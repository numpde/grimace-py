"""Producer-free verification for one count-free writer EOS artifact."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from collections.abc import Mapping

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .residual_constraints import ResidualFactorKey
from .residual_constraints import ResidualStore
from .residual_constraints import tetra_parity_var
from .writer_envelope_terms import _identity_digest
from .writer_envelope_terms import _identity_envelope
from .writer_graph_obligations import WriterGraphCompletionStatus
from .writer_execution_evidence import WriterGraphObligationWorkEvidence
from .writer_execution_evidence import WriterResidualPropagationWorkEvidence
from .writer_capabilities import _WriterExecutionCapabilityKind
from .writer_events import WriterLocalOrderClosed
from .writer_stereo import WriterStereoLifecycleEvidence
from .writer_stereo import WriterStereoLifecycleOutcomeKind
from .writer_terminal_certificates import WriterTerminalCertificate
from .writer_terminal_certificates import WriterTerminalCertificateKind
from .writer_snapshot_closed_terms import writer_frontier_cursor_from_closed_terms
from .writer_support_artifact_offline_verifier import _decode_transition_term
from .writer_support_artifact_offline_verifier import _local_order_parity
from .writer_support_artifact_offline_verifier import _specified_tetra_site_for_transition
from .writer_support_artifact_offline_verifier import _check_transition_result
from .writer_support_artifact_fact_verifier import _check_prepared_identity
from .writer_support_artifact_fact_verifier import _check_source_snapshot_identity
from .writer_prepared_identity import writer_prepared_identity
from .prepared_runtime import prepare_south_star_mol_from_facts
from .prepared_runtime import SouthStarWriterSurface
from .writer_terminalization_artifact_checker import verify_writer_terminalization_artifact_consistency
from .writer_terminalization_terms import WriterTerminalizationTerm
from .writer_stereo import EMPTY_RESIDUAL_SNAPSHOT
from .writer_component_completion_replay import (
    WriterComponentCompletionReplayError,
    replay_completed_component_prefix,
)
from .writer_residual_transition_terms import TetraLocalOrderFactorClosureTransitionTerm
from .writer_residual_transition_terms import WriterResidualTransitionKind

_TERM_PATH = (
    "grimace._south_star1.writer_terminalization_terms."
    "WriterTerminalizationTerm"
)
_TETRA_PATH = (
    "grimace._south_star1.writer_residual_transition_terms."
    "TetraLocalOrderFactorClosureTransitionTerm"
)


@dataclass(frozen=True, slots=True)
class WriterTerminalizationArtifactFactsVerification:
    accepted: bool
    terminalization_mode: str | None = None
    semantically_replayed_operations: tuple[str, ...] = ()
    checked_obligation_families: tuple[str, ...] = ()
    unchecked_obligation_families: tuple[str, ...] = ()
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class _ExpectedTerminalizationEvidence:
    graph_work: tuple[WriterGraphObligationWorkEvidence, ...]
    residual_work: tuple[WriterResidualPropagationWorkEvidence, ...]
    lifecycle: tuple[WriterStereoLifecycleEvidence, ...]
    capabilities: frozenset[_WriterExecutionCapabilityKind]
    certificates: tuple[WriterTerminalCertificate, ...]
    terminal_support_key: tuple[object, ...]
    operations: tuple[str, ...]


def verify_writer_terminalization_artifact_for_facts(
    *, facts, runtime_options, artifact, policy=None
) -> WriterTerminalizationArtifactFactsVerification:
    try:
        structural = verify_writer_terminalization_artifact_consistency(artifact)
        if not structural.accepted:
            _violation(structural.reason or "structural_rejection")
        prepared = prepare_south_star_mol_from_facts(
            facts, writer_surface=SouthStarWriterSurface(), policy=policy
        )
        expected_identity = _identity_envelope(
            writer_prepared_identity(prepared, runtime_options)
        )
        _check_prepared_identity(artifact, expected_identity)
        _check_source_snapshot_identity(
            artifact["source_snapshot"],
            expected_identity=expected_identity,
            runtime_options=runtime_options,
        )
        objects = {item["object_id"]: item for item in artifact["objects"]}
        projection = objects[artifact["roots"]["terminal_projection_ref"]]["payload"]
        support = objects[artifact["roots"]["terminal_support_ref"]]["payload"]
        source_state = _unique_state(
            projection["source_cursor"], support["source_state_digest"], "source"
        )
        finalized_state = _unique_state(
            projection["finalized_cursor"],
            support["finalized_state_digest"],
            "finalized",
        )
        term = _decode_transition_term(
            support["terminalization_term"], expected_path=_TERM_PATH
        )
        if not isinstance(term, WriterTerminalizationTerm):
            _violation("terminalization_term_kind_mismatch")
        expected = _reconstruct_terminalization_evidence_for_facts(
            facts=facts,
            term=term,
            support=support,
            source=source_state,
            finalized=finalized_state,
            rooted_at_atom=runtime_options.rooted_at_atom,
        )
        return WriterTerminalizationArtifactFactsVerification(
            accepted=True,
            terminalization_mode=term.stereo_mode,
            semantically_replayed_operations=expected.operations,
            checked_obligation_families=tuple(
                family
                for family in (
                    "terminal_residual_work",
                    "terminal_stereo_lifecycle",
                    "terminal_graph_obligation_work",
                )
                if support["obligation_manifests"][family]
            ),
        )
    except SouthStarError as exc:
        return WriterTerminalizationArtifactFactsVerification(
            accepted=False,
            reason=exc.args[-1] if exc.args else "terminalization_verification_error",
        )
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        return WriterTerminalizationArtifactFactsVerification(
            accepted=False,
            reason=f"malformed_terminalization_artifact:{type(exc).__name__}:{exc}",
        )


def replay_terminal_support_payload_for_facts(
    *, facts, support, source_state, finalized_state, rooted_at_atom: int = -1
):
    """Replay a shared terminal-support payload without artifact routing."""
    term = _decode_transition_term(
        support["terminalization_term"], expected_path=_TERM_PATH
    )
    expected = _reconstruct_terminalization_evidence_for_facts(
        facts=facts,
        term=term,
        support=support,
        source=source_state,
        finalized=finalized_state,
        rooted_at_atom=rooted_at_atom,
    )
    return term, expected.operations


def _reconstruct_terminalization_evidence_for_facts(
    *, facts, term, support, source, finalized, rooted_at_atom: int
) -> _ExpectedTerminalizationEvidence:
    graph_work = _check_term_and_graph(
        facts=facts,
        term=term,
        support=support,
        source=source,
        finalized=finalized,
        rooted_at_atom=rooted_at_atom,
    )
    operations, residual_work, capabilities, lifecycle = _replay_stereo(
        facts=facts,
        term=term,
        support=support,
        source=source,
        finalized=finalized,
    )
    status = WriterGraphCompletionStatus(True, (), ())
    certificates = (
        WriterTerminalCertificate(
            kind=WriterTerminalCertificateKind.GRAPH_COMPLETE,
            source_state=source,
            finalized_state=finalized,
            graph_completion_status=status,
            graph_obligation_work_evidence=graph_work,
        ),
        WriterTerminalCertificate(
            kind=WriterTerminalCertificateKind.STEREO_TERMINALIZED,
            source_state=source,
            finalized_state=finalized,
            terminal_stereo_lifecycle_evidence=lifecycle,
            terminal_execution_capabilities=capabilities,
            terminal_residual_work_evidence=residual_work,
        ),
        WriterTerminalCertificate(
            kind=WriterTerminalCertificateKind.FINALIZED_STATE,
            source_state=source,
            finalized_state=finalized,
            graph_obligation_work_evidence=graph_work,
            terminal_stereo_lifecycle_evidence=lifecycle,
            terminal_execution_capabilities=capabilities,
            terminal_residual_work_evidence=residual_work,
        ),
    )
    support_key = (
        source,
        finalized,
        support["parent_weight"],
        support["terminal_ordinal"],
        capabilities,
        residual_work,
        lifecycle,
        graph_work,
        certificates,
    )
    _check_reconstructed_support_identity(
        support=support,
        graph_work=graph_work,
        residual_work=residual_work,
        lifecycle=lifecycle,
        capabilities=capabilities,
        certificates=certificates,
        support_key=support_key,
    )
    return _ExpectedTerminalizationEvidence(
        graph_work=graph_work,
        residual_work=residual_work,
        lifecycle=lifecycle,
        capabilities=capabilities,
        certificates=certificates,
        terminal_support_key=support_key,
        operations=operations,
    )


def _unique_state(cursor_terms, digest, role):
    cursor = writer_frontier_cursor_from_closed_terms(cursor_terms["terms"])
    matches = tuple(
        state for state, _weight in cursor.weighted_states
        if _identity_digest(state) == digest
    )
    if len(matches) != 1:
        _violation(f"terminal_{role}_state_anchor_mismatch")
    return matches[0]


def _check_term_and_graph(
    *, facts, term, support, source, finalized, rooted_at_atom: int
):
    if support["terminalization_term_digest"] != _identity_digest(
        support["terminalization_term"]
    ):
        _violation("terminalization_term_digest_mismatch")
    fact_bonds = {bond.id: bond for bond in facts.bonds}
    closure_bonds = tuple(closure.bond for closure in source.ring_state.closed_closures)
    for closure in source.ring_state.closed_closures:
        _check_closed_closure(fact_bonds=fact_bonds, closure=closure)
    if (
        term.source_state_digest != _identity_digest(source)
        or term.finalized_state_digest != _identity_digest(finalized)
        or int(term.active_atom) != int(source.active.atom)
    ):
        _violation("terminalization_state_identity_mismatch")
    expected_status = WriterGraphCompletionStatus(True, (), ())
    if term.graph_completion_status != expected_status:
        _violation("terminal_graph_completion_status_mismatch")
    if (
        source.obligations.pending_entry is not None
        or source.branch_stack
        or source.ring_state.open_endpoints
        or source.ring_state.label_state.allocated
        or set(source.visited_atoms) != {atom.id for atom in facts.atoms}
        or set(source.written_bonds) & set(closure_bonds)
        or len(closure_bonds) != len(set(closure_bonds))
        or set(source.written_bonds) | set(closure_bonds) != set(fact_bonds)
        or source.active.atom not in source.visited_atoms
        or not source.active.atom_emitted
        or int(source.component_cursor.component_index) != len(facts.components) - 1
    ):
        _violation("terminal_graph_completion_mismatch")
    try:
        replay_completed_component_prefix(
            facts=facts,
            state=source,
            completed_component_index=len(facts.components) - 1,
            require_final=True,
            rooted_at_atom=rooted_at_atom,
        )
    except WriterComponentCompletionReplayError as exc:
        if str(exc) == "component_boundary_root_vector_mismatch":
            _violation("terminal_component_roots_mismatch")
        _violation("terminal_graph_completion_mismatch")
    _check_component_and_active_frame(facts=facts, source=source)
    graph_items = support["obligation_manifests"]["terminal_graph_obligation_work"]
    if len(graph_items) != 1:
        _violation("terminal_graph_manifest_count_mismatch")
    if tuple(item["evidence_digest"] for item in graph_items) != (
        term.graph_obligation_work_digests
    ):
        _violation("terminal_graph_work_digest_mismatch")
    component = facts.components[source.component_cursor.component_index]
    expected_graph_work = WriterGraphObligationWorkEvidence(
        operation="writer graph obligation context",
        component_index=source.component_cursor.component_index,
        component_atom_count=len(component.atoms),
        component_bond_count=len(component.bonds),
        edge_obligation_count=len(component.bonds),
        residual_attachment_count=0,
        residual_attachment_action_count=0,
        boundary_incidence_count=0,
        closure_candidate_count=0,
        live_branch_return_closure_candidate_count=0,
        deferred_branch_return_closure_candidate_count=0,
        deferred_control_live_closure_candidate_count=0,
        unsupported_closure_candidate_count=0,
        open_closure_count=0,
        closed_closure_count=sum(
            closure.bond in component.bonds
            for closure in source.ring_state.closed_closures
        ),
        max_attachment_atom_count=0,
        max_attachment_boundary_count=0,
        max_attachment_cyclic_rank=0,
    )
    if term.graph_obligation_work_digests != (_identity_digest(expected_graph_work),):
        _violation("terminal_graph_work_replay_mismatch")
    _check_graph_manifest(
        item=graph_items[0],
        source_digest=term.source_state_digest,
        successor_digest=term.finalized_state_digest,
        evidence_digest=_identity_digest(expected_graph_work),
    )
    for field in (
        "component_cursor", "active", "branch_stack", "visited_atoms",
        "written_bonds", "obligations", "ring_state", "policy_state",
    ):
        if getattr(source, field) != getattr(finalized, field):
            _violation("terminal_finalized_non_stereo_state_mismatch")
    if (
        term.source_residual_snapshot_digest
        != _identity_digest(source.stereo_state.residual_snapshot)
        or term.finalized_residual_snapshot_digest
        != _identity_digest(finalized.stereo_state.residual_snapshot)
    ):
        _violation("terminal_residual_state_anchor_mismatch")
    return (expected_graph_work,)


def _check_component_and_active_frame(*, facts, source) -> None:
    roots = source.component_cursor.component_roots
    if (
        len(roots) != len(facts.components)
        or any(root not in component.atoms for root, component in zip(roots, facts.components))
    ):
        _violation("terminal_component_roots_mismatch")
    component = facts.components[source.component_cursor.component_index]
    active = source.active
    if active.atom not in component.atoms:
        _violation("terminal_active_component_mismatch")
    if active.parent is None:
        if active.incoming_bond is not None:
            _violation("terminal_active_frame_mismatch")
        return
    bonds = tuple(bond for bond in facts.bonds if bond.id == active.incoming_bond)
    if (
        len(bonds) != 1
        or {bonds[0].a, bonds[0].b} != {active.parent, active.atom}
    ):
        _violation("terminal_active_frame_mismatch")


def _check_closed_closure(*, fact_bonds, closure) -> None:
    bond = fact_bonds.get(closure.bond)
    if bond is None:
        _violation("terminal_closed_closure_unknown_bond")
    if {closure.first_atom, closure.second_atom} != {bond.a, bond.b}:
        _violation("terminal_closed_closure_endpoint_mismatch")
    expected_label_text = (
        str(closure.label.value)
        if closure.label.value < 10
        else f"%{closure.label.value}"
    )
    if (
        closure.label.value <= 0
        or closure.label.text != expected_label_text
        or closure.first_endpoint_text != closure.label.text
        or closure.second_endpoint_text != closure.label.text
    ):
        _violation("terminal_closed_closure_label_mismatch")
    markers = (closure.first_endpoint_bond_text, closure.second_endpoint_bond_text)
    order = getattr(bond.order, "value", bond.order)
    required = "=" if order == "double" else "#" if order == "triple" else ""
    if required:
        if markers.count(required) != 1 or any(item not in ("", required) for item in markers):
            _violation("terminal_closed_closure_marker_mismatch")
        if (
            closure.first_endpoint_direction_mark.value
            or closure.second_endpoint_direction_mark.value
        ):
            _violation("terminal_closed_closure_direction_mismatch")
    elif any(item not in ("", "-") for item in markers):
        _violation("terminal_closed_closure_marker_mismatch")


def _check_graph_manifest(*, item, source_digest, successor_digest, evidence_digest):
    expected = {
        "family": "terminal_graph_obligation_work",
        "operation": "writer graph obligation context",
        "source_digest": source_digest,
        "successor_digest": successor_digest,
        "is_noop": source_digest == successor_digest,
        "is_empty": False,
        "is_discharged": False,
        "terminal_clean": True,
        "ring_summary": None,
        "evidence_digest": evidence_digest,
        "transition_term": None,
        "transition_digest": None,
        "coupling_term": None,
        "coupling_term_digest": None,
        "linked_lifecycle_digests": [],
        "linked_residual_work_digests": [],
    }
    for key, value in expected.items():
        if item[key] != value:
            _violation("terminal_graph_manifest_mismatch")


def _replay_stereo(*, facts, term, support, source, finalized):
    manifests = support["obligation_manifests"]
    residual_items = manifests["terminal_residual_work"]
    lifecycle_items = manifests["terminal_stereo_lifecycle"]
    if tuple(item["evidence_digest"] for item in residual_items) != term.terminal_residual_work_digests:
        _violation("terminal_residual_work_digest_mismatch")
    if tuple(item["evidence_digest"] for item in lifecycle_items) != term.terminal_stereo_lifecycle_digests:
        _violation("terminal_lifecycle_digest_mismatch")
    if len(lifecycle_items) != 1:
        _violation("terminal_lifecycle_count_mismatch")
    lifecycle = lifecycle_items[0]
    if (
        lifecycle["lifecycle_event_kind"] != "local_order_closed"
        or lifecycle["source_digest"] != term.source_state_digest
        or lifecycle["successor_digest"] != term.finalized_state_digest
        or tuple(lifecycle["residual_work_digests"])
        != term.terminal_residual_work_digests
        or tuple(lifecycle["lifecycle_capabilities"])
        != term.terminal_execution_capabilities
    ):
        _violation("terminal_lifecycle_provenance_mismatch")
    source_orders = source.stereo_state.local_orders
    final_orders = finalized.stereo_state.local_orders
    source_matches = tuple(item for item in source_orders if item.atom == source.active.atom)
    final_matches = tuple(item for item in final_orders if item.atom == source.active.atom)
    if len(final_matches) != 1 or not final_matches[0].closed:
        _violation("terminal_local_order_lifecycle_mismatch")
    if source_matches:
        if source_matches[0].closed or (
            final_matches[0].order[: len(source_matches[0].order)]
            != source_matches[0].order
        ):
            _violation("terminal_local_order_lifecycle_mismatch")
        expected_final = final_matches[0]
    else:
        expected_final = final_matches[0]
        if final_matches[0].order:
            _violation("terminal_local_order_lifecycle_mismatch")
    if len(source_matches) > 1 or expected_final != final_matches[0]:
        _violation("terminal_local_order_lifecycle_mismatch")
    if term.stereo_mode == "noop":
        if residual_items or term.terminal_execution_capabilities:
            _violation("terminal_false_noop")
        if (
            source.stereo_state.residual_snapshot != EMPTY_RESIDUAL_SNAPSHOT
            or finalized.stereo_state.residual_snapshot != EMPTY_RESIDUAL_SNAPSHOT
        ):
            _violation("terminal_false_noop")
        _check_only_local_order_closed(source, finalized)
        residual_work = ()
        capabilities = frozenset()
        operations = ()
        outcome = WriterStereoLifecycleOutcomeKind.EVENT_RECORDED
        _check_residual_manifests_noop(
            support=support,
            term=term,
        )
    else:
        if term.stereo_mode != "tetra_local_order_factor_closure" or len(residual_items) != 1:
            _violation("terminal_stereo_mode_mismatch")
        item = residual_items[0]
        if item["operation"] != "tetrahedral local-order factor closure":
            _violation("terminal_residual_operation_mismatch")
        transition = _decode_transition_term(item["transition_term"], expected_path=_TETRA_PATH)
        if not isinstance(transition, TetraLocalOrderFactorClosureTransitionTerm):
            _violation("terminal_transition_term_kind_mismatch")
        _replay_tetra(facts, transition, source, finalized)
        capabilities = frozenset((
            _WriterExecutionCapabilityKind.RESIDUAL_FACTOR_DISCHARGE,
            _WriterExecutionCapabilityKind.RESIDUAL_PROPAGATION,
            _WriterExecutionCapabilityKind.TETRA_LOCAL_ORDER_RESTRICTION,
        ))
        expected_caps = tuple(sorted(capability.value for capability in capabilities))
        if term.terminal_execution_capabilities != expected_caps:
            _violation("terminal_capability_mismatch")
        residual_work = (WriterResidualPropagationWorkEvidence(
            operation="tetrahedral local-order factor closure",
            result_kind=transition.propagation_result.kind,
            component_variables=transition.propagation_result.stats.component_variables,
            component_factor_keys=transition.propagation_result.stats.component_factor_keys,
            checked_candidate_rows=transition.propagation_result.stats.checked_candidate_rows,
            largest_factor_scope=transition.propagation_result.stats.largest_factor_scope,
            largest_candidate_row_count=(
                transition.propagation_result.stats.largest_candidate_row_count
            ),
            transition_term=transition,
        ),)
        operations = (item["operation"],)
        outcome = WriterStereoLifecycleOutcomeKind.RECORD_AND_RESTRICT
        _check_tetra_residual_manifest(
            item=item,
            term=term,
            residual_work=residual_work,
        )
    lifecycle_evidence = WriterStereoLifecycleEvidence(
        event=WriterLocalOrderClosed(source.active.atom),
        source_residual_snapshot=source.stereo_state.residual_snapshot,
        successor_residual_snapshot=finalized.stereo_state.residual_snapshot,
        source_atom_occurrences=source.stereo_state.atom_occurrences,
        successor_atom_occurrences=finalized.stereo_state.atom_occurrences,
        source_bond_occurrences=source.stereo_state.bond_occurrences,
        successor_bond_occurrences=finalized.stereo_state.bond_occurrences,
        source_local_orders=source.stereo_state.local_orders,
        successor_local_orders=finalized.stereo_state.local_orders,
        capabilities=capabilities,
        residual_work_evidence=residual_work,
        outcome_kind=outcome,
    )
    _check_lifecycle_manifest(
        item=lifecycle,
        term=term,
        lifecycle=lifecycle_evidence,
        residual_work=residual_work,
        capabilities=capabilities,
        outcome=outcome,
    )
    return operations, residual_work, capabilities, (lifecycle_evidence,)


def _replay_tetra(facts, term, source, finalized):
    if term.kind is not WriterResidualTransitionKind.TETRA_LOCAL_ORDER_FACTOR_CLOSURE:
        _violation("terminal_transition_kind_mismatch")
    if (
        term.source_snapshot != source.stereo_state.residual_snapshot
        or term.successor_snapshot != finalized.stereo_state.residual_snapshot
        or int(term.atom) != int(source.active.atom)
    ):
        _violation("terminal_transition_state_anchor_mismatch")
    site = _specified_tetra_site_for_transition(
        facts=facts,
        site=int(term.site),
        atom=int(term.atom),
        violation_prefix="terminal_tetra",
    )
    if tuple(term.reference_order) != tuple(site.reference_order):
        _violation("terminal_tetra_reference_order_mismatch")
    final_order = next(item.order for item in finalized.stereo_state.local_orders if item.atom == term.atom)
    if tuple(term.local_order) != tuple(final_order):
        _violation("terminal_tetra_local_order_mismatch")
    parity = _local_order_parity(reference_order=site.reference_order, local_order=term.local_order)
    if (
        term.target_parity is not parity
        or term.constraint_var != tetra_parity_var(term.site)
        or term.constraint_value is not parity
        or term.discharged_factor_keys != (ResidualFactorKey("tetra_site", (int(term.site),)),)
        or term.projected_variables != (term.constraint_var,)
    ):
        _violation("terminal_tetra_restriction_mismatch")
    store = ResidualStore.from_value_snapshot(term.source_snapshot)
    result = store.restrict_many_and_propagate(((term.constraint_var, parity),))
    _check_transition_result(expected=term.propagation_result, actual=result, violation_prefix="terminal_tetra")
    if (
        result.stats.component_variables != term.affected_variables
        or result.stats.component_factor_keys != term.affected_factor_keys
    ):
        _violation("terminal_tetra_affected_component_mismatch")
    try:
        store.discharge_satisfied_factors(term.discharged_factor_keys)
    except ValueError:
        _violation("terminal_tetra_discharge_failed")
    if store.value_snapshot() != term.successor_snapshot or term.successor_snapshot != EMPTY_RESIDUAL_SNAPSHOT:
        _violation("terminal_final_residual_not_empty")
    _check_only_local_order_closed(source, finalized)


def _check_residual_manifests_noop(*, support, term) -> None:
    if (
        support["obligation_manifests"]["terminal_residual_work"]
        or term.terminal_residual_work_digests
        or support["terminal_residual_work_evidence_digest"] != _identity_digest(())
    ):
        _violation("terminal_false_noop_residual_work")


def _check_tetra_residual_manifest(*, item, term, residual_work) -> None:
    lifecycle_digest = term.terminal_stereo_lifecycle_digests
    expected = {
        "family": "terminal_residual_work",
        "operation": "tetrahedral local-order factor closure",
        "source_digest": term.source_state_digest,
        "successor_digest": term.finalized_state_digest,
        "is_noop": False,
        "is_empty": False,
        "is_discharged": False,
        "terminal_clean": False,
        "ring_summary": None,
        "evidence_digest": _identity_digest(residual_work[0]),
        "transition_digest": _identity_digest(residual_work[0].transition_term),
        "coupling_term": None,
        "coupling_term_digest": None,
        "linked_lifecycle_digests": list(lifecycle_digest),
        "linked_residual_work_digests": [],
    }
    for key, value in expected.items():
        if item[key] != value:
            _violation("terminal_tetra_residual_manifest_mismatch")


def _check_lifecycle_manifest(
    *, item, term, lifecycle, residual_work, capabilities, outcome
) -> None:
    residual_digests = [_identity_digest(item) for item in residual_work]
    residual_operations = [item.operation for item in residual_work]
    expected = {
        "family": "terminal_stereo_lifecycle",
        "operation": "WriterStereoLifecycleEvidence",
        "source_digest": term.source_state_digest,
        "successor_digest": term.finalized_state_digest,
        "is_noop": False,
        "is_empty": False,
        "is_discharged": False,
        "terminal_clean": True,
        "ring_summary": None,
        "evidence_digest": _identity_digest(lifecycle),
        "transition_term": None,
        "transition_digest": None,
        "coupling_term": None,
        "coupling_term_digest": None,
        "linked_lifecycle_digests": [],
        "linked_residual_work_digests": residual_digests,
        "lifecycle_event_kind": "local_order_closed",
        "lifecycle_capabilities": sorted(capability.value for capability in capabilities),
        "lifecycle_outcome_kind": outcome.value,
        "residual_snapshot_changed": (
            lifecycle.source_residual_snapshot
            != lifecycle.successor_residual_snapshot
        ),
        "source_residual_snapshot_digest": _identity_digest(
            lifecycle.source_residual_snapshot
        ),
        "successor_residual_snapshot_digest": _identity_digest(
            lifecycle.successor_residual_snapshot
        ),
        "local_orders_changed": True,
        "residual_work_digests": residual_digests,
        "residual_work_operations": residual_operations,
        "certificate_kind": None,
        "certificate_capability": None,
        "certificate_lifecycle_digest": None,
    }
    for key, value in expected.items():
        if item[key] != value:
            _violation("terminal_lifecycle_identity_mismatch")


def _check_reconstructed_support_identity(
    *, support, graph_work, residual_work, lifecycle, capabilities, certificates,
    support_key
) -> None:
    expected = {
        "terminal_execution_capabilities_digest": _identity_digest(capabilities),
        "terminal_residual_work_evidence_digest": _identity_digest(residual_work),
        "terminal_stereo_lifecycle_evidence_digest": _identity_digest(lifecycle),
        "graph_obligation_work_evidence_digest": _identity_digest(graph_work),
        "terminal_support_key_digest": _identity_digest(support_key),
        "terminal_certificate_digests": [
            _identity_digest(certificate) for certificate in certificates
        ],
    }
    for key, value in expected.items():
        if support[key] != value:
            _violation(f"terminal_support_{key}_mismatch")
    expected_identity = {
        "source_state_digest": _identity_digest(support_key[0]),
        "finalized_state_digest": _identity_digest(support_key[1]),
        "parent_weight": support["parent_weight"],
        "terminal_ordinal": support["terminal_ordinal"],
        "terminal_support_key_digest": _identity_digest(support_key),
        "terminal_execution_capabilities_digest": _identity_digest(capabilities),
        "terminal_residual_work_evidence_digest": _identity_digest(residual_work),
        "terminal_stereo_lifecycle_evidence_digest": _identity_digest(lifecycle),
        "graph_obligation_work_evidence_digest": _identity_digest(graph_work),
        "terminal_certificate_digests": [
            _identity_digest(certificate) for certificate in certificates
        ],
    }
    if support["digest"] != _identity_digest(expected_identity):
        _violation("terminal_support_identity_digest_mismatch")


def _check_only_local_order_closed(source, finalized):
    if (
        source.stereo_state.atom_occurrences != finalized.stereo_state.atom_occurrences
        or source.stereo_state.bond_occurrences != finalized.stereo_state.bond_occurrences
    ):
        _violation("terminal_unrelated_stereo_state_changed")
    source_orders = source.stereo_state.local_orders
    final_active = tuple(
        item for item in finalized.stereo_state.local_orders
        if item.atom == source.active.atom
    )
    expected_orders = tuple(
        final_active[0] if item.atom == source.active.atom else item
        for item in source_orders
    )
    if not any(item.atom == source.active.atom for item in source_orders):
        expected_orders += tuple(
            item
            for item in finalized.stereo_state.local_orders
            if item.atom == source.active.atom
        )
        expected_orders = tuple(sorted(expected_orders, key=lambda item: int(item.atom)))
    if expected_orders != finalized.stereo_state.local_orders:
        _violation("terminal_unrelated_stereo_state_changed")


def _violation(reason):
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer terminalization facts violation: {reason}",
    )


__all__ = (
    "WriterTerminalizationArtifactFactsVerification",
    "verify_writer_terminalization_artifact_for_facts",
    "replay_terminal_support_payload_for_facts",
)
