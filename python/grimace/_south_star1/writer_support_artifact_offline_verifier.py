"""Producer-free offline relation replay for writer support artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .facts import AtomFacts
from .facts import BondFacts
from .facts import BondOrder
from .facts import DirectionalSiteFacts
from .facts import DirectionalValue
from .facts import LigandKind
from .facts import MoleculeFacts
from .facts import SiteStatus
from .facts import TetrahedralSiteFacts
from .facts import TetraValue
from .policy import DirectionMark
from .policy import TetraToken
from .residual_constraints import ResidualFactorKey
from .residual_constraints import DirectionalBondEmissionFactorValueSnapshot
from .residual_constraints import DirectionalNormalizedSign
from .residual_constraints import DirectionalSiteCarrierModel
from .residual_constraints import DirectionalSiteFactorValueSnapshot
from .residual_constraints import ResidualPropagationKind
from .residual_constraints import ResidualPropagationResult
from .residual_constraints import ResidualPropagationStats
from .residual_constraints import ResidualStore
from .residual_constraints import ResidualStoreValueSnapshot
from .residual_constraints import TetraLocalParity
from .residual_constraints import TetraResidualFactorValueSnapshot
from .residual_constraints import TetraTokenParityFactorValueSnapshot
from .residual_constraints import VarId
from .residual_constraints import directional_site_carrier_var
from .residual_constraints import normalized_sign_from_mark
from .residual_constraints import tetra_parity_var
from .residual_constraints import tetra_token_var
from .writer_residual_transition_terms import (
    DirectionalCarrierMarkRestrictionTransitionTerm,
)
from .writer_residual_transition_terms import (
    DirectionalRingEndpointProjectionTransitionTerm,
)
from .writer_residual_transition_terms import (
    DirectionalRingPairRestrictionTransitionTerm,
)
from .writer_residual_transition_terms import (
    SharedDirectionalRingEndpointProjectionTransitionTerm,
)
from .writer_atom_text_lifecycle import bracket_atom_text
from .writer_count_dag_envelope import count_dag_node_by_id
from .writer_count_dag_envelope import validate_writer_count_certificate_dag_envelope
from .writer_envelope_terms import _identity_digest
from .writer_envelope_terms import _digest_terms_bounded
from .writer_envelope_terms import _term
from .writer_envelope_work import WriterEnvelopeWorkBudget
from .writer_envelope_work import default_writer_envelope_work_budget
from .writer_residual_transition_terms import (
    TetraAtomTokenRestrictionTransitionTerm,
)
from .writer_residual_transition_terms import (
    TetraLocalOrderFactorClosureTransitionTerm,
)
from .writer_residual_transition_terms import WriterResidualTransitionKind


OBJECT_KIND_OFFLINE_COVERAGE = {
    "source_snapshot": "identity_checked",
    "count_envelope": "arithmetic_checked",
    "count_dag": "arithmetic_checked",
    "frontier_product": "structurally_checked",
    "replay_path": "partially_offline_checked",
    "branch_support": "partially_offline_checked",
    "text_projection": "partially_offline_checked",
    "terminal_projection": "partially_offline_checked",
    "terminal_support": "partially_offline_checked",
    "support_string": "partially_offline_checked",
    "support_image_coverage": "structurally_checked",
    "support_image": "structurally_checked",
}

_OFFLINE_UNCHECKED_OBJECT_KINDS = (
    "count_envelope",
    "frontier_product",
    "terminal_support",
    "support_image_coverage",
    "support_image",
)

_PATH_PREFIX = "grimace._south_star1."


class OfflineResidualReplayDisposition(Enum):
    SEMANTICALLY_REPLAYED = "semantically_replayed"
    DECLARED_OUT_OF_SCOPE = "declared_out_of_scope"


_ALLOWED_TETRA_TRANSITION_ENUMS = {
    _PATH_PREFIX + "facts.SiteStatus": SiteStatus,
    _PATH_PREFIX + "facts.TetraValue": TetraValue,
    _PATH_PREFIX + "facts.DirectionalValue": DirectionalValue,
    _PATH_PREFIX + "policy.DirectionMark": DirectionMark,
    _PATH_PREFIX + "policy.TetraToken": TetraToken,
    _PATH_PREFIX + "residual_constraints.ResidualPropagationKind": (
        ResidualPropagationKind
    ),
    _PATH_PREFIX + "residual_constraints.TetraLocalParity": TetraLocalParity,
    _PATH_PREFIX + "residual_constraints.DirectionalNormalizedSign": (
        DirectionalNormalizedSign
    ),
    _PATH_PREFIX + "writer_residual_transition_terms.WriterResidualTransitionKind": (
        WriterResidualTransitionKind
    ),
}
_ALLOWED_TETRA_TRANSITION_DATACLASSES = {
    _PATH_PREFIX + "residual_constraints.ResidualFactorKey": ResidualFactorKey,
    _PATH_PREFIX + "residual_constraints.ResidualPropagationResult": (
        ResidualPropagationResult
    ),
    _PATH_PREFIX + "residual_constraints.ResidualPropagationStats": (
        ResidualPropagationStats
    ),
    _PATH_PREFIX + "residual_constraints.ResidualStoreValueSnapshot": (
        ResidualStoreValueSnapshot
    ),
    _PATH_PREFIX + "residual_constraints.TetraResidualFactorValueSnapshot": (
        TetraResidualFactorValueSnapshot
    ),
    _PATH_PREFIX + "residual_constraints.TetraTokenParityFactorValueSnapshot": (
        TetraTokenParityFactorValueSnapshot
    ),
    _PATH_PREFIX + "residual_constraints.DirectionalSiteCarrierModel": (
        DirectionalSiteCarrierModel
    ),
    _PATH_PREFIX + "residual_constraints.DirectionalSiteFactorValueSnapshot": (
        DirectionalSiteFactorValueSnapshot
    ),
    _PATH_PREFIX + "residual_constraints."
    "DirectionalBondEmissionFactorValueSnapshot": (
        DirectionalBondEmissionFactorValueSnapshot
    ),
    _PATH_PREFIX + "residual_constraints.VarId": VarId,
    _PATH_PREFIX + "writer_residual_transition_terms."
    "DirectionalCarrierMarkRestrictionTransitionTerm": (
        DirectionalCarrierMarkRestrictionTransitionTerm
    ),
    _PATH_PREFIX + "writer_residual_transition_terms."
    "DirectionalRingEndpointProjectionTransitionTerm": (
        DirectionalRingEndpointProjectionTransitionTerm
    ),
    _PATH_PREFIX + "writer_residual_transition_terms."
    "DirectionalRingPairRestrictionTransitionTerm": (
        DirectionalRingPairRestrictionTransitionTerm
    ),
    _PATH_PREFIX + "writer_residual_transition_terms."
    "SharedDirectionalRingEndpointProjectionTransitionTerm": (
        SharedDirectionalRingEndpointProjectionTransitionTerm
    ),
    _PATH_PREFIX + "writer_residual_transition_terms."
    "TetraAtomTokenRestrictionTransitionTerm": (
        TetraAtomTokenRestrictionTransitionTerm
    ),
    _PATH_PREFIX + "writer_residual_transition_terms."
    "TetraLocalOrderFactorClosureTransitionTerm": (
        TetraLocalOrderFactorClosureTransitionTerm
    ),
}
_ALLOWED_TETRA_TRANSITION_DATACLASS_FIELDS = {
    _PATH_PREFIX + "residual_constraints.ResidualFactorKey": (
        frozenset(("kind", "key"))
    ),
    _PATH_PREFIX + "residual_constraints.ResidualPropagationResult": (
        frozenset(("kind", "stats"))
    ),
    _PATH_PREFIX + "residual_constraints.ResidualPropagationStats": (
        frozenset(
            (
                "component_variables",
                "component_factor_keys",
                "checked_candidate_rows",
                "largest_factor_scope",
                "largest_candidate_row_count",
            )
        )
    ),
    _PATH_PREFIX + "residual_constraints.ResidualStoreValueSnapshot": (
        frozenset(("domains", "assignments", "factors"))
    ),
    _PATH_PREFIX + "residual_constraints.TetraResidualFactorValueSnapshot": (
        frozenset(
            ("key", "scope", "status", "target", "reference_order", "local_order")
        )
    ),
    _PATH_PREFIX + "residual_constraints.TetraTokenParityFactorValueSnapshot": (
        frozenset(("key", "scope", "status", "target"))
    ),
    _PATH_PREFIX + "residual_constraints.DirectionalSiteCarrierModel": (
        frozenset(
            (
                "site",
                "bond",
                "side",
                "endpoint_orientation_factor",
                "ligand_factor",
            )
        )
    ),
    _PATH_PREFIX + "residual_constraints.DirectionalSiteFactorValueSnapshot": (
        frozenset(("key", "scope", "sides", "status", "target"))
    ),
    _PATH_PREFIX + "residual_constraints."
    "DirectionalBondEmissionFactorValueSnapshot": (
        frozenset(("key", "scope", "models", "allowed_marks"))
    ),
    _PATH_PREFIX + "residual_constraints.VarId": frozenset(("kind", "key")),
    _PATH_PREFIX + "writer_residual_transition_terms."
    "DirectionalCarrierMarkRestrictionTransitionTerm": (
        frozenset(
            (
                "kind",
                "source_snapshot",
                "source_snapshot_digest",
                "bond",
                "parent",
                "child",
                "direction_mark",
                "canonical_orientation",
                "carrier_models",
                "restrictions",
                "affected_variables",
                "affected_factor_keys",
                "propagation_result",
                "discharged_factor_keys",
                "projected_variables",
                "successor_snapshot",
                "successor_snapshot_digest",
            )
        )
    ),
    _PATH_PREFIX + "writer_residual_transition_terms."
    "DirectionalRingEndpointProjectionTransitionTerm": (
        frozenset(
            (
                "kind", "source_snapshot", "source_snapshot_digest", "bond",
                "endpoint_atom", "partner_atom", "ring_label_value",
                "ring_label_text", "endpoint_text", "bond_text",
                "direction_mark", "carrier_model",
                "compatible_second_endpoint_choices", "domain_intersections",
                "affected_variables", "affected_factor_keys",
                "propagation_result", "projected_variables",
                "discharged_factor_keys", "successor_snapshot",
                "successor_snapshot_digest",
            )
        )
    ),
    _PATH_PREFIX + "writer_residual_transition_terms."
    "DirectionalRingPairRestrictionTransitionTerm": (
        frozenset((
            "kind", "source_snapshot", "source_snapshot_digest", "bond",
            "first_atom", "second_atom", "ring_label_value", "ring_label_text",
            "first_endpoint_text", "first_endpoint_bond_text",
            "first_endpoint_direction_mark", "second_endpoint_text",
            "second_endpoint_bond_text", "second_endpoint_direction_mark",
            "first_canonical_orientation", "second_canonical_orientation",
            "carrier_models", "compatible_second_endpoint_choices",
            "restrictions", "bond_occurrence_parent", "bond_occurrence_child",
            "bond_occurrence_mark", "affected_variables", "affected_factor_keys",
            "propagation_result", "discharged_factor_keys",
            "projected_variables", "successor_snapshot",
            "successor_snapshot_digest",
        ))
    ),
    _PATH_PREFIX + "writer_residual_transition_terms."
    "SharedDirectionalRingEndpointProjectionTransitionTerm": (
        frozenset(
            (
                "kind", "source_snapshot", "source_snapshot_digest", "bond",
                "endpoint_atom", "partner_atom", "ring_label_value",
                "ring_label_text", "endpoint_text", "bond_text",
                "direction_mark", "carrier_models",
                "compatible_second_endpoint_choices", "domain_intersections",
                "affected_variables", "affected_factor_keys",
                "propagation_result", "projected_variables",
                "discharged_factor_keys", "successor_snapshot",
                "successor_snapshot_digest",
            )
        )
    ),
    _PATH_PREFIX + "writer_residual_transition_terms."
    "TetraAtomTokenRestrictionTransitionTerm": (
        frozenset(
            (
                "kind",
                "source_snapshot",
                "source_snapshot_digest",
                "atom",
                "site",
                "token",
                "constraint_var",
                "constraint_value",
                "affected_variables",
                "affected_factor_keys",
                "propagation_result",
                "projected_variables",
                "discharged_factor_keys",
                "successor_snapshot",
                "successor_snapshot_digest",
            )
        )
    ),
    _PATH_PREFIX + "writer_residual_transition_terms."
    "TetraLocalOrderFactorClosureTransitionTerm": (
        frozenset(
            (
                "kind",
                "source_snapshot",
                "source_snapshot_digest",
                "atom",
                "site",
                "local_order",
                "reference_order",
                "target_parity",
                "constraint_var",
                "constraint_value",
                "affected_variables",
                "affected_factor_keys",
                "propagation_result",
                "projected_variables",
                "discharged_factor_keys",
                "successor_snapshot",
                "successor_snapshot_digest",
            )
        )
    ),
}


@dataclass(frozen=True, slots=True)
class WriterSupportArtifactOfflineReplayResult:
    accepted: bool
    checked_object_kinds: tuple[str, ...] = ()
    unchecked_object_kinds: tuple[str, ...] = ()
    checked_relation_families: tuple[str, ...] = ()
    checked_obligation_families: tuple[str, ...] = ()
    unchecked_obligation_families: tuple[str, ...] = ()
    empty_obligation_families: tuple[str, ...] = ()
    offline_replay_complete: bool = False
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class OfflineObligationClassification:
    accepted: bool
    residual_obligations_present: bool = False
    stereo_obligations_present: bool = False
    graph_obligations_present: bool = False
    unchecked_families: tuple[str, ...] = ()
    checked_families: tuple[str, ...] = ()
    checked_empty_families: tuple[str, ...] = ()
    semantically_replayed_operations: tuple[str, ...] = ()
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class CountDagArithmeticVerification:
    accepted: bool
    support_count: int | None = None
    completion_count: int | None = None
    checked_node_kinds: tuple[str, ...] = ()
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class SupportImageCoverageVerification:
    accepted: bool
    support_count: int | None = None
    witness_count: int | None = None
    relation_families: tuple[str, ...] = ()
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class SupportStringReplayPathVerification:
    accepted: bool
    checked_support_strings: int = 0
    checked_projection_steps: int = 0
    relation_families: tuple[str, ...] = ()
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class BranchProjectionIdentityVerification:
    accepted: bool
    checked_text_projections: int = 0
    checked_branch_supports: int = 0
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class GraphRingBranchDeltaVerification:
    accepted: bool
    checked_branches: int = 0
    checked_atom_steps: int = 0
    checked_bond_steps: int = 0
    checked_branch_steps: int = 0
    checked_ring_steps: int = 0
    structural_only_branches: int = 0
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class LocalBranchSuccessorEvidenceVerification:
    accepted: bool
    checked_branches: int = 0
    checked_plain_atom_text_branches: int = 0
    checked_bracket_atom_text_branches: int = 0
    checked_closure_bond_text_branches: int = 0
    checked_directional_coupled_branches: int = 0
    structurally_checked_branches: int = 0
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class TerminalSupportIdentityVerification:
    accepted: bool
    checked_terminal_projections: int = 0
    checked_terminal_supports: int = 0
    checked_terminal_paths: int = 0
    reason: str | None = None


def verify_writer_support_artifact_offline_replay(
    *,
    facts: MoleculeFacts,
    artifact: Mapping[str, object],
    budget: WriterEnvelopeWorkBudget | None = None,
) -> WriterSupportArtifactOfflineReplayResult:
    try:
        objects = _object_by_id(artifact)
        _check_object_kinds_classified(objects)
        root = _require_object(objects, artifact["roots"]["support_image_root"])
        count = _require_object(objects, root["payload"]["count_ref"])
        count_dag = _require_object(objects, count["payload"]["count_dag_ref"])
        arithmetic = verify_count_dag_arithmetic(
            count_dag=count_dag["payload"],
            count_object=count["payload"],
            budget=budget,
        )
        if not arithmetic.accepted:
            _offline_violation(arithmetic.reason or "count_dag_arithmetic_rejected")
        coverage = verify_support_image_coverage_offline(
            artifact=artifact,
            objects=objects,
        )
        if not coverage.accepted:
            _offline_violation(coverage.reason or "support_image_coverage_rejected")
        replay_paths = verify_support_string_replay_paths_offline(
            artifact=artifact,
            objects=objects,
        )
        if not replay_paths.accepted:
            _offline_violation(replay_paths.reason or "support_string_replay_path_rejected")
        branch_identities = verify_branch_projection_identities_offline(
            artifact=artifact,
            objects=objects,
        )
        if not branch_identities.accepted:
            _offline_violation(
                branch_identities.reason or "branch_projection_identity_rejected"
            )
        graph_ring = verify_graph_ring_branch_deltas_offline(
            facts=facts,
            artifact=artifact,
            objects=objects,
            budget=budget,
        )
        if not graph_ring.accepted:
            _offline_violation(
                graph_ring.reason or "graph_ring_branch_delta_rejected"
            )
        local_evidence = verify_local_branch_successor_evidence_offline(
            facts=facts,
            artifact=artifact,
            objects=objects,
            budget=budget,
        )
        if not local_evidence.accepted:
            _offline_violation(
                local_evidence.reason or "local_branch_successor_evidence_rejected"
            )
        terminal = verify_terminal_support_identities_offline(
            artifact=artifact,
            objects=objects,
        )
        if not terminal.accepted:
            _offline_violation(
                terminal.reason or "terminal_support_identity_rejected"
            )
        obligations = classify_residual_stereo_obligations_offline(
            facts=facts,
            artifact=artifact,
            objects=objects,
        )
        if not obligations.accepted:
            _offline_violation(
                obligations.reason or "offline_obligation_classification_rejected"
            )
        checked_object_kinds = {
            "branch_support",
            "count_dag",
            "count_envelope",
            "frontier_product",
            "source_snapshot",
            "support_string",
            "replay_path",
            "support_image",
            "support_image_coverage",
            "terminal_projection",
            "terminal_support",
            "text_projection",
        }
        checked_relations: set[str] = {
            "count_dag_arithmetic",
            *coverage.relation_families,
            *replay_paths.relation_families,
            "branch_projection_identity",
            "graph_ring_branch_delta",
            "local_branch_successor_evidence",
            "terminal_support_identity",
            "residual_stereo_obligation_classification",
            *obligations.checked_empty_families,
        }
        support_refs = root["payload"]["support_string_refs"]
        source = _require_object(objects, artifact["roots"]["source_ref"])
        source_is_initial = (
            source["payload"]["decoder_boundary"]["consumed_token_count"] == 0
        )
        for ref in support_refs:
            support = _require_object(objects, ref)
            _check_support_string_offline(
                facts=facts,
                support=support,
                objects=objects,
                checked_object_kinds=checked_object_kinds,
                checked_relations=checked_relations,
                source_is_initial=source_is_initial,
            )
        unchecked = ()
        replay_complete = not obligations.unchecked_families
        return WriterSupportArtifactOfflineReplayResult(
            accepted=True,
            checked_object_kinds=tuple(sorted(checked_object_kinds)),
            unchecked_object_kinds=unchecked,
            checked_relation_families=tuple(sorted(checked_relations)),
            checked_obligation_families=obligations.checked_families,
            unchecked_obligation_families=obligations.unchecked_families,
            empty_obligation_families=obligations.checked_empty_families,
            offline_replay_complete=replay_complete,
        )
    except SouthStarError as exc:
        return WriterSupportArtifactOfflineReplayResult(
            accepted=False,
            reason=exc.args[-1] if exc.args else "offline_replay_error",
        )
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        return WriterSupportArtifactOfflineReplayResult(
            accepted=False,
            reason=f"malformed_artifact:{type(exc).__name__}",
        )


def validate_writer_bracket_atom_text_against_facts(
    *,
    facts: MoleculeFacts,
    rendered_text: str,
) -> AtomFacts:
    matches = []
    for atom in facts.atoms:
        try:
            if _bracket_atom_text_matches_facts(
                facts=facts,
                atom=atom,
                rendered_text=rendered_text,
            ):
                matches.append(atom)
        except SouthStarError:
            continue
    if len(matches) != 1:
        _offline_violation("bracket_atom_text_facts_mismatch")
    return matches[0]


def _bracket_atom_text_matches_facts(
    *,
    facts: MoleculeFacts,
    atom: AtomFacts,
    rendered_text: str,
) -> bool:
    try:
        return bracket_atom_text(atom) == rendered_text
    except SouthStarError:
        pass
    return _tetra_bracket_atom_text_matches_facts(
        facts=facts,
        atom=atom,
        rendered_text=rendered_text,
    )


def _tetra_bracket_atom_text_matches_facts(
    *,
    facts: MoleculeFacts,
    atom: AtomFacts,
    rendered_text: str,
) -> bool:
    if not rendered_text.startswith("[") or not rendered_text.endswith("]"):
        return False
    inner = rendered_text[1:-1]
    if "@" not in inner:
        return False
    if inner.startswith("C@@H"):
        token = "@@"
        suffix = inner[len("C@@H") :]
    elif inner.startswith("C@H"):
        token = "@"
        suffix = inner[len("C@H") :]
    else:
        return False
    if suffix:
        return False
    if token not in {"@", "@@"}:
        return False
    if atom.symbol != "C":
        return False
    if atom.isotope is not None:
        return False
    if atom.formal_charge != 0:
        return False
    if atom.is_aromatic:
        return False
    if atom.explicit_h_count != 0:
        return False
    if atom.no_implicit:
        return False
    return any(
        site.center == atom.id
        and site.status is SiteStatus.SPECIFIED
        and _tetra_site_h_count(facts=facts, atom=atom, site=site) == 1
        for site in facts.stereo.tetrahedral
    )


def _tetra_site_h_count(
    *,
    facts: MoleculeFacts,
    atom: AtomFacts,
    site: TetrahedralSiteFacts,
) -> int:
    site_occurrence_ids = set(site.ligand_occurrences)
    occurrence_h_count = sum(
        1
        for occurrence in facts.ligand_occurrences
        if occurrence.id in site_occurrence_ids
        and occurrence.atom == atom.id
        and occurrence.kind is LigandKind.IMPLICIT_H
    )
    if atom.implicit_h_count not in {0, 1}:
        return atom.implicit_h_count
    if occurrence_h_count not in {0, 1}:
        return occurrence_h_count
    if atom.implicit_h_count == 1 or occurrence_h_count == 1:
        return 1
    return 0


def verify_count_dag_arithmetic(
    *,
    count_dag: Mapping[str, object],
    count_object: Mapping[str, object],
    budget: WriterEnvelopeWorkBudget | None = None,
) -> CountDagArithmeticVerification:
    try:
        validate_writer_count_certificate_dag_envelope(count_dag, budget=budget)
        nodes = count_dag_node_by_id(dict(count_dag))
        checked: set[str] = set()
        support_root = count_dag["roots"]["support_count_root"]
        completion_root = count_dag["roots"]["completion_count_root"]
        support_count = None if support_root is None else _node_count(
            nodes,
            support_root,
            field="support_count",
            checked=checked,
        )
        completion_count = None if completion_root is None else _node_count(
            nodes,
            completion_root,
            field="completion_count",
            checked=checked,
        )
        if support_count != count_object["support_count"]:
            _offline_violation("count_dag_support_count_mismatch")
        if completion_count != count_object["completion_count"]:
            _offline_violation("count_dag_completion_count_mismatch")
        for node_id in count_dag["roots"]["choice_count_roots"]:
            _check_node_arithmetic(nodes, node_id, checked=checked)
        terminal_root = count_dag["roots"]["terminal_choice_count_root"]
        if terminal_root is not None:
            _check_node_arithmetic(nodes, terminal_root, checked=checked)
        if count_dag["digest"] != count_object["count_dag_digest"]:
            _offline_violation("count_dag_digest_mismatch")
        if count_dag["metrics"]["node_count"] != count_object["count_dag_node_count"]:
            _offline_violation("count_dag_node_count_mismatch")
        if count_dag["metrics"]["edge_count"] != count_object["count_dag_edge_count"]:
            _offline_violation("count_dag_edge_count_mismatch")
        return CountDagArithmeticVerification(
            accepted=True,
            support_count=int(support_count),
            completion_count=int(completion_count),
            checked_node_kinds=tuple(sorted(checked)),
        )
    except SouthStarError as exc:
        return CountDagArithmeticVerification(
            accepted=False,
            reason=exc.args[-1] if exc.args else "count_dag_arithmetic_error",
        )
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        return CountDagArithmeticVerification(
            accepted=False,
            reason=f"malformed_count_dag:{type(exc).__name__}",
        )


def verify_support_image_coverage_offline(
    *,
    artifact: Mapping[str, object],
    objects: Mapping[str, Mapping[str, object]],
) -> SupportImageCoverageVerification:
    try:
        root = _require_object(objects, artifact["roots"]["support_image_root"])
        if root["kind"] != "support_image":
            _offline_violation("support_image_root_kind_mismatch")
        root_payload = root["payload"]
        support_refs = root_payload["support_string_refs"]
        if len(set(support_refs)) != len(support_refs):
            _offline_violation("coverage_duplicate_support_string_ref")
        support_objects = [_require_object(objects, ref) for ref in support_refs]
        support_strings = [item["payload"]["string"] for item in support_objects]
        if root_payload["support_strings"] != support_strings:
            _offline_violation("coverage_support_string_order_mismatch")
        if len(set(support_strings)) != len(support_strings):
            _offline_violation("coverage_duplicate_support_string_text")
        if root_payload["distinct_count"] != len(support_refs):
            _offline_violation("support_image_distinct_count_mismatch")

        count = _require_object(objects, root_payload["count_ref"])
        if count["kind"] != "count_envelope":
            _offline_violation("coverage_count_ref_kind_mismatch")
        count_payload = count["payload"]
        if root_payload["distinct_count"] != count_payload["support_count"]:
            _offline_violation("coverage_count_support_total_mismatch")
        if root_payload["witness_count"] != count_payload["completion_count"]:
            _offline_violation("coverage_count_completion_total_mismatch")

        coverage = _require_object(objects, root_payload["coverage_ref"])
        if coverage["kind"] != "support_image_coverage":
            _offline_violation("coverage_ref_kind_mismatch")
        payload = coverage["payload"]
        if payload["distinct_count"] != root_payload["distinct_count"]:
            _offline_violation("coverage_distinct_count_mismatch")
        if payload["support_count"] != count_payload["support_count"]:
            _offline_violation("coverage_support_count_mismatch")

        assigned: list[str] = []
        support_ref_set = set(support_refs)
        for bucket in payload["text_buckets"]:
            refs = bucket["string_refs"]
            if bucket["support_count"] != len(refs):
                _offline_violation("coverage_text_bucket_count_mismatch")
            for ref in refs:
                if ref not in support_ref_set:
                    _offline_violation("coverage_text_bucket_unknown_ref")
                support = _require_object(objects, ref)
                emitted_texts = support["payload"]["emitted_texts"]
                if not emitted_texts:
                    _offline_violation("coverage_empty_string_in_text_bucket")
                first_projection = _require_object(
                    objects,
                    support["payload"]["text_projection_refs"][0],
                )
                if not _same_text_projection_core(
                    bucket["text_projection"],
                    first_projection["payload"],
                ):
                    _offline_violation("coverage_text_projection_mismatch")
            assigned.extend(refs)

        empty_refs = [
            ref
            for ref in support_refs
            if not _require_object(objects, ref)["payload"]["emitted_texts"]
        ]
        terminal = payload["terminal_bucket"]
        if terminal is None:
            if empty_refs:
                _offline_violation("coverage_terminal_bucket_missing")
        else:
            if terminal["support_count"] != len(empty_refs):
                _offline_violation("coverage_terminal_bucket_count_mismatch")
            if empty_refs:
                if terminal["string_ref"] != empty_refs[0]:
                    _offline_violation("coverage_terminal_string_ref_mismatch")
                support = _require_object(objects, empty_refs[0])
                terminal_projection = _require_object(
                    objects,
                    support["payload"]["terminal_projection_ref"],
                )
                if terminal["terminal_projection"] != terminal_projection["payload"]:
                    _offline_violation("coverage_terminal_projection_mismatch")
                assigned.extend(empty_refs)
            elif terminal["string_ref"] is not None:
                _offline_violation("coverage_terminal_unexpected_string_ref")

        if len(assigned) != len(set(assigned)):
            _offline_violation("coverage_duplicate_assignment")
        if sorted(assigned) != sorted(support_refs):
            _offline_violation("coverage_partition_mismatch")
        total = sum(int(bucket["support_count"]) for bucket in payload["text_buckets"])
        if terminal is not None:
            total += int(terminal["support_count"])
        if total != root_payload["distinct_count"]:
            _offline_violation("coverage_support_total_mismatch")
        return SupportImageCoverageVerification(
            accepted=True,
            support_count=int(root_payload["distinct_count"]),
            witness_count=int(root_payload["witness_count"]),
            relation_families=("support_image_coverage",),
        )
    except SouthStarError as exc:
        return SupportImageCoverageVerification(
            accepted=False,
            reason=exc.args[-1] if exc.args else "support_image_coverage_error",
        )
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        return SupportImageCoverageVerification(
            accepted=False,
            reason=f"malformed_coverage:{type(exc).__name__}",
        )


def verify_support_string_replay_paths_offline(
    *,
    artifact: Mapping[str, object],
    objects: Mapping[str, Mapping[str, object]],
) -> SupportStringReplayPathVerification:
    try:
        root = _require_object(objects, artifact["roots"]["support_image_root"])
        source = _require_object(objects, root["payload"]["source_ref"])
        if source["kind"] != "source_snapshot":
            _offline_violation("replay_path_source_ref_kind_mismatch")
        source_cursor = source["payload"]["cursor"]
        checked_steps = 0
        for ref in root["payload"]["support_string_refs"]:
            support = _require_object(objects, ref)
            _check_support_string_replay_path(
                support=support,
                source_cursor=source_cursor,
                objects=objects,
            )
            checked_steps += len(support["payload"]["text_projection_refs"])
        return SupportStringReplayPathVerification(
            accepted=True,
            checked_support_strings=len(root["payload"]["support_string_refs"]),
            checked_projection_steps=checked_steps,
            relation_families=("support_string_replay_path",),
        )
    except SouthStarError as exc:
        return SupportStringReplayPathVerification(
            accepted=False,
            reason=exc.args[-1] if exc.args else "support_string_replay_path_error",
        )
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        return SupportStringReplayPathVerification(
            accepted=False,
            reason=f"malformed_replay_path:{type(exc).__name__}",
        )


def verify_branch_projection_identities_offline(
    *,
    artifact: Mapping[str, object],
    objects: Mapping[str, Mapping[str, object]],
) -> BranchProjectionIdentityVerification:
    try:
        root = _require_object(objects, artifact["roots"]["support_image_root"])
        seen_projection_refs: set[str] = set()
        checked_branch_refs: set[str] = set()
        for support_ref in root["payload"]["support_string_refs"]:
            support = _require_object(objects, support_ref)
            for projection_ref in support["payload"]["text_projection_refs"]:
                if projection_ref in seen_projection_refs:
                    continue
                seen_projection_refs.add(projection_ref)
                projection = _require_object(objects, projection_ref)
                if projection["kind"] != "text_projection":
                    _offline_violation("branch_projection_text_ref_kind_mismatch")
                _check_text_projection_branch_identities(
                    projection=projection,
                    objects=objects,
                    checked_branch_refs=checked_branch_refs,
                )
        return BranchProjectionIdentityVerification(
            accepted=True,
            checked_text_projections=len(seen_projection_refs),
            checked_branch_supports=len(checked_branch_refs),
        )
    except SouthStarError as exc:
        return BranchProjectionIdentityVerification(
            accepted=False,
            reason=exc.args[-1] if exc.args else "branch_projection_identity_error",
        )
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        return BranchProjectionIdentityVerification(
            accepted=False,
            reason=f"malformed_branch_projection_identity:{type(exc).__name__}",
        )


def verify_transition_branch_projection_identity_offline(
    *, projection_ref: str, branch_ref: str, objects: Mapping[str, Mapping[str, object]]
) -> BranchProjectionIdentityVerification:
    try:
        projection = _require_object(objects, projection_ref)
        if projection["payload"]["branch_support_refs"] != [branch_ref]:
            _offline_violation("branch_projection_selected_ref_mismatch")
        checked_branch_refs: set[str] = set()
        _check_text_projection_branch_identities(
            projection=projection,
            objects=objects,
            checked_branch_refs=checked_branch_refs,
        )
        return BranchProjectionIdentityVerification(
            accepted=True,
            checked_text_projections=1,
            checked_branch_supports=len(checked_branch_refs),
        )
    except SouthStarError as exc:
        return BranchProjectionIdentityVerification(
            accepted=False,
            reason=exc.args[-1] if exc.args else "branch_projection_identity_error",
        )
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        return BranchProjectionIdentityVerification(
            accepted=False,
            reason=f"malformed_branch_projection_identity:{type(exc).__name__}",
        )


def verify_local_branch_successor_evidence_offline(
    *,
    facts: MoleculeFacts,
    artifact: Mapping[str, object],
    objects: Mapping[str, Mapping[str, object]],
    budget: WriterEnvelopeWorkBudget | None = None,
    branch_refs: tuple[str, ...] | None = None,
) -> LocalBranchSuccessorEvidenceVerification:
    try:
        budget = default_writer_envelope_work_budget(budget)
        if branch_refs is None:
            branch_refs = _branch_refs_from_support_artifact(artifact=artifact, objects=objects)
        plain_atom_count = 0
        bracket_atom_count = 0
        closure_count = 0
        directional_count = 0
        structural_count = 0
        for branch_ref in branch_refs:
            branch = _require_object(objects, branch_ref)
            if branch["kind"] != "branch_support":
                _offline_violation("local_branch_support_ref_kind_mismatch")
            kind = _check_branch_local_evidence(
                facts=facts,
                branch=branch,
                budget=budget,
            )
            if kind == "plain_atom_text":
                plain_atom_count += 1
            elif kind == "bracket_atom_text":
                bracket_atom_count += 1
            elif kind == "closure_bond_text":
                closure_count += 1
            elif kind == "directional_ring_closure_bond_text":
                directional_count += 1
            elif kind == "other_structural":
                structural_count += 1
        return LocalBranchSuccessorEvidenceVerification(
            accepted=True,
            checked_branches=len(branch_refs),
            checked_plain_atom_text_branches=plain_atom_count,
            checked_bracket_atom_text_branches=bracket_atom_count,
            checked_closure_bond_text_branches=closure_count,
            checked_directional_coupled_branches=directional_count,
            structurally_checked_branches=structural_count,
        )
    except SouthStarError as exc:
        return LocalBranchSuccessorEvidenceVerification(
            accepted=False,
            reason=exc.args[-1] if exc.args else "local_branch_evidence_error",
        )
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        return LocalBranchSuccessorEvidenceVerification(
            accepted=False,
            reason=f"malformed_local_branch_evidence:{type(exc).__name__}",
        )


def verify_graph_ring_branch_deltas_offline(
    *,
    facts: MoleculeFacts,
    artifact: Mapping[str, object],
    objects: Mapping[str, Mapping[str, object]],
    budget: WriterEnvelopeWorkBudget | None = None,
    branch_refs: tuple[str, ...] | None = None,
) -> GraphRingBranchDeltaVerification:
    try:
        budget = default_writer_envelope_work_budget(budget)
        if branch_refs is None:
            branch_refs = _branch_refs_from_support_artifact(artifact=artifact, objects=objects)
        atom_count = 0
        bond_count = 0
        branch_count = 0
        ring_count = 0
        structural_count = 0
        for branch_ref in branch_refs:
            branch = _require_object(objects, branch_ref)
            if branch["kind"] != "branch_support":
                _offline_violation("graph_ring_branch_support_ref_kind_mismatch")
            kind = _check_graph_ring_branch_delta(
                facts=facts,
                branch=branch,
                budget=budget,
            )
            if kind in ("atom_start", "atom_advance"):
                atom_count += 1
            elif kind == "bond_advance":
                bond_count += 1
            elif kind in ("branch_open", "branch_return"):
                branch_count += 1
            elif kind in (
                "ring_endpoint_open",
                "ring_endpoint_pair",
                "ring_endpoint_pair_non_single",
            ):
                ring_count += 1
            elif kind == "other_structural":
                structural_count += 1
        return GraphRingBranchDeltaVerification(
            accepted=True,
            checked_branches=len(branch_refs),
            checked_atom_steps=atom_count,
            checked_bond_steps=bond_count,
            checked_branch_steps=branch_count,
            checked_ring_steps=ring_count,
            structural_only_branches=structural_count,
        )
    except SouthStarError as exc:
        return GraphRingBranchDeltaVerification(
            accepted=False,
            reason=exc.args[-1] if exc.args else "graph_ring_branch_delta_error",
        )
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        return GraphRingBranchDeltaVerification(
            accepted=False,
            reason=f"malformed_graph_ring_branch_delta:{type(exc).__name__}",
        )


def classify_residual_stereo_obligations_offline(
    *,
    facts: MoleculeFacts,
    artifact: Mapping[str, object],
    objects: Mapping[str, Mapping[str, object]],
    branch_refs: tuple[str, ...] | None = None,
    include_terminals: bool = True,
) -> OfflineObligationClassification:
    try:
        manifests_by_family = {
            "residual_work": [],
            "finite_relation_work": [],
            "graph_obligation_work": [],
            "stereo_lifecycle": [],
            "residual_attachment_lifecycle": [],
            "closure_candidate_lifecycle": [],
            "directional_ring_closure_lifecycle": [],
            "terminal_residual_work": [],
            "terminal_stereo_lifecycle": [],
            "terminal_graph_obligation_work": [],
        }
        replayed_residual_digests: set[str] = set()
        replayed_lifecycle_digests: set[str] = set()
        replayed_directional_ring_closure_digests: set[str] = set()
        replayed_operations: list[str] = []
        ring_endpoint_choices = _ring_endpoint_choices_from_artifact(
            artifact=artifact,
            objects=objects,
        )
        if branch_refs is None:
            branch_refs = _branch_refs_from_support_artifact(artifact=artifact, objects=objects)
        for branch_ref in branch_refs:
            branch = _require_object(objects, branch_ref)
            _check_branch_obligation_ring_summaries(branch)
            for family, items in branch["payload"]["obligation_manifests"].items():
                if family == "residual_work":
                    manifests_by_family[family].extend(
                        _classify_branch_residual_work_manifests(
                            facts=facts,
                            branch=branch,
                            items=items,
                            objects=objects,
                            replayed_residual_digests=replayed_residual_digests,
                            replayed_operations=replayed_operations,
                            ring_endpoint_choices=ring_endpoint_choices,
                        )
                    )
                else:
                    manifests_by_family[family].extend(items)
            _classify_branch_replayed_lifecycle_manifests(
                branch=branch,
                replayed_residual_digests=replayed_residual_digests,
                replayed_lifecycle_digests=replayed_lifecycle_digests,
            )
            _classify_branch_directional_ring_closure_lifecycles(
                branch=branch,
                facts=facts,
                objects=objects,
                replayed_residual_digests=replayed_residual_digests,
                replayed_lifecycle_digests=replayed_lifecycle_digests,
                replayed_directional_ring_closure_digests=(
                    replayed_directional_ring_closure_digests
                ),
            )
        if include_terminals:
            root = _require_object(objects, artifact["roots"]["support_image_root"])
            for support_ref in root["payload"]["support_string_refs"]:
                support = _require_object(objects, support_ref)
                for terminal_ref in support["payload"]["terminal_support_refs"]:
                    terminal = _require_object(objects, terminal_ref)
                    for family, items in terminal["payload"]["obligation_manifests"].items():
                        manifests_by_family[family].extend(items)
        unchecked = tuple(dict.fromkeys(
            _unchecked_obligation_family_name(family, items, facts=facts)
            for family, items in sorted(manifests_by_family.items())
            if items and not _obligation_manifests_checked(
                items,
                replayed_residual_digests=replayed_residual_digests,
                replayed_lifecycle_digests=replayed_lifecycle_digests,
                replayed_directional_ring_closure_digests=replayed_directional_ring_closure_digests,
            )
        ))
        checked = tuple(
            family
            for family, items in sorted(manifests_by_family.items())
            if items and _obligation_manifests_checked(
                items,
                replayed_residual_digests=replayed_residual_digests,
                replayed_lifecycle_digests=replayed_lifecycle_digests,
                replayed_directional_ring_closure_digests=replayed_directional_ring_closure_digests,
            )
        )
        checked_empty = tuple(
            f"{family}_checked_empty"
            for family, items in sorted(manifests_by_family.items())
            if not items
        )
        return OfflineObligationClassification(
            accepted=True,
            residual_obligations_present=any(
                manifests_by_family[family]
                for family in (
                    "residual_work",
                    "residual_attachment_lifecycle",
                    "terminal_residual_work",
                )
            ),
            stereo_obligations_present=any(
                manifests_by_family[family]
                for family in (
                    "stereo_lifecycle",
                    "directional_ring_closure_lifecycle",
                    "terminal_stereo_lifecycle",
                )
            ),
            graph_obligations_present=any(
                manifests_by_family[family]
                for family in (
                    "graph_obligation_work",
                    "terminal_graph_obligation_work",
                )
            ),
            unchecked_families=unchecked,
            checked_families=checked,
            checked_empty_families=checked_empty,
            semantically_replayed_operations=tuple(replayed_operations),
        )
    except SouthStarError as exc:
        return OfflineObligationClassification(
            accepted=False,
            reason=exc.args[-1] if exc.args else "offline_obligation_error",
        )
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        return OfflineObligationClassification(
            accepted=False,
            reason=f"malformed_offline_obligation:{type(exc).__name__}",
        )


def verify_branch_obligations_offline(
    *,
    facts: MoleculeFacts,
    artifact: Mapping[str, object],
    objects: Mapping[str, Mapping[str, object]],
    branch_ref: str,
) -> OfflineObligationClassification:
    """Replay obligation evidence for one explicitly selected branch only."""
    return classify_residual_stereo_obligations_offline(
        facts=facts,
        artifact=artifact,
        objects=objects,
        branch_refs=(branch_ref,),
        include_terminals=False,
    )


def _ring_endpoint_choices_from_artifact(
    *,
    artifact: Mapping[str, object],
    objects: Mapping[str, Mapping[str, object]],
) -> dict[int, tuple[tuple[str, DirectionMark], ...]]:
    del objects
    policy = _term_field_value(artifact["prepared_identity"]["terms"], "policy")
    domains = policy[4]
    out = {}
    for bond, slot_kind, choices in domains:
        if slot_kind != "ring_endpoint":
            continue
        expanded = []
        for _name, base_text, permits_direction in choices:
            expanded.append((base_text, DirectionMark.ABSENT))
            if permits_direction:
                expanded.extend(((base_text, DirectionMark.FWD), (base_text, DirectionMark.REV)))
        if bond in out:
            _offline_violation("directional_non_single_ring_policy_domain_duplicate")
        out[bond] = tuple(expanded)
    return out


def _obligation_manifests_checked(
    items: list[object],
    *,
    replayed_residual_digests: set[str],
    replayed_lifecycle_digests: set[str],
    replayed_directional_ring_closure_digests: set[str],
) -> bool:
    return all(
        _obligation_manifest_checked(
            item,
            replayed_residual_digests=replayed_residual_digests,
            replayed_lifecycle_digests=replayed_lifecycle_digests,
            replayed_directional_ring_closure_digests=replayed_directional_ring_closure_digests,
        )
        for item in items
    )


def _obligation_manifest_checked(
    item: Mapping[str, object],
    *,
    replayed_residual_digests: set[str],
    replayed_lifecycle_digests: set[str],
    replayed_directional_ring_closure_digests: set[str],
) -> bool:
    family = item["family"]
    if family == "residual_work":
        return item["evidence_digest"] in replayed_residual_digests
    if family == "stereo_lifecycle":
        linked = tuple(item["linked_residual_work_digests"])
        if linked:
            return item["evidence_digest"] in replayed_lifecycle_digests
        return bool(
            item["is_noop"]
            or item["is_empty"]
            or item["is_discharged"]
            or item["terminal_clean"]
            or item["evidence_digest"] in replayed_lifecycle_digests
        )
    if family == "directional_ring_closure_lifecycle":
        return item["evidence_digest"] in replayed_directional_ring_closure_digests
    return bool(
        item["is_noop"]
        or item["is_empty"]
        or item["is_discharged"]
        or item["terminal_clean"]
        or _ring_obligation_manifest_checked(item)
    )


def _classify_branch_residual_work_manifests(
    *,
    facts: MoleculeFacts,
    branch: Mapping[str, object],
    items: list[object],
    objects: Mapping[str, Mapping[str, object]],
    replayed_residual_digests: set[str],
    replayed_operations: list[str],
    ring_endpoint_choices: Mapping[int, tuple[tuple[str, DirectionMark], ...]],
) -> tuple[Mapping[str, object], ...]:
    _check_branch_residual_lifecycle_links(branch=branch, residual_items=items)
    for item in items:
        disposition = _validate_tetra_residual_manifest_if_known(
            facts=facts,
            branch=branch,
            item=item,
            objects=objects,
            ring_endpoint_choices=ring_endpoint_choices,
        )
        if disposition is OfflineResidualReplayDisposition.SEMANTICALLY_REPLAYED:
            replayed_residual_digests.add(item["evidence_digest"])
            replayed_operations.append(item["operation"])
    return tuple(items)


def _classify_branch_replayed_lifecycle_manifests(
    *,
    branch: Mapping[str, object],
    replayed_residual_digests: set[str],
    replayed_lifecycle_digests: set[str],
) -> None:
    residual_items = branch["payload"]["obligation_manifests"]["residual_work"]
    residual_by_digest = {item["evidence_digest"]: item for item in residual_items}
    for lifecycle in branch["payload"]["obligation_manifests"]["stereo_lifecycle"]:
        linked = lifecycle["linked_residual_work_digests"]
        if not linked:
            continue
        if any(digest not in residual_by_digest for digest in linked):
            _offline_violation("residual_lifecycle_replayed_digest_missing")
        if lifecycle["residual_work_digests"] != linked:
            _offline_violation("residual_lifecycle_replayed_digest_mismatch")
        if any(digest not in replayed_residual_digests for digest in linked):
            continue
        linked_items = [residual_by_digest[digest] for digest in linked]
        expected_operations = [item["operation"] for item in linked_items]
        if lifecycle["residual_work_operations"] != expected_operations:
            _offline_violation("residual_lifecycle_replayed_operation_mismatch")
        replayed_lifecycle_digests.add(lifecycle["evidence_digest"])


def _classify_branch_directional_ring_closure_lifecycles(
    *,
    branch: Mapping[str, object],
    facts: MoleculeFacts,
    objects: Mapping[str, Mapping[str, object]],
    replayed_residual_digests: set[str],
    replayed_lifecycle_digests: set[str],
    replayed_directional_ring_closure_digests: set[str],
) -> None:
    payload = branch["payload"]
    items = payload["obligation_manifests"]["directional_ring_closure_lifecycle"]
    if not items:
        return
    local = payload["local_evidence"]
    if local["kind"] != "directional_ring_closure_bond_text":
        _offline_violation("directional_non_single_ring_coupled_digest_mismatch")
    expected = [item["coupling_term_digest"] for item in items]
    if len(expected) != len(set(expected)):
        _offline_violation("directional_ring_coupling_duplicate")
    if local["manifest"]["directional_coupled_digests"] != expected:
        _offline_violation("directional_non_single_ring_coupled_digest_mismatch")
    residual_by_digest = {
        value["evidence_digest"]: value
        for value in payload["obligation_manifests"]["residual_work"]
    }
    lifecycle_by_digest = {
        value["evidence_digest"]: value
        for value in payload["obligation_manifests"]["stereo_lifecycle"]
    }
    source_state, successor_state = _branch_writer_state_terms(
        branch=branch,
        objects=objects,
    )
    source_ring = _term_field_value(source_state, "ring_state")
    successor_ring = _term_field_value(successor_state, "ring_state")
    source_residual = _term_field_value(_term_field_value(source_state, "stereo_state"), "residual_snapshot")
    successor_residual = _term_field_value(_term_field_value(successor_state, "stereo_state"), "residual_snapshot")
    closure_manifests = local["manifest"]["closure_bond_text"]
    events = [
        event for event in payload["graph_ring_delta"]["manifest"]["event_manifests"]
        if event["kind"] in ("ring_endpoint_emitted", "ring_endpoint_paired")
    ]
    if len(events) != 1:
        _offline_violation("directional_ring_coupling_event_mismatch")
    event = events[0]
    for item in items:
        term = item["coupling_term"]
        if item["coupling_term_digest"] != _closed_term_digest(term):
            _offline_violation("directional_non_single_ring_coupled_digest_mismatch")
        if (
            _term_field_value(term, "source_state_digest") != payload["source_state_digest"]
            or _term_field_value(term, "successor_state_digest") != payload["successor_state_digest"]
        ):
            _offline_violation("directional_ring_coupling_state_mismatch")
        if (
            _term_field_value(term, "source_ring_state_digest") != _closed_term_digest(source_ring)
            or _term_field_value(term, "successor_ring_state_digest") != _closed_term_digest(successor_ring)
        ):
            _offline_violation("directional_ring_coupling_ring_state_mismatch")
        if (
            _term_field_value(term, "source_residual_snapshot_digest") != _closed_term_digest(source_residual)
            or _term_field_value(term, "successor_residual_snapshot_digest") != _closed_term_digest(successor_residual)
        ):
            _offline_violation("directional_ring_coupling_residual_state_mismatch")
        event_kind = _term_field_value(term, "event_kind")
        bond = _term_field_value(term, "bond")
        label = event["label"]
        expected_opening_atom = (
            event["endpoint_atom"]
            if event_kind == "ring_endpoint_emitted"
            else event["partner_atom"]
        )
        expected_closing_atom = (
            event["partner_atom"]
            if event_kind == "ring_endpoint_emitted"
            else event["endpoint_atom"]
        )
        if (
            event_kind != event["kind"]
            or bond != event["bond"]
            or _term_field_value(term, "bond_order") != "double"
            or _term_field_value(term, "label_value") != _term_field_value(label, "value")
            or _term_field_value(term, "label_text") != _term_field_value(label, "text")
            or _term_field_value(term, "opening_atom") != expected_opening_atom
            or _term_field_value(term, "closing_atom") != expected_closing_atom
        ):
            _offline_violation("directional_ring_coupling_event_identity_mismatch")
        graph_bond = _facts_bond(facts=facts, bond=bond)
        if graph_bond.order is not BondOrder.DOUBLE or _facts_bond_is_bridge(facts=facts, bond=bond):
            _offline_violation("directional_ring_coupling_event_mismatch")
        closure_event_kind = (
            "endpoint_emitted"
            if event_kind == "ring_endpoint_emitted"
            else "endpoint_paired"
        )
        matching_closures = [
            closure for closure in closure_manifests
            if closure["bond"] == bond
            and closure["event_kind"] == closure_event_kind
            and closure["label"] == label
            and closure["opening_atom"] == _term_field_value(term, "opening_atom")
            and closure["closing_atom"] == _term_field_value(term, "closing_atom")
        ]
        if len(matching_closures) != 1:
            _offline_violation("directional_ring_coupling_closure_manifest_mismatch")
        closure = matching_closures[0]
        if _term_field_value(term, "closure_manifest_digest") != _identity_digest(closure):
            _offline_violation("directional_ring_coupling_closure_manifest_mismatch")
        if (
            _term_field_value(term, "opening_marker") != closure["opening_marker"]
            or _term_field_value(term, "closing_marker") != closure["closing_marker"]
            or _term_field_value(term, "marker_side") != closure["marker_side"]
            or closure["bond_order"] != "double"
            or sorted((closure["opening_marker"], closure["closing_marker"])) != ["", "="]
        ):
            _offline_violation("directional_ring_coupling_marker_mismatch")
        stereo_digest = _term_field_value(term, "stereo_lifecycle_digest")
        residual_digests = tuple(_term_field_value(term, "residual_work_digests"))
        lifecycle = lifecycle_by_digest.get(stereo_digest)
        if lifecycle is None:
            _offline_violation("directional_ring_coupling_lifecycle_branch_mismatch")
        if stereo_digest not in replayed_lifecycle_digests:
            continue
        expected_operation = (
            "directional ring endpoint projection"
            if event_kind == "ring_endpoint_emitted"
            else "directional ring pair restriction"
        )
        if (
            lifecycle["operation"] != "WriterStereoLifecycleEvidence"
            or lifecycle["lifecycle_event_kind"] != event_kind
            or lifecycle["source_digest"] != payload["source_state_digest"]
            or lifecycle["successor_digest"] != payload["successor_state_digest"]
            or lifecycle["source_residual_snapshot_digest"] != _term_field_value(term, "source_residual_snapshot_digest")
            or lifecycle["successor_residual_snapshot_digest"] != _term_field_value(term, "successor_residual_snapshot_digest")
        ):
            _offline_violation("directional_ring_coupling_lifecycle_branch_mismatch")
        if lifecycle["residual_work_digests"] != list(residual_digests):
            _offline_violation("directional_ring_coupling_residual_state_mismatch")
        if any(digest not in residual_by_digest for digest in residual_digests):
            _offline_violation("directional_ring_coupling_residual_branch_mismatch")
        if any(digest not in replayed_residual_digests for digest in residual_digests):
            continue
        if (
            len(residual_digests) != 1
            or residual_by_digest[residual_digests[0]]["operation"] != expected_operation
            or lifecycle["residual_work_operations"] != [expected_operation]
        ):
            _offline_violation("directional_ring_coupling_residual_branch_mismatch")
        closed_digest = _term_field_value(term, "closed_closure_record_digest")
        if closure["closed_closure_record_digest"] != closed_digest:
            _offline_violation("directional_ring_coupling_closed_record_mismatch")
        if event_kind == "ring_endpoint_emitted":
            if closed_digest is not None:
                _offline_violation("directional_ring_coupling_closed_record_mismatch")
            source_open = [
                record for record in _term_field_value(source_ring, "open_endpoints")
                if int(_term_field_value(record, "bond")) == int(bond)
            ]
            successor_open = [
                record for record in _term_field_value(successor_ring, "open_endpoints")
                if int(_term_field_value(record, "bond")) == int(bond)
            ]
            if source_open or len(successor_open) != 1:
                _offline_violation("directional_ring_coupling_closed_record_mismatch")
            opened = successor_open[0]
            if (
                _term_field_value(opened, "first_atom") != expected_opening_atom
                or _term_field_value(opened, "second_atom") != expected_closing_atom
                or _term_field_value(opened, "label") != label
                or _term_field_value(opened, "first_endpoint_bond_text") != closure["opening_marker"]
            ):
                _offline_violation("directional_ring_coupling_event_identity_mismatch")
        else:
            closed = [
                record for record in _term_field_value(successor_ring, "closed_closures")
                if int(_term_field_value(record, "bond")) == int(bond)
            ]
            if len(closed) != 1 or closed_digest != _closed_term_digest(closed[0]):
                _offline_violation("directional_ring_coupling_closed_record_mismatch")
            record = closed[0]
            if (
                _term_field_value(record, "first_atom") != expected_opening_atom
                or _term_field_value(record, "second_atom") != expected_closing_atom
                or _term_field_value(record, "label") != label
                or _term_field_value(record, "first_endpoint_bond_text") != closure["opening_marker"]
                or _term_field_value(record, "second_endpoint_bond_text") != closure["closing_marker"]
            ):
                _offline_violation("directional_ring_coupling_closed_record_mismatch")
        replayed_directional_ring_closure_digests.add(item["evidence_digest"])


def _check_branch_residual_lifecycle_links(
    *,
    branch: Mapping[str, object],
    residual_items: list[object],
) -> None:
    residual_by_digest = {}
    for residual in residual_items:
        digest = residual["evidence_digest"]
        if digest in residual_by_digest:
            _offline_violation("residual_lifecycle_residual_digest_duplicate")
        residual_by_digest[digest] = residual
        links = residual["linked_lifecycle_digests"]
        if len(set(links)) != len(links):
            _offline_violation("residual_lifecycle_forward_link_duplicate")
    lifecycle_by_digest = {}
    lifecycle_items = branch["payload"]["obligation_manifests"]["stereo_lifecycle"]
    for lifecycle in lifecycle_items:
        digest = lifecycle["evidence_digest"]
        if digest in lifecycle_by_digest:
            _offline_violation("residual_lifecycle_lifecycle_digest_duplicate")
        lifecycle_by_digest[digest] = lifecycle
        links = lifecycle["linked_residual_work_digests"]
        if len(set(links)) != len(links):
            _offline_violation("residual_lifecycle_reverse_link_duplicate")
        if links != lifecycle["residual_work_digests"]:
            _offline_violation("residual_lifecycle_reverse_link_provenance_mismatch")

    for residual_digest, residual in residual_by_digest.items():
        expected_links = [
            lifecycle["evidence_digest"]
            for lifecycle in lifecycle_items
            if residual_digest in lifecycle["residual_work_digests"]
        ]
        if residual["linked_lifecycle_digests"] != expected_links:
            _offline_violation("residual_lifecycle_forward_link_provenance_mismatch")
        for lifecycle_digest in residual["linked_lifecycle_digests"]:
            lifecycle = lifecycle_by_digest.get(lifecycle_digest)
            if lifecycle is None:
                _offline_violation("residual_lifecycle_forward_link_missing")
            if residual_digest not in lifecycle["linked_residual_work_digests"]:
                _offline_violation("residual_lifecycle_forward_link_unreciprocated")

    for lifecycle_digest, lifecycle in lifecycle_by_digest.items():
        for residual_digest in lifecycle["linked_residual_work_digests"]:
            residual = residual_by_digest.get(residual_digest)
            if residual is None:
                _offline_violation("residual_lifecycle_reverse_link_missing")
            if lifecycle_digest not in residual["linked_lifecycle_digests"]:
                _offline_violation("residual_lifecycle_reverse_link_unreciprocated")


def _validate_tetra_residual_manifest_if_known(
    *,
    facts: MoleculeFacts,
    branch: Mapping[str, object],
    item: Mapping[str, object],
    objects: Mapping[str, Mapping[str, object]],
    ring_endpoint_choices: Mapping[int, tuple[tuple[str, DirectionMark], ...]],
) -> OfflineResidualReplayDisposition:
    operation = item["operation"]
    if operation == "tetrahedral atom-token restriction":
        _check_tetra_residual_manifest_core(
            branch=branch,
            item=item,
            expected_event_kind="atom_emitted",
            expected_capability="tetra_token_restriction",
            expected_lifecycle_capabilities=(
                "residual_propagation",
                "tetra_token_restriction",
            ),
            expected_certificate_kind="tetra_token_restricted",
            expected_changed_field="residual_snapshot_changed",
        )
        _replay_tetra_atom_token_transition(
            branch=branch,
            item=item,
            facts=facts,
            objects=objects,
        )
        return OfflineResidualReplayDisposition.SEMANTICALLY_REPLAYED
    if operation == "directional ring pair restriction":
        if item["transition_term"] is None:
            if _directional_ring_pair_transition_term_required_offline(
                facts=facts,
                branch=branch,
                ring_endpoint_choices=ring_endpoint_choices,
            ):
                _offline_violation("directional_ring_pair_transition_missing")
            return OfflineResidualReplayDisposition.DECLARED_OUT_OF_SCOPE
        _check_directional_ring_pair_manifest_core(
            branch=branch,
            item=item,
            facts=facts,
        )
        _replay_directional_ring_pair_transition(
            branch=branch,
            item=item,
            facts=facts,
            objects=objects,
            ring_endpoint_choices=ring_endpoint_choices,
        )
        return OfflineResidualReplayDisposition.SEMANTICALLY_REPLAYED
    if operation == "tetrahedral local-order factor closure":
        _check_tetra_residual_manifest_core(
            branch=branch,
            item=item,
            expected_event_kind="local_order_closed",
            expected_capability="tetra_local_order_restriction",
            expected_lifecycle_capabilities=(
                "residual_factor_discharge",
                "residual_propagation",
                "tetra_local_order_restriction",
            ),
            expected_certificate_kind="tetra_local_order_restricted",
            expected_changed_field="local_orders_changed",
        )
        _check_tetra_local_order_residual(branch=branch, facts=facts)
        _replay_tetra_local_order_transition(
            branch=branch,
            item=item,
            facts=facts,
            objects=objects,
        )
        return OfflineResidualReplayDisposition.SEMANTICALLY_REPLAYED
    if operation == "directional carrier-mark restriction":
        expected_lifecycle_capabilities = (
            "directional_carrier_restriction",
            "directional_site_compatibility",
            "residual_factor_discharge",
            "residual_propagation",
        )
        if _directional_carrier_transition_site_count_offline(
            facts=facts,
            branch=branch,
        ) == 2:
            expected_lifecycle_capabilities = (
                "directional_carrier_restriction",
                "directional_site_compatibility",
                "residual_factor_discharge",
                "residual_propagation",
                "shared_directional_carrier_restriction",
            )
        _check_tetra_residual_manifest_core(
            branch=branch,
            item=item,
            expected_event_kind="bond_emitted",
            expected_capability="directional_carrier_restriction",
            expected_lifecycle_capabilities=expected_lifecycle_capabilities,
            expected_certificate_kind="directional_carrier_restricted",
            expected_changed_field="residual_snapshot_changed",
        )
        if item["transition_term"] is None:
            if _directional_carrier_transition_term_required_offline(
                facts=facts,
                branch=branch,
            ):
                _offline_violation("tetra_residual_transition_missing")
            return OfflineResidualReplayDisposition.DECLARED_OUT_OF_SCOPE
        _replay_directional_carrier_transition(
            branch=branch,
            item=item,
            facts=facts,
            objects=objects,
        )
        return OfflineResidualReplayDisposition.SEMANTICALLY_REPLAYED
    if operation == "directional ring endpoint projection":
        _check_directional_ring_projection_manifest_core(
            branch=branch,
            item=item,
            facts=facts,
        )
        term_required = _directional_ring_endpoint_transition_term_required_offline(
            facts=facts,
            branch=branch,
            ring_endpoint_choices=ring_endpoint_choices,
        )
        if not term_required:
            return OfflineResidualReplayDisposition.DECLARED_OUT_OF_SCOPE
        if item["transition_term"] is None:
            _offline_violation("directional_ring_projection_transition_missing")
        _replay_directional_ring_endpoint_projection_transition(
            branch=branch,
            item=item,
            facts=facts,
            objects=objects,
            ring_endpoint_choices=ring_endpoint_choices,
        )
        return OfflineResidualReplayDisposition.SEMANTICALLY_REPLAYED
    return OfflineResidualReplayDisposition.DECLARED_OUT_OF_SCOPE


def _directional_ring_endpoint_transition_term_required_offline(
    *,
    facts: MoleculeFacts,
    branch: Mapping[str, object],
    ring_endpoint_choices: Mapping[int, tuple[tuple[str, DirectionMark], ...]],
) -> bool:
    delta = branch["payload"]["graph_ring_delta"]
    events = [
        event for event in delta["manifest"]["event_manifests"]
        if event["kind"] == "ring_endpoint_emitted"
    ]
    if delta["kind"] != "ring_endpoint_open" or len(events) != 1:
        return False
    event = events[0]
    bond = _facts_bond(facts=facts, bond=event["bond"])
    if bond.order not in (BondOrder.SINGLE, BondOrder.DOUBLE) or _facts_bond_is_bridge(
        facts=facts,
        bond=event["bond"],
    ):
        return False
    choices = ring_endpoint_choices.get(event["bond"], ())
    if bond.order is BondOrder.SINGLE:
        if event["bond_text"] != "" or set(choices) != {
            ("", DirectionMark.ABSENT), ("", DirectionMark.FWD), ("", DirectionMark.REV)
        }:
            return False
    elif (
        event["bond_text"] not in ("", "=")
        or event["direction_mark"]["value"] != DirectionMark.ABSENT.value
        or set(choices) != {("", DirectionMark.ABSENT), ("=", DirectionMark.ABSENT)}
    ):
        return False
    sites = _directional_sites_for_facts_bond(facts=facts, bond=event["bond"])
    if len(sites) not in (1, 2) or any(
        site.status is not SiteStatus.SPECIFIED for site in sites
    ):
        return False
    models = _expected_directional_models_for_facts_bond(
        facts=facts,
        sites=sites,
        bond=event["bond"],
    )
    if bond.order is BondOrder.DOUBLE:
        return len(sites) == 1 and len(models) == 1
    return len(sites) in (1, 2) and len(models) == len(sites)


def _directional_ring_pair_transition_term_required_offline(
    *,
    facts: MoleculeFacts,
    branch: Mapping[str, object],
    ring_endpoint_choices: Mapping[int, tuple[tuple[str, DirectionMark], ...]],
) -> bool:
    delta = branch["payload"]["graph_ring_delta"]
    if delta["kind"] not in ("ring_endpoint_pair", "ring_endpoint_pair_non_single"):
        return False
    events = [
        event for event in delta["manifest"]["event_manifests"]
        if event["kind"] == "ring_endpoint_paired"
    ]
    if len(events) != 1:
        return False
    event = events[0]
    bond = _facts_bond(facts=facts, bond=event["bond"])
    if bond.order not in (BondOrder.SINGLE, BondOrder.DOUBLE) or _facts_bond_is_bridge(
        facts=facts,
        bond=event["bond"],
    ):
        return False
    choices = ring_endpoint_choices.get(event["bond"], ())
    if bond.order is BondOrder.SINGLE:
        if event["first_endpoint_bond_text"] != "" or event["bond_text"] != "":
            return False
    elif (
        sorted((event["first_endpoint_bond_text"], event["bond_text"])) != ["", "="]
        or event["first_endpoint_direction_mark"]["value"] != DirectionMark.ABSENT.value
        or event["direction_mark"]["value"] != DirectionMark.ABSENT.value
        or set(choices) != {("", DirectionMark.ABSENT), ("=", DirectionMark.ABSENT)}
    ):
        return False
    sites = _directional_sites_for_facts_bond(facts=facts, bond=event["bond"])
    if len(sites) not in (1, 2) or any(
        site.status is not SiteStatus.SPECIFIED for site in sites
    ):
        return False
    models = _expected_directional_models_for_facts_bond(
        facts=facts,
        sites=sites,
        bond=event["bond"],
    )
    if bond.order is BondOrder.DOUBLE:
        return len(models) == 1
    return len(models) == len(sites)


def _check_tetra_residual_manifest_core(
    *,
    branch: Mapping[str, object],
    item: Mapping[str, object],
    expected_event_kind: str,
    expected_capability: str,
    expected_lifecycle_capabilities: tuple[str, ...],
    expected_certificate_kind: str,
    expected_changed_field: str,
) -> None:
    payload = branch["payload"]
    if item["family"] != "residual_work":
        _offline_violation("tetra_residual_family_mismatch")
    if item["source_digest"] != payload["source_state_digest"]:
        _offline_violation("tetra_residual_source_digest_mismatch")
    if item["successor_digest"] != payload["successor_state_digest"]:
        _offline_violation("tetra_residual_successor_digest_mismatch")
    if item["is_noop"] or item["is_empty"] or item["is_discharged"]:
        _offline_violation("tetra_residual_unexpected_discharge_flag")
    if item["terminal_clean"]:
        _offline_violation("tetra_residual_unexpected_terminal_clean")
    if item["ring_summary"] is not None:
        _offline_violation("tetra_residual_unexpected_ring_summary")
    if not item["evidence_digest"]:
        _offline_violation("tetra_residual_evidence_digest_missing")
    linked_digests = item["linked_lifecycle_digests"]
    if not linked_digests:
        _offline_violation("tetra_residual_lifecycle_link_missing")
    if len(set(linked_digests)) != len(linked_digests):
        _offline_violation("tetra_residual_lifecycle_link_duplicate")
    lifecycle_by_digest = {}
    reverse_linked_digests = set()
    for lifecycle in payload["obligation_manifests"]["stereo_lifecycle"]:
        lifecycle_digest = lifecycle["evidence_digest"]
        if lifecycle_digest in lifecycle_by_digest:
            _offline_violation("tetra_residual_lifecycle_digest_duplicate")
        lifecycle_by_digest[lifecycle_digest] = lifecycle
        reverse_links = lifecycle["linked_residual_work_digests"]
        if len(set(reverse_links)) != len(reverse_links):
            _offline_violation("tetra_residual_reverse_link_duplicate")
        if item["evidence_digest"] in reverse_links:
            reverse_linked_digests.add(lifecycle_digest)
    if set(linked_digests) != reverse_linked_digests:
        _offline_violation("tetra_residual_lifecycle_link_mismatch")
    linked_lifecycles = []
    for digest in linked_digests:
        lifecycle = lifecycle_by_digest.get(digest)
        if lifecycle is None:
            _offline_violation("tetra_residual_lifecycle_link_missing")
        if item["evidence_digest"] not in lifecycle["linked_residual_work_digests"]:
            _offline_violation("tetra_residual_reverse_link_missing")
        if lifecycle["source_digest"] != item["source_digest"]:
            _offline_violation("tetra_residual_lifecycle_source_mismatch")
        if lifecycle["successor_digest"] != item["successor_digest"]:
            _offline_violation("tetra_residual_lifecycle_successor_mismatch")
        if not lifecycle["is_discharged"]:
            _offline_violation("tetra_residual_lifecycle_not_discharged")
        linked_lifecycles.append(lifecycle)
    allowed_operations = {
        "WriterStereoLifecycleEvidence",
        "WriterStereoBranchCertificate",
    }
    if any(lifecycle["operation"] not in allowed_operations for lifecycle in linked_lifecycles):
        _offline_violation("tetra_residual_lifecycle_operation_mismatch")
    raw = [
        lifecycle
        for lifecycle in linked_lifecycles
        if lifecycle["operation"] == "WriterStereoLifecycleEvidence"
    ]
    certificates = [
        lifecycle
        for lifecycle in linked_lifecycles
        if lifecycle["operation"] == "WriterStereoBranchCertificate"
        and lifecycle["certificate_kind"] == expected_certificate_kind
    ]
    if len(raw) != 1 or len(certificates) != 1:
        _offline_violation("tetra_residual_lifecycle_evidence_missing")
    raw_lifecycle = raw[0]
    certificate = certificates[0]
    _check_tetra_raw_lifecycle_provenance(
        lifecycle=raw_lifecycle,
        item=item,
        expected_event_kind=expected_event_kind,
        expected_lifecycle_capabilities=expected_lifecycle_capabilities,
        expected_changed_field=expected_changed_field,
    )
    _check_tetra_certificate_lifecycle_provenance(
        certificate=certificate,
        raw_lifecycle=raw_lifecycle,
        item=item,
        expected_capability=expected_capability,
        expected_certificate_kind=expected_certificate_kind,
    )


def _check_tetra_raw_lifecycle_provenance(
    *,
    lifecycle: Mapping[str, object],
    item: Mapping[str, object],
    expected_event_kind: str,
    expected_lifecycle_capabilities: tuple[str, ...],
    expected_changed_field: str,
) -> None:
    if lifecycle["lifecycle_event_kind"] != expected_event_kind:
        _offline_violation("tetra_residual_lifecycle_event_kind_mismatch")
    if lifecycle["lifecycle_capabilities"] != list(expected_lifecycle_capabilities):
        _offline_violation("tetra_residual_lifecycle_capabilities_mismatch")
    if lifecycle["lifecycle_outcome_kind"] not in {
        "residual_restricted",
        "record_and_restrict",
    }:
        _offline_violation("tetra_residual_lifecycle_outcome_kind_mismatch")
    if not lifecycle[expected_changed_field]:
        _offline_violation("tetra_residual_lifecycle_change_flag_mismatch")
    if lifecycle["residual_work_digests"] != [item["evidence_digest"]]:
        _offline_violation("tetra_residual_lifecycle_work_digest_mismatch")
    if lifecycle["residual_work_operations"] != [item["operation"]]:
        _offline_violation("tetra_residual_lifecycle_work_operation_mismatch")
    if lifecycle["certificate_kind"] is not None:
        _offline_violation("tetra_residual_lifecycle_certificate_kind_mismatch")
    if lifecycle["certificate_capability"] is not None:
        _offline_violation("tetra_residual_lifecycle_certificate_capability_mismatch")
    if lifecycle["certificate_lifecycle_digest"] is not None:
        _offline_violation("tetra_residual_lifecycle_certificate_digest_mismatch")


def _check_tetra_certificate_lifecycle_provenance(
    *,
    certificate: Mapping[str, object],
    raw_lifecycle: Mapping[str, object],
    item: Mapping[str, object],
    expected_capability: str,
    expected_certificate_kind: str,
) -> None:
    if certificate["lifecycle_event_kind"] != raw_lifecycle["lifecycle_event_kind"]:
        _offline_violation("tetra_residual_certificate_event_kind_mismatch")
    if certificate["lifecycle_capabilities"] != [expected_capability]:
        _offline_violation("tetra_residual_certificate_capabilities_mismatch")
    if certificate["certificate_capability"] != expected_capability:
        _offline_violation("tetra_residual_certificate_capability_mismatch")
    if certificate["lifecycle_outcome_kind"] != raw_lifecycle["lifecycle_outcome_kind"]:
        _offline_violation("tetra_residual_certificate_outcome_kind_mismatch")
    if certificate["residual_snapshot_changed"] != raw_lifecycle["residual_snapshot_changed"]:
        _offline_violation("tetra_residual_certificate_change_flag_mismatch")
    if certificate["local_orders_changed"] != raw_lifecycle["local_orders_changed"]:
        _offline_violation("tetra_residual_certificate_change_flag_mismatch")
    if certificate["residual_work_digests"] != raw_lifecycle["residual_work_digests"]:
        _offline_violation("tetra_residual_certificate_work_digest_mismatch")
    if certificate["residual_work_digests"] != [item["evidence_digest"]]:
        _offline_violation("tetra_residual_certificate_work_digest_mismatch")
    if certificate["residual_work_operations"] != raw_lifecycle["residual_work_operations"]:
        _offline_violation("tetra_residual_certificate_work_operation_mismatch")
    if certificate["certificate_kind"] != expected_certificate_kind:
        _offline_violation("tetra_residual_certificate_kind_mismatch")
    if certificate["certificate_lifecycle_digest"] != raw_lifecycle["evidence_digest"]:
        _offline_violation("tetra_residual_certificate_lifecycle_digest_mismatch")


def _specified_tetra_centers(facts: MoleculeFacts) -> set[object]:
    return {
        _term(site.center)
        for site in facts.stereo.tetrahedral
        if site.status is SiteStatus.SPECIFIED
    }


def _tetra_token_from_rendered_text(text: str) -> str:
    if text.startswith("[C@@H]"):
        return "@@"
    if text.startswith("[C@H]"):
        return "@"
    _offline_violation("tetra_atom_token_residual_text_mismatch")


def _single_atom_emitted_event(
    *,
    events: list[Mapping[str, object]],
    violation_prefix: str,
) -> Mapping[str, object]:
    atom_events = [
        event
        for event in events
        if event["kind"] == "atom_emitted"
    ]
    if len(atom_events) != 1:
        _offline_violation(f"{violation_prefix}_atom_event_count")
    return atom_events[0]


def _bond_between_facts_atoms(
    *,
    facts: MoleculeFacts,
    bond_id: object,
    left_atom: object,
    right_atom: object,
) -> bool:
    for bond in facts.bonds:
        if _term(bond.id) != bond_id:
            continue
        endpoints = {_term(bond.a), _term(bond.b)}
        return endpoints == {left_atom, right_atom}
    return False


def _require_specified_tetra_center(
    *,
    facts: MoleculeFacts,
    atom: object,
    violation: str,
) -> None:
    if atom not in _specified_tetra_centers(facts):
        _offline_violation(violation)


def _replay_tetra_atom_token_transition(
    *,
    branch: Mapping[str, object],
    item: Mapping[str, object],
    facts: MoleculeFacts,
    objects: Mapping[str, Mapping[str, object]],
) -> None:
    term = _transition_from_manifest(item)
    if not isinstance(term, TetraAtomTokenRestrictionTransitionTerm):
        _offline_violation("tetra_atom_token_transition_kind_mismatch")
    if term.kind is not WriterResidualTransitionKind.TETRA_ATOM_TOKEN_RESTRICTION:
        _offline_violation("tetra_atom_token_transition_kind_mismatch")
    if term.source_snapshot_digest != _identity_digest(term.source_snapshot):
        _offline_violation("tetra_atom_token_source_residual_digest_mismatch")
    if term.successor_snapshot_digest != _identity_digest(term.successor_snapshot):
        _offline_violation("tetra_atom_token_successor_residual_digest_mismatch")
    _check_transition_manifest_digest(item=item, term=term)
    _check_transition_lifecycle_residual_binding(
        branch=branch,
        item=item,
        source_digest=term.source_snapshot_digest,
        successor_digest=term.successor_snapshot_digest,
    )
    source_state, successor_state = _branch_writer_state_terms(
        branch=branch,
        objects=objects,
    )
    _check_transition_state_residual_anchors(
        source_state=source_state,
        successor_state=successor_state,
        source_snapshot=term.source_snapshot,
        successor_snapshot=term.successor_snapshot,
    )
    delta = branch["payload"]["graph_ring_delta"]
    if delta["kind"] not in ("atom_start", "atom_advance", "bond_advance"):
        _offline_violation("tetra_atom_token_residual_delta_kind_mismatch")
    event = _single_atom_emitted_event(
        events=delta["manifest"]["event_manifests"],
        violation_prefix="tetra_atom_token_residual",
    )
    if event["atom"] != int(term.atom):
        _offline_violation("tetra_atom_token_residual_atom_mismatch")
    if event["tetra_token"]["value"] != term.token.value:
        _offline_violation("tetra_atom_token_residual_token_mismatch")
    _specified_tetra_site_for_transition(
        facts=facts,
        site=int(term.site),
        atom=int(term.atom),
        violation_prefix="tetra_atom_token",
    )
    expected_var = tetra_token_var(term.site)
    if term.constraint_var != expected_var or term.constraint_value is not term.token:
        _offline_violation("tetra_atom_token_transition_constraint_mismatch")
    store = ResidualStore.from_value_snapshot(term.source_snapshot)
    result = store.restrict_many_and_propagate(
        ((term.constraint_var, term.constraint_value),)
    )
    _check_transition_result(
        expected=term.propagation_result,
        actual=result,
        violation_prefix="tetra_atom_token",
    )
    if result.stats.component_variables != term.affected_variables:
        _offline_violation("tetra_atom_token_affected_variables_mismatch")
    if result.stats.component_factor_keys != term.affected_factor_keys:
        _offline_violation("tetra_atom_token_affected_factors_mismatch")
    if term.projected_variables or term.discharged_factor_keys:
        _offline_violation("tetra_atom_token_projection_or_discharge_mismatch")
    if store.value_snapshot() != term.successor_snapshot:
        _offline_violation("tetra_atom_token_successor_residual_mismatch")


def _check_directional_ring_projection_manifest_core(
    *,
    branch: Mapping[str, object],
    item: Mapping[str, object],
    facts: MoleculeFacts,
) -> None:
    payload = branch["payload"]
    if item["family"] != "residual_work":
        _offline_violation("directional_ring_projection_family_mismatch")
    if item["source_digest"] != payload["source_state_digest"]:
        _offline_violation("directional_ring_projection_source_digest_mismatch")
    if item["successor_digest"] != payload["successor_state_digest"]:
        _offline_violation("directional_ring_projection_successor_digest_mismatch")
    raw = _linked_raw_tetra_lifecycle(branch=branch, item=item)
    if raw["lifecycle_event_kind"] != "ring_endpoint_emitted":
        _offline_violation("directional_ring_projection_lifecycle_event_mismatch")
    expected_capabilities = [
        "directional_ring_pair_compatibility",
        "residual_propagation",
    ]
    if _directional_ring_transition_site_count_offline(
        facts=facts,
        branch=branch,
    ) == 2:
        expected_capabilities.append("shared_directional_carrier_restriction")
    if raw["lifecycle_capabilities"] != expected_capabilities:
        _offline_violation("directional_ring_projection_lifecycle_capabilities_mismatch")
    if raw["residual_work_digests"] != [item["evidence_digest"]]:
        _offline_violation("directional_ring_projection_lifecycle_work_mismatch")
    if raw["residual_work_operations"] != [item["operation"]]:
        _offline_violation("directional_ring_projection_lifecycle_work_mismatch")


def _check_directional_ring_pair_manifest_core(
    *,
    branch: Mapping[str, object],
    item: Mapping[str, object],
    facts: MoleculeFacts,
) -> None:
    payload = branch["payload"]
    if item["family"] != "residual_work":
        _offline_violation("directional_ring_pair_family_mismatch")
    if item["source_digest"] != payload["source_state_digest"]:
        _offline_violation("directional_ring_pair_source_digest_mismatch")
    if item["successor_digest"] != payload["successor_state_digest"]:
        _offline_violation("directional_ring_pair_successor_digest_mismatch")
    if item["transition_term"] is None:
        _offline_violation("directional_ring_pair_transition_missing")
    raw = _linked_raw_tetra_lifecycle(branch=branch, item=item)
    if raw["lifecycle_event_kind"] != "ring_endpoint_paired":
        _offline_violation("directional_ring_pair_lifecycle_event_mismatch")
    expected_capabilities = [
        "directional_carrier_restriction",
        "directional_ring_pair_compatibility",
        "directional_site_compatibility",
        "residual_factor_discharge",
        "residual_propagation",
    ]
    if _directional_ring_transition_site_count_offline(
        facts=facts,
        branch=branch,
    ) == 2:
        expected_capabilities.append("shared_directional_carrier_restriction")
    if raw["lifecycle_capabilities"] != expected_capabilities:
        _offline_violation("directional_ring_pair_lifecycle_capabilities_mismatch")
    if raw["residual_work_digests"] != [item["evidence_digest"]]:
        _offline_violation("directional_ring_pair_lifecycle_work_mismatch")
    if raw["residual_work_operations"] != [item["operation"]]:
        _offline_violation("directional_ring_pair_lifecycle_work_mismatch")
    linked = [
        lifecycle
        for lifecycle in payload["obligation_manifests"]["stereo_lifecycle"]
        if item["evidence_digest"] in lifecycle["linked_residual_work_digests"]
        and lifecycle["operation"] == "WriterStereoBranchCertificate"
    ]
    expected_certificates = {
        ("directional_carrier_restricted", "directional_carrier_restriction"),
        ("directional_ring_pair_restricted", "directional_ring_pair_compatibility"),
        ("residual_factor_discharged", "residual_factor_discharge"),
    }
    actual_certificates = {
        (item["certificate_kind"], item["certificate_capability"])
        for item in linked
    }
    if actual_certificates != expected_certificates or len(linked) != 3:
        _offline_violation("directional_ring_pair_lifecycle_certificates_mismatch")


def _replay_directional_ring_pair_transition(
    *,
    branch: Mapping[str, object],
    item: Mapping[str, object],
    facts: MoleculeFacts,
    objects: Mapping[str, Mapping[str, object]],
    ring_endpoint_choices: Mapping[int, tuple[tuple[str, DirectionMark], ...]],
) -> None:
    term = _transition_from_manifest(item)
    if not isinstance(term, DirectionalRingPairRestrictionTransitionTerm):
        _offline_violation("directional_ring_pair_transition_kind_mismatch")
    if term.kind is not WriterResidualTransitionKind.DIRECTIONAL_RING_PAIR_RESTRICTION:
        _offline_violation("directional_ring_pair_transition_kind_mismatch")
    if term.source_snapshot_digest != _identity_digest(term.source_snapshot):
        _offline_violation("directional_ring_pair_source_residual_digest_mismatch")
    if term.successor_snapshot_digest != _identity_digest(term.successor_snapshot):
        _offline_violation("directional_ring_pair_successor_residual_digest_mismatch")
    _check_transition_manifest_digest(item=item, term=term)
    _check_transition_lifecycle_residual_binding(
        branch=branch,
        item=item,
        source_digest=term.source_snapshot_digest,
        successor_digest=term.successor_snapshot_digest,
        violation_prefix=(
            "shared_directional_ring_transition"
            if len(term.carrier_models) == 2
            else "directional_ring_pair_transition"
        ),
    )
    source_state, successor_state = _branch_writer_state_terms(
        branch=branch,
        objects=objects,
    )
    _check_transition_state_residual_anchors(
        source_state=source_state,
        successor_state=successor_state,
        source_snapshot=term.source_snapshot,
        successor_snapshot=term.successor_snapshot,
        violation_prefix="directional_ring_pair",
    )
    _check_directional_ring_pair_event_and_state(
        branch=branch,
        term=term,
        source_state=source_state,
        successor_state=successor_state,
    )
    graph_bond = _facts_bond(facts=facts, bond=term.bond)
    if graph_bond.order not in (BondOrder.SINGLE, BondOrder.DOUBLE) or _facts_bond_is_bridge(
        facts=facts,
        bond=term.bond,
    ):
        _offline_violation("directional_ring_pair_bond_scope_mismatch")
    is_double = graph_bond.order is BondOrder.DOUBLE
    if is_double:
        if set(ring_endpoint_choices.get(int(term.bond), ())) != {
            ("", DirectionMark.ABSENT), ("=", DirectionMark.ABSENT)
        }:
            _offline_violation("directional_non_single_ring_policy_domain_mismatch")
        if (
            term.first_endpoint_direction_mark is not DirectionMark.ABSENT
            or term.second_endpoint_direction_mark is not DirectionMark.ABSENT
        ):
            _offline_violation("directional_non_single_ring_direction_mark_mismatch")
        if sorted((term.first_endpoint_bond_text, term.second_endpoint_bond_text)) != ["", "="]:
            _offline_violation("directional_non_single_ring_marker_count_mismatch")
    elif term.first_endpoint_bond_text != "" or term.second_endpoint_bond_text != "":
        _offline_violation("directional_ring_pair_bond_text_mismatch")
    sites = _expected_directional_sites_for_facts_bond(facts=facts, bond=term.bond)
    models = _expected_directional_models_for_facts_bond(
        facts=facts,
        sites=sites,
        bond=term.bond,
    )
    expected_model_count = 1 if is_double else len(sites)
    if (
        expected_model_count not in (1, 2)
        or len(models) != expected_model_count
        or term.carrier_models != models
    ):
        reason = (
            "shared_directional_ring_model_mismatch"
            if len(sites) == 2
            else "directional_ring_pair_carrier_model_mismatch"
        )
        _offline_violation(reason)
    first_orientation = _facts_bond_orientation(
        facts=facts,
        bond=term.bond,
        parent=term.first_atom,
        child=term.second_atom,
    )
    if (
        term.first_canonical_orientation != first_orientation
        or term.second_canonical_orientation != -first_orientation
    ):
        _offline_violation("directional_ring_pair_canonical_orientation_mismatch")
    if is_double:
        second = "" if term.first_endpoint_bond_text == "=" else "="
        compatible = ((second, DirectionMark.ABSENT),)
        restrictions_by_choice = {
            compatible[0]: ((
                directional_site_carrier_var(models[0].site, term.bond),
                DirectionalNormalizedSign.ABSENT,
            ),)
        }
    else:
        rows = _expected_shared_directional_ring_choice_rows(
            facts=facts,
            bond=term.bond,
            first_atom=term.first_atom,
            second_atom=term.second_atom,
            first_mark=term.first_endpoint_direction_mark,
            candidate_second_choices=ring_endpoint_choices.get(int(term.bond), ()),
            models=models,
        )
        compatible = tuple(choice for choice, _restrictions in rows)
        restrictions_by_choice = dict(rows)
    if term.compatible_second_endpoint_choices != compatible:
        reason = (
            "shared_directional_ring_choice_relation_mismatch"
            if len(models) == 2
            else "directional_ring_pair_compatible_choices_mismatch"
        )
        _offline_violation(reason)
    selected = (term.second_endpoint_bond_text, term.second_endpoint_direction_mark)
    if selected not in compatible:
        _offline_violation("directional_ring_pair_selected_choice_mismatch")
    expected_restrictions = restrictions_by_choice[selected]
    if term.restrictions != expected_restrictions:
        reason = (
            "shared_directional_ring_restriction_mismatch"
            if len(models) == 2
            else "directional_ring_pair_restriction_mismatch"
        )
        _offline_violation(reason)
    if not is_double:
        _check_directional_source_factor_snapshots(
            facts=facts,
            sites=sites,
            bond=term.bond,
            source_snapshot=term.source_snapshot,
            models=models,
        )
    expected_occurrence = _expected_directional_ring_pair_occurrence(term)
    if (
        term.bond_occurrence_parent,
        term.bond_occurrence_child,
        term.bond_occurrence_mark,
    ) != expected_occurrence:
        _offline_violation("directional_ring_pair_bond_occurrence_mismatch")
    store = ResidualStore.from_value_snapshot(term.source_snapshot)
    result = store.restrict_many_and_propagate(expected_restrictions)
    _check_transition_result(
        expected=term.propagation_result,
        actual=result,
        violation_prefix="directional_ring_pair",
    )
    if result.stats.component_variables != term.affected_variables:
        _offline_violation("directional_ring_pair_affected_variables_mismatch")
    if result.stats.component_factor_keys != term.affected_factor_keys:
        _offline_violation("directional_ring_pair_affected_factors_mismatch")
    expected_discharged = _expected_directional_discharge_keys(
        facts=facts,
        sites=sites,
        bond=term.bond,
        source_state=source_state,
    )
    if term.discharged_factor_keys != expected_discharged:
        _offline_violation("directional_ring_pair_discharge_factor_mismatch")
    try:
        store.discharge_satisfied_factors(expected_discharged)
    except ValueError:
        _offline_violation("directional_ring_pair_discharge_replay_failed")
    expected_projected = tuple(sorted(
        (
            var for var in dict(term.source_snapshot.domains)
            if var not in dict(store.value_snapshot().domains)
        ),
        key=lambda var: (var.kind, tuple(repr(value) for value in var.key)),
    ))
    if term.projected_variables != expected_projected:
        _offline_violation("directional_ring_pair_projected_variables_mismatch")
    if store.value_snapshot() != term.successor_snapshot:
        _offline_violation("directional_ring_pair_successor_residual_mismatch")


def _expected_shared_directional_ring_choice_rows(
    *,
    facts: MoleculeFacts,
    bond: object,
    first_atom: object,
    second_atom: object,
    first_mark: DirectionMark,
    candidate_second_choices: tuple[tuple[str, DirectionMark], ...],
    models: tuple[DirectionalSiteCarrierModel, ...],
) -> tuple[
    tuple[
        tuple[str, DirectionMark],
        tuple[tuple[VarId, DirectionalNormalizedSign], ...],
    ],
    ...,
]:
    first_orientation = _facts_bond_orientation(
        facts=facts,
        bond=bond,
        parent=first_atom,
        child=second_atom,
    )
    rows = []
    for choice in candidate_second_choices:
        _bond_text, second_mark = choice
        restrictions = []
        valid = True
        for model in models:
            normalized = []
            if first_mark is not DirectionMark.ABSENT:
                normalized.append(normalized_sign_from_mark(
                    mark=first_mark,
                    canonical_orientation=first_orientation,
                    model=model,
                ))
            if second_mark is not DirectionMark.ABSENT:
                normalized.append(normalized_sign_from_mark(
                    mark=second_mark,
                    canonical_orientation=-first_orientation,
                    model=model,
                ))
            if not normalized:
                value = DirectionalNormalizedSign.ABSENT
            elif len(set(normalized)) == 1:
                value = normalized[0]
            else:
                valid = False
                break
            restrictions.append((
                directional_site_carrier_var(model.site, bond),
                value,
            ))
        if valid:
            rows.append((choice, tuple(restrictions)))
    return tuple(rows)


def _expected_directional_ring_pair_occurrence(
    term: DirectionalRingPairRestrictionTransitionTerm,
) -> tuple[AtomId, AtomId, DirectionMark]:
    if term.first_endpoint_direction_mark is not DirectionMark.ABSENT:
        return (
            term.first_atom,
            term.second_atom,
            term.first_endpoint_direction_mark,
        )
    if term.second_endpoint_direction_mark is not DirectionMark.ABSENT:
        return (
            term.second_atom,
            term.first_atom,
            term.second_endpoint_direction_mark,
        )
    return (term.first_atom, term.second_atom, DirectionMark.ABSENT)


def _replay_directional_ring_endpoint_projection_transition(
    *,
    branch: Mapping[str, object],
    item: Mapping[str, object],
    facts: MoleculeFacts,
    objects: Mapping[str, Mapping[str, object]],
    ring_endpoint_choices: Mapping[int, tuple[tuple[str, DirectionMark], ...]],
) -> None:
    term = _transition_from_manifest(item)
    if not isinstance(
        term,
        (
            DirectionalRingEndpointProjectionTransitionTerm,
            SharedDirectionalRingEndpointProjectionTransitionTerm,
        ),
    ):
        _offline_violation("directional_ring_projection_transition_kind_mismatch")
    if term.kind is not WriterResidualTransitionKind.DIRECTIONAL_RING_ENDPOINT_PROJECTION:
        _offline_violation("directional_ring_projection_transition_kind_mismatch")
    if term.source_snapshot_digest != _identity_digest(term.source_snapshot):
        _offline_violation("directional_ring_projection_source_residual_digest_mismatch")
    if term.successor_snapshot_digest != _identity_digest(term.successor_snapshot):
        _offline_violation("directional_ring_projection_successor_residual_digest_mismatch")
    _check_transition_manifest_digest(item=item, term=term)
    _check_transition_lifecycle_residual_binding(
        branch=branch,
        item=item,
        source_digest=term.source_snapshot_digest,
        successor_digest=term.successor_snapshot_digest,
        violation_prefix=(
            "shared_directional_ring_transition"
            if isinstance(
                term,
                SharedDirectionalRingEndpointProjectionTransitionTerm,
            )
            else "directional_ring_projection_transition"
        ),
    )
    source_state, successor_state = _branch_writer_state_terms(
        branch=branch,
        objects=objects,
    )
    _check_transition_state_residual_anchors(
        source_state=source_state,
        successor_state=successor_state,
        source_snapshot=term.source_snapshot,
        successor_snapshot=term.successor_snapshot,
        violation_prefix="directional_ring_projection",
    )
    _check_directional_ring_projection_event_and_state(
        branch=branch,
        term=term,
        source_state=source_state,
        successor_state=successor_state,
    )
    graph_bond = _facts_bond(facts=facts, bond=term.bond)
    if graph_bond.order not in (BondOrder.SINGLE, BondOrder.DOUBLE) or _facts_bond_is_bridge(
        facts=facts,
        bond=term.bond,
    ):
        _offline_violation("directional_ring_projection_bond_scope_mismatch")
    is_double = graph_bond.order is BondOrder.DOUBLE
    if is_double:
        if set(ring_endpoint_choices.get(int(term.bond), ())) != {
            ("", DirectionMark.ABSENT), ("=", DirectionMark.ABSENT)
        }:
            _offline_violation("directional_non_single_ring_policy_domain_mismatch")
        if term.direction_mark is not DirectionMark.ABSENT:
            _offline_violation("directional_non_single_ring_direction_mark_mismatch")
    elif term.bond_text != "":
        _offline_violation("directional_ring_projection_bond_text_mismatch")
    sites = _expected_directional_sites_for_facts_bond(facts=facts, bond=term.bond)
    models = _expected_directional_models_for_facts_bond(
        facts=facts,
        sites=sites,
        bond=term.bond,
    )
    if isinstance(term, SharedDirectionalRingEndpointProjectionTransitionTerm):
        if len(sites) != 2 or len(models) != 2 or term.carrier_models != models:
            _offline_violation("shared_directional_ring_model_mismatch")
    elif len(sites) != 1 or len(models) != 1 or term.carrier_model != models[0]:
        reason = (
            "shared_directional_ring_model_mismatch"
            if len(sites) == 2
            else "directional_ring_projection_carrier_model_mismatch"
        )
        _offline_violation(reason)
    candidate_seconds = ring_endpoint_choices.get(int(term.bond), ())
    if is_double:
        expected_seconds = tuple(
            choice
            for choice in candidate_seconds
            if sorted((term.bond_text, choice[0])) == ["", "="]
        )
        rows = tuple(
            (
                choice,
                ((
                    directional_site_carrier_var(models[0].site, term.bond),
                    DirectionalNormalizedSign.ABSENT,
                ),),
            )
            for choice in expected_seconds
        )
    else:
        rows = _expected_shared_directional_ring_choice_rows(
            facts=facts,
            bond=term.bond,
            first_atom=term.endpoint_atom,
            second_atom=term.partner_atom,
            first_mark=term.direction_mark,
            candidate_second_choices=candidate_seconds,
            models=models,
        )
        expected_seconds = tuple(choice for choice, _restrictions in rows)
    if term.compatible_second_endpoint_choices != expected_seconds:
        reason = (
            "shared_directional_ring_choice_relation_mismatch"
            if len(models) == 2
            else "directional_ring_projection_compatible_seconds_mismatch"
        )
        _offline_violation(reason)
    values_by_var = {
        directional_site_carrier_var(model.site, term.bond): []
        for model in models
    }
    for _choice, restrictions in rows:
        for var, value in restrictions:
            if value not in values_by_var[var]:
                values_by_var[var].append(value)
    expected_intersections = tuple(
        (
            var,
            (
                tuple(
                    value
                    for value in (
                        DirectionalNormalizedSign.ABSENT,
                        DirectionalNormalizedSign.POSITIVE,
                        DirectionalNormalizedSign.NEGATIVE,
                    )
                    if value in values
                )
                if len(models) == 2
                else tuple(values)
            ),
        )
        for var, values in sorted(
            values_by_var.items(),
            key=lambda item: (item[0].kind, tuple(repr(value) for value in item[0].key)),
        )
    )
    if term.domain_intersections != expected_intersections:
        reason = (
            "shared_directional_ring_intersection_mismatch"
            if len(models) == 2
            else "directional_ring_projection_domain_intersection_mismatch"
        )
        _offline_violation(reason)
    if not is_double:
        _check_directional_source_factor_snapshots(
            facts=facts,
            sites=sites,
            bond=term.bond,
            source_snapshot=term.source_snapshot,
            models=models,
        )
    store = ResidualStore.from_value_snapshot(term.source_snapshot)
    result = store.intersect_domains_and_propagate(expected_intersections)
    _check_transition_result(
        expected=term.propagation_result,
        actual=result,
        violation_prefix="directional_ring_projection",
    )
    if result.stats.component_variables != term.affected_variables:
        _offline_violation("directional_ring_projection_affected_variables_mismatch")
    if result.stats.component_factor_keys != term.affected_factor_keys:
        _offline_violation("directional_ring_projection_affected_factors_mismatch")
    if term.projected_variables or term.discharged_factor_keys:
        _offline_violation("directional_ring_projection_projection_or_discharge_mismatch")
    if len(models) == 2 and (
        term.source_snapshot.assignments or term.successor_snapshot.assignments
    ):
        _offline_violation("shared_directional_ring_assignment_materialized")
    if store.value_snapshot() != term.successor_snapshot:
        _offline_violation("directional_ring_projection_successor_residual_mismatch")


def _check_directional_ring_projection_event_and_state(
    *,
    branch: Mapping[str, object],
    term: (
        DirectionalRingEndpointProjectionTransitionTerm
        | SharedDirectionalRingEndpointProjectionTransitionTerm
    ),
    source_state: Mapping[str, object],
    successor_state: Mapping[str, object],
) -> None:
    delta = branch["payload"]["graph_ring_delta"]
    if delta["kind"] != "ring_endpoint_open":
        _offline_violation("directional_ring_projection_delta_kind_mismatch")
    events = [
        event for event in delta["manifest"]["event_manifests"]
        if event["kind"] == "ring_endpoint_emitted"
    ]
    if len(events) != 1:
        _offline_violation("directional_ring_projection_event_mismatch")
    event = events[0]
    expected_event = {
        "bond": int(term.bond),
        "endpoint_atom": int(term.endpoint_atom),
        "partner_atom": int(term.partner_atom),
        "label": {
            "__dataclass__": "grimace._south_star1.writer_state.WriterClosureLabel",
            "fields": [["value", term.ring_label_value], ["text", term.ring_label_text]],
        },
        "endpoint_text": term.endpoint_text,
        "bond_text": term.bond_text,
        "direction_mark": {"__enum__": "grimace._south_star1.policy.DirectionMark", "value": term.direction_mark.value},
        "side": "open",
    }
    for field, expected in expected_event.items():
        if event[field] != expected:
            _offline_violation(f"directional_ring_projection_event_{field}_mismatch")
    source_ring = _term_field_value(source_state, "ring_state")
    successor_ring = _term_field_value(successor_state, "ring_state")
    source_open = tuple(_term_field_value(source_ring, "open_endpoints"))
    successor_open = tuple(_term_field_value(successor_ring, "open_endpoints"))
    if any(int(_term_field_value(endpoint, "bond")) == int(term.bond) for endpoint in source_open):
        _offline_violation("directional_ring_projection_source_open_endpoint_mismatch")
    matching = [
        endpoint for endpoint in successor_open
        if int(_term_field_value(endpoint, "bond")) == int(term.bond)
    ]
    if len(matching) != 1:
        _offline_violation("directional_ring_projection_successor_open_endpoint_mismatch")
    endpoint = matching[0]
    expected_fields = {
        "first_atom": int(term.endpoint_atom),
        "second_atom": int(term.partner_atom),
        "label": expected_event["label"],
        "first_endpoint_text": term.endpoint_text,
        "first_endpoint_bond_text": term.bond_text,
        "first_endpoint_direction_mark": expected_event["direction_mark"],
    }
    for field, expected in expected_fields.items():
        if _term_field_value(endpoint, field) != expected:
            _offline_violation("directional_ring_projection_successor_open_endpoint_mismatch")
    if tuple(_term_field_value(source_ring, "closed_closures")) != tuple(
        _term_field_value(successor_ring, "closed_closures")
    ):
        _offline_violation("directional_ring_projection_closed_closure_mismatch")
    label = _writer_closure_label_term(
        value=term.ring_label_value,
        text=term.ring_label_text,
    )
    successor_labels = _term_field_value(successor_ring, "label_state")
    if (
        tuple(_term_field_value(successor_labels, "allocated")).count(label) != 1
        or label in tuple(_term_field_value(successor_labels, "reusable"))
    ):
        _offline_violation("directional_ring_projection_label_state_mismatch")
    if _state_bond_occurrence_records(source_state) != _state_bond_occurrence_records(successor_state):
        _offline_violation("directional_ring_projection_bond_occurrence_mismatch")


def _check_directional_ring_pair_event_and_state(
    *,
    branch: Mapping[str, object],
    term: DirectionalRingPairRestrictionTransitionTerm,
    source_state: Mapping[str, object],
    successor_state: Mapping[str, object],
) -> None:
    delta = branch["payload"]["graph_ring_delta"]
    if delta["kind"] not in ("ring_endpoint_pair", "ring_endpoint_pair_non_single"):
        _offline_violation("directional_ring_pair_delta_kind_mismatch")
    events = delta["manifest"]["event_manifests"]
    pair_events = [event for event in events if event["kind"] == "ring_endpoint_paired"]
    if len(pair_events) != 1:
        _offline_violation("directional_ring_pair_event_mismatch")
    event = pair_events[0]
    label = _writer_closure_label_term(
        value=term.ring_label_value,
        text=term.ring_label_text,
    )
    expected_event = {
        "bond": int(term.bond),
        "endpoint_atom": int(term.second_atom),
        "partner_atom": int(term.first_atom),
        "label": label,
        "endpoint_text": term.second_endpoint_text,
        "bond_text": term.second_endpoint_bond_text,
        "direction_mark": _direction_mark_term(term.second_endpoint_direction_mark),
        "first_endpoint_bond_text": term.first_endpoint_bond_text,
        "first_endpoint_direction_mark": _direction_mark_term(
            term.first_endpoint_direction_mark
        ),
        "side": "close",
    }
    for field, expected in expected_event.items():
        if event[field] != expected:
            _offline_violation(f"directional_ring_pair_event_{field}_mismatch")
    releases = [event for event in events if event["kind"] == "ring_label_released"]
    if len(releases) != 1 or releases[0]["label"] != label:
        _offline_violation("directional_ring_pair_label_release_event_mismatch")
    if releases[0]["destination"] != "reusable":
        _offline_violation("directional_ring_pair_label_release_event_mismatch")

    source_ring = _term_field_value(source_state, "ring_state")
    successor_ring = _term_field_value(successor_state, "ring_state")
    source_open = tuple(_term_field_value(source_ring, "open_endpoints"))
    successor_open = tuple(_term_field_value(successor_ring, "open_endpoints"))
    source_matches = [
        endpoint for endpoint in source_open
        if int(_term_field_value(endpoint, "bond")) == int(term.bond)
    ]
    if len(source_matches) != 1:
        _offline_violation("directional_ring_pair_open_endpoint_anchor_mismatch")
    open_endpoint = source_matches[0]
    expected_open = {
        "first_atom": int(term.first_atom),
        "second_atom": int(term.second_atom),
        "label": label,
        "first_endpoint_text": term.first_endpoint_text,
        "first_endpoint_bond_text": term.first_endpoint_bond_text,
        "first_endpoint_direction_mark": _direction_mark_term(
            term.first_endpoint_direction_mark
        ),
    }
    if any(
        _term_field_value(open_endpoint, field) != expected
        for field, expected in expected_open.items()
    ):
        _offline_violation("directional_ring_pair_open_endpoint_anchor_mismatch")
    if any(
        int(_term_field_value(endpoint, "bond")) == int(term.bond)
        for endpoint in successor_open
    ):
        _offline_violation("directional_ring_pair_open_endpoint_anchor_mismatch")
    unrelated_source_open = tuple(
        endpoint for endpoint in source_open if endpoint is not open_endpoint
    )
    if successor_open != unrelated_source_open:
        _offline_violation("directional_ring_pair_open_endpoint_anchor_mismatch")

    source_closed = tuple(_term_field_value(source_ring, "closed_closures"))
    successor_closed = tuple(_term_field_value(successor_ring, "closed_closures"))
    if any(
        int(_term_field_value(closure, "bond")) == int(term.bond)
        for closure in source_closed
    ):
        _offline_violation("directional_ring_pair_closed_closure_anchor_mismatch")
    closed_matches = [
        closure for closure in successor_closed
        if int(_term_field_value(closure, "bond")) == int(term.bond)
    ]
    if len(closed_matches) != 1:
        _offline_violation("directional_ring_pair_closed_closure_anchor_mismatch")
    closure = closed_matches[0]
    expected_closed = {
        "first_atom": int(term.first_atom),
        "second_atom": int(term.second_atom),
        "label": label,
        "first_endpoint_text": term.first_endpoint_text,
        "second_endpoint_text": term.second_endpoint_text,
        "first_endpoint_bond_text": term.first_endpoint_bond_text,
        "second_endpoint_bond_text": term.second_endpoint_bond_text,
        "first_endpoint_direction_mark": _direction_mark_term(
            term.first_endpoint_direction_mark
        ),
        "second_endpoint_direction_mark": _direction_mark_term(
            term.second_endpoint_direction_mark
        ),
    }
    if any(
        _term_field_value(closure, field) != expected
        for field, expected in expected_closed.items()
    ):
        _offline_violation("directional_ring_pair_closed_closure_anchor_mismatch")
    if successor_closed != source_closed + (closure,):
        _offline_violation("directional_ring_pair_closed_closure_anchor_mismatch")

    source_labels = _term_field_value(source_ring, "label_state")
    successor_labels = _term_field_value(successor_ring, "label_state")
    source_allocated = tuple(_term_field_value(source_labels, "allocated"))
    source_reusable = tuple(_term_field_value(source_labels, "reusable"))
    successor_allocated = tuple(_term_field_value(successor_labels, "allocated"))
    successor_reusable = tuple(_term_field_value(successor_labels, "reusable"))
    if (
        source_allocated.count(label) != 1
        or label in source_reusable
        or label in successor_allocated
        or successor_reusable.count(label) != 1
        or successor_allocated != tuple(item for item in source_allocated if item != label)
        or successor_reusable != source_reusable + (label,)
    ):
        _offline_violation("directional_ring_pair_label_state_mismatch")

    source_occurrences = _state_bond_occurrence_records(source_state)
    successor_occurrences = _state_bond_occurrence_records(successor_state)
    if any(
        int(_term_field_value(record, "bond")) == int(term.bond)
        for record in source_occurrences
    ):
        _offline_violation("directional_ring_pair_bond_occurrence_mismatch")
    expected_record = {
        "__dataclass__": "grimace._south_star1.writer_stereo.WriterBondOccurrenceRecord",
        "fields": [
            ["bond", int(term.bond)],
            ["parent", int(term.bond_occurrence_parent)],
            ["child", int(term.bond_occurrence_child)],
            ["mark", _direction_mark_term(term.bond_occurrence_mark)],
        ],
    }
    if successor_occurrences != source_occurrences + (expected_record,):
        _offline_violation("directional_ring_pair_bond_occurrence_mismatch")


def _writer_closure_label_term(*, value: int, text: str) -> dict[str, object]:
    return {
        "__dataclass__": "grimace._south_star1.writer_state.WriterClosureLabel",
        "fields": [["value", value], ["text", text]],
    }


def _direction_mark_term(mark: DirectionMark) -> dict[str, object]:
    return {
        "__enum__": "grimace._south_star1.policy.DirectionMark",
        "value": mark.value,
    }


def _replay_tetra_local_order_transition(
    *,
    branch: Mapping[str, object],
    item: Mapping[str, object],
    facts: MoleculeFacts,
    objects: Mapping[str, Mapping[str, object]],
) -> None:
    term = _transition_from_manifest(item)
    if not isinstance(term, TetraLocalOrderFactorClosureTransitionTerm):
        _offline_violation("tetra_local_order_transition_kind_mismatch")
    if term.kind is not WriterResidualTransitionKind.TETRA_LOCAL_ORDER_FACTOR_CLOSURE:
        _offline_violation("tetra_local_order_transition_kind_mismatch")
    if term.source_snapshot_digest != _identity_digest(term.source_snapshot):
        _offline_violation("tetra_local_order_source_residual_digest_mismatch")
    if term.successor_snapshot_digest != _identity_digest(term.successor_snapshot):
        _offline_violation("tetra_local_order_successor_residual_digest_mismatch")
    _check_transition_manifest_digest(item=item, term=term)
    _check_transition_lifecycle_residual_binding(
        branch=branch,
        item=item,
        source_digest=term.source_snapshot_digest,
        successor_digest=term.successor_snapshot_digest,
    )
    source_state, successor_state = _branch_writer_state_terms(
        branch=branch,
        objects=objects,
    )
    _check_transition_state_residual_anchors(
        source_state=source_state,
        successor_state=successor_state,
        source_snapshot=term.source_snapshot,
        successor_snapshot=term.successor_snapshot,
    )
    site = _specified_tetra_site_for_transition(
        facts=facts,
        site=int(term.site),
        atom=int(term.atom),
        violation_prefix="tetra_local_order",
    )
    if tuple(int(item) for item in term.reference_order) != tuple(
        int(item) for item in site.reference_order
    ):
        _offline_violation("tetra_local_order_reference_order_mismatch")
    if set(int(item) for item in term.local_order) != set(
        int(item) for item in site.reference_order
    ):
        _offline_violation("tetra_local_order_local_order_mismatch")
    _check_transition_local_order_event_binding(
        branch=branch,
        term=term,
        source_state=source_state,
        successor_state=successor_state,
    )
    expected_parity = _local_order_parity(
        reference_order=site.reference_order,
        local_order=term.local_order,
    )
    if term.target_parity is not expected_parity:
        _offline_violation("tetra_local_order_target_parity_mismatch")
    expected_var = tetra_parity_var(term.site)
    if (
        term.constraint_var != expected_var
        or term.constraint_value is not expected_parity
    ):
        _offline_violation("tetra_local_order_transition_constraint_mismatch")
    expected_factor = ResidualFactorKey("tetra_site", (int(term.site),))
    if term.discharged_factor_keys != (expected_factor,):
        _offline_violation("tetra_local_order_discharge_factor_mismatch")
    if term.projected_variables != (term.constraint_var,):
        _offline_violation("tetra_local_order_projected_variables_mismatch")
    if term.constraint_var not in dict(term.source_snapshot.domains):
        _offline_violation("tetra_local_order_projected_before_closure")
    store = ResidualStore.from_value_snapshot(term.source_snapshot)
    result = store.restrict_many_and_propagate(
        ((term.constraint_var, term.constraint_value),)
    )
    _check_transition_result(
        expected=term.propagation_result,
        actual=result,
        violation_prefix="tetra_local_order",
    )
    if result.stats.component_variables != term.affected_variables:
        _offline_violation("tetra_local_order_affected_variables_mismatch")
    if result.stats.component_factor_keys != term.affected_factor_keys:
        _offline_violation("tetra_local_order_affected_factors_mismatch")
    try:
        store.discharge_satisfied_factors(term.discharged_factor_keys)
    except ValueError:
        _offline_violation("tetra_local_order_discharge_replay_failed")
    if store.value_snapshot() != term.successor_snapshot:
        _offline_violation("tetra_local_order_successor_residual_mismatch")


def _replay_directional_carrier_transition(
    *,
    branch: Mapping[str, object],
    item: Mapping[str, object],
    facts: MoleculeFacts,
    objects: Mapping[str, Mapping[str, object]],
) -> None:
    term = _transition_from_manifest(item)
    if not isinstance(term, DirectionalCarrierMarkRestrictionTransitionTerm):
        _offline_violation("directional_carrier_transition_kind_mismatch")
    if (
        term.kind
        is not WriterResidualTransitionKind.DIRECTIONAL_CARRIER_MARK_RESTRICTION
    ):
        _offline_violation("directional_carrier_transition_kind_mismatch")
    if term.source_snapshot_digest != _identity_digest(term.source_snapshot):
        _offline_violation("directional_carrier_source_residual_digest_mismatch")
    if term.successor_snapshot_digest != _identity_digest(term.successor_snapshot):
        _offline_violation("directional_carrier_successor_residual_digest_mismatch")
    _check_transition_manifest_digest(item=item, term=term)
    _check_transition_lifecycle_residual_binding(
        branch=branch,
        item=item,
        source_digest=term.source_snapshot_digest,
        successor_digest=term.successor_snapshot_digest,
    )
    source_state, successor_state = _branch_writer_state_terms(
        branch=branch,
        objects=objects,
    )
    _check_transition_state_residual_anchors(
        source_state=source_state,
        successor_state=successor_state,
        source_snapshot=term.source_snapshot,
        successor_snapshot=term.successor_snapshot,
        violation_prefix="directional_carrier",
    )
    _check_directional_bond_event_binding(
        branch=branch,
        term=term,
        source_state=source_state,
        successor_state=successor_state,
    )
    if term.canonical_orientation not in (-1, 1):
        _offline_violation("directional_carrier_canonical_orientation_mismatch")
    expected_orientation = _facts_bond_orientation(
        facts=facts,
        bond=term.bond,
        parent=term.parent,
        child=term.child,
    )
    if term.canonical_orientation != expected_orientation:
        _offline_violation("directional_carrier_canonical_orientation_mismatch")
    sites = _expected_directional_sites_for_facts_bond(
        facts=facts,
        bond=term.bond,
    )
    if not 1 <= len(sites) <= 2:
        _offline_violation("directional_carrier_site_scope_mismatch")
    expected_models = _expected_directional_models_for_facts_bond(
        facts=facts,
        sites=sites,
        bond=term.bond,
    )
    if term.carrier_models != expected_models:
        _offline_violation("directional_carrier_model_mismatch")
    expected_restrictions = tuple(
        (
            directional_site_carrier_var(model.site, model.bond),
            normalized_sign_from_mark(
                mark=term.direction_mark,
                canonical_orientation=term.canonical_orientation,
                model=model,
            ),
        )
        for model in expected_models
    )
    if term.restrictions != expected_restrictions:
        _offline_violation("directional_carrier_restriction_mismatch")
    if _facts_bond(facts=facts, bond=term.bond).order is BondOrder.SINGLE:
        _check_directional_source_factor_snapshots(
            facts=facts,
            sites=sites,
            bond=term.bond,
            source_snapshot=term.source_snapshot,
            models=expected_models,
        )
    store = ResidualStore.from_value_snapshot(term.source_snapshot)
    result = store.restrict_many_and_propagate(term.restrictions)
    _check_transition_result(
        expected=term.propagation_result,
        actual=result,
        violation_prefix="directional_carrier",
    )
    if result.stats.component_variables != term.affected_variables:
        _offline_violation("directional_carrier_affected_variables_mismatch")
    if result.stats.component_factor_keys != term.affected_factor_keys:
        _offline_violation("directional_carrier_affected_factors_mismatch")
    expected_discharged = _expected_directional_discharge_keys(
        facts=facts,
        sites=sites,
        bond=term.bond,
        source_state=source_state,
    )
    if term.discharged_factor_keys != expected_discharged:
        _offline_violation("directional_carrier_discharge_factor_mismatch")
    try:
        store.discharge_satisfied_factors(term.discharged_factor_keys)
    except ValueError:
        _offline_violation("directional_carrier_discharge_replay_failed")
    expected_projected = tuple(
        sorted(
            (
                var
                for var in dict(term.source_snapshot.domains)
                if var not in dict(store.value_snapshot().domains)
            ),
            key=lambda var: (var.kind, tuple(repr(item) for item in var.key)),
        )
    )
    if term.projected_variables != expected_projected:
        _offline_violation("directional_carrier_projected_variables_mismatch")
    if store.value_snapshot() != term.successor_snapshot:
        _offline_violation("directional_carrier_successor_residual_mismatch")


def _directional_carrier_transition_term_required_offline(
    *,
    facts: MoleculeFacts,
    branch: Mapping[str, object],
) -> bool:
    event = _single_bond_emitted_event(
        events=branch["payload"]["graph_ring_delta"]["manifest"]["event_manifests"],
        violation_prefix="directional_carrier_residual",
    )
    bond = event["bond"]
    graph_bond = _facts_bond(facts=facts, bond=bond)
    if graph_bond.order not in (BondOrder.SINGLE, BondOrder.DOUBLE):
        return False
    sites = _expected_directional_sites_for_facts_bond(facts=facts, bond=bond)
    if not 1 <= len(sites) <= 2:
        return False
    models = _expected_directional_models_for_facts_bond(
        facts=facts,
        sites=sites,
        bond=bond,
    )
    return len(models) == len(sites)


def _directional_carrier_transition_site_count_offline(
    *,
    facts: MoleculeFacts,
    branch: Mapping[str, object],
) -> int:
    event = _single_bond_emitted_event(
        events=branch["payload"]["graph_ring_delta"]["manifest"]["event_manifests"],
        violation_prefix="directional_carrier_residual",
    )
    bond = event["bond"]
    graph_bond = _facts_bond(facts=facts, bond=bond)
    if graph_bond.order not in (BondOrder.SINGLE, BondOrder.DOUBLE):
        return 0
    sites = _expected_directional_sites_for_facts_bond(facts=facts, bond=bond)
    if not 1 <= len(sites) <= 2:
        return 0
    models = _expected_directional_models_for_facts_bond(
        facts=facts,
        sites=sites,
        bond=bond,
    )
    if len(models) != len(sites):
        return 0
    return len(sites)


def _directional_ring_transition_site_count_offline(
    *,
    facts: MoleculeFacts,
    branch: Mapping[str, object],
) -> int:
    events = [
        event
        for event in branch["payload"]["graph_ring_delta"]["manifest"]["event_manifests"]
        if event["kind"] in ("ring_endpoint_emitted", "ring_endpoint_paired")
    ]
    if len(events) != 1:
        return 0
    bond = events[0]["bond"]
    sites = _directional_sites_for_facts_bond(facts=facts, bond=bond)
    if len(sites) not in (1, 2) or any(
        site.status is not SiteStatus.SPECIFIED for site in sites
    ):
        return 0
    models = _expected_directional_models_for_facts_bond(
        facts=facts,
        sites=sites,
        bond=bond,
    )
    return len(sites) if len(sites) == len(models) else 0


def _check_transition_manifest_digest(*, item: Mapping[str, object], term: object) -> None:
    if item["transition_digest"] != _identity_digest(term):
        _offline_violation("tetra_residual_transition_digest_mismatch")


def _check_transition_result(
    *,
    expected: ResidualPropagationResult,
    actual: ResidualPropagationResult,
    violation_prefix: str,
) -> None:
    if actual.kind is not expected.kind:
        _offline_violation(f"{violation_prefix}_result_kind_mismatch")
    if actual.stats != expected.stats:
        _offline_violation(f"{violation_prefix}_result_stats_mismatch")


def _check_transition_lifecycle_residual_binding(
    *,
    branch: Mapping[str, object],
    item: Mapping[str, object],
    source_digest: str,
    successor_digest: str,
    violation_prefix: str = "tetra_residual_transition",
) -> None:
    raw_lifecycle = _linked_raw_tetra_lifecycle(branch=branch, item=item)
    if raw_lifecycle["source_residual_snapshot_digest"] != source_digest:
        _offline_violation(f"{violation_prefix}_source_lifecycle_mismatch")
    if raw_lifecycle["successor_residual_snapshot_digest"] != successor_digest:
        _offline_violation(f"{violation_prefix}_successor_lifecycle_mismatch")


def _branch_writer_state_terms(
    *,
    branch: Mapping[str, object],
    objects: Mapping[str, Mapping[str, object]],
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    branch_ref = branch["object_id"]
    projections = [
        item
        for item in objects.values()
        if item["kind"] == "text_projection"
        and branch_ref in item["payload"]["branch_support_refs"]
    ]
    if len(projections) != 1:
        _offline_violation("branch_projection_support_ref_ambiguous")
    projection_payload = projections[0]["payload"]
    source_state = _state_term_from_cursor(
        cursor=projection_payload["source_cursor"],
        digest=branch["payload"]["source_state_digest"],
        missing_reason="branch_source_state_term_missing",
        ambiguous_reason="branch_source_state_term_ambiguous",
    )
    successor_state = _state_term_from_cursor(
        cursor=projection_payload["successor_cursor"],
        digest=branch["payload"]["successor_state_digest"],
        missing_reason="branch_successor_state_term_missing",
        ambiguous_reason="branch_successor_state_term_ambiguous",
    )
    return source_state, successor_state


def _state_term_from_cursor(
    *,
    cursor: Mapping[str, object],
    digest: str,
    missing_reason: str,
    ambiguous_reason: str,
) -> Mapping[str, object]:
    cursor_terms = cursor["terms"]
    if (
        not isinstance(cursor_terms, Mapping)
        or cursor_terms.get("__dataclass__")
        != "grimace._south_star1.writer_frontier.WriterFrontierCursor"
    ):
        _offline_violation("branch_cursor_term_kind_mismatch")
    matches = [
        state
        for state, _weight in _term_field_value(cursor_terms, "weighted_states")
        if _closed_term_digest(state) == digest
    ]
    if not matches:
        _offline_violation(missing_reason)
    if len(matches) != 1:
        _offline_violation(ambiguous_reason)
    return matches[0]


def _check_transition_state_residual_anchors(
    *,
    source_state: Mapping[str, object],
    successor_state: Mapping[str, object],
    source_snapshot: ResidualStoreValueSnapshot,
    successor_snapshot: ResidualStoreValueSnapshot,
    violation_prefix: str = "tetra_residual_transition",
) -> None:
    if _state_residual_snapshot(source_state) != _term(source_snapshot):
        _offline_violation(f"{violation_prefix}_source_state_anchor_mismatch")
    if _state_residual_snapshot(successor_state) != _term(successor_snapshot):
        _offline_violation(f"{violation_prefix}_successor_state_anchor_mismatch")


def _state_residual_snapshot(state: Mapping[str, object]) -> object:
    stereo = _term_field_value(state, "stereo_state")
    return _term_field_value(stereo, "residual_snapshot")


def _state_local_order_records(state: Mapping[str, object]) -> tuple[object, ...]:
    stereo = _term_field_value(state, "stereo_state")
    return tuple(_term_field_value(stereo, "local_orders"))


def _state_bond_occurrence_records(state: Mapping[str, object]) -> tuple[object, ...]:
    stereo = _term_field_value(state, "stereo_state")
    return tuple(_term_field_value(stereo, "bond_occurrences"))


def _check_directional_bond_event_binding(
    *,
    branch: Mapping[str, object],
    term: DirectionalCarrierMarkRestrictionTransitionTerm,
    source_state: Mapping[str, object],
    successor_state: Mapping[str, object],
) -> None:
    delta = branch["payload"]["graph_ring_delta"]
    if delta["kind"] != "bond_advance":
        _offline_violation("directional_carrier_residual_delta_kind_mismatch")
    event = _single_bond_emitted_event(
        events=delta["manifest"]["event_manifests"],
        violation_prefix="directional_carrier_residual",
    )
    if event["bond"] != int(term.bond):
        _offline_violation("directional_carrier_residual_bond_mismatch")
    if event["parent"] != int(term.parent) or event["child"] != int(term.child):
        _offline_violation("directional_carrier_residual_endpoint_mismatch")
    if event["direction_mark"]["value"] != term.direction_mark.value:
        _offline_violation("directional_carrier_residual_mark_mismatch")
    expected_record = {
        "__dataclass__": "grimace._south_star1.writer_stereo.WriterBondOccurrenceRecord",
        "fields": [
            ["bond", int(term.bond)],
            ["parent", int(term.parent)],
            ["child", int(term.child)],
            [
                "mark",
                {
                    "__enum__": "grimace._south_star1.policy.DirectionMark",
                    "value": term.direction_mark.value,
                },
            ],
        ],
    }
    source_records = _state_bond_occurrence_records(source_state)
    successor_records = _state_bond_occurrence_records(successor_state)
    if any(int(_term_field_value(record, "bond")) == int(term.bond) for record in source_records):
        _offline_violation("directional_carrier_source_bond_occurrence_mismatch")
    if successor_records != source_records + (expected_record,):
        _offline_violation("directional_carrier_successor_bond_occurrence_mismatch")


def _single_bond_emitted_event(
    *,
    events: object,
    violation_prefix: str,
) -> Mapping[str, object]:
    matches = [
        event
        for event in events
        if isinstance(event, Mapping) and event.get("kind") == "bond_emitted"
    ]
    if len(matches) != 1:
        _offline_violation(f"{violation_prefix}_event_mismatch")
    return matches[0]


def _directional_sites_for_facts_bond(
    *,
    facts: MoleculeFacts,
    bond: object,
) -> tuple[DirectionalSiteFacts, ...]:
    occurrence_by_id = {occurrence.id: occurrence for occurrence in facts.ligand_occurrences}
    sites = []
    for site in facts.stereo.directional:
        ligand_ids = site.left_ligands + site.right_ligands
        if any(occurrence_by_id[item].bond == bond for item in ligand_ids):
            sites.append(site)
    return tuple(sites)


def _expected_directional_sites_for_facts_bond(
    *,
    facts: MoleculeFacts,
    bond: object,
) -> tuple[DirectionalSiteFacts, ...]:
    sites = _directional_sites_for_facts_bond(facts=facts, bond=bond)
    if len(sites) > 2:
        _offline_violation("directional_carrier_site_scope_mismatch")
    for site in sites:
        if site.status is not SiteStatus.SPECIFIED:
            _offline_violation("directional_carrier_site_status_mismatch")
    return sites


def _expected_directional_models_for_facts_bond(
    *,
    facts: MoleculeFacts,
    sites: tuple[DirectionalSiteFacts, ...],
    bond: object,
) -> tuple[DirectionalSiteCarrierModel, ...]:
    models = []
    for site in sites:
        site_models = _facts_directional_models_for_bond(
            facts=facts,
            site=site,
            bond=bond,
        )
        if len(site_models) != 1:
            _offline_violation("directional_carrier_model_mismatch")
        models.extend(site_models)
    return tuple(
        sorted(
            models,
            key=lambda model: (
                int(model.site),
                int(model.bond),
                model.side,
                model.endpoint_orientation_factor,
                model.ligand_factor,
            ),
        )
    )


def _facts_directional_models_for_bond(
    *,
    facts: MoleculeFacts,
    site: DirectionalSiteFacts,
    bond: object,
) -> tuple[DirectionalSiteCarrierModel, ...]:
    occurrence_by_id = {occurrence.id: occurrence for occurrence in facts.ligand_occurrences}
    left_reference, right_reference = _directional_reference_pair_from_facts(site)
    models = []
    for side, endpoint, side_ligands, reference in (
        ("left", site.left_endpoint, site.left_ligands, left_reference),
        ("right", site.right_endpoint, site.right_ligands, right_reference),
    ):
        matches = [
            occurrence_by_id[item]
            for item in side_ligands
            if occurrence_by_id[item].bond == bond
        ]
        if len(matches) > 1:
            _offline_violation("directional_carrier_model_mismatch")
        if not matches:
            continue
        occurrence = matches[0]
        if occurrence.kind is not LigandKind.NEIGHBOR_ATOM:
            _offline_violation("directional_carrier_non_neighbor_ligand")
        models.append(
            DirectionalSiteCarrierModel(
                site=site.id,
                bond=bond,
                side=side,
                endpoint_orientation_factor=_facts_endpoint_orientation_factor(
                    facts=facts,
                    bond=bond,
                    endpoint=endpoint,
                ),
                ligand_factor=(
                    1
                    if tuple(side_ligands).index(occurrence.id)
                    == tuple(side_ligands).index(reference)
                    else -1
                ),
            )
        )
    return tuple(
        sorted(
            models,
            key=lambda model: (
                int(model.site),
                int(model.bond),
                model.side,
                model.endpoint_orientation_factor,
                model.ligand_factor,
            ),
        )
    )


def _directional_reference_pair_from_facts(
    site: DirectionalSiteFacts,
) -> tuple[object, object]:
    if site.reference_pair is not None:
        return site.reference_pair
    if not site.left_ligands or not site.right_ligands:
        _offline_violation("directional_carrier_reference_pair_mismatch")
    return (site.left_ligands[0], site.right_ligands[0])


def _facts_bond_orientation(
    *,
    facts: MoleculeFacts,
    bond: object,
    parent: object,
    child: object,
) -> int:
    graph_bond = _facts_bond(facts=facts, bond=bond)
    if graph_bond.a == parent and graph_bond.b == child:
        return 1
    if graph_bond.a == child and graph_bond.b == parent:
        return -1
    _offline_violation("directional_carrier_canonical_orientation_mismatch")


def _facts_endpoint_orientation_factor(
    *,
    facts: MoleculeFacts,
    bond: object,
    endpoint: object,
) -> int:
    graph_bond = _facts_bond(facts=facts, bond=bond)
    if graph_bond.a == endpoint:
        return 1
    if graph_bond.b == endpoint:
        return -1
    _offline_violation("directional_carrier_model_mismatch")


def _facts_bond(*, facts: MoleculeFacts, bond: object) -> BondFacts:
    matches = [item for item in facts.bonds if item.id == bond]
    if len(matches) != 1:
        _offline_violation("directional_carrier_bond_mismatch")
    return matches[0]


def _facts_bond_is_bridge(*, facts: MoleculeFacts, bond: object) -> bool:
    graph_bond = _facts_bond(facts=facts, bond=bond)
    target = graph_bond.b
    seen = {graph_bond.a}
    stack = [graph_bond.a]
    while stack:
        atom = stack.pop()
        for incident in facts.bonds:
            if incident.id == bond:
                continue
            if incident.a == atom:
                neighbor = incident.b
            elif incident.b == atom:
                neighbor = incident.a
            else:
                continue
            if neighbor == target:
                return False
            if neighbor in seen:
                continue
            seen.add(neighbor)
            stack.append(neighbor)
    return True


def _check_directional_source_factor_snapshots(
    *,
    facts: MoleculeFacts,
    sites: tuple[DirectionalSiteFacts, ...],
    bond: object,
    source_snapshot: ResidualStoreValueSnapshot,
    models: tuple[DirectionalSiteCarrierModel, ...],
) -> None:
    factor_by_key = {factor.key: factor for factor in source_snapshot.factors}
    for site in sites:
        site_key = ResidualFactorKey("directional_site", (int(site.id),))
        site_factor = factor_by_key.get(site_key)
        if not isinstance(site_factor, DirectionalSiteFactorValueSnapshot):
            _offline_violation("directional_carrier_source_site_factor_mismatch")
        site_models = _facts_directional_models_for_site(facts=facts, site=site)
        expected_scope = tuple(
            directional_site_carrier_var(model.site, model.bond)
            for model in site_models
        )
        expected_sides = tuple(
            (var, model.side)
            for var, model in zip(expected_scope, site_models)
        )
        if (
            site_factor.scope != expected_scope
            or site_factor.sides != expected_sides
            or site_factor.status is not site.status
            or site_factor.target is not site.target
        ):
            _offline_violation("directional_carrier_source_site_factor_mismatch")
    bond_key = ResidualFactorKey("directional_bond_emission", (int(bond),))
    bond_factor = factor_by_key.get(bond_key)
    if not isinstance(bond_factor, DirectionalBondEmissionFactorValueSnapshot):
        _offline_violation("directional_carrier_source_bond_factor_mismatch")
    expected_vars = tuple(directional_site_carrier_var(model.site, model.bond) for model in models)
    if (
        bond_factor.scope != expected_vars
        or bond_factor.models != models
        or bond_factor.allowed_marks
        != (DirectionMark.ABSENT, DirectionMark.FWD, DirectionMark.REV)
    ):
        _offline_violation("directional_carrier_source_bond_factor_mismatch")


def _facts_directional_models_for_site(
    *,
    facts: MoleculeFacts,
    site: DirectionalSiteFacts,
) -> tuple[DirectionalSiteCarrierModel, ...]:
    occurrence_by_id = {occurrence.id: occurrence for occurrence in facts.ligand_occurrences}
    bonds = []
    for occurrence_id in site.left_ligands + site.right_ligands:
        occurrence = occurrence_by_id[occurrence_id]
        if occurrence.kind is LigandKind.NEIGHBOR_ATOM:
            bonds.append(occurrence.bond)
    models = []
    for bond in bonds:
        models.extend(_facts_directional_models_for_bond(facts=facts, site=site, bond=bond))
    return tuple(sorted(models, key=lambda model: (int(model.site), int(model.bond), model.side)))


def _expected_directional_discharge_keys(
    *,
    facts: MoleculeFacts,
    sites: tuple[DirectionalSiteFacts, ...],
    bond: object,
    source_state: Mapping[str, object],
) -> tuple[ResidualFactorKey, ...]:
    keys = [ResidualFactorKey("directional_bond_emission", (int(bond),))]
    emitted = {
        int(_term_field_value(record, "bond"))
        for record in _state_bond_occurrence_records(source_state)
    } | {int(bond)}
    for site in sorted(sites, key=lambda item: int(item.id)):
        site_bonds = {
            int(model.bond)
            for model in _facts_directional_models_for_site(facts=facts, site=site)
        }
        if site_bonds.issubset(emitted):
            keys.append(ResidualFactorKey("directional_site", (int(site.id),)))
    return tuple(keys)


def _closed_term_digest(term: object) -> str:
    return _digest_terms_bounded(
        term,
        budget=default_writer_envelope_work_budget(None),
        operation="support_artifact.offline.closed_term_digest",
    )


def _term_field_value(term: Mapping[str, object], name: str) -> object:
    for field_name, value in term["fields"]:
        if field_name == name:
            return value
    _offline_violation("closed_term_field_missing")


def _linked_raw_tetra_lifecycle(
    *,
    branch: Mapping[str, object],
    item: Mapping[str, object],
) -> Mapping[str, object]:
    matches = [
        lifecycle
        for lifecycle in branch["payload"]["obligation_manifests"]["stereo_lifecycle"]
        if lifecycle["operation"] == "WriterStereoLifecycleEvidence"
        and lifecycle["evidence_digest"] in item["linked_lifecycle_digests"]
    ]
    if len(matches) != 1:
        _offline_violation("tetra_residual_raw_lifecycle_binding_mismatch")
    return matches[0]


def _check_transition_local_order_event_binding(
    *,
    branch: Mapping[str, object],
    term: TetraLocalOrderFactorClosureTransitionTerm,
    source_state: Mapping[str, object],
    successor_state: Mapping[str, object],
) -> None:
    events = branch["payload"]["graph_ring_delta"]["manifest"]["event_manifests"]
    closed = [
        event
        for event in events
        if event["kind"] == "local_order_closed"
    ]
    if len(closed) != 1:
        _offline_violation("tetra_local_order_residual_close_event_count")
    event = closed[0]
    if event["atom"] != int(term.atom):
        _offline_violation("tetra_local_order_event_atom_mismatch")
    for field in (
        "site",
        "local_order",
        "reference_order",
        "source_local_order_record_digest",
        "successor_local_order_record_digest",
        "local_order_identity_digest",
    ):
        if field not in event:
            _offline_violation("tetra_local_order_event_identity_missing")
    if event["site"] != int(term.site):
        _offline_violation("tetra_local_order_event_site_mismatch")
    if tuple(event["local_order"]) != tuple(int(item) for item in term.local_order):
        _offline_violation("tetra_local_order_event_order_mismatch")
    if tuple(event["reference_order"]) != tuple(
        int(item) for item in term.reference_order
    ):
        _offline_violation("tetra_local_order_event_reference_order_mismatch")
    identity = {
        "site": event["site"],
        "atom": event["atom"],
        "local_order": event["local_order"],
        "reference_order": event["reference_order"],
        "source_local_order_record_digest": event[
            "source_local_order_record_digest"
        ],
        "successor_local_order_record_digest": event[
            "successor_local_order_record_digest"
        ],
    }
    if event["local_order_identity_digest"] != _identity_digest(identity):
        _offline_violation("tetra_local_order_event_identity_digest_mismatch")
    _check_transition_local_order_state_record_anchors(
        event=event,
        term=term,
        source_state=source_state,
        successor_state=successor_state,
    )


def _check_transition_local_order_state_record_anchors(
    *,
    event: Mapping[str, object],
    term: TetraLocalOrderFactorClosureTransitionTerm,
    source_state: Mapping[str, object],
    successor_state: Mapping[str, object],
) -> None:
    source_record = _local_order_record_for_site(
        _state_local_order_records(source_state),
        atom=int(term.atom),
        digest=event["source_local_order_record_digest"],
        mismatch_reason="tetra_local_order_source_record_anchor_mismatch",
    )
    successor_record = _local_order_record_for_site(
        _state_local_order_records(successor_state),
        atom=int(term.atom),
        digest=event["successor_local_order_record_digest"],
        mismatch_reason="tetra_local_order_successor_record_anchor_mismatch",
    )
    if _term_field_value(source_record, "closed"):
        _offline_violation("tetra_local_order_source_record_not_open")
    if not _term_field_value(successor_record, "closed"):
        _offline_violation("tetra_local_order_successor_record_not_closed")
    if tuple(_term_field_value(successor_record, "order")) != tuple(
        int(item) for item in term.local_order
    ):
        _offline_violation("tetra_local_order_successor_record_order_mismatch")


def _local_order_record_for_site(
    records: tuple[object, ...],
    *,
    atom: int,
    digest: str,
    mismatch_reason: str,
) -> Mapping[str, object]:
    matches = [
        record
        for record in records
        if isinstance(record, Mapping)
        and int(_term_field_value(record, "atom")) == atom
        and _closed_term_digest(record) == digest
    ]
    if len(matches) != 1:
        _offline_violation(mismatch_reason)
    return matches[0]


def _transition_from_manifest(item: Mapping[str, object]) -> object:
    if item["transition_term"] is None:
        _offline_violation("tetra_residual_transition_missing")
    operation = item["operation"]
    if operation == "tetrahedral atom-token restriction":
        return _decode_transition_term(
            item["transition_term"],
            expected_path=(
                "grimace._south_star1.writer_residual_transition_terms."
                "TetraAtomTokenRestrictionTransitionTerm"
            ),
        )
    if operation == "tetrahedral local-order factor closure":
        return _decode_transition_term(
            item["transition_term"],
            expected_path=(
                "grimace._south_star1.writer_residual_transition_terms."
                "TetraLocalOrderFactorClosureTransitionTerm"
            ),
        )
    if operation == "directional carrier-mark restriction":
        return _decode_transition_term(
            item["transition_term"],
            expected_path=(
                "grimace._south_star1.writer_residual_transition_terms."
                "DirectionalCarrierMarkRestrictionTransitionTerm"
            ),
        )
    if operation == "directional ring endpoint projection":
        shared_path = (
            "grimace._south_star1.writer_residual_transition_terms."
            "SharedDirectionalRingEndpointProjectionTransitionTerm"
        )
        if item["transition_term"].get("__dataclass__") == shared_path:
            return _decode_transition_term(
                item["transition_term"],
                expected_path=shared_path,
            )
        return _decode_transition_term(
            item["transition_term"],
            expected_path=(
                "grimace._south_star1.writer_residual_transition_terms."
                "DirectionalRingEndpointProjectionTransitionTerm"
            ),
        )
    if operation == "directional ring pair restriction":
        return _decode_transition_term(
            item["transition_term"],
            expected_path=(
                "grimace._south_star1.writer_residual_transition_terms."
                "DirectionalRingPairRestrictionTransitionTerm"
            ),
        )
    _offline_violation("tetra_residual_transition_operation_mismatch")


def _decode_transition_term(term: object, *, expected_path: str) -> object:
    if not isinstance(term, Mapping):
        _offline_violation("transition_term_shape_mismatch")
    if frozenset(term) != frozenset(("__dataclass__", "fields")):
        _offline_violation("transition_term_shape_mismatch")
    if term.get("__dataclass__") != expected_path:
        _offline_violation("transition_term_class_mismatch")
    return _value_from_term(term)


def _value_from_term(term: object) -> object:
    if term is None or isinstance(term, (str, bool, int)):
        return term
    if isinstance(term, Mapping):
        if "__enum__" in term:
            if frozenset(term) != frozenset(("__enum__", "value")):
                _offline_violation("transition_term_enum_shape_mismatch")
            enum_cls = _ALLOWED_TETRA_TRANSITION_ENUMS.get(term["__enum__"])
            if enum_cls is None:
                _offline_violation("transition_term_enum_class_mismatch")
            try:
                return enum_cls(term["value"])
            except ValueError:
                _offline_violation("transition_term_enum_value_mismatch")
        if "__dataclass__" in term:
            if frozenset(term) != frozenset(("__dataclass__", "fields")):
                _offline_violation("transition_term_dataclass_shape_mismatch")
            cls = _ALLOWED_TETRA_TRANSITION_DATACLASSES.get(term["__dataclass__"])
            if cls is None:
                _offline_violation("transition_term_dataclass_class_mismatch")
            fields = term["fields"]
            if not isinstance(fields, list):
                _offline_violation("transition_term_dataclass_fields_mismatch")
            values = {}
            for item in fields:
                if (
                    not isinstance(item, list)
                    or len(item) != 2
                    or not isinstance(item[0], str)
                ):
                    _offline_violation("transition_term_dataclass_field_mismatch")
                if item[0] in values:
                    _offline_violation("transition_term_dataclass_field_duplicate")
                values[item[0]] = _value_from_term(item[1])
            expected = _ALLOWED_TETRA_TRANSITION_DATACLASS_FIELDS[
                term["__dataclass__"]
            ]
            if frozenset(values) != expected:
                _offline_violation("transition_term_dataclass_fields_mismatch")
            return cls(**values)
        _offline_violation("transition_term_mapping_shape_mismatch")
    if isinstance(term, list):
        return tuple(_value_from_term(item) for item in term)
    _offline_violation("transition_term_shape_mismatch")


def _local_order_parity(
    *,
    reference_order: tuple[object, ...],
    local_order: tuple[object, ...],
) -> TetraLocalParity:
    if set(reference_order) != set(local_order):
        _offline_violation("tetra_local_order_reference_order_mismatch")
    positions = {item: index for index, item in enumerate(reference_order)}
    indices = tuple(positions[item] for item in local_order)
    inversions = 0
    for index, left in enumerate(indices):
        for right in indices[index + 1:]:
            if left > right:
                inversions += 1
    return TetraLocalParity.EVEN if inversions % 2 == 0 else TetraLocalParity.ODD


def _specified_tetra_site_for_transition(
    *,
    facts: MoleculeFacts,
    site: int,
    atom: int,
    violation_prefix: str,
) -> TetrahedralSiteFacts:
    matches = [
        item
        for item in facts.stereo.tetrahedral
        if int(item.id) == site
    ]
    if len(matches) != 1:
        _offline_violation(f"{violation_prefix}_site_mismatch")
    result = matches[0]
    if result.status is not SiteStatus.SPECIFIED:
        _offline_violation(f"{violation_prefix}_site_status_mismatch")
    if int(result.center) != atom:
        _offline_violation(f"{violation_prefix}_center_mismatch")
    return result


def _check_tetra_atom_token_residual(
    *,
    branch: Mapping[str, object],
    facts: MoleculeFacts,
) -> None:
    text = branch["payload"]["emitted_text"]
    if text not in ("[C@H]", "[C@@H]"):
        _offline_violation("tetra_atom_token_residual_text_mismatch")
    matching_atom = validate_writer_bracket_atom_text_against_facts(
        facts=facts,
        rendered_text=text,
    )
    expected_token = _tetra_token_from_rendered_text(text)
    delta = branch["payload"]["graph_ring_delta"]
    if delta["kind"] not in ("atom_start", "atom_advance", "bond_advance"):
        _offline_violation("tetra_atom_token_residual_delta_kind_mismatch")
    event = _single_atom_emitted_event(
        events=delta["manifest"]["event_manifests"],
        violation_prefix="tetra_atom_token_residual",
    )
    if event["text"] != text:
        _offline_violation("tetra_atom_token_residual_event_text_mismatch")
    if event["atom"] != _term(matching_atom.id):
        _offline_violation("tetra_atom_token_residual_atom_mismatch")
    _require_specified_tetra_center(
        facts=facts,
        atom=event["atom"],
        violation="tetra_atom_token_residual_center_mismatch",
    )
    token = event["tetra_token"]
    if token["value"] not in ("@", "@@"):
        _offline_violation("tetra_atom_token_residual_event_token_missing")
    if token["value"] != expected_token:
        _offline_violation("tetra_atom_token_residual_token_mismatch")


def _check_tetra_local_order_residual(
    *,
    branch: Mapping[str, object],
    facts: MoleculeFacts,
) -> None:
    if not facts.stereo.tetrahedral:
        _offline_violation("tetra_local_order_residual_site_missing")
    delta = branch["payload"]["graph_ring_delta"]
    if delta["kind"] not in ("atom_start", "atom_advance", "bond_advance"):
        _offline_violation("tetra_local_order_residual_delta_kind_mismatch")
    events = delta["manifest"]["event_manifests"]
    closed_atoms = [
        event["atom"]
        for event in events
        if event["kind"] == "local_order_closed"
    ]
    if len(closed_atoms) != 1:
        _offline_violation("tetra_local_order_residual_close_event_count")
    specified_centers = {
        _term(site.center)
        for site in facts.stereo.tetrahedral
        if site.status is SiteStatus.SPECIFIED
    }
    if closed_atoms[0] not in specified_centers:
        _offline_violation("tetra_local_order_residual_center_mismatch")
    emitted = _single_atom_emitted_event(
        events=events,
        violation_prefix="tetra_local_order_residual",
    )
    if emitted["parent"] != closed_atoms[0]:
        _offline_violation("tetra_local_order_residual_parent_mismatch")
    if not _bond_between_facts_atoms(
        facts=facts,
        bond_id=emitted["incoming_bond"],
        left_atom=closed_atoms[0],
        right_atom=emitted["atom"],
    ):
        _offline_violation("tetra_local_order_residual_bond_mismatch")


def _unchecked_obligation_family_name(
    family: str,
    items: list[object],
    *,
    facts: MoleculeFacts,
) -> str:
    directional_bonds = [
        bond
        for bond in facts.bonds
        if _expected_directional_sites_for_facts_bond(facts=facts, bond=bond.id)
    ]
    if any(bond.order is not BondOrder.SINGLE for bond in directional_bonds):
        return "directional_non_single_ring_transition_replay"
    if any(
        len(_expected_directional_sites_for_facts_bond(facts=facts, bond=bond.id)) > 1
        for bond in directional_bonds
    ):
        return "shared_directional_ring_transition_replay"
    if family == "residual_work" and all(
        item["operation"]
        in {
            "tetrahedral atom-token restriction",
            "tetrahedral local-order factor closure",
        }
        for item in items
    ):
        return "tetra_residual_link_provenance_replay"
    if family == "residual_work" and any(
        item["operation"] == "directional ring pair restriction"
        for item in items
    ):
        return "directional_ring_pair_transition_replay"
    if family == "stereo_lifecycle" and any(
        item["residual_work_operations"] == ["directional ring pair restriction"]
        for item in items
    ):
        return "directional_ring_pair_transition_replay"
    return family


def _ring_obligation_manifest_checked(item: Mapping[str, object]) -> bool:
    summary = item["ring_summary"]
    if summary is None:
        return False
    family = item["family"]
    if not summary["is_exact"]:
        return False
    if family == "finite_relation_work":
        if item["operation"] not in (
            "closure endpoint open relation",
            "closure endpoint pair relation",
        ):
            return False
        if summary["operation"] != item["operation"]:
            return False
        return bool(summary["is_exhausted"] and summary["is_discharged"])
    if family == "graph_obligation_work":
        if item["operation"] != "writer graph obligation context":
            return False
        if summary["operation"] != item["operation"]:
            return False
        kind = summary["relation_kind"]
        if kind == "ring_endpoint_open":
            return bool(
                summary["is_complete"]
                and summary["pending_before_count"] == 0
                and summary["pending_after_count"] == 1
            )
        if kind in ("ring_endpoint_pair", "ring_endpoint_pair_non_single"):
            return bool(
                summary["is_complete"]
                and summary["is_discharged"]
                and summary["pending_before_count"] == 1
                and summary["pending_after_count"] == 0
                and summary["closed_after_count"] == 1
            )
    return False


def _check_branch_obligation_ring_summaries(branch: Mapping[str, object]) -> None:
    payload = branch["payload"]
    delta = payload["graph_ring_delta"]
    kind = delta["kind"]
    manifests = payload["obligation_manifests"]
    ring_kind = kind in (
        "ring_endpoint_open",
        "ring_endpoint_pair",
        "ring_endpoint_pair_non_single",
    )
    for family in ("finite_relation_work", "graph_obligation_work"):
        for item in manifests[family]:
            summary = item["ring_summary"]
            if not ring_kind:
                if summary is not None:
                    _offline_violation("non_ring_obligation_has_ring_summary")
                continue
            if summary is None:
                _offline_violation("ring_obligation_summary_missing")
            _check_ring_obligation_summary_against_delta(
                family=family,
                item=item,
                summary=summary,
                delta=delta,
            )


def _check_ring_obligation_summary_against_delta(
    *,
    family: str,
    item: Mapping[str, object],
    summary: Mapping[str, object],
    delta: Mapping[str, object],
) -> None:
    kind = delta["kind"]
    if summary["relation_kind"] != kind:
        _offline_violation("ring_obligation_relation_kind_mismatch")
    event_kind = (
        "ring_endpoint_emitted"
        if kind == "ring_endpoint_open"
        else "ring_endpoint_paired"
    )
    events = delta["manifest"]["event_manifests"]
    ring_events = [event for event in events if event["kind"] == event_kind]
    if len(ring_events) != 1:
        _offline_violation("ring_obligation_event_count_mismatch")
    event = ring_events[0]
    for field, event_field in (
        ("bond", "bond"),
        ("endpoint_atom", "endpoint_atom"),
        ("partner_atom", "partner_atom"),
        ("ring_label", "label"),
        ("side", "side"),
    ):
        if summary[field] != event[event_field]:
            _offline_violation(f"ring_obligation_{field}_mismatch")
    marker = event["bond_text"]
    marker_count = int(bool(event["bond_text"]))
    if kind != "ring_endpoint_open":
        marker = marker or event["first_endpoint_bond_text"]
        marker_count += int(bool(event["first_endpoint_bond_text"]))
    if summary["marker"] != marker:
        _offline_violation("ring_obligation_marker_mismatch")
    if summary["marker_count"] != marker_count:
        _offline_violation("ring_obligation_marker_count_mismatch")
    expected_operation = (
        "closure endpoint open relation"
        if family == "finite_relation_work" and kind == "ring_endpoint_open"
        else "closure endpoint pair relation"
        if family == "finite_relation_work"
        else "writer graph obligation context"
    )
    if item["operation"] != expected_operation or summary["operation"] != expected_operation:
        _offline_violation("ring_obligation_operation_mismatch")
    closing = kind in ("ring_endpoint_pair", "ring_endpoint_pair_non_single")
    if summary["pending_before_count"] != (1 if closing else 0):
        _offline_violation("ring_obligation_pending_before_mismatch")
    if summary["pending_after_count"] != (0 if closing else 1):
        _offline_violation("ring_obligation_pending_after_mismatch")
    if summary["closed_after_count"] != (1 if closing else 0):
        _offline_violation("ring_obligation_closed_after_mismatch")
    if closing and not any(event["kind"] == "ring_label_released" for event in events):
        _offline_violation("ring_obligation_pair_lacks_label_release")


def verify_terminal_support_identities_offline(
    *,
    artifact: Mapping[str, object],
    objects: Mapping[str, Mapping[str, object]],
) -> TerminalSupportIdentityVerification:
    try:
        root = _require_object(objects, artifact["roots"]["support_image_root"])
        terminal_projection_refs: set[str] = set()
        terminal_support_refs: set[str] = set()
        checked_paths = 0
        for support_ref in root["payload"]["support_string_refs"]:
            support = _require_object(objects, support_ref)
            _check_support_terminal_path(
                support=support,
                objects=objects,
                terminal_projection_refs=terminal_projection_refs,
                terminal_support_refs=terminal_support_refs,
            )
            checked_paths += 1
        for projection_ref in terminal_projection_refs:
            projection = _require_object(objects, projection_ref)
            _check_terminal_projection_identity(projection)
        for support_ref in terminal_support_refs:
            terminal_support = _require_object(objects, support_ref)
            _check_terminal_support_identity(terminal_support)
        _check_terminal_bucket_identity(
            root=root,
            objects=objects,
        )
        return TerminalSupportIdentityVerification(
            accepted=True,
            checked_terminal_projections=len(terminal_projection_refs),
            checked_terminal_supports=len(terminal_support_refs),
            checked_terminal_paths=checked_paths,
        )
    except SouthStarError as exc:
        return TerminalSupportIdentityVerification(
            accepted=False,
            reason=exc.args[-1] if exc.args else "terminal_support_identity_error",
        )
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        return TerminalSupportIdentityVerification(
            accepted=False,
            reason=f"malformed_terminal_support_identity:{type(exc).__name__}",
        )


def _check_support_terminal_path(
    *,
    support: Mapping[str, object],
    objects: Mapping[str, Mapping[str, object]],
    terminal_projection_refs: set[str],
    terminal_support_refs: set[str],
) -> None:
    payload = support["payload"]
    terminal = _require_object(objects, payload["terminal_projection_ref"])
    if terminal["kind"] != "terminal_projection":
        _offline_violation("terminal_projection_ref_kind_mismatch")
    terminal_projection_refs.add(payload["terminal_projection_ref"])
    replay = _require_object(objects, payload["replay_path_ref"])
    if (
        terminal["payload"]["source_cursor"]["digest"]
        != replay["payload"]["final_cursor_digest"]
    ):
        _offline_violation("terminal_projection_source_cursor_mismatch")
    refs = payload["terminal_support_refs"]
    if not refs:
        _offline_violation("terminal_support_refs_missing")
    identity_by_digest = {
        identity["digest"]: identity
        for identity in terminal["payload"]["terminal_support_identities"]
    }
    if len(set(refs)) != len(refs):
        _offline_violation("terminal_support_ref_duplicate")
    support_digests: list[str] = []
    for ref in refs:
        terminal_support = _require_object(objects, ref)
        if terminal_support["kind"] != "terminal_support":
            _offline_violation("terminal_support_ref_kind_mismatch")
        terminal_support_refs.add(ref)
        digest = terminal_support["payload"]["digest"]
        support_digests.append(digest)
        if digest not in identity_by_digest:
            _offline_violation("terminal_support_not_in_projection")
        if (
            _terminal_support_identity_payload(terminal_support["payload"])
            != identity_by_digest[digest]
        ):
            _offline_violation("terminal_support_identity_mismatch")
    if set(support_digests) != set(identity_by_digest):
        _offline_violation("terminal_projection_support_set_mismatch")


def _check_terminal_projection_identity(projection: Mapping[str, object]) -> None:
    payload = projection["payload"]
    if payload["support_count"] < 0:
        _offline_violation("terminal_projection_negative_support_count")
    if payload["completion_count"] < 0:
        _offline_violation("terminal_projection_negative_completion_count")
    identities = payload["terminal_support_identities"]
    if payload["multiplicity"] != len(identities):
        _offline_violation("terminal_projection_multiplicity_mismatch")
    if len(payload["terminal_certificate_digests"]) != len(identities):
        _offline_violation("terminal_projection_certificate_digest_count_mismatch")
    ordinals = [identity["terminal_ordinal"] for identity in identities]
    if len(set(ordinals)) != len(ordinals):
        _offline_violation("terminal_projection_duplicate_ordinal")
    keys = [identity["terminal_support_key_digest"] for identity in identities]
    if len(set(keys)) != len(keys):
        _offline_violation("terminal_projection_duplicate_key_digest")
    for identity in identities:
        _check_terminal_support_payload(identity)


def _check_terminal_support_identity(terminal_support: Mapping[str, object]) -> None:
    _check_terminal_support_payload(terminal_support["payload"])


def _terminal_support_identity_payload(payload: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in payload.items()
        if key not in ("obligation_summary", "obligation_manifests")
    }


def _check_terminal_support_payload(payload: Mapping[str, object]) -> None:
    if payload["parent_weight"] <= 0:
        _offline_violation("terminal_support_parent_weight_nonpositive")
    if payload["terminal_ordinal"] < 0:
        _offline_violation("terminal_support_ordinal_negative")
    for field in (
        "source_state_digest",
        "finalized_state_digest",
        "terminal_support_key_digest",
        "digest",
    ):
        if not payload[field]:
            _offline_violation("terminal_support_identity_digest_missing")
    if not payload["terminal_certificate_digests"]:
        _offline_violation("terminal_support_certificate_digests_missing")


def _check_terminal_bucket_identity(
    *,
    root: Mapping[str, object],
    objects: Mapping[str, Mapping[str, object]],
) -> None:
    coverage = _require_object(objects, root["payload"]["coverage_ref"])
    terminal_bucket = coverage["payload"]["terminal_bucket"]
    empty_refs = [
        ref
        for ref in root["payload"]["support_string_refs"]
        if not _require_object(objects, ref)["payload"]["emitted_texts"]
    ]
    if terminal_bucket is None:
        if empty_refs:
            _offline_violation("terminal_bucket_missing_for_empty_string")
        return
    if terminal_bucket["support_count"] != len(empty_refs):
        _offline_violation("terminal_bucket_support_count_mismatch")
    if not empty_refs:
        if terminal_bucket["string_ref"] is not None:
            _offline_violation("terminal_bucket_unexpected_string_ref")
        return
    if terminal_bucket["string_ref"] != empty_refs[0]:
        _offline_violation("terminal_bucket_string_ref_mismatch")
    support = _require_object(objects, empty_refs[0])
    terminal = _require_object(objects, support["payload"]["terminal_projection_ref"])
    if terminal_bucket["terminal_projection"] != terminal["payload"]:
        _offline_violation("terminal_bucket_projection_mismatch")
    terminal_digests = {
        _require_object(objects, ref)["payload"]["digest"]
        for ref in support["payload"]["terminal_support_refs"]
    }
    projection_digests = {
        identity["digest"]
        for identity in terminal["payload"]["terminal_support_identities"]
    }
    if terminal_digests != projection_digests:
        _offline_violation("terminal_bucket_support_identity_mismatch")


def _branch_support_refs_for_root(
    *,
    root: Mapping[str, object],
    objects: Mapping[str, Mapping[str, object]],
) -> tuple[str, ...]:
    refs: set[str] = set()
    for support_ref in root["payload"]["support_string_refs"]:
        support = _require_object(objects, support_ref)
        for projection_ref in support["payload"]["text_projection_refs"]:
            projection = _require_object(objects, projection_ref)
            refs.update(projection["payload"]["branch_support_refs"])
    return tuple(sorted(refs))


def _branch_refs_from_support_artifact(
    *,
    artifact: Mapping[str, object],
    objects: Mapping[str, Mapping[str, object]],
) -> tuple[str, ...]:
    root = _require_object(objects, artifact["roots"]["support_image_root"])
    return _branch_support_refs_for_root(root=root, objects=objects)


def _branch_ref_from_transition_artifact(
    *, artifact: Mapping[str, object], objects: Mapping[str, Mapping[str, object]]
) -> str:
    branch_ref = artifact["roots"]["branch_support_ref"]
    branch = _require_object(objects, branch_ref)
    if branch["kind"] != "branch_support":
        _offline_violation("transition_branch_support_ref_kind_mismatch")
    return branch_ref


def _check_graph_ring_branch_delta(
    *,
    facts: MoleculeFacts,
    branch: Mapping[str, object],
    budget: WriterEnvelopeWorkBudget,
) -> str:
    payload = branch["payload"]
    delta = payload["graph_ring_delta"]
    kind = delta["kind"]
    manifest = delta["manifest"]
    expected_digest = _identity_digest(
        {"kind": kind, "manifest": manifest},
        budget=budget,
        operation=f"offline.graph_ring_delta.{kind}.digest",
    )
    if delta["digest"] != expected_digest:
        _offline_violation("graph_ring_delta_digest_mismatch")
    _check_graph_ring_delta_common(payload=payload, manifest=manifest)
    events = manifest["event_manifests"]
    if kind == "other_structural":
        return kind
    if kind in ("atom_start", "atom_advance"):
        _check_atom_delta_events(facts=facts, branch_payload=payload, events=events)
        return kind
    if kind == "bond_advance":
        _check_bond_delta_events(facts=facts, branch_payload=payload, events=events)
        return kind
    if kind in ("branch_open", "branch_return"):
        _check_branch_delta_events(facts=facts, kind=kind, events=events)
        return kind
    if kind in (
        "ring_endpoint_open",
        "ring_endpoint_pair",
        "ring_endpoint_pair_non_single",
    ):
        _check_ring_delta_events(
            facts=facts,
            branch_payload=payload,
            kind=kind,
            events=events,
        )
        return kind
    _offline_violation("graph_ring_delta_unknown_kind")


def _check_graph_ring_delta_common(
    *,
    payload: Mapping[str, object],
    manifest: Mapping[str, object],
) -> None:
    for field in (
        "source_state_digest",
        "successor_state_digest",
        "source_cursor_digest",
        "successor_cursor_digest",
        "transition_kind",
        "emitted_text",
        "graph_action_surface_digest",
        "successor_state_certificate_digest",
        "checked_branch_certificate_digest",
    ):
        if manifest[field] != payload[field]:
            _offline_violation(f"graph_ring_delta_{field}_mismatch")
    if manifest["local_evidence_digest"] != payload["local_evidence"]["digest"]:
        _offline_violation("graph_ring_delta_local_evidence_digest_mismatch")
    if payload["parent_weight"] <= 0:
        _offline_violation("graph_ring_delta_parent_weight_nonpositive")
    if payload["branch_ordinal"] < 0:
        _offline_violation("graph_ring_delta_branch_ordinal_negative")
    if not payload["source_state_digest"] or not payload["successor_state_digest"]:
        _offline_violation("graph_ring_delta_state_digest_missing")
    if not payload["successor_state_certificate_digest"]:
        _offline_violation("graph_ring_delta_successor_certificate_digest_missing")
    if not payload["checked_branch_certificate_digest"]:
        _offline_violation("graph_ring_delta_checked_certificate_digest_missing")


def _check_atom_delta_events(
    *,
    facts: MoleculeFacts,
    branch_payload: Mapping[str, object],
    events: object,
) -> None:
    atom_events = [event for event in events if event["kind"] == "atom_emitted"]
    if len(atom_events) != 1:
        _offline_violation("graph_ring_atom_event_count_mismatch")
    event = atom_events[0]
    atom = _atom_by_term(facts, event["atom"])
    if event["text"] != branch_payload["emitted_text"]:
        _offline_violation("graph_ring_atom_event_text_mismatch")
    if atom.symbol not in event["text"]:
        _offline_violation("graph_ring_atom_event_symbol_mismatch")
    if event["incoming_bond"] is not None:
        bond = _bond_by_term(facts, event["incoming_bond"])
        if event["atom"] not in (_term(bond.a), _term(bond.b)):
            _offline_violation("graph_ring_atom_incoming_bond_endpoint_mismatch")


def _check_bond_delta_events(
    *,
    facts: MoleculeFacts,
    branch_payload: Mapping[str, object],
    events: object,
) -> None:
    bond_events = [event for event in events if event["kind"] == "bond_emitted"]
    if len(bond_events) != 1:
        _offline_violation("graph_ring_bond_event_count_mismatch")
    event = bond_events[0]
    bond = _bond_by_term(facts, event["bond"])
    endpoints = {_term(bond.a), _term(bond.b)}
    if {event["parent"], event["child"]} != endpoints:
        _offline_violation("graph_ring_bond_endpoint_mismatch")
    expected_marker = _bond_order_marker(bond.order)
    if event["text"] != expected_marker:
        expected_direction_text = _direction_mark_text(event["direction_mark"])
        accepts_direction_text = (
            expected_marker == ""
            and expected_direction_text != ""
            and event["text"] == expected_direction_text
        )
        if not accepts_direction_text:
            _offline_violation(
                "graph_ring_bond_marker_mismatch:"
                f"bond={event['bond']};"
                f"parent={event['parent']};"
                f"child={event['child']};"
                f"expected_marker={expected_marker!r};"
                f"observed_text={event['text']!r};"
                f"expected_direction_text={expected_direction_text!r};"
                f"direction_mark={event['direction_mark']!r};"
                f"emitted_text={branch_payload['emitted_text']!r};"
                f"local_evidence_kind={branch_payload['local_evidence']['kind']};"
                "expected_marker_side=bond_advance;"
                "observed_marker_side=bond_advance;"
                f"closure_text_pair={_closure_text_pair(branch_payload)};"
                f"successor_certificate="
                f"{branch_payload['successor_state_certificate_digest']};"
                f"checked_branch_certificate="
                f"{branch_payload['checked_branch_certificate_digest']}"
            )
    if event["text"] and event["text"] not in branch_payload["emitted_text"]:
        _offline_violation("graph_ring_bond_event_text_mismatch")


def _direction_mark_text(mark: object) -> str:
    if mark == _term(DirectionMark.FWD):
        return "/"
    if mark == _term(DirectionMark.REV):
        return "\\"
    if mark == _term(DirectionMark.ABSENT):
        return ""
    _offline_violation("graph_ring_bond_direction_mark_unknown")


def _closure_text_pair(branch_payload: Mapping[str, object]) -> object:
    local_evidence = branch_payload["local_evidence"]
    manifest = local_evidence["manifest"]
    if local_evidence["kind"] == "closure_bond_text":
        items = manifest["items"]
    elif local_evidence["kind"] == "directional_ring_closure_bond_text":
        items = manifest["closure_bond_text"]
    else:
        return None
    return tuple(
        (
            item["opening_marker"],
            item["closing_marker"],
            item["marker_side"],
        )
        for item in items
    )


def _check_branch_delta_events(
    *,
    facts: MoleculeFacts,
    kind: str,
    events: object,
) -> None:
    event_kind = "branch_opened" if kind == "branch_open" else "branch_closed"
    branch_events = [event for event in events if event["kind"] == event_kind]
    if len(branch_events) != 1:
        _offline_violation("graph_ring_branch_event_count_mismatch")
    event = branch_events[0]
    if kind == "branch_open":
        bond = _bond_by_term(facts, event["bond"])
        if {event["parent"], event["child"]} != {_term(bond.a), _term(bond.b)}:
            _offline_violation("graph_ring_branch_open_endpoint_mismatch")
    else:
        _atom_by_term(facts, event["atom"])


def _check_ring_delta_events(
    *,
    facts: MoleculeFacts,
    branch_payload: Mapping[str, object],
    kind: str,
    events: object,
) -> None:
    event_kind = (
        "ring_endpoint_emitted"
        if kind == "ring_endpoint_open"
        else "ring_endpoint_paired"
    )
    ring_events = [event for event in events if event["kind"] == event_kind]
    if len(ring_events) != 1:
        _offline_violation("graph_ring_endpoint_event_count_mismatch")
    event = ring_events[0]
    bond = _bond_by_term(facts, event["bond"])
    endpoints = {_term(bond.a), _term(bond.b)}
    if {event["endpoint_atom"], event["partner_atom"]} != endpoints:
        _offline_violation("graph_ring_endpoint_atoms_mismatch")
    if event["endpoint_text"] not in branch_payload["emitted_text"]:
        _offline_violation("graph_ring_endpoint_text_mismatch")
    if event["bond_text"] and event["bond_text"] not in branch_payload["emitted_text"]:
        _offline_violation("graph_ring_endpoint_bond_text_mismatch")
    marker = _bond_order_marker(bond.order)
    if marker and kind == "ring_endpoint_pair_non_single":
        marker_count = int(event["bond_text"] == marker)
        if event_kind == "ring_endpoint_paired":
            marker_count += int(event["first_endpoint_bond_text"] == marker)
        if marker_count != 1:
            _offline_violation("graph_ring_non_single_marker_count_mismatch")
    if kind == "ring_endpoint_pair_non_single":
        local_kind = branch_payload["local_evidence"]["kind"]
        if local_kind not in (
            "closure_bond_text",
            "directional_ring_closure_bond_text",
        ):
            _offline_violation("graph_ring_non_single_missing_closure_evidence")
        labels = _closure_evidence_labels(branch_payload["local_evidence"])
        if not any(event["label"] == label for label in labels):
            _offline_violation("graph_ring_endpoint_label_mismatch")


def _closure_evidence_labels(local_evidence: Mapping[str, object]) -> tuple[object, ...]:
    manifest = local_evidence["manifest"]
    if local_evidence["kind"] == "closure_bond_text":
        items = manifest["items"]
    elif local_evidence["kind"] == "directional_ring_closure_bond_text":
        items = manifest["closure_bond_text"]
    else:
        return ()
    return tuple(item["label"] for item in items)


def _check_branch_local_evidence(
    *,
    facts: MoleculeFacts,
    branch: Mapping[str, object],
    budget: WriterEnvelopeWorkBudget,
) -> str:
    payload = branch["payload"]
    evidence = payload["local_evidence"]
    if not payload["successor_state_certificate_digest"]:
        _offline_violation("local_branch_successor_certificate_digest_missing")
    if not payload["checked_branch_certificate_digest"]:
        _offline_violation("local_branch_checked_certificate_digest_missing")
    kind = evidence["kind"]
    expected_digest = _identity_digest(
        {"kind": kind, "manifest": evidence["manifest"]},
        budget=budget,
        operation=f"offline.local_branch_evidence.{kind}.digest",
    )
    if evidence["digest"] != expected_digest:
        _offline_violation("local_branch_evidence_digest_mismatch")
    if kind == "other_structural":
        if evidence["manifest"]:
            _offline_violation("local_branch_other_structural_manifest_not_empty")
        return kind
    if kind == "plain_atom_text":
        _check_plain_atom_text_local_evidence(
            facts=facts,
            branch_payload=payload,
            manifest=evidence["manifest"],
        )
        return kind
    if kind == "bracket_atom_text":
        _check_bracket_atom_text_local_evidence(
            facts=facts,
            branch_payload=payload,
            manifest=evidence["manifest"],
        )
        return kind
    if kind == "closure_bond_text":
        _check_closure_bond_text_local_evidence(
            facts=facts,
            branch_payload=payload,
            items=evidence["manifest"]["items"],
        )
        return kind
    if kind == "directional_ring_closure_bond_text":
        manifest = evidence["manifest"]
        _check_closure_bond_text_local_evidence(
            facts=facts,
            branch_payload=payload,
            items=manifest["closure_bond_text"],
        )
        if manifest["directional_coupled_count"] != len(
            manifest["directional_coupled_digests"]
        ):
            _offline_violation("local_directional_coupled_count_mismatch")
        return kind
    _offline_violation("local_branch_unknown_evidence_kind")


def _check_plain_atom_text_local_evidence(
    *,
    facts: MoleculeFacts,
    branch_payload: Mapping[str, object],
    manifest: Mapping[str, object],
) -> None:
    if branch_payload["emitted_text"] != manifest["rendered_text"]:
        _offline_violation("local_plain_atom_text_rendered_text_mismatch")
    if manifest["bracket_required"]:
        _offline_violation("local_plain_atom_text_bracket_required")
    atom = _atom_by_term(facts, manifest["atom_id"])
    if atom.symbol != manifest["element"]:
        _offline_violation("local_plain_atom_text_element_mismatch")
    if atom.is_aromatic != manifest["aromatic"]:
        _offline_violation("local_plain_atom_text_aromatic_mismatch")
    if atom.isotope is not None:
        _offline_violation("local_plain_atom_text_isotope_present")
    if atom.formal_charge != 0:
        _offline_violation("local_plain_atom_text_charge_present")
    if atom.symbol != manifest["rendered_text"]:
        _offline_violation("local_plain_atom_text_facts_mismatch")


def _check_bracket_atom_text_local_evidence(
    *,
    facts: MoleculeFacts,
    branch_payload: Mapping[str, object],
    manifest: Mapping[str, object],
) -> None:
    if branch_payload["emitted_text"] != manifest["rendered_text"]:
        _offline_violation("local_bracket_atom_text_rendered_text_mismatch")
    if not manifest["bracket_required"]:
        _offline_violation("local_bracket_atom_text_bracket_not_required")
    atom = _atom_by_term(facts, manifest["atom_id"])
    if atom.symbol != manifest["element"]:
        _offline_violation("local_bracket_atom_text_element_mismatch")
    if atom.isotope != manifest["isotope"]:
        _offline_violation("local_bracket_atom_text_isotope_mismatch")
    if atom.formal_charge != manifest["formal_charge"]:
        _offline_violation("local_bracket_atom_text_charge_mismatch")
    if atom.implicit_h_count != manifest["hydrogen_count"]:
        _offline_violation("local_bracket_atom_text_hydrogen_count_mismatch")
    if atom.is_aromatic != manifest["aromatic"]:
        _offline_violation("local_bracket_atom_text_aromatic_mismatch")
    if not _bracket_atom_text_matches_facts(
        facts=facts,
        atom=atom,
        rendered_text=manifest["rendered_text"],
    ):
        _offline_violation("local_bracket_atom_text_facts_mismatch")


def _check_closure_bond_text_local_evidence(
    *,
    facts: MoleculeFacts,
    branch_payload: Mapping[str, object],
    items: object,
) -> None:
    if not items:
        _offline_violation("local_closure_bond_text_items_missing")
    for item in items:
        bond = _bond_by_term(facts, item["bond"])
        expected_order = _bond_order_value(bond.order)
        if item["bond_order"] != expected_order:
            _offline_violation("local_closure_bond_order_mismatch")
        marker = {"double": "=", "triple": "#"}[item["bond_order"]]
        marker_count = (
            int(item["opening_marker"] == marker)
            + int(item["closing_marker"] == marker)
        )
        if marker_count == 0:
            _offline_violation("local_closure_marker_missing")
        if marker_count > 1:
            _offline_violation("local_closure_marker_duplicate")
        if item["marker_side"] == "opening" and item["opening_marker"] != marker:
            _offline_violation("local_closure_marker_side_mismatch")
        if item["marker_side"] == "closing" and item["closing_marker"] != marker:
            _offline_violation("local_closure_marker_side_mismatch")
        if (
            item["event_kind"] == "endpoint_emitted"
            and marker not in branch_payload["emitted_text"]
        ):
            _offline_violation("local_closure_branch_marker_missing")
        if item["event_kind"] == "endpoint_paired" and item["closing_marker"]:
            if marker not in branch_payload["emitted_text"]:
                _offline_violation("local_closure_branch_marker_missing")


def _check_text_projection_branch_identities(
    *,
    projection: Mapping[str, object],
    objects: Mapping[str, Mapping[str, object]],
    checked_branch_refs: set[str],
) -> None:
    payload = projection["payload"]
    branch_refs = payload["branch_support_refs"]
    if not branch_refs:
        _offline_violation("branch_projection_support_refs_missing")
    if len(set(branch_refs)) != len(branch_refs):
        _offline_violation("branch_projection_duplicate_support_ref")
    if int(payload["immediate_multiplicity"]) != len(branch_refs):
        _offline_violation("branch_projection_multiplicity_mismatch")
    branch_digests = set(payload["branch_certificate_digests"])
    if len(branch_digests) != len(branch_refs):
        _offline_violation("branch_projection_certificate_digest_count_mismatch")
    for branch_ref in branch_refs:
        branch = _require_object(objects, branch_ref)
        if branch["kind"] != "branch_support":
            _offline_violation("branch_projection_support_ref_kind_mismatch")
        checked_branch_refs.add(branch_ref)
        branch_payload = branch["payload"]
        if branch_payload["emitted_text"] != payload["emitted_text"]:
            _offline_violation("branch_projection_emitted_text_mismatch")
        if branch_payload["source_cursor_digest"] != payload["source_cursor"]["digest"]:
            _offline_violation("branch_projection_source_cursor_mismatch")
        if (
            branch_payload["successor_cursor_digest"]
            != payload["successor_cursor"]["digest"]
        ):
            _offline_violation("branch_projection_successor_cursor_mismatch")
        if branch_payload["checked_branch_certificate_digest"] not in branch_digests:
            _offline_violation("branch_projection_certificate_digest_mismatch")


def _check_support_string_replay_path(
    *,
    support: Mapping[str, object],
    source_cursor: Mapping[str, object],
    objects: Mapping[str, Mapping[str, object]],
) -> None:
    payload = support["payload"]
    emitted_texts = payload["emitted_texts"]
    text_refs = payload["text_projection_refs"]
    if payload["string"] != "".join(emitted_texts):
        _offline_violation("replay_path_support_string_join_mismatch")
    if len(text_refs) != len(emitted_texts):
        _offline_violation("replay_path_text_projection_count_mismatch")
    replay = _require_object(objects, payload["replay_path_ref"])
    if replay["kind"] != "replay_path":
        _offline_violation("replay_path_ref_kind_mismatch")
    replay_payload = replay["payload"]
    if replay_payload["emitted_texts"] != emitted_texts:
        _offline_violation("replay_path_emitted_texts_mismatch")
    if replay_payload["text_projection_refs"] != text_refs:
        _offline_violation("replay_path_text_projection_refs_mismatch")

    current_cursor = source_cursor
    for index, (projection_ref, emitted_text) in enumerate(
        zip(text_refs, emitted_texts, strict=True)
    ):
        projection = _require_object(objects, projection_ref)
        if projection["kind"] != "text_projection":
            _offline_violation("replay_path_text_projection_ref_kind_mismatch")
        projection_payload = projection["payload"]
        if projection_payload["emitted_text"] != emitted_text:
            _offline_violation("replay_path_projection_text_mismatch")
        if projection_payload["source_cursor"] != current_cursor:
            _offline_violation("replay_path_projection_source_cursor_mismatch")
        current_cursor = projection_payload["successor_cursor"]
        if (
            index == len(text_refs) - 1
            and current_cursor["digest"] != replay_payload["final_cursor_digest"]
        ):
            _offline_violation("replay_path_final_cursor_mismatch")
    if (
        not text_refs
        and source_cursor["digest"] != replay_payload["final_cursor_digest"]
    ):
        _offline_violation("replay_path_empty_final_cursor_mismatch")

    terminal = _require_object(objects, payload["terminal_projection_ref"])
    if terminal["kind"] != "terminal_projection":
        _offline_violation("replay_path_terminal_projection_ref_kind_mismatch")
    terminal_payload = terminal["payload"]
    if (
        terminal_payload["source_cursor"]["digest"]
        != replay_payload["final_cursor_digest"]
    ):
        _offline_violation("replay_path_terminal_source_cursor_mismatch")
    terminal_identities = {
        item["digest"]
        for item in terminal_payload["terminal_support_identities"]
    }
    for terminal_ref in payload["terminal_support_refs"]:
        terminal_support = _require_object(objects, terminal_ref)
        if terminal_support["kind"] != "terminal_support":
            _offline_violation("replay_path_terminal_support_ref_kind_mismatch")
        if terminal_support["payload"]["digest"] not in terminal_identities:
            _offline_violation("replay_path_terminal_support_identity_mismatch")


def _node_count(
    nodes: Mapping[str, Mapping[str, object]],
    node_id: str,
    *,
    field: str,
    checked: set[str],
) -> int:
    _check_node_arithmetic(nodes, node_id, checked=checked)
    node = nodes[node_id]
    value = node[field]
    if not isinstance(value, int):
        _offline_violation("count_dag_node_count_not_int")
    return value


def _same_text_projection_core(
    left: Mapping[str, object],
    right: Mapping[str, object],
) -> bool:
    return (
        left.get("emitted_text") == right.get("emitted_text")
        and left.get("source_cursor") == right.get("source_cursor")
        and left.get("successor_cursor") == right.get("successor_cursor")
    )


def _check_node_arithmetic(
    nodes: Mapping[str, Mapping[str, object]],
    node_id: str,
    *,
    checked: set[str],
) -> None:
    node = _require_count_node(nodes, node_id)
    kind = str(node["kind"])
    checked.add(kind)
    if kind == "writer_text_support_count":
        child = node["state_support_count_node_id"]
        if node["support_count"] != _node_count(
            nodes,
            child,
            field="support_count",
            checked=checked,
        ):
            _offline_violation("text_support_count_mismatch")
        return
    if kind == "writer_text_state_support_count":
        total = int(node["terminal_count"])
        for child in node["choice_term_node_ids"]:
            total += _node_count(nodes, child, field="support_count", checked=checked)
        if node["support_count"] != total:
            _offline_violation("state_support_count_mismatch")
        return
    if kind == "writer_text_choice_support_count_term":
        child = node["successor_support_count_node_id"]
        if node["support_count"] != _node_count(
            nodes,
            child,
            field="support_count",
            checked=checked,
        ):
            _offline_violation("choice_support_count_mismatch")
        return
    if kind == "writer_cursor_completion_count":
        total = 0
        for entry in node["state_count_entries"]:
            total += int(entry["cursor_weight"]) * _node_count(
                nodes,
                entry["state_count_node_id"],
                field="completion_count",
                checked=checked,
            )
        if node["completion_count"] != total:
            _offline_violation("cursor_completion_count_mismatch")
        return
    if kind == "writer_state_completion_count":
        total = int(node["terminal_count"])
        for child in node["branch_term_node_ids"]:
            total += _node_count(
                nodes,
                child,
                field="successor_count",
                checked=checked,
            )
        if node["completion_count"] != total:
            _offline_violation("state_completion_count_mismatch")
        return
    if kind == "writer_branch_completion_term":
        child = node["successor_count_node_id"]
        if node["successor_count"] != _node_count(
            nodes,
            child,
            field="completion_count",
            checked=checked,
        ):
            _offline_violation("branch_successor_count_mismatch")
        return
    if kind == "writer_text_choice_count":
        support_child = node["support_count_node_id"]
        completion_child = node["completion_count_node_id"]
        if node["support_count"] != _node_count(
            nodes,
            support_child,
            field="support_count",
            checked=checked,
        ):
            _offline_violation("text_choice_support_count_mismatch")
        if node["completion_count"] != _node_count(
            nodes,
            completion_child,
            field="completion_count",
            checked=checked,
        ):
            _offline_violation("text_choice_completion_count_mismatch")
        return
    if kind == "writer_terminal_choice_count":
        if not isinstance(node["support_count"], int):
            _offline_violation("terminal_choice_support_count_not_int")
        if not isinstance(node["completion_count"], int):
            _offline_violation("terminal_choice_completion_count_not_int")
        return
    _offline_violation("unknown_count_dag_node_kind")


def _require_count_node(
    nodes: Mapping[str, Mapping[str, object]],
    node_id: object,
) -> Mapping[str, object]:
    if not isinstance(node_id, str) or node_id not in nodes:
        _offline_violation("count_dag_child_missing")
    return nodes[node_id]


def _check_support_string_offline(
    *,
    facts: MoleculeFacts,
    support: Mapping[str, object],
    objects: Mapping[str, Mapping[str, object]],
    checked_object_kinds: set[str],
    checked_relations: set[str],
    source_is_initial: bool,
) -> None:
    payload = support["payload"]
    emitted_texts = payload["emitted_texts"]
    if payload["string"] != "".join(emitted_texts):
        _offline_violation("support_string_text_mismatch")
    replay = _require_object(objects, payload["replay_path_ref"])
    if replay["kind"] != "replay_path":
        _offline_violation("replay_path_kind_mismatch")
    if replay["payload"]["emitted_texts"] != emitted_texts:
        _offline_violation("replay_path_text_mismatch")
    text_refs = payload["text_projection_refs"]
    if len(text_refs) != len(emitted_texts):
        _offline_violation("text_projection_count_mismatch")
    for ref, emitted_text in zip(text_refs, emitted_texts, strict=True):
        projection = _require_object(objects, ref)
        if projection["kind"] != "text_projection":
            _offline_violation("text_projection_ref_kind_mismatch")
        if projection["payload"]["emitted_text"] != emitted_text:
            _offline_violation("text_projection_emitted_text_mismatch")
        checked_object_kinds.add("text_projection")
        if emitted_text.startswith("[") or emitted_text.endswith("]"):
            validate_writer_bracket_atom_text_against_facts(
                facts=facts,
                rendered_text=emitted_text,
            )
            checked_relations.add("bracket_atom_text")
    terminal = _require_object(objects, payload["terminal_projection_ref"])
    if terminal["kind"] != "terminal_projection":
        _offline_violation("terminal_projection_ref_kind_mismatch")
    if "digest" not in terminal["payload"]:
        _offline_violation("terminal_projection_digest_missing")
    if source_is_initial and _non_single_cyclic_bonds(facts):
        _check_non_single_ring_closure_text(
            facts=facts,
            support_string=str(payload["string"]),
        )
        checked_relations.add("closure_bond_text")


def _check_non_single_ring_closure_text(
    *,
    facts: MoleculeFacts,
    support_string: str,
) -> None:
    bonds = _non_single_cyclic_bonds(facts)
    marker_counts = {
        "=": sum(1 for bond in bonds if bond.order == BondOrder.DOUBLE),
        "#": sum(1 for bond in bonds if bond.order == BondOrder.TRIPLE),
    }
    for marker, expected_count in marker_counts.items():
        if expected_count and support_string.count(marker) != expected_count:
            _offline_violation("closure_bond_text_marker_count_mismatch")
        if not expected_count and marker in support_string:
            _offline_violation("closure_bond_text_unexpected_marker")
    if not _has_ring_label_pair(support_string):
        _offline_violation("closure_bond_text_ring_label_missing")


def _atom_by_term(facts: MoleculeFacts, atom_term: object) -> AtomFacts:
    for atom in facts.atoms:
        if _term(atom.id) == atom_term:
            return atom
    _offline_violation("local_atom_text_atom_missing")


def _bond_by_term(facts: MoleculeFacts, bond_term: object) -> BondFacts:
    for bond in facts.bonds:
        if _term(bond.id) == bond_term:
            return bond
    _offline_violation("local_closure_bond_missing")


def _bond_order_value(order: BondOrder) -> str:
    if order == BondOrder.DOUBLE:
        return "double"
    if order == BondOrder.TRIPLE:
        return "triple"
    _offline_violation("local_closure_bond_order_unsupported")


def _bond_order_marker(order: BondOrder) -> str:
    if order == BondOrder.SINGLE:
        return ""
    if order == BondOrder.DOUBLE:
        return "="
    if order == BondOrder.TRIPLE:
        return "#"
    _offline_violation("graph_ring_bond_order_unsupported")


def _non_single_cyclic_bonds(facts: MoleculeFacts) -> tuple[BondFacts, ...]:
    return tuple(
        bond
        for bond in facts.bonds
        if bond.order in (BondOrder.DOUBLE, BondOrder.TRIPLE)
        and _bond_is_cyclic(facts, bond)
    )


def _bond_is_cyclic(facts: MoleculeFacts, bond: BondFacts) -> bool:
    adjacency: dict[object, list[object]] = {}
    for item in facts.bonds:
        if item.id == bond.id:
            continue
        adjacency.setdefault(item.a, []).append(item.b)
        adjacency.setdefault(item.b, []).append(item.a)
    pending = [bond.a]
    seen = set()
    while pending:
        atom = pending.pop()
        if atom == bond.b:
            return True
        if atom in seen:
            continue
        seen.add(atom)
        pending.extend(adjacency.get(atom, ()))
    return False


def _has_ring_label_pair(support_string: str) -> bool:
    return any(support_string.count(str(value)) >= 2 for value in range(1, 10))


def _object_by_id(
    artifact: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    return {
        item["object_id"]: item
        for item in artifact["objects"]
        if isinstance(item, Mapping)
    }


def _check_object_kinds_classified(
    objects: Mapping[str, Mapping[str, object]],
) -> None:
    unknown = {
        str(item["kind"])
        for item in objects.values()
        if item["kind"] not in OBJECT_KIND_OFFLINE_COVERAGE
    }
    if unknown:
        _offline_violation("offline_coverage_ledger_missing_object_kind")


def _require_object(
    objects: Mapping[str, Mapping[str, object]],
    object_id: object,
) -> Mapping[str, object]:
    if not isinstance(object_id, str) or object_id not in objects:
        _offline_violation("object_ref_missing")
    return objects[object_id]


def _offline_violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer support artifact offline replay violation: {kind}",
    )


__all__ = (
    "OBJECT_KIND_OFFLINE_COVERAGE",
    "BranchProjectionIdentityVerification",
    "CountDagArithmeticVerification",
    "GraphRingBranchDeltaVerification",
    "LocalBranchSuccessorEvidenceVerification",
    "OfflineObligationClassification",
    "SupportImageCoverageVerification",
    "SupportStringReplayPathVerification",
    "TerminalSupportIdentityVerification",
    "WriterSupportArtifactOfflineReplayResult",
    "classify_residual_stereo_obligations_offline",
    "validate_writer_bracket_atom_text_against_facts",
    "verify_branch_projection_identities_offline",
    "verify_branch_obligations_offline",
    "verify_count_dag_arithmetic",
    "verify_graph_ring_branch_deltas_offline",
    "verify_local_branch_successor_evidence_offline",
    "verify_support_image_coverage_offline",
    "verify_support_string_replay_paths_offline",
    "verify_terminal_support_identities_offline",
    "verify_transition_branch_projection_identity_offline",
    "verify_writer_support_artifact_offline_replay",
)
