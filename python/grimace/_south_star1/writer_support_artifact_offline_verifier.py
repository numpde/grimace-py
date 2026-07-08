"""Producer-free offline relation replay for writer support artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .facts import AtomFacts
from .facts import BondFacts
from .facts import BondOrder
from .facts import MoleculeFacts
from .writer_atom_text_lifecycle import bracket_atom_text
from .writer_count_dag_envelope import count_dag_node_by_id
from .writer_count_dag_envelope import validate_writer_count_certificate_dag_envelope
from .writer_envelope_terms import _identity_digest
from .writer_envelope_terms import _term
from .writer_envelope_work import WriterEnvelopeWorkBudget
from .writer_envelope_work import default_writer_envelope_work_budget


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


@dataclass(frozen=True, slots=True)
class WriterSupportArtifactOfflineReplayResult:
    accepted: bool
    checked_object_kinds: tuple[str, ...] = ()
    unchecked_object_kinds: tuple[str, ...] = ()
    checked_relation_families: tuple[str, ...] = ()
    offline_replay_complete: bool = False
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
        checked_object_kinds = {
            "branch_support",
            "count_dag",
            "count_envelope",
            "source_snapshot",
            "support_string",
            "replay_path",
            "support_image",
            "support_image_coverage",
            "terminal_projection",
        }
        checked_relations: set[str] = {
            "count_dag_arithmetic",
            *coverage.relation_families,
            *replay_paths.relation_families,
            "branch_projection_identity",
            "graph_ring_branch_delta",
            "local_branch_successor_evidence",
            "terminal_support_identity",
        }
        support_refs = root["payload"]["support_string_refs"]
        for ref in support_refs:
            support = _require_object(objects, ref)
            _check_support_string_offline(
                facts=facts,
                support=support,
                objects=objects,
                checked_object_kinds=checked_object_kinds,
                checked_relations=checked_relations,
            )
        unchecked = tuple(
            kind
            for kind in _OFFLINE_UNCHECKED_OBJECT_KINDS
            if any(item["kind"] == kind for item in objects.values())
        )
        return WriterSupportArtifactOfflineReplayResult(
            accepted=True,
            checked_object_kinds=tuple(sorted(checked_object_kinds)),
            unchecked_object_kinds=unchecked,
            checked_relation_families=tuple(sorted(checked_relations)),
            offline_replay_complete=False,
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
            if bracket_atom_text(atom) == rendered_text:
                matches.append(atom)
        except SouthStarError:
            continue
    if len(matches) != 1:
        _offline_violation("bracket_atom_text_facts_mismatch")
    return matches[0]


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


def verify_local_branch_successor_evidence_offline(
    *,
    facts: MoleculeFacts,
    artifact: Mapping[str, object],
    objects: Mapping[str, Mapping[str, object]],
    budget: WriterEnvelopeWorkBudget | None = None,
) -> LocalBranchSuccessorEvidenceVerification:
    try:
        budget = default_writer_envelope_work_budget(budget)
        root = _require_object(objects, artifact["roots"]["support_image_root"])
        branch_refs = _branch_support_refs_for_root(root=root, objects=objects)
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
) -> GraphRingBranchDeltaVerification:
    try:
        budget = default_writer_envelope_work_budget(budget)
        root = _require_object(objects, artifact["roots"]["support_image_root"])
        branch_refs = _branch_support_refs_for_root(root=root, objects=objects)
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
        if terminal_support["payload"] != identity_by_digest[digest]:
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
    if event["text"] != _bond_order_marker(bond.order):
        _offline_violation("graph_ring_bond_marker_mismatch")
    if event["text"] and event["text"] not in branch_payload["emitted_text"]:
        _offline_violation("graph_ring_bond_event_text_mismatch")


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
    if bracket_atom_text(atom) != manifest["rendered_text"]:
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
    if (
        _non_single_cyclic_bonds(facts)
        or "=" in payload["string"]
        or "#" in payload["string"]
    ):
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
    "SupportImageCoverageVerification",
    "SupportStringReplayPathVerification",
    "TerminalSupportIdentityVerification",
    "WriterSupportArtifactOfflineReplayResult",
    "validate_writer_bracket_atom_text_against_facts",
    "verify_branch_projection_identities_offline",
    "verify_count_dag_arithmetic",
    "verify_graph_ring_branch_deltas_offline",
    "verify_local_branch_successor_evidence_offline",
    "verify_support_image_coverage_offline",
    "verify_support_string_replay_paths_offline",
    "verify_terminal_support_identities_offline",
    "verify_writer_support_artifact_offline_replay",
)
