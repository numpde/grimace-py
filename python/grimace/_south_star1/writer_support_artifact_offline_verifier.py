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
from .writer_envelope_work import WriterEnvelopeWorkBudget


OBJECT_KIND_OFFLINE_COVERAGE = {
    "source_snapshot": "identity_checked",
    "count_envelope": "arithmetic_checked",
    "count_dag": "arithmetic_checked",
    "frontier_product": "structurally_checked",
    "replay_path": "partially_offline_checked",
    "text_projection": "partially_offline_checked",
    "terminal_projection": "identity_shape_checked",
    "terminal_support": "structurally_checked",
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
        checked_object_kinds = {
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
    "CountDagArithmeticVerification",
    "SupportImageCoverageVerification",
    "WriterSupportArtifactOfflineReplayResult",
    "validate_writer_bracket_atom_text_against_facts",
    "verify_count_dag_arithmetic",
    "verify_support_image_coverage_offline",
    "verify_writer_support_artifact_offline_replay",
)
