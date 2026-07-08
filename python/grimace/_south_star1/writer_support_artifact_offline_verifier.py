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


OBJECT_KIND_OFFLINE_COVERAGE = {
    "source_snapshot": "identity_checked",
    "count_envelope": "structurally_checked",
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


def verify_writer_support_artifact_offline_replay(
    *,
    facts: MoleculeFacts,
    artifact: Mapping[str, object],
) -> WriterSupportArtifactOfflineReplayResult:
    try:
        objects = _object_by_id(artifact)
        _check_object_kinds_classified(objects)
        checked_object_kinds = {
            "source_snapshot",
            "support_string",
            "replay_path",
            "terminal_projection",
        }
        checked_relations: set[str] = set()
        root = _require_object(objects, artifact["roots"]["support_image_root"])
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
    "WriterSupportArtifactOfflineReplayResult",
    "validate_writer_bracket_atom_text_against_facts",
    "verify_writer_support_artifact_offline_replay",
)
