"""Shared navigation and digest primitives for writer artifact tests."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping

from grimace._south_star1.writer_envelope_terms import _digest_terms_bounded
from grimace._south_star1.writer_envelope_terms import _identity_digest
from grimace._south_star1.writer_envelope_work import WriterEnvelopeWorkBudget
from grimace._south_star1.writer_envelope_work import default_writer_envelope_work_budget


def _budget(budget: WriterEnvelopeWorkBudget | None) -> WriterEnvelopeWorkBudget:
    return default_writer_envelope_work_budget(budget)


def artifact_object_by_id(
    artifact: Mapping[str, object],
    object_id: str,
) -> MutableMapping[str, object]:
    objects = artifact.get("objects")
    if not isinstance(objects, list):
        raise AssertionError("artifact objects must be a list")
    matches = []
    for item in objects:
        if not isinstance(item, MutableMapping):
            raise AssertionError("artifact object must be a mutable mapping")
        if item.get("object_id") == object_id:
            matches.append(item)
    if len(matches) != 1:
        raise AssertionError(
            f"expected exactly one artifact object id {object_id!r}, found {len(matches)}"
        )
    return matches[0]


def artifact_objects_by_kind(
    artifact: Mapping[str, object],
    kind: str,
) -> tuple[MutableMapping[str, object], ...]:
    objects = artifact.get("objects")
    if not isinstance(objects, list):
        raise AssertionError("artifact objects must be a list")
    result = []
    for item in objects:
        if not isinstance(item, MutableMapping):
            raise AssertionError("artifact object must be a mutable mapping")
        if item.get("kind") == kind:
            result.append(item)
    return tuple(result)


def unique_artifact_object_by_kind(
    artifact: Mapping[str, object],
    kind: str,
) -> MutableMapping[str, object]:
    matches = artifact_objects_by_kind(artifact, kind)
    if len(matches) != 1:
        raise AssertionError(
            f"expected exactly one artifact object kind {kind!r}, found {len(matches)}"
        )
    return matches[0]


def closed_term_field(term: Mapping[str, object], name: str) -> object:
    fields = term.get("fields")
    if not isinstance(fields, list):
        raise AssertionError("closed term fields must be a list")
    matches = []
    for field in fields:
        if not isinstance(field, list) or len(field) != 2:
            raise AssertionError("closed term field must be a two-element list")
        if field[0] == name:
            matches.append(field[1])
    if len(matches) != 1:
        raise AssertionError(
            f"expected exactly one closed term field {name!r}, found {len(matches)}"
        )
    return matches[0]


def set_closed_term_field(
    term: MutableMapping[str, object], name: str, value: object
) -> None:
    fields = term.get("fields")
    if not isinstance(fields, list):
        raise AssertionError("closed term fields must be a list")
    matches = [field for field in fields if isinstance(field, list) and len(field) == 2 and field[0] == name]
    if len(matches) != 1:
        raise AssertionError(
            f"expected exactly one closed term field {name!r}, found {len(matches)}"
        )
    matches[0][1] = value


def set_nested_closed_term_field(
    term: MutableMapping[str, object], *path: str, value: object
) -> None:
    if not path:
        raise AssertionError("nested closed term path must not be empty")
    current: object = term
    for name in path[:-1]:
        if not isinstance(current, MutableMapping):
            raise AssertionError(f"closed term path {name!r} is not a mapping")
        current = closed_term_field(current, name)
    if not isinstance(current, MutableMapping):
        raise AssertionError("nested closed term target is not a mapping")
    set_closed_term_field(current, path[-1], value)


def find_closed_term(value: object, marker: str) -> MutableMapping[str, object] | None:
    if isinstance(value, MutableMapping):
        if marker in value:
            return value
        for child in value.values():
            found = find_closed_term(child, marker)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_closed_term(child, marker)
            if found is not None:
                return found
    return None


def closed_term_digest(
    term: object,
    *,
    operation: str,
    budget: WriterEnvelopeWorkBudget | None = None,
) -> str:
    return _identity_digest(term, budget=_budget(budget), operation=operation)


def refresh_closed_term_digest_field(
    container: MutableMapping[str, object],
    *,
    term_field: str,
    digest_field: str,
    operation: str,
    budget: WriterEnvelopeWorkBudget | None = None,
) -> str:
    digest = closed_term_digest(container[term_field], operation=operation, budget=budget)
    container[digest_field] = digest
    return digest


def refresh_cursor_digest(
    cursor: MutableMapping[str, object],
    *,
    operation: str,
    budget: WriterEnvelopeWorkBudget | None = None,
) -> str:
    digest = _digest_terms_bounded(
        cursor["terms"],
        budget=_budget(budget),
        operation=operation,
    )
    cursor["digest"] = digest
    return digest


def refresh_kind_manifest_digest(
    value: MutableMapping[str, object],
    *,
    operation: str,
    budget: WriterEnvelopeWorkBudget | None = None,
) -> str:
    digest = _identity_digest(
        {"kind": value["kind"], "manifest": value["manifest"]},
        budget=_budget(budget),
        operation=operation,
    )
    value["digest"] = digest
    return digest


__all__ = (
    "artifact_object_by_id",
    "artifact_objects_by_kind",
    "unique_artifact_object_by_kind",
    "closed_term_field",
    "set_closed_term_field",
    "set_nested_closed_term_field",
    "find_closed_term",
    "closed_term_digest",
    "refresh_closed_term_digest_field",
    "refresh_cursor_digest",
    "refresh_kind_manifest_digest",
)
