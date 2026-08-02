"""Disjoint fast and slow qualification shards for accepted writer cases."""

from __future__ import annotations

from contextvars import ContextVar

from tests.south_star1.default_writer_capability_ledger import (
    ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES,
)
from tests.south_star1.default_writer_capability_ledger import (
    DefaultWriterCapabilityCase,
)


SLOW_COUPLED_CASE_NAMES = (
    "zero_h_tetrahedral",
    "adjacent_specified_tetrahedral",
    "remote_coupled_tetrahedral_a",
    "remote_coupled_tetrahedral_b",
)

FAST_ACCEPTED_CASES = tuple(
    case
    for case in ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES
    if case.name not in SLOW_COUPLED_CASE_NAMES
)
SLOW_COUPLED_CASES = tuple(
    case
    for case in ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES
    if case.name in SLOW_COUPLED_CASE_NAMES
)

MATERIALIZED_ARTIFACT_QUALIFIED_CASES = tuple(
    case
    for case in ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES
    if case.qualification_authority == "materialized_support_artifact"
)
CONTINUATION_PROOF_QUALIFIED_CASES = tuple(
    case
    for case in ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES
    if case.qualification_authority == "continuation_proof_complete"
)
assert not {
    case.name for case in MATERIALIZED_ARTIFACT_QUALIFIED_CASES
} & {
    case.name for case in CONTINUATION_PROOF_QUALIFIED_CASES
}
assert {
    case.name for case in MATERIALIZED_ARTIFACT_QUALIFIED_CASES
} | {
    case.name for case in CONTINUATION_PROOF_QUALIFIED_CASES
} == {
    case.name for case in ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES
}
assert all(
    case.qualification_authority is not None
    for case in ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES
)

SLOW_QUALIFICATION_SHARDS = {
    "zero-h-adjacent": (
        "zero_h_tetrahedral",
        "adjacent_specified_tetrahedral",
    ),
    "remote-a": (
        "remote_coupled_tetrahedral_a",
    ),
    "remote-b": (
        "remote_coupled_tetrahedral_b",
    ),
}

_SHARD_NAME_SETS = tuple(
    frozenset(names) for names in SLOW_QUALIFICATION_SHARDS.values()
)
assert all(
    left.isdisjoint(right)
    for index, left in enumerate(_SHARD_NAME_SETS)
    for right in _SHARD_NAME_SETS[index + 1 :]
)
assert frozenset().union(*_SHARD_NAME_SETS) == frozenset(SLOW_COUPLED_CASE_NAMES)

_SELECTED_SLOW_CASES: ContextVar[tuple[DefaultWriterCapabilityCase, ...] | None] = (
    ContextVar("south_star1_selected_slow_cases", default=None)
)


def slow_cases_for_shard(name: str) -> tuple[DefaultWriterCapabilityCase, ...]:
    if not name or name not in SLOW_QUALIFICATION_SHARDS:
        raise ValueError(f"unknown slow qualification shard: {name!r}")
    selected_names = SLOW_QUALIFICATION_SHARDS[name]
    return tuple(
        case
        for case in ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES
        if case.name in selected_names
    )


def bind_slow_qualification_shard(
    name: str,
) -> object:
    return _SELECTED_SLOW_CASES.set(slow_cases_for_shard(name))


def reset_slow_qualification_shard(token: object) -> None:
    _SELECTED_SLOW_CASES.reset(token)


def selected_slow_qualification_cases() -> tuple[DefaultWriterCapabilityCase, ...]:
    selected = _SELECTED_SLOW_CASES.get()
    if selected is None:
        raise RuntimeError(
            "slow qualification shard selection is required before loading slow tests"
        )
    return selected
