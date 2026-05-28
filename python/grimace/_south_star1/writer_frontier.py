"""Determinized frontier over writer-shaped transition states."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .ids import AtomId
from .writer_state import ComponentCursor
from .writer_state import ObligationState
from .writer_state import WriterAtomFrame
from .writer_state import WriterPolicyState
from .writer_state import WriterRingState
from .writer_state import WriterState
from .writer_state import WriterStateKey
from .writer_state import WriterStereoState
from .writer_state import writer_state_from_key
from .writer_state import writer_state_key
from .writer_transitions import legal_writer_transitions
from .writer_transitions import validate_writer_supported_prepared
from .writer_transitions import writer_state_is_eos

if TYPE_CHECKING:
    from .prepared_runtime import SouthStarPreparedMol
    from .prepared_runtime import SouthStarRuntimeOptions


@dataclass(frozen=True, slots=True)
class WriterFrontierState:
    states: frozenset[WriterStateKey]


@dataclass(frozen=True, slots=True)
class WriterFrontierChoice:
    emitted_text: str
    successor: WriterFrontierState
    immediate_multiplicity: int
    support_count: int | None = None
    completion_count: int | None = None


@dataclass(frozen=True, slots=True)
class WriterFrontierChoices:
    eos_available: bool
    choices: tuple[WriterFrontierChoice, ...]


@dataclass(frozen=True, slots=True)
class _GroupedWriterFrontierTransitions:
    eos_available: bool
    grouped_by_text: dict[str, set[WriterStateKey]]
    weighted_by_text: dict[str, Counter[WriterStateKey]]


def initial_writer_frontier(
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions,
) -> WriterFrontierState:
    from .prepared_runtime import require_writer_shaped_runtime_options
    from .prepared_runtime import runtime_root_atom_for_prepared

    require_writer_shaped_runtime_options(runtime_options)
    runtime_root_atom_for_prepared(runtime_options, prepared=prepared)
    validate_writer_supported_prepared(prepared)
    root_domains = _root_domains_for_runtime(prepared, runtime_options)
    states = []
    for roots in product(*(atoms for _, atoms in root_domains)):
        root_tuple = tuple(roots)
        if not root_tuple:
            continue
        states.append(
            writer_state_key(
                WriterState(
                    component_cursor=ComponentCursor(
                        component_index=0,
                        component_roots=root_tuple,
                    ),
                    active=WriterAtomFrame(
                        atom=root_tuple[0],
                        parent=None,
                        incoming_bond=None,
                        atom_emitted=False,
                    ),
                    branch_stack=(),
                    visited_atoms=frozenset(),
                    written_bonds=frozenset(),
                    obligations=ObligationState(),
                    ring_state=WriterRingState(),
                    stereo_state=WriterStereoState(),
                    policy_state=WriterPolicyState(),
                )
            )
        )
    return WriterFrontierState(states=frozenset(states))


def writer_frontier_choices(
    prepared: SouthStarPreparedMol,
    frontier: WriterFrontierState,
) -> WriterFrontierChoices:
    grouped = _group_writer_frontier_transitions(prepared, frontier)
    support_memo: dict[WriterFrontierState, int] = {}
    completion_memo: dict[WriterStateKey, int] = {}
    choices = []
    for text in sorted(grouped.grouped_by_text):
        successor = WriterFrontierState(
            states=frozenset(grouped.grouped_by_text[text])
        )
        weighted_successors = grouped.weighted_by_text[text]
        choices.append(
            WriterFrontierChoice(
                emitted_text=text,
                successor=successor,
                immediate_multiplicity=sum(weighted_successors.values()),
                support_count=_count_writer_frontier_support(
                    prepared,
                    successor,
                    support_memo,
                ),
                completion_count=_count_weighted_successor_completions(
                    prepared,
                    weighted_successors,
                    completion_memo,
                ),
            )
        )
    return WriterFrontierChoices(
        eos_available=grouped.eos_available,
        choices=tuple(choices),
    )


def _group_writer_frontier_transitions(
    prepared: SouthStarPreparedMol,
    frontier: WriterFrontierState,
) -> _GroupedWriterFrontierTransitions:
    grouped: dict[str, set[WriterStateKey]] = {}
    weighted: dict[str, Counter[WriterStateKey]] = {}
    eos_available = False
    for key in frontier.states:
        state = writer_state_from_key(key)
        if writer_state_is_eos(prepared, state):
            eos_available = True
        for transition in legal_writer_transitions(prepared, state):
            successor_key = writer_state_key(transition.successor)
            grouped.setdefault(transition.emitted_text, set()).add(successor_key)
            weighted.setdefault(transition.emitted_text, Counter())[successor_key] += 1
    return _GroupedWriterFrontierTransitions(
        eos_available=eos_available,
        grouped_by_text=grouped,
        weighted_by_text=weighted,
    )


def count_writer_frontier_support(
    prepared: SouthStarPreparedMol,
    frontier: WriterFrontierState,
) -> int:
    return _count_writer_frontier_support(prepared, frontier, {})


def _count_writer_frontier_support(
    prepared: SouthStarPreparedMol,
    frontier: WriterFrontierState,
    memo: dict[WriterFrontierState, int],
) -> int:
    cached = memo.get(frontier)
    if cached is not None:
        return cached
    grouped = _group_writer_frontier_transitions(prepared, frontier)
    total = 1 if grouped.eos_available else 0
    for text, successor_keys in grouped.grouped_by_text.items():
        successor = WriterFrontierState(states=frozenset(successor_keys))
        total += _count_writer_frontier_support(prepared, successor, memo)
    memo[frontier] = total
    return total


def count_writer_witness_completions(
    prepared: SouthStarPreparedMol,
    frontier: WriterFrontierState,
) -> int:
    memo: dict[WriterStateKey, int] = {}

    return sum(
        _count_writer_state_completions(prepared, key, memo)
        for key in frontier.states
    )


def _count_weighted_successor_completions(
    prepared: SouthStarPreparedMol,
    weighted_successors: Counter[WriterStateKey],
    memo: dict[WriterStateKey, int],
) -> int:
    return sum(
        multiplicity * _count_writer_state_completions(prepared, key, memo)
        for key, multiplicity in weighted_successors.items()
    )


def _count_writer_state_completions(
    prepared: SouthStarPreparedMol,
    key: WriterStateKey,
    memo: dict[WriterStateKey, int],
) -> int:
    cached = memo.get(key)
    if cached is not None:
        return cached
    state = writer_state_from_key(key)
    total = 1 if writer_state_is_eos(prepared, state) else 0
    for transition in legal_writer_transitions(prepared, state):
        total += _count_writer_state_completions(
            prepared,
            writer_state_key(transition.successor),
            memo,
        )
    memo[key] = total
    return total


def iter_writer_frontier_support(
    prepared: SouthStarPreparedMol,
    frontier: WriterFrontierState,
) -> Iterator[str]:
    def rec(current: WriterFrontierState, prefix: str) -> Iterator[str]:
        choices = writer_frontier_choices(prepared, current)
        if choices.eos_available:
            yield prefix
        for choice in choices.choices:
            yield from rec(choice.successor, prefix + choice.emitted_text)

    yield from rec(frontier, "")


def _root_domains_for_runtime(
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions,
) -> tuple[tuple[object, tuple[AtomId, ...]], ...]:
    if runtime_options.rooted_at_atom < 0:
        return prepared.all_root_domains
    atom = AtomId(runtime_options.rooted_at_atom)
    try:
        return prepared.component_root_domains_by_explicit_root[atom]
    except KeyError as exc:
        raise SouthStarError(
            SouthStarErrorKind.INVALID_FACTS,
            f"rooted_at_atom is not present in prepared molecule: {int(atom)}",
        ) from exc


__all__ = (
    "WriterFrontierChoice",
    "WriterFrontierChoices",
    "WriterFrontierState",
    "count_writer_frontier_support",
    "count_writer_witness_completions",
    "initial_writer_frontier",
    "iter_writer_frontier_support",
    "writer_frontier_choices",
)
