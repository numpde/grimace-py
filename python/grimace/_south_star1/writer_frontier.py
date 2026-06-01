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
from .writer_state import writer_state_from_key
from .writer_state import writer_state_key
from .writer_state import writer_state_key_sort_tuple
from .writer_stereo import empty_writer_stereo_state
from .writer_transitions import finalize_writer_terminal_state
from .writer_transitions import legal_writer_transitions
from .writer_transitions import validate_writer_supported_prepared
from .writer_transitions import validate_writer_transition_graph_surface

if TYPE_CHECKING:
    from .prepared_runtime import SouthStarPreparedMol
    from .prepared_runtime import SouthStarRuntimeOptions


@dataclass(frozen=True, slots=True)
class WriterFrontierState:
    states: frozenset[WriterStateKey]


@dataclass(frozen=True, slots=True)
class WriterFrontierCursor:
    weighted_states: tuple[tuple[WriterStateKey, int], ...]

    def __post_init__(self) -> None:
        merged: Counter[WriterStateKey] = Counter()
        for key, weight in self.weighted_states:
            if weight < 0:
                raise ValueError("writer frontier cursor weights must be nonnegative")
            if weight:
                merged[key] += weight
        object.__setattr__(
            self,
            "weighted_states",
            tuple(
                sorted(
                    merged.items(),
                    key=lambda item: writer_state_key_sort_tuple(item[0]),
                )
            ),
        )

    @property
    def support_state(self) -> WriterFrontierState:
        return WriterFrontierState(
            states=frozenset(key for key, _ in self.weighted_states)
        )


@dataclass(frozen=True, slots=True)
class WriterFrontierTerminal:
    support_count: int
    completion_count: int
    multiplicity: int
    finalized_cursor: WriterFrontierCursor


@dataclass(frozen=True, slots=True)
class WriterFrontierChoice:
    emitted_text: str
    successor: WriterFrontierCursor
    immediate_multiplicity: int
    support_count: int | None = None
    completion_count: int | None = None


@dataclass(frozen=True, slots=True)
class WriterFrontierChoices:
    terminal: WriterFrontierTerminal | None
    choices: tuple[WriterFrontierChoice, ...]


@dataclass(frozen=True, slots=True)
class _GroupedWriterFrontierTransitions:
    terminal_by_key: Counter[WriterStateKey]
    grouped_by_text: dict[str, set[WriterStateKey]]
    weighted_by_text: dict[str, Counter[WriterStateKey]]


def initial_writer_frontier_cursor(
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions,
) -> WriterFrontierCursor:
    return _initial_writer_frontier_cursor(
        prepared,
        runtime_options,
        validate_prepared=validate_writer_supported_prepared,
    )


def initial_writer_transition_frontier_cursor(
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions,
) -> WriterFrontierCursor:
    return _initial_writer_frontier_cursor(
        prepared,
        runtime_options,
        validate_prepared=validate_writer_transition_graph_surface,
    )


def _initial_writer_frontier_cursor(
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions,
    *,
    validate_prepared,
) -> WriterFrontierCursor:
    from .prepared_runtime import require_writer_shaped_runtime_options
    from .prepared_runtime import runtime_root_atom_for_prepared

    require_writer_shaped_runtime_options(runtime_options)
    runtime_root_atom_for_prepared(runtime_options, prepared=prepared)
    validate_prepared(prepared)
    root_domains = _root_domains_for_runtime(prepared, runtime_options)
    weighted_states = []
    for roots in product(*(atoms for _, atoms in root_domains)):
        root_tuple = tuple(roots)
        if not root_tuple:
            continue
        weighted_states.append(
            (
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
                        stereo_state=empty_writer_stereo_state(),
                        policy_state=WriterPolicyState(),
                    )
                ),
                1,
            )
        )
    return WriterFrontierCursor(weighted_states=tuple(weighted_states))


def _cursor_from_support_state(frontier: WriterFrontierState) -> WriterFrontierCursor:
    return WriterFrontierCursor(
        weighted_states=tuple((key, 1) for key in frontier.states)
    )


def writer_frontier_choices(
    prepared: SouthStarPreparedMol,
    cursor: WriterFrontierCursor,
) -> WriterFrontierChoices:
    grouped = _group_writer_frontier_transitions(prepared, cursor)
    support_memo: dict[WriterFrontierState, int] = {}
    completion_memo: dict[WriterStateKey, int] = {}
    choices = []
    for text in sorted(grouped.grouped_by_text):
        successor = WriterFrontierCursor(
            weighted_states=tuple(grouped.weighted_by_text[text].items())
        )
        weighted_successors = grouped.weighted_by_text[text]
        support_count = _count_writer_frontier_support(
            prepared,
            successor.support_state,
            support_memo,
        )
        completion_count = _count_weighted_successor_completions(
            prepared,
            weighted_successors,
            completion_memo,
        )
        if support_count == 0 and completion_count == 0:
            continue
        choices.append(
            WriterFrontierChoice(
                emitted_text=text,
                successor=successor,
                immediate_multiplicity=sum(weighted_successors.values()),
                support_count=support_count,
                completion_count=completion_count,
            )
        )
    terminal = None
    if grouped.terminal_by_key:
        finalized_cursor = WriterFrontierCursor(
            weighted_states=tuple(grouped.terminal_by_key.items())
        )
        terminal_weight = sum(grouped.terminal_by_key.values())
        terminal = WriterFrontierTerminal(
            support_count=1,
            completion_count=terminal_weight,
            multiplicity=terminal_weight,
            finalized_cursor=finalized_cursor,
        )
    return WriterFrontierChoices(
        terminal=terminal,
        choices=tuple(choices),
    )


def _writer_frontier_raw_successors_for_streaming(
    prepared: SouthStarPreparedMol,
    cursor: WriterFrontierCursor,
) -> tuple[tuple[str, WriterFrontierCursor], ...]:
    grouped = _group_writer_frontier_transitions(prepared, cursor)
    return _successors_from_grouped(grouped)


def _successors_from_grouped(
    grouped: _GroupedWriterFrontierTransitions,
) -> tuple[tuple[str, WriterFrontierCursor], ...]:
    return tuple(
        (
            text,
            WriterFrontierCursor(
                weighted_states=tuple(grouped.weighted_by_text[text].items())
            ),
        )
        for text in sorted(grouped.grouped_by_text)
    )


def _group_writer_frontier_transitions(
    prepared: SouthStarPreparedMol,
    cursor: WriterFrontierCursor,
) -> _GroupedWriterFrontierTransitions:
    grouped: dict[str, set[WriterStateKey]] = {}
    weighted: dict[str, Counter[WriterStateKey]] = {}
    terminal_by_key: Counter[WriterStateKey] = Counter()
    for key, parent_weight in cursor.weighted_states:
        state = writer_state_from_key(key)
        finalized = finalize_writer_terminal_state(prepared, state)
        if finalized is not None:
            terminal_by_key[writer_state_key(finalized)] += parent_weight
        for transition in legal_writer_transitions(prepared, state):
            successor_key = writer_state_key(transition.successor)
            grouped.setdefault(transition.emitted_text, set()).add(successor_key)
            weighted.setdefault(
                transition.emitted_text,
                Counter(),
            )[successor_key] += parent_weight
    return _GroupedWriterFrontierTransitions(
        terminal_by_key=terminal_by_key,
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
    grouped = _group_writer_frontier_transitions(
        prepared,
        _cursor_from_support_state(frontier),
    )
    total = 1 if grouped.terminal_by_key else 0
    for text, successor_keys in grouped.grouped_by_text.items():
        successor = WriterFrontierState(states=frozenset(successor_keys))
        total += _count_writer_frontier_support(prepared, successor, memo)
    memo[frontier] = total
    return total


def count_writer_cursor_completions(
    prepared: SouthStarPreparedMol,
    cursor: WriterFrontierCursor,
) -> int:
    memo: dict[WriterStateKey, int] = {}

    return sum(
        weight * _count_writer_state_completions(prepared, key, memo)
        for key, weight in cursor.weighted_states
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
    total = 1 if finalize_writer_terminal_state(prepared, state) is not None else 0
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
    cursor: WriterFrontierCursor,
) -> Iterator[str]:
    def rec(current: WriterFrontierCursor, prefix: str) -> Iterator[str]:
        grouped = _group_writer_frontier_transitions(prepared, current)
        if grouped.terminal_by_key:
            yield prefix
        for text, successor in _successors_from_grouped(grouped):
            yield from rec(successor, prefix + text)

    yield from rec(cursor, "")


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
    "WriterFrontierCursor",
    "WriterFrontierState",
    "WriterFrontierTerminal",
    "count_writer_cursor_completions",
    "count_writer_frontier_support",
    "initial_writer_frontier_cursor",
    "initial_writer_transition_frontier_cursor",
    "iter_writer_frontier_support",
    "writer_frontier_choices",
)
