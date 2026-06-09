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
from .writer_transitions import _WriterActiveEmittedGraphPolicyBlocker
from .writer_transitions import _WriterNextTokenFrontierSupport
from .writer_transitions import _WriterTopLevelScheduleOutcome
from .writer_transitions import _legal_writer_schedule_outcome
from .writer_transitions import _raise_for_top_level_schedule_outcome_blockers
from .writer_transitions import validate_writer_supported_prepared
from .writer_transitions import validate_writer_transition_prepared

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


@dataclass(frozen=True, slots=True)
class _WriterFrontierStateScheduleOutcome:
    state_key: WriterStateKey
    parent_weight: int
    finalized_state_key: WriterStateKey | None
    schedule_outcome: _WriterTopLevelScheduleOutcome

    @property
    def blocked(self) -> bool:
        return bool(self.schedule_outcome.graph_policy_blockers)

    @property
    def graph_policy_blockers(
        self,
    ) -> tuple[_WriterActiveEmittedGraphPolicyBlocker, ...]:
        return self.schedule_outcome.graph_policy_blockers


@dataclass(frozen=True, slots=True)
class _WriterFrontierNextTokenSupport:
    state_key: WriterStateKey
    parent_weight: int
    schedule_support: _WriterNextTokenFrontierSupport
    successor_key: WriterStateKey

    @property
    def emitted_text(self) -> str:
        return self.schedule_support.emitted_text

    @property
    def graph_action_surface(self):
        return self.schedule_support.graph_action_surface

    @property
    def policy_family(self):
        return self.schedule_support.policy_family


@dataclass(frozen=True, slots=True)
class _WriterFrontierNextTokenEntry:
    emitted_text: str
    supports: tuple[_WriterFrontierNextTokenSupport, ...]

    @property
    def successor_keys(self) -> frozenset[WriterStateKey]:
        return frozenset(
            support.successor_key
            for support in self.supports
        )

    @property
    def weighted_successors(self) -> Counter[WriterStateKey]:
        weighted: Counter[WriterStateKey] = Counter()

        for support in self.supports:
            weighted[support.successor_key] += support.parent_weight

        return weighted

    @property
    def immediate_multiplicity(self) -> int:
        return sum(self.weighted_successors.values())

    @property
    def policy_families(self):
        return tuple(
            support.policy_family
            for support in self.supports
        )


@dataclass(frozen=True, slots=True)
class _WriterFrontierScheduleOutcome:
    state_outcomes: tuple[_WriterFrontierStateScheduleOutcome, ...]
    terminal_by_key: Counter[WriterStateKey]
    grouped_by_text: dict[str, set[WriterStateKey]]
    weighted_by_text: dict[str, Counter[WriterStateKey]]
    next_token_frontier: tuple[_WriterFrontierNextTokenEntry, ...] = ()

    @property
    def blocked_state_outcomes(
        self,
    ) -> tuple[_WriterFrontierStateScheduleOutcome, ...]:
        return tuple(
            state_outcome
            for state_outcome in self.state_outcomes
            if state_outcome.blocked
        )

    @property
    def graph_policy_blockers(
        self,
    ) -> tuple[_WriterActiveEmittedGraphPolicyBlocker, ...]:
        return tuple(
            blocker
            for state_outcome in self.blocked_state_outcomes
            for blocker in state_outcome.graph_policy_blockers
        )

    @property
    def blocked(self) -> bool:
        return bool(self.graph_policy_blockers)

    @property
    def grouped_transitions(self) -> _GroupedWriterFrontierTransitions:
        return _GroupedWriterFrontierTransitions(
            terminal_by_key=self.terminal_by_key,
            grouped_by_text=self.grouped_by_text,
            weighted_by_text=self.weighted_by_text,
        )

    @property
    def next_token_supports(
        self,
    ) -> tuple[_WriterFrontierNextTokenSupport, ...]:
        return tuple(
            support
            for entry in self.next_token_frontier
            for support in entry.supports
        )

    @property
    def grouped_by_text_from_next_token_frontier(
        self,
    ) -> dict[str, set[WriterStateKey]]:
        return {
            entry.emitted_text: set(entry.successor_keys)
            for entry in self.next_token_frontier
        }

    @property
    def weighted_by_text_from_next_token_frontier(
        self,
    ) -> dict[str, Counter[WriterStateKey]]:
        return {
            entry.emitted_text: entry.weighted_successors
            for entry in self.next_token_frontier
        }


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
        validate_prepared=validate_writer_transition_prepared,
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


def _writer_frontier_next_token_entries_from_supports(
    supports: tuple[_WriterFrontierNextTokenSupport, ...],
) -> tuple[_WriterFrontierNextTokenEntry, ...]:
    grouped: dict[str, list[_WriterFrontierNextTokenSupport]] = {}
    order: list[str] = []

    for support in supports:
        emitted_text = support.emitted_text

        if emitted_text not in grouped:
            grouped[emitted_text] = []
            order.append(emitted_text)

        grouped[emitted_text].append(support)

    return tuple(
        _WriterFrontierNextTokenEntry(
            emitted_text=emitted_text,
            supports=tuple(grouped[emitted_text]),
        )
        for emitted_text in order
    )


def _writer_frontier_schedule_outcome(
    prepared: SouthStarPreparedMol,
    cursor: WriterFrontierCursor,
    *,
    stop_after_first_blocked: bool = False,
) -> _WriterFrontierScheduleOutcome:
    grouped: dict[str, set[WriterStateKey]] = {}
    weighted: dict[str, Counter[WriterStateKey]] = {}
    terminal_by_key: Counter[WriterStateKey] = Counter()
    state_outcomes: list[_WriterFrontierStateScheduleOutcome] = []
    frontier_supports: list[_WriterFrontierNextTokenSupport] = []

    for key, parent_weight in cursor.weighted_states:
        state = writer_state_from_key(key)

        finalized = finalize_writer_terminal_state(prepared, state)
        finalized_key = None

        if finalized is not None:
            finalized_key = writer_state_key(finalized)
            terminal_by_key[finalized_key] += parent_weight

        schedule_outcome = _legal_writer_schedule_outcome(prepared, state)

        state_outcome = _WriterFrontierStateScheduleOutcome(
            state_key=key,
            parent_weight=parent_weight,
            finalized_state_key=finalized_key,
            schedule_outcome=schedule_outcome,
        )
        state_outcomes.append(state_outcome)

        if state_outcome.blocked:
            if stop_after_first_blocked:
                break

            continue

        for entry in schedule_outcome.selected_next_token_frontier:
            for support in entry.supports:
                successor_key = writer_state_key(support.transition.successor)

                frontier_support = _WriterFrontierNextTokenSupport(
                    state_key=key,
                    parent_weight=parent_weight,
                    schedule_support=support,
                    successor_key=successor_key,
                )
                frontier_supports.append(frontier_support)

                grouped.setdefault(entry.emitted_text, set()).add(successor_key)
                weighted.setdefault(
                    entry.emitted_text,
                    Counter(),
                )[successor_key] += parent_weight

    return _WriterFrontierScheduleOutcome(
        state_outcomes=tuple(state_outcomes),
        terminal_by_key=terminal_by_key,
        grouped_by_text=grouped,
        weighted_by_text=weighted,
        next_token_frontier=(
            _writer_frontier_next_token_entries_from_supports(
                tuple(frontier_supports)
            )
        ),
    )


def _raise_for_writer_frontier_schedule_outcome_blockers(
    outcome: _WriterFrontierScheduleOutcome,
) -> None:
    for state_outcome in outcome.blocked_state_outcomes:
        _raise_for_top_level_schedule_outcome_blockers(
            state_outcome.schedule_outcome
        )


def _group_writer_frontier_transitions(
    prepared: SouthStarPreparedMol,
    cursor: WriterFrontierCursor,
) -> _GroupedWriterFrontierTransitions:
    outcome = _writer_frontier_schedule_outcome(
        prepared,
        cursor,
        stop_after_first_blocked=True,
    )

    _raise_for_writer_frontier_schedule_outcome_blockers(outcome)

    return outcome.grouped_transitions


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
    outcome = _legal_writer_schedule_outcome(prepared, state)
    _raise_for_top_level_schedule_outcome_blockers(outcome)

    for entry in outcome.selected_next_token_frontier:
        for support in entry.supports:
            total += _count_writer_state_completions(
                prepared,
                writer_state_key(support.transition.successor),
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
