"""Determinized frontier over writer-shaped transition states."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .ids import AtomId
from .policy import SerializationLanguageMode
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


@dataclass(frozen=True, slots=True)
class WriterFrontierChoices:
    eos_available: bool
    choices: tuple[WriterFrontierChoice, ...]


def initial_writer_frontier(
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions,
) -> WriterFrontierState:
    if runtime_options.serialization_language is not SerializationLanguageMode.WRITER_SHAPED:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_POLICY,
            "writer frontier requires serialization_language=WRITER_SHAPED",
        )
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
    grouped: dict[str, set[WriterStateKey]] = {}
    raw_counts: dict[str, int] = {}
    eos_available = False
    for key in frontier.states:
        state = writer_state_from_key(key)
        if writer_state_is_eos(prepared, state):
            eos_available = True
        for transition in legal_writer_transitions(prepared, state):
            grouped.setdefault(transition.emitted_text, set()).add(
                writer_state_key(transition.successor)
            )
            raw_counts[transition.emitted_text] = raw_counts.get(transition.emitted_text, 0) + 1

    return WriterFrontierChoices(
        eos_available=eos_available,
        choices=tuple(
            WriterFrontierChoice(
                emitted_text=text,
                successor=WriterFrontierState(states=frozenset(grouped[text])),
                immediate_multiplicity=raw_counts[text],
            )
            for text in sorted(grouped)
        ),
    )


def count_writer_frontier_support(
    prepared: SouthStarPreparedMol,
    frontier: WriterFrontierState,
) -> int:
    memo: dict[WriterFrontierState, int] = {}

    def rec(current: WriterFrontierState) -> int:
        cached = memo.get(current)
        if cached is not None:
            return cached
        choices = writer_frontier_choices(prepared, current)
        total = 1 if choices.eos_available else 0
        for choice in choices.choices:
            total += rec(choice.successor)
        memo[current] = total
        return total

    return rec(frontier)


def count_writer_witness_completions(
    prepared: SouthStarPreparedMol,
    frontier: WriterFrontierState,
) -> int:
    memo: dict[WriterStateKey, int] = {}

    def rec(key: WriterStateKey) -> int:
        cached = memo.get(key)
        if cached is not None:
            return cached
        state = writer_state_from_key(key)
        total = 1 if writer_state_is_eos(prepared, state) else 0
        for transition in legal_writer_transitions(prepared, state):
            total += rec(writer_state_key(transition.successor))
        memo[key] = total
        return total

    return sum(rec(key) for key in frontier.states)


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
        raise ValueError(
            f"rooted atom is not present in prepared molecule: {atom!r}"
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
