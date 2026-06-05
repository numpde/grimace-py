"""Private decoder-state walkers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import grimace._core as _core
from grimace._runtime_states import _BaseDecoderState, _StateTransitions


_U64_MAX = (1 << 64) - 1
_TransitionChooser = Callable[[_StateTransitions], int]
_IndexSampler = Callable[[int], int]
_WeightedIndexSampler = Callable[[tuple[int, ...]], int]


@dataclass(frozen=True, slots=True)
class _TokenWalkResult:
    tokens: tuple[str, ...]
    choice_counts: tuple[int, ...]
    choice_tokens: tuple[str, ...]
    choice_branch_counts: tuple[int, ...]


def _uniform_token_chooser(sample_index: _IndexSampler) -> _TransitionChooser:
    def choose(transitions: _StateTransitions) -> int:
        return sample_index(len(transitions))

    return choose


def _branch_multiplicity_chooser(
    sample_weighted_index: _WeightedIndexSampler,
) -> _TransitionChooser:
    def choose(transitions: _StateTransitions) -> int:
        return sample_weighted_index(
            tuple(transition.branch_count for transition in transitions)
        )

    return choose


def _seeded_uniform_token_chooser(seed: int) -> _TransitionChooser:
    sampler = _splitmix64_sampler(seed)
    return _uniform_token_chooser(sampler.uniform_index)


def _seeded_branch_multiplicity_chooser(seed: int) -> _TransitionChooser:
    sampler = _splitmix64_sampler(seed)
    return _branch_multiplicity_chooser(sampler.weighted_index)


def _splitmix64_sampler(seed: int) -> _core._SplitMix64Sampler:
    if type(seed) is not int or not 0 <= seed <= _U64_MAX:
        raise ValueError("walk seed must be an unsigned 64-bit integer")
    return _core._SplitMix64Sampler(seed)


def _walk_token_transitions(
    initial_state: _BaseDecoderState,
    choose_index: _TransitionChooser,
) -> _TokenWalkResult:
    tokens: list[str] = []
    choice_counts: list[int] = []
    choice_tokens: list[str] = []
    choice_branch_counts: list[int] = []

    state = initial_state
    while not state.is_terminal():
        # Accepted states may still have outgoing transitions in composed
        # runtimes; walking stops on acceptance, not on absence of choices.
        transitions = state._token_state_transitions()
        if not transitions:
            raise RuntimeError(
                "nonterminal decoder state has no token transitions"
            )
        selected_idx = choose_index(transitions)
        if type(selected_idx) is not int:
            raise TypeError("token transition chooser must return an int")
        if not 0 <= selected_idx < len(transitions):
            raise IndexError(
                f"token transition index {selected_idx} is outside "
                f"0..{len(transitions) - 1}"
            )

        choice_counts.append(len(transitions))
        choice_tokens.extend(transition.text for transition in transitions)
        choice_branch_counts.extend(
            transition.branch_count for transition in transitions
        )

        selected = transitions[selected_idx]
        tokens.append(selected.text)
        state = selected.state_factory()

    return _TokenWalkResult(
        tokens=tuple(tokens),
        choice_counts=tuple(choice_counts),
        choice_tokens=tuple(choice_tokens),
        choice_branch_counts=tuple(choice_branch_counts),
    )
