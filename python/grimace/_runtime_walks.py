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
    selected_indices: tuple[int, ...]
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


def _seeded_branch_preserving_chooser(seed: int) -> _TransitionChooser:
    sampler = _splitmix64_sampler(seed)
    return _uniform_token_chooser(sampler.uniform_index)


def _splitmix64_sampler(seed: int) -> _core._SplitMix64Sampler:
    return _core._SplitMix64Sampler(_validate_walk_seed(seed))


def _validate_walk_seed(seed: int) -> int:
    if type(seed) is not int or not 0 <= seed <= _U64_MAX:
        raise ValueError("walk seed must be an unsigned 64-bit integer")
    return seed


def _selected_transition_index(
    transitions: _StateTransitions,
    choose_index: _TransitionChooser,
    *,
    label: str,
) -> int:
    if not transitions:
        raise RuntimeError(f"nonterminal decoder state has no {label} transitions")
    selected_idx = choose_index(transitions)
    if type(selected_idx) is not int:
        raise TypeError(f"{label} transition chooser must return an int")
    if not 0 <= selected_idx < len(transitions):
        raise IndexError(
            f"{label} transition index {selected_idx} is outside "
            f"0..{len(transitions) - 1}"
        )
    return selected_idx


def _token_index_by_text(transitions: _StateTransitions) -> dict[str, int]:
    indices: dict[str, int] = {}
    for idx, transition in enumerate(transitions):
        if transition.text in indices:
            raise RuntimeError("token transitions must have unique text")
        indices[transition.text] = idx
    return indices


def _record_token_step(
    *,
    token_transitions: _StateTransitions,
    selected_token_index: int,
    tokens: list[str],
    selected_indices: list[int],
    choice_counts: list[int],
    choice_tokens: list[str],
    choice_branch_counts: list[int],
) -> None:
    selected_token = token_transitions[selected_token_index].text
    choice_counts.append(len(token_transitions))
    choice_tokens.extend(transition.text for transition in token_transitions)
    choice_branch_counts.extend(
        transition.branch_count for transition in token_transitions
    )
    tokens.append(selected_token)
    selected_indices.append(selected_token_index)


def _walk_token_transitions(
    initial_state: _BaseDecoderState,
    choose_index: _TransitionChooser,
) -> _TokenWalkResult:
    tokens: list[str] = []
    selected_indices: list[int] = []
    choice_counts: list[int] = []
    choice_tokens: list[str] = []
    choice_branch_counts: list[int] = []

    state = initial_state
    while not state.is_terminal():
        # Accepted states may still have outgoing transitions in composed
        # runtimes; walking stops on acceptance, not on absence of choices.
        transitions = state._token_state_transitions()
        selected_idx = _selected_transition_index(
            transitions,
            choose_index,
            label="token",
        )
        _record_token_step(
            token_transitions=transitions,
            selected_token_index=selected_idx,
            tokens=tokens,
            selected_indices=selected_indices,
            choice_counts=choice_counts,
            choice_tokens=choice_tokens,
            choice_branch_counts=choice_branch_counts,
        )

        selected = transitions[selected_idx]
        state = selected.state_factory()

    return _TokenWalkResult(
        tokens=tuple(tokens),
        selected_indices=tuple(selected_indices),
        choice_counts=tuple(choice_counts),
        choice_tokens=tuple(choice_tokens),
        choice_branch_counts=tuple(choice_branch_counts),
    )


def _walk_branch_transitions(
    initial_state: _BaseDecoderState,
    choose_index: _TransitionChooser,
) -> _TokenWalkResult:
    tokens: list[str] = []
    selected_indices: list[int] = []
    choice_counts: list[int] = []
    choice_tokens: list[str] = []
    choice_branch_counts: list[int] = []

    state = initial_state
    while not state.is_terminal():
        # Branch-preserving sampling chooses from branch transitions but
        # reports the same token buckets as determinized sampling.
        token_transitions = state._token_state_transitions()
        if not token_transitions:
            raise RuntimeError(
                "nonterminal decoder state has no token transitions"
            )
        token_indices = _token_index_by_text(token_transitions)

        branch_transitions = state._branch_state_transitions()
        selected_branch_idx = _selected_transition_index(
            branch_transitions,
            choose_index,
            label="branch",
        )
        selected_branch = branch_transitions[selected_branch_idx]
        try:
            selected_token_index = token_indices[selected_branch.text]
        except KeyError as exc:
            raise RuntimeError(
                "selected branch transition is missing from token transitions"
            ) from exc

        _record_token_step(
            token_transitions=token_transitions,
            selected_token_index=selected_token_index,
            tokens=tokens,
            selected_indices=selected_indices,
            choice_counts=choice_counts,
            choice_tokens=choice_tokens,
            choice_branch_counts=choice_branch_counts,
        )
        state = selected_branch.state_factory()

    return _TokenWalkResult(
        tokens=tuple(tokens),
        selected_indices=tuple(selected_indices),
        choice_counts=tuple(choice_counts),
        choice_tokens=tuple(choice_tokens),
        choice_branch_counts=tuple(choice_branch_counts),
    )
