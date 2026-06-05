"""Private decoder-state walkers."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

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

    def step_payloads(
        self,
    ) -> Iterator[tuple[str, int, tuple[str, ...], tuple[int, ...]]]:
        starts = self._validated_choice_starts()
        for token, selected_index, choice_count, offset in zip(
            self.tokens,
            self.selected_indices,
            self.choice_counts,
            starts,
            strict=True,
        ):
            stop = offset + choice_count
            yield (
                token,
                selected_index,
                self.choice_tokens[offset:stop],
                self.choice_branch_counts[offset:stop],
            )

    def _validated_choice_starts(self) -> tuple[int, ...]:
        if len(self.tokens) != len(self.selected_indices):
            raise ValueError("walk token count does not match selected-index count")
        if len(self.tokens) != len(self.choice_counts):
            raise ValueError("walk token count does not match choice-count count")
        if len(self.choice_tokens) != len(self.choice_branch_counts):
            raise ValueError("walk choice token and branch-count lengths differ")
        if not all(isinstance(token, str) for token in self.tokens):
            raise TypeError("walk result tokens must be strings")
        if not all(isinstance(token, str) for token in self.choice_tokens):
            raise TypeError("walk result choice tokens must be strings")
        if not all(
            type(branch_count) is int and branch_count > 0
            for branch_count in self.choice_branch_counts
        ):
            raise ValueError(
                "walk result choice branch counts must be positive ints"
            )
        if not all(
            type(choice_count) is int and choice_count > 0
            for choice_count in self.choice_counts
        ):
            raise ValueError("walk result choice counts must be positive ints")
        starts: list[int] = []
        offset = 0
        for token, selected_index, choice_count in zip(
            self.tokens,
            self.selected_indices,
            self.choice_counts,
            strict=True,
        ):
            if (
                type(selected_index) is not int
                or not 0 <= selected_index < choice_count
            ):
                raise ValueError("walk result selected indices must be ints in range")
            if offset + choice_count > len(self.choice_tokens):
                raise ValueError("walk choice counts do not span choice payload")
            choice_tokens = self.choice_tokens[offset : offset + choice_count]
            if len(set(choice_tokens)) != choice_count:
                raise ValueError("walk choice tokens must be unique per step")
            if token != self.choice_tokens[offset + selected_index]:
                raise ValueError("walk selected tokens do not match choice payload")
            starts.append(offset)
            offset += choice_count
        if offset != len(self.choice_tokens):
            raise ValueError("walk choice counts do not span choice payload")

        return tuple(starts)


@dataclass(slots=True)
class _TokenWalkBuilder:
    tokens: list[str] = field(default_factory=list)
    selected_indices: list[int] = field(default_factory=list)
    choice_counts: list[int] = field(default_factory=list)
    choice_tokens: list[str] = field(default_factory=list)
    choice_branch_counts: list[int] = field(default_factory=list)

    def record(
        self,
        *,
        token_transitions: _StateTransitions,
        selected_token_index: int,
    ) -> None:
        selected_token = token_transitions[selected_token_index].text
        self.choice_counts.append(len(token_transitions))
        self.choice_tokens.extend(transition.text for transition in token_transitions)
        self.choice_branch_counts.extend(
            transition.branch_count for transition in token_transitions
        )
        self.tokens.append(selected_token)
        self.selected_indices.append(selected_token_index)

    def result(self) -> _TokenWalkResult:
        return _TokenWalkResult(
            tokens=tuple(self.tokens),
            selected_indices=tuple(self.selected_indices),
            choice_counts=tuple(self.choice_counts),
            choice_tokens=tuple(self.choice_tokens),
            choice_branch_counts=tuple(self.choice_branch_counts),
        )


def _uniform_transition_chooser(sample_index: _IndexSampler) -> _TransitionChooser:
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


def _seeded_uniform_transition_chooser(seed: int) -> _TransitionChooser:
    sampler = _splitmix64_sampler(seed)
    return _uniform_transition_chooser(sampler.uniform_index)


def _seeded_branch_multiplicity_chooser(seed: int) -> _TransitionChooser:
    sampler = _splitmix64_sampler(seed)
    return _branch_multiplicity_chooser(sampler.weighted_index)


def _splitmix64_sampler(seed: int) -> _core._SplitMix64Sampler:
    validated_seed = _validate_walk_seed(seed)
    return _core._SplitMix64Sampler(validated_seed)


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


def _walk_token_transitions(
    initial_state: _BaseDecoderState,
    choose_index: _TransitionChooser,
) -> _TokenWalkResult:
    builder = _TokenWalkBuilder()
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
        builder.record(
            token_transitions=transitions,
            selected_token_index=selected_idx,
        )

        selected = transitions[selected_idx]
        state = selected.state_factory()

    return builder.result()


def _walk_branch_transitions(
    initial_state: _BaseDecoderState,
    choose_index: _TransitionChooser,
) -> _TokenWalkResult:
    builder = _TokenWalkBuilder()
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

        builder.record(
            token_transitions=token_transitions,
            selected_token_index=selected_token_index,
        )
        state = selected_branch.state_factory()

    return builder.result()
