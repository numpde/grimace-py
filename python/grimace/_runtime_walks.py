"""Private decoder-state walkers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from grimace._runtime_states import _BaseDecoderState, _StateTransitions


_TransitionChooser = Callable[[_StateTransitions], int]


@dataclass(frozen=True, slots=True)
class _TokenWalkResult:
    tokens: tuple[str, ...]
    choice_counts: tuple[int, ...]
    choice_tokens: tuple[str, ...]
    choice_branch_counts: tuple[int, ...]


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
