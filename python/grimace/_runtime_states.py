"""Decoder state adapters for the public runtime."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol, TypeAlias

import grimace._core as _core


DecoderCacheKey: TypeAlias = tuple[object, ...]
_StateTransitionFactory: TypeAlias = Callable[[], "_BaseDecoderState"]


@dataclass(frozen=True, slots=True)
class _StateTransition:
    text: str
    branch_count: int
    state_factory: _StateTransitionFactory

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("decoder state transition text must be a string")
        if type(self.branch_count) is not int or self.branch_count <= 0:
            raise ValueError("decoder state transition branch_count must be positive")


_StateTransitions: TypeAlias = tuple[_StateTransition, ...]


class _BaseDecoderState(Protocol):
    def prefix(self) -> str: ...
    def is_terminal(self) -> bool: ...
    def copy(self) -> "_BaseDecoderState": ...
    def cache_key(self) -> DecoderCacheKey: ...
    def _branch_state_transitions(self) -> _StateTransitions: ...
    def _token_state_transitions(self) -> _StateTransitions: ...


def _realize_state_transitions(
    transitions: _StateTransitions,
) -> tuple[tuple[str, _BaseDecoderState], ...]:
    return tuple(
        (transition.text, transition.state_factory())
        for transition in transitions
    )


def _branch_transition(
    text: str,
    state_factory: _StateTransitionFactory,
) -> _StateTransition:
    return _StateTransition(text, 1, state_factory)


def _counts_by_text(texts: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for text in texts:
        counts[text] = counts.get(text, 0) + 1
    return counts


def _token_branch_counts(decoder: object) -> tuple[tuple[str, int], ...]:
    branch_counts = _counts_by_text(decoder.next_choice_texts())
    token_texts = tuple(decoder.next_token_support())
    token_text_set = set(token_texts)
    if len(token_text_set) != len(token_texts):
        raise RuntimeError("decoder token support must contain unique texts")
    if token_text_set != set(branch_counts):
        raise RuntimeError("decoder token support and branch choices disagree")
    return tuple((text, branch_counts[text]) for text in token_texts)


def _advance_choice_state(decoder: object, chosen_idx: int) -> "_CoreStateAdapter":
    next_decoder = decoder.copy()
    next_decoder.advance_choice(chosen_idx)
    return _CoreStateAdapter(next_decoder)


def _advance_token_state(decoder: object, chosen_token: str) -> "_CoreStateAdapter":
    next_decoder = decoder.copy()
    next_decoder.advance_token(chosen_token)
    return _CoreStateAdapter(next_decoder)


class _CoreStateAdapter:
    __slots__ = ("_decoder",)

    def __init__(self, decoder: object) -> None:
        self._decoder = decoder

    def _branch_state_transitions(self) -> _StateTransitions:
        decoder = self._decoder
        return tuple(
            _branch_transition(
                text,
                lambda chosen_idx=chosen_idx: _advance_choice_state(
                    decoder,
                    chosen_idx,
                ),
            )
            for chosen_idx, text in enumerate(decoder.next_choice_texts())
        )

    def prefix(self) -> str:
        return self._decoder.prefix()

    def is_terminal(self) -> bool:
        return bool(self._decoder.is_terminal())

    def copy(self) -> "_CoreStateAdapter":
        return type(self)(self._decoder.copy())

    def cache_key(self) -> DecoderCacheKey:
        return ("core", self._decoder.cache_key())

    def _token_state_transitions(self) -> _StateTransitions:
        decoder = self._decoder
        return tuple(
            _StateTransition(
                text,
                branch_count,
                lambda text=text: _advance_token_state(decoder, text),
            )
            for text, branch_count in _token_branch_counts(decoder)
        )


class _LazyAllRootsConnectedStereoState:
    __slots__ = ("_prepared", "_atom_count")

    def __init__(
        self,
        prepared: object,
        atom_count: int,
    ) -> None:
        if atom_count <= 0:
            raise ValueError("Lazy all-roots stereo state requires at least one root")
        self._prepared = prepared
        self._atom_count = atom_count

    def _root_decoder(self, root_idx: int) -> object:
        return _core.RootedConnectedStereoDecoder(self._prepared, root_idx)

    def _branch_state_transitions(self) -> _StateTransitions:
        transitions: list[_StateTransition] = []
        for root_idx in range(self._atom_count):
            decoder = self._root_decoder(root_idx)
            for chosen_idx, text in enumerate(decoder.next_choice_texts()):
                transitions.append(
                    _branch_transition(
                        text,
                        lambda root_idx=root_idx, chosen_idx=chosen_idx: (
                            _advance_choice_state(
                                self._root_decoder(root_idx),
                                chosen_idx,
                            )
                        ),
                    )
                )
        return tuple(transitions)

    def _token_state_transitions(self) -> _StateTransitions:
        counts: dict[str, int] = {}
        root_indices_by_text: dict[str, list[int]] = {}
        for root_idx in range(self._atom_count):
            decoder = self._root_decoder(root_idx)
            for text, branch_count in _token_branch_counts(decoder):
                counts[text] = counts.get(text, 0) + branch_count
                root_indices_by_text.setdefault(text, []).append(root_idx)

        return tuple(
            _StateTransition(
                text,
                counts[text],
                lambda text=text, root_indices=tuple(root_indices): (
                    _merge_state_adapters(
                        tuple(
                            _advance_token_state(self._root_decoder(root_idx), text)
                            for root_idx in root_indices
                        )
                    )
                ),
            )
            for text, root_indices in root_indices_by_text.items()
        )

    def prefix(self) -> str:
        return ""

    def is_terminal(self) -> bool:
        return False

    def copy(self) -> "_LazyAllRootsConnectedStereoState":
        return type(self)(self._prepared, self._atom_count)

    def cache_key(self) -> DecoderCacheKey:
        return (
            "lazy_all_roots_connected_stereo",
            self._prepared.policy_digest,
            self._prepared.identity_smiles,
            self._atom_count,
        )


class _MergedStateAdapter:
    __slots__ = ("_states",)

    def __init__(self, states: tuple[_BaseDecoderState, ...]) -> None:
        if not states:
            raise ValueError("Merged decoder state requires at least one branch")
        self._states = states

    def _branch_state_transitions(self) -> _StateTransitions:
        transitions: list[_StateTransition] = []
        for state in self._states:
            if state.is_terminal():
                continue
            transitions.extend(state._branch_state_transitions())
        return tuple(transitions)

    def prefix(self) -> str:
        prefix = self._states[0].prefix()
        for state in self._states[1:]:
            if state.prefix() != prefix:
                raise RuntimeError("Merged decoder states diverged on prefix")
        return prefix

    def is_terminal(self) -> bool:
        return any(state.is_terminal() for state in self._states)

    def copy(self) -> "_MergedStateAdapter":
        return type(self)(tuple(state.copy() for state in self._states))

    def cache_key(self) -> DecoderCacheKey:
        return (
            "merged",
            tuple(sorted((_state_cache_key(state) for state in self._states), key=repr)),
        )

    def _token_state_transitions(self) -> _StateTransitions:
        counts: dict[str, int] = {}
        factories_by_text: dict[str, list[_StateTransitionFactory]] = {}
        for state in self._states:
            if state.is_terminal():
                continue
            for transition in state._token_state_transitions():
                counts[transition.text] = (
                    counts.get(transition.text, 0) + transition.branch_count
                )
                factories_by_text.setdefault(transition.text, []).append(
                    transition.state_factory
                )
        return tuple(
            _StateTransition(
                text,
                counts[text],
                lambda factories=tuple(factories): _merge_state_adapters(
                    tuple(factory() for factory in factories)
                ),
            )
            for text, factories in factories_by_text.items()
        )


class _DisconnectedStateAdapter:
    __slots__ = ("_fragment_states", "_fragment_idx", "_completed_prefix")

    def __init__(
        self,
        fragment_states: tuple[_BaseDecoderState, ...],
        *,
        fragment_idx: int = 0,
        completed_prefix: str = "",
    ) -> None:
        if not fragment_states:
            raise ValueError("Disconnected decoder requires at least one fragment state")
        self._fragment_states = fragment_states
        self._fragment_idx = fragment_idx
        self._completed_prefix = completed_prefix

    def _active_state(self) -> _BaseDecoderState:
        return self._fragment_states[self._fragment_idx]

    def _with_active_state(
        self,
        state: _BaseDecoderState,
    ) -> "_DisconnectedStateAdapter":
        return type(self)(
            self._fragment_states[: self._fragment_idx]
            + (state,)
            + self._fragment_states[self._fragment_idx + 1 :],
            fragment_idx=self._fragment_idx,
            completed_prefix=self._completed_prefix,
        )

    def _advance_fragment(
        self,
        active: _BaseDecoderState,
    ) -> "_DisconnectedStateAdapter":
        return type(self)(
            self._fragment_states,
            fragment_idx=self._fragment_idx + 1,
            completed_prefix=f"{self._completed_prefix}{active.prefix()}.",
        )

    def _wrap_active_transitions(
        self,
        transitions: _StateTransitions,
    ) -> _StateTransitions:
        return tuple(
            _StateTransition(
                transition.text,
                transition.branch_count,
                lambda factory=transition.state_factory: self._with_active_state(
                    factory()
                ),
            )
            for transition in transitions
        )

    def _fragment_separator_transition(
        self,
        active: _BaseDecoderState,
    ) -> _StateTransitions:
        if self._fragment_idx + 1 == len(self._fragment_states):
            return ()
        return (_branch_transition(".", lambda: self._advance_fragment(active)),)

    def _active_state_transitions(
        self,
        transitions: _StateTransitions,
        active: _BaseDecoderState,
    ) -> _StateTransitions:
        wrapped = self._wrap_active_transitions(transitions)
        if active.is_terminal():
            return wrapped + self._fragment_separator_transition(active)
        return wrapped

    def _branch_state_transitions(self) -> _StateTransitions:
        active = self._active_state()
        return self._active_state_transitions(
            active._branch_state_transitions(),
            active,
        )

    def _token_state_transitions(self) -> _StateTransitions:
        active = self._active_state()
        return self._active_state_transitions(
            active._token_state_transitions(),
            active,
        )

    def prefix(self) -> str:
        return f"{self._completed_prefix}{self._active_state().prefix()}"

    def is_terminal(self) -> bool:
        active = self._active_state()
        return (
            active.is_terminal()
            and self._fragment_idx + 1 == len(self._fragment_states)
        )

    def copy(self) -> "_DisconnectedStateAdapter":
        return type(self)(
            tuple(state.copy() for state in self._fragment_states),
            fragment_idx=self._fragment_idx,
            completed_prefix=self._completed_prefix,
        )

    def cache_key(self) -> DecoderCacheKey:
        return (
            "disconnected",
            self._fragment_idx,
            self._completed_prefix,
            tuple(_state_cache_key(state) for state in self._fragment_states),
        )


def _merge_state_adapters(
    states: tuple[_BaseDecoderState, ...],
) -> _BaseDecoderState:
    if not states:
        raise ValueError("Cannot merge an empty decoder state set")
    flattened: list[_BaseDecoderState] = []
    for state in states:
        if isinstance(state, _MergedStateAdapter):
            flattened.extend(state._states)
        else:
            flattened.append(state)
    if len(flattened) == 1:
        return flattened[0]
    return _MergedStateAdapter(tuple(flattened))


def _state_cache_key(state: _BaseDecoderState) -> DecoderCacheKey:
    key = state.cache_key()
    if not isinstance(key, tuple):
        raise TypeError("decoder state cache_key() must return a tuple")
    try:
        hash(key)
    except TypeError as exc:
        raise TypeError("decoder state cache_key() must be hashable") from exc
    return key
