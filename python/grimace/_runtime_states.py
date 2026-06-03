"""Decoder state adapters for the public runtime."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import Protocol, TypeAlias, cast

import grimace._core as _core


DecoderCacheKey: TypeAlias = tuple[object, ...]
_StateTransitionFactory: TypeAlias = Callable[[], "_BaseDecoderState"]
_StateTransitions: TypeAlias = tuple[tuple[str, _StateTransitionFactory], ...]


class _BaseDecoderState(Protocol):
    def prefix(self) -> str: ...
    def is_terminal(self) -> bool: ...
    def copy(self) -> "_BaseDecoderState": ...
    def cache_key(self) -> Hashable: ...
    def _choice_state_transitions(self) -> _StateTransitions: ...
    def _grouped_state_transitions(self) -> _StateTransitions: ...


def _realize_state_transitions(
    transitions: _StateTransitions,
) -> tuple[tuple[str, _BaseDecoderState], ...]:
    return tuple(
        (text, state_factory())
        for text, state_factory in transitions
    )


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

    def _choice_state_transitions(self) -> _StateTransitions:
        decoder = self._decoder
        return tuple(
            (
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

    def _grouped_state_transitions(self) -> _StateTransitions:
        decoder = self._decoder
        return tuple(
            (
                text,
                lambda text=text: _advance_token_state(decoder, text),
            )
            for text in decoder.next_token_support()
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

    def _choice_state_transitions(self) -> _StateTransitions:
        transitions: list[tuple[str, _StateTransitionFactory]] = []
        for root_idx in range(self._atom_count):
            decoder = self._root_decoder(root_idx)
            for chosen_idx, text in enumerate(decoder.next_choice_texts()):
                transitions.append(
                    (
                        text,
                        lambda decoder=decoder, chosen_idx=chosen_idx: (
                            _advance_choice_state(decoder, chosen_idx)
                        ),
                    )
                )
        return tuple(transitions)

    def _grouped_state_transitions(self) -> _StateTransitions:
        buckets: dict[str, list[object]] = {}
        for root_idx in range(self._atom_count):
            decoder = self._root_decoder(root_idx)
            for text in decoder.next_token_support():
                buckets.setdefault(text, []).append(decoder)

        return tuple(
            (
                text,
                lambda text=text, decoders=tuple(decoders): _merge_state_adapters(
                    tuple(
                        _advance_token_state(decoder, text)
                        for decoder in decoders
                    )
                ),
            )
            for text, decoders in buckets.items()
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

    def _choice_state_transitions(self) -> _StateTransitions:
        transitions: list[tuple[str, _StateTransitionFactory]] = []
        for state in self._states:
            if state.is_terminal():
                continue
            transitions.extend(state._choice_state_transitions())
        return tuple(transitions)

    def prefix(self) -> str:
        prefix = self._states[0].prefix()
        for state in self._states[1:]:
            if state.prefix() != prefix:
                raise RuntimeError("Merged decoder states diverged on prefix")
        return prefix

    def is_terminal(self) -> bool:
        return all(state.is_terminal() for state in self._states)

    def copy(self) -> "_MergedStateAdapter":
        return type(self)(tuple(state.copy() for state in self._states))

    def cache_key(self) -> DecoderCacheKey:
        return (
            "merged",
            tuple(sorted((_state_cache_key(state) for state in self._states), key=repr)),
        )

    def _grouped_state_transitions(self) -> _StateTransitions:
        grouped: dict[str, list[_StateTransitionFactory]] = {}
        for state in self._states:
            if state.is_terminal():
                continue
            for text, state_factory in state._grouped_state_transitions():
                grouped.setdefault(text, []).append(state_factory)
        return tuple(
            (
                text,
                lambda factories=tuple(factories): _merge_state_adapters(
                    tuple(factory() for factory in factories)
                ),
            )
            for text, factories in grouped.items()
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

    def _active_successor_transitions(
        self,
        transitions: _StateTransitions,
    ) -> _StateTransitions:
        return tuple(
            (
                text,
                lambda factory=factory: self._with_active_state(factory()),
            )
            for text, factory in transitions
        )

    def _fragment_separator_transition(
        self,
        active: _BaseDecoderState,
    ) -> _StateTransitions:
        if self._fragment_idx + 1 == len(self._fragment_states):
            return ()
        return ((".", lambda: self._advance_fragment(active)),)

    def _choice_state_transitions(self) -> _StateTransitions:
        active = self._active_state()
        if not active.is_terminal():
            return self._active_successor_transitions(active._choice_state_transitions())
        return self._fragment_separator_transition(active)

    def _grouped_state_transitions(self) -> _StateTransitions:
        active = self._active_state()
        if not active.is_terminal():
            return self._active_successor_transitions(active._grouped_state_transitions())
        return self._fragment_separator_transition(active)

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
    if isinstance(key, tuple):
        return cast(DecoderCacheKey, key)
    return ("raw", key)
