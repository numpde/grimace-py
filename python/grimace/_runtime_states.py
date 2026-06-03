"""Decoder state adapters for the public runtime."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from typing import Protocol, TypeAlias, cast


DecoderCacheKey: TypeAlias = tuple[object, ...]
_StateFactory: TypeAlias = Callable[[], "_BaseDecoderState"]
_StateEntries: TypeAlias = tuple[tuple[str, _StateFactory], ...]


class _BaseDecoderState(Protocol):
    def prefix(self) -> str: ...
    def is_terminal(self) -> bool: ...
    def copy(self) -> "_BaseDecoderState": ...
    def cache_key(self) -> Hashable: ...
    def choice_successor_states(
        self,
    ) -> tuple[tuple[str, "_BaseDecoderState"], ...]: ...
    def grouped_successor_states(
        self,
    ) -> tuple[tuple[str, "_BaseDecoderState"], ...]: ...


def _state_factory(state: _BaseDecoderState) -> _StateFactory:
    return lambda: state


def _eager_state_entries(
    successors: tuple[tuple[str, _BaseDecoderState], ...],
) -> _StateEntries:
    return tuple((text, _state_factory(successor)) for text, successor in successors)


class _CoreStateAdapter:
    __slots__ = ("_decoder",)

    def __init__(self, decoder: object) -> None:
        self._decoder = decoder

    @staticmethod
    def _successor_states(
        successors: Sequence[tuple[str, object]],
    ) -> tuple[tuple[str, _BaseDecoderState], ...]:
        if not successors:
            return ()
        successor_states: list[tuple[str, _BaseDecoderState]] = []
        for text, next_decoder in successors:
            next_state = _CoreStateAdapter.__new__(_CoreStateAdapter)
            next_state._decoder = next_decoder
            successor_states.append((text, next_state))
        return tuple(successor_states)

    def choice_successor_states(self) -> tuple[tuple[str, _BaseDecoderState], ...]:
        return self._successor_states(self._decoder.choice_successors())

    def prefix(self) -> str:
        return self._decoder.prefix()

    def is_terminal(self) -> bool:
        return bool(self._decoder.is_terminal())

    def copy(self) -> "_CoreStateAdapter":
        return type(self)(self._decoder.copy())

    def cache_key(self) -> DecoderCacheKey:
        return ("core", self._decoder.cache_key())

    def grouped_successor_states(self) -> tuple[tuple[str, _BaseDecoderState], ...]:
        return self._successor_states(self._decoder.grouped_successors())


class _MergedStateAdapter:
    __slots__ = ("_states",)

    def __init__(self, states: tuple[_BaseDecoderState, ...]) -> None:
        if not states:
            raise ValueError("Merged decoder state requires at least one branch")
        self._states = states

    def choice_successor_states(self) -> tuple[tuple[str, _BaseDecoderState], ...]:
        successor_states: list[tuple[str, _BaseDecoderState]] = []
        for state in self._states:
            if state.is_terminal():
                continue
            successor_states.extend(_choice_successor_states(state))
        return tuple(successor_states)

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

    def grouped_successor_states(self) -> tuple[tuple[str, _BaseDecoderState], ...]:
        grouped: dict[str, list[_BaseDecoderState]] = {}
        for state in self._states:
            for text, successor in _grouped_successor_states(state):
                grouped.setdefault(text, []).append(successor)
        return tuple(
            (text, _merge_choice_successor_states(tuple(successors)))
            for text, successors in grouped.items()
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

    def _active_successor_states(
        self,
        successors: tuple[tuple[str, _BaseDecoderState], ...],
    ) -> tuple[tuple[str, _BaseDecoderState], ...]:
        return tuple(
            (text, self._with_active_state(successor))
            for text, successor in successors
        )

    def _fragment_separator_successor(
        self,
        active: _BaseDecoderState,
    ) -> tuple[tuple[str, _BaseDecoderState], ...]:
        if self._fragment_idx + 1 == len(self._fragment_states):
            return ()
        return ((".", self._advance_fragment(active)),)

    def choice_successor_states(self) -> tuple[tuple[str, _BaseDecoderState], ...]:
        active = self._active_state()
        if not active.is_terminal():
            return self._active_successor_states(_choice_successor_states(active))
        return self._fragment_separator_successor(active)

    def _active_successor_entries(
        self,
        entries: _StateEntries,
    ) -> _StateEntries:
        return tuple(
            (
                text,
                lambda factory=factory: self._with_active_state(factory()),
            )
            for text, factory in entries
        )

    def _fragment_separator_entry(
        self,
        active: _BaseDecoderState,
    ) -> _StateEntries:
        if self._fragment_idx + 1 == len(self._fragment_states):
            return ()
        return ((".", lambda: self._advance_fragment(active)),)

    def _choice_state_entries(self) -> _StateEntries:
        active = self._active_state()
        if not active.is_terminal():
            return self._active_successor_entries(_choice_state_entries(active))
        return self._fragment_separator_entry(active)

    def grouped_successor_states(self) -> tuple[tuple[str, _BaseDecoderState], ...]:
        active = self._active_state()
        if not active.is_terminal():
            return self._active_successor_states(
                _grouped_successor_states(active)
            )
        return self._fragment_separator_successor(active)

    def _grouped_state_entries(self) -> _StateEntries:
        active = self._active_state()
        if not active.is_terminal():
            return self._active_successor_entries(_grouped_state_entries(active))
        return self._fragment_separator_entry(active)

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


def _merge_choice_successor_states(
    states: tuple[_BaseDecoderState, ...],
) -> _BaseDecoderState:
    flattened: list[_BaseDecoderState] = []
    for state in states:
        if isinstance(state, _MergedStateAdapter):
            flattened.extend(state._states)
        else:
            flattened.append(state)
    if len(flattened) == 1:
        return flattened[0]
    return _MergedStateAdapter(tuple(flattened))


def _choice_successor_states(
    state: _BaseDecoderState,
) -> tuple[tuple[str, _BaseDecoderState], ...]:
    return state.choice_successor_states()


def _choice_state_entries(state: _BaseDecoderState) -> _StateEntries:
    entries = getattr(state, "_choice_state_entries", None)
    if entries is not None:
        return entries()
    return _eager_state_entries(_choice_successor_states(state))


def _grouped_successor_states(
    state: _BaseDecoderState,
) -> tuple[tuple[str, _BaseDecoderState], ...]:
    return state.grouped_successor_states()


def _grouped_state_entries(state: _BaseDecoderState) -> _StateEntries:
    entries = getattr(state, "_grouped_state_entries", None)
    if entries is not None:
        return entries()
    return _eager_state_entries(_grouped_successor_states(state))


def _determinized_choice_successors(
    state: _BaseDecoderState,
) -> tuple[tuple[str, _BaseDecoderState], ...]:
    """Return one successor per token text by merging same-text branches."""
    return _grouped_successor_states(state)


def _state_cache_key(state: _BaseDecoderState) -> DecoderCacheKey:
    key = state.cache_key()
    if isinstance(key, tuple):
        return cast(DecoderCacheKey, key)
    return ("raw", key)


def _reachable_terminal_prefixes(
    state: _BaseDecoderState,
    *,
    memo: dict[DecoderCacheKey, frozenset[str]] | None = None,
) -> frozenset[str]:
    """Return every terminal prefix reachable from one internal decoder state."""
    if memo is None:
        memo = {}

    key = _state_cache_key(state)
    cached = memo.get(key)
    if cached is not None:
        return cached

    if state.is_terminal():
        terminal = frozenset({state.prefix()})
        memo[key] = terminal
        return terminal

    outputs: set[str] = set()
    for _, successor in _choice_successor_states(state):
        outputs.update(_reachable_terminal_prefixes(successor, memo=memo))
    resolved = frozenset(outputs)
    memo[key] = resolved
    return resolved
