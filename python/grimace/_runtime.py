"""Internal runtime bridge between RDKit input and the Rust core."""

from __future__ import annotations

import importlib
from collections.abc import Hashable, Iterator, Sequence
from itertools import product
from typing import Protocol, TypeAlias, cast

from grimace._runtime_graphs import (
    as_disconnected_prepared_mol as _as_disconnected_prepared_mol,
    atom_count as _atom_count,
    connected_prepared_mol_fragment_or_none as _connected_prepared_mol_fragment_or_none,
    prepared_mol_fragment_plans as _prepared_mol_fragment_plans,
    prepare_core_graph_for_static_inventory as _prepare_core_graph_for_static_inventory,
    prepare_smiles_graph,
)
from grimace._runtime_inputs import (
    MolToSmilesFlags,
    make_flags as _make_flags,
    prepare_runtime_input as _prepare_runtime_input,
)

_core = importlib.import_module("grimace._core")
from grimace._reference.prepared_graph import (
    CONNECTED_NONSTEREO_SURFACE,
    CONNECTED_STEREO_SURFACE,
    PREPARED_SMILES_GRAPH_SCHEMA_VERSION,
)

DecoderCacheKey: TypeAlias = tuple[object, ...]


class _BaseDecoderState(Protocol):
    def prefix(self) -> str: ...
    def is_terminal(self) -> bool: ...
    def copy(self) -> object: ...
    def cache_key(self) -> Hashable: ...


class _CoreDecoderState(_BaseDecoderState, Protocol):
    def choice_successors(self) -> Sequence[tuple[str, object]]: ...
    def grouped_successors(self) -> Sequence[tuple[str, object]]: ...


class _AdapterDecoderState(_BaseDecoderState, Protocol):
    def choice_successor_states(self) -> tuple[tuple[str, object], ...]: ...
    def grouped_successor_states(self) -> tuple[tuple[str, object], ...]: ...


def _connected_fragment_support(
    fragment_mol: object,
    *,
    flags: MolToSmilesFlags,
    rooted_at_atom: int | None,
) -> set[str]:
    if rooted_at_atom is not None:
        walker = _make_walker(
            fragment_mol,
            flags.with_rooted_at_atom(rooted_at_atom),
        )
        return set(walker.enumerate_support())

    # For all-roots support, build or unwrap the prepared graph once and reuse
    # it across roots instead of reparsing the same fragment each time.
    atom_count = _atom_count(fragment_mol)
    connected_fragment = _connected_prepared_mol_fragment_or_none(
        fragment_mol,
        rooted_at_atom=-1,
    )
    prepared_or_fragment = (
        fragment_mol if connected_fragment is None else connected_fragment[0]
    )

    support: set[str] = set()
    local_root_indices = (0,) if atom_count == 0 else range(atom_count)
    for local_root_idx in local_root_indices:
        walker = _make_walker(
            prepared_or_fragment,
            flags.with_rooted_at_atom(local_root_idx),
        )
        support.update(walker.enumerate_support())
    return support


def _fragmented_prepared_support(
    prepared: object,
    *,
    flags: MolToSmilesFlags,
) -> set[str]:
    fragment_supports: list[tuple[str, ...]] = []
    rooted_at_atom = None if flags.rooted_at_atom < 0 else flags.rooted_at_atom

    for plan in _prepared_mol_fragment_plans(
        prepared,
        rooted_at_atom=rooted_at_atom,
    ):
        support = _connected_fragment_support(
            plan.fragment,
            flags=flags,
            rooted_at_atom=plan.rooted_at_atom,
        )
        fragment_supports.append(tuple(sorted(support)))

    return {".".join(parts) for parts in product(*fragment_supports)}


class MolToSmilesChoice:
    __slots__ = ("text", "next_state")

    def __init__(self, text: str, next_state: object) -> None:
        self.text = text
        self.next_state = next_state


def _public_decoder_choices(
    decoder_type: type,
    successors: Sequence[tuple[str, object]],
) -> tuple[MolToSmilesChoice, ...]:
    if not successors:
        return ()
    decoder_new = decoder_type.__new__
    choice_new = MolToSmilesChoice.__new__
    choices = [None] * len(successors)
    for idx, (text, successor) in enumerate(successors):
        next_state = decoder_new(decoder_type)
        next_state._state = successor
        next_state._choices_cache = None
        choice = choice_new(MolToSmilesChoice)
        choice.text = text
        choice.next_state = next_state
        choices[idx] = choice
    return cast(tuple[MolToSmilesChoice, ...], tuple(choices))


class _CoreStateAdapter:
    __slots__ = ("_decoder",)

    def __init__(self, decoder: object) -> None:
        self._decoder = decoder

    @staticmethod
    def _successor_states(
        successors: Sequence[tuple[str, object]],
    ) -> tuple[tuple[str, object], ...]:
        if not successors:
            return ()
        successor_states = [None] * len(successors)
        for idx, (text, next_decoder) in enumerate(successors):
            next_state = _CoreStateAdapter.__new__(_CoreStateAdapter)
            next_state._decoder = next_decoder
            successor_states[idx] = (text, next_state)
        return cast(tuple[tuple[str, object], ...], tuple(successor_states))

    def choice_successor_states(self) -> tuple[tuple[str, object], ...]:
        return self._successor_states(self._decoder.choice_successors())

    def prefix(self) -> str:
        return self._decoder.prefix()

    def is_terminal(self) -> bool:
        return bool(self._decoder.is_terminal())

    def copy(self) -> "_CoreStateAdapter":
        return type(self)(self._decoder.copy())

    def cache_key(self) -> DecoderCacheKey:
        return ("core", self._decoder.cache_key())

    def grouped_successor_states(self) -> tuple[tuple[str, object], ...]:
        return self._successor_states(self._decoder.grouped_successors())


class _MergedStateAdapter:
    __slots__ = ("_states",)

    def __init__(self, states: tuple[object, ...]) -> None:
        if not states:
            raise ValueError("Merged decoder state requires at least one branch")
        self._states = states

    def choice_successor_states(self) -> tuple[tuple[str, object], ...]:
        successor_states: list[tuple[str, object]] = []
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

    def grouped_successor_states(self) -> tuple[tuple[str, object], ...]:
        grouped: dict[str, list[object]] = {}
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
        fragment_states: tuple[object, ...],
        *,
        fragment_idx: int = 0,
        completed_prefix: str = "",
    ) -> None:
        if not fragment_states:
            raise ValueError("Disconnected decoder requires at least one fragment state")
        self._fragment_states = fragment_states
        self._fragment_idx = fragment_idx
        self._completed_prefix = completed_prefix

    def _active_state(self) -> object:
        return self._fragment_states[self._fragment_idx]

    def choice_successor_states(self) -> tuple[tuple[str, object], ...]:
        active = self._active_state()
        if not active.is_terminal():
            successor_states: list[tuple[str, object]] = []
            adapter_type = type(self)
            for text, successor in _choice_successor_states(active):
                successor_states.append(
                    (
                        text,
                        adapter_type(
                            self._fragment_states[: self._fragment_idx]
                            + (successor,)
                            + self._fragment_states[self._fragment_idx + 1 :],
                            fragment_idx=self._fragment_idx,
                            completed_prefix=self._completed_prefix,
                        ),
                    )
                )
            return tuple(successor_states)
        if self._fragment_idx + 1 < len(self._fragment_states):
            return (
                (
                    ".",
                    type(self)(
                        self._fragment_states,
                        fragment_idx=self._fragment_idx + 1,
                        completed_prefix=f"{self._completed_prefix}{active.prefix()}.",
                    ),
                ),
            )
        return ()

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

    def grouped_successor_states(self) -> tuple[tuple[str, object], ...]:
        active = self._active_state()
        if not active.is_terminal():
            return tuple(
                (
                    text,
                    type(self)(
                        self._fragment_states[: self._fragment_idx]
                        + (successor,)
                        + self._fragment_states[self._fragment_idx + 1 :],
                        fragment_idx=self._fragment_idx,
                        completed_prefix=self._completed_prefix,
                    ),
                )
                for text, successor in _grouped_successor_states(active)
            )
        if self._fragment_idx + 1 < len(self._fragment_states):
            return (
                (
                    ".",
                    type(self)(
                        self._fragment_states,
                        fragment_idx=self._fragment_idx + 1,
                        completed_prefix=f"{self._completed_prefix}{active.prefix()}.",
                    ),
                ),
            )
        return ()


def _merge_choice_successor_states(states: tuple[object, ...]) -> object:
    flattened: list[object] = []
    for state in states:
        if isinstance(state, _MergedStateAdapter):
            flattened.extend(state._states)
        else:
            flattened.append(state)
    if len(flattened) == 1:
        return flattened[0]
    return _MergedStateAdapter(tuple(flattened))


def _choice_successor_states(state: _AdapterDecoderState | _CoreDecoderState) -> tuple[tuple[str, object], ...]:
    if isinstance(state, (_CoreStateAdapter, _MergedStateAdapter, _DisconnectedStateAdapter)):
        return state.choice_successor_states()
    successors = state.choice_successors()
    if not successors:
        return ()
    return tuple(successors)


def _grouped_successor_states(state: _AdapterDecoderState | _CoreDecoderState) -> tuple[tuple[str, object], ...]:
    if isinstance(state, (_CoreStateAdapter, _MergedStateAdapter, _DisconnectedStateAdapter)):
        return state.grouped_successor_states()
    successors = state.grouped_successors()
    if not successors:
        return ()
    return tuple(successors)


def _determinized_choice_successors(
    state: _AdapterDecoderState | _CoreDecoderState,
) -> tuple[tuple[str, object], ...]:
    """Return one successor per token text by merging same-text branches."""
    return _grouped_successor_states(state)


def _state_cache_key(state: _BaseDecoderState) -> DecoderCacheKey:
    key = state.cache_key()
    if isinstance(key, tuple):
        return cast(DecoderCacheKey, key)
    return ("raw", key)


def _reachable_terminal_prefixes(
    state: _AdapterDecoderState | _CoreDecoderState,
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


def _instantiate_core_object(
    mol_or_prepared: object,
    flags: MolToSmilesFlags,
    *,
    stereo_type: type,
    nonstereo_type: type,
) -> object:
    connected_fragment = _connected_prepared_mol_fragment_or_none(
        mol_or_prepared,
        rooted_at_atom=flags.rooted_at_atom,
    )
    if connected_fragment is not None:
        fragment, rooted_at_atom = connected_fragment
        mol_or_prepared = fragment
        flags = flags.with_rooted_at_atom(rooted_at_atom)

    prepared = prepare_smiles_graph(mol_or_prepared, flags=flags)
    core_type = stereo_type if prepared.surface_kind == CONNECTED_STEREO_SURFACE else nonstereo_type
    return core_type(prepared, flags.rooted_at_atom)


def _make_walker(
    mol_or_prepared: object,
    flags: MolToSmilesFlags,
) -> object:
    return _instantiate_core_object(
        mol_or_prepared,
        flags,
        stereo_type=_core.RootedConnectedStereoWalker,
        nonstereo_type=_core.RootedConnectedNonStereoWalker,
    )


def _make_decoder(
    mol_or_prepared: object,
    flags: MolToSmilesFlags,
) -> object:
    return _instantiate_core_object(
        mol_or_prepared,
        flags,
        stereo_type=_core.RootedConnectedStereoDecoder,
        nonstereo_type=_core.RootedConnectedNonStereoDecoder,
    )


def _make_connected_state_adapter(
    mol_or_prepared: object,
    flags: MolToSmilesFlags,
) -> _CoreStateAdapter:
    return _CoreStateAdapter(_make_decoder(mol_or_prepared, flags))


def _make_fragment_state_adapter(
    fragment_mol: object,
    *,
    flags: MolToSmilesFlags,
    rooted_at_atom: int | None,
) -> object:
    if rooted_at_atom is not None:
        return _make_connected_state_adapter(
            fragment_mol,
            flags.with_rooted_at_atom(rooted_at_atom),
        )

    atom_count = _atom_count(fragment_mol)
    if atom_count == 0:
        return _make_connected_state_adapter(
            fragment_mol,
            flags.with_rooted_at_atom(0),
        )

    fragment_for_preparation = fragment_mol
    connected_fragment = _connected_prepared_mol_fragment_or_none(
        fragment_mol,
        rooted_at_atom=-1,
    )
    if connected_fragment is not None:
        fragment_for_preparation, _ = connected_fragment
    prepared_fragment = prepare_smiles_graph(
        fragment_for_preparation,
        flags=flags.with_rooted_at_atom(0),
    )
    if prepared_fragment.surface_kind == CONNECTED_NONSTEREO_SURFACE:
        return _make_connected_state_adapter(
            prepared_fragment,
            flags.with_rooted_at_atom(-1),
        )

    states = tuple(
        _make_decoder(prepared_fragment, flags.with_rooted_at_atom(local_root_idx))
        for local_root_idx in range(atom_count)
    )
    if len(states) == 1:
        return states[0]
    return _MergedStateAdapter(states)


def _make_disconnected_decoder(
    prepared: object,
    flags: MolToSmilesFlags,
) -> _DisconnectedStateAdapter:
    rooted_at_atom = None if flags.rooted_at_atom < 0 else flags.rooted_at_atom
    fragment_states = tuple(
        _make_fragment_state_adapter(
            plan.fragment,
            flags=flags,
            rooted_at_atom=plan.rooted_at_atom,
        )
        for plan in _prepared_mol_fragment_plans(
            prepared,
            rooted_at_atom=rooted_at_atom,
        )
    )
    return _DisconnectedStateAdapter(fragment_states)


def _make_decoder_state_impl(
    mol_or_prepared: object,
    *,
    flags: MolToSmilesFlags,
) -> object:
    disconnected = _as_disconnected_prepared_mol(mol_or_prepared)
    if disconnected is not None:
        return _make_disconnected_decoder(disconnected, flags)
    if flags.rooted_at_atom < 0:
        return _make_fragment_state_adapter(
            mol_or_prepared,
            flags=flags,
            rooted_at_atom=None,
        )
    return _make_connected_state_adapter(mol_or_prepared, flags)


class _PublicDecoderBase:
    __slots__ = ("_state", "_choices_cache")

    def __init__(
        self,
        mol_or_prepared: object,
        *,
        isomeric_smiles: bool = True,
        kekule_smiles: bool = False,
        rooted_at_atom: int = -1,
        canonical: bool = True,
        all_bonds_explicit: bool = False,
        all_hs_explicit: bool = False,
        do_random: bool = False,
        ignore_atom_map_numbers: bool = False,
    ) -> None:
        flags = _make_flags(
            isomeric_smiles=isomeric_smiles,
            kekule_smiles=kekule_smiles,
            rooted_at_atom=rooted_at_atom,
            canonical=canonical,
            all_bonds_explicit=all_bonds_explicit,
            all_hs_explicit=all_hs_explicit,
            do_random=do_random,
            ignore_atom_map_numbers=ignore_atom_map_numbers,
        )
        self._state = _make_decoder_state_impl(
            _prepare_runtime_input(mol_or_prepared, flags=flags),
            flags=flags,
        )
        self._choices_cache = None

    @classmethod
    def _from_parts(
        cls,
        state_impl: object,
    ) -> "_PublicDecoderBase":
        decoder = cls.__new__(cls)
        decoder._state = state_impl
        decoder._choices_cache = None
        return decoder

    @property
    def prefix(self) -> str:
        return self._state.prefix()

    @property
    def is_terminal(self) -> bool:
        if self._choices_cache is not None:
            return not self._choices_cache
        return not self.next_choices

    def copy(self) -> "_PublicDecoderBase":
        return type(self)._from_parts(self._state.copy())

    @property
    def next_choices(self) -> tuple[MolToSmilesChoice, ...]:
        if self._choices_cache is None:
            self._choices_cache = self.choices()
        return self._choices_cache


class MolToSmilesDecoder(_PublicDecoderBase):
    def choices(self) -> tuple[MolToSmilesChoice, ...]:
        return _public_decoder_choices(
            type(self),
            _choice_successor_states(self._state),
        )


class MolToSmilesDeterminizedDecoder(_PublicDecoderBase):
    def choices(self) -> tuple[MolToSmilesChoice, ...]:
        return _public_decoder_choices(
            type(self),
            _determinized_choice_successors(self._state),
        )


def _exact_token_inventory_from_decoder(
    mol_or_prepared: object,
    *,
    flags: MolToSmilesFlags,
) -> tuple[str, ...]:
    inventory: set[str] = set()
    visited_state_keys: set[DecoderCacheKey] = set()

    root_indices = (-1,) if flags.rooted_at_atom < 0 else (flags.rooted_at_atom,)
    for root_idx in root_indices:
        stack = [
            _make_decoder_state_impl(
                mol_or_prepared,
                flags=flags.with_rooted_at_atom(root_idx),
            )
        ]

        while stack:
            state = stack.pop()
            state_key = _state_cache_key(state)
            if state_key in visited_state_keys:
                continue
            visited_state_keys.add(state_key)
            grouped_successors = _determinized_choice_successors(state)
            inventory.update(text for text, _ in grouped_successors)
            stack.extend(successor for _, successor in grouped_successors)

    return tuple(sorted(inventory))


def _connected_token_inventory_superset(
    mol_or_prepared: object,
    *,
    flags: MolToSmilesFlags,
) -> tuple[str, ...]:
    rooted_at_atom = flags.rooted_at_atom
    connected_fragment = _connected_prepared_mol_fragment_or_none(
        mol_or_prepared,
        rooted_at_atom=flags.rooted_at_atom,
    )
    if connected_fragment is not None:
        fragment, rooted_at_atom = connected_fragment
        mol_or_prepared = fragment
        flags = flags.with_rooted_at_atom(rooted_at_atom)

    prepared = _prepare_core_graph_for_static_inventory(
        mol_or_prepared,
        flags=flags,
    )
    return tuple(prepared.token_inventory_superset(rooted_at_atom))


def _fragmented_prepared_token_inventory_superset(
    prepared: object,
    *,
    flags: MolToSmilesFlags,
) -> tuple[str, ...]:
    rooted_at_atom = None if flags.rooted_at_atom < 0 else flags.rooted_at_atom
    inventory: set[str] = set()
    fragment_plans = _prepared_mol_fragment_plans(
        prepared,
        rooted_at_atom=rooted_at_atom,
    )

    for plan in fragment_plans:
        fragment_root = -1 if plan.rooted_at_atom is None else plan.rooted_at_atom
        inventory.update(
            _connected_token_inventory_superset(
                plan.fragment,
                flags=flags.with_rooted_at_atom(fragment_root),
            )
        )

    if len(fragment_plans) > 1:
        inventory.add(".")

    return tuple(sorted(inventory))


def mol_to_smiles_enum(
    mol_or_prepared: object,
    *,
    isomeric_smiles: bool = True,
    kekule_smiles: bool = False,
    rooted_at_atom: int = -1,
    canonical: bool = True,
    all_bonds_explicit: bool = False,
    all_hs_explicit: bool = False,
    do_random: bool = False,
    ignore_atom_map_numbers: bool = False,
) -> Iterator[str]:
    flags = _make_flags(
        isomeric_smiles=isomeric_smiles,
        kekule_smiles=kekule_smiles,
        rooted_at_atom=rooted_at_atom,
        canonical=canonical,
        all_bonds_explicit=all_bonds_explicit,
        all_hs_explicit=all_hs_explicit,
        do_random=do_random,
        ignore_atom_map_numbers=ignore_atom_map_numbers,
    )
    mol_or_prepared = _prepare_runtime_input(mol_or_prepared, flags=flags)
    disconnected = _as_disconnected_prepared_mol(mol_or_prepared)
    if disconnected is not None:
        return iter(
            sorted(
                _fragmented_prepared_support(
                    disconnected,
                    flags=flags,
                )
            )
        )
    if flags.rooted_at_atom < 0:
        return iter(
            sorted(
                _connected_fragment_support(
                    mol_or_prepared,
                    flags=flags,
                    rooted_at_atom=None,
                )
            )
        )
    walker = _make_walker(mol_or_prepared, flags)
    return iter(walker.enumerate_support())


def mol_to_smiles_support(
    mol_or_prepared: object,
    *,
    isomeric_smiles: bool = True,
    kekule_smiles: bool = False,
    rooted_at_atom: int = -1,
    canonical: bool = True,
    all_bonds_explicit: bool = False,
    all_hs_explicit: bool = False,
    do_random: bool = False,
    ignore_atom_map_numbers: bool = False,
) -> set[str]:
    return set(
        mol_to_smiles_enum(
            mol_or_prepared,
            isomeric_smiles=isomeric_smiles,
            kekule_smiles=kekule_smiles,
            rooted_at_atom=rooted_at_atom,
            canonical=canonical,
            all_bonds_explicit=all_bonds_explicit,
            all_hs_explicit=all_hs_explicit,
            do_random=do_random,
            ignore_atom_map_numbers=ignore_atom_map_numbers,
        )
    )


def mol_to_smiles_token_inventory(
    mol_or_prepared: object,
    *,
    isomeric_smiles: bool = True,
    kekule_smiles: bool = False,
    rooted_at_atom: int = -1,
    canonical: bool = True,
    all_bonds_explicit: bool = False,
    all_hs_explicit: bool = False,
    do_random: bool = False,
    ignore_atom_map_numbers: bool = False,
) -> tuple[str, ...]:
    """Return the exact decoder token inventory under the public runtime flags."""

    flags = _make_flags(
        isomeric_smiles=isomeric_smiles,
        kekule_smiles=kekule_smiles,
        rooted_at_atom=rooted_at_atom,
        canonical=canonical,
        all_bonds_explicit=all_bonds_explicit,
        all_hs_explicit=all_hs_explicit,
        do_random=do_random,
        ignore_atom_map_numbers=ignore_atom_map_numbers,
    )
    mol_or_prepared = _prepare_runtime_input(mol_or_prepared, flags=flags)
    return _exact_token_inventory_from_decoder(
        mol_or_prepared,
        flags=flags,
    )


def mol_to_smiles_token_inventory_superset(
    mol_or_prepared: object,
    *,
    isomeric_smiles: bool = True,
    kekule_smiles: bool = False,
    rooted_at_atom: int = -1,
    canonical: bool = True,
    all_bonds_explicit: bool = False,
    all_hs_explicit: bool = False,
    do_random: bool = False,
    ignore_atom_map_numbers: bool = False,
) -> tuple[str, ...]:
    """Return a conservative static superset of reachable decoder tokens."""

    flags = _make_flags(
        isomeric_smiles=isomeric_smiles,
        kekule_smiles=kekule_smiles,
        rooted_at_atom=rooted_at_atom,
        canonical=canonical,
        all_bonds_explicit=all_bonds_explicit,
        all_hs_explicit=all_hs_explicit,
        do_random=do_random,
        ignore_atom_map_numbers=ignore_atom_map_numbers,
    )
    mol_or_prepared = _prepare_runtime_input(mol_or_prepared, flags=flags)
    disconnected = _as_disconnected_prepared_mol(mol_or_prepared)
    if disconnected is not None:
        return _fragmented_prepared_token_inventory_superset(
            disconnected,
            flags=flags,
        )
    return _connected_token_inventory_superset(
        mol_or_prepared,
        flags=flags,
    )


def enumerate_rooted_connected_nonstereo_smiles_support(
    mol_or_prepared: object,
    root_idx: int,
) -> set[str]:
    return mol_to_smiles_support(
        mol_or_prepared,
        isomeric_smiles=False,
        rooted_at_atom=root_idx,
        canonical=False,
        do_random=True,
    )


def enumerate_rooted_connected_stereo_smiles_support(
    mol_or_prepared: object,
    root_idx: int,
) -> set[str]:
    return mol_to_smiles_support(
        mol_or_prepared,
        isomeric_smiles=True,
        rooted_at_atom=root_idx,
        canonical=False,
        do_random=True,
    )


def make_nonstereo_walker(
    mol_or_prepared: object,
    root_idx: int,
) -> _core.RootedConnectedNonStereoWalker:
    flags = _make_flags(
        isomeric_smiles=False,
        rooted_at_atom=root_idx,
        canonical=False,
        do_random=True,
    )
    return cast(
        _core.RootedConnectedNonStereoWalker,
        _instantiate_core_object(
            mol_or_prepared,
            flags,
            stereo_type=_core.RootedConnectedStereoWalker,
            nonstereo_type=_core.RootedConnectedNonStereoWalker,
        ),
    )


def make_stereo_walker(
    mol_or_prepared: object,
    root_idx: int,
) -> _core.RootedConnectedStereoWalker:
    flags = _make_flags(
        isomeric_smiles=True,
        rooted_at_atom=root_idx,
        canonical=False,
        do_random=True,
    )
    return cast(
        _core.RootedConnectedStereoWalker,
        _instantiate_core_object(
            mol_or_prepared,
            flags,
            stereo_type=_core.RootedConnectedStereoWalker,
            nonstereo_type=_core.RootedConnectedNonStereoWalker,
        ),
    )


def prepared_smiles_graph_schema_version() -> int:
    core_version = _core.prepared_smiles_graph_schema_version()
    if core_version != PREPARED_SMILES_GRAPH_SCHEMA_VERSION:
        raise RuntimeError(
            "Python RDKit bridge and Rust core disagree on prepared graph schema "
            f"version: python={PREPARED_SMILES_GRAPH_SCHEMA_VERSION}, core={core_version}"
        )
    return core_version


__all__ = [
    "CONNECTED_NONSTEREO_SURFACE",
    "CONNECTED_STEREO_SURFACE",
    "MolToSmilesDecoder",
    "MolToSmilesDeterminizedDecoder",
    "MolToSmilesFlags",
    "PREPARED_SMILES_GRAPH_SCHEMA_VERSION",
    "enumerate_rooted_connected_nonstereo_smiles_support",
    "enumerate_rooted_connected_stereo_smiles_support",
    "make_nonstereo_walker",
    "make_stereo_walker",
    "mol_to_smiles_enum",
    "mol_to_smiles_support",
    "prepare_smiles_graph",
    "prepared_smiles_graph_schema_version",
]
