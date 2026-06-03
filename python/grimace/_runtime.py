"""Internal runtime bridge between RDKit input and the Rust core."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from itertools import product
from typing import cast

import grimace._core as _core
from grimace._runtime_graphs import (
    as_disconnected_prepared_mol as _as_disconnected_prepared_mol,
    atom_count as _atom_count,
    connected_prepared_mol_fragment_or_none as _connected_prepared_mol_fragment_or_none,
    prepared_mol_fragment_plans as _prepared_mol_fragment_plans,
    prepare_core_graph_for_static_inventory as _prepare_core_graph_for_static_inventory,
    prepare_smiles_graph as _prepare_smiles_graph,
)
from grimace._runtime_inputs import (
    MolToSmilesFlags as _MolToSmilesFlags,
    _internal_option_kwargs,
    _make_flags_from_internal_options,
    prepare_runtime_input as _prepare_runtime_input,
)
from grimace._runtime_states import (
    _BaseDecoderState,
    DecoderCacheKey,
    _CoreStateAdapter,
    _DisconnectedStateAdapter,
    _LazyAllRootsConnectedStereoState,
    _StateTransitionFactory,
    _StateTransitions,
    _grouped_successor_states,
    _state_cache_key,
)
from grimace._reference.prepared_graph import (
    CONNECTED_NONSTEREO_SURFACE as _CONNECTED_NONSTEREO_SURFACE,
    CONNECTED_STEREO_SURFACE as _CONNECTED_STEREO_SURFACE,
    PREPARED_SMILES_GRAPH_SCHEMA_VERSION as _PREPARED_SMILES_GRAPH_SCHEMA_VERSION,
)


def _connected_fragment_support(
    fragment_mol: object,
    *,
    flags: _MolToSmilesFlags,
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
    flags: _MolToSmilesFlags,
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
    __slots__ = ("text", "_next_state", "_next_state_factory")

    def __init__(self, text: str, next_state: object) -> None:
        self.text = text
        self.next_state = next_state

    @classmethod
    def _from_next_state_factory(
        cls,
        text: str,
        next_state_factory: Callable[[], object],
    ) -> "MolToSmilesChoice":
        choice = cls.__new__(cls)
        choice.text = text
        choice._next_state = None
        choice._next_state_factory = next_state_factory
        return choice

    @property
    def next_state(self) -> object:
        factory = self._next_state_factory
        if factory is not None:
            self._next_state = factory()
            self._next_state_factory = None
        return self._next_state

    @next_state.setter
    def next_state(self, next_state: object) -> None:
        self._next_state = next_state
        self._next_state_factory = None


def _instantiate_core_object(
    mol_or_prepared: object,
    flags: _MolToSmilesFlags,
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

    prepared = _prepare_smiles_graph(mol_or_prepared, flags=flags)
    core_type = (
        stereo_type
        if prepared.surface_kind == _CONNECTED_STEREO_SURFACE
        else nonstereo_type
    )
    return core_type(prepared, flags.rooted_at_atom)


def _make_walker(
    mol_or_prepared: object,
    flags: _MolToSmilesFlags,
) -> object:
    return _instantiate_core_object(
        mol_or_prepared,
        flags,
        stereo_type=_core.RootedConnectedStereoWalker,
        nonstereo_type=_core.RootedConnectedNonStereoWalker,
    )


def _make_decoder(
    mol_or_prepared: object,
    flags: _MolToSmilesFlags,
) -> object:
    return _instantiate_core_object(
        mol_or_prepared,
        flags,
        stereo_type=_core.RootedConnectedStereoDecoder,
        nonstereo_type=_core.RootedConnectedNonStereoDecoder,
    )


def _make_connected_state_adapter(
    mol_or_prepared: object,
    flags: _MolToSmilesFlags,
) -> _CoreStateAdapter:
    return _CoreStateAdapter(_make_decoder(mol_or_prepared, flags))


def _make_fragment_state_adapter(
    fragment_mol: object,
    *,
    flags: _MolToSmilesFlags,
    rooted_at_atom: int | None,
) -> _BaseDecoderState:
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
    prepared_fragment = _prepare_smiles_graph(
        fragment_for_preparation,
        flags=flags.with_rooted_at_atom(0),
    )
    if prepared_fragment.surface_kind == _CONNECTED_NONSTEREO_SURFACE:
        return _make_connected_state_adapter(
            prepared_fragment,
            flags.with_rooted_at_atom(-1),
        )

    return _LazyAllRootsConnectedStereoState(
        prepared_fragment,
        atom_count,
    )


def _make_disconnected_decoder(
    prepared: object,
    flags: _MolToSmilesFlags,
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
    flags: _MolToSmilesFlags,
) -> _BaseDecoderState:
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
        flags = _make_flags_from_internal_options(locals())
        self._state = _make_decoder_state_impl(
            _prepare_runtime_input(mol_or_prepared, flags=flags),
            flags=flags,
        )
        self._choices_cache = None

    @classmethod
    def _from_parts(
        cls,
        state_impl: _BaseDecoderState,
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
        return self._state.is_terminal()

    def copy(self) -> "_PublicDecoderBase":
        return type(self)._from_parts(self._state.copy())

    def _cache_key(self) -> DecoderCacheKey:
        return _state_cache_key(self._state)

    @property
    def next_choices(self) -> tuple[MolToSmilesChoice, ...]:
        if self._choices_cache is None:
            self._choices_cache = self.choices()
        return self._choices_cache


def _public_decoder_choice(
    decoder_type: type[_PublicDecoderBase],
    text: str,
    state_factory: _StateTransitionFactory,
) -> MolToSmilesChoice:
    return MolToSmilesChoice._from_next_state_factory(
        text,
        lambda: decoder_type._from_parts(state_factory()),
    )


def _public_decoder_choices(
    decoder_type: type[_PublicDecoderBase],
    transitions: _StateTransitions,
) -> tuple[MolToSmilesChoice, ...]:
    if not transitions:
        return ()
    return tuple(
        _public_decoder_choice(decoder_type, text, state_factory)
        for text, state_factory in transitions
    )


class MolToSmilesDecoder(_PublicDecoderBase):
    def choices(self) -> tuple[MolToSmilesChoice, ...]:
        return _public_decoder_choices(
            type(self),
            self._state._choice_state_transitions(),
        )


class MolToSmilesDeterminizedDecoder(_PublicDecoderBase):
    def choices(self) -> tuple[MolToSmilesChoice, ...]:
        return _public_decoder_choices(
            type(self),
            self._state._grouped_state_transitions(),
        )


def _exact_token_inventory_from_decoder(
    mol_or_prepared: object,
    *,
    flags: _MolToSmilesFlags,
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
            grouped_successors = _grouped_successor_states(state)
            inventory.update(text for text, _ in grouped_successors)
            stack.extend(successor for _, successor in grouped_successors)

    return tuple(sorted(inventory))


def _connected_token_inventory_superset(
    mol_or_prepared: object,
    *,
    flags: _MolToSmilesFlags,
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
    flags: _MolToSmilesFlags,
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
    flags = _make_flags_from_internal_options(locals())
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
            **_internal_option_kwargs(locals()),
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

    flags = _make_flags_from_internal_options(locals())
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

    flags = _make_flags_from_internal_options(locals())
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


def _rooted_connected_options(
    root_idx: int,
    *,
    isomeric_smiles: bool,
) -> dict[str, object]:
    return {
        "isomeric_smiles": isomeric_smiles,
        "rooted_at_atom": root_idx,
        "canonical": False,
        "do_random": True,
    }


def enumerate_rooted_connected_nonstereo_smiles_support(
    mol_or_prepared: object,
    root_idx: int,
) -> set[str]:
    return mol_to_smiles_support(
        mol_or_prepared,
        **_rooted_connected_options(root_idx, isomeric_smiles=False),
    )


def enumerate_rooted_connected_stereo_smiles_support(
    mol_or_prepared: object,
    root_idx: int,
) -> set[str]:
    return mol_to_smiles_support(
        mol_or_prepared,
        **_rooted_connected_options(root_idx, isomeric_smiles=True),
    )


def make_nonstereo_walker(
    mol_or_prepared: object,
    root_idx: int,
) -> _core.RootedConnectedNonStereoWalker:
    flags = _make_flags_from_internal_options(
        _rooted_connected_options(root_idx, isomeric_smiles=False)
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
    flags = _make_flags_from_internal_options(
        _rooted_connected_options(root_idx, isomeric_smiles=True)
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
    if core_version != _PREPARED_SMILES_GRAPH_SCHEMA_VERSION:
        raise RuntimeError(
            "Python RDKit bridge and Rust core disagree on prepared graph schema "
            f"version: python={_PREPARED_SMILES_GRAPH_SCHEMA_VERSION}, core={core_version}"
        )
    return core_version
