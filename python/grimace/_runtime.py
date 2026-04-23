"""Internal runtime bridge between RDKit input and the Rust core."""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from dataclasses import dataclass, replace
from itertools import product
from typing import cast

from rdkit import Chem

_core = importlib.import_module("grimace._core")
from grimace._reference.prepared_graph import (
    CONNECTED_NONSTEREO_SURFACE,
    CONNECTED_STEREO_SURFACE,
    PREPARED_SMILES_GRAPH_SCHEMA_VERSION,
    PreparedSmilesGraph as ReferencePreparedSmilesGraph,
    prepare_smiles_graph_from_mol_to_smiles_kwargs,
)

@dataclass(frozen=True, slots=True)
class MolToSmilesFlags:
    isomeric_smiles: bool = True
    kekule_smiles: bool = False
    rooted_at_atom: int = -1
    canonical: bool = True
    all_bonds_explicit: bool = False
    all_hs_explicit: bool = False
    do_random: bool = False
    ignore_atom_map_numbers: bool = False

    def with_rooted_at_atom(self, rooted_at_atom: int) -> "MolToSmilesFlags":
        return replace(self, rooted_at_atom=rooted_at_atom)


def _prepared_bond_dirs(prepared: object) -> tuple[str, ...]:
    return tuple(str(value) for value in getattr(prepared, "bond_dirs", ()))


def _requires_stereo_runtime_surface(
    mol_or_prepared: object,
    *,
    flags: MolToSmilesFlags,
) -> bool:
    if flags.isomeric_smiles:
        return True
    if not flags.all_bonds_explicit:
        return False
    if isinstance(mol_or_prepared, Chem.Mol):
        return any(
            bond.GetStereo() != Chem.BondStereo.STEREONONE or bond.GetBondDir() != Chem.BondDir.NONE
            for bond in mol_or_prepared.GetBonds()
        )
    if getattr(mol_or_prepared, "surface_kind", None) != CONNECTED_STEREO_SURFACE:
        return False
    return any(bond_dir != "NONE" for bond_dir in _prepared_bond_dirs(mol_or_prepared))


def _runtime_surface_kind(
    mol_or_prepared: object,
    *,
    flags: MolToSmilesFlags,
) -> str:
    # The runtime is mostly keyed by isomeric_smiles, but explicit single-bond
    # directions still require the stereo surface even when isomeric_smiles=False.
    if _requires_stereo_runtime_surface(mol_or_prepared, flags=flags):
        return CONNECTED_STEREO_SURFACE
    return CONNECTED_NONSTEREO_SURFACE


@dataclass(frozen=True, slots=True)
class _FragmentPlan:
    mol: Chem.Mol
    rooted_at_atom: int | None


def _validate_surface_kind(
    prepared: object,
    *,
    surface_kind: str,
) -> None:
    if prepared.surface_kind != surface_kind:
        raise ValueError(
            f"PreparedSmilesGraph surface_kind={prepared.surface_kind!r} does not match "
            f"the requested surface_kind={surface_kind!r}"
        )


def _validate_writer_flags(
    prepared: object,
    flags: MolToSmilesFlags,
) -> None:
    if isinstance(prepared, _core.PreparedSmilesGraph):
        prepared_data = prepared.to_dict()
        actual = (
            bool(prepared_data["writer_do_isomeric_smiles"]),
            bool(prepared_data["writer_kekule_smiles"]),
            bool(prepared_data["writer_all_bonds_explicit"]),
            bool(prepared_data["writer_all_hs_explicit"]),
            bool(prepared_data["writer_ignore_atom_map_numbers"]),
        )
    else:
        actual = (
            prepared.writer_do_isomeric_smiles,
            prepared.writer_kekule_smiles,
            prepared.writer_all_bonds_explicit,
            prepared.writer_all_hs_explicit,
            prepared.writer_ignore_atom_map_numbers,
        )
    expected = (
        bool(flags.isomeric_smiles),
        bool(flags.kekule_smiles),
        bool(flags.all_bonds_explicit),
        bool(flags.all_hs_explicit),
        bool(flags.ignore_atom_map_numbers),
    )
    if actual != expected:
        raise ValueError(
            "PreparedSmilesGraph writer flags do not match the requested public runtime options"
        )


def _validate_supported_flags(flags: MolToSmilesFlags) -> None:
    if flags.rooted_at_atom < -1:
        raise NotImplementedError("MolToSmiles runtime requires rootedAtAtom == -1 or rootedAtAtom >= 0")
    if flags.canonical:
        raise NotImplementedError("MolToSmiles runtime requires canonical=False")
    if not flags.do_random:
        raise NotImplementedError("MolToSmiles runtime requires doRandom=True")


def _ensure_singly_connected_molecule(mol: Chem.Mol) -> None:
    if mol.GetNumAtoms() == 0:
        return
    if len(Chem.GetMolFrags(mol)) != 1:
        raise NotImplementedError(
            "MolToSmiles runtime currently supports only singly-connected molecules"
        )


def _is_disconnected_molecule(mol_or_prepared: object) -> bool:
    return isinstance(mol_or_prepared, Chem.Mol) and len(Chem.GetMolFrags(mol_or_prepared)) > 1


def _fragment_plans_for_molecule(
    mol: Chem.Mol,
    *,
    rooted_at_atom: int,
) -> tuple[_FragmentPlan, ...]:
    if rooted_at_atom < 0 or rooted_at_atom >= mol.GetNumAtoms():
        raise IndexError("root_idx out of range")

    fragments = Chem.GetMolFrags(mol)
    fragment_mols = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    global_to_local: dict[int, tuple[int, int]] = {}
    for fragment_idx, fragment_atom_indices in enumerate(fragments):
        for local_idx, global_idx in enumerate(fragment_atom_indices):
            global_to_local[global_idx] = (fragment_idx, local_idx)

    rooted_fragment_idx, rooted_local_idx = global_to_local[rooted_at_atom]
    plans: list[_FragmentPlan] = []
    for fragment_idx, fragment_mol in enumerate(fragment_mols):
        if fragment_idx == rooted_fragment_idx:
            plans.append(_FragmentPlan(fragment_mol, rooted_local_idx))
        else:
            plans.append(_FragmentPlan(fragment_mol, None))
    return tuple(plans)


def _atom_count(mol_or_prepared: object) -> int:
    if isinstance(mol_or_prepared, Chem.Mol):
        return mol_or_prepared.GetNumAtoms()
    if isinstance(mol_or_prepared, ReferencePreparedSmilesGraph):
        return mol_or_prepared.atom_count
    if isinstance(mol_or_prepared, _core.PreparedSmilesGraph):
        return cast(int, mol_or_prepared.to_dict()["atom_count"])
    raise TypeError(f"Unsupported molecule/prepared type: {type(mol_or_prepared)!r}")


def _connected_fragment_support(
    fragment_mol: object,
    *,
    flags: MolToSmilesFlags,
    rooted_at_atom: int | None,
) -> set[str]:
    if rooted_at_atom is not None:
        return mol_to_smiles_support(
            fragment_mol,
            isomeric_smiles=flags.isomeric_smiles,
            kekule_smiles=flags.kekule_smiles,
            rooted_at_atom=rooted_at_atom,
            canonical=flags.canonical,
            all_bonds_explicit=flags.all_bonds_explicit,
            all_hs_explicit=flags.all_hs_explicit,
            do_random=flags.do_random,
            ignore_atom_map_numbers=flags.ignore_atom_map_numbers,
        )

    atom_count = _atom_count(fragment_mol)
    if atom_count == 0:
        return {
            ""
        }

    # For all-roots support on an RDKit molecule, build the prepared graph once
    # and reuse it across roots instead of reparsing the same fragment each time.
    prepared_or_fragment = (
        prepare_smiles_graph(fragment_mol, flags=flags)
        if isinstance(fragment_mol, Chem.Mol)
        else fragment_mol
    )

    support: set[str] = set()
    for local_root_idx in range(atom_count):
        support.update(
            mol_to_smiles_support(
                prepared_or_fragment,
                isomeric_smiles=flags.isomeric_smiles,
                kekule_smiles=flags.kekule_smiles,
                rooted_at_atom=local_root_idx,
                canonical=flags.canonical,
                all_bonds_explicit=flags.all_bonds_explicit,
                all_hs_explicit=flags.all_hs_explicit,
                do_random=flags.do_random,
                ignore_atom_map_numbers=flags.ignore_atom_map_numbers,
            )
        )
    return support


def _fragmented_mol_to_smiles_support(
    mol: Chem.Mol,
    *,
    flags: MolToSmilesFlags,
) -> set[str]:
    fragment_supports: list[tuple[str, ...]] = []
    rooted_at_atom = None if flags.rooted_at_atom < 0 else flags.rooted_at_atom

    for plan in _fragment_plans_for_token_inventory(mol, rooted_at_atom=rooted_at_atom):
        support = _connected_fragment_support(
            plan.mol,
            flags=flags,
            rooted_at_atom=plan.rooted_at_atom,
        )
        fragment_supports.append(tuple(sorted(support)))

    return {".".join(parts) for parts in product(*fragment_supports)}


def _fragment_plans_for_token_inventory(
    mol: Chem.Mol,
    *,
    rooted_at_atom: int | None,
) -> tuple[_FragmentPlan, ...]:
    if rooted_at_atom is None:
        return tuple(
            _FragmentPlan(fragment_mol, None)
            for fragment_mol in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        )
    return _fragment_plans_for_molecule(mol, rooted_at_atom=rooted_at_atom)


def _fragmented_mol_to_smiles_token_inventory(
    mol: Chem.Mol,
    *,
    isomeric_smiles: bool,
    kekule_smiles: bool,
    rooted_at_atom: int | None,
    canonical: bool,
    all_bonds_explicit: bool,
    all_hs_explicit: bool,
    do_random: bool,
    ignore_atom_map_numbers: bool,
) -> tuple[str, ...]:
    inventory: set[str] = set()

    for plan in _fragment_plans_for_token_inventory(mol, rooted_at_atom=rooted_at_atom):
        inventory.update(
            mol_to_smiles_token_inventory(
                plan.mol,
                isomeric_smiles=isomeric_smiles,
                kekule_smiles=kekule_smiles,
                rooted_at_atom=plan.rooted_at_atom,
                canonical=canonical,
                all_bonds_explicit=all_bonds_explicit,
                all_hs_explicit=all_hs_explicit,
                do_random=do_random,
                ignore_atom_map_numbers=ignore_atom_map_numbers,
            )
        )

    if len(Chem.GetMolFrags(mol)) > 1:
        inventory.add(".")

    return tuple(sorted(inventory))


@dataclass(frozen=True, slots=True)
class MolToSmilesChoice:
    text: str
    next_state: object


class _CoreStateAdapter:
    __slots__ = ("_decoder",)

    def __init__(self, decoder: object) -> None:
        self._decoder = decoder

    def choice_successor_states(self) -> tuple[tuple[str, object], ...]:
        return tuple(
            (text, type(self)(next_decoder))
            for text, next_decoder in self._decoder.choice_successors()
        )

    def choices(self) -> tuple[MolToSmilesChoice, ...]:
        return tuple(
            MolToSmilesChoice(text=text, next_state=next_state)
            for text, next_state in self.choice_successor_states()
        )

    def prefix(self) -> str:
        return self._decoder.prefix()

    def is_terminal(self) -> bool:
        return bool(self._decoder.is_terminal())

    def copy(self) -> "_CoreStateAdapter":
        return type(self)(self._decoder.copy())

    def cache_key(self) -> str:
        return repr(("core", self._decoder.cache_key()))

    def grouped_successor_states(self) -> tuple[tuple[str, object], ...]:
        return tuple(
            (text, type(self)(next_decoder))
            for text, next_decoder in self._decoder.grouped_successors()
        )

    def _advance_token(self, text: str) -> "_CoreStateAdapter":
        next_decoder = self._decoder.copy()
        next_decoder.advance_token(text)
        return type(self)(next_decoder)


class _MergedStateAdapter:
    __slots__ = ("_states",)

    def __init__(self, states: tuple[object, ...]) -> None:
        if not states:
            raise ValueError("Merged decoder state requires at least one branch")
        self._states = states

    def choice_successor_states(self) -> tuple[tuple[str, object], ...]:
        return tuple(
            (text, next_state)
            for state in self._states
            if not state.is_terminal()
            for text, next_state in state.choice_successor_states()
        )

    def choices(self) -> tuple[MolToSmilesChoice, ...]:
        return tuple(
            MolToSmilesChoice(text=text, next_state=next_state)
            for text, next_state in self.choice_successor_states()
        )

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

    def cache_key(self) -> str:
        return repr(("merged", tuple(sorted(state.cache_key() for state in self._states))))

    def grouped_successor_states(self) -> tuple[tuple[str, object], ...]:
        grouped: dict[str, list[object]] = {}
        for state in self._states:
            for text, successor in state.grouped_successor_states():
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
                for text, successor in active.choice_successor_states()
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

    def choices(self) -> tuple[MolToSmilesChoice, ...]:
        return tuple(
            MolToSmilesChoice(text=text, next_state=next_state)
            for text, next_state in self.choice_successor_states()
        )

    def prefix(self) -> str:
        return f"{self._completed_prefix}{self._active_state().prefix()}"

    def is_terminal(self) -> bool:
        active = self._active_state()
        return active.is_terminal() and self._fragment_idx + 1 == len(self._fragment_states)

    def copy(self) -> "_DisconnectedStateAdapter":
        return type(self)(
            tuple(state.copy() for state in self._fragment_states),
            fragment_idx=self._fragment_idx,
            completed_prefix=self._completed_prefix,
        )

    def cache_key(self) -> str:
        return repr(
            (
                "disconnected",
                self._fragment_idx,
                self._completed_prefix,
                tuple(state.cache_key() for state in self._fragment_states),
            )
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
                for text, successor in active.grouped_successor_states()
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


def _determinized_choice_successors(state: object) -> tuple[tuple[str, object], ...]:
    """Return one successor per token text by merging same-text branches."""
    return state.grouped_successor_states()


def _state_cache_key(state: object) -> str:
    return cast(str, state.cache_key())


def _reachable_terminal_prefixes(
    state: object,
    *,
    memo: dict[str, frozenset[str]] | None = None,
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
    for _, successor in state.choice_successor_states():
        outputs.update(_reachable_terminal_prefixes(successor, memo=memo))
    resolved = frozenset(outputs)
    memo[key] = resolved
    return resolved


def prepare_smiles_graph(
    mol_or_prepared: object,
    *,
    flags: MolToSmilesFlags,
) -> _core.PreparedSmilesGraph:
    surface_kind = _runtime_surface_kind(mol_or_prepared, flags=flags)
    if isinstance(mol_or_prepared, _core.PreparedSmilesGraph):
        _validate_surface_kind(mol_or_prepared, surface_kind=surface_kind)
        _validate_writer_flags(mol_or_prepared, flags)
        return mol_or_prepared

    if isinstance(mol_or_prepared, ReferencePreparedSmilesGraph):
        _validate_surface_kind(mol_or_prepared, surface_kind=surface_kind)
        _validate_writer_flags(mol_or_prepared, flags)
        return _core.PreparedSmilesGraph(mol_or_prepared)

    _ensure_singly_connected_molecule(mol_or_prepared)
    reference_prepared = prepare_smiles_graph_from_mol_to_smiles_kwargs(
        mol_or_prepared,
        surface_kind=surface_kind,
        isomeric_smiles=flags.isomeric_smiles,
        kekule_smiles=flags.kekule_smiles,
        all_bonds_explicit=flags.all_bonds_explicit,
        all_hs_explicit=flags.all_hs_explicit,
        ignore_atom_map_numbers=flags.ignore_atom_map_numbers,
    )
    return _core.PreparedSmilesGraph(reference_prepared)


make_prepared_graph = prepare_smiles_graph


def _make_flags(
    *,
    isomeric_smiles: bool = True,
    kekule_smiles: bool = False,
    rooted_at_atom: int = -1,
    canonical: bool = True,
    all_bonds_explicit: bool = False,
    all_hs_explicit: bool = False,
    do_random: bool = False,
    ignore_atom_map_numbers: bool = False,
) -> MolToSmilesFlags:
    return MolToSmilesFlags(
        isomeric_smiles=isomeric_smiles,
        kekule_smiles=kekule_smiles,
        rooted_at_atom=rooted_at_atom,
        canonical=canonical,
        all_bonds_explicit=all_bonds_explicit,
        all_hs_explicit=all_hs_explicit,
        do_random=do_random,
        ignore_atom_map_numbers=ignore_atom_map_numbers,
    )


def _instantiate_core_object(
    mol_or_prepared: object,
    flags: MolToSmilesFlags,
    *,
    stereo_type: type,
    nonstereo_type: type,
) -> object:
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
        return _make_connected_state_adapter(fragment_mol, flags.with_rooted_at_atom(rooted_at_atom))

    atom_count = _atom_count(fragment_mol)
    if atom_count == 0:
        return _make_connected_state_adapter(fragment_mol, flags.with_rooted_at_atom(0))

    states = tuple(
        _make_connected_state_adapter(fragment_mol, flags.with_rooted_at_atom(local_root_idx))
        for local_root_idx in range(atom_count)
    )
    if len(states) == 1:
        return states[0]
    return _MergedStateAdapter(states)


def _make_disconnected_decoder(
    mol: Chem.Mol,
    flags: MolToSmilesFlags,
) -> _DisconnectedStateAdapter:
    rooted_at_atom = None if flags.rooted_at_atom < 0 else flags.rooted_at_atom
    fragment_states = tuple(
        _make_fragment_state_adapter(
            plan.mol,
            flags=flags,
            rooted_at_atom=plan.rooted_at_atom,
        )
        for plan in _fragment_plans_for_token_inventory(mol, rooted_at_atom=rooted_at_atom)
    )
    return _DisconnectedStateAdapter(fragment_states)


def _make_decoder_state_impl(
    mol_or_prepared: object,
    *,
    flags: MolToSmilesFlags,
) -> object:
    if _is_disconnected_molecule(mol_or_prepared):
        return _make_disconnected_decoder(cast(Chem.Mol, mol_or_prepared), flags)
    if flags.rooted_at_atom < 0:
        return _make_fragment_state_adapter(
            mol_or_prepared,
            flags=flags,
            rooted_at_atom=None,
        )
    return _make_connected_state_adapter(mol_or_prepared, flags)


class _PublicDecoderBase:
    __slots__ = ("_state", "_choices_cache")
    _prefer_choice_cache_for_terminal = True

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
        _validate_supported_flags(flags)
        self._state = _make_decoder_state_impl(mol_or_prepared, flags=flags)
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
        if type(self)._prefer_choice_cache_for_terminal:
            return not self.next_choices
        return self._state.is_terminal()

    def copy(self) -> "_PublicDecoderBase":
        return type(self)._from_parts(self._state.copy())

    @property
    def next_choices(self) -> tuple[MolToSmilesChoice, ...]:
        if self._choices_cache is None:
            self._choices_cache = self.choices()
        return self._choices_cache


class MolToSmilesDecoder(_PublicDecoderBase):
    def choices(self) -> tuple[MolToSmilesChoice, ...]:
        return tuple(
            MolToSmilesChoice(
                text=text,
                next_state=type(self)._from_parts(successor),
            )
            for text, successor in self._state.choice_successor_states()
        )


class MolToSmilesDeterminizedDecoder(_PublicDecoderBase):
    def choices(self) -> tuple[MolToSmilesChoice, ...]:
        return tuple(
            MolToSmilesChoice(
                text=text,
                next_state=type(self)._from_parts(successor),
            )
            for text, successor in _determinized_choice_successors(self._state)
        )


def _token_inventory_root_indices(
    mol_or_prepared: object,
    *,
    rooted_at_atom: int | None,
) -> tuple[int, ...]:
    atom_count = _atom_count(mol_or_prepared)
    if atom_count == 0:
        return (0,)
    if rooted_at_atom is None:
        return tuple(range(atom_count))
    return (rooted_at_atom,)


def _exact_token_inventory_from_decoder(
    mol_or_prepared: object,
    *,
    isomeric_smiles: bool,
    kekule_smiles: bool,
    rooted_at_atom: int | None,
    canonical: bool,
    all_bonds_explicit: bool,
    all_hs_explicit: bool,
    do_random: bool,
    ignore_atom_map_numbers: bool,
) -> tuple[str, ...]:
    inventory: set[str] = set()
    visited_state_keys: set[str] = set()

    for root_idx in _token_inventory_root_indices(
        mol_or_prepared,
        rooted_at_atom=rooted_at_atom,
    ):
        decoder = MolToSmilesDecoder(
            mol_or_prepared,
            isomeric_smiles=isomeric_smiles,
            kekule_smiles=kekule_smiles,
            rooted_at_atom=root_idx,
            canonical=canonical,
            all_bonds_explicit=all_bonds_explicit,
            all_hs_explicit=all_hs_explicit,
            do_random=do_random,
            ignore_atom_map_numbers=ignore_atom_map_numbers,
        )
        stack = [decoder._state]

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
    _validate_supported_flags(flags)
    if _is_disconnected_molecule(mol_or_prepared):
        if flags.rooted_at_atom < 0:
            return iter(
                sorted(
                    _fragmented_mol_to_smiles_support(
                        cast(Chem.Mol, mol_or_prepared),
                        flags=flags,
                    )
                )
            )
        return iter(
            sorted(
                _fragmented_mol_to_smiles_support(
                    cast(Chem.Mol, mol_or_prepared),
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
    rooted_at_atom: int | None = None,
    canonical: bool = True,
    all_bonds_explicit: bool = False,
    all_hs_explicit: bool = False,
    do_random: bool = False,
    ignore_atom_map_numbers: bool = False,
) -> tuple[str, ...]:
    """Return the exact decoder token inventory under the public runtime flags."""

    effective_root = 0 if rooted_at_atom is None else rooted_at_atom
    flags = _make_flags(
        isomeric_smiles=isomeric_smiles,
        kekule_smiles=kekule_smiles,
        rooted_at_atom=effective_root,
        canonical=canonical,
        all_bonds_explicit=all_bonds_explicit,
        all_hs_explicit=all_hs_explicit,
        do_random=do_random,
        ignore_atom_map_numbers=ignore_atom_map_numbers,
    )
    _validate_supported_flags(flags)
    return _exact_token_inventory_from_decoder(
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
    "make_prepared_graph",
    "make_stereo_walker",
    "mol_to_smiles_enum",
    "mol_to_smiles_support",
    "prepare_smiles_graph",
    "prepared_smiles_graph_schema_version",
]
