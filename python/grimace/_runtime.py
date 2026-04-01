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

    @property
    def surface_kind(self) -> str:
        if self.isomeric_smiles:
            return CONNECTED_STEREO_SURFACE
        return CONNECTED_NONSTEREO_SURFACE

    def with_rooted_at_atom(self, rooted_at_atom: int) -> "MolToSmilesFlags":
        return replace(self, rooted_at_atom=rooted_at_atom)


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

    support: set[str] = set()
    for local_root_idx in range(atom_count):
        support.update(
            mol_to_smiles_support(
                fragment_mol,
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
class _ChoiceImpl:
    text: str
    next_state: object


class _CoreStateAdapter:
    __slots__ = ("_decoder",)

    def __init__(self, decoder: object) -> None:
        self._decoder = decoder

    def _choice_with_advanced_decoder(self, text: str) -> _ChoiceImpl:
        next_decoder = self._decoder.copy()
        next_decoder.advance_token(text)
        return _ChoiceImpl(text=text, next_state=type(self)(next_decoder))

    def choices(self) -> tuple[_ChoiceImpl, ...]:
        choice_texts = tuple(self._decoder.next_token_support())
        return tuple(
            self._choice_with_advanced_decoder(text)
            for text in choice_texts
        )

    def prefix(self) -> str:
        return self._decoder.prefix()

    def is_terminal(self) -> bool:
        return bool(self._decoder.is_terminal())

    def copy(self) -> "_CoreStateAdapter":
        return type(self)(self._decoder.copy())


class _MergedStateAdapter:
    __slots__ = ("_states",)

    def __init__(self, states: tuple[object, ...]) -> None:
        if not states:
            raise ValueError("Merged decoder state requires at least one branch")
        self._states = states

    def choices(self) -> tuple[_ChoiceImpl, ...]:
        return tuple(
            choice
            for state in self._states
            if not state.is_terminal()
            for choice in state.choices()
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

    def choices(self) -> tuple[_ChoiceImpl, ...]:
        active = self._active_state()
        if not active.is_terminal():
            return tuple(
                _ChoiceImpl(
                    text=choice.text,
                    next_state=type(self)(
                        self._fragment_states[: self._fragment_idx]
                        + (choice.next_state,)
                        + self._fragment_states[self._fragment_idx + 1 :],
                        fragment_idx=self._fragment_idx,
                        completed_prefix=self._completed_prefix,
                    ),
                )
                for choice in active.choices()
            )
        if self._fragment_idx + 1 < len(self._fragment_states):
            next_active = self._fragment_states[self._fragment_idx + 1]
            return (
                _ChoiceImpl(
                    text=".",
                    next_state=type(self)(
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
        return active.is_terminal() and self._fragment_idx + 1 == len(self._fragment_states)

    def copy(self) -> "_DisconnectedStateAdapter":
        return type(self)(
            tuple(state.copy() for state in self._fragment_states),
            fragment_idx=self._fragment_idx,
            completed_prefix=self._completed_prefix,
        )


def prepare_smiles_graph(
    mol_or_prepared: object,
    *,
    flags: MolToSmilesFlags,
) -> _core.PreparedSmilesGraph:
    if isinstance(mol_or_prepared, _core.PreparedSmilesGraph):
        _validate_surface_kind(mol_or_prepared, surface_kind=flags.surface_kind)
        _validate_writer_flags(mol_or_prepared, flags)
        return mol_or_prepared

    if isinstance(mol_or_prepared, ReferencePreparedSmilesGraph):
        _validate_surface_kind(mol_or_prepared, surface_kind=flags.surface_kind)
        _validate_writer_flags(mol_or_prepared, flags)
        return _core.PreparedSmilesGraph(mol_or_prepared)

    _ensure_singly_connected_molecule(mol_or_prepared)
    reference_prepared = prepare_smiles_graph_from_mol_to_smiles_kwargs(
        mol_or_prepared,
        surface_kind=flags.surface_kind,
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
    core_type = stereo_type if flags.isomeric_smiles else nonstereo_type
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


class MolToSmilesDecoder:
    __slots__ = ("_state",)

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
        if _is_disconnected_molecule(mol_or_prepared):
            self._state = _make_disconnected_decoder(cast(Chem.Mol, mol_or_prepared), flags)
        elif flags.rooted_at_atom < 0:
            self._state = _make_fragment_state_adapter(
                mol_or_prepared,
                flags=flags,
                rooted_at_atom=None,
            )
        else:
            self._state = _make_connected_state_adapter(mol_or_prepared, flags)

    @classmethod
    def _from_parts(
        cls,
        state_impl: object,
    ) -> "MolToSmilesDecoder":
        decoder = cls.__new__(cls)
        decoder._state = state_impl
        return decoder

    def choices(self) -> tuple[_ChoiceImpl, ...]:
        return tuple(
            _ChoiceImpl(
                text=choice.text,
                next_state=type(self)._from_parts(choice.next_state),
            )
            for choice in self._state.choices()
        )

    @property
    def prefix(self) -> str:
        return self._state.prefix()

    @property
    def is_terminal(self) -> bool:
        return self._state.is_terminal()

    def copy(self) -> "MolToSmilesDecoder":
        return type(self)._from_parts(self._state.copy())


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
    """Build a cheap token inventory from local prepared-graph structure."""

    if _is_disconnected_molecule(mol_or_prepared):
        return _fragmented_mol_to_smiles_token_inventory(
            cast(Chem.Mol, mol_or_prepared),
            isomeric_smiles=isomeric_smiles,
            kekule_smiles=kekule_smiles,
            rooted_at_atom=rooted_at_atom,
            canonical=canonical,
            all_bonds_explicit=all_bonds_explicit,
            all_hs_explicit=all_hs_explicit,
            do_random=do_random,
            ignore_atom_map_numbers=ignore_atom_map_numbers,
        )

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
    prepared = prepare_smiles_graph(mol_or_prepared, flags=flags)
    prepared_data = prepared.to_dict()

    atom_tokens = cast(list[str], prepared_data["atom_tokens"])
    neighbors = cast(list[list[int]], prepared_data["neighbors"])
    neighbor_bond_tokens = cast(list[list[str]], prepared_data["neighbor_bond_tokens"])

    inventory: set[str] = set()

    chiral_tags = cast(list[str], prepared_data.get("atom_chiral_tags", []))
    explicit_h_counts = cast(list[int], prepared_data.get("atom_explicit_h_counts", []))
    implicit_h_counts = cast(list[int], prepared_data.get("atom_implicit_h_counts", []))
    isotopes = cast(list[int], prepared_data.get("atom_isotopes", []))
    formal_charges = cast(list[int], prepared_data.get("atom_formal_charges", []))
    atom_map_numbers = cast(list[int], prepared_data.get("atom_map_numbers", []))

    def _format_hydrogen_count(hydrogen_count: int) -> str:
        if hydrogen_count == 0:
            return ""
        if hydrogen_count == 1:
            return "H"
        return f"H{hydrogen_count}"

    def _format_charge(formal_charge: int) -> str:
        if formal_charge == 0:
            return ""
        sign = "+" if formal_charge > 0 else "-"
        magnitude = abs(formal_charge)
        if magnitude == 1:
            return sign
        return f"{sign}{magnitude}"

    def _ring_label_text(label: int) -> str:
        if label < 10:
            return str(label)
        if label < 100:
            return f"%{label}"
        return f"%({label})"

    for atom_idx, base_token in enumerate(atom_tokens):
        if flags.isomeric_smiles and chiral_tags and chiral_tags[atom_idx] != "CHI_UNSPECIFIED":
            hydrogen_count = explicit_h_counts[atom_idx] + implicit_h_counts[atom_idx]
            isotope = isotopes[atom_idx]
            formal_charge = formal_charges[atom_idx]
            atom_map_number = 0 if flags.ignore_atom_map_numbers else atom_map_numbers[atom_idx]
            symbol = base_token
            for stereo_mark in ("@", "@@"):
                parts = ["["]
                if isotope:
                    parts.append(str(isotope))
                parts.append(symbol)
                parts.append(stereo_mark)
                parts.append(_format_hydrogen_count(hydrogen_count))
                parts.append(_format_charge(formal_charge))
                if atom_map_number:
                    parts.append(f":{atom_map_number}")
                parts.append("]")
                inventory.add("".join(parts))
        else:
            inventory.add(base_token)

    has_branching = False
    for begin_idx, bonded_tokens in enumerate(neighbor_bond_tokens):
        if len(neighbors[begin_idx]) > 2:
            has_branching = True
        for token in bonded_tokens:
            if token:
                inventory.add(token)

    if has_branching:
        inventory.update({"(", ")"})

    bond_count = cast(int, prepared_data["bond_count"])
    atom_count = cast(int, prepared_data["atom_count"])
    ring_rank = max(0, bond_count - atom_count + 1)
    for label in range(1, ring_rank + 1):
        inventory.add(_ring_label_text(label))

    if flags.isomeric_smiles:
        bond_dirs = cast(list[str], prepared_data.get("bond_dirs", []))
        if any(bond_dir != "NONE" for bond_dir in bond_dirs):
            inventory.update({"/", "\\"})

    return tuple(sorted(inventory))


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
