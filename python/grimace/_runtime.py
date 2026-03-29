"""Internal runtime bridge between RDKit input and the Rust core."""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from dataclasses import dataclass

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
    if flags.rooted_at_atom < 0:
        raise NotImplementedError("MolToSmiles runtime requires rootedAtAtom >= 0")
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


def _make_walker(
    mol_or_prepared: object,
    flags: MolToSmilesFlags,
) -> object:
    prepared = prepare_smiles_graph(mol_or_prepared, flags=flags)
    if flags.isomeric_smiles:
        return _core.RootedConnectedStereoWalker(prepared, flags.rooted_at_atom)
    return _core.RootedConnectedNonStereoWalker(prepared, flags.rooted_at_atom)


def _make_decoder(
    mol_or_prepared: object,
    flags: MolToSmilesFlags,
) -> object:
    prepared = prepare_smiles_graph(mol_or_prepared, flags=flags)
    if flags.isomeric_smiles:
        return _core.RootedConnectedStereoDecoder(prepared, flags.rooted_at_atom)
    return _core.RootedConnectedNonStereoDecoder(prepared, flags.rooted_at_atom)


class MolToSmilesDecoder:
    __slots__ = ("_decoder",)

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
        flags = MolToSmilesFlags(
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
        self._decoder = _make_decoder(mol_or_prepared, flags)

    @classmethod
    def _from_parts(
        cls,
        decoder_impl: object,
    ) -> "MolToSmilesDecoder":
        decoder = cls.__new__(cls)
        decoder._decoder = decoder_impl
        return decoder

    def nextTokens(self) -> tuple[str, ...]:
        return tuple(self._decoder.next_token_support())

    def advance(self, token: str) -> "MolToSmilesDecoder":
        self._decoder.advance_token(token)
        return self

    def prefix(self) -> str:
        return self._decoder.prefix()

    def isTerminal(self) -> bool:
        return self._decoder.is_terminal()

    def copy(self) -> "MolToSmilesDecoder":
        return type(self)._from_parts(self._decoder.copy())


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
    flags = MolToSmilesFlags(
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
    prepared = prepare_smiles_graph(
        mol_or_prepared,
        flags=MolToSmilesFlags(
            isomeric_smiles=False,
            rooted_at_atom=root_idx,
            canonical=False,
            do_random=True,
        ),
    )
    return _core.RootedConnectedNonStereoWalker(prepared, root_idx)


def make_stereo_walker(
    mol_or_prepared: object,
    root_idx: int,
) -> _core.RootedConnectedStereoWalker:
    prepared = prepare_smiles_graph(
        mol_or_prepared,
        flags=MolToSmilesFlags(
            isomeric_smiles=True,
            rooted_at_atom=root_idx,
            canonical=False,
            do_random=True,
        ),
    )
    return _core.RootedConnectedStereoWalker(prepared, root_idx)


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
