"""Internal runtime bridge between RDKit input and the Rust core."""

from __future__ import annotations

import importlib
from dataclasses import dataclass

from rdkit import Chem

_core = importlib.import_module("smiles_next_token._core")
from smiles_next_token._reference.prepared_graph import (
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
            "PreparedSmilesGraph writer flags do not match the requested MolToSmilesSupport options"
        )


def _validate_supported_flags(flags: MolToSmilesFlags) -> None:
    if flags.rooted_at_atom < 0:
        raise NotImplementedError("MolToSmilesSupport requires rootedAtAtom >= 0")
    if flags.canonical:
        raise NotImplementedError("MolToSmilesSupport requires canonical=False")
    if not flags.do_random:
        raise NotImplementedError("MolToSmilesSupport requires doRandom=True")


def _ensure_singly_connected_molecule(mol: Chem.Mol) -> None:
    if mol.GetNumAtoms() == 0:
        return
    if len(Chem.GetMolFrags(mol)) != 1:
        raise NotImplementedError(
            "MolToSmilesSupport currently supports only singly-connected molecules"
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
    prepared = prepare_smiles_graph(mol_or_prepared, flags=flags)
    return set(
        _core.mol_to_smiles_support(
            prepared,
            flags.rooted_at_atom,
            flags.isomeric_smiles,
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
    "MolToSmilesFlags",
    "PREPARED_SMILES_GRAPH_SCHEMA_VERSION",
    "enumerate_rooted_connected_nonstereo_smiles_support",
    "enumerate_rooted_connected_stereo_smiles_support",
    "make_nonstereo_walker",
    "make_prepared_graph",
    "make_stereo_walker",
    "mol_to_smiles_support",
    "prepare_smiles_graph",
    "prepared_smiles_graph_schema_version",
]
