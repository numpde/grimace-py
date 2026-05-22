"""Public runtime input normalization."""

from __future__ import annotations

from dataclasses import dataclass, replace

import grimace._prepared_mol as _prepared_mol
from grimace._mol_to_smiles_options import (
    MOL_TO_SMILES_OPTIONS,
    MOL_TO_SMILES_PREPARED_OPTIONS,
    coerce_internal_options,
    coerce_option,
)
from grimace._prepared_mol import PreparedMol
from grimace._reference.prepared_graph import (
    CONNECTED_NONSTEREO_SURFACE,
    CONNECTED_STEREO_SURFACE,
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


def make_flags(
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
        **coerce_internal_options(
            MOL_TO_SMILES_OPTIONS,
            locals(),
            context="MolToSmiles runtime",
        )
    )


def writer_flag_kwargs(flags: MolToSmilesFlags) -> dict[str, bool]:
    return {
        spec.internal_name: bool(getattr(flags, spec.internal_name))
        for spec in MOL_TO_SMILES_PREPARED_OPTIONS
    }


def runtime_surface_kind(
    mol_or_prepared: object,
    *,
    flags: MolToSmilesFlags,
) -> str:
    if _requires_stereo_runtime_surface(mol_or_prepared, flags=flags):
        return CONNECTED_STEREO_SURFACE
    return CONNECTED_NONSTEREO_SURFACE


def prepare_runtime_input(
    mol_or_prepared: object,
    *,
    flags: MolToSmilesFlags,
) -> object:
    _validate_supported_flags(flags)
    if isinstance(mol_or_prepared, PreparedMol):
        _validate_prepared_mol_writer_flags(mol_or_prepared, flags)
        return mol_or_prepared
    if _prepared_mol._is_rdkit_mol(mol_or_prepared):
        return _prepared_mol.PrepareMol(
            mol_or_prepared,
            **_runtime_public_writer_flag_kwargs(flags),
        )
    return mol_or_prepared


def ensure_singly_connected_molecule(mol: object) -> None:
    if _prepared_mol._rdkit_mol_atom_count(mol) == 0:
        return
    if _prepared_mol._rdkit_mol_fragment_count(mol) != 1:
        raise NotImplementedError(
            "MolToSmiles runtime currently supports only singly-connected molecules"
        )


def _requires_stereo_runtime_surface(
    mol_or_prepared: object,
    *,
    flags: MolToSmilesFlags,
) -> bool:
    if flags.isomeric_smiles:
        return True
    if not flags.all_bonds_explicit:
        return False
    if _prepared_mol._is_rdkit_mol(mol_or_prepared):
        return _prepared_mol._rdkit_mol_requires_stereo_surface(mol_or_prepared)
    if getattr(mol_or_prepared, "surface_kind", None) != CONNECTED_STEREO_SURFACE:
        return False
    return any(
        str(bond_dir) != "NONE"
        for bond_dir in getattr(mol_or_prepared, "bond_dirs", ())
    )


def _runtime_public_writer_flag_kwargs(flags: MolToSmilesFlags) -> dict[str, bool]:
    return {
        spec.public_name: bool(getattr(flags, spec.internal_name))
        for spec in MOL_TO_SMILES_PREPARED_OPTIONS
    }


def _validate_prepared_mol_writer_flags(
    prepared: PreparedMol,
    flags: MolToSmilesFlags,
) -> None:
    if not _prepared_mol._matches_writer_flags(
        prepared,
        **writer_flag_kwargs(flags),
    ):
        raise ValueError(
            "PreparedMol writer flags do not match the requested public runtime options"
        )


def _validate_supported_flags(flags: MolToSmilesFlags) -> None:
    normalized = {
        spec.internal_name: coerce_option(
            spec,
            getattr(flags, spec.internal_name),
            context="MolToSmiles runtime",
        )
        for spec in MOL_TO_SMILES_OPTIONS
    }
    if bool(normalized["canonical"]) or not bool(normalized["do_random"]):
        raise NotImplementedError(
            "MolToSmiles runtime currently supports only canonical=False and "
            "doRandom=True; the public signatures keep RDKit-like defaults for "
            "surface compatibility, so pass those two flags explicitly."
        )
