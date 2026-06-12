"""Public runtime input normalization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import grimace._prepared_mol as _prepared_mol
from grimace._mol_to_smiles_options import (
    MOL_TO_SMILES_OPTIONS,
    MOL_TO_SMILES_PREPARED_OPTIONS,
    coerce_internal_options,
    internal_option_values,
    normalize_root_atom,
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

    def __post_init__(self) -> None:
        for spec in MOL_TO_SMILES_OPTIONS:
            value = getattr(self, spec.internal_name)
            if spec.value_rule == "bool_like":
                if type(value) is not bool:
                    raise TypeError(
                        f"MolToSmilesFlags.{spec.internal_name} must be a bool"
                    )
            elif spec.value_rule == "root_atom":
                if type(value) is not int:
                    raise TypeError(
                        "MolToSmilesFlags.rooted_at_atom must be an int"
                    )
                normalized = normalize_root_atom(value)
                if normalized != value:
                    object.__setattr__(self, spec.internal_name, normalized)
            else:
                raise RuntimeError(
                    f"unsupported MolToSmiles option value_rule: {spec.value_rule!r}"
                )

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
    return _make_flags_from_internal_options(locals())


def _internal_option_kwargs(values: Mapping[str, object]) -> dict[str, object]:
    return coerce_internal_options(
        MOL_TO_SMILES_OPTIONS,
        internal_option_values(MOL_TO_SMILES_OPTIONS, values),
        context="MolToSmiles runtime",
    )


def _make_flags_from_internal_options(
    values: Mapping[str, object],
) -> MolToSmilesFlags:
    return MolToSmilesFlags(**_internal_option_kwargs(values))


def writer_flag_kwargs(flags: MolToSmilesFlags) -> dict[str, bool]:
    return {
        spec.internal_name: getattr(flags, spec.internal_name)
        for spec in MOL_TO_SMILES_PREPARED_OPTIONS
    }


def prepare_runtime_input(
    mol_or_prepared: object,
    *,
    flags: MolToSmilesFlags,
) -> object:
    _validate_supported_flags(flags)
    if isinstance(mol_or_prepared, _prepared_mol.PreparedMol):
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


def _runtime_public_writer_flag_kwargs(flags: MolToSmilesFlags) -> dict[str, bool]:
    return {
        spec.public_name: getattr(flags, spec.internal_name)
        for spec in MOL_TO_SMILES_PREPARED_OPTIONS
    }


def _validate_prepared_mol_writer_flags(
    prepared: _prepared_mol.PreparedMol,
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
    if flags.canonical or not flags.do_random:
        raise NotImplementedError(
            "MolToSmiles runtime currently supports only canonical=False and "
            "doRandom=True; the public signatures keep RDKit-like defaults for "
            "surface compatibility, so pass those two flags explicitly."
        )
