"""Prepared molecule wrapper and RDKit preparation boundary."""

from __future__ import annotations

from rdkit import Chem

import grimace._core as _core
from grimace._mol_to_smiles_options import (
    MOL_TO_SMILES_PREPARED_OPTIONS,
    coerce_public_options,
)


class PreparedMol:
    """Opaque prepared molecule returned by PrepareMol."""

    __slots__ = ("_inner",)

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError(
            "PreparedMol cannot be constructed directly; use PrepareMol or "
            "PreparedMol.from_bytes"
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("PreparedMol is immutable")

    def to_bytes(self) -> bytes:
        return self._inner.to_bytes()

    @staticmethod
    def from_bytes(data: bytes) -> "PreparedMol":
        if not isinstance(data, bytes):
            raise TypeError("PreparedMol.from_bytes requires bytes")
        return _make_prepared_mol(_core.PreparedMol.from_bytes(data))


def _make_prepared_mol(inner: object) -> PreparedMol:
    prepared = object.__new__(PreparedMol)
    object.__setattr__(prepared, "_inner", inner)
    return prepared


def _matches_writer_flags(
    prepared: PreparedMol,
    *,
    isomeric_smiles: bool,
    kekule_smiles: bool,
    all_bonds_explicit: bool,
    all_hs_explicit: bool,
    ignore_atom_map_numbers: bool,
) -> bool:
    return prepared._inner.matches_writer_flags(
        isomeric_smiles=isomeric_smiles,
        kekule_smiles=kekule_smiles,
        all_bonds_explicit=all_bonds_explicit,
        all_hs_explicit=all_hs_explicit,
        ignore_atom_map_numbers=ignore_atom_map_numbers,
    )


def _fragment_count(prepared: PreparedMol) -> int:
    return prepared._inner.fragment_count()


def _atom_count(prepared: PreparedMol) -> int:
    return prepared._inner.atom_count()


def _rooted_fragments(
    prepared: PreparedMol,
    *,
    rooted_at_atom: int | None,
) -> tuple[tuple[object, int | None], ...]:
    return tuple(prepared._inner.rooted_fragments(rooted_at_atom))


def _is_rdkit_mol(value: object) -> bool:
    return isinstance(value, Chem.Mol)


def _rdkit_mol_requires_stereo_surface(mol: Chem.Mol) -> bool:
    return any(
        bond.GetStereo() != Chem.BondStereo.STEREONONE
        or bond.GetBondDir() != Chem.BondDir.NONE
        for bond in mol.GetBonds()
    )


def _rdkit_mol_atom_count(mol: Chem.Mol) -> int:
    return mol.GetNumAtoms()


def _rdkit_mol_fragment_count(mol: Chem.Mol) -> int:
    return len(Chem.GetMolFrags(mol))


def _rdkit_mol_fragment_mols_and_atom_indices(
    mol: Chem.Mol,
) -> tuple[tuple[Chem.Mol, tuple[int, ...]], ...]:
    atom_indices_by_fragment: list[tuple[int, ...]] = []
    fragment_mols = Chem.GetMolFrags(
        mol,
        asMols=True,
        sanitizeFrags=False,
        fragsMolAtomMapping=atom_indices_by_fragment,
    )
    return tuple(zip(fragment_mols, atom_indices_by_fragment, strict=True))


def PrepareMol(
    mol: Chem.Mol,
    *,
    isomericSmiles: bool = True,
    kekuleSmiles: bool = False,
    allBondsExplicit: bool = False,
    allHsExplicit: bool = False,
    ignoreAtomMapNumbers: bool = False,
) -> PreparedMol:
    if not isinstance(mol, Chem.Mol):
        raise TypeError("PrepareMol requires an RDKit Chem.Mol")

    writer_options = coerce_public_options(
        MOL_TO_SMILES_PREPARED_OPTIONS,
        locals(),
        context="PrepareMol",
    )

    import grimace._runtime_graphs as runtime_graphs
    import grimace._runtime_inputs as runtime_inputs
    runtime_flags = runtime_inputs.MolToSmilesFlags(**writer_options)

    if mol.GetNumAtoms() == 0:
        fragments = [
            (
                [],
                runtime_graphs.prepare_smiles_graph(mol, flags=runtime_flags),
            )
        ]
    else:
        fragments = [
            (
                list(atom_indices),
                runtime_graphs.prepare_smiles_graph(
                    fragment_mol,
                    flags=runtime_flags,
                ),
            )
            for fragment_mol, atom_indices in _rdkit_mol_fragment_mols_and_atom_indices(
                mol
            )
        ]

    return _make_prepared_mol(
        _core.PreparedMol._from_parts(
            **writer_options,
            fragments=fragments,
        )
    )
