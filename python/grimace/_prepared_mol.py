"""Prepared molecule wrapper and RDKit preparation boundary."""

from __future__ import annotations

import importlib
from numbers import Integral

from rdkit import Chem


_PREPARED_MOL_SCHEMA_VERSION = 1


def _core_module() -> object:
    return importlib.import_module("grimace._core")


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
        return _make_prepared_mol(_core_module().PreparedMol.from_bytes(data))


def _make_prepared_mol(inner: object) -> PreparedMol:
    prepared = object.__new__(PreparedMol)
    object.__setattr__(prepared, "_inner", inner)
    return prepared


def _prepared_mol_writer_flag_values(
    prepared: PreparedMol,
) -> tuple[bool, bool, bool, bool, bool]:
    return prepared._inner.writer_flag_values()


def _prepared_mol_fragment_count(prepared: PreparedMol) -> int:
    return prepared._inner.fragment_count()


def _prepared_mol_fragment_atom_indices(
    prepared: PreparedMol,
    fragment_idx: int,
) -> tuple[int, ...]:
    return tuple(prepared._inner.fragment_atom_indices(fragment_idx))


def _prepared_mol_fragment_graph(prepared: PreparedMol, fragment_idx: int) -> object:
    return prepared._inner.fragment_prepared_graph(fragment_idx)


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

    writer_isomeric_smiles = _coerce_bool_flag("isomericSmiles", isomericSmiles)
    writer_kekule_smiles = _coerce_bool_flag("kekuleSmiles", kekuleSmiles)
    writer_all_bonds_explicit = _coerce_bool_flag(
        "allBondsExplicit",
        allBondsExplicit,
    )
    writer_all_hs_explicit = _coerce_bool_flag("allHsExplicit", allHsExplicit)
    writer_ignore_atom_map_numbers = _coerce_bool_flag(
        "ignoreAtomMapNumbers",
        ignoreAtomMapNumbers,
    )

    runtime = importlib.import_module("grimace._runtime")
    runtime_flags = runtime.MolToSmilesFlags(
        isomeric_smiles=writer_isomeric_smiles,
        kekule_smiles=writer_kekule_smiles,
        all_bonds_explicit=writer_all_bonds_explicit,
        all_hs_explicit=writer_all_hs_explicit,
        ignore_atom_map_numbers=writer_ignore_atom_map_numbers,
    )

    if mol.GetNumAtoms() == 0:
        fragments = [
            {
                "atom_indices": [],
                "prepared_graph": runtime.prepare_smiles_graph(mol, flags=runtime_flags),
            }
        ]
    else:
        fragments = [
            {
                "atom_indices": list(atom_indices),
                "prepared_graph": runtime.prepare_smiles_graph(
                    fragment_mol,
                    flags=runtime_flags,
                ),
            }
            for fragment_mol, atom_indices in _rdkit_mol_fragment_mols_and_atom_indices(
                mol
            )
        ]

    payload = {
        "schema_version": _PREPARED_MOL_SCHEMA_VERSION,
        "writer_flags": {
            "isomeric_smiles": writer_isomeric_smiles,
            "kekule_smiles": writer_kekule_smiles,
            "all_bonds_explicit": writer_all_bonds_explicit,
            "all_hs_explicit": writer_all_hs_explicit,
            "ignore_atom_map_numbers": writer_ignore_atom_map_numbers,
        },
        "fragments": fragments,
    }
    return _make_prepared_mol(_core_module().PreparedMol(payload))


def _coerce_bool_flag(name: str, value: object) -> bool:
    if value is None:
        return False
    if not isinstance(value, Integral):
        raise TypeError(
            f"PrepareMol requires {name} to follow RDKit's Python binding "
            "and be a bool, int, or None"
        )
    return bool(value)
