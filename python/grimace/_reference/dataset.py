from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TextIO

from rdkit import Chem

from grimace._reference._paths import DEFAULT_MOLECULE_SOURCE_PATH

_REQUIRED_COLUMNS = ("CID", "iupac_name", "SMILES")


@dataclass(frozen=True)
class MoleculeCase:
    cid: str
    name: str
    smiles: str


def molecule_is_connected(mol: Chem.Mol) -> bool:
    return mol.GetNumAtoms() == 0 or len(Chem.GetMolFrags(mol)) == 1


def molecule_has_stereochemistry(mol: Chem.Mol) -> bool:
    if any(
        atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        for atom in mol.GetAtoms()
    ):
        return True
    if any(
        bond.GetStereo() != Chem.BondStereo.STEREONONE
        or bond.GetBondDir() != Chem.BondDir.NONE
        for bond in mol.GetBonds()
    ):
        return True
    return bool(mol.GetStereoGroups())


def _iter_tsv_rows(handle: TextIO) -> Iterator[dict[str, str]]:
    reader = csv.DictReader(handle, delimiter="\t")
    fieldnames = tuple(reader.fieldnames or ())
    missing = tuple(column for column in _REQUIRED_COLUMNS if column not in fieldnames)
    if missing:
        raise ValueError(
            f"Molecule fixture lacks required column(s): {', '.join(missing)}"
        )
    for row_number, row in enumerate(reader, start=2):
        if row.get(None):
            raise ValueError(f"Molecule fixture row {row_number} has too many columns")
        yield row


def iter_molecule_cases(
    path: str | Path,
    *,
    limit: int | None = None,
    max_smiles_length: int | None = None,
) -> Iterator[MoleculeCase]:
    source_path = Path(path)
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None")
    if max_smiles_length is not None and max_smiles_length < 0:
        raise ValueError("max_smiles_length must be non-negative or None")

    with gzip.open(source_path, "rt", encoding="utf-8", newline="") as handle:
        yielded = 0
        for row in _iter_tsv_rows(handle):
            case = MoleculeCase(
                cid=row["CID"],
                name=row["iupac_name"],
                smiles=row["SMILES"],
            )
            if max_smiles_length is not None and len(case.smiles) > max_smiles_length:
                continue
            if limit is not None and yielded >= limit:
                break
            yield case
            yielded += 1


def load_molecule_cases(
    path: str | Path,
    *,
    limit: int | None = None,
    max_smiles_length: int | None = None,
) -> list[MoleculeCase]:
    return list(iter_molecule_cases(path, limit=limit, max_smiles_length=max_smiles_length))


def iter_default_molecule_cases(
    *,
    limit: int | None = None,
    max_smiles_length: int | None = None,
) -> Iterator[MoleculeCase]:
    return iter_molecule_cases(
        DEFAULT_MOLECULE_SOURCE_PATH,
        limit=limit,
        max_smiles_length=max_smiles_length,
    )


def load_default_molecule_cases(
    *,
    limit: int | None = None,
    max_smiles_length: int | None = None,
) -> list[MoleculeCase]:
    return load_molecule_cases(
        DEFAULT_MOLECULE_SOURCE_PATH,
        limit=limit,
        max_smiles_length=max_smiles_length,
    )


def iter_default_connected_nonstereo_molecule_cases(
    *,
    limit: int | None = None,
    max_smiles_length: int | None = None,
) -> Iterator[MoleculeCase]:
    yielded = 0
    for case in iter_default_molecule_cases(max_smiles_length=max_smiles_length):
        mol = Chem.MolFromSmiles(case.smiles)
        if mol is None:
            continue
        if not molecule_is_connected(mol):
            continue
        if molecule_has_stereochemistry(mol):
            continue

        if limit is not None and yielded >= limit:
            break
        yield case
        yielded += 1


def load_default_connected_nonstereo_molecule_cases(
    *,
    limit: int | None = None,
    max_smiles_length: int | None = None,
) -> list[MoleculeCase]:
    return list(
        iter_default_connected_nonstereo_molecule_cases(
            limit=limit,
            max_smiles_length=max_smiles_length,
        )
    )
