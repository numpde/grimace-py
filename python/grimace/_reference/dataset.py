from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from rdkit import Chem

from grimace._reference._paths import DEFAULT_MOLECULE_SOURCE_PATH, resolve_bundled_reference_path


@dataclass(frozen=True)
class MoleculeCase:
    cid: str
    name: str
    smiles: str


def molecule_is_connected(mol: Chem.Mol) -> bool:
    return mol.GetNumAtoms() == 0 or len(Chem.GetMolFrags(mol)) == 1


def molecule_has_stereochemistry(mol: Chem.Mol) -> bool:
    if any(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for atom in mol.GetAtoms()):
        return True
    if any(
        bond.GetStereo() != Chem.BondStereo.STEREONONE or bond.GetBondDir() != Chem.BondDir.NONE
        for bond in mol.GetBonds()
    ):
        return True
    return bool(mol.GetStereoGroups())


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
        reader = csv.DictReader(handle, delimiter="\t")
        yielded = 0
        for row in reader:
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


def _resolve_input_source_path(input_source: Mapping[str, Any]) -> Path:
    return resolve_bundled_reference_path(str(input_source["path"]))


def _input_source_filters(input_source: Mapping[str, Any]) -> Mapping[str, Any]:
    filters = input_source.get("filters", {})
    if not isinstance(filters, Mapping):
        raise TypeError("input_source.filters must be a JSON object")
    return filters


def iter_molecule_cases_from_input_source(
    input_source: Mapping[str, Any],
    *,
    limit: int | None = None,
    max_smiles_length: int | None = None,
) -> Iterator[MoleculeCase]:
    if not isinstance(input_source, Mapping):
        raise TypeError("input_source must be a JSON object")
    if input_source.get("kind") != "default_fixture":
        raise ValueError(f"Unsupported input_source kind: {input_source.get('kind')!r}")

    filters = _input_source_filters(input_source)
    connected_only = bool(filters.get("connected_only", False))
    stereochemistry = filters.get("stereochemistry", "allow")
    if stereochemistry not in {"allow", "forbid"}:
        raise ValueError(f"Unsupported input_source stereochemistry mode: {stereochemistry!r}")

    yielded = 0
    for case in iter_molecule_cases(
        _resolve_input_source_path(input_source),
        max_smiles_length=max_smiles_length,
    ):
        mol = None
        if connected_only or stereochemistry != "allow":
            mol = Chem.MolFromSmiles(case.smiles)
            if mol is None:
                continue
        if connected_only and mol is not None and not molecule_is_connected(mol):
            continue
        if stereochemistry == "forbid" and mol is not None and molecule_has_stereochemistry(mol):
            continue

        if limit is not None and yielded >= limit:
            break
        yield case
        yielded += 1


def load_molecule_cases_from_input_source(
    input_source: Mapping[str, Any],
    *,
    limit: int | None = None,
    max_smiles_length: int | None = None,
) -> list[MoleculeCase]:
    return list(
        iter_molecule_cases_from_input_source(
            input_source,
            limit=limit,
            max_smiles_length=max_smiles_length,
        )
    )


def iter_default_connected_nonstereo_molecule_cases(
    *,
    limit: int | None = None,
    max_smiles_length: int | None = None,
) -> Iterator[MoleculeCase]:
    return iter_molecule_cases_from_input_source(
        {
            "kind": "default_fixture",
            "path": str(DEFAULT_MOLECULE_SOURCE_PATH.name),
            "filters": {
                "connected_only": True,
                "stereochemistry": "forbid",
            },
        },
        limit=limit,
        max_smiles_length=max_smiles_length,
    )


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
