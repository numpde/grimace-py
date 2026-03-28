"""Shared curated and dataset-backed case selectors."""

from __future__ import annotations

from rdkit import Chem

from smiles_next_token.reference import (
    load_default_molecule_cases,
    molecule_is_connected,
)


NONSTEREO_CURATED_ROOT_CASES: tuple[tuple[str, int], ...] = (
    ("Cc1ccccc1", 0),
    ("Cc1ccccc1", 1),
    ("C1CCCC=C1", 0),
    ("c1ccncc1", 0),
    ("O=[Ti]=O", 0),
    ("O=[Ti]=O", 1),
    ("C[Ge]", 0),
    ("C[Ge]", 1),
)

NONSTEREO_AWKWARD_CASES: tuple[str, ...] = (
    "O=[Ti]=O",
    "O=[Cr](=O)=O",
    "Cl[Fe]Cl",
    "F[Mg]",
    "C[Ge]",
    "C[Al]",
    "C[Cu]",
    "O=[Se]=O",
    "C[Si]",
    "O=[P]=O",
)

STEREO_CURATED_CASES: tuple[str, ...] = (
    "F[C@H](Cl)Br",
    "F[C@](Cl)(Br)I",
    "C[C@H](O)[C@@H](F)Cl",
    "F/C=C\\Cl",
    "F/C=C/C",
    "C(/C=C/Cl)Cl",
    "C(=C/Cl)\\Cl",
    "CC/C=C\\CCO",
    "C/C=C/C=O",
    "C/C=C/C=C/C(=O)O",
    "C/C=C(\\C)/C(=O)O",
    "C/C(=N\\\\OC(=O)NC)/SC",
    "F/C(Cl)=C/F",
)

STEREO_WALKER_CURATED_CASES: tuple[tuple[str, int], ...] = (
    ("F[C@H](Cl)Br", 0),
    ("F[C@](Cl)(Br)I", 0),
    ("C[C@H](O)[C@@H](F)Cl", 0),
    ("F/C=C\\Cl", 0),
    ("C/C=C/C=C/C(=O)O", 0),
    ("C/C=C(\\C)/C(=O)O", 0),
    ("C/C(=N\\\\OC(=O)NC)/SC", 0),
    ("F/C(Cl)=C/F", 0),
)

STEREO_ATOM_CURATED_CASES: tuple[str, ...] = (
    "F[C@H](Cl)Br",
    "F[C@](Cl)(Br)I",
    "C[C@H](O)[C@@H](F)Cl",
)

STEREO_BOND_CURATED_CASES: tuple[str, ...] = (
    "F/C=C\\Cl",
    "F/C=C/C",
    "C(/C=C/Cl)Cl",
    "C(=C/Cl)\\Cl",
    "CC/C=C\\CCO",
    "C/C=C/C=O",
)


def load_connected_atom_stereo_cases(*, limit: int, max_smiles_length: int) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    for case in load_default_molecule_cases(limit=5000, max_smiles_length=max_smiles_length):
        mol = Chem.MolFromSmiles(case.smiles)
        if mol is None or not molecule_is_connected(mol):
            continue
        if any(
            str(bond.GetStereo()) != "STEREONONE" or str(bond.GetBondDir()) != "NONE"
            for bond in mol.GetBonds()
        ):
            continue
        if not any(str(atom.GetChiralTag()) != "CHI_UNSPECIFIED" for atom in mol.GetAtoms()):
            continue
        selected.append((case.cid, case.smiles))
        if len(selected) >= limit:
            break
    return selected


def load_connected_multi_atom_stereo_cases(
    *,
    limit: int,
    max_smiles_length: int,
) -> list[tuple[str, str, int]]:
    selected: list[tuple[str, str, int]] = []
    for case in load_default_molecule_cases(limit=50000, max_smiles_length=max_smiles_length):
        mol = Chem.MolFromSmiles(case.smiles)
        if mol is None or not molecule_is_connected(mol):
            continue
        if any(
            str(bond.GetStereo()) != "STEREONONE" or str(bond.GetBondDir()) != "NONE"
            for bond in mol.GetBonds()
        ):
            continue
        chiral_count = sum(
            1 for atom in mol.GetAtoms() if str(atom.GetChiralTag()) != "CHI_UNSPECIFIED"
        )
        if chiral_count < 3:
            continue
        selected.append((case.cid, case.smiles, chiral_count))
        if len(selected) >= limit:
            break
    return selected


def load_connected_bond_stereo_cases(*, limit: int, max_smiles_length: int) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    for case in load_default_molecule_cases(limit=5000, max_smiles_length=max_smiles_length):
        mol = Chem.MolFromSmiles(case.smiles)
        if mol is None or not molecule_is_connected(mol):
            continue
        if any(atom.GetIsAromatic() for atom in mol.GetAtoms()):
            continue
        if any(bond.IsInRing() for bond in mol.GetBonds()):
            continue
        if any(str(atom.GetChiralTag()) != "CHI_UNSPECIFIED" for atom in mol.GetAtoms()):
            continue
        stereo_bonds = [bond for bond in mol.GetBonds() if str(bond.GetStereo()) != "STEREONONE"]
        if len(stereo_bonds) != 1:
            continue
        stereo_bond = stereo_bonds[0]
        if (
            stereo_bond.GetBeginAtom().GetAtomicNum() != 6
            or stereo_bond.GetEndAtom().GetAtomicNum() != 6
        ):
            continue
        begin_idx = stereo_bond.GetBeginAtomIdx()
        end_idx = stereo_bond.GetEndAtomIdx()
        begin_single_substituents = sum(
            1
            for neighbor in mol.GetAtomWithIdx(begin_idx).GetNeighbors()
            if neighbor.GetIdx() != end_idx
            and mol.GetBondBetweenAtoms(begin_idx, neighbor.GetIdx()).GetBondType()
            == Chem.BondType.SINGLE
        )
        end_single_substituents = sum(
            1
            for neighbor in mol.GetAtomWithIdx(end_idx).GetNeighbors()
            if neighbor.GetIdx() != begin_idx
            and mol.GetBondBetweenAtoms(end_idx, neighbor.GetIdx()).GetBondType()
            == Chem.BondType.SINGLE
        )
        if begin_single_substituents > 1 or end_single_substituents > 1:
            continue
        selected.append((case.cid, case.smiles))
        if len(selected) >= limit:
            break
    return selected
