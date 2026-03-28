"""Helpers for parsing test molecules."""

from __future__ import annotations

from rdkit import Chem


def parse_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise AssertionError(f"Failed to parse SMILES: {smiles!r}")
    return mol
