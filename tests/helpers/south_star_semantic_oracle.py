from __future__ import annotations

from rdkit import Chem


def parse_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise AssertionError(f"failed to parse SMILES {smiles!r}")
    return mol


def graph_signature(smiles: str) -> str:
    return Chem.MolToSmiles(
        parse_smiles(smiles),
        canonical=True,
        isomericSmiles=False,
    )


def semantic_signature(smiles: str) -> str:
    return Chem.MolToSmiles(
        parse_smiles(smiles),
        canonical=True,
        isomericSmiles=True,
    )


def semantic_oracle_accepts(*, source_smiles: str, candidate_smiles: str) -> bool:
    return (
        graph_signature(source_smiles) == graph_signature(candidate_smiles)
        and semantic_signature(source_smiles) == semantic_signature(candidate_smiles)
    )
