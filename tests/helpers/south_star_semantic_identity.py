from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from rdkit import Chem
from rdkit import rdBase


SOUTH_STAR_PARSER_DEPENDENCY_BASIS = "rdkit_parser_dependency"
SOUTH_STAR_GRAPH_IDENTITY_BASIS = "rdkit_canonical_nonisomeric_parseback"
SOUTH_STAR_STEREO_IDENTITY_BASIS = "rdkit_canonical_isomeric_parseback"


@dataclass(frozen=True, slots=True)
class SouthStarSemanticIdentityCheck:
    passed: bool
    basis: str
    detail: str


@dataclass(frozen=True, slots=True)
class SouthStarSemanticIdentityReport:
    parser_dependency: SouthStarSemanticIdentityCheck
    graph_identity: SouthStarSemanticIdentityCheck
    stereo_identity: SouthStarSemanticIdentityCheck

    @property
    def accepted(self) -> bool:
        return (
            self.parser_dependency.passed
            and self.graph_identity.passed
            and self.stereo_identity.passed
        )


def south_star_semantic_identity_report(
    *,
    source_smiles: str,
    candidate_smiles: str,
) -> SouthStarSemanticIdentityReport:
    source_mol = parse_smiles(source_smiles)
    candidate_mol = _try_parse_smiles(candidate_smiles)
    if candidate_mol is None:
        parser_failure = SouthStarSemanticIdentityCheck(
            passed=False,
            basis=SOUTH_STAR_PARSER_DEPENDENCY_BASIS,
            detail="candidate SMILES did not parse through the current parser",
        )
        return SouthStarSemanticIdentityReport(
            parser_dependency=parser_failure,
            graph_identity=SouthStarSemanticIdentityCheck(
                passed=False,
                basis=SOUTH_STAR_GRAPH_IDENTITY_BASIS,
                detail="candidate graph identity is unavailable because parsing failed",
            ),
            stereo_identity=SouthStarSemanticIdentityCheck(
                passed=False,
                basis=SOUTH_STAR_STEREO_IDENTITY_BASIS,
                detail=(
                    "candidate stereo identity is unavailable because parsing failed"
                ),
            ),
        )

    source_graph = graph_signature_for_mol(source_mol)
    candidate_graph = graph_signature_for_mol(candidate_mol)
    source_semantics = semantic_signature_for_mol(source_mol)
    candidate_semantics = semantic_signature_for_mol(candidate_mol)
    return SouthStarSemanticIdentityReport(
        parser_dependency=SouthStarSemanticIdentityCheck(
            passed=True,
            basis=SOUTH_STAR_PARSER_DEPENDENCY_BASIS,
            detail="candidate SMILES parsed through the current parser",
        ),
        graph_identity=SouthStarSemanticIdentityCheck(
            passed=source_graph == candidate_graph,
            basis=SOUTH_STAR_GRAPH_IDENTITY_BASIS,
            detail=(
                "candidate graph matches source"
                if source_graph == candidate_graph
                else "candidate graph differs from source"
            ),
        ),
        stereo_identity=SouthStarSemanticIdentityCheck(
            passed=source_semantics == candidate_semantics,
            basis=SOUTH_STAR_STEREO_IDENTITY_BASIS,
            detail=(
                "candidate stereo semantics match source"
                if source_semantics == candidate_semantics
                else "candidate stereo semantics differ from source"
            ),
        ),
    )


def parse_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise AssertionError(f"failed to parse SMILES {smiles!r}")
    return mol


@lru_cache(maxsize=None)
def graph_signature(smiles: str) -> str:
    return graph_signature_for_mol(parse_smiles(smiles))


@lru_cache(maxsize=None)
def semantic_signature(smiles: str) -> str:
    return semantic_signature_for_mol(parse_smiles(smiles))


def graph_signature_for_mol(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=False,
    )


def semantic_signature_for_mol(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True,
    )


def _try_parse_smiles(smiles: str) -> Chem.Mol | None:
    with rdBase.BlockLogs():
        return Chem.MolFromSmiles(smiles)
