from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit import rdBase

from tests.helpers.south_star_grammar_conformance import (
    SOUTH_STAR_GRAMMAR_CONFORMANCE_BASIS,
    south_star_grammar_conformance,
)


@dataclass(frozen=True, slots=True)
class SouthStarConformanceCheck:
    passed: bool
    basis: str
    detail: str


@dataclass(frozen=True, slots=True)
class SouthStarConformanceReport:
    rdkit_parseability: SouthStarConformanceCheck
    graph_equivalence: SouthStarConformanceCheck
    stereo_equivalence: SouthStarConformanceCheck
    grammar_conformance: SouthStarConformanceCheck

    @property
    def annotation_conformance(self) -> SouthStarConformanceCheck:
        return self.grammar_conformance

    @property
    def accepted(self) -> bool:
        return all(
            (
                self.rdkit_parseability.passed,
                self.graph_equivalence.passed,
                self.stereo_equivalence.passed,
                self.grammar_conformance.passed,
            )
        )

    @property
    def rejection_reasons(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, check in (
                ("rdkit_parseability", self.rdkit_parseability),
                ("graph_equivalence", self.graph_equivalence),
                ("stereo_equivalence", self.stereo_equivalence),
                ("grammar_conformance", self.grammar_conformance),
            )
            if not check.passed
        )


def parse_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise AssertionError(f"failed to parse SMILES {smiles!r}")
    return mol


def graph_signature(smiles: str) -> str:
    return _graph_signature_for_mol(parse_smiles(smiles))


def semantic_signature(smiles: str) -> str:
    return _semantic_signature_for_mol(parse_smiles(smiles))


def semantic_oracle_accepts(*, source_smiles: str, candidate_smiles: str) -> bool:
    return south_star_conformance_report(
        source_smiles=source_smiles,
        candidate_smiles=candidate_smiles,
    ).accepted


def south_star_conformance_report(
    *,
    source_smiles: str,
    candidate_smiles: str,
) -> SouthStarConformanceReport:
    source_mol = parse_smiles(source_smiles)
    grammar_conformance = south_star_grammar_conformance(candidate_smiles)
    candidate_mol = _try_parse_smiles(candidate_smiles)

    if candidate_mol is None:
        failed = SouthStarConformanceCheck(
            passed=False,
            basis="rdkit_parser",
            detail="candidate SMILES did not parse",
        )
        return SouthStarConformanceReport(
            rdkit_parseability=failed,
            graph_equivalence=SouthStarConformanceCheck(
                passed=False,
                basis="canonical_nonisomeric_smiles",
                detail="candidate graph is unavailable because parsing failed",
            ),
            stereo_equivalence=SouthStarConformanceCheck(
                passed=False,
                basis="canonical_isomeric_smiles",
                detail="candidate stereo is unavailable because parsing failed",
            ),
            grammar_conformance=SouthStarConformanceCheck(
                passed=grammar_conformance.passed,
                basis=SOUTH_STAR_GRAMMAR_CONFORMANCE_BASIS,
                detail=grammar_conformance.detail,
            ),
        )

    source_graph = _graph_signature_for_mol(source_mol)
    candidate_graph = _graph_signature_for_mol(candidate_mol)
    source_semantics = _semantic_signature_for_mol(source_mol)
    candidate_semantics = _semantic_signature_for_mol(candidate_mol)
    return SouthStarConformanceReport(
        rdkit_parseability=SouthStarConformanceCheck(
            passed=True,
            basis="rdkit_parser",
            detail="candidate SMILES parsed",
        ),
        graph_equivalence=SouthStarConformanceCheck(
            passed=source_graph == candidate_graph,
            basis="canonical_nonisomeric_smiles",
            detail=(
                "candidate graph matches source"
                if source_graph == candidate_graph
                else "candidate graph differs from source"
            ),
        ),
        stereo_equivalence=SouthStarConformanceCheck(
            passed=source_semantics == candidate_semantics,
            basis="canonical_isomeric_smiles",
            detail=(
                "candidate stereo semantics match source"
                if source_semantics == candidate_semantics
                else "candidate stereo semantics differ from source"
            ),
        ),
        grammar_conformance=SouthStarConformanceCheck(
            passed=grammar_conformance.passed,
            basis=SOUTH_STAR_GRAMMAR_CONFORMANCE_BASIS,
            detail=grammar_conformance.detail,
        ),
    )


def _try_parse_smiles(smiles: str) -> Chem.Mol | None:
    with rdBase.BlockLogs():
        return Chem.MolFromSmiles(smiles)


def _graph_signature_for_mol(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=False,
    )


def _semantic_signature_for_mol(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True,
    )
