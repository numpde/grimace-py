from __future__ import annotations

from dataclasses import dataclass

from tests.helpers.south_star_grammar_conformance import (
    SOUTH_STAR_GRAMMAR_CONFORMANCE_BASIS,
    south_star_grammar_conformance,
)
from tests.helpers.south_star_semantic_identity import (
    SOUTH_STAR_GRAPH_IDENTITY_BASIS,
    SOUTH_STAR_PARSER_DEPENDENCY_BASIS,
    SOUTH_STAR_STEREO_IDENTITY_BASIS,
    graph_signature,
    parse_smiles,
    semantic_signature,
    south_star_semantic_identity_report,
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
    grammar_conformance = south_star_grammar_conformance(candidate_smiles)
    identity_report = south_star_semantic_identity_report(
        source_smiles=source_smiles,
        candidate_smiles=candidate_smiles,
    )

    if not identity_report.parser_dependency.passed:
        return SouthStarConformanceReport(
            rdkit_parseability=SouthStarConformanceCheck(
                passed=False,
                basis=SOUTH_STAR_PARSER_DEPENDENCY_BASIS,
                detail=identity_report.parser_dependency.detail,
            ),
            graph_equivalence=SouthStarConformanceCheck(
                passed=False,
                basis=SOUTH_STAR_GRAPH_IDENTITY_BASIS,
                detail=identity_report.graph_identity.detail,
            ),
            stereo_equivalence=SouthStarConformanceCheck(
                passed=False,
                basis=SOUTH_STAR_STEREO_IDENTITY_BASIS,
                detail=identity_report.stereo_identity.detail,
            ),
            grammar_conformance=SouthStarConformanceCheck(
                passed=grammar_conformance.passed,
                basis=SOUTH_STAR_GRAMMAR_CONFORMANCE_BASIS,
                detail=grammar_conformance.detail,
            ),
        )

    return SouthStarConformanceReport(
        rdkit_parseability=SouthStarConformanceCheck(
            passed=True,
            basis=SOUTH_STAR_PARSER_DEPENDENCY_BASIS,
            detail=identity_report.parser_dependency.detail,
        ),
        graph_equivalence=SouthStarConformanceCheck(
            passed=identity_report.graph_identity.passed,
            basis=SOUTH_STAR_GRAPH_IDENTITY_BASIS,
            detail=identity_report.graph_identity.detail,
        ),
        stereo_equivalence=SouthStarConformanceCheck(
            passed=identity_report.stereo_identity.passed,
            basis=SOUTH_STAR_STEREO_IDENTITY_BASIS,
            detail=identity_report.stereo_identity.detail,
        ),
        grammar_conformance=SouthStarConformanceCheck(
            passed=grammar_conformance.passed,
            basis=SOUTH_STAR_GRAMMAR_CONFORMANCE_BASIS,
            detail=grammar_conformance.detail,
        ),
    )
