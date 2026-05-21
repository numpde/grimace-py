from __future__ import annotations

"""Test-only South Star semantic spec oracle skeleton.

This module is evidence, not generation authority. It deliberately consumes
candidate strings produced elsewhere and checks them against the declared South
Star grammar and graph/stereo semantic identity contract. Runtime/package code
must not import this helper, and tests must not treat parser/filter acceptance
as the mechanism by which EnumS support is generated.
"""

from dataclasses import dataclass

from tests.helpers.south_star_semantic_oracle import SouthStarConformanceReport
from tests.helpers.south_star_semantic_oracle import south_star_conformance_report


SOUTH_STAR_SPEC_ORACLE_BASIS = "test_only_south_star_semantic_spec_oracle"
SOUTH_STAR_SPEC_ORACLE_GENERATION_AUTHORITY = "not_generation_authority"


@dataclass(frozen=True, slots=True)
class SouthStarSpecOracleCandidateReport:
    candidate_smiles: str
    conformance_report: SouthStarConformanceReport

    @property
    def accepted(self) -> bool:
        return self.conformance_report.accepted

    @property
    def rejection_reasons(self) -> tuple[str, ...]:
        return self.conformance_report.rejection_reasons


@dataclass(frozen=True, slots=True)
class SouthStarSpecOracleReport:
    source_smiles: str
    candidate_reports: tuple[SouthStarSpecOracleCandidateReport, ...]
    basis: str = SOUTH_STAR_SPEC_ORACLE_BASIS
    generation_authority: str = SOUTH_STAR_SPEC_ORACLE_GENERATION_AUTHORITY

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_reports)

    @property
    def accepted_count(self) -> int:
        return sum(1 for report in self.candidate_reports if report.accepted)

    @property
    def all_accepted(self) -> bool:
        return self.accepted_count == self.candidate_count

    @property
    def rejected_candidates(self) -> tuple[SouthStarSpecOracleCandidateReport, ...]:
        return tuple(report for report in self.candidate_reports if not report.accepted)


def south_star_spec_oracle_report(
    *,
    source_smiles: str,
    candidate_smiles: tuple[str, ...],
) -> SouthStarSpecOracleReport:
    return SouthStarSpecOracleReport(
        source_smiles=source_smiles,
        candidate_reports=tuple(
            SouthStarSpecOracleCandidateReport(
                candidate_smiles=candidate,
                conformance_report=south_star_conformance_report(
                    source_smiles=source_smiles,
                    candidate_smiles=candidate,
                ),
            )
            for candidate in candidate_smiles
        ),
    )
