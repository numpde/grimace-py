from __future__ import annotations

from dataclasses import dataclass

from tests.helpers.south_star_enum_s import SouthStarEnumSPrototypeResult
from tests.helpers.south_star_semantic_oracle import (
    SouthStarConformanceReport,
    south_star_conformance_report,
)
from tests.helpers.south_star_semantics import SouthStarSemanticCase


@dataclass(frozen=True, slots=True)
class SouthStarOutputCorrectnessResult:
    case_id: str
    output: str
    report: SouthStarConformanceReport


def evaluate_south_star_output_correctness(
    *,
    case: SouthStarSemanticCase,
    enum_result: SouthStarEnumSPrototypeResult,
) -> tuple[SouthStarOutputCorrectnessResult, ...]:
    if enum_result.case_id != case.case_id:
        raise ValueError(
            f"EnumS result case {enum_result.case_id!r} does not match "
            f"semantic case {case.case_id!r}"
        )

    return tuple(
        SouthStarOutputCorrectnessResult(
            case_id=case.case_id,
            output=output,
            report=south_star_conformance_report(
                source_smiles=case.source_smiles,
                candidate_smiles=output,
            ),
        )
        for output in enum_result.outputs
    )
