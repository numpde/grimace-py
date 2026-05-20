from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from grimace._south_star.enum_s import (
    SouthStarEnumSGenerationDiagnostics,
    _mol_to_smiles_enum_s_graph_native_for_mol,
)
from grimace._south_star.molecule_facts import SouthStarMoleculeFacts
from grimace._south_star.policies import DEFAULT_SOUTH_STAR_POLICY_SET
from tests.helpers.south_star_exact_support import SouthStarExpandedSupportCase
from tests.helpers.south_star_semantic_oracle import (
    parse_smiles,
    south_star_conformance_report,
)


@dataclass(frozen=True, slots=True)
class SouthStarComplexityTimingBreakdown:
    fact_extraction_seconds: float
    generation_seconds: float
    conformance_seconds: float


@dataclass(frozen=True, slots=True)
class SouthStarComplexityDiagnostic:
    case_id: str
    generation_diagnostics: SouthStarEnumSGenerationDiagnostics
    timing: SouthStarComplexityTimingBreakdown


def south_star_complexity_diagnostic_for_case(
    case: SouthStarExpandedSupportCase,
) -> SouthStarComplexityDiagnostic:
    """Return inspectable complexity/timing evidence for one South Star case.

    Timings are diagnostic metadata only. Tests may assert that the shape is
    populated and internally consistent, but must not assert wall-clock
    thresholds.
    """

    fact_start = perf_counter()
    mol = parse_smiles(case.source_smiles)
    molecule_facts = SouthStarMoleculeFacts.from_mol(mol)
    fact_extraction_seconds = perf_counter() - fact_start

    generation_start = perf_counter()
    result = _mol_to_smiles_enum_s_graph_native_for_mol(
        mol,
        case_id=case.case_id,
        policy_set=DEFAULT_SOUTH_STAR_POLICY_SET,
        molecule_facts=molecule_facts,
    )
    generation_seconds = perf_counter() - generation_start

    if result.generation_diagnostics is None:
        raise AssertionError("South Star complexity diagnostics require generation data")

    conformance_start = perf_counter()
    for output in result.outputs:
        south_star_conformance_report(
            source_smiles=case.source_smiles,
            candidate_smiles=output,
        )
    conformance_seconds = perf_counter() - conformance_start

    return SouthStarComplexityDiagnostic(
        case_id=case.case_id,
        generation_diagnostics=result.generation_diagnostics,
        timing=SouthStarComplexityTimingBreakdown(
            fact_extraction_seconds=fact_extraction_seconds,
            generation_seconds=generation_seconds,
            conformance_seconds=conformance_seconds,
        ),
    )
