from __future__ import annotations

from dataclasses import dataclass

from tests.helpers.south_star_exact_support import (
    load_south_star_exact_first_domain_cases,
    load_south_star_expanded_support_cases,
)


SOUTH_STAR_ENUM_S_BENCHMARK_POLICY_SET: dict[str, str] = {
    "annotation_policy": "maximal_eligible_carrier",
    "fragment_order_policy": "all_fragment_orders",
    "output_order_policy": "first_occurrence_deduplication",
}
SOUTH_STAR_ENUM_S_BENCHMARK_SCOPE_NOTE = (
    "Measures the private South Star semantic enumerator "
    "mol_to_smiles_enum_s_graph_native on pinned semantic fixtures. "
    "This is not an RDKit writer-parity benchmark."
)
SOUTH_STAR_ENUM_S_CASE_MANIFEST_SCOPE_NOTE = (
    "Deterministic case manifest for the private South Star semantic "
    "enumerator benchmark. This records fixture coverage only; timing reports "
    "are point-in-time snapshots and may cover any subset of these cases."
)


@dataclass(frozen=True, slots=True)
class SouthStarEnumSBenchmarkCase:
    case_id: str
    fixture_family: str
    domain_label: str
    source_smiles: str
    expected_output_count: int


def south_star_enum_s_benchmark_cases() -> tuple[SouthStarEnumSBenchmarkCase, ...]:
    return tuple(
        SouthStarEnumSBenchmarkCase(
            case_id=case.case_id,
            fixture_family="exact_first_domain",
            domain_label="first_domain_directional_bond_stereo",
            source_smiles=case.source_smiles,
            expected_output_count=len(case.expected_support),
        )
        for case in load_south_star_exact_first_domain_cases()
    ) + tuple(
        SouthStarEnumSBenchmarkCase(
            case_id=case.case_id,
            fixture_family="expanded_support",
            domain_label=case.feature_area,
            source_smiles=case.source_smiles,
            expected_output_count=len(case.expected_support),
        )
        for case in load_south_star_expanded_support_cases()
    )
