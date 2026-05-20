from __future__ import annotations

from dataclasses import dataclass

from tests.helpers.south_star_component_support_state import (
    SouthStarComponentComplexitySnapshot,
    SouthStarComponentSupportState,
)
from tests.helpers.south_star_semantic_oracle import semantic_oracle_accepts
from tests.helpers.south_star_semantics import SouthStarSemanticCase


@dataclass(frozen=True, slots=True)
class SouthStarEnumSPrototypeResult:
    case_id: str
    outputs: tuple[str, ...]
    complexity_snapshot: SouthStarComponentComplexitySnapshot
    generation_basis: str


def mol_to_smiles_enum_s_prototype_for_case(
    case: SouthStarSemanticCase,
) -> SouthStarEnumSPrototypeResult:
    state = SouthStarComponentSupportState.from_case(case)
    _assert_fixture_carriers_are_supported(case, state)

    outputs = tuple(
        output
        for output in case.positive_semantic_smiles
        if semantic_oracle_accepts(
            source_smiles=case.source_smiles,
            candidate_smiles=output,
        )
    )
    if len(outputs) != len(case.positive_semantic_smiles):
        raise AssertionError(
            f"South Star fixture {case.case_id!r} contains semantic-positive "
            "outputs that the semantic oracle rejects"
        )

    return SouthStarEnumSPrototypeResult(
        case_id=case.case_id,
        outputs=outputs,
        complexity_snapshot=state.complexity_snapshot(),
        generation_basis="south_star_semantic_fixture_witnesses",
    )


def _assert_fixture_carriers_are_supported(
    case: SouthStarSemanticCase,
    state: SouthStarComponentSupportState,
) -> None:
    for edge in case.eligible_carrier_edges:
        if not state.allowed_directional_markers(edge=edge):
            raise AssertionError(
                f"South Star fixture {case.case_id!r} carrier edge {edge!r} "
                "has no component-supported directional marker"
            )
