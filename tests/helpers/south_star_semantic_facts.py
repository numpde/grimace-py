from __future__ import annotations

from dataclasses import dataclass

from tests.helpers.south_star_annotation_policy import (
    DIRECTIONAL_MARKERS,
    SemanticCarrierOpportunity,
    SurvivingSemanticAssignment,
    normalized_edge,
)
from tests.helpers.south_star_semantics import SouthStarSemanticCase


@dataclass(frozen=True, slots=True)
class SouthStarSemanticFacts:
    case_id: str
    source_smiles: str
    carrier_opportunities: tuple[SemanticCarrierOpportunity, ...]


def south_star_semantic_facts_from_case(
    case: SouthStarSemanticCase,
) -> SouthStarSemanticFacts:
    return SouthStarSemanticFacts(
        case_id=case.case_id,
        source_smiles=case.source_smiles,
        carrier_opportunities=tuple(
            SemanticCarrierOpportunity(edge=edge)
            for edge in case.eligible_carrier_edges
        ),
    )


def prototype_surviving_assignments_from_case(
    case: SouthStarSemanticCase,
) -> tuple[SurvivingSemanticAssignment, ...]:
    return (
        SurvivingSemanticAssignment(
            assignment_id=f"{case.case_id}:prototype-all-marker-survivor",
            marker_options_by_edge={
                normalized_edge(edge): DIRECTIONAL_MARKERS
                for edge in case.eligible_carrier_edges
            },
        ),
    )
