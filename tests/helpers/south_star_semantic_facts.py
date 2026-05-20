from __future__ import annotations

from dataclasses import dataclass

from grimace._south_star.annotation_policy import (
    DIRECTIONAL_MARKERS,
    SemanticCarrierOpportunity,
    SurvivingSemanticAssignment,
    normalized_edge,
)
from grimace._south_star.components import (
    SouthStarSemanticStereoComponent,
    extract_south_star_components,
)
from tests.helpers.south_star_semantic_oracle import parse_smiles
from tests.helpers.south_star_semantics import SouthStarSemanticCase


@dataclass(frozen=True, slots=True)
class SouthStarSemanticFacts:
    case_id: str
    source_smiles: str
    components: tuple[SouthStarSemanticStereoComponent, ...]
    carrier_opportunities: tuple[SemanticCarrierOpportunity, ...]


def south_star_semantic_facts_from_case(
    case: SouthStarSemanticCase,
) -> SouthStarSemanticFacts:
    extraction = extract_south_star_components(parse_smiles(case.source_smiles))
    extraction.fail_if_unsupported()
    carrier_edges = tuple(
        dict.fromkeys(
            edge
            for component in extraction.components
            for edge in component.eligible_carrier_edges
        )
    )
    return SouthStarSemanticFacts(
        case_id=case.case_id,
        source_smiles=case.source_smiles,
        components=extraction.components,
        carrier_opportunities=tuple(
            SemanticCarrierOpportunity(edge=edge)
            for edge in carrier_edges
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
