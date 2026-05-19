from __future__ import annotations

from dataclasses import dataclass

from tests.helpers.south_star_annotation_policy import (
    DIRECTIONAL_MARKERS,
    AnnotationPolicy,
    AnnotationPolicyDecision,
    EmittedEdgeBasis,
    MaximalEligibleCarrierAnnotationPolicy,
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


@dataclass(frozen=True, slots=True)
class SouthStarSemanticDiagnostic:
    semantic_facts: SouthStarSemanticFacts
    annotation_policy_decisions: tuple[AnnotationPolicyDecision, ...]


def south_star_semantic_diagnostic(
    case: SouthStarSemanticCase,
    *,
    annotation_policy: AnnotationPolicy | None = None,
) -> SouthStarSemanticDiagnostic:
    policy = annotation_policy or MaximalEligibleCarrierAnnotationPolicy()
    semantic_facts = SouthStarSemanticFacts(
        case_id=case.case_id,
        source_smiles=case.source_smiles,
        carrier_opportunities=tuple(
            SemanticCarrierOpportunity(edge=edge)
            for edge in case.eligible_carrier_edges
        ),
    )
    survivor = SurvivingSemanticAssignment(
        assignment_id=f"{case.case_id}:fixture-survivor",
        marker_options_by_edge={
            normalized_edge(edge): DIRECTIONAL_MARKERS
            for edge in case.eligible_carrier_edges
        },
    )
    decisions = tuple(
        policy.decision(
            carrier_opportunities=semantic_facts.carrier_opportunities,
            emitted_edge=EmittedEdgeBasis(edge=edge),
            surviving_assignments=(survivor,),
        )
        for edge in case.eligible_carrier_edges
    )
    return SouthStarSemanticDiagnostic(
        semantic_facts=semantic_facts,
        annotation_policy_decisions=decisions,
    )
