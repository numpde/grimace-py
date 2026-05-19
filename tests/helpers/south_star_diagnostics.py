from __future__ import annotations

from dataclasses import dataclass

from tests.helpers.south_star_annotation_policy import (
    AnnotationPolicy,
    AnnotationPolicyDecision,
    EmittedEdgeBasis,
    MaximalEligibleCarrierAnnotationPolicy,
)
from tests.helpers.south_star_semantic_facts import (
    SouthStarSemanticFacts,
    prototype_surviving_assignments_from_case,
    south_star_semantic_facts_from_case,
)
from tests.helpers.south_star_semantics import SouthStarSemanticCase


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
    semantic_facts = south_star_semantic_facts_from_case(case)
    surviving_assignments = prototype_surviving_assignments_from_case(case)
    decisions = tuple(
        policy.decision(
            carrier_opportunities=semantic_facts.carrier_opportunities,
            emitted_edge=EmittedEdgeBasis(edge=edge),
            surviving_assignments=surviving_assignments,
        )
        for edge in case.eligible_carrier_edges
    )
    return SouthStarSemanticDiagnostic(
        semantic_facts=semantic_facts,
        annotation_policy_decisions=decisions,
    )
