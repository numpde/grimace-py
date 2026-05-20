from __future__ import annotations

from dataclasses import dataclass

from grimace._south_star.annotation_policy import (
    DIRECTIONAL_MARKERS,
    AnnotationPolicy,
    AnnotationPolicyDecision,
    Edge,
    EmittedEdgeBasis,
    MaximalEligibleCarrierAnnotationPolicy,
    SurvivingSemanticAssignment,
    normalized_edge,
)
from tests.helpers.south_star_semantic_facts import (
    SouthStarSemanticFacts,
    prototype_surviving_assignments_from_case,
    south_star_semantic_facts_from_case,
)
from tests.helpers.south_star_semantics import SouthStarSemanticCase


@dataclass(frozen=True, slots=True)
class SouthStarDirectionalMarkerSupport:
    edge: Edge
    marker: str
    token_allowed: bool
    reason: str
    annotation_policy_decision: AnnotationPolicyDecision


@dataclass(frozen=True, slots=True)
class SouthStarOnlineSupportBoundary:
    semantic_facts: SouthStarSemanticFacts
    surviving_assignments: tuple[SurvivingSemanticAssignment, ...]
    annotation_policy: AnnotationPolicy

    @classmethod
    def from_case(
        cls,
        case: SouthStarSemanticCase,
        *,
        annotation_policy: AnnotationPolicy | None = None,
    ) -> SouthStarOnlineSupportBoundary:
        return cls(
            semantic_facts=south_star_semantic_facts_from_case(case),
            surviving_assignments=prototype_surviving_assignments_from_case(case),
            annotation_policy=(
                annotation_policy or MaximalEligibleCarrierAnnotationPolicy()
            ),
        )

    def explain_directional_marker(
        self,
        *,
        edge: Edge,
        marker: str,
    ) -> SouthStarDirectionalMarkerSupport:
        if marker not in DIRECTIONAL_MARKERS:
            raise ValueError(f"directional marker must be one of {DIRECTIONAL_MARKERS}")

        normalized = normalized_edge(edge)
        decision = self.annotation_policy.decision(
            carrier_opportunities=self.semantic_facts.carrier_opportunities,
            emitted_edge=EmittedEdgeBasis(edge=normalized),
            surviving_assignments=self.surviving_assignments,
        )
        if not decision.marker_required:
            return SouthStarDirectionalMarkerSupport(
                edge=normalized,
                marker=marker,
                token_allowed=False,
                reason="marker_not_required_by_annotation_policy",
                annotation_policy_decision=decision,
            )
        if marker not in decision.allowed_markers:
            return SouthStarDirectionalMarkerSupport(
                edge=normalized,
                marker=marker,
                token_allowed=False,
                reason="marker_rejected_by_surviving_semantic_assignments",
                annotation_policy_decision=decision,
            )
        return SouthStarDirectionalMarkerSupport(
            edge=normalized,
            marker=marker,
            token_allowed=True,
            reason="marker_allowed_by_annotation_policy",
            annotation_policy_decision=decision,
        )

    def allowed_directional_markers(self, *, edge: Edge) -> tuple[str, ...]:
        return tuple(
            marker
            for marker in DIRECTIONAL_MARKERS
            if self.explain_directional_marker(edge=edge, marker=marker).token_allowed
        )
