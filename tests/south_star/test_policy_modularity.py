from __future__ import annotations

import unittest

from grimace._south_star.annotation_policy import (
    AnnotationPolicyDecision,
    EmittedEdgeBasis,
    SemanticCarrierOpportunity,
    SurvivingSemanticAssignment,
    normalized_edge,
)
from tests.helpers.south_star_semantics import load_south_star_semantic_cases
from tests.helpers.south_star_support_boundary import SouthStarOnlineSupportBoundary


class SlashOnlyPolicy:
    def decision(
        self,
        *,
        carrier_opportunities: tuple[SemanticCarrierOpportunity, ...],
        emitted_edge: EmittedEdgeBasis,
        surviving_assignments: tuple[SurvivingSemanticAssignment, ...],
    ) -> AnnotationPolicyDecision:
        del carrier_opportunities, surviving_assignments
        return AnnotationPolicyDecision(
            edge=normalized_edge(emitted_edge.edge),
            marker_required=True,
            allowed_markers=("/",),
        )


class NoMarkerPolicy:
    def decision(
        self,
        *,
        carrier_opportunities: tuple[SemanticCarrierOpportunity, ...],
        emitted_edge: EmittedEdgeBasis,
        surviving_assignments: tuple[SurvivingSemanticAssignment, ...],
    ) -> AnnotationPolicyDecision:
        del carrier_opportunities, surviving_assignments
        return AnnotationPolicyDecision(
            edge=normalized_edge(emitted_edge.edge),
            marker_required=False,
            allowed_markers=(),
        )


class SouthStarPolicyModularityTests(unittest.TestCase):
    def test_support_boundary_consumes_injected_policy_allowed_markers(self) -> None:
        for case in load_south_star_semantic_cases():
            boundary = SouthStarOnlineSupportBoundary.from_case(
                case,
                annotation_policy=SlashOnlyPolicy(),
            )
            edge = case.eligible_carrier_edges[0]

            with self.subTest(case_id=case.case_id):
                self.assertEqual(("/",), boundary.allowed_directional_markers(edge=edge))
                self.assertTrue(
                    boundary.explain_directional_marker(
                        edge=edge,
                        marker="/",
                    ).token_allowed
                )
                rejected = boundary.explain_directional_marker(
                    edge=edge,
                    marker="\\",
                )
                self.assertFalse(rejected.token_allowed)
                self.assertEqual(
                    "marker_rejected_by_surviving_semantic_assignments",
                    rejected.reason,
                )

    def test_support_boundary_consumes_injected_policy_requirement(self) -> None:
        for case in load_south_star_semantic_cases():
            boundary = SouthStarOnlineSupportBoundary.from_case(
                case,
                annotation_policy=NoMarkerPolicy(),
            )
            edge = case.eligible_carrier_edges[0]

            with self.subTest(case_id=case.case_id):
                self.assertEqual((), boundary.allowed_directional_markers(edge=edge))
                support = boundary.explain_directional_marker(edge=edge, marker="/")
                self.assertFalse(support.token_allowed)
                self.assertEqual(
                    "marker_not_required_by_annotation_policy",
                    support.reason,
                )
