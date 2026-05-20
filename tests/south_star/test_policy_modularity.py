from __future__ import annotations

import unittest

from grimace._south_star.annotation_policy import (
    AnnotationPolicyDecision,
    EmittedEdgeBasis,
    NoMarkerAnnotationPolicyStub,
    SOUTH_STAR_ANNOTATION_POLICY_CANDIDATES,
    SemanticCarrierOpportunity,
    SurvivingSemanticAssignment,
    normalized_edge,
)
from grimace._south_star.policies import DEFAULT_SOUTH_STAR_POLICY_SET
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


class SouthStarPolicyModularityTests(unittest.TestCase):
    def test_maximal_policy_remains_default(self) -> None:
        self.assertEqual(
            "maximal_eligible_carrier",
            DEFAULT_SOUTH_STAR_POLICY_SET.annotation_policy.name,
        )

    def test_policy_candidate_registry_names_deferred_concepts(self) -> None:
        candidates_by_name = {
            candidate.name: candidate
            for candidate in SOUTH_STAR_ANNOTATION_POLICY_CANDIDATES
        }

        self.assertEqual(
            "default",
            candidates_by_name["maximal_eligible_carrier"].status,
        )
        self.assertEqual(
            "deferred_candidate",
            candidates_by_name["minimal_sufficient"].status,
        )
        self.assertEqual(
            "deferred_candidate",
            candidates_by_name["canonical_semantic"].status,
        )
        self.assertEqual(
            "comparison_candidate",
            candidates_by_name["rdkit_writer_like"].status,
        )
        self.assertIn(
            "not as South Star semantic authority",
            candidates_by_name["rdkit_writer_like"].description,
        )
        self.assertEqual(
            "test_stub",
            candidates_by_name["no_marker_policy_stub"].status,
        )

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
                annotation_policy=NoMarkerAnnotationPolicyStub(),
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
