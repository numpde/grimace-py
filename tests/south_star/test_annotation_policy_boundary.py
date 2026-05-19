from __future__ import annotations

import unittest

from tests.helpers.south_star_annotation_policy import (
    DIRECTIONAL_MARKERS,
    EmittedEdgeBasis,
    MaximalEligibleCarrierAnnotationPolicy,
    SemanticCarrierOpportunity,
    SurvivingSemanticAssignment,
    normalized_edge,
)
from tests.helpers.south_star_semantics import load_south_star_semantic_cases


def _carrier_opportunities(
    edges: tuple[tuple[int, int], ...],
) -> tuple[SemanticCarrierOpportunity, ...]:
    return tuple(SemanticCarrierOpportunity(edge=edge) for edge in edges)


def _all_marker_assignment(
    edges: tuple[tuple[int, int], ...],
) -> SurvivingSemanticAssignment:
    return SurvivingSemanticAssignment(
        assignment_id="all-markers",
        marker_options_by_edge={
            normalized_edge(edge): DIRECTIONAL_MARKERS for edge in edges
        },
    )


class SouthStarAnnotationPolicyBoundaryTests(unittest.TestCase):
    def test_maximal_policy_requires_each_eligible_carrier_edge(self) -> None:
        policy = MaximalEligibleCarrierAnnotationPolicy()
        for case in load_south_star_semantic_cases():
            opportunities = _carrier_opportunities(case.eligible_carrier_edges)
            survivor = _all_marker_assignment(case.eligible_carrier_edges)

            required_count = 0
            for edge in case.eligible_carrier_edges:
                decision = policy.decision(
                    carrier_opportunities=opportunities,
                    emitted_edge=EmittedEdgeBasis(edge=edge),
                    surviving_assignments=(survivor,),
                )
                required_count += int(decision.marker_required)
                self.assertEqual(DIRECTIONAL_MARKERS, decision.allowed_markers)

            self.assertEqual(
                case.maximal_eligible_carrier.required_marker_edge_count,
                required_count,
            )

    def test_noncarrier_edge_does_not_require_directional_marker(self) -> None:
        policy = MaximalEligibleCarrierAnnotationPolicy()
        decision = policy.decision(
            carrier_opportunities=(SemanticCarrierOpportunity(edge=(0, 1)),),
            emitted_edge=EmittedEdgeBasis(edge=(1, 2)),
            surviving_assignments=(
                SurvivingSemanticAssignment(
                    assignment_id="survivor",
                    marker_options_by_edge={(0, 1): DIRECTIONAL_MARKERS},
                ),
            ),
        )

        self.assertFalse(decision.marker_required)
        self.assertEqual((), decision.allowed_markers)

    def test_allowed_markers_are_union_of_surviving_assignments(self) -> None:
        policy = MaximalEligibleCarrierAnnotationPolicy()
        decision = policy.decision(
            carrier_opportunities=(SemanticCarrierOpportunity(edge=(0, 1)),),
            emitted_edge=EmittedEdgeBasis(edge=(1, 0)),
            surviving_assignments=(
                SurvivingSemanticAssignment(
                    assignment_id="slash",
                    marker_options_by_edge={(0, 1): ("/",)},
                ),
                SurvivingSemanticAssignment(
                    assignment_id="backslash",
                    marker_options_by_edge={(0, 1): ("\\",)},
                ),
            ),
        )

        self.assertTrue(decision.marker_required)
        self.assertEqual(DIRECTIONAL_MARKERS, decision.allowed_markers)

    def test_invalid_survivor_marker_options_fail_fast(self) -> None:
        policy = MaximalEligibleCarrierAnnotationPolicy()
        with self.assertRaisesRegex(ValueError, "invalid marker options"):
            policy.decision(
                carrier_opportunities=(SemanticCarrierOpportunity(edge=(0, 1)),),
                emitted_edge=EmittedEdgeBasis(edge=(0, 1)),
                surviving_assignments=(
                    SurvivingSemanticAssignment(
                        assignment_id="bad-marker",
                        marker_options_by_edge={(0, 1): ("?",)},
                    ),
                ),
            )
