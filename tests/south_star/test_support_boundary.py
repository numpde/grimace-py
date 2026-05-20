from __future__ import annotations

import unittest

from grimace._south_star.annotation_policy import DIRECTIONAL_MARKERS
from tests.helpers.south_star_semantics import load_south_star_semantic_cases
from tests.helpers.south_star_support_boundary import SouthStarOnlineSupportBoundary


class SouthStarOnlineSupportBoundaryTests(unittest.TestCase):
    def test_fixture_carriers_allow_directional_markers_through_policy(self) -> None:
        for case in load_south_star_semantic_cases():
            boundary = SouthStarOnlineSupportBoundary.from_case(case)

            for edge in case.eligible_carrier_edges:
                with self.subTest(case_id=case.case_id, edge=edge):
                    self.assertEqual(
                        DIRECTIONAL_MARKERS,
                        boundary.allowed_directional_markers(edge=edge),
                    )

    def test_noncarrier_edge_rejects_directional_markers_with_reason(self) -> None:
        for case in load_south_star_semantic_cases():
            boundary = SouthStarOnlineSupportBoundary.from_case(case)

            support = boundary.explain_directional_marker(
                edge=(10_000, 10_001),
                marker="/",
            )

            with self.subTest(case_id=case.case_id):
                self.assertFalse(support.token_allowed)
                self.assertEqual(
                    "marker_not_required_by_annotation_policy",
                    support.reason,
                )

    def test_invalid_directional_marker_fails_fast(self) -> None:
        for case in load_south_star_semantic_cases():
            boundary = SouthStarOnlineSupportBoundary.from_case(case)

            with self.subTest(case_id=case.case_id):
                with self.assertRaisesRegex(ValueError, "directional marker"):
                    boundary.explain_directional_marker(
                        edge=case.eligible_carrier_edges[0],
                        marker="?",
                    )
