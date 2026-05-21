from __future__ import annotations

import unittest

from grimace._south_star.ring_labels import (
    SouthStarFirstEncounterRingClosureLabelPolicy,
)
from grimace._south_star.ring_labels import closure_id_for_edge


class SouthStarRingLabelPolicyTests(unittest.TestCase):
    def test_single_closure_edge_gets_first_label(self) -> None:
        assignments = SouthStarFirstEncounterRingClosureLabelPolicy().assignments_for_edges(
            ((5, 1),)
        )

        self.assertEqual(1, len(assignments))
        self.assertEqual("1-5", assignments[0].closure_id)
        self.assertEqual((1, 5), assignments[0].edge)
        self.assertEqual("1", assignments[0].label)

    def test_multiple_closure_edges_keep_first_encounter_label_order(self) -> None:
        assignments = SouthStarFirstEncounterRingClosureLabelPolicy().assignments_for_edges(
            ((4, 0), (7, 2), (3, 6))
        )

        self.assertEqual(
            (
                ("0-4", (0, 4), "1"),
                ("2-7", (2, 7), "2"),
                ("3-6", (3, 6), "3"),
            ),
            tuple(
                (assignment.closure_id, assignment.edge, assignment.label)
                for assignment in assignments
            ),
        )

    def test_duplicate_closure_edges_fail_fast_after_normalization(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique edges"):
            SouthStarFirstEncounterRingClosureLabelPolicy().assignments_for_edges(
                ((1, 4), (4, 1))
            )

    def test_unsupported_label_range_fails_fast(self) -> None:
        closure_edges = tuple((idx, idx + 10) for idx in range(10))

        with self.assertRaisesRegex(NotImplementedError, "at most 9"):
            SouthStarFirstEncounterRingClosureLabelPolicy().assignments_for_edges(
                closure_edges
            )

    def test_closure_id_normalizes_edge(self) -> None:
        self.assertEqual("2-8", closure_id_for_edge((8, 2)))


if __name__ == "__main__":
    unittest.main()
