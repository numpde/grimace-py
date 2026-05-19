from __future__ import annotations

import unittest

from tests.helpers.south_star_comparison import south_star_comparison_labels
from tests.helpers.south_star_semantics import load_south_star_semantic_cases


class SouthStarComparisonLabelTests(unittest.TestCase):
    def test_comparison_labels_keep_semantic_and_writer_results_separate(self) -> None:
        for case in load_south_star_semantic_cases():
            labels = south_star_comparison_labels(case, rdkit_sample_count=32)

            self.assertEqual(
                len(case.positive_semantic_smiles) + len(case.negative_semantic_smiles),
                len(labels),
            )
            for label in labels:
                with self.subTest(
                    case_id=label.case_id,
                    candidate=label.candidate_smiles,
                ):
                    self.assertEqual(
                        label.expected_semantic_acceptance,
                        label.semantic_oracle_accepts,
                    )
                    self.assertIsInstance(label.rdkit_writer_observed, bool)
                    self.assertIsInstance(label.grimace_parity_support_accepts, bool)

    def test_negative_semantic_witnesses_are_not_reclassified_by_writer_results(self) -> None:
        for case in load_south_star_semantic_cases():
            labels = south_star_comparison_labels(case, rdkit_sample_count=32)

            for label in labels:
                if label.expected_semantic_acceptance:
                    continue
                with self.subTest(
                    case_id=label.case_id,
                    candidate=label.candidate_smiles,
                ):
                    self.assertFalse(label.semantic_oracle_accepts)
