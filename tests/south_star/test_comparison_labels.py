from __future__ import annotations

import unittest

from tests.helpers.south_star_comparison import RDKIT_PARITY_ONLY
from tests.helpers.south_star_comparison import SOUTH_STAR_INTERSECTION
from tests.helpers.south_star_comparison import SOUTH_STAR_ONLY
from tests.helpers.south_star_comparison import (
    south_star_expanded_parity_comparison_report,
)
from tests.helpers.south_star_comparison import south_star_parity_comparison_report
from tests.helpers.south_star_comparison import south_star_comparison_labels
from tests.helpers.south_star_domain_manifest import (
    SOUTH_STAR_GRAPH_NATIVE_REGRESSION_AUTHORITY,
)
from tests.helpers.south_star_exact_support import (
    load_south_star_expanded_support_cases,
)
from tests.helpers.south_star_semantics import load_south_star_semantic_cases


class SouthStarComparisonLabelTests(unittest.TestCase):
    def assertDiagnosticPartition(self, report) -> None:
        self.assertEqual(
            report.south_star_support_size,
            len(report.south_star_only) + len(report.intersection),
        )
        self.assertEqual(
            report.rdkit_parity_support_size,
            len(report.rdkit_parity_only) + len(report.intersection),
        )
        self.assertEqual(len(report.intersection), report.intersection_size)
        for classification in report.classifications:
            self.assertIn(
                classification.membership,
                (
                    SOUTH_STAR_INTERSECTION,
                    SOUTH_STAR_ONLY,
                    RDKIT_PARITY_ONLY,
                ),
            )
            self.assertTrue(
                classification.conformance_report.rdkit_parseability.passed
            )

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

    def test_support_report_classifies_differences_without_requiring_equality(self) -> None:
        for case in load_south_star_semantic_cases():
            report = south_star_parity_comparison_report(case)

            with self.subTest(case_id=case.case_id):
                self.assertEqual(case.case_id, report.case_id)
                self.assertDiagnosticPartition(report)
                self.assertEqual(
                    report.south_star_support_size
                    + report.rdkit_parity_support_size
                    - report.intersection_size,
                    len(report.classifications),
                )

    def test_expanded_support_reports_are_diagnostic_only(self) -> None:
        cases = tuple(
            case
            for case in load_south_star_expanded_support_cases()
            if case.support_authority != SOUTH_STAR_GRAPH_NATIVE_REGRESSION_AUTHORITY
        )

        self.assertNotEqual((), cases)
        for case in cases:
            report = south_star_expanded_parity_comparison_report(case)

            with self.subTest(case_id=case.case_id):
                self.assertEqual(case.case_id, report.case_id)
                self.assertDiagnosticPartition(report)
