from __future__ import annotations

import json
from pathlib import Path
import unittest

from tests.helpers.south_star_exact_support import (
    load_south_star_exact_first_domain_cases,
    load_south_star_expanded_support_cases,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_REPORT = (
    REPO_ROOT / "notes" / "perf_reports" / "south_star_enum_s_v1.json"
)


class SouthStarBenchmarkArtifactTests(unittest.TestCase):
    def test_benchmark_artifact_covers_current_pinned_cases(self) -> None:
        report = json.loads(BENCHMARK_REPORT.read_text())

        expected_rows = {
            case.case_id: {
                "fixture_family": "exact_first_domain",
                "domain_label": "first_domain_directional_bond_stereo",
                "source_smiles": case.source_smiles,
                "expected_output_count": len(case.expected_support),
            }
            for case in load_south_star_exact_first_domain_cases()
        }
        expected_rows.update(
            {
                case.case_id: {
                    "fixture_family": "expanded_support",
                    "domain_label": case.feature_area,
                    "source_smiles": case.source_smiles,
                    "expected_output_count": len(case.expected_support),
                }
                for case in load_south_star_expanded_support_cases()
            }
        )

        rows = report["rows"]
        row_by_id = {row["case_id"]: row for row in rows}

        self.assertEqual(len(rows), len(row_by_id))
        self.assertEqual(set(expected_rows), set(row_by_id))
        for case_id, expected in expected_rows.items():
            with self.subTest(case_id=case_id):
                row = row_by_id[case_id]
                for key, value in expected.items():
                    self.assertEqual(value, row[key])
                self.assertEqual(row["expected_output_count"], row["output_count"])

    def test_benchmark_artifact_records_private_semantic_scope(self) -> None:
        report = json.loads(BENCHMARK_REPORT.read_text())

        self.assertEqual("south_star_semantic_enum_s_benchmark", report["kind"])
        self.assertEqual(
            {
                "annotation_policy": "maximal_eligible_carrier",
                "fragment_order_policy": "all_fragment_orders",
                "output_order_policy": "first_occurrence_deduplication",
            },
            report["policy_set"],
        )
        self.assertIn("private South Star semantic enumerator", report["scope_note"])
        self.assertIn("not an RDKit writer-parity benchmark", report["scope_note"])
        self.assertRegex(report["metadata"]["git_commit"], r"^[0-9a-f]{40}$")
        self.assertRegex(report["metadata"]["rdkit_version"], r"^\d{4}\.\d{2}\.\d+$")
        self.assertGreater(report["repeats"], 0)
        self.assertGreaterEqual(report["warmups"], 0)


if __name__ == "__main__":
    unittest.main()
