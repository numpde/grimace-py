from __future__ import annotations

import json
from pathlib import Path
import unittest

from tests.helpers.south_star_enum_s_benchmark_cases import (
    SOUTH_STAR_ENUM_S_BENCHMARK_POLICY_SET,
    SOUTH_STAR_ENUM_S_BENCHMARK_SCOPE_NOTE,
    SOUTH_STAR_ENUM_S_CASE_MANIFEST_SCOPE_NOTE,
    south_star_enum_s_benchmark_cases,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_REPORT = (
    REPO_ROOT / "notes" / "perf_reports" / "south_star_enum_s_v1.json"
)
CASE_MANIFEST = (
    REPO_ROOT / "notes" / "perf_reports" / "south_star_enum_s_case_manifest_v1.json"
)


class SouthStarBenchmarkArtifactTests(unittest.TestCase):
    def test_case_manifest_covers_current_pinned_cases(self) -> None:
        manifest = json.loads(CASE_MANIFEST.read_text())

        self.assertEqual("south_star_semantic_enum_s_case_manifest", manifest["kind"])
        self.assertEqual(SOUTH_STAR_ENUM_S_BENCHMARK_POLICY_SET, manifest["policy_set"])
        self.assertEqual(
            SOUTH_STAR_ENUM_S_CASE_MANIFEST_SCOPE_NOTE,
            manifest["scope_note"],
        )
        expected_rows = _expected_case_rows()
        rows = manifest["rows"]
        row_by_id = {row["case_id"]: row for row in rows}

        self.assertEqual(len(rows), len(row_by_id))
        self.assertEqual(set(expected_rows), set(row_by_id))
        for case_id, expected in expected_rows.items():
            with self.subTest(case_id=case_id):
                row = row_by_id[case_id]
                for key, value in expected.items():
                    self.assertEqual(value, row[key])

    def test_benchmark_artifact_rows_are_current_manifest_cases(self) -> None:
        manifest = json.loads(CASE_MANIFEST.read_text())
        report = json.loads(BENCHMARK_REPORT.read_text())

        manifest_by_id = {row["case_id"]: row for row in manifest["rows"]}
        rows = report["rows"]
        row_by_id = {row["case_id"]: row for row in rows}

        self.assertEqual(len(rows), len(row_by_id))
        self.assertTrue(set(row_by_id) <= set(manifest_by_id))
        for case_id, row in row_by_id.items():
            with self.subTest(case_id=case_id):
                manifest_row = manifest_by_id[case_id]
                for key in (
                    "fixture_family",
                    "domain_label",
                    "source_smiles",
                    "expected_output_count",
                ):
                    self.assertEqual(manifest_row[key], row[key])
                self.assertEqual(row["expected_output_count"], row["output_count"])

    def test_benchmark_artifact_records_private_semantic_scope(self) -> None:
        report = json.loads(BENCHMARK_REPORT.read_text())

        self.assertEqual("south_star_semantic_enum_s_benchmark", report["kind"])
        self.assertEqual(SOUTH_STAR_ENUM_S_BENCHMARK_POLICY_SET, report["policy_set"])
        self.assertEqual(SOUTH_STAR_ENUM_S_BENCHMARK_SCOPE_NOTE, report["scope_note"])
        self.assertRegex(report["metadata"]["git_commit"], r"^[0-9a-f]{40}$")
        self.assertRegex(report["metadata"]["rdkit_version"], r"^\d{4}\.\d{2}\.\d+$")
        self.assertGreater(report["repeats"], 0)
        self.assertGreaterEqual(report["warmups"], 0)


def _expected_case_rows() -> dict[str, dict[str, object]]:
    return {
        case.case_id: {
            "fixture_family": case.fixture_family,
            "domain_label": case.domain_label,
            "source_smiles": case.source_smiles,
            "expected_output_count": case.expected_output_count,
        }
        for case in south_star_enum_s_benchmark_cases()
    }


if __name__ == "__main__":
    unittest.main()
