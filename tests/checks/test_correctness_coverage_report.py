from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tests.helpers.pinned_rdkit_fixtures import (
    PINNED_RDKIT_EXACT_SMALL_SUPPORT,
    PINNED_RDKIT_KNOWN_QUIRKS,
    PINNED_RDKIT_KNOWN_STEREO_GAPS,
    PINNED_RDKIT_PARITY_FIXTURE_FAMILIES,
    PINNED_RDKIT_ROOTED_RANDOM,
    PINNED_RDKIT_SERIALIZER_REGRESSIONS,
    PINNED_RDKIT_WRITER_MEMBERSHIP,
    PINNED_RDKIT_WRITER_SUPPORT_COUNTS,
)
from tests.helpers.rdkit_serializer_coverage import (
    COVERAGE_STATUS_COVERED,
    COVERAGE_STATUS_KNOWN_GAP,
    COVERAGE_STATUS_OUT_OF_SCOPE,
    DEFAULT_RDKIT_SERIALIZER_VERSION,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "report_correctness_coverage.py"
SERIALIZER_REPORT_SCRIPT = (
    ROOT / "scripts" / "report_rdkit_serializer_coverage.py"
)
RDKIT_VERSION = DEFAULT_RDKIT_SERIALIZER_VERSION
REPORT_FIXTURE_FAMILIES = (
    *PINNED_RDKIT_PARITY_FIXTURE_FAMILIES,
    PINNED_RDKIT_WRITER_SUPPORT_COUNTS,
    PINNED_RDKIT_KNOWN_QUIRKS,
    PINNED_RDKIT_KNOWN_STEREO_GAPS,
)


def _load_report_module():
    spec = importlib.util.spec_from_file_location(
        "report_correctness_coverage",
        SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPORT = _load_report_module()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _minimal_case_payload(family: str) -> dict[str, object]:
    return {
        "rdkit_version": "2099.01.1",
        "cases": [
            {
                "id": f"{family}_case",
                "source": "Local contract test",
            }
        ],
    }


def _minimal_coverage_payload(
    status: str = COVERAGE_STATUS_COVERED,
    *,
    links: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    if links is None:
        links = [
            {
                "fixture": (
                    "tests/fixtures/rdkit_exact_small_support/2099.01.1.json"
                ),
                "cases": [
                    f"{PINNED_RDKIT_EXACT_SMALL_SUPPORT}_case",
                ],
                "note": "contract test",
            }
        ]
    return {
        "rdkit_version": "2099.01.1",
        "entries": [
            {
                "status": status,
                "grimace_links": links,
            }
        ],
    }


def _write_minimal_fixture_families(
    fixture_root: Path,
    *,
    unclassified_family: str | None = None,
) -> None:
    for family in REPORT_FIXTURE_FAMILIES:
        payload = _minimal_case_payload(family)
        if family == unclassified_family:
            payload["cases"] = [
                {
                    "id": "unclassified_case",
                    "source": "contract test",
                }
            ]
        _write_json(fixture_root / family / "2099.01.1.json", payload)


def _write_serializer_ledger(fixture_root: Path, payload: object) -> None:
    _write_json(
        fixture_root / "rdkit_upstream_serializer_coverage" / "2099.01.1.json",
        payload,
    )


class CorrectnessCoverageReportTests(unittest.TestCase):
    def test_report_fixture_scope_comes_from_pinned_fixture_constants(self) -> None:
        summary = REPORT.build_summary(ROOT)

        self.assertEqual(
            REPORT_FIXTURE_FAMILIES,
            tuple(summary["fixture_cases"]),
        )

    def test_report_fails_when_authoritative_fixture_family_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_root = Path(tmpdir) / "tests" / "fixtures"
            fixture_root.mkdir(parents=True)

            with self.assertRaisesRegex(
                FileNotFoundError,
                "missing pinned RDKit fixture family",
            ):
                REPORT.build_summary(Path(tmpdir))

    def test_report_fails_when_serializer_ledger_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_root = Path(tmpdir) / "tests" / "fixtures"
            _write_minimal_fixture_families(fixture_root)

            with self.assertRaisesRegex(
                FileNotFoundError,
                "missing RDKit serializer coverage ledger",
            ):
                REPORT.build_summary(Path(tmpdir))

    def test_report_fails_on_unclassified_fixture_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_root = Path(tmpdir) / "tests" / "fixtures"
            _write_minimal_fixture_families(
                fixture_root,
                unclassified_family=PINNED_RDKIT_WRITER_MEMBERSHIP,
            )
            _write_serializer_ledger(
                fixture_root,
                _minimal_coverage_payload(),
            )

            with self.assertRaisesRegex(ValueError, "unclassified source"):
                REPORT.build_summary(Path(tmpdir))

    def test_report_fails_on_unknown_serializer_ledger_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_root = Path(tmpdir) / "tests" / "fixtures"
            _write_minimal_fixture_families(fixture_root)
            _write_serializer_ledger(
                fixture_root,
                _minimal_coverage_payload("typo-status"),
            )

            with self.assertRaisesRegex(ValueError, "unknown ledger status"):
                REPORT.build_summary(Path(tmpdir))

    def test_report_fails_on_covered_serializer_entry_without_links(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_root = Path(tmpdir) / "tests" / "fixtures"
            _write_minimal_fixture_families(fixture_root)
            _write_serializer_ledger(
                fixture_root,
                _minimal_coverage_payload(links=[]),
            )

            with self.assertRaisesRegex(
                ValueError,
                "must link executable fixture cases",
            ):
                REPORT.build_summary(Path(tmpdir))

    def test_report_fails_on_out_of_scope_serializer_entry_with_links(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_root = Path(tmpdir) / "tests" / "fixtures"
            _write_minimal_fixture_families(fixture_root)
            _write_serializer_ledger(
                fixture_root,
                _minimal_coverage_payload(COVERAGE_STATUS_OUT_OF_SCOPE),
            )

            with self.assertRaisesRegex(
                ValueError,
                "must not link executable fixture cases",
            ):
                REPORT.build_summary(Path(tmpdir))

    def test_report_fails_on_serializer_link_to_missing_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_root = Path(tmpdir) / "tests" / "fixtures"
            _write_minimal_fixture_families(fixture_root)
            _write_serializer_ledger(
                fixture_root,
                _minimal_coverage_payload(
                    links=[
                        {
                            "fixture": (
                                "tests/fixtures/rdkit_exact_small_support/"
                                "2099.01.1.json"
                            ),
                            "cases": ["missing_case"],
                            "note": "contract test",
                        }
                    ],
                ),
            )

            with self.assertRaisesRegex(ValueError, "references missing cases"):
                REPORT.build_summary(Path(tmpdir))

    def test_report_fails_on_serializer_ledger_entry_without_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_root = Path(tmpdir) / "tests" / "fixtures"
            _write_minimal_fixture_families(fixture_root)
            _write_serializer_ledger(
                fixture_root,
                {
                    "rdkit_version": "2099.01.1",
                    "entries": [
                        {
                            "grimace_links": [],
                        }
                    ],
                },
            )

            with self.assertRaisesRegex(ValueError, "without status"):
                REPORT.build_summary(Path(tmpdir))

    def test_report_fails_on_malformed_checked_in_fixture_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_root = Path(tmpdir) / "tests" / "fixtures"
            _write_minimal_fixture_families(fixture_root)
            _write_serializer_ledger(fixture_root, _minimal_coverage_payload())
            malformed_fixture = fixture_root / "extra" / "bad.json"
            malformed_fixture.parent.mkdir(parents=True)
            malformed_fixture.write_text("[", encoding="utf-8")

            with self.assertRaises(json.JSONDecodeError):
                REPORT.build_summary(Path(tmpdir))

    def test_report_fails_on_non_object_checked_in_fixture_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_root = Path(tmpdir) / "tests" / "fixtures"
            _write_minimal_fixture_families(fixture_root)
            _write_serializer_ledger(fixture_root, _minimal_coverage_payload())
            _write_json(fixture_root / "extra" / "list.json", [])

            with self.assertRaisesRegex(ValueError, "must contain a JSON object"):
                REPORT.build_summary(Path(tmpdir))

    def test_report_fails_on_empty_serializer_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_root = Path(tmpdir) / "tests" / "fixtures"
            _write_minimal_fixture_families(fixture_root)
            _write_serializer_ledger(
                fixture_root,
                {
                    "rdkit_version": "2099.01.1",
                    "entries": [],
                },
            )

            with self.assertRaisesRegex(ValueError, "at least one entry"):
                REPORT.build_summary(Path(tmpdir))

    def test_summary_counts_current_checked_in_fixture_baseline(self) -> None:
        summary = REPORT.build_summary(ROOT)

        self.assertEqual(
            {
                PINNED_RDKIT_EXACT_SMALL_SUPPORT: {RDKIT_VERSION: 76},
                PINNED_RDKIT_ROOTED_RANDOM: {RDKIT_VERSION: 1},
                PINNED_RDKIT_SERIALIZER_REGRESSIONS: {RDKIT_VERSION: 130},
                PINNED_RDKIT_WRITER_MEMBERSHIP: {RDKIT_VERSION: 56},
                PINNED_RDKIT_WRITER_SUPPORT_COUNTS: {RDKIT_VERSION: 18},
                PINNED_RDKIT_KNOWN_QUIRKS: {RDKIT_VERSION: 1},
                PINNED_RDKIT_KNOWN_STEREO_GAPS: {RDKIT_VERSION: 16},
            },
            summary["fixture_cases"],
        )
        self.assertEqual(
            {
                "upstream-rdkit": 171,
                "local-probe": 30,
                "dataset-derived": 49,
                "random-writer-observation": 31,
                "known-rdkit-gap": 16,
                "rdkit-quirk": 1,
            },
            summary["source_classes"],
        )
        self.assertEqual(
            {
                COVERAGE_STATUS_COVERED: 54,
                COVERAGE_STATUS_KNOWN_GAP: 6,
                COVERAGE_STATUS_OUT_OF_SCOPE: 209,
            },
            summary["serializer_ledger_statuses"][RDKIT_VERSION],
        )

    def test_cli_json_output_runs(self) -> None:
        summary = REPORT.build_summary(ROOT)
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--format", "json"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(summary, json.loads(proc.stdout))


class SerializerCoverageReportCliTests(unittest.TestCase):
    def test_rejects_unknown_status_filter(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                str(SERIALIZER_REPORT_SCRIPT),
                "--status",
                "typo-status",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("invalid choice", proc.stderr)

    def test_rejects_negative_limit(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(SERIALIZER_REPORT_SCRIPT), "--limit", "-1"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("--limit must be nonnegative", proc.stderr)


if __name__ == "__main__":
    unittest.main()
