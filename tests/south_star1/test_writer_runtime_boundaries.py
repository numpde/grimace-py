"""Architectural boundary tests for the writer-shaped runtime stack."""

from __future__ import annotations

import unittest
from pathlib import Path

from tests.helpers.module_boundaries import import_from_observations
from tests.helpers.module_boundaries import scan_module_boundaries


REPO_ROOT = Path(__file__).resolve().parents[2]
SOUTH_STAR_ROOT = REPO_ROOT / "python" / "grimace" / "_south_star1"
WRITER_RUNTIME_PATH = SOUTH_STAR_ROOT / "writer_runtime.py"
WRITER_SUPPORT_PATH = SOUTH_STAR_ROOT / "writer_support.py"
WRITER_BRANCH_CERTIFICATES_PATH = (
    SOUTH_STAR_ROOT / "writer_branch_certificates.py"
)
WRITER_CAPABILITY_CERTIFICATES_PATH = (
    SOUTH_STAR_ROOT / "writer_capability_certificates.py"
)
WRITER_CLOSURE_CANDIDATE_BRANCH_CERTIFICATES_PATH = (
    SOUTH_STAR_ROOT / "writer_closure_candidate_branch_certificates.py"
)
WRITER_CLOSURE_CANDIDATE_LIFECYCLE_PATH = (
    SOUTH_STAR_ROOT / "writer_closure_candidate_lifecycle.py"
)
WRITER_FRONTIER_CERTIFICATES_PATH = (
    SOUTH_STAR_ROOT / "writer_frontier_certificates.py"
)
WRITER_COUNT_CERTIFICATES_PATH = (
    SOUTH_STAR_ROOT / "writer_count_certificates.py"
)
WRITER_ONLINE_DECODER_CERTIFICATES_PATH = (
    SOUTH_STAR_ROOT / "writer_online_decoder_certificates.py"
)
WRITER_DIAGNOSTIC_CERTIFICATES_PATH = (
    SOUTH_STAR_ROOT / "writer_diagnostic_certificates.py"
)
WRITER_RESIDUAL_ATTACHMENT_BRANCH_CERTIFICATES_PATH = (
    SOUTH_STAR_ROOT / "writer_residual_attachment_branch_certificates.py"
)
WRITER_RESIDUAL_ATTACHMENT_LIFECYCLE_PATH = (
    SOUTH_STAR_ROOT / "writer_residual_attachment_lifecycle.py"
)
WRITER_STEREO_BRANCH_CERTIFICATES_PATH = (
    SOUTH_STAR_ROOT / "writer_stereo_branch_certificates.py"
)
WRITER_TERMINAL_CERTIFICATES_PATH = (
    SOUTH_STAR_ROOT / "writer_terminal_certificates.py"
)
WRITER_PROJECTION_CERTIFICATES_PATH = (
    SOUTH_STAR_ROOT / "writer_projection_certificates.py"
)
WRITER_SNAPSHOT_CERTIFICATES_PATH = (
    SOUTH_STAR_ROOT / "writer_snapshot_certificates.py"
)
WRITER_SUPPORT_CERTIFICATES_PATH = (
    SOUTH_STAR_ROOT / "writer_support_certificates.py"
)


class WriterRuntimeBoundaryTest(unittest.TestCase):
    def test_writer_runtime_stays_below_adapters(self) -> None:
        scan = scan_module_boundaries(
            WRITER_RUNTIME_PATH,
            banned_modules={
                "audit_rdkit",
                "online_decoder_api",
                "rdkit_adapter",
                "support_artifact",
                "support_artifact_checker",
                "support_enumeration",
                "writer_online_decoder",
                "writer_support",
            },
            banned_calls={
                "count_writer_runtime_completions",
                "count_writer_runtime_support",
                "enumerate_prepared_writer_shaped_support",
                "iter_writer_runtime_support",
                "make_writer_shaped_online_decoder",
            },
        )

        self.assertEqual(scan.violations, ())

    def test_writer_support_adapter_routes_through_runtime(self) -> None:
        scan = scan_module_boundaries(
            WRITER_SUPPORT_PATH,
            banned_modules={
                "rdkit_adapter",
                "support_artifact",
                "support_artifact_checker",
                "support_enumeration",
                "writer_frontier",
                "writer_transitions",
            },
            banned_calls={
                "count_writer_cursor_completions",
                "count_writer_frontier_support",
                "count_writer_runtime_branch_completions",
                "initial_writer_frontier_cursor",
                "iter_writer_frontier_support",
                "writer_runtime_branch_transitions",
            },
        )
        snapshot_imports = import_from_observations(
            WRITER_SUPPORT_PATH,
            module_root="writer_snapshot",
        )

        self.assertEqual(scan.violations, ())
        self.assertEqual(len(snapshot_imports), 1)
        self.assertTrue(snapshot_imports[0].inside_type_checking)

    def test_closure_candidate_lifecycle_stays_below_runtime(self) -> None:
        scan = scan_module_boundaries(
            WRITER_CLOSURE_CANDIDATE_LIFECYCLE_PATH,
            banned_modules={
                "audit_rdkit",
                "rdkit_adapter",
                "writer_online_decoder",
                "writer_runtime",
                "writer_support",
            },
        )

        self.assertEqual(scan.violations, ())

    def test_frontier_certificates_stay_below_runtime(self) -> None:
        scan = scan_module_boundaries(
            WRITER_FRONTIER_CERTIFICATES_PATH,
            banned_modules={
                "audit_rdkit",
                "rdkit_adapter",
                "writer_online_decoder",
                "writer_runtime",
                "writer_support",
            },
        )

        self.assertEqual(scan.violations, ())

    def test_count_certificates_stay_below_runtime(self) -> None:
        scan = scan_module_boundaries(
            WRITER_COUNT_CERTIFICATES_PATH,
            banned_modules={
                "audit_rdkit",
                "rdkit_adapter",
                "writer_online_decoder",
                "writer_runtime",
                "writer_support",
            },
        )

        self.assertEqual(scan.violations, ())

    def test_diagnostic_certificates_stay_below_runtime(self) -> None:
        scan = scan_module_boundaries(
            WRITER_DIAGNOSTIC_CERTIFICATES_PATH,
            banned_modules={
                "audit_rdkit",
                "rdkit_adapter",
                "writer_online_decoder",
                "writer_runtime",
                "writer_support",
            },
        )

        self.assertEqual(scan.violations, ())

    def test_branch_certificates_stay_below_runtime(self) -> None:
        scan = scan_module_boundaries(
            WRITER_BRANCH_CERTIFICATES_PATH,
            banned_modules={
                "audit_rdkit",
                "rdkit_adapter",
                "writer_online_decoder",
                "writer_runtime",
                "writer_support",
            },
        )

        self.assertEqual(scan.violations, ())

    def test_capability_certificates_stay_below_runtime(self) -> None:
        scan = scan_module_boundaries(
            WRITER_CAPABILITY_CERTIFICATES_PATH,
            banned_modules={
                "audit_rdkit",
                "rdkit_adapter",
                "writer_online_decoder",
                "writer_runtime",
                "writer_support",
            },
        )

        self.assertEqual(scan.violations, ())

    def test_closure_candidate_branch_certificates_stay_below_runtime(
        self,
    ) -> None:
        scan = scan_module_boundaries(
            WRITER_CLOSURE_CANDIDATE_BRANCH_CERTIFICATES_PATH,
            banned_modules={
                "audit_rdkit",
                "rdkit_adapter",
                "writer_online_decoder",
                "writer_runtime",
                "writer_support",
            },
        )

        self.assertEqual(scan.violations, ())

    def test_residual_attachment_lifecycle_stays_below_runtime(self) -> None:
        scan = scan_module_boundaries(
            WRITER_RESIDUAL_ATTACHMENT_LIFECYCLE_PATH,
            banned_modules={
                "audit_rdkit",
                "rdkit_adapter",
                "writer_online_decoder",
                "writer_runtime",
                "writer_support",
            },
        )

        self.assertEqual(scan.violations, ())

    def test_residual_attachment_branch_certificates_stay_below_runtime(
        self,
    ) -> None:
        scan = scan_module_boundaries(
            WRITER_RESIDUAL_ATTACHMENT_BRANCH_CERTIFICATES_PATH,
            banned_modules={
                "audit_rdkit",
                "rdkit_adapter",
                "writer_online_decoder",
                "writer_runtime",
                "writer_support",
            },
        )

        self.assertEqual(scan.violations, ())

    def test_stereo_branch_certificates_stay_below_runtime(self) -> None:
        scan = scan_module_boundaries(
            WRITER_STEREO_BRANCH_CERTIFICATES_PATH,
            banned_modules={
                "audit_rdkit",
                "rdkit_adapter",
                "writer_online_decoder",
                "writer_runtime",
                "writer_support",
            },
        )

        self.assertEqual(scan.violations, ())

    def test_terminal_certificates_stay_below_runtime(self) -> None:
        scan = scan_module_boundaries(
            WRITER_TERMINAL_CERTIFICATES_PATH,
            banned_modules={
                "audit_rdkit",
                "rdkit_adapter",
                "writer_online_decoder",
                "writer_runtime",
                "writer_support",
            },
        )

        self.assertEqual(scan.violations, ())

    def test_projection_certificates_stay_below_runtime(self) -> None:
        scan = scan_module_boundaries(
            WRITER_PROJECTION_CERTIFICATES_PATH,
            banned_modules={
                "audit_rdkit",
                "rdkit_adapter",
                "writer_online_decoder",
                "writer_runtime",
                "writer_support",
            },
        )

        self.assertEqual(scan.violations, ())

    def test_online_decoder_certificates_stay_below_runtime(self) -> None:
        scan = scan_module_boundaries(
            WRITER_ONLINE_DECODER_CERTIFICATES_PATH,
            banned_modules={
                "audit_rdkit",
                "rdkit_adapter",
                "writer_online_decoder",
                "writer_runtime",
                "writer_support",
                "writer_frontier",
            },
        )

        self.assertEqual(scan.violations, ())

    def test_snapshot_certificates_stay_below_runtime(self) -> None:
        scan = scan_module_boundaries(
            WRITER_SNAPSHOT_CERTIFICATES_PATH,
            banned_modules={
                "audit_rdkit",
                "rdkit_adapter",
                "writer_online_decoder",
                "writer_runtime",
                "writer_support",
            },
        )

        self.assertEqual(scan.violations, ())

    def test_support_certificates_stay_below_runtime(self) -> None:
        scan = scan_module_boundaries(
            WRITER_SUPPORT_CERTIFICATES_PATH,
            banned_modules={
                "audit_rdkit",
                "rdkit_adapter",
                "writer_online_decoder",
                "writer_runtime",
                "writer_support",
            },
        )

        self.assertEqual(scan.violations, ())


if __name__ == "__main__":
    unittest.main()
