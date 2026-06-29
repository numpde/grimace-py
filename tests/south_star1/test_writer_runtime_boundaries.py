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
                "enumerate_prepared_writer_shaped_support",
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
                "initial_writer_frontier_cursor",
                "iter_writer_frontier_support",
            },
        )
        snapshot_imports = import_from_observations(
            WRITER_SUPPORT_PATH,
            module_root="writer_snapshot",
        )

        self.assertEqual(scan.violations, ())
        self.assertEqual(len(snapshot_imports), 1)
        self.assertTrue(snapshot_imports[0].inside_type_checking)


if __name__ == "__main__":
    unittest.main()
