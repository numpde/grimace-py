from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest

from tests.helpers.kernel import CORE_MODULE


REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATE_REFERENCE_ARTIFACTS_SCRIPT = REPO_ROOT / "scripts" / "generate_reference_artifacts.py"
RECORD_PERF_HOTSPOTS_SCRIPT = REPO_ROOT / "scripts" / "record_perf_hotspots.py"


class LocalScriptsSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if CORE_MODULE is None:
            raise unittest.SkipTest("private Rust extension is not installed")

    def test_generate_reference_artifacts_script_can_noop_import_cleanly(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                str(GENERATE_REFERENCE_ARTIFACTS_SCRIPT),
                "--skip-core",
                "--skip-metrics",
            ],
            cwd=REPO_ROOT,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)

    def test_record_perf_hotspots_help_loads_without_source_tree_pythonpath(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(RECORD_PERF_HOTSPOTS_SCRIPT), "--help"],
            cwd=REPO_ROOT,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
        self.assertIn("Record an internal perf-hotspot history entry", proc.stdout)


if __name__ == "__main__":
    unittest.main()
