from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]
RECORD_PERF_HOTSPOTS_SCRIPT = REPO_ROOT / "scripts" / "record_perf_hotspots.py"
PREPARED_ZSTD_DICT_SCRIPT = (
    REPO_ROOT / "scripts" / "generate_prepared_mol_zstd_dictionary.py"
)


class LocalScriptsSmokeTests(unittest.TestCase):
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

    def test_prepared_mol_zstd_dictionary_generator_help_loads_without_deps(
        self,
    ) -> None:
        proc = subprocess.run(
            [sys.executable, "-I", str(PREPARED_ZSTD_DICT_SCRIPT), "--help"],
            cwd=REPO_ROOT,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
        self.assertIn(
            "Generate the production PreparedMol zstd dictionary artifact",
            proc.stdout,
        )


if __name__ == "__main__":
    unittest.main()
