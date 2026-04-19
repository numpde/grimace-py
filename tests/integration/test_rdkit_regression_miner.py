from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from tests.helpers.kernel import CORE_MODULE


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "mine_rdkit_regressions.py"


def _load_miner_module():
    spec = importlib.util.spec_from_file_location("mine_rdkit_regressions", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load miner script from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MINER = _load_miner_module()


class RdkitRegressionMinerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if CORE_MODULE is None:
            raise unittest.SkipTest("private Rust extension is not installed")

    def test_support_comparison_classifies_clean_uncertain_and_mismatches(self) -> None:
        self.assertEqual(
            "clean",
            MINER._classify_support_comparison(
                sampled_support={"A"},
                grimace_support={"A"},
                plateau_reached=False,
            )["status"],
        )
        self.assertEqual(
            "uncertain",
            MINER._classify_support_comparison(
                sampled_support={"A"},
                grimace_support={"A", "B"},
                plateau_reached=False,
            )["status"],
        )
        grimace_only = MINER._classify_support_comparison(
            sampled_support={"A"},
            grimace_support={"A", "B"},
            plateau_reached=True,
        )
        self.assertEqual("grimace_only", grimace_only["status"])
        self.assertEqual(["B"], grimace_only["grimace_only_preview"])

        rdkit_only = MINER._classify_support_comparison(
            sampled_support={"A", "C"},
            grimace_support={"A", "B"},
            plateau_reached=True,
        )
        self.assertEqual("rdkit_only", rdkit_only["status"])
        self.assertEqual(["C"], rdkit_only["rdkit_only_preview"])

    def _run_worker(self, *extra_args: str) -> dict[str, object]:
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = "python:." if not existing_pythonpath else f"python:.:{existing_pythonpath}"
        proc = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), *extra_args],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
        return json.loads(proc.stdout.strip())

    def test_worker_sampled_mode_reports_clean_plateau_case(self) -> None:
        payload = self._run_worker(
            "--worker",
            "--smiles",
            "CC",
            "--rooted-at-atom",
            "0",
            "--isomeric",
            "false",
            "--kekule",
            "false",
            "--all-bonds-explicit",
            "false",
            "--all-hs-explicit",
            "false",
            "--ignore-atom-map-numbers",
            "false",
            "--rdkit-mode",
            "sampled",
            "--draws-per-round",
            "3",
            "--stagnation-rounds",
            "2",
            "--max-draws",
            "12",
            "--seed",
            "123",
        )

        self.assertEqual("clean", payload["status"])
        self.assertEqual(1, payload["support_size"])
        self.assertEqual(1, payload["sampled_size"])
        self.assertTrue(payload["plateau_reached"])
        self.assertTrue(payload["contains"])

    def test_worker_sampled_mode_reports_uncertain_when_budget_ends_early(self) -> None:
        payload = self._run_worker(
            "--worker",
            "--smiles",
            "CCO",
            "--rooted-at-atom",
            "-1",
            "--isomeric",
            "false",
            "--kekule",
            "false",
            "--all-bonds-explicit",
            "false",
            "--all-hs-explicit",
            "false",
            "--ignore-atom-map-numbers",
            "false",
            "--rdkit-mode",
            "sampled",
            "--draws-per-round",
            "1",
            "--stagnation-rounds",
            "3",
            "--max-draws",
            "1",
            "--seed",
            "123",
        )

        self.assertEqual("uncertain", payload["status"])
        self.assertFalse(payload["plateau_reached"])
        self.assertTrue(payload["contains"])
        self.assertLess(payload["sampled_size"], payload["support_size"])


if __name__ == "__main__":
    unittest.main()
