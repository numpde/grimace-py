from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
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

    def _run_controller(self, *extra_args: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = "python:." if not existing_pythonpath else f"python:.:{existing_pythonpath}"
        return subprocess.run(
            [sys.executable, str(SCRIPT_PATH), *extra_args],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

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

    def test_load_resume_state_uses_last_recorded_cid(self) -> None:
        config = MINER.ScanConfig(
            root_mode="none",
            isomeric_smiles=True,
            kekule_smiles=False,
            all_bonds_explicit=False,
            all_hs_explicit=False,
            ignore_atom_map_numbers=False,
            rdkit_mode="sampled",
            draws_per_round=40,
            stagnation_rounds=5,
            max_draws=400,
            seed=12345,
            connected_mode="connected",
            max_atoms=30,
            limit=100,
            start_after=None,
            timeout=12.0,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "scan.jsonl"
            path.write_text(
                "\n".join(
                    (
                        json.dumps({"record_type": "mode", "mode": MINER.asdict(config)}),
                        json.dumps({"record_type": "case", "cid": "100", "checked": 1}),
                        json.dumps({"record_type": "timeout", "cid": "200", "checked": 2}),
                        json.dumps({"record_type": "stop", "reason": "limit", "checked": 2}),
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            state = MINER._load_resume_state(str(path), config)

        self.assertEqual(2, state.checked)
        self.assertEqual("200", state.start_after)

    def test_load_resume_state_rejects_incompatible_mode(self) -> None:
        config = MINER.ScanConfig(
            root_mode="none",
            isomeric_smiles=True,
            kekule_smiles=False,
            all_bonds_explicit=False,
            all_hs_explicit=False,
            ignore_atom_map_numbers=False,
            rdkit_mode="deterministic",
            draws_per_round=40,
            stagnation_rounds=5,
            max_draws=400,
            seed=12345,
            connected_mode="connected",
            max_atoms=30,
            limit=100,
            start_after=None,
            timeout=12.0,
        )
        incompatible_mode = {
            "root_mode": "zero",
            "isomeric_smiles": True,
            "kekule_smiles": False,
            "all_bonds_explicit": False,
            "all_hs_explicit": False,
            "ignore_atom_map_numbers": False,
            "rdkit_mode": "deterministic",
            "draws_per_round": 40,
            "stagnation_rounds": 5,
            "max_draws": 400,
            "seed": 12345,
            "connected_mode": "connected",
            "max_atoms": 30,
            "limit": 100,
            "start_after": None,
            "timeout": 12.0,
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "scan.jsonl"
            path.write_text(
                json.dumps({"record_type": "mode", "mode": incompatible_mode}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "does not match"):
                MINER._load_resume_state(str(path), config)

    def test_controller_jsonl_resume_continues_checked_counter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_path = Path(tmp_dir) / "scan.jsonl"
            base_args = (
                "--root",
                "none",
                "--isomeric",
                "false",
                "--connected",
                "connected",
                "--max-atoms",
                "4",
                "--timeout",
                "5",
                "--jsonl-output",
                str(jsonl_path),
            )
            first = self._run_controller(
                *base_args,
                "--limit",
                "2",
            )
            self.assertEqual(0, first.returncode, msg=first.stderr or first.stdout)

            second = self._run_controller(
                *base_args,
                "--limit",
                "4",
                "--resume-jsonl",
            )
            self.assertEqual(0, second.returncode, msg=second.stderr or second.stdout)

            records = [
                json.loads(line)
                for line in jsonl_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        case_records = [record for record in records if record.get("record_type") == "case"]
        mode_records = [record for record in records if record.get("record_type") == "mode"]
        self.assertEqual([1, 2, 3, 4], [record["checked"] for record in case_records])
        self.assertEqual(2, len(mode_records))
        self.assertEqual(0, mode_records[0]["resume_checked"])
        self.assertEqual(2, mode_records[1]["resume_checked"])
        self.assertEqual(case_records[1]["cid"], mode_records[1]["resume_start_after"])


if __name__ == "__main__":
    unittest.main()
