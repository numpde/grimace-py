from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "mine_rdkit_writer_support_count_candidates.py"


def _load_miner_module():
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        spec = importlib.util.spec_from_file_location(
            "mine_rdkit_writer_support_count_candidates",
            SCRIPT,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to load {SCRIPT}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPT.parent))


MINER = _load_miner_module()
SURFACES = sys.modules["rdkit_writer_support_count_surfaces"]


class RdkitWriterSupportCountCandidateMinerTests(unittest.TestCase):
    def test_surface_flags_are_explicit_supported_writer_surface(self) -> None:
        flags = MINER.surface_flags("nonisomeric__random")

        self.assertEqual(
            {
                "isomericSmiles": False,
                "canonical": False,
                "doRandom": True,
                "kekuleSmiles": False,
                "allBondsExplicit": False,
                "allHsExplicit": False,
                "ignoreAtomMapNumbers": False,
            },
            flags,
        )

        flags["canonical"] = True
        self.assertFalse(MINER.surface_flags("nonisomeric__random")["canonical"])

    def test_candidate_mining_surfaces_roundtrip_to_shard_names(self) -> None:
        for surface in MINER.CANDIDATE_MINING_SURFACE_FLAGS:
            with self.subTest(surface=surface):
                self.assertEqual(surface, SURFACES.surface_name(MINER.surface_flags(surface)))

    def test_case_id_is_stable_and_generator_safe(self) -> None:
        self.assertEqual(
            "pubchem_2244_nonisomeric_random_unrooted",
            MINER.case_id_for(
                cid="2244",
                surface="nonisomeric__random",
                rooted_at_atom=-1,
            ),
        )
        self.assertEqual(
            "pubchem_2244_isomeric_random_root0",
            MINER.case_id_for(
                cid="2244",
                surface="isomeric__random",
                rooted_at_atom=0,
            ),
        )

    def test_support_buckets_match_mining_policy(self) -> None:
        self.assertEqual("small", MINER.support_bucket(100))
        self.assertEqual("medium", MINER.support_bucket(101))
        self.assertEqual("medium", MINER.support_bucket(2_000))
        self.assertEqual("large", MINER.support_bucket(2_001))
        self.assertEqual("too_large", MINER.support_bucket(10_001))

    def test_cli_refuses_to_overwrite_existing_output_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "candidates.json"
            output_path.write_text("{}\n", encoding="utf-8")

            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--output",
                    str(output_path),
                    "--limit",
                    "0",
                    "--allow-outside-repo",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertNotEqual(0, proc.returncode)
        self.assertIn("already exists", proc.stderr)

    def test_cli_rejects_outside_repo_output_without_explicit_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--output",
                    str(Path(tmpdir) / "candidates.json"),
                    "--limit",
                    "0",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertNotEqual(0, proc.returncode)
        self.assertIn("output path must be under", proc.stderr)


if __name__ == "__main__":
    unittest.main()
