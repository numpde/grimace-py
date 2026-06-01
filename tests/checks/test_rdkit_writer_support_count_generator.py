from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "generate_rdkit_writer_support_counts.py"


def _load_generator_module():
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        return _load_generator_module_from_path()
    finally:
        sys.path.remove(str(SCRIPT.parent))


def _load_generator_module_from_path():
    spec = importlib.util.spec_from_file_location(
        "generate_rdkit_writer_support_counts",
        SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GENERATOR = _load_generator_module()


class RdkitWriterSupportCountGeneratorTests(unittest.TestCase):
    def test_surface_name_matches_fixture_naming_contract(self) -> None:
        base_flags = {
            "isomericSmiles": False,
            "canonical": False,
            "doRandom": True,
            "kekuleSmiles": False,
            "allBondsExplicit": False,
            "allHsExplicit": False,
            "ignoreAtomMapNumbers": False,
        }

        cases = (
            ({}, "nonisomeric__random"),
            ({"isomericSmiles": True}, "isomeric__random"),
            ({"kekuleSmiles": True}, "nonisomeric__random_kekule"),
            (
                {"allBondsExplicit": True},
                "nonisomeric__random_all_bonds_explicit",
            ),
            ({"allHsExplicit": True}, "nonisomeric__random_all_hs_explicit"),
            (
                {"ignoreAtomMapNumbers": True},
                "nonisomeric__random_ignore_atom_maps",
            ),
            (
                {
                    "isomericSmiles": True,
                    "kekuleSmiles": True,
                    "allBondsExplicit": True,
                    "allHsExplicit": True,
                    "ignoreAtomMapNumbers": True,
                },
                "isomeric__random_kekule_all_bonds_explicit_"
                "all_hs_explicit_ignore_atom_maps",
            ),
        )
        for overrides, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(
                    expected,
                    GENERATOR.surface_name(dict(base_flags, **overrides)),
                )

    def test_missing_variant_estimate_handles_degenerate_counts(self) -> None:
        self.assertEqual(0.0, GENERATOR.estimated_missing_variants(0, 0))
        self.assertEqual(2.0, GENERATOR.estimated_missing_variants(2, 1))
        self.assertEqual(float("inf"), GENERATOR.estimated_missing_variants(1, 0))

    def test_adaptive_saturation_requires_all_evidence_terms(self) -> None:
        kwargs = {
            "draw_count": 20_000,
            "support_count": 304,
            "consecutive_draws_without_new_variant": 12_000,
            "singleton_count": 0,
            "doubleton_count": 0,
            "min_draws": 20_000,
            "unseen_mass_threshold": 0.0001,
            "allowed_missing_variants": 1.0,
        }
        self.assertTrue(GENERATOR.run_is_saturated(**kwargs))

        for field, value in (
            ("draw_count", 19_999),
            ("consecutive_draws_without_new_variant", 9_999),
            ("singleton_count", 3),
        ):
            bad_kwargs = dict(kwargs, **{field: value})
            with self.subTest(field=field):
                self.assertFalse(GENERATOR.run_is_saturated(**bad_kwargs))

    def test_cli_refuses_to_overwrite_existing_output_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.json"
            output_path = Path(tmpdir) / "nonisomeric__random.json"
            input_path.write_text('{"cases": []}\n', encoding="utf-8")
            output_path.write_text("{}\n", encoding="utf-8")

            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--seed",
                    "1",
                    "--seed",
                    "2",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertNotEqual(0, proc.returncode)
        self.assertIn("already exists", proc.stderr)


if __name__ == "__main__":
    unittest.main()
