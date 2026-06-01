from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "generate_rdkit_writer_support_counts.py"


def _load_generator_module():
    spec = importlib.util.spec_from_file_location(
        "generate_rdkit_writer_support_counts",
        SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATOR = _load_generator_module()


class RdkitWriterSupportCountGeneratorTests(unittest.TestCase):
    def test_surface_name_matches_fixture_naming_contract(self) -> None:
        flags = {
            "isomericSmiles": False,
            "canonical": False,
            "doRandom": True,
            "kekuleSmiles": False,
            "allBondsExplicit": False,
            "allHsExplicit": False,
            "ignoreAtomMapNumbers": False,
        }

        self.assertEqual("nonisomeric__random", GENERATOR.surface_name(flags))

        flags["isomericSmiles"] = True
        self.assertEqual("isomeric__random", GENERATOR.surface_name(flags))

        flags["isomericSmiles"] = False
        flags["kekuleSmiles"] = True
        self.assertEqual("nonisomeric__random_kekule", GENERATOR.surface_name(flags))

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


if __name__ == "__main__":
    unittest.main()
