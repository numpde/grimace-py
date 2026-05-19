from __future__ import annotations

import unittest

from tests import run_south_star_semantics


class SouthStarHarnessTests(unittest.TestCase):
    def test_runner_is_dedicated_to_south_star_package(self) -> None:
        self.assertNotEqual((), run_south_star_semantics.SOUTH_STAR_SEMANTIC_MODULES)
        self.assertTrue(
            all(
                module_name.startswith("tests.south_star.")
                for module_name in run_south_star_semantics.SOUTH_STAR_SEMANTIC_MODULES
            )
        )

    def test_runner_does_not_name_rdkit_writer_parity_modules(self) -> None:
        forbidden_fragments = (
            "pinned_rdkit",
            "rdkit_serialization",
            "run_pinned_rdkit_parity",
            "writer_membership",
        )

        for module_name in run_south_star_semantics.SOUTH_STAR_SEMANTIC_MODULES:
            with self.subTest(module_name=module_name):
                self.assertFalse(
                    any(fragment in module_name for fragment in forbidden_fragments)
                )
