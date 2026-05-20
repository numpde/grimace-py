from __future__ import annotations

import unittest

from tests.helpers.south_star_component_support_state import (
    SouthStarComponentSupportState,
)
from tests.helpers.south_star_enum_s import mol_to_smiles_enum_s_prototype_for_case
from tests.helpers.south_star_semantics import load_south_star_semantic_cases


class SouthStarEnumSPrototypeTests(unittest.TestCase):
    def test_prototype_returns_fixture_positive_semantic_outputs(self) -> None:
        for case in load_south_star_semantic_cases():
            result = mol_to_smiles_enum_s_prototype_for_case(case)

            with self.subTest(case_id=case.case_id):
                self.assertEqual(case.case_id, result.case_id)
                self.assertEqual(case.positive_semantic_smiles, result.outputs)
                self.assertEqual(
                    "south_star_semantic_fixture_witnesses",
                    result.generation_basis,
                )

    def test_prototype_excludes_negative_semantic_witnesses(self) -> None:
        for case in load_south_star_semantic_cases():
            result = mol_to_smiles_enum_s_prototype_for_case(case)
            negative_outputs = {
                negative.smiles for negative in case.negative_semantic_smiles
            }

            with self.subTest(case_id=case.case_id):
                self.assertFalse(negative_outputs.intersection(result.outputs))

    def test_prototype_exposes_support_state_complexity_snapshot(self) -> None:
        for case in load_south_star_semantic_cases():
            result = mol_to_smiles_enum_s_prototype_for_case(case)
            expected = SouthStarComponentSupportState.from_case(
                case
            ).complexity_snapshot()

            with self.subTest(case_id=case.case_id):
                self.assertEqual(expected, result.complexity_snapshot)
