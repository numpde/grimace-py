from __future__ import annotations

import unittest

from tests.helpers.south_star_enum_s import mol_to_smiles_enum_s_prototype_for_case
from tests.helpers.south_star_output_correctness import (
    evaluate_south_star_output_correctness,
)
from tests.helpers.south_star_semantics import load_south_star_semantic_cases


class SouthStarOutputCorrectnessHarnessTests(unittest.TestCase):
    def test_every_enum_s_prototype_output_passes_semantic_conformance(self) -> None:
        for case in load_south_star_semantic_cases():
            enum_result = mol_to_smiles_enum_s_prototype_for_case(case)
            correctness_results = evaluate_south_star_output_correctness(
                case=case,
                enum_result=enum_result,
            )

            with self.subTest(case_id=case.case_id):
                self.assertEqual(len(enum_result.outputs), len(correctness_results))

            for correctness in correctness_results:
                with self.subTest(
                    case_id=case.case_id,
                    output=correctness.output,
                ):
                    self.assertTrue(
                        correctness.report.accepted,
                        correctness.report.rejection_reasons,
                    )
                    self.assertTrue(correctness.report.rdkit_parseability.passed)
                    self.assertTrue(correctness.report.graph_equivalence.passed)
                    self.assertTrue(correctness.report.stereo_equivalence.passed)

    def test_harness_rejects_mismatched_case_and_enum_result(self) -> None:
        cases = load_south_star_semantic_cases()
        self.assertGreaterEqual(len(cases), 2)
        enum_result = mol_to_smiles_enum_s_prototype_for_case(cases[0])

        with self.assertRaisesRegex(ValueError, "does not match"):
            evaluate_south_star_output_correctness(
                case=cases[1],
                enum_result=enum_result,
            )
