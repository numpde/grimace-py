from __future__ import annotations

import unittest

from tests.helpers.south_star_enum_s import mol_to_smiles_enum_s_graph_native_for_case
from tests.helpers.south_star_exact_support import (
    load_south_star_exact_first_domain_cases,
)
from tests.helpers.south_star_semantic_oracle import semantic_oracle_accepts
from tests.helpers.south_star_semantics import load_south_star_semantic_cases


class SouthStarFirstDomainCompletenessTests(unittest.TestCase):
    def test_graph_native_support_matches_exact_first_domain_fixtures(self) -> None:
        semantic_cases = {
            case.case_id: case for case in load_south_star_semantic_cases()
        }
        for exact_case in load_south_star_exact_first_domain_cases():
            semantic_case = semantic_cases[exact_case.case_id]
            result = mol_to_smiles_enum_s_graph_native_for_case(semantic_case)

            with self.subTest(case_id=exact_case.case_id):
                self.assertEqual(exact_case.source_smiles, semantic_case.source_smiles)
                self.assertEqual(exact_case.expected_support, result.outputs)

    def test_exact_first_domain_fixtures_are_semantic_evidence(self) -> None:
        for exact_case in load_south_star_exact_first_domain_cases():
            for smiles in exact_case.expected_support:
                with self.subTest(case_id=exact_case.case_id, smiles=smiles):
                    self.assertTrue(
                        semantic_oracle_accepts(
                            source_smiles=exact_case.source_smiles,
                            candidate_smiles=smiles,
                        )
                    )


if __name__ == "__main__":
    unittest.main()
