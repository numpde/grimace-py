from __future__ import annotations

import unittest

from grimace._south_star.enum_s import mol_to_smiles_enum_s_graph_native_for_case
from tests.helpers.south_star_exact_support import (
    load_south_star_exact_first_domain_cases,
)
from tests.helpers.south_star_first_domain_oracle import (
    independent_first_domain_support_for_case,
)
from tests.helpers.south_star_first_domain_proof_inputs import (
    first_domain_proof_inputs_from_shared_spine,
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

    def test_temporary_witness_matches_exact_first_domain_fixtures(self) -> None:
        # This witness is intentionally separate from graph-native EnumS. It is
        # completeness evidence for the declared first-domain language, not a
        # second implementation path to promote into runtime code.
        semantic_cases = {
            case.case_id: case for case in load_south_star_semantic_cases()
        }
        for exact_case in load_south_star_exact_first_domain_cases():
            semantic_case = semantic_cases[exact_case.case_id]
            oracle_support = independent_first_domain_support_for_case(semantic_case)

            with self.subTest(case_id=exact_case.case_id):
                self.assertEqual(
                    frozenset(exact_case.expected_support),
                    frozenset(oracle_support),
                )

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

    def test_first_domain_proof_inputs_come_from_shared_spine(self) -> None:
        semantic_cases = {
            case.case_id: case for case in load_south_star_semantic_cases()
        }
        for exact_case in load_south_star_exact_first_domain_cases():
            semantic_case = semantic_cases[exact_case.case_id]
            proof_inputs = first_domain_proof_inputs_from_shared_spine(semantic_case)

            with self.subTest(case_id=exact_case.case_id):
                self.assertEqual(exact_case.case_id, proof_inputs.case_id)
                self.assertEqual(exact_case.source_smiles, proof_inputs.source_smiles)
                self.assertFalse(proof_inputs.expected_support_strings_used)
                self.assertEqual(
                    "maximal_eligible_carrier",
                    proof_inputs.annotation_policy_name,
                )
                self.assertGreater(proof_inputs.atom_count, 0)
                self.assertGreater(proof_inputs.bond_count, 0)
                self.assertGreater(proof_inputs.component_count, 0)
                self.assertGreater(proof_inputs.carrier_opportunity_count, 0)
                self.assertGreater(proof_inputs.traversal_count, 0)
                self.assertGreater(proof_inputs.traversal_event_count, 0)
                self.assertGreater(proof_inputs.marker_slot_count, 0)
                self.assertGreater(proof_inputs.carrier_context_count, 0)
                self.assertEqual(0, proof_inputs.renderer_input_count)
                self.assertEqual(
                    proof_inputs.marker_slot_count,
                    proof_inputs.equation_count,
                )


if __name__ == "__main__":
    unittest.main()
