from __future__ import annotations

import os
import unittest

from grimace._south_star.enum_s import mol_to_smiles_enum_s_graph_native
from tests.helpers.south_star_derived_support import (
    derived_support_proof_for_case,
    load_south_star_derived_support_cases,
    support_digest_sha256,
)
from tests.helpers.south_star_semantic_oracle import graph_signature
from tests.helpers.south_star_semantic_oracle import semantic_signature


SOUTH_STAR_DERIVED_SUPPORT_DIAGNOSTIC_ENV = (
    "RUN_SOUTH_STAR_DERIVED_SUPPORT_DIAGNOSTICS"
)


class SouthStarDerivedSupportFixtureTests(unittest.TestCase):
    def test_derived_fixture_materializes_expected_support_digest(self) -> None:
        cases = load_south_star_derived_support_cases()

        self.assertNotEqual((), cases)
        for case in cases:
            proof = derived_support_proof_for_case(case)
            with self.subTest(case_id=case.case_id):
                self.assertEqual(case.expected_product_count, len(proof.outputs))
                self.assertEqual(
                    case.expected_product_count,
                    proof.estimated_product_size,
                )
                self.assertEqual(
                    tuple(
                        ref.expected_output_count for ref in case.fragment_refs
                    ),
                    proof.fragment_output_counts,
                )
                self.assertEqual(case.fragment_order_policy, proof.fragment_order_policy)
                self.assertEqual(case.output_order_policy, proof.output_order_policy)
                self.assertEqual(case.expected_digest_sha256, proof.digest_sha256)
                self.assertTrue(set(case.sentinel_outputs) <= set(proof.outputs))

    def test_graph_native_runtime_matches_derived_support_digest(self) -> None:
        for case in load_south_star_derived_support_cases():
            proof = derived_support_proof_for_case(case)
            result = mol_to_smiles_enum_s_graph_native(
                case.source_smiles,
                case_id=case.case_id,
            )

            with self.subTest(case_id=case.case_id):
                self.assertEqual(proof.digest_sha256, support_digest_sha256(result.outputs))
                self.assertEqual(case.expected_product_count, len(result.outputs))
                self.assertEqual(proof.outputs, result.outputs)

    def test_derived_support_sentinel_outputs_preserve_semantics(self) -> None:
        for case in load_south_star_derived_support_cases():
            source_graph = graph_signature(case.source_smiles)
            source_semantics = semantic_signature(case.source_smiles)

            with self.subTest(case_id=case.case_id):
                self.assertTrue(
                    all(
                        graph_signature(output) == source_graph
                        and semantic_signature(output) == source_semantics
                        for output in case.sentinel_outputs
                    )
                )

    @unittest.skipUnless(
        os.environ.get(SOUTH_STAR_DERIVED_SUPPORT_DIAGNOSTIC_ENV) == "1",
        f"set {SOUTH_STAR_DERIVED_SUPPORT_DIAGNOSTIC_ENV}=1 to parse "
        "all derived support outputs",
    )
    def test_derived_support_all_outputs_preserve_semantics(self) -> None:
        for case in load_south_star_derived_support_cases():
            proof = derived_support_proof_for_case(case)
            source_graph = graph_signature(case.source_smiles)
            source_semantics = semantic_signature(case.source_smiles)

            with self.subTest(case_id=case.case_id):
                self.assertTrue(
                    all(
                        graph_signature(output) == source_graph
                        and semantic_signature(output) == source_semantics
                        for output in proof.outputs
                    )
                )


if __name__ == "__main__":
    unittest.main()
