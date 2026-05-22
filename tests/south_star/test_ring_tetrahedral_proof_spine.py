from __future__ import annotations

import unittest

from grimace._south_star.enum_s import mol_to_smiles_enum_s_graph_native
from tests.helpers.south_star_ring_tetrahedral_proof_spine import (
    POLYCYCLIC_RING_TETRAHEDRAL_PROOF_AUTHORITY,
    polycyclic_ring_tetrahedral_proof_spine,
)


class SouthStarRingTetrahedralProofSpineTests(unittest.TestCase):
    def test_polycyclic_ring_tetrahedral_spine_handles_clean_witness(self) -> None:
        proof = polycyclic_ring_tetrahedral_proof_spine("F[C@H]1CC2CCC1C2")

        self.assertEqual(
            POLYCYCLIC_RING_TETRAHEDRAL_PROOF_AUTHORITY,
            proof.support_authority,
        )
        self.assertGreater(proof.traversal_count, 0)
        self.assertGreater(proof.output_count, 0)
        self.assertGreater(proof.closure_event_count, 0)
        self.assertGreater(proof.renderer_input_count, 0)
        self.assertEqual(1, proof.obligation_count)

    def test_polycyclic_ring_tetrahedral_spine_handles_minimality_stress(
        self,
    ) -> None:
        proof = polycyclic_ring_tetrahedral_proof_spine("F[C@H]1CC2CC1C2")

        self.assertGreater(proof.traversal_count, 0)
        self.assertGreater(proof.output_count, 0)
        self.assertEqual(1, proof.obligation_count)

    def test_polycyclic_ring_tetrahedral_spine_rejects_supported_monocycle(
        self,
    ) -> None:
        with self.assertRaisesRegex(NotImplementedError, "fused or polycyclic"):
            polycyclic_ring_tetrahedral_proof_spine("F[C@H]1CCCC(C)C1")

    def test_runtime_support_matches_proof_spine_for_domain_witnesses(
        self,
    ) -> None:
        cases = (
            "F[C@H]1CC2CCC1C2",
            "F[C@H]1CC2CC1C2",
            "C1CC2CCCC2[C@H]1F",
            "F[C@H]1CCC2CC1C2",
        )

        for source_smiles in cases:
            with self.subTest(source_smiles=source_smiles):
                proof = polycyclic_ring_tetrahedral_proof_spine(source_smiles)
                result = mol_to_smiles_enum_s_graph_native(source_smiles)

                self.assertEqual(proof.outputs, result.outputs)


if __name__ == "__main__":
    unittest.main()
