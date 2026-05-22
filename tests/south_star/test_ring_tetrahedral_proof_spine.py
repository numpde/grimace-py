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
        with self.assertRaisesRegex(NotImplementedError, "frontier"):
            polycyclic_ring_tetrahedral_proof_spine("F[C@H]1CCCC(C)C1")

    def test_proof_spine_does_not_open_runtime_support(self) -> None:
        with self.assertRaisesRegex(
            NotImplementedError,
            "fused_or_polycyclic_ring|ring_tetrahedral_interaction",
        ):
            mol_to_smiles_enum_s_graph_native("F[C@H]1CC2CCC1C2")


if __name__ == "__main__":
    unittest.main()
