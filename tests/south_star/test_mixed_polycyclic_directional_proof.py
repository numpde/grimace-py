from __future__ import annotations

import unittest

from tests.helpers.south_star_mixed_polycyclic_directional_proof import (
    mixed_polycyclic_directional_proof,
)


class SouthStarMixedPolycyclicDirectionalProofTests(unittest.TestCase):
    def test_connected_supported_proof_generates_semantic_outputs(self) -> None:
        proof = mixed_polycyclic_directional_proof("F[C@H]1CC2CCC1C2/C=C/Cl")

        self.assertEqual((), proof.unsupported_categories)
        self.assertEqual((1,), proof.ring_tetrahedral_center_atom_indices)
        self.assertEqual(("component:0",), proof.directional_component_ids)
        self.assertEqual(("bond:8",), proof.directional_feature_ids)
        self.assertEqual(1, proof.ring_tetrahedral_obligation_count)
        self.assertEqual(1, proof.directional_component_count)
        self.assertEqual(0, proof.directional_coupling_cause_count)
        self.assertEqual(2, proof.component_assignment_count)
        self.assertEqual(3160, proof.traversal_count)
        self.assertEqual(3160, proof.raw_output_count)
        self.assertEqual(3160, proof.output_count)
        self.assertGreater(proof.closure_event_count, 0)
        self.assertGreater(proof.marker_slot_count, 0)
        self.assertGreater(proof.renderer_input_count, 0)
        self.assertTrue(proof.semantic_parseback_passed)

    def test_disconnected_frontier_remains_second_wave(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "one fragment"):
            mixed_polycyclic_directional_proof("F[C@H]1CC2CCC1C2.F/C=C/Cl")

    def test_polycyclic_tetra_without_directional_component_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            NotImplementedError,
            "exactly one directional component",
        ):
            mixed_polycyclic_directional_proof("F[C@H]1CC2CCC1C2")


if __name__ == "__main__":
    unittest.main()
