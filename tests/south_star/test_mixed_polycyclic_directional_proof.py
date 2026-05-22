from __future__ import annotations

import unittest

from grimace._south_star.enum_s import mol_to_smiles_enum_s_graph_native
from grimace._south_star.support_gates import south_star_support_gate_report
from tests.helpers.south_star_mixed_polycyclic_directional_proof import (
    mixed_polycyclic_directional_proof,
)
from tests.helpers.south_star_semantic_oracle import graph_signature
from tests.helpers.south_star_semantic_oracle import parse_smiles
from tests.helpers.south_star_semantic_oracle import semantic_signature


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

    def test_disconnected_runtime_composes_supported_fragments(self) -> None:
        source_smiles = "F[C@H]1CC2CCC1C2.F/C=C/Cl"

        report = south_star_support_gate_report(parse_smiles(source_smiles))
        result = mol_to_smiles_enum_s_graph_native(
            source_smiles,
            case_id="disconnected_polycyclic_tetrahedral_directional_fragments",
        )
        diagnostics = result.generation_diagnostics
        if diagnostics is None:
            raise AssertionError("disconnected stress case requires diagnostics")

        self.assertTrue(report.supported, report.unsupported_features)
        self.assertEqual(2, diagnostics.fragment_count)
        self.assertEqual((784, 12), diagnostics.fragment_output_counts)
        self.assertEqual(2, diagnostics.fragment_order_count)
        self.assertEqual(18816, diagnostics.estimated_product_size)
        self.assertEqual(18816, diagnostics.raw_output_count)
        self.assertEqual(18816, diagnostics.output_count)

        source_graph = graph_signature(source_smiles)
        source_semantics = semantic_signature(source_smiles)
        sentinel_outputs = (
            result.outputs[0],
            result.outputs[len(result.outputs) // 2],
            result.outputs[-1],
        )
        self.assertTrue(
            all(
                graph_signature(output) == source_graph
                and semantic_signature(output) == source_semantics
                for output in sentinel_outputs
            )
        )

    def test_polycyclic_tetra_without_directional_component_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            NotImplementedError,
            "exactly one directional component",
        ):
            mixed_polycyclic_directional_proof("F[C@H]1CC2CCC1C2")


if __name__ == "__main__":
    unittest.main()
