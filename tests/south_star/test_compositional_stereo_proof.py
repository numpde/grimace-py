from __future__ import annotations

import unittest

from tests.helpers.south_star_compositional_stereo_proof import (
    compositional_stereo_proof_report,
)


class SouthStarCompositionalStereoProofTests(unittest.TestCase):
    def test_separated_tetrahedral_centers_are_independent_product(self) -> None:
        report = compositional_stereo_proof_report("F[C@H](Cl)C[C@H](Br)I")

        self.assertTrue(report.supported)
        self.assertEqual("independent_product", report.classification)
        self.assertEqual(("tetrahedral:1", "tetrahedral:4"), _obligation_ids(report))
        self.assertEqual(
            (("tetrahedral:1",), ("tetrahedral:4",)),
            _component_obligation_ids(report),
        )
        self.assertEqual(4, report.assignment_count_before_rendering)
        self.assertEqual(48, report.proof_output_count)
        self.assertEqual(48, report.runtime_output_count)
        self.assertTrue(report.runtime_outputs_match_proof)
        self.assertTrue(report.semantic_parseback_passed)

    def test_adjacent_tetrahedral_centers_are_coupled_component(self) -> None:
        report = compositional_stereo_proof_report("F[C@H](Cl)[C@H](Br)I")

        self.assertTrue(report.supported)
        self.assertEqual("coupled_component", report.classification)
        self.assertEqual(
            (("tetrahedral:1", "tetrahedral:3"),),
            _component_obligation_ids(report),
        )
        self.assertEqual(
            ("adjacent_tetrahedral_centers",),
            report.components[0].coupling_reasons,
        )
        self.assertEqual(4, report.assignment_count_before_rendering)
        self.assertEqual(40, report.proof_output_count)
        self.assertEqual(40, report.runtime_output_count)
        self.assertTrue(report.runtime_outputs_match_proof)
        self.assertTrue(report.semantic_parseback_passed)

    def test_disconnected_tetrahedral_fragments_are_independent_product(self) -> None:
        report = compositional_stereo_proof_report("F[C@H](Cl)Br.F[C@H](Cl)I")

        self.assertTrue(report.supported)
        self.assertEqual("independent_product", report.classification)
        self.assertEqual(
            (("tetrahedral:1",), ("tetrahedral:5",)),
            _component_obligation_ids(report),
        )
        self.assertEqual(4, report.assignment_count_before_rendering)
        self.assertEqual(288, report.proof_output_count)
        self.assertEqual(288, report.runtime_output_count)
        self.assertTrue(report.runtime_outputs_match_proof)
        self.assertTrue(report.semantic_parseback_passed)

    def test_shared_directional_and_tetrahedral_atoms_are_coupled(self) -> None:
        report = compositional_stereo_proof_report("F/C=C/[C@H](Cl)Br")

        self.assertTrue(report.supported)
        self.assertEqual("coupled_component", report.classification)
        self.assertEqual(
            (("tetrahedral:3", "directional:component:0"),),
            _component_obligation_ids(report),
        )
        self.assertEqual(
            ("shared_directional_obligation_atom",),
            report.components[0].coupling_reasons,
        )
        self.assertEqual(40, report.proof_output_count)
        self.assertEqual(40, report.runtime_output_count)
        self.assertTrue(report.runtime_outputs_match_proof)

    def test_separated_directional_components_remain_independent_product(self) -> None:
        report = compositional_stereo_proof_report("F/C=C/C/C=C/Cl")

        self.assertTrue(report.supported)
        self.assertEqual("independent_product", report.classification)
        self.assertEqual(
            (("directional:component:0",), ("directional:component:1",)),
            _component_obligation_ids(report),
        )
        self.assertEqual(4, report.assignment_count_before_rendering)
        self.assertEqual(48, report.proof_output_count)
        self.assertEqual(48, report.runtime_output_count)
        self.assertTrue(report.runtime_outputs_match_proof)

    def test_mixed_ring_tetrahedral_centers_share_ring_system(self) -> None:
        report = compositional_stereo_proof_report(
            "F[C@H]1CCCC([C@H](Cl)Br)C1"
        )

        self.assertTrue(report.supported)
        self.assertEqual("coupled_component", report.classification)
        self.assertEqual(
            (("tetrahedral:1", "tetrahedral:6"),),
            _component_obligation_ids(report),
        )
        self.assertEqual(
            ("shared_ring_tetrahedral_system",),
            report.components[0].coupling_reasons,
        )
        self.assertEqual(576, report.proof_output_count)
        self.assertEqual(576, report.runtime_output_count)
        self.assertTrue(report.runtime_outputs_match_proof)

    def test_polycyclic_mixed_boundary_remains_unsupported(self) -> None:
        report = compositional_stereo_proof_report("F[C@H]1CC2CCC1C2/C=C/Cl")

        self.assertFalse(report.supported)
        self.assertIn("fused_or_polycyclic_ring", report.unsupported_categories)
        self.assertIn("ring_tetrahedral_interaction", report.unsupported_categories)
        self.assertIsNone(report.proof_output_count)
        self.assertIsNone(report.runtime_output_count)
        self.assertIsNone(report.runtime_outputs_match_proof)
        self.assertIsNone(report.semantic_parseback_passed)


def _obligation_ids(report):
    return tuple(obligation.obligation_id for obligation in report.obligations)


def _component_obligation_ids(report):
    return tuple(component.obligation_ids for component in report.components)


if __name__ == "__main__":
    unittest.main()
