from __future__ import annotations

import unittest

from rdkit import Chem

from tests.helpers.south_star_semantic_oracle import parse_smiles
from tests.helpers.south_star_semantics import load_south_star_semantic_cases
from tests.helpers.south_star_support_gates import south_star_support_gate_report


class SouthStarSupportGateTests(unittest.TestCase):
    def test_current_semantic_fixtures_are_inside_first_gate_scope(self) -> None:
        for case in load_south_star_semantic_cases():
            report = south_star_support_gate_report(parse_smiles(case.source_smiles))

            with self.subTest(case_id=case.case_id):
                self.assertTrue(report.supported, report.unsupported_features)

    def test_atom_stereo_is_fail_fast_unsupported(self) -> None:
        report = south_star_support_gate_report(parse_smiles("C[C@H](F)Cl"))

        self.assertIn("atom_stereo", report.categories)
        with self.assertRaisesRegex(NotImplementedError, "atom_stereo"):
            report.fail_if_unsupported()

    def test_dative_and_metal_surfaces_are_fail_fast_unsupported(self) -> None:
        report = south_star_support_gate_report(parse_smiles("[NH3]->[Cu]"))

        self.assertIn("dative_bond", report.categories)
        self.assertIn("metal_atom", report.categories)

    def test_query_surfaces_are_fail_fast_unsupported(self) -> None:
        query = Chem.MolFromSmarts("[#6]-[#8]")
        self.assertIsNotNone(query)
        report = south_star_support_gate_report(query)

        self.assertIn("query_atom", report.categories)
        self.assertIn("query_bond", report.categories)

    def test_disconnected_molecules_are_fail_fast_unsupported(self) -> None:
        report = south_star_support_gate_report(parse_smiles("C/C=C\\C.CCO"))

        self.assertIn("disconnected_molecule", report.categories)

    def test_ring_stereo_is_fail_fast_unsupported(self) -> None:
        report = south_star_support_gate_report(parse_smiles("C1/C=C\\CCCCC1"))

        self.assertIn("ring_stereo", report.categories)
