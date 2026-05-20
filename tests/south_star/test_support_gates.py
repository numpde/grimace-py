from __future__ import annotations

import unittest

from rdkit import Chem

from tests.helpers.south_star_domain_manifest import SOUTH_STAR_PRIVATE_DOMAIN
from tests.helpers.south_star_semantic_oracle import parse_smiles
from tests.helpers.south_star_semantics import load_south_star_semantic_cases
from grimace._south_star.support_gates import south_star_support_gate_report


class SouthStarSupportGateTests(unittest.TestCase):
    def assertUnsupportedCategory(self, category: str, categories: frozenset[str]) -> None:
        self.assertIn(category, SOUTH_STAR_PRIVATE_DOMAIN.unsupported_feature_categories)
        self.assertIn(category, categories)

    def test_current_semantic_fixtures_are_inside_first_gate_scope(self) -> None:
        for case in load_south_star_semantic_cases():
            report = south_star_support_gate_report(parse_smiles(case.source_smiles))

            with self.subTest(case_id=case.case_id):
                self.assertTrue(report.supported, report.unsupported_features)

    def test_supported_tetrahedral_atom_stereo_is_inside_gate_scope(self) -> None:
        report = south_star_support_gate_report(parse_smiles("C[C@H](F)Cl"))

        self.assertTrue(report.supported, report.unsupported_features)

    def test_unsupported_atom_stereo_is_fail_fast_unsupported(self) -> None:
        mol = parse_smiles("CCF")
        mol.GetAtomWithIdx(1).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        report = south_star_support_gate_report(mol)

        self.assertUnsupportedCategory("atom_stereo", report.categories)
        with self.assertRaisesRegex(NotImplementedError, "atom_stereo"):
            report.fail_if_unsupported()

    def test_dative_and_metal_surfaces_are_fail_fast_unsupported(self) -> None:
        report = south_star_support_gate_report(parse_smiles("[NH3]->[Cu]"))

        self.assertUnsupportedCategory("dative_bond", report.categories)
        self.assertUnsupportedCategory("metal_atom", report.categories)

    def test_query_surfaces_are_fail_fast_unsupported(self) -> None:
        query = Chem.MolFromSmarts("[#6]-[#8]")
        self.assertIsNotNone(query)
        report = south_star_support_gate_report(query)

        self.assertUnsupportedCategory("query_atom", report.categories)
        self.assertUnsupportedCategory("query_bond", report.categories)

    def test_supported_disconnected_stereo_fragments_are_inside_gate_scope(self) -> None:
        report = south_star_support_gate_report(parse_smiles("F/C=C\\Cl.O"))

        self.assertTrue(report.supported, report.unsupported_features)

    def test_unsupported_disconnected_fragments_are_fail_fast_unsupported(self) -> None:
        report = south_star_support_gate_report(parse_smiles("C#N.O"))

        self.assertUnsupportedCategory("disconnected_molecule", report.categories)

    def test_markerless_disconnected_fragments_are_supported(self) -> None:
        report = south_star_support_gate_report(parse_smiles("C1CCCCC1.O"))

        self.assertTrue(report.supported, report.unsupported_features)

    def test_simple_saturated_monocycles_are_supported(self) -> None:
        report = south_star_support_gate_report(parse_smiles("C1CCCCC1"))

        self.assertTrue(report.supported, report.unsupported_features)

    def test_branched_saturated_monocycles_are_supported(self) -> None:
        report = south_star_support_gate_report(parse_smiles("CC1CCCCC1"))

        self.assertTrue(report.supported, report.unsupported_features)

    def test_unsaturated_nonstereo_monocycles_are_supported(self) -> None:
        report = south_star_support_gate_report(parse_smiles("C1=CCCCC1"))

        self.assertTrue(report.supported, report.unsupported_features)

    def test_fused_rings_are_fail_fast_unsupported(self) -> None:
        report = south_star_support_gate_report(parse_smiles("C1CC2CCCC2C1"))

        self.assertUnsupportedCategory("ring_molecule", report.categories)

    def test_ring_stereo_has_specific_fail_fast_reason(self) -> None:
        report = south_star_support_gate_report(parse_smiles("C1/C=C\\CCCCC1"))

        self.assertUnsupportedCategory("ring_stereo", report.categories)

    def test_unsupported_atom_text_is_fail_fast_unsupported(self) -> None:
        report = south_star_support_gate_report(parse_smiles("[SiH3]C"))

        self.assertUnsupportedCategory("unsupported_atom_text", report.categories)

    def test_unsupported_bond_types_are_fail_fast_unsupported(self) -> None:
        report = south_star_support_gate_report(parse_smiles("C#N"))

        self.assertUnsupportedCategory("unsupported_bond_type", report.categories)

    def test_aromatic_bonds_are_outside_first_domain(self) -> None:
        report = south_star_support_gate_report(parse_smiles("c1ccccc1"))

        self.assertUnsupportedCategory("unsupported_bond_type", report.categories)
        self.assertUnsupportedCategory("ring_molecule", report.categories)

    def test_aromatic_directional_surfaces_have_specific_reason(self) -> None:
        mol = parse_smiles("c1ccccc1")
        bond = mol.GetBondWithIdx(0)
        bond.SetBondDir(Chem.BondDir.ENDUPRIGHT)

        report = south_star_support_gate_report(mol)

        self.assertUnsupportedCategory(
            "aromatic_directional_surface",
            report.categories,
        )

    def test_stereo_without_carrier_basis_is_fail_fast_unsupported(self) -> None:
        mol = parse_smiles("FC=CCl")
        bond = mol.GetBondBetweenAtoms(1, 2)
        self.assertIsNotNone(bond)
        bond.SetStereo(Chem.BondStereo.STEREOZ)

        report = south_star_support_gate_report(mol)

        self.assertUnsupportedCategory(
            "unstated_component_equation",
            report.categories,
        )
