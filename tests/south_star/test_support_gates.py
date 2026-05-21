from __future__ import annotations

import unittest

from rdkit import Chem

from grimace._south_star.support_gates import south_star_support_gate_report
from tests.helpers.south_star_domain_manifest import SOUTH_STAR_PRIVATE_DOMAIN
from tests.helpers.south_star_semantic_oracle import parse_smiles
from tests.helpers.south_star_semantics import load_south_star_semantic_cases


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

    def test_explicit_bracket_hydrogen_slice_is_inside_gate_scope(self) -> None:
        report = south_star_support_gate_report(parse_smiles("[H][H]"))

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

    def test_fused_rings_are_inside_polycyclic_skeleton_scope(self) -> None:
        report = south_star_support_gate_report(parse_smiles("C1CC2CCCC2C1"))

        self.assertTrue(report.supported, report.unsupported_features)

    def test_spiro_rings_are_inside_polycyclic_skeleton_scope(self) -> None:
        report = south_star_support_gate_report(parse_smiles("C1CCC2(CC1)CCCC2"))

        self.assertTrue(report.supported, report.unsupported_features)

    def test_bridged_rings_are_inside_polycyclic_skeleton_scope(self) -> None:
        report = south_star_support_gate_report(parse_smiles("C1CC2CCC1C2"))

        self.assertTrue(report.supported, report.unsupported_features)

    def test_modeled_ring_stereo_monocycles_are_supported(self) -> None:
        report = south_star_support_gate_report(parse_smiles("C1/C=C\\CCCCC1"))

        self.assertTrue(report.supported, report.unsupported_features)

    def test_directional_polycycle_without_rdkit_stereo_is_skeleton_scope(
        self,
    ) -> None:
        report = south_star_support_gate_report(parse_smiles("C1/C=C\\C2CCCC2C1"))

        self.assertTrue(report.supported, report.unsupported_features)

    def test_polycyclic_with_external_alkene_stereo_stays_fail_fast(self) -> None:
        report = south_star_support_gate_report(parse_smiles("F/C=C\\C1CC2CCC1C2"))

        self.assertUnsupportedCategory(
            "fused_or_polycyclic_ring",
            report.categories,
        )
        self.assertNotIn("ring_stereo", report.categories)

    def test_ring_tetrahedral_interactions_are_specific_unsupported_surface(
        self,
    ) -> None:
        report = south_star_support_gate_report(parse_smiles("F[C@H]1CCCC(C)C1"))

        self.assertUnsupportedCategory(
            "ring_tetrahedral_interaction",
            report.categories,
        )
        self.assertNotIn("ring_molecule", report.categories)

    def test_ring_adjacent_tetrahedral_center_is_specific_unsupported_surface(
        self,
    ) -> None:
        report = south_star_support_gate_report(parse_smiles("F[C@H](Cl)C1CCCCC1"))

        self.assertUnsupportedCategory(
            "ring_tetrahedral_interaction",
            report.categories,
        )
        self.assertNotIn("ring_molecule", report.categories)

    def test_fused_ring_tetrahedral_interaction_keeps_ring_system_blocker(
        self,
    ) -> None:
        report = south_star_support_gate_report(parse_smiles("C1CC2CCCC2[C@H]1F"))

        self.assertUnsupportedCategory(
            "ring_tetrahedral_interaction",
            report.categories,
        )
        self.assertUnsupportedCategory(
            "fused_or_polycyclic_ring",
            report.categories,
        )

    def test_unsupported_atom_text_is_fail_fast_unsupported(self) -> None:
        report = south_star_support_gate_report(parse_smiles("[SiH3]C"))

        self.assertUnsupportedCategory("unsupported_atom_text", report.categories)

    def test_unsupported_bracket_atom_modifiers_are_fail_fast_unsupported(self) -> None:
        cases = (
            ("[2H][H]", "unsupported_atom_isotope"),
            ("[H+]", "unsupported_atom_charge"),
            ("[H]", "unsupported_radical_atom"),
            ("[CH3:1]C", "unsupported_atom_map"),
        )

        for smiles, category in cases:
            with self.subTest(smiles=smiles):
                report = south_star_support_gate_report(parse_smiles(smiles))
                self.assertUnsupportedCategory(category, report.categories)

    def test_unsupported_bond_types_are_fail_fast_unsupported(self) -> None:
        report = south_star_support_gate_report(parse_smiles("C#N"))

        self.assertUnsupportedCategory("unsupported_bond_type", report.categories)

    def test_aromatic_bonds_are_outside_first_domain(self) -> None:
        report = south_star_support_gate_report(parse_smiles("c1ccccc1"))

        self.assertUnsupportedCategory("aromatic_ring_surface", report.categories)
        self.assertUnsupportedCategory("unsupported_bond_type", report.categories)
        self.assertUnsupportedCategory("ring_molecule", report.categories)

    def test_kekule_looking_aromatic_text_is_still_aromatic_after_parsing(
        self,
    ) -> None:
        report = south_star_support_gate_report(parse_smiles("C1=CC=CC=C1"))

        self.assertUnsupportedCategory("aromatic_ring_surface", report.categories)

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
