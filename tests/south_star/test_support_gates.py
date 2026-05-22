from __future__ import annotations

import unittest

from rdkit import Chem

from grimace._south_star.support_gates import south_star_support_gate_report
from tests.helpers.south_star_domain_manifest import SOUTH_STAR_PRIVATE_DOMAIN
from tests.helpers.south_star_semantic_oracle import parse_smiles
from tests.helpers.south_star_semantics import load_south_star_semantic_cases


class SouthStarSupportGateTests(unittest.TestCase):
    def assertUnsupportedCategory(self, category: str, categories: frozenset[str]) -> None:
        self.assertIn(category, SOUTH_STAR_PRIVATE_DOMAIN.support_gate_blocker_categories)
        self.assertIn(category, categories)

    def unsupported_bond_type_mol(self) -> Chem.Mol:
        mol = Chem.RWMol()
        begin_idx = mol.AddAtom(Chem.Atom(6))
        end_idx = mol.AddAtom(Chem.Atom(6))
        mol.AddBond(begin_idx, end_idx, Chem.BondType.UNSPECIFIED)
        return mol.GetMol()

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
        report = south_star_support_gate_report(parse_smiles("[Na+].O"))

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

    def test_polycyclic_ring_stereo_is_inside_gate_scope(self) -> None:
        report = south_star_support_gate_report(parse_smiles("C1CCC/C=C\\C2C1C2"))

        self.assertTrue(report.supported, report.unsupported_features)

    def test_polycyclic_with_external_alkene_stereo_stays_fail_fast(self) -> None:
        report = south_star_support_gate_report(parse_smiles("F/C=C\\C1CC2CCC1C2"))

        self.assertUnsupportedCategory(
            "fused_or_polycyclic_ring",
            report.categories,
        )
        self.assertNotIn("ring_stereo", report.categories)

    def test_ring_tetrahedral_monocycle_is_inside_gate_scope(
        self,
    ) -> None:
        report = south_star_support_gate_report(parse_smiles("F[C@H]1CCCC(C)C1"))

        self.assertTrue(report.supported, report.unsupported_features)
        self.assertNotIn("ring_molecule", report.categories)

    def test_ring_adjacent_tetrahedral_monocycle_is_inside_gate_scope(
        self,
    ) -> None:
        report = south_star_support_gate_report(parse_smiles("F[C@H](Cl)C1CCCCC1"))

        self.assertTrue(report.supported, report.unsupported_features)
        self.assertNotIn("ring_molecule", report.categories)

    def test_exocyclic_directional_branch_on_monocycle_is_inside_gate_scope(
        self,
    ) -> None:
        report = south_star_support_gate_report(parse_smiles("C1CC(/C=C/Cl)CCC1"))

        self.assertTrue(report.supported, report.unsupported_features)
        self.assertNotIn("ring_molecule", report.categories)
        self.assertNotIn("ring_stereo", report.categories)

    def test_ring_tetrahedral_with_directional_branch_is_inside_gate_scope(
        self,
    ) -> None:
        report = south_star_support_gate_report(parse_smiles("F[C@H]1CCCC(/C=C/Cl)C1"))

        self.assertTrue(report.supported, report.unsupported_features)
        self.assertNotIn("ring_molecule", report.categories)
        self.assertNotIn("ring_tetrahedral_interaction", report.categories)

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
        report = south_star_support_gate_report(parse_smiles("[Na+]"))

        self.assertUnsupportedCategory("unsupported_atom_text", report.categories)

    def test_renderer_capable_bracket_atom_modifiers_are_inside_gate_scope(
        self,
    ) -> None:
        cases = (
            "[2H][H]",
            "[H+]",
            "[Cl-]",
            "[NH4+]",
            "[CH3:1]C",
            "[AsH3]",
            "[GeH4]",
            "[SeH]",
            "[SiH3]C",
            "[SbH3]",
        )

        for smiles in cases:
            with self.subTest(smiles=smiles):
                report = south_star_support_gate_report(parse_smiles(smiles))
                self.assertTrue(report.supported, report.unsupported_features)

    def test_radical_bracket_atom_modifier_is_inside_gate_scope(self) -> None:
        report = south_star_support_gate_report(parse_smiles("[CH3]"))

        self.assertTrue(report.supported, report.unsupported_features)

    def test_unsupported_bond_types_are_fail_fast_unsupported(self) -> None:
        report = south_star_support_gate_report(self.unsupported_bond_type_mol())

        self.assertUnsupportedCategory("unsupported_bond_type", report.categories)

    def test_quadruple_bond_text_is_inside_gate_scope(self) -> None:
        report = south_star_support_gate_report(parse_smiles("C$C"))

        self.assertTrue(report.supported, report.unsupported_features)

    def test_markerless_aromatic_monocycle_is_inside_gate_scope(self) -> None:
        cases = (
            "c1ccccc1",
            "c1ccncc1",
            "c1ccoc1",
            "c1ccccc1C",
            "c1ccncc1C",
            "c1ccoc1C",
        )

        for smiles in cases:
            with self.subTest(smiles=smiles):
                report = south_star_support_gate_report(parse_smiles(smiles))

                self.assertTrue(report.supported, report.unsupported_features)

    def test_kekule_looking_aromatic_monocycle_shares_supported_fact_scope(
        self,
    ) -> None:
        cases = ("C1=CC=CC=C1", "[si]1ccccc1")

        for smiles in cases:
            with self.subTest(smiles=smiles):
                report = south_star_support_gate_report(parse_smiles(smiles))
                self.assertTrue(report.supported, report.unsupported_features)

    def test_unmodified_fused_aromatic_rings_are_inside_gate_scope(self) -> None:
        cases = (
            "c1ccc2ccccc2c1",
            "c1ccc2ncccc2c1",
            "c1ccc2occc2c1",
        )

        for smiles in cases:
            with self.subTest(smiles=smiles):
                report = south_star_support_gate_report(parse_smiles(smiles))
                self.assertTrue(report.supported, report.unsupported_features)

    def test_modified_aromatic_atoms_are_inside_first_atom_text_scope(self) -> None:
        cases = (
            "c1cc[nH]c1",
            "c1cc[15nH]c1",
            "[nH:7]1cccc1",
            "c1cc[nH+]cc1",
            "c1cc[n+]([O-])cc1",
        )

        for smiles in cases:
            with self.subTest(smiles=smiles):
                report = south_star_support_gate_report(parse_smiles(smiles))
                self.assertTrue(report.supported, report.unsupported_features)

    def test_bracket_only_aromatic_element_text_is_inside_gate_scope(self) -> None:
        cases = (
            "[se]1cccc1",
            "[15se]1cccc1",
            "[se:7]1cccc1",
            "[15se:7]1cccc1",
            "[te]1cccc1",
            "[15te]1cccc1",
            "[te:7]1cccc1",
            "[15te:7]1cccc1",
        )

        for smiles in cases:
            with self.subTest(smiles=smiles):
                report = south_star_support_gate_report(parse_smiles(smiles))
                self.assertTrue(report.supported, report.unsupported_features)

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
