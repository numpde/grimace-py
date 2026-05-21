from __future__ import annotations

import unittest

from rdkit import Chem

from grimace._south_star.aromatic_policy import (
    DEFAULT_SOUTH_STAR_AROMATIC_POLICY_CONTRACT,
)
from grimace._south_star.molecule_facts import SouthStarMoleculeFacts
from grimace._south_star.support_gates import south_star_support_gate_report
from tests.helpers.south_star_semantic_oracle import parse_smiles


class SouthStarAromaticBoundaryTests(unittest.TestCase):
    def test_active_contract_names_non_aromatic_fact_boundary(self) -> None:
        contract = DEFAULT_SOUTH_STAR_AROMATIC_POLICY_CONTRACT

        self.assertEqual("non_aromatic_molecule_facts", contract.name)
        self.assertEqual("non_aromatic_molecule_facts", contract.molecule_fact_contract)
        self.assertEqual(
            "non_aromatic_organic_and_bracket_atom_text",
            contract.atom_text_policy,
        )
        self.assertEqual(
            "non_aromatic_single_double_bond_text",
            contract.bond_text_policy,
        )
        self.assertEqual(
            "non_aromatic_parse_back_graph_stereo_identity",
            contract.semantic_equivalence_relation,
        )
        self.assertEqual(
            "unsupported_aromatic_directional_overlay",
            contract.directional_surface_policy,
        )
        self.assertFalse(contract.supports_aromatic_facts)

    def test_sanitized_aromatic_and_kekule_spelling_share_aromatic_facts(
        self,
    ) -> None:
        cases = ("c1ccccc1", "C1=CC=CC=C1")

        for smiles in cases:
            facts = SouthStarMoleculeFacts.from_mol(parse_smiles(smiles))

            with self.subTest(smiles=smiles):
                self.assertIn("aromatic_ring_surface", facts.unsupported_categories)
                self.assertIn("unsupported_bond_type", facts.unsupported_categories)
                self.assertTrue(any(atom.is_aromatic for atom in facts.atom_text_facts))
                self.assertTrue(any(bond.is_aromatic for bond in facts.bond_text_facts))
                self.assertEqual(
                    {"AROMATIC"},
                    {bond.bond_type for bond in facts.bond_text_facts},
                )

    def test_deliberately_kekulized_non_aromatic_facts_are_a_different_contract(
        self,
    ) -> None:
        mol = parse_smiles("c1ccccc1")
        Chem.Kekulize(mol, clearAromaticFlags=True)

        facts = SouthStarMoleculeFacts.from_mol(mol)

        self.assertNotIn("aromatic_ring_surface", facts.unsupported_categories)
        self.assertNotIn("aromatic_directional_surface", facts.unsupported_categories)
        self.assertTrue(facts.supported, facts.unsupported_categories)
        self.assertFalse(any(atom.is_aromatic for atom in facts.atom_text_facts))
        self.assertFalse(any(bond.is_aromatic for bond in facts.bond_text_facts))
        self.assertEqual(
            {"SINGLE", "DOUBLE"},
            {bond.bond_type for bond in facts.bond_text_facts},
        )

    def test_aromatic_directional_surface_remains_a_fail_fast_overlay(self) -> None:
        mol = parse_smiles("c1ccccc1")
        mol.GetBondWithIdx(0).SetBondDir(Chem.BondDir.ENDUPRIGHT)
        facts = SouthStarMoleculeFacts.from_mol(mol)

        self.assertIn("aromatic_ring_surface", facts.unsupported_categories)
        self.assertIn("aromatic_directional_surface", facts.unsupported_categories)

    def test_support_gate_reasons_name_active_contract(self) -> None:
        contract = DEFAULT_SOUTH_STAR_AROMATIC_POLICY_CONTRACT
        mol = parse_smiles("c1ccccc1")
        mol.GetBondWithIdx(0).SetBondDir(Chem.BondDir.ENDUPRIGHT)

        report = south_star_support_gate_report(mol)
        reasons_by_category = {
            feature.category: feature.reason for feature in report.unsupported_features
        }

        self.assertIn(contract.name, reasons_by_category["aromatic_ring_surface"])
        self.assertIn(
            contract.directional_surface_policy,
            reasons_by_category["aromatic_directional_surface"],
        )


if __name__ == "__main__":
    unittest.main()
