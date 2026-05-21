from __future__ import annotations

import unittest

from rdkit import Chem

from grimace._south_star.molecule_facts import SouthStarMoleculeFacts
from tests.helpers.south_star_semantic_oracle import parse_smiles


class SouthStarAromaticBoundaryTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
