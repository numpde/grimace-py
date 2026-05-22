"""Tests for the isolated South Star 1 RDKit ingestion boundary."""

from __future__ import annotations

import unittest

from rdkit import Chem

from grimace._south_star1.facts import BondOrder
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.rdkit_adapter import molecule_facts_from_rdkit


class RdkitAdapterTest(unittest.TestCase):
    def test_snapshots_simple_nonstereo_molecule_facts(self) -> None:
        mol = Chem.MolFromSmiles("CCO")

        facts = molecule_facts_from_rdkit(mol)

        self.assertEqual(tuple(atom.symbol for atom in facts.atoms), ("C", "C", "O"))
        self.assertEqual(tuple(bond.order for bond in facts.bonds), (
            BondOrder.SINGLE,
            BondOrder.SINGLE,
        ))
        self.assertEqual(facts.components[0].atoms, (AtomId(0), AtomId(1), AtomId(2)))
        self.assertEqual(facts.components[0].bonds, (BondId(0), BondId(1)))

    def test_snapshots_disconnected_components_without_reordering_atoms(self) -> None:
        mol = Chem.MolFromSmiles("CO.CC")

        facts = molecule_facts_from_rdkit(mol)

        self.assertEqual(
            tuple(component.atoms for component in facts.components),
            ((AtomId(0), AtomId(1)), (AtomId(2), AtomId(3))),
        )

    def test_rejects_rdkit_atom_stereo_until_stereo_adapter_is_explicit(self) -> None:
        mol = Chem.MolFromSmiles("F[C@H](Cl)Br")

        with self.assertRaisesRegex(NotImplementedError, "atom stereo"):
            molecule_facts_from_rdkit(mol)

    def test_rejects_rdkit_bond_stereo_until_stereo_adapter_is_explicit(self) -> None:
        mol = Chem.MolFromSmiles("F/C=C/Cl")

        with self.assertRaisesRegex(NotImplementedError, "bond stereo"):
            molecule_facts_from_rdkit(mol)


if __name__ == "__main__":
    unittest.main()
