"""Tests for the isolated South Star 1 RDKit ingestion boundary."""

from __future__ import annotations

import unittest

from rdkit import Chem

from grimace._south_star1.facts import BondOrder
from grimace._south_star1.facts import LigandKind
from grimace._south_star1.facts import SiteStatus
from grimace._south_star1.facts import TetraValue
from grimace._south_star1.fact_isomorphism import facts_are_isomorphic
from grimace._south_star1.graph_index import build_graph_index
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ordinary_semantics import OrdinarySmilesSemantics
from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.rdkit_adapter import RdkitOrdinaryExtractionOptions
from grimace._south_star1.rdkit_adapter import molecule_facts_from_rdkit
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_rdkit
from grimace._south_star1.skeleton import enumerate_traversal_skeletons
from grimace._south_star1.support_enumeration import enumerate_stereo_support


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

    def test_ordinary_adapter_normalizes_non_graph_hydrogens(self) -> None:
        mol = Chem.MolFromSmiles("[C@H](F)(Cl)Br")

        facts = ordinary_molecule_facts_from_rdkit(
            mol,
            RdkitOrdinaryExtractionOptions(
                extract_specified_tetrahedral=False,
                reject_unsupported_stereo=False,
            ),
        )

        center = facts.atoms[0]
        self.assertEqual(center.explicit_h_count, 0)
        self.assertEqual(center.implicit_h_count, 1)
        self.assertFalse(center.no_implicit)
        self.assertEqual(len(facts.stereo.tetrahedral), 1)
        occurrence_by_id = {
            occurrence.id: occurrence
            for occurrence in facts.ligand_occurrences
        }
        tetra = facts.stereo.tetrahedral[0]
        self.assertEqual(
            sum(
                occurrence_by_id[occurrence_id].kind is LigandKind.IMPLICIT_H
                for occurrence_id in tetra.ligand_occurrences
            ),
            1,
        )
        ordinary_policy_for_facts(facts)

    def test_ordinary_adapter_promotes_rdkit_tetrahedral_stereo(self) -> None:
        mol = Chem.MolFromSmiles("[C@H](F)(Cl)Br")

        facts = ordinary_molecule_facts_from_rdkit(mol)

        self.assertEqual(len(facts.stereo.tetrahedral), 1)
        site = facts.stereo.tetrahedral[0]
        self.assertEqual(site.status, SiteStatus.SPECIFIED)
        self.assertEqual(site.target, TetraValue.PLUS)
        self.assertEqual(set(site.reference_order), set(site.ligand_occurrences))
        ordinary_policy_for_facts(facts)

    def test_ordinary_adapter_distinguishes_tetrahedral_enantiomer_tags(self) -> None:
        clockwise = ordinary_molecule_facts_from_rdkit(
            Chem.MolFromSmiles("[C@H](F)(Cl)Br")
        )
        counterclockwise = ordinary_molecule_facts_from_rdkit(
            Chem.MolFromSmiles("[C@@H](F)(Cl)Br")
        )

        self.assertEqual(
            clockwise.stereo.tetrahedral[0].ligand_occurrences,
            counterclockwise.stereo.tetrahedral[0].ligand_occurrences,
        )
        self.assertNotEqual(
            clockwise.stereo.tetrahedral[0].target,
            counterclockwise.stereo.tetrahedral[0].target,
        )

    def test_tetrahedral_center_root_support_roundtrips_to_isomorphic_facts(self) -> None:
        facts = ordinary_molecule_facts_from_rdkit(
            Chem.MolFromSmiles("[C@H](F)(Cl)Br")
        )
        policy = ordinary_policy_for_facts(facts)
        skeletons = tuple(
            skeleton
            for skeleton in enumerate_traversal_skeletons(
                facts,
                build_graph_index(facts),
                policy,
            )
            if skeleton.roots == (AtomId(0),)
        )
        image = enumerate_stereo_support(
            facts=facts,
            policy=policy,
            semantics=OrdinarySmilesSemantics(),
            skeletons=skeletons,
        )

        self.assertGreater(image.distinct_count, 0)
        for text in image.strings:
            parsed = Chem.MolFromSmiles(text)
            self.assertIsNotNone(parsed, text)
            reparsed = ordinary_molecule_facts_from_rdkit(parsed)
            compare = facts_are_isomorphic(facts, reparsed)
            self.assertTrue(compare.isomorphic, (text, compare.reason))


if __name__ == "__main__":
    unittest.main()
