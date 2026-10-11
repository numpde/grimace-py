from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import patch

from rdkit import Chem

from grimace._south_star2.preparation import (
    PreparedMolecule,
    TetrahedralAtom,
    VirtualHydrogen,
    prepare_molecule,
)


class SouthStar2PreparationTests(unittest.TestCase):
    def test_prepares_simple_real_graphs_from_rdkit_facts(self) -> None:
        cases = (
            (
                "CCO",
                ("C", "C", "O"),
                (3, 2, 1),
                ((0, 1, ""), (1, 2, "")),
            ),
            (
                "CC(C)O",
                ("C", "C", "C", "O"),
                (3, 1, 3, 1),
                ((0, 1, ""), (1, 2, ""), (1, 3, "")),
            ),
            (
                "C1CC1",
                ("C", "C", "C"),
                (2, 2, 2),
                ((0, 1, ""), (1, 2, ""), (2, 0, "")),
            ),
        )
        for smiles, atoms, hydrogens, bonds in cases:
            with self.subTest(smiles=smiles):
                mol = Chem.MolFromSmiles(smiles)
                self.assertIsNotNone(mol)
                assert mol is not None
                prepared = prepare_molecule(mol)
                self.assertEqual(prepared, PreparedMolecule(atoms, bonds))
                self.assertEqual(
                    tuple(atom.GetNumImplicitHs() for atom in mol.GetAtoms()),
                    hydrogens,
                )
                self.assertEqual(
                    tuple(
                        (4 if atom.GetAtomicNum() == 6 else 2) - atom.GetDegree()
                        for atom in mol.GetAtoms()
                    ),
                    hydrogens,
                )
                self.assertEqual(
                    prepared.bonds,
                    tuple(
                        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), "")
                        for bond in mol.GetBonds()
                    ),
                )

    def test_prepared_facts_cannot_contain_mutable_lists(self) -> None:
        with self.assertRaisesRegex(TypeError, "immutable tuples"):
            PreparedMolecule(["C"], ())  # type: ignore[arg-type]

        with self.assertRaisesRegex(ValueError, "C or O tokens"):
            PreparedMolecule(("N",), ())
        with self.assertRaisesRegex(ValueError, "elided single-bond token"):
            PreparedMolecule(("C", "C"), ((0, 1, "-"),))

        with self.assertRaisesRegex(TypeError, "immutable tuples"):
            TetrahedralAtom(
                [0, 1, 2, VirtualHydrogen.INSTANCE], ("[C@H]", "[C@@H]")  # type: ignore[arg-type]
            )

        mol = Chem.MolFromSmiles("C")
        assert mol is not None
        prepared = prepare_molecule(mol)
        with self.assertRaises(FrozenInstanceError):
            prepared.atoms = ("O",)  # type: ignore[misc]

        tetrahedral = TetrahedralAtom(
            (0, 1, 2, VirtualHydrogen.INSTANCE), ("[C@H]", "[C@@H]")
        )
        with self.assertRaises(FrozenInstanceError):
            tetrahedral.text_by_parity = ("[C@@H]", "[C@H]")  # type: ignore[misc]

    def test_rejects_non_molecules(self) -> None:
        with self.assertRaisesRegex(TypeError, "RDKit Mol"):
            prepare_molecule("CCO")

    def test_preparation_does_not_mutate_the_input_molecule(self) -> None:
        mol = Chem.MolFromSmiles("CC(C)O")
        assert mol is not None
        atoms_before = tuple(
            (
                atom.GetAtomicNum(),
                atom.GetFormalCharge(),
                atom.GetIsotope(),
                atom.GetAtomMapNum(),
                atom.GetChiralTag(),
                atom.GetNumImplicitHs(),
            )
            for atom in mol.GetAtoms()
        )
        bonds_before = tuple(
            (
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                bond.GetBondType(),
                bond.GetStereo(),
                bond.GetBondDir(),
            )
            for bond in mol.GetBonds()
        )

        prepare_molecule(mol)

        self.assertEqual(
            atoms_before,
            tuple(
                (
                    atom.GetAtomicNum(),
                    atom.GetFormalCharge(),
                    atom.GetIsotope(),
                    atom.GetAtomMapNum(),
                    atom.GetChiralTag(),
                    atom.GetNumImplicitHs(),
                )
                for atom in mol.GetAtoms()
            ),
        )
        self.assertEqual(
            bonds_before,
            tuple(
                (
                    bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx(),
                    bond.GetBondType(),
                    bond.GetStereo(),
                    bond.GetBondDir(),
                )
                for bond in mol.GetBonds()
            ),
        )

    def test_prepares_supported_tetrahedral_stereo_from_incident_bond_order(self) -> None:
        ccw = Chem.MolFromSmiles("C[C@H](O)CC")
        cw = Chem.MolFromSmiles("C[C@@H](O)CC")
        assert ccw is not None and cw is not None
        ccw_fact = prepare_molecule(ccw).atoms[1]
        cw_fact = prepare_molecule(cw).atoms[1]
        self.assertIsInstance(ccw_fact, TetrahedralAtom)
        self.assertIsInstance(cw_fact, TetrahedralAtom)
        assert isinstance(ccw_fact, TetrahedralAtom)
        assert isinstance(cw_fact, TetrahedralAtom)
        self.assertEqual(
            ccw_fact.reference_order,
            (
                *tuple(bond.GetIdx() for bond in ccw.GetAtomWithIdx(1).GetBonds()),
                VirtualHydrogen.INSTANCE,
            ),
        )
        self.assertEqual(ccw_fact.text_by_parity, ("[C@H]", "[C@@H]"))
        self.assertEqual(cw_fact.text_by_parity, ("[C@@H]", "[C@H]"))

    def test_rejects_tetrahedral_stereo_outside_the_supported_hydrogen_scope(self) -> None:
        mol = Chem.MolFromSmiles("C[C@](O)(CC)CCC")
        assert mol is not None
        with self.assertRaisesRegex(ValueError, "degree-three carbon"):
            prepare_molecule(mol)

    def test_rejects_unsaturated_and_aromatic_bonds(self) -> None:
        expected_bond_details = (
            ("C=C", "BondType.DOUBLE", "is_aromatic = False"),
            ("c1ccccc1", "BondType.AROMATIC", "is_aromatic = True"),
        )
        for smiles, bond_type, aromaticity in expected_bond_details:
            with self.subTest(smiles=smiles):
                mol = Chem.MolFromSmiles(smiles)
                assert mol is not None
                with self.assertRaisesRegex(ValueError, bond_type) as caught:
                    prepare_molecule(mol)
                self.assertIn(aromaticity, str(caught.exception))

    def test_build_writer_explains_missing_compiled_extension(self) -> None:
        prepared = PreparedMolecule(("C",), ())
        with patch.dict("sys.modules", {"grimace._core": None}):
            with self.assertRaisesRegex(ImportError, "compiled grimace._core extension") as caught:
                prepared.build_writer()
        self.assertIsInstance(caught.exception.__cause__, ModuleNotFoundError)
        assert isinstance(caught.exception.__cause__, ModuleNotFoundError)
        self.assertEqual(caught.exception.__cause__.name, "grimace._core")

    def test_rejects_an_aromatic_atom_without_aromatic_bonds(self) -> None:
        editable = Chem.RWMol()
        atom = Chem.Atom(6)
        atom.SetIsAromatic(True)
        editable.AddAtom(atom)
        mol = editable.GetMol()
        mol.UpdatePropertyCache(strict=False)
        self.assertFalse(mol.NeedsUpdatePropertyCache())
        with self.assertRaisesRegex(ValueError, "aromatic atom"):
            prepare_molecule(mol)

    def test_rejects_atom_features_without_a_spelling_rule(self) -> None:
        for smiles in ("[13CH4]", "[CH3:7]", "[CH3]", "[O-]C", "N"):
            with self.subTest(smiles=smiles):
                mol = Chem.MolFromSmiles(smiles)
                assert mol is not None
                with self.assertRaises(ValueError):
                    prepare_molecule(mol)

    def test_atom_feature_rejections_report_the_offending_scalar_or_tag(self) -> None:
        cases = (
            ("[O-]C", "formal_charge = -1"),
            ("[13CH4]", "isotope = 13"),
            ("[CH3:7]", "atom_map_num = 7"),
            ("[CH3]", "radical_electrons = 1"),
        )
        for smiles, evidence in cases:
            with self.subTest(smiles=smiles):
                mol = Chem.MolFromSmiles(smiles)
                assert mol is not None
                with self.assertRaises(ValueError) as caught:
                    prepare_molecule(mol)
                self.assertIn(evidence, str(caught.exception))

        mol = Chem.MolFromSmiles("CCO")
        assert mol is not None
        unsupported_tag = Chem.ChiralType.CHI_OTHER
        mol.GetAtomWithIdx(1).SetChiralTag(unsupported_tag)
        with self.assertRaises(ValueError) as caught:
            prepare_molecule(mol)
        self.assertIn(repr(unsupported_tag), str(caught.exception))

    def test_rejects_no_implicit_hydrogen_and_explicit_hydrogen_states(self) -> None:
        no_implicit = Chem.MolFromSmiles("CCO")
        assert no_implicit is not None
        no_implicit.GetAtomWithIdx(0).SetNoImplicit(True)
        with self.assertRaisesRegex(ValueError, "no-implicit-hydrogen"):
            prepare_molecule(no_implicit)

        explicit = Chem.MolFromSmiles("CCO")
        assert explicit is not None
        explicit.GetAtomWithIdx(0).SetNumExplicitHs(1)
        with self.assertRaisesRegex(ValueError, "explicit hydrogen state"):
            prepare_molecule(explicit)

    def test_rejects_a_stale_rdkit_property_cache(self) -> None:
        editable = Chem.RWMol()
        editable.AddAtom(Chem.Atom(6))
        mol = editable.GetMol()
        self.assertTrue(mol.NeedsUpdatePropertyCache())
        with self.assertRaisesRegex(ValueError, "current RDKit property cache"):
            prepare_molecule(mol)


if __name__ == "__main__":
    unittest.main()
