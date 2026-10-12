from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError
import json
from types import ModuleType
from unittest.mock import patch

from rdkit import Chem

from tests.helpers.fixture_paths import checked_in_fixture_path

from grimace._south_star2.preparation import (
    AtomFacts,
    PreparedMolecule,
    TetrahedralAtom,
    VirtualHydrogen,
    prepare_molecule,
)


CHEMISTRY_FACTS_PATH = checked_in_fixture_path(
    "opensmiles_examples", "native_chemistry_facts.json"
)


def _chemistry_cases() -> tuple[dict[str, object], ...]:
    fixture = json.loads(CHEMISTRY_FACTS_PATH.read_text())
    return tuple(fixture["cases"])


def _case_molecule(case: dict[str, object]) -> Chem.Mol:
    params = Chem.SmilesParserParams()
    params.removeHs = not case.get("preserve_graph_hydrogens", False)
    mol = Chem.MolFromSmiles(case["smiles"], params)
    if mol is None:
        raise AssertionError(f"RDKit could not parse compatibility input {case['smiles']!r}")
    return mol


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
                self.assertEqual(tuple(atom.element for atom in prepared.atoms), atoms)
                self.assertEqual(tuple(atom.non_graph_hydrogens for atom in prepared.atoms), hydrogens)
                self.assertEqual(prepared.bonds, bonds)
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

        with self.assertRaisesRegex(TypeError, "one immutable bool per atom"):
            PreparedMolecule(
                (AtomFacts("C", False, None, 0, 4, 0),),
                (),
                source_atom_class_property_present=(1,),  # type: ignore[arg-type]
            )

        with self.assertRaisesRegex(ValueError, "elided, single, aromatic, double, or triple"):
            PreparedMolecule((
                AtomFacts("N", False, None, 0, 3, 0),
                AtomFacts("N", False, None, 0, 3, 0),
            ), ((0, 1, "->"),))

        with self.assertRaisesRegex(TypeError, "immutable tuples"):
            TetrahedralAtom(
                AtomFacts("C", False, None, 0, 1, 0),
                [0, 1, 2, VirtualHydrogen.INSTANCE],  # type: ignore[arg-type]
                True,
            )

        mol = Chem.MolFromSmiles("C")
        assert mol is not None
        prepared = prepare_molecule(mol)
        with self.assertRaises(FrozenInstanceError):
            prepared.atoms = ("O",)  # type: ignore[misc]

        tetrahedral = TetrahedralAtom(
            AtomFacts("C", False, None, 0, 1, 0),
            (0, 1, 2, VirtualHydrogen.INSTANCE),
            True,
        )
        with self.assertRaises(FrozenInstanceError):
            tetrahedral.at_for_even_parity = False  # type: ignore[misc]

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
        self.assertTrue(ccw_fact.at_for_even_parity)
        self.assertFalse(cw_fact.at_for_even_parity)

    def test_rejects_tetrahedral_stereo_outside_the_supported_hydrogen_scope(self) -> None:
        mol = Chem.MolFromSmiles("C[C@](O)(CC)CCC")
        assert mol is not None
        with self.assertRaisesRegex(ValueError, "degree-three carbon"):
            prepare_molecule(mol)

    def test_prepares_supported_single_double_triple_and_aromatic_bonds(self) -> None:
        for smiles, expected in (
            ("C=C", ("=",)),
            ("C#N", ("#",)),
            ("c1ccccc1", (":",) * 6),
            ("c1ccccc1-c2ccccc2", (":",) * 5 + ("-",) + (":",) * 7),
        ):
            with self.subTest(smiles=smiles):
                mol = Chem.MolFromSmiles(smiles)
                assert mol is not None
                self.assertEqual(
                    expected,
                    tuple(token for _, _, token in prepare_molecule(mol).bonds),
                )

    def test_unpaired_source_direction_errors_name_sorted_bond_ids(self) -> None:
        for smiles, marked_bonds, expected_ids in (
            ("CC=C", (0,), "[0]"),
            ("CC=CC", (2, 0), "[0, 2]"),
        ):
            with self.subTest(smiles=smiles, marked_bonds=marked_bonds):
                mol = Chem.MolFromSmiles(smiles)
                assert mol is not None
                for bond_id in marked_bonds:
                    mol.GetBondWithIdx(bond_id).SetBondDir(Chem.BondDir.ENDUPRIGHT)
                with self.assertRaises(ValueError) as caught:
                    prepare_molecule(mol)
                self.assertIn(expected_ids, str(caught.exception))
                self.assertIn("each must be a configured double-bond reference carrier", str(caught.exception))

    def test_build_writer_explains_missing_compiled_extension(self) -> None:
        prepared = PreparedMolecule((AtomFacts("C", False, None, 0, 4, 0),), ())
        with patch.dict("sys.modules", {"grimace._core": None}):
            with self.assertRaisesRegex(ImportError, "compiled grimace._core extension") as caught:
                prepared.build_writer()
        self.assertIsInstance(caught.exception.__cause__, ModuleNotFoundError)
        assert isinstance(caught.exception.__cause__, ModuleNotFoundError)
        self.assertEqual(caught.exception.__cause__.name, "grimace._core")

    def test_preparation_transports_aromaticity_without_reperception(self) -> None:
        editable = Chem.RWMol()
        atom = Chem.Atom(6)
        atom.SetIsAromatic(True)
        editable.AddAtom(atom)
        mol = editable.GetMol()
        mol.UpdatePropertyCache(strict=False)
        self.assertFalse(mol.NeedsUpdatePropertyCache())
        self.assertTrue(prepare_molecule(mol).atoms[0].aromatic)

    def test_rejects_radicals_and_unsupported_atom_stereo(self) -> None:
        radical = Chem.MolFromSmiles("[CH3]")
        assert radical is not None
        with self.assertRaisesRegex(ValueError, "radical spelling is unsupported"):
            prepare_molecule(radical)

        mol = Chem.MolFromSmiles("CCO")
        assert mol is not None
        unsupported_tag = Chem.ChiralType.CHI_OTHER
        mol.GetAtomWithIdx(1).SetChiralTag(unsupported_tag)
        with self.assertRaises(ValueError) as caught:
            prepare_molecule(mol)
        self.assertIn(repr(unsupported_tag), str(caught.exception))

    def test_prepared_hydrogen_facts_distinguish_graph_and_atom_field_hydrogen(self) -> None:
        methane = Chem.MolFromSmiles("C")
        bracket_methane = Chem.MolFromSmiles("[CH4]")
        explicit_params = Chem.SmilesParserParams()
        explicit_params.removeHs = False
        graph_methane = Chem.MolFromSmiles("[H]C([H])([H])[H]", explicit_params)
        assert methane is not None and bracket_methane is not None and graph_methane is not None

        methane_fact = prepare_molecule(methane).atoms[0]
        bracket_fact = prepare_molecule(bracket_methane).atoms[0]
        graph_facts = prepare_molecule(graph_methane)
        self.assertEqual(methane_fact, bracket_fact)
        self.assertEqual(4, methane_fact.non_graph_hydrogens)
        self.assertEqual(5, len(graph_facts.atoms))
        self.assertEqual(4, sum(atom.element == "H" for atom in graph_facts.atoms))
        self.assertEqual(0, graph_facts.atoms[1].non_graph_hydrogens)
        self.assertEqual(4, len(graph_facts.bonds))

        self.assertFalse(methane.GetAtomWithIdx(0).GetNoImplicit())
        self.assertTrue(bracket_methane.GetAtomWithIdx(0).GetNoImplicit())

    def test_query_atoms_and_bonds_fail_at_the_preparation_boundary(self) -> None:
        query_atom = Chem.MolFromSmarts("[#6]")
        query_bond = Chem.MolFromSmarts("C~C")
        assert query_atom is not None and query_bond is not None
        query_atom.UpdatePropertyCache(strict=False)
        query_bond.UpdatePropertyCache(strict=False)
        with self.assertRaisesRegex(ValueError, "query atom"):
            prepare_molecule(query_atom)
        with self.assertRaisesRegex(ValueError, "query bond"):
            prepare_molecule(query_bond)

    def test_styled_writer_rejects_new_unqualified_chemistry(self) -> None:
        for smiles in ("N", "c1ccccc1", "[13CH4]", "[CH3:7]C", "C#N", "C#C"):
            with self.subTest(smiles=smiles):
                mol = Chem.MolFromSmiles(smiles)
                assert mol is not None
                prepared = prepare_molecule(mol)
                with self.assertRaisesRegex(ValueError, "only qualified|not qualified"):
                    prepared.build_writer(style="rdkit_random_noncanonical")

    def test_explicit_zero_atom_class_is_preserved_as_source_context(self) -> None:
        mol = Chem.MolFromSmiles("C")
        assert mol is not None
        atom = mol.GetAtomWithIdx(0)
        atom.SetProp("molAtomMapNumber", "0")

        prepared = prepare_molecule(mol)
        self.assertEqual((0,), tuple(
            value.atom_class if isinstance(value, AtomFacts) else value.facts.atom_class
            for value in prepared.atoms
        ))
        self.assertEqual((True,), prepared.source_atom_class_property_present)
        with self.assertRaisesRegex(ValueError, "only qualified"):
            prepared.build_writer(style="rdkit_random_noncanonical")

        core = ModuleType("grimace._core")
        core._SouthStar2PreparedWriter = lambda *args: args  # type: ignore[attr-defined]
        with patch.dict("sys.modules", {"grimace._core": core}):
            payload = prepared.build_writer()
        self.assertEqual(("C", False, None, 0, 4, 0, None, (), True), payload[0][0])

    def test_preparation_matches_version_independent_chemistry_fact_fixture(self) -> None:
        for case in _chemistry_cases():
            with self.subTest(case=case["id"]):
                mol = _case_molecule(case)
                prepared = prepare_molecule(mol)
                atoms = tuple(case["atoms"])
                atom_ids = {row[0]: index for index, row in enumerate(atoms)}
                self.assertEqual(len(atoms), len(prepared.atoms))
                for row, prepared_atom in zip(atoms, prepared.atoms, strict=True):
                    facts = prepared_atom.facts if isinstance(prepared_atom, TetrahedralAtom) else prepared_atom
                    self.assertEqual(
                        (row[1], row[2], row[3], row[4], row[5], row[6]),
                        (
                            facts.element,
                            facts.aromatic,
                            facts.isotope,
                            facts.formal_charge,
                            facts.non_graph_hydrogens,
                            facts.atom_class,
                        ),
                    )
                self.assertEqual(
                    tuple(atom.HasProp("molAtomMapNumber") > 0 for atom in mol.GetAtoms()),
                    prepared.source_atom_class_property_present,
                )
                token_for_kind = {
                    "single": "",
                    "double": "=",
                    "triple": "#",
                    "aromatic": ":",
                }
                self.assertEqual(
                    tuple(
                        (
                            atom_ids[left],
                            atom_ids[right],
                            "-"
                            if kind == "single"
                            and atoms[atom_ids[left]][2]
                            and atoms[atom_ids[right]][2]
                            else token_for_kind[kind],
                        )
                        for left, right, kind in case["bonds"]
                    ),
                    prepared.bonds,
                )

    def test_rejects_a_stale_rdkit_property_cache(self) -> None:
        editable = Chem.RWMol()
        editable.AddAtom(Chem.Atom(6))
        mol = editable.GetMol()
        self.assertTrue(mol.NeedsUpdatePropertyCache())
        with self.assertRaisesRegex(ValueError, "current RDKit property cache"):
            prepare_molecule(mol)


if __name__ == "__main__":
    unittest.main()
