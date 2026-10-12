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
                (AtomFacts("C", False, None, 0, 4, 0, None),),
                (),
                source_atom_class_property_present=(1,),  # type: ignore[arg-type]
            )

        with self.assertRaisesRegex(ValueError, "elided, single, aromatic, double, or triple"):
            PreparedMolecule((
                AtomFacts("N", False, None, 0, 3, 0, None),
                AtomFacts("N", False, None, 0, 3, 0, None),
            ), ((0, 1, "->"),))

        with self.assertRaisesRegex(TypeError, "immutable tuples"):
            TetrahedralAtom(
                AtomFacts("C", False, None, 0, 1, 0, None),
                [0, 1, 2, VirtualHydrogen.INSTANCE],  # type: ignore[arg-type]
                True,
            )

        mol = Chem.MolFromSmiles("C")
        assert mol is not None
        prepared = prepare_molecule(mol)
        with self.assertRaises(FrozenInstanceError):
            prepared.atoms = ("O",)  # type: ignore[misc]

        tetrahedral = TetrahedralAtom(
            AtomFacts("C", False, None, 0, 1, 0, None),
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

    def test_kekulized_aromatic_flags_preserve_prepared_source_facts(self) -> None:
        for smiles in ("c1ccccc1", "c1ccccc1-c2ccccc2", "O=c1cccc[nH]1", "c1ccccc1/C=C/C"):
            with self.subTest(smiles=smiles):
                mol = Chem.MolFromSmiles(smiles)
                assert mol is not None
                expected = prepare_molecule(mol)
                kekulized = Chem.Mol(mol)
                Chem.Kekulize(kekulized, clearAromaticFlags=False)
                self.assertFalse(kekulized.NeedsUpdatePropertyCache())
                before = tuple((bond.GetIsAromatic(), bond.GetBondType()) for bond in kekulized.GetBonds())
                self.assertEqual(expected, prepare_molecule(kekulized))
                self.assertEqual(before, tuple(
                    (bond.GetIsAromatic(), bond.GetBondType()) for bond in kekulized.GetBonds()
                ))

                cleared = Chem.Mol(mol)
                Chem.Kekulize(cleared, clearAromaticFlags=True)
                fixed = prepare_molecule(cleared)
                self.assertTrue(all(not atom.aromatic and atom.aromatic_total_valence is None for atom in fixed.atoms))
                self.assertNotIn(":", tuple(token for _, _, token in fixed.bonds))
                self.assertEqual(tuple(
                    "=" if bond.GetBondType() == Chem.BondType.DOUBLE else ""
                    for bond in cleared.GetBonds()
                ), tuple(token for _, _, token in fixed.bonds))

    def test_rejects_contradictory_aromatic_bond_facts(self) -> None:
        for change, reason in (
            (lambda mol: mol.GetBondWithIdx(0).SetIsAromatic(False), "inconsistent"),
            (lambda mol: mol.GetBondWithIdx(0).SetBondType(Chem.BondType.TRIPLE), "inconsistent"),
            (lambda mol: mol.GetAtomWithIdx(0).SetIsAromatic(False), "both endpoint atoms"),
            (lambda mol: mol.GetBondWithIdx(0).SetBondDir(Chem.BondDir.ENDUPRIGHT), "no source stereo or direction"),
            (lambda mol: mol.GetBondWithIdx(0).SetStereo(Chem.BondStereo.STEREOANY), "no source stereo or direction"),
        ):
            with self.subTest(reason=reason):
                mol = Chem.MolFromSmiles("c1ccccc1")
                assert mol is not None
                change(mol)
                with self.assertRaisesRegex(ValueError, reason):
                    prepare_molecule(mol)

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
        prepared = PreparedMolecule((AtomFacts("C", False, None, 0, 4, 0, None),), ())
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

    def test_rdkit_writer_rejects_new_unqualified_chemistry(self) -> None:
        for smiles in ("N", "c1ccccc1", "[13CH4]", "[CH3:7]C", "C#N", "C#C"):
            with self.subTest(smiles=smiles):
                mol = Chem.MolFromSmiles(smiles)
                assert mol is not None
                prepared = prepare_molecule(mol)
                with self.assertRaisesRegex(ValueError, "only qualified|not qualified"):
                    prepared.build_writer(writer_regime="rdkit_random_noncanonical")

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
            prepared.build_writer(writer_regime="rdkit_random_noncanonical")

        core = ModuleType("grimace._core")
        core._SouthStar2PreparedWriter = lambda *args: args  # type: ignore[attr-defined]
        with patch.dict("sys.modules", {"grimace._core": core}):
            payload = prepared.build_writer()
        self.assertEqual(("C", False, None, 0, 4, 0, None, (), True, None), payload[0][0])

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


class AromaticValencePreparationTests(unittest.TestCase):
    def test_cached_source_valence_is_transported_only_for_aromatic_atoms(self) -> None:
        for case in _chemistry_cases():
            with self.subTest(case=case["id"]):
                mol = _case_molecule(case)
                prepared = prepare_molecule(mol)
                core = ModuleType("grimace._core")
                core._SouthStar2PreparedWriter = lambda *args: args
                with patch.dict("sys.modules", {"grimace._core": core}):
                    payload = prepared.build_writer()
                for source, atom, wire in zip(mol.GetAtoms(), prepared.atoms, payload[0], strict=True):
                    facts = atom.facts if isinstance(atom, TetrahedralAtom) else atom
                    expected = source.GetTotalValence() if source.GetIsAromatic() else None
                    self.assertEqual(facts.aromatic_total_valence, expected)
                    self.assertEqual(facts.aromatic, source.GetIsAromatic())
                    self.assertEqual(len(wire), 10)
                    self.assertEqual(wire[9], expected)

    def test_native_boundary_rejects_stale_or_inconsistent_valence_facts(self) -> None:
        from grimace._core import _SouthStar2PreparedWriter
        current = ("C", False, None, 0, 4, 0, None, (), False, None)
        cases = (
            (current[:-1], "expected ten atom-fact fields"),
            (current[:-1] + (4,), "must be absent for nonaromatic atoms"),
            (("C", True, None, 0, 1, 0, None, (), False, None), "must be present for aromatic atoms"),
            (("C", True, None, 0, 1, 0, None, (), False, True), "unsigned integer or None"),
            (("C", True, None, 0, 1, 0, None, (), False, -1), "unsigned integer or None"),
        )
        for wire, reason in cases:
            with self.subTest(wire=wire):
                with self.assertRaisesRegex(ValueError, reason):
                    _SouthStar2PreparedWriter([wire], [], "generic_any_order")
