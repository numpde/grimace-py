from __future__ import annotations

import unittest

from grimace._reference import (
    DEFAULT_MOLECULE_SOURCE_PATH,
    iter_default_molecule_cases,
    load_default_connected_nonstereo_molecule_cases,
    load_default_molecule_cases,
    molecule_has_stereochemistry,
    molecule_is_connected,
)
from rdkit import Chem


class ReferenceDatasetTest(unittest.TestCase):
    def test_default_molecule_source_exists(self) -> None:
        self.assertTrue(DEFAULT_MOLECULE_SOURCE_PATH.is_file())

    def test_default_molecule_source_loads_expected_first_rows(self) -> None:
        cases = load_default_molecule_cases(limit=2)

        self.assertEqual(2, len(cases))
        self.assertEqual("702", cases[0].cid)
        self.assertEqual("ethanol", cases[0].name)
        self.assertEqual("CCO", cases[0].smiles)
        self.assertEqual("23978", cases[1].cid)
        self.assertEqual("copper", cases[1].name)
        self.assertEqual("[Cu]", cases[1].smiles)

    def test_iter_default_molecule_cases_honors_limit(self) -> None:
        cases = list(iter_default_molecule_cases(limit=3))
        self.assertEqual(3, len(cases))

    def test_load_default_molecule_cases_can_filter_by_smiles_length(self) -> None:
        cases = load_default_molecule_cases(limit=5, max_smiles_length=3)
        self.assertEqual(5, len(cases))
        self.assertTrue(all(len(case.smiles) <= 3 for case in cases))

    def test_connected_nonstereo_loader_filters_surface(self) -> None:
        cases = load_default_connected_nonstereo_molecule_cases(limit=10, max_smiles_length=12)

        self.assertEqual(10, len(cases))
        for case in cases:
            with self.subTest(cid=case.cid):
                mol = Chem.MolFromSmiles(case.smiles)
                self.assertIsNotNone(mol)
                self.assertTrue(molecule_is_connected(mol))
                self.assertFalse(molecule_has_stereochemistry(mol))


if __name__ == "__main__":
    unittest.main()
