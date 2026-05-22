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
from grimace._reference.dataset import iter_molecule_cases_from_input_source
from rdkit import Chem

_DEFAULT_INPUT_SOURCE = {
    "kind": "default_fixture",
    "path": "tests/fixtures/top_100000_CIDs.tsv.gz",
}


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

    def test_input_source_without_filters_uses_general_surface(self) -> None:
        cases = list(iter_molecule_cases_from_input_source(_DEFAULT_INPUT_SOURCE, limit=2))

        self.assertEqual(["702", "23978"], [case.cid for case in cases])

    def test_input_source_requires_kind(self) -> None:
        input_source = {"path": _DEFAULT_INPUT_SOURCE["path"]}

        with self.assertRaisesRegex(ValueError, "input_source.kind is required"):
            list(iter_molecule_cases_from_input_source(input_source, limit=1))

    def test_input_source_requires_path(self) -> None:
        input_source = {"kind": "default_fixture"}

        with self.assertRaisesRegex(ValueError, "input_source.path is required"):
            list(iter_molecule_cases_from_input_source(input_source, limit=1))

    def test_connected_nonstereo_loader_filters_surface(self) -> None:
        cases = load_default_connected_nonstereo_molecule_cases(limit=10, max_smiles_length=12)

        self.assertEqual(10, len(cases))
        for case in cases:
            with self.subTest(cid=case.cid):
                mol = Chem.MolFromSmiles(case.smiles)
                self.assertIsNotNone(mol)
                self.assertTrue(molecule_is_connected(mol))
                self.assertFalse(molecule_has_stereochemistry(mol))

    def test_input_source_filters_must_be_an_object(self) -> None:
        input_source = {**_DEFAULT_INPUT_SOURCE, "filters": []}

        with self.assertRaisesRegex(TypeError, "input_source.filters must be a JSON object"):
            list(iter_molecule_cases_from_input_source(input_source, limit=1))

    def test_input_source_filters_must_define_exact_known_keys(self) -> None:
        cases = [
            {
                "filters": {
                    "connected_only": True,
                },
                "message": r"missing=\['stereochemistry'\]; extra=\[\]",
            },
            {
                "filters": {
                    "connected_only": True,
                    "stereochemistry": "forbid",
                    "unknown": True,
                },
                "message": r"missing=\[\]; extra=\['unknown'\]",
            },
        ]
        for case in cases:
            with self.subTest(filters=case["filters"]):
                input_source = {**_DEFAULT_INPUT_SOURCE, "filters": case["filters"]}
                with self.assertRaisesRegex(ValueError, case["message"]):
                    list(iter_molecule_cases_from_input_source(input_source, limit=1))

    def test_input_source_connected_only_filter_must_be_boolean(self) -> None:
        input_source = {
            **_DEFAULT_INPUT_SOURCE,
            "filters": {
                "connected_only": "false",
                "stereochemistry": "allow",
            },
        }

        with self.assertRaisesRegex(TypeError, "connected_only must be a JSON boolean"):
            list(iter_molecule_cases_from_input_source(input_source, limit=1))

    def test_input_source_stereochemistry_filter_must_be_string(self) -> None:
        input_source = {
            **_DEFAULT_INPUT_SOURCE,
            "filters": {
                "connected_only": False,
                "stereochemistry": False,
            },
        }

        with self.assertRaisesRegex(TypeError, "stereochemistry must be a string"):
            list(iter_molecule_cases_from_input_source(input_source, limit=1))

    def test_input_source_stereochemistry_filter_must_be_supported(self) -> None:
        input_source = {
            **_DEFAULT_INPUT_SOURCE,
            "filters": {
                "connected_only": False,
                "stereochemistry": "require",
            },
        }

        with self.assertRaisesRegex(ValueError, "Unsupported input_source stereochemistry mode: 'require'"):
            list(iter_molecule_cases_from_input_source(input_source, limit=1))


if __name__ == "__main__":
    unittest.main()
