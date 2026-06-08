from __future__ import annotations

import gzip
from pathlib import Path
import tempfile
import unittest

from rdkit import Chem

from grimace._reference._paths import DEFAULT_MOLECULE_SOURCE_PATH
from grimace._reference.dataset import (
    iter_default_molecule_cases,
    load_default_connected_nonstereo_molecule_cases,
    load_default_molecule_cases,
    load_molecule_cases,
    molecule_has_stereochemistry,
    molecule_is_connected,
)


class ReferenceDatasetTest(unittest.TestCase):
    def _write_fixture(self, tmpdir: str, text: str) -> Path:
        path = Path(tmpdir) / "molecules.tsv.gz"
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            handle.write(text)
        return path

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

    def test_molecule_fixture_reader_rejects_malformed_shape(self) -> None:
        cases = (
            ("CID\tSMILES\n1\tCCO\n", "lacks required column"),
            (
                "CID\tiupac_name\tSMILES\n1\tethanol\tCCO\textra\n",
                "too many columns",
            ),
        )
        for text, message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as tmpdir:
                    path = self._write_fixture(tmpdir, text)
                    with self.assertRaisesRegex(ValueError, message):
                        load_molecule_cases(path)

    def test_connected_nonstereo_loader_filters_surface(self) -> None:
        cases = load_default_connected_nonstereo_molecule_cases(
            limit=10,
            max_smiles_length=12,
        )

        self.assertEqual(10, len(cases))
        for case in cases:
            with self.subTest(cid=case.cid):
                mol = Chem.MolFromSmiles(case.smiles)
                self.assertIsNotNone(mol)
                self.assertTrue(molecule_is_connected(mol))
                self.assertFalse(molecule_has_stereochemistry(mol))


if __name__ == "__main__":
    unittest.main()
