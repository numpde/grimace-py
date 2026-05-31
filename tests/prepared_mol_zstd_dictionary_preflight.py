from __future__ import annotations

import unittest

from rdkit import Chem, rdBase
import zstandard as zstd

import grimace
from scripts import generate_prepared_mol_zstd_dictionary as generator


class PreparedMolZstdDictionaryPreflightTests(unittest.TestCase):
    def test_generation_environment_matches_recipe(self) -> None:
        self.assertEqual(generator.EXPECTED_RDKIT_VERSION, rdBase.rdkitVersion)
        self.assertEqual(generator.EXPECTED_ZSTANDARD_VERSION, zstd.__version__)
        self.assertEqual(generator.EXPECTED_ZSTANDARD_BACKEND, zstd.backend)
        self.assertEqual(generator.EXPECTED_ZSTD_LIBRARY_VERSION, zstd.ZSTD_VERSION)

    def test_installed_prepared_mol_smoke(self) -> None:
        mol = Chem.MolFromSmiles("CCO")
        self.assertIsNotNone(mol)

        prepared = grimace.PrepareMol(mol, **generator.WRITER_OPTIONS)
        payload = prepared.to_bytes()
        restored = grimace.PreparedMol.from_bytes(payload)

        self.assertTrue(payload.startswith(generator.RAW_PREPARED_MOL_MAGIC))
        self.assertEqual(payload, restored.to_bytes())


if __name__ == "__main__":
    unittest.main()
