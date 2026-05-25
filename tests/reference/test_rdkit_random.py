from __future__ import annotations

import unittest

from rdkit import Chem

from grimace._reference.rdkit_random import (
    sample_and_validate_rdkit_random,
    sample_rdkit_random_smiles,
    sample_rdkit_random_smiles_from_root,
)
from grimace._reference.dataset import load_default_molecule_cases
from tests.helpers.policies import load_default_policy


class RdkitRandomReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_default_policy()

    def test_sampling_is_reproducible_for_fixed_policy(self) -> None:
        mol = Chem.MolFromSmiles("Cc1ccccc1")
        self.assertIsNotNone(mol)

        first = sample_rdkit_random_smiles(mol, self.policy)
        second = sample_rdkit_random_smiles(mol, self.policy)

        self.assertEqual(first, second)

    def test_sampled_outputs_roundtrip_on_default_dataset_slice(self) -> None:
        for case in load_default_molecule_cases(limit=10):
            with self.subTest(cid=case.cid, smiles=case.smiles):
                mol = Chem.MolFromSmiles(case.smiles)
                self.assertIsNotNone(mol)
                result = sample_and_validate_rdkit_random(mol, self.policy)
                self.assertGreaterEqual(result.distinct_count, 1)
                self.assertTrue(result.is_valid)

    def test_rooted_sampling_is_reproducible_for_fixed_policy(self) -> None:
        mol = Chem.MolFromSmiles("C1CCCC=C1")
        self.assertIsNotNone(mol)

        first = sample_rdkit_random_smiles_from_root(mol, self.policy, 0)
        second = sample_rdkit_random_smiles_from_root(mol, self.policy, 0)

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
