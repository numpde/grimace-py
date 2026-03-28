from __future__ import annotations

import unittest

from rdkit import Chem, rdBase

from smiles_next_token.reference import (
    DEFAULT_RDKIT_RANDOM_POLICY_PATH,
    ReferencePolicy,
    load_default_molecule_cases,
    sample_and_validate_rdkit_random,
    sample_rdkit_random_smiles,
    sample_rdkit_random_smiles_from_root,
)


class RdkitRandomReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = ReferencePolicy.from_path(DEFAULT_RDKIT_RANDOM_POLICY_PATH)

    def test_policy_declares_rdkit_version(self) -> None:
        self.assertRegex(str(self.policy.data["rdkit_version"]), r"^\d{4}\.\d{2}\.\d+$")

    def test_policy_rdkit_version_matches_installed_rdkit_when_fixture_is_regenerated(self) -> None:
        policy_version = str(self.policy.data["rdkit_version"])
        if policy_version != rdBase.rdkitVersion:
            raise unittest.SkipTest(
                "policy fixture targets a different RDKit build than the local environment"
            )
        self.assertEqual(rdBase.rdkitVersion, policy_version)

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
