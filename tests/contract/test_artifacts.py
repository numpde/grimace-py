from __future__ import annotations

import unittest

from smiles_next_token._reference import (
    DEFAULT_RDKIT_RANDOM_CONNECTED_NONSTEREO_POLICY_PATH,
    DEFAULT_RDKIT_RANDOM_POLICY_PATH,
    ReferencePolicy,
    build_core_exact_sets_artifact,
    build_full_metrics_artifact,
    molecule_has_stereochemistry,
    molecule_is_connected,
)
from rdkit import Chem


class ReferenceArtifactsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = ReferencePolicy.from_path(DEFAULT_RDKIT_RANDOM_POLICY_PATH)
        cls.connected_nonstereo_policy = ReferencePolicy.from_path(
            DEFAULT_RDKIT_RANDOM_CONNECTED_NONSTEREO_POLICY_PATH
        )

    def test_core_exact_sets_artifact_includes_sampled_sets(self) -> None:
        artifact = build_core_exact_sets_artifact(self.policy, limit=2)

        self.assertEqual("rdkit_random_v1", artifact["policy_name"])
        self.assertEqual("general", artifact["branch_family"])
        self.assertEqual(2, artifact["case_count"])
        self.assertEqual({"kind": "first_n", "count": 2}, artifact["selection"])
        self.assertIn("sampled_set", artifact["cases"][0])

    def test_full_metrics_artifact_omits_sampled_sets(self) -> None:
        artifact = build_full_metrics_artifact(self.policy, limit=2)

        self.assertEqual("rdkit_random_v1", artifact["policy_name"])
        self.assertEqual("general", artifact["branch_family"])
        self.assertEqual(2, artifact["case_count"])
        self.assertEqual({"kind": "first_n", "count": 2}, artifact["selection"])
        self.assertNotIn("sampled_set", artifact["cases"][0])

    def test_full_metrics_artifact_tracks_length_capped_selection(self) -> None:
        artifact = build_full_metrics_artifact(self.policy, limit=2, max_smiles_length=3)

        self.assertEqual(
            {
                "kind": "first_n_with_max_smiles_length",
                "count": 2,
                "max_smiles_length": 3,
            },
            artifact["selection"],
        )

    def test_connected_nonstereo_policy_artifact_respects_input_filter(self) -> None:
        artifact = build_core_exact_sets_artifact(self.connected_nonstereo_policy, limit=5)

        self.assertEqual("rdkit_random_connected_nonstereo_v1", artifact["policy_name"])
        self.assertEqual("connected_nonstereo", artifact["branch_family"])
        self.assertEqual(
            {
                "connected_only": True,
                "stereochemistry": "forbid",
            },
            artifact["input_source"]["filters"],
        )
        for case in artifact["cases"]:
            with self.subTest(cid=case["cid"]):
                mol = Chem.MolFromSmiles(case["input_smiles"])
                self.assertIsNotNone(mol)
                self.assertTrue(molecule_is_connected(mol))
                self.assertFalse(molecule_has_stereochemistry(mol))


if __name__ == "__main__":
    unittest.main()
