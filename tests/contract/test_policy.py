from __future__ import annotations

from pathlib import Path
import unittest

from smiles_next_token.reference import (
    DEFAULT_RDKIT_RANDOM_CONNECTED_NONSTEREO_POLICY_PATH,
    DEFAULT_RDKIT_RANDOM_POLICY_PATH,
    ReferencePolicy,
)


class ReferencePolicyTest(unittest.TestCase):
    def test_default_policy_path_exists(self) -> None:
        self.assertTrue(DEFAULT_RDKIT_RANDOM_POLICY_PATH.is_file())
        self.assertTrue(DEFAULT_RDKIT_RANDOM_CONNECTED_NONSTEREO_POLICY_PATH.is_file())

    def test_policy_name_matches_filename(self) -> None:
        policy = ReferencePolicy.from_path(DEFAULT_RDKIT_RANDOM_POLICY_PATH)
        self.assertEqual("rdkit_random_v1", policy.policy_name)
        self.assertEqual("rdkit_random_v1", DEFAULT_RDKIT_RANDOM_POLICY_PATH.stem)
        self.assertEqual("general", policy.branch_family)

    def test_policy_tracks_explicit_sampling_and_identity_settings(self) -> None:
        policy = ReferencePolicy.from_path(DEFAULT_RDKIT_RANDOM_POLICY_PATH)

        self.assertEqual(
            {
                "seed",
                "draw_budget",
                "isomericSmiles",
                "kekuleSmiles",
                "rootedAtAtom",
                "canonical",
                "allBondsExplicit",
                "allHsExplicit",
                "doRandom",
                "ignoreAtomMapNumbers",
            },
            set(policy.data["sampling"]),
        )
        self.assertEqual(
            {
                "parse_with_rdkit",
                "canonical",
                "isomericSmiles",
                "kekuleSmiles",
                "rootedAtAtom",
                "allBondsExplicit",
                "allHsExplicit",
                "doRandom",
                "ignoreAtomMapNumbers",
            },
            set(policy.data["identity_check"]),
        )

    def test_snapshot_paths_include_policy_name_and_digest(self) -> None:
        policy = ReferencePolicy.from_path(DEFAULT_RDKIT_RANDOM_POLICY_PATH)
        digest = policy.digest()

        self.assertEqual(
            Path("tests/fixtures/reference/rdkit_random/branches/general/snapshots/rdkit_random_v1")
            / digest,
            policy.snapshot_dir(Path("tests/fixtures/reference")),
        )
        self.assertEqual(
            Path("tests/fixtures/reference/rdkit_random/branches/general/snapshots/rdkit_random_v1")
            / digest
            / "core_exact_sets.json",
            policy.core_exact_sets_path(Path("tests/fixtures/reference")),
        )
        self.assertEqual(
            Path("tests/fixtures/reference/rdkit_random/branches/general/snapshots/rdkit_random_v1")
            / digest
            / "full_metrics.json.gz",
            policy.full_metrics_path(Path("tests/fixtures/reference")),
        )
        self.assertEqual(
            Path("tests/fixtures/reference/rdkit_random/branches/general/snapshots/rdkit_random_v1")
            / digest
            / "first_1000_len_le_40_metrics.json.gz",
            policy.metrics_path("first_1000_len_le_40", Path("tests/fixtures/reference")),
        )

    def test_connected_nonstereo_policy_tracks_input_surface_filters(self) -> None:
        policy = ReferencePolicy.from_path(DEFAULT_RDKIT_RANDOM_CONNECTED_NONSTEREO_POLICY_PATH)

        self.assertEqual("rdkit_random_connected_nonstereo_v1", policy.policy_name)
        self.assertEqual("connected_nonstereo", policy.branch_family)
        self.assertEqual(
            {
                "connected_only": True,
                "stereochemistry": "forbid",
            },
            policy.data["input_source"]["filters"],
        )
        self.assertTrue(policy.data["sampling"]["isomericSmiles"])
        self.assertTrue(policy.data["identity_check"]["isomericSmiles"])


if __name__ == "__main__":
    unittest.main()
