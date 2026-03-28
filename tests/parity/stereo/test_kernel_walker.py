from __future__ import annotations

import random
import unittest

from smiles_next_token.reference import (
    CONNECTED_STEREO_SURFACE,
    enumerate_rooted_connected_stereo_smiles_support,
    prepare_smiles_graph,
)
from tests.helpers.assertions import assert_prefix_options_match_outputs
from tests.helpers.cases import (
    STEREO_WALKER_CURATED_CASES,
    load_connected_atom_stereo_cases,
    load_connected_bond_stereo_cases,
)
from tests.helpers.kernel import CORE_MODULE
from tests.helpers.mols import parse_smiles
from tests.helpers.policies import load_connected_nonstereo_policy


class CoreRootedConnectedStereoWalkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if CORE_MODULE is None:
            raise unittest.SkipTest("private Rust extension is not installed")
        cls.policy = load_connected_nonstereo_policy()

    def test_core_stereo_walker_exact_support_matches_reference_on_curated_cases(self) -> None:
        for smiles, root_idx in STEREO_WALKER_CURATED_CASES:
            with self.subTest(smiles=smiles, root_idx=root_idx):
                prepared = prepare_smiles_graph(
                    parse_smiles(smiles),
                    self.policy,
                    surface_kind=CONNECTED_STEREO_SURFACE,
                )
                walker = CORE_MODULE.RootedConnectedStereoWalker(prepared, root_idx)

                observed = set(walker.enumerate_support())
                expected = enumerate_rooted_connected_stereo_smiles_support(prepared, root_idx)
                self.assertEqual(expected, observed)

    def test_core_stereo_walker_exact_support_matches_reference_on_dataset_slice(self) -> None:
        cases = load_connected_atom_stereo_cases(limit=2, max_smiles_length=16)
        bond_cases = load_connected_bond_stereo_cases(limit=1, max_smiles_length=18)
        self.assertEqual(2, len(cases))
        self.assertEqual(1, len(bond_cases))

        for cid, smiles in [*cases, *bond_cases]:
            prepared = prepare_smiles_graph(
                parse_smiles(smiles),
                self.policy,
                surface_kind=CONNECTED_STEREO_SURFACE,
            )
            for root_idx in range(prepared.atom_count):
                with self.subTest(cid=cid, smiles=smiles, root_idx=root_idx):
                    walker = CORE_MODULE.RootedConnectedStereoWalker(prepared, root_idx)
                    observed = set(walker.enumerate_support())
                    expected = enumerate_rooted_connected_stereo_smiles_support(prepared, root_idx)
                    self.assertEqual(expected, observed)

    def test_core_stereo_walker_sampled_paths_stay_within_exact_support(self) -> None:
        cases = [
            ("F[C@H](Cl)Br", 0),
            ("F/C=C\\Cl", 0),
            ("F/C(Cl)=C/F", 0),
        ]

        for smiles, root_idx in cases:
            prepared = prepare_smiles_graph(
                parse_smiles(smiles),
                self.policy,
                surface_kind=CONNECTED_STEREO_SURFACE,
            )
            expected = enumerate_rooted_connected_stereo_smiles_support(prepared, root_idx)

            for seed in range(3):
                with self.subTest(smiles=smiles, root_idx=root_idx, seed=seed):
                    rng = random.Random(seed)
                    walker = CORE_MODULE.RootedConnectedStereoWalker(prepared, root_idx)
                    state = walker.initial_state()
                    while not walker.is_terminal(state):
                        options = tuple(walker.next_token_support(state))
                        assert_prefix_options_match_outputs(self, state.prefix, options, expected)
                        chosen_token = rng.choice(options)
                        self.assertTrue(any(output.startswith(state.prefix + chosen_token) for output in expected))
                        state = walker.advance_token(state, chosen_token)
                    self.assertIn(state.prefix, expected)

    def test_core_stereo_walker_rejects_invalid_token(self) -> None:
        prepared = prepare_smiles_graph(
            parse_smiles("F/C=C\\Cl"),
            self.policy,
            surface_kind=CONNECTED_STEREO_SURFACE,
        )
        walker = CORE_MODULE.RootedConnectedStereoWalker(prepared, 0)
        state = walker.initial_state()

        with self.assertRaisesRegex(KeyError, "choices"):
            walker.advance_token(state, "(")


if __name__ == "__main__":
    unittest.main()
