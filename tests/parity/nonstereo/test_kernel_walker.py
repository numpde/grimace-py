from __future__ import annotations

import random
import unittest

from smiles_next_token.reference import (
    RootedConnectedNonStereoWalker as PythonRootedConnectedNonStereoWalker,
    enumerate_rooted_nonstereo_smiles_support,
    load_default_connected_nonstereo_molecule_cases,
    prepare_smiles_graph,
    validate_rooted_nonstereo_smiles_support,
)
from tests.helpers.cases import NONSTEREO_CURATED_ROOT_CASES
from tests.helpers.kernel import CORE_MODULE
from tests.helpers.mols import parse_smiles
from tests.helpers.policies import load_connected_nonstereo_policy


class CoreRootedNextTokenWalkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if CORE_MODULE is None:
            raise unittest.SkipTest("private Rust extension is not installed")
        cls.policy = load_connected_nonstereo_policy()

    def test_core_walker_exact_support_matches_python_reference_on_curated_cases(self) -> None:
        for smiles, root_idx in NONSTEREO_CURATED_ROOT_CASES:
            with self.subTest(smiles=smiles, root_idx=root_idx):
                prepared = prepare_smiles_graph(parse_smiles(smiles), self.policy)
                core_walker = CORE_MODULE.RootedConnectedNonStereoWalker(prepared, root_idx)

                expected = enumerate_rooted_nonstereo_smiles_support(prepared, root_idx)
                observed = set(core_walker.enumerate_support())

                self.assertEqual(expected, observed)
                self.assertEqual(
                    [],
                    validate_rooted_nonstereo_smiles_support(prepared, root_idx, None, observed),
                )

    def test_core_walker_support_reporting_matches_python_walker_on_sampled_walks(self) -> None:
        cases = [
            ("Cc1ccccc1", 1),
            ("C1CCCC=C1", 0),
            ("O=[Ti]=O", 1),
            ("C[Ge]", 0),
        ]

        for smiles, root_idx in cases:
            prepared = prepare_smiles_graph(parse_smiles(smiles), self.policy)

            for seed in range(3):
                with self.subTest(smiles=smiles, root_idx=root_idx, seed=seed):
                    rng = random.Random(seed)
                    python_walker = PythonRootedConnectedNonStereoWalker(prepared, root_idx)
                    core_walker = CORE_MODULE.RootedConnectedNonStereoWalker(prepared, root_idx)
                    python_state = python_walker.initial_state()
                    core_state = core_walker.initial_state()

                    while not python_walker.is_terminal(python_state):
                        python_options = python_walker.next_token_support(python_state)
                        core_options = tuple(core_walker.next_token_support(core_state))
                        self.assertEqual(python_options, core_options)

                        chosen_token = rng.choice(python_options)
                        python_state = python_walker.advance_token(python_state, chosen_token)
                        core_state = core_walker.advance_token(core_state, chosen_token)

                        self.assertEqual(python_state.prefix, core_state.prefix)

                    self.assertTrue(core_walker.is_terminal(core_state))
                    self.assertEqual(python_state.prefix, core_state.prefix)

    def test_core_walker_matches_python_walker_on_dataset_slice_sampled_paths(self) -> None:
        cases = load_default_connected_nonstereo_molecule_cases(limit=12, max_smiles_length=10)
        self.assertEqual(12, len(cases))

        for case in cases:
            prepared = prepare_smiles_graph(parse_smiles(case.smiles), self.policy)

            for root_idx in range(prepared.atom_count):
                for seed in range(2):
                    with self.subTest(
                        cid=case.cid,
                        smiles=case.smiles,
                        root_idx=root_idx,
                        seed=seed,
                    ):
                        rng = random.Random(seed)
                        python_walker = PythonRootedConnectedNonStereoWalker(prepared, root_idx)
                        core_walker = CORE_MODULE.RootedConnectedNonStereoWalker(prepared, root_idx)
                        python_state = python_walker.initial_state()
                        core_state = core_walker.initial_state()

                        while not python_walker.is_terminal(python_state):
                            python_options = python_walker.next_token_support(python_state)
                            core_options = tuple(core_walker.next_token_support(core_state))
                            self.assertEqual(python_options, core_options)

                            chosen_token = rng.choice(python_options)
                            python_state = python_walker.advance_token(python_state, chosen_token)
                            core_state = core_walker.advance_token(core_state, chosen_token)

                            self.assertEqual(python_state.prefix, core_state.prefix)

                        self.assertTrue(core_walker.is_terminal(core_state))
                        self.assertEqual(python_state.prefix, core_state.prefix)

    def test_core_walker_exact_support_matches_python_reference_on_dataset_slice(self) -> None:
        cases = load_default_connected_nonstereo_molecule_cases(limit=20, max_smiles_length=10)
        self.assertEqual(20, len(cases))

        for case in cases:
            prepared = prepare_smiles_graph(parse_smiles(case.smiles), self.policy)

            for root_idx in range(prepared.atom_count):
                with self.subTest(cid=case.cid, smiles=case.smiles, root_idx=root_idx):
                    core_walker = CORE_MODULE.RootedConnectedNonStereoWalker(prepared, root_idx)
                    observed = set(core_walker.enumerate_support())
                    expected = enumerate_rooted_nonstereo_smiles_support(prepared, root_idx)
                    self.assertEqual(expected, observed)

    def test_core_walker_rejects_invalid_token(self) -> None:
        prepared = prepare_smiles_graph(parse_smiles("CCO"), self.policy)
        core_walker = CORE_MODULE.RootedConnectedNonStereoWalker(prepared, 0)
        state = core_walker.initial_state()

        with self.assertRaisesRegex(KeyError, "choices"):
            core_walker.advance_token(state, "(")


if __name__ == "__main__":
    unittest.main()
