from __future__ import annotations

import random
import unittest

from smiles_next_token.reference import (
    RootedConnectedNonStereoWalker,
    RootedConnectedNonStereoWalkerState,
    enumerate_rooted_nonstereo_smiles_support,
    load_default_connected_nonstereo_molecule_cases,
    prepare_smiles_graph,
    validate_rooted_nonstereo_smiles_support,
)
from tests.helpers.cases import NONSTEREO_CURATED_ROOT_CASES
from tests.helpers.mols import parse_smiles
from tests.helpers.policies import load_connected_nonstereo_policy


def enumerate_walker_support(
    walker: RootedConnectedNonStereoWalker,
    state: RootedConnectedNonStereoWalkerState,
) -> set[str]:
    successors_by_token = walker._successors_by_token(state)
    if not successors_by_token:
        assert walker.is_terminal(state)
        return {state.prefix}

    results: set[str] = set()
    for token in sorted(successors_by_token):
        for successor in successors_by_token[token]:
            results.update(enumerate_walker_support(walker, successor))
    return results


class RootedNextTokenWalkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_connected_nonstereo_policy()

    def test_initial_support_is_the_root_atom_token(self) -> None:
        cases = [
            ("Cc1ccccc1", 0, ("C",)),
            ("Cc1ccccc1", 1, ("c",)),
            ("O=[Ti]=O", 1, ("[Ti]",)),
        ]

        for smiles, root_idx, expected in cases:
            with self.subTest(smiles=smiles, root_idx=root_idx):
                walker = RootedConnectedNonStereoWalker.from_mol(
                    parse_smiles(smiles),
                    root_idx,
                    self.policy,
                )
                self.assertEqual(expected, walker.next_token_support(walker.initial_state()))

    def test_walker_exact_support_matches_rooted_enumerator_on_curated_cases(self) -> None:
        for smiles, root_idx in NONSTEREO_CURATED_ROOT_CASES:
            with self.subTest(smiles=smiles, root_idx=root_idx):
                prepared = prepare_smiles_graph(parse_smiles(smiles), self.policy)
                walker = RootedConnectedNonStereoWalker(prepared, root_idx)

                expected = enumerate_rooted_nonstereo_smiles_support(prepared, root_idx)
                observed = enumerate_walker_support(walker, walker.initial_state())

                self.assertEqual(expected, observed)
                self.assertEqual(
                    [],
                    validate_rooted_nonstereo_smiles_support(prepared, root_idx, None, observed),
                )

    def test_support_reporting_matches_exact_successor_tokens_on_sampled_walks(self) -> None:
        cases = [
            ("Cc1ccccc1", 1),
            ("C1CCCC=C1", 0),
            ("O=[Ti]=O", 1),
            ("C[Ge]", 0),
        ]

        for smiles, root_idx in cases:
            prepared = prepare_smiles_graph(parse_smiles(smiles), self.policy)
            expected_support = enumerate_rooted_nonstereo_smiles_support(prepared, root_idx)

            for seed in range(3):
                with self.subTest(smiles=smiles, root_idx=root_idx, seed=seed):
                    rng = random.Random(seed)
                    walker = RootedConnectedNonStereoWalker(prepared, root_idx)
                    state = walker.initial_state()
                    seen_tokens: list[str] = []

                    while not walker.is_terminal(state):
                        exact_successors = walker._successors_by_token(walker.clone_state(state))
                        self.assertEqual(tuple(sorted(exact_successors)), walker.next_token_support(state))
                        options = walker.next_token_support(state)
                        chosen_token = rng.choice(options)
                        seen_tokens.append(chosen_token)
                        walker.advance_token(state, chosen_token, rng=rng)

                    self.assertEqual(state.prefix, "".join(seen_tokens))
                    self.assertIn(state.prefix, expected_support)

    def test_walker_exact_support_matches_rooted_enumerator_on_dataset_slice(self) -> None:
        cases = load_default_connected_nonstereo_molecule_cases(limit=20, max_smiles_length=10)
        self.assertEqual(20, len(cases))

        for case in cases:
            prepared = prepare_smiles_graph(parse_smiles(case.smiles), self.policy)

            for root_idx in range(prepared.atom_count):
                with self.subTest(cid=case.cid, smiles=case.smiles, root_idx=root_idx):
                    walker = RootedConnectedNonStereoWalker(prepared, root_idx)
                    expected = enumerate_rooted_nonstereo_smiles_support(prepared, root_idx)
                    observed = enumerate_walker_support(walker, walker.initial_state())

                    self.assertEqual(expected, observed)

    def test_invalid_token_is_rejected(self) -> None:
        walker = RootedConnectedNonStereoWalker.from_mol(parse_smiles("CCO"), 0, self.policy)
        state = walker.initial_state()

        with self.assertRaisesRegex(KeyError, "choices"):
            walker.advance_token(state, "(")


if __name__ == "__main__":
    unittest.main()
