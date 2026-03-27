from __future__ import annotations

import random
import unittest

from rdkit import Chem

from smiles_next_token.reference import (
    DEFAULT_RDKIT_RANDOM_CONNECTED_NONSTEREO_POLICY_PATH,
    ReferencePolicy,
    RootedConnectedNonStereoWalker,
    RootedConnectedNonStereoWalkerState,
    enumerate_rooted_nonstereo_smiles_support,
    load_default_connected_nonstereo_molecule_cases,
    prepare_smiles_graph,
    validate_rooted_nonstereo_smiles_support,
)


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
        cls.policy = ReferencePolicy.from_path(DEFAULT_RDKIT_RANDOM_CONNECTED_NONSTEREO_POLICY_PATH)

    def test_initial_support_is_the_root_atom_token(self) -> None:
        cases = [
            ("Cc1ccccc1", 0, ("C",)),
            ("Cc1ccccc1", 1, ("c",)),
            ("O=[Ti]=O", 1, ("[Ti]",)),
        ]

        for smiles, root_idx, expected in cases:
            with self.subTest(smiles=smiles, root_idx=root_idx):
                mol = Chem.MolFromSmiles(smiles)
                self.assertIsNotNone(mol)
                assert mol is not None
                walker = RootedConnectedNonStereoWalker.from_mol(
                    mol,
                    root_idx,
                    self.policy,
                )
                self.assertEqual(expected, walker.next_token_support(walker.initial_state()))

    def test_walker_exact_support_matches_rooted_enumerator_on_curated_cases(self) -> None:
        cases = [
            ("Cc1ccccc1", 0),
            ("Cc1ccccc1", 1),
            ("C1CCCC=C1", 0),
            ("c1ccncc1", 0),
            ("O=[Ti]=O", 0),
            ("O=[Ti]=O", 1),
            ("C[Ge]", 0),
            ("C[Ge]", 1),
        ]

        for smiles, root_idx in cases:
            with self.subTest(smiles=smiles, root_idx=root_idx):
                mol = Chem.MolFromSmiles(smiles)
                self.assertIsNotNone(mol)
                assert mol is not None
                prepared = prepare_smiles_graph(mol, self.policy)
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
            mol = Chem.MolFromSmiles(smiles)
            self.assertIsNotNone(mol)
            assert mol is not None
            prepared = prepare_smiles_graph(mol, self.policy)
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
            mol = Chem.MolFromSmiles(case.smiles)
            self.assertIsNotNone(mol)
            assert mol is not None
            prepared = prepare_smiles_graph(mol, self.policy)

            for root_idx in range(prepared.atom_count):
                with self.subTest(cid=case.cid, smiles=case.smiles, root_idx=root_idx):
                    walker = RootedConnectedNonStereoWalker(prepared, root_idx)
                    expected = enumerate_rooted_nonstereo_smiles_support(prepared, root_idx)
                    observed = enumerate_walker_support(walker, walker.initial_state())

                    self.assertEqual(expected, observed)

    def test_invalid_token_is_rejected(self) -> None:
        mol = Chem.MolFromSmiles("CCO")
        self.assertIsNotNone(mol)
        assert mol is not None
        walker = RootedConnectedNonStereoWalker.from_mol(mol, 0, self.policy)
        state = walker.initial_state()

        with self.assertRaisesRegex(KeyError, "choices"):
            walker.advance_token(state, "(")


if __name__ == "__main__":
    unittest.main()
