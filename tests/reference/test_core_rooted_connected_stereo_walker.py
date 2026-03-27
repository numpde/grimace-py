from __future__ import annotations

import random
import statistics
import time
import unittest

from rdkit import Chem

from smiles_next_token.reference import (
    CONNECTED_STEREO_SURFACE,
    DEFAULT_RDKIT_RANDOM_CONNECTED_NONSTEREO_POLICY_PATH,
    ReferencePolicy,
    enumerate_rooted_connected_stereo_smiles_support,
    load_default_molecule_cases,
    molecule_is_connected,
    prepare_smiles_graph,
    validate_rooted_connected_stereo_smiles_support,
)

try:
    from smiles_next_token import _core
except ImportError:  # pragma: no cover - exercised only when the extension is absent
    _core = None


def _load_connected_atom_stereo_cases(*, limit: int, max_smiles_length: int) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    for case in load_default_molecule_cases(limit=5000, max_smiles_length=max_smiles_length):
        mol = Chem.MolFromSmiles(case.smiles)
        if mol is None or not molecule_is_connected(mol):
            continue
        if any(str(bond.GetStereo()) != "STEREONONE" or str(bond.GetBondDir()) != "NONE" for bond in mol.GetBonds()):
            continue
        if not any(str(atom.GetChiralTag()) != "CHI_UNSPECIFIED" for atom in mol.GetAtoms()):
            continue
        selected.append((case.cid, case.smiles))
        if len(selected) >= limit:
            break
    return selected


def _load_connected_bond_stereo_cases(*, limit: int, max_smiles_length: int) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    for case in load_default_molecule_cases(limit=5000, max_smiles_length=max_smiles_length):
        mol = Chem.MolFromSmiles(case.smiles)
        if mol is None or not molecule_is_connected(mol):
            continue
        if any(atom.GetIsAromatic() for atom in mol.GetAtoms()):
            continue
        if any(bond.IsInRing() for bond in mol.GetBonds()):
            continue
        if any(str(atom.GetChiralTag()) != "CHI_UNSPECIFIED" for atom in mol.GetAtoms()):
            continue
        stereo_bonds = [bond for bond in mol.GetBonds() if str(bond.GetStereo()) != "STEREONONE"]
        if len(stereo_bonds) != 1:
            continue
        selected.append((case.cid, case.smiles))
        if len(selected) >= limit:
            break
    return selected


class CoreRootedConnectedStereoWalkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if _core is None:
            raise unittest.SkipTest("private Rust extension is not installed")
        cls.policy = ReferencePolicy.from_path(DEFAULT_RDKIT_RANDOM_CONNECTED_NONSTEREO_POLICY_PATH)

    def test_core_stereo_walker_exact_support_matches_reference_on_curated_cases(self) -> None:
        cases = [
            ("F[C@H](Cl)Br", 0),
            ("F[C@](Cl)(Br)I", 0),
            ("C[C@H](O)[C@@H](F)Cl", 0),
            ("F/C=C\\Cl", 0),
            ("C/C=C/C=C/C(=O)O", 0),
            ("C/C=C(\\C)/C(=O)O", 0),
            ("C/C(=N\\\\OC(=O)NC)/SC", 0),
            ("F/C(Cl)=C/F", 0),
        ]

        for smiles, root_idx in cases:
            with self.subTest(smiles=smiles, root_idx=root_idx):
                mol = Chem.MolFromSmiles(smiles)
                self.assertIsNotNone(mol)
                assert mol is not None
                prepared = prepare_smiles_graph(mol, self.policy, surface_kind=CONNECTED_STEREO_SURFACE)
                walker = _core.RootedConnectedStereoWalker(prepared, root_idx)

                observed = set(walker.enumerate_support())
                expected = enumerate_rooted_connected_stereo_smiles_support(prepared, root_idx)
                self.assertEqual(expected, observed)
                self.assertEqual(
                    [],
                    validate_rooted_connected_stereo_smiles_support(prepared, root_idx, None, observed),
                )

    def test_core_stereo_walker_exact_support_matches_reference_on_dataset_slice(self) -> None:
        cases = _load_connected_atom_stereo_cases(limit=2, max_smiles_length=16)
        bond_cases = _load_connected_bond_stereo_cases(limit=1, max_smiles_length=18)
        self.assertEqual(2, len(cases))
        self.assertEqual(1, len(bond_cases))

        for cid, smiles in [*cases, *bond_cases]:
            mol = Chem.MolFromSmiles(smiles)
            self.assertIsNotNone(mol)
            assert mol is not None
            prepared = prepare_smiles_graph(mol, self.policy, surface_kind=CONNECTED_STEREO_SURFACE)
            for root_idx in range(prepared.atom_count):
                with self.subTest(cid=cid, smiles=smiles, root_idx=root_idx):
                    walker = _core.RootedConnectedStereoWalker(prepared, root_idx)
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
            mol = Chem.MolFromSmiles(smiles)
            self.assertIsNotNone(mol)
            assert mol is not None
            prepared = prepare_smiles_graph(mol, self.policy, surface_kind=CONNECTED_STEREO_SURFACE)
            expected = enumerate_rooted_connected_stereo_smiles_support(prepared, root_idx)

            for seed in range(3):
                with self.subTest(smiles=smiles, root_idx=root_idx, seed=seed):
                    rng = random.Random(seed)
                    walker = _core.RootedConnectedStereoWalker(prepared, root_idx)
                    state = walker.initial_state()
                    while not walker.is_terminal(state):
                        options = tuple(walker.next_token_support(state))
                        self.assertTrue(options)
                        self.assertTrue(
                            any(output.startswith(state.prefix + token) for output in expected for token in options)
                        )
                        chosen_token = rng.choice(options)
                        self.assertTrue(any(output.startswith(state.prefix + chosen_token) for output in expected))
                        state = walker.advance_token(state, chosen_token)
                    self.assertIn(state.prefix, expected)

    def test_core_stereo_walker_rejects_invalid_token(self) -> None:
        mol = Chem.MolFromSmiles("F/C=C\\Cl")
        self.assertIsNotNone(mol)
        assert mol is not None
        prepared = prepare_smiles_graph(mol, self.policy, surface_kind=CONNECTED_STEREO_SURFACE)
        walker = _core.RootedConnectedStereoWalker(prepared, 0)
        state = walker.initial_state()

        with self.assertRaisesRegex(KeyError, "choices"):
            walker.advance_token(state, "(")

    def test_core_stereo_next_token_path_beats_exact_support_on_representative_cases(self) -> None:
        cases = [
            ("F/C=C\\Cl", 0),
            ("F/C(Cl)=C/F", 0),
            ("C/C=C/C=C/C(=O)O", 0),
            ("C/C(=N\\\\OC(=O)NC)/SC", 0),
            ("C[C@H](O)[C@@H](F)Cl", 0),
        ]
        total_next_token = 0.0
        total_exact = 0.0

        for smiles, root_idx in cases:
            with self.subTest(smiles=smiles, root_idx=root_idx):
                mol = Chem.MolFromSmiles(smiles)
                self.assertIsNotNone(mol)
                assert mol is not None
                prepared = prepare_smiles_graph(mol, self.policy, surface_kind=CONNECTED_STEREO_SURFACE)
                walker = _core.RootedConnectedStereoWalker(prepared, root_idx)
                kernel_prepared = _core.PreparedSmilesGraph(prepared)

                def next_token_path() -> None:
                    state = walker.initial_state()
                    while not walker.is_terminal(state):
                        token = walker.next_token_support(state)[0]
                        state = walker.advance_token(state, token)

                def exact_support() -> None:
                    kernel_prepared.enumerate_rooted_connected_stereo_support(root_idx)

                next_token_time = self._median_runtime(next_token_path)
                exact_time = self._median_runtime(exact_support)
                total_next_token += next_token_time
                total_exact += exact_time

                self.assertLess(
                    next_token_time,
                    exact_time,
                    msg=(
                        f"Expected stereo next-token path to beat exact support for {smiles!r} "
                        f"at root {root_idx}; next_token={next_token_time:.6f}s exact={exact_time:.6f}s"
                    ),
                )

        self.assertLess(
            total_next_token,
            total_exact * 0.2,
            msg=(
                "Expected aggregate stereo next-token runtime to stay well below exact-support runtime; "
                f"next_token={total_next_token:.6f}s exact={total_exact:.6f}s"
            ),
        )

    def test_core_stereo_next_token_path_beats_rooted_rdkit_random_calls_on_longer_atom_stereo_cases(self) -> None:
        cases = [
            ("CN1CCC[C@H]1C2=CN=CC=C2", 0, 16),
            ("C1=CC(=C(C=C1C[C@@H](C(=O)O)N)O)O", 0, 16),
            ("C[C@@H]1CC[C@H]([C@@H](C1)O)C(C)C", 0, 16),
            ("C[C@@H](C1=CC2=C(C=C1)C=C(C=C2)OC)C(=O)O", 0, 16),
        ]
        sampling = self.policy.data["sampling"]
        total_rdkit = 0.0
        total_next_token = 0.0

        for smiles, root_idx, draw_count in cases:
            with self.subTest(smiles=smiles, root_idx=root_idx, draw_count=draw_count):
                mol = Chem.MolFromSmiles(smiles)
                self.assertIsNotNone(mol)
                assert mol is not None
                prepared = prepare_smiles_graph(mol, self.policy, surface_kind=CONNECTED_STEREO_SURFACE)
                walker = _core.RootedConnectedStereoWalker(prepared, root_idx)

                def rdkit_draws() -> None:
                    for _ in range(draw_count):
                        Chem.MolToSmiles(
                            mol,
                            canonical=False,
                            doRandom=True,
                            rootedAtAtom=root_idx,
                            isomericSmiles=bool(sampling["isomericSmiles"]),
                            kekuleSmiles=bool(sampling["kekuleSmiles"]),
                            allBondsExplicit=bool(sampling["allBondsExplicit"]),
                            allHsExplicit=bool(sampling["allHsExplicit"]),
                            ignoreAtomMapNumbers=bool(sampling["ignoreAtomMapNumbers"]),
                        )

                def next_token_path() -> None:
                    state = walker.initial_state()
                    while not walker.is_terminal(state):
                        token = walker.next_token_support(state)[0]
                        state = walker.advance_token(state, token)

                rdkit_time = self._median_runtime(rdkit_draws)
                next_token_time = self._median_runtime(next_token_path)
                total_rdkit += rdkit_time
                total_next_token += next_token_time

                self.assertLess(
                    next_token_time,
                    rdkit_time,
                    msg=(
                        f"Expected stereo next-token path to beat {draw_count} rooted "
                        f"RDKit random draws for {smiles!r} at root {root_idx}; "
                        f"rdkit={rdkit_time:.6f}s next_token={next_token_time:.6f}s"
                    ),
                )

        self.assertLess(
            total_next_token,
            total_rdkit * 0.75,
            msg=(
                "Expected the aggregate stereo next-token runtime on longer atom-stereo cases "
                f"to stay below the aggregate RDKit random-draw runtime; "
                f"rdkit={total_rdkit:.6f}s next_token={total_next_token:.6f}s"
            ),
        )

    def _median_runtime(self, fn, *, repeats: int = 5) -> float:
        fn()
        timings: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            timings.append(time.perf_counter() - start)
        return statistics.median(timings)


if __name__ == "__main__":
    unittest.main()
