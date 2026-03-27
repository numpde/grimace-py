from __future__ import annotations

import statistics
import time
import unittest

from rdkit import Chem

from smiles_next_token.reference import (
    DEFAULT_RDKIT_RANDOM_CONNECTED_NONSTEREO_POLICY_PATH,
    ReferencePolicy,
    enumerate_rooted_nonstereo_smiles_support,
    load_default_connected_nonstereo_molecule_cases,
    prepare_smiles_graph,
    validate_rooted_nonstereo_smiles_support,
)

try:
    from smiles_next_token import _core
except ImportError:  # pragma: no cover - exercised only when the extension is absent
    _core = None


class CoreRootedEnumeratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if _core is None:
            raise unittest.SkipTest("private Rust extension is not installed")
        cls.policy = ReferencePolicy.from_path(DEFAULT_RDKIT_RANDOM_CONNECTED_NONSTEREO_POLICY_PATH)

    def test_kernel_matches_python_reference_on_curated_cases(self) -> None:
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
                kernel_prepared = _core.PreparedSmilesGraph(prepared)

                python_support = enumerate_rooted_nonstereo_smiles_support(prepared, root_idx)
                kernel_support = set(
                    kernel_prepared.enumerate_rooted_connected_nonstereo_support(root_idx)
                )

                self.assertEqual(python_support, kernel_support)
                self.assertEqual(
                    [],
                    validate_rooted_nonstereo_smiles_support(
                        prepared,
                        root_idx,
                        None,
                        kernel_support,
                    ),
                )

    def test_kernel_matches_python_reference_for_all_roots_on_dataset_slice(self) -> None:
        cases = load_default_connected_nonstereo_molecule_cases(limit=80, max_smiles_length=14)
        self.assertEqual(80, len(cases))

        for case in cases:
            mol = Chem.MolFromSmiles(case.smiles)
            self.assertIsNotNone(mol)
            assert mol is not None

            prepared = prepare_smiles_graph(mol, self.policy)
            kernel_prepared = _core.PreparedSmilesGraph(prepared)
            for root_idx in range(prepared.atom_count):
                with self.subTest(cid=case.cid, smiles=case.smiles, root_idx=root_idx):
                    python_support = enumerate_rooted_nonstereo_smiles_support(prepared, root_idx)
                    kernel_support = set(
                        kernel_prepared.enumerate_rooted_connected_nonstereo_support(root_idx)
                    )

                    self.assertEqual(python_support, kernel_support)
                    self.assertEqual(
                        [],
                        validate_rooted_nonstereo_smiles_support(
                            prepared,
                            root_idx,
                            None,
                            kernel_support,
                        ),
                    )

    def test_kernel_matches_python_reference_on_curated_awkward_cases(self) -> None:
        for smiles in [
            "O=[Ti]=O",
            "O=[Cr](=O)=O",
            "Cl[Fe]Cl",
            "F[Mg]",
            "C[Ge]",
            "C[Al]",
            "C[Cu]",
            "O=[Se]=O",
            "C[Si]",
            "O=[P]=O",
        ]:
            mol = Chem.MolFromSmiles(smiles)
            self.assertIsNotNone(mol)
            assert mol is not None

            prepared = prepare_smiles_graph(mol, self.policy)
            kernel_prepared = _core.PreparedSmilesGraph(prepared)
            for root_idx in range(prepared.atom_count):
                with self.subTest(smiles=smiles, root_idx=root_idx):
                    python_support = enumerate_rooted_nonstereo_smiles_support(prepared, root_idx)
                    kernel_support = set(
                        kernel_prepared.enumerate_rooted_connected_nonstereo_support(root_idx)
                    )

                    self.assertEqual(python_support, kernel_support)
                    self.assertEqual(
                        [],
                        validate_rooted_nonstereo_smiles_support(
                            prepared,
                            root_idx,
                            None,
                            kernel_support,
                        ),
                    )

    def test_kernel_next_token_path_beats_a_few_rooted_rdkit_random_calls(self) -> None:
        cases = [
            ("Cc1ccccc1", 1, 8),
            ("CC(=O)Oc1ccccc1C(=O)O", 0, 8),
            ("C1COCCC12CO2", 0, 8),
            ("O=[Ti]=O", 1, 8),
        ]
        sampling = self.policy.data["sampling"]
        total_rdkit = 0.0
        total_kernel = 0.0

        for smiles, root_idx, draw_count in cases:
            with self.subTest(smiles=smiles, root_idx=root_idx, draw_count=draw_count):
                mol = Chem.MolFromSmiles(smiles)
                self.assertIsNotNone(mol)
                assert mol is not None

                prepared = prepare_smiles_graph(mol, self.policy)
                kernel_walker = _core.RootedConnectedNonStereoWalker(prepared, root_idx)

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

                def kernel_next_token_path() -> None:
                    state = kernel_walker.initial_state()
                    while not kernel_walker.is_terminal(state):
                        token = kernel_walker.next_token_support(state)[0]
                        state = kernel_walker.advance_token(state, token)

                rdkit_time = self._median_runtime(rdkit_draws)
                kernel_time = self._median_runtime(kernel_next_token_path)
                total_rdkit += rdkit_time
                total_kernel += kernel_time

                self.assertLess(
                    kernel_time,
                    rdkit_time,
                    msg=(
                        f"Expected kernel next-token path to beat {draw_count} rooted "
                        f"RDKit random draws for {smiles!r} at root {root_idx}; "
                        f"rdkit={rdkit_time:.6f}s kernel={kernel_time:.6f}s"
                    ),
                )

        self.assertLess(
            total_kernel,
            total_rdkit * 0.75,
            msg=(
                "Expected the aggregate kernel runtime to stay comfortably below the "
                f"aggregate RDKit random-draw runtime; rdkit={total_rdkit:.6f}s "
                f"kernel={total_kernel:.6f}s"
            ),
        )

    def _median_runtime(self, fn, *, repeats: int = 7) -> float:
        fn()
        timings: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            timings.append(time.perf_counter() - start)
        return statistics.median(timings)


if __name__ == "__main__":
    unittest.main()
