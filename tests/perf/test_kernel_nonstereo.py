from __future__ import annotations

import os
import statistics
import time
import unittest

from rdkit import Chem

from grimace._reference import (
    enumerate_rooted_nonstereo_smiles_support,
    load_default_connected_nonstereo_molecule_cases,
    prepare_smiles_graph,
)
from tests.helpers.kernel import CORE_MODULE
from tests.helpers.mols import parse_smiles
from tests.helpers.policies import load_connected_nonstereo_policy


@unittest.skipUnless(
    os.environ.get("RUN_PERF_TESTS") == "1",
    "set RUN_PERF_TESTS=1 to run performance checks",
)
class KernelNonStereoPerfTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if CORE_MODULE is None:
            raise unittest.SkipTest("private Rust extension is not installed")
        cls.policy = load_connected_nonstereo_policy()

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
                mol = parse_smiles(smiles)
                prepared = prepare_smiles_graph(mol, self.policy)
                kernel_walker = CORE_MODULE.RootedConnectedNonStereoWalker(prepared, root_idx)

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

                self.assertLess(kernel_time, rdkit_time)

        self.assertLess(total_kernel, total_rdkit * 0.75)

    def test_kernel_exact_support_still_handles_dataset_slice(self) -> None:
        cases = load_default_connected_nonstereo_molecule_cases(limit=8, max_smiles_length=12)
        self.assertEqual(8, len(cases))
        for case in cases:
            prepared = prepare_smiles_graph(parse_smiles(case.smiles), self.policy)
            support = enumerate_rooted_nonstereo_smiles_support(prepared, 0)
            self.assertTrue(support)

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
