from __future__ import annotations

import os
import statistics
import time
import unittest

from rdkit import Chem

from smiles_next_token._reference import CONNECTED_STEREO_SURFACE, prepare_smiles_graph
from tests.helpers.kernel import CORE_MODULE
from tests.helpers.mols import parse_smiles
from tests.helpers.policies import load_connected_nonstereo_policy


@unittest.skipUnless(
    os.environ.get("RUN_PERF_TESTS") == "1",
    "set RUN_PERF_TESTS=1 to run performance checks",
)
class KernelStereoPerfTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if CORE_MODULE is None:
            raise unittest.SkipTest("private Rust extension is not installed")
        cls.policy = load_connected_nonstereo_policy()

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
                prepared = prepare_smiles_graph(
                    parse_smiles(smiles),
                    self.policy,
                    surface_kind=CONNECTED_STEREO_SURFACE,
                )
                walker = CORE_MODULE.RootedConnectedStereoWalker(prepared, root_idx)
                kernel_prepared = CORE_MODULE.PreparedSmilesGraph(prepared)

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

                self.assertLess(next_token_time, exact_time)

        self.assertLess(total_next_token, total_exact * 0.2)

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
                mol = parse_smiles(smiles)
                prepared = prepare_smiles_graph(
                    mol,
                    self.policy,
                    surface_kind=CONNECTED_STEREO_SURFACE,
                )
                walker = CORE_MODULE.RootedConnectedStereoWalker(prepared, root_idx)

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

                self.assertLess(next_token_time, rdkit_time)

        self.assertLess(total_next_token, total_rdkit * 0.75)

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
