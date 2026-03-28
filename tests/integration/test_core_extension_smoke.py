from __future__ import annotations

import unittest

from smiles_next_token.reference import prepare_smiles_graph
from tests.helpers.kernel import CORE_MODULE
from tests.helpers.mols import parse_smiles
from tests.helpers.policies import load_connected_nonstereo_policy


class CoreExtensionSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if CORE_MODULE is None:
            raise unittest.SkipTest("private Rust extension is not installed")
        cls.policy = load_connected_nonstereo_policy()

    def test_core_objects_construct_and_advance(self) -> None:
        mol = parse_smiles("CCO")
        prepared = prepare_smiles_graph(mol, self.policy)
        kernel_prepared = CORE_MODULE.PreparedSmilesGraph(prepared)
        walker = CORE_MODULE.RootedConnectedNonStereoWalker(prepared, 0)
        state = walker.initial_state()

        self.assertEqual(prepared.atom_count, kernel_prepared.atom_count)
        self.assertTrue(walker.next_token_support(state))


if __name__ == "__main__":
    unittest.main()
