from __future__ import annotations

import unittest

from smiles_next_token.reference import (
    RootedConnectedNonStereoWalker,
    enumerate_rooted_connected_stereo_smiles_support,
    enumerate_rooted_nonstereo_smiles_support,
    prepare_smiles_graph,
)
from tests.helpers.mols import parse_smiles
from tests.helpers.policies import load_connected_nonstereo_policy


class PythonApiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_connected_nonstereo_policy()

    def test_reference_surface_end_to_end_smoke(self) -> None:
        mol = parse_smiles("CCO")
        prepared = prepare_smiles_graph(mol, self.policy)

        support = enumerate_rooted_nonstereo_smiles_support(prepared, 0)
        walker = RootedConnectedNonStereoWalker(prepared, 0)

        self.assertTrue(support)
        self.assertTrue(walker.next_token_support(walker.initial_state()))

    def test_stereo_reference_surface_smoke(self) -> None:
        mol = parse_smiles("F/C=C\\Cl")
        support = enumerate_rooted_connected_stereo_smiles_support(mol, 0, self.policy)

        self.assertTrue(support)


if __name__ == "__main__":
    unittest.main()
