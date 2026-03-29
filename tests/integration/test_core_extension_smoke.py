from __future__ import annotations

import unittest

from tests.helpers.kernel import CORE_MODULE
from tests.helpers.mols import parse_smiles


class CoreExtensionSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if CORE_MODULE is None:
            raise unittest.SkipTest("private Rust extension is not installed")

    def test_core_objects_construct_and_advance(self) -> None:
        from grimace import _runtime

        mol = parse_smiles("CCO")
        prepared = _runtime.prepare_smiles_graph(
            mol,
            flags=_runtime.MolToSmilesFlags(
                isomeric_smiles=False,
                rooted_at_atom=0,
                canonical=False,
                do_random=True,
            ),
        )
        kernel_prepared = CORE_MODULE.PreparedSmilesGraph(prepared)
        walker = _runtime.make_nonstereo_walker(prepared, 0)
        state = walker.initial_state()

        self.assertEqual(prepared.atom_count, kernel_prepared.atom_count)
        self.assertTrue(walker.next_token_support(state))


if __name__ == "__main__":
    unittest.main()
