from __future__ import annotations

import unittest

import smiles_next_token
from smiles_next_token._reference import (
    prepare_smiles_graph as prepare_reference_smiles_graph,
)
from tests.helpers.kernel import CORE_MODULE
from tests.helpers.mols import parse_smiles
from tests.helpers.policies import load_connected_nonstereo_policy


class PythonApiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_connected_nonstereo_policy()

    def test_top_level_api_exposes_only_narrow_runtime_surface(self) -> None:
        self.assertTrue(hasattr(smiles_next_token, "ReferencePolicy"))
        self.assertFalse(hasattr(smiles_next_token, "prepare_smiles_graph"))
        self.assertFalse(hasattr(smiles_next_token, "make_prepared_graph"))
        self.assertFalse(hasattr(smiles_next_token, "make_nonstereo_walker"))
        self.assertFalse(hasattr(smiles_next_token, "make_stereo_walker"))
        self.assertFalse(hasattr(smiles_next_token, "PreparedSmilesGraph"))
        self.assertFalse(hasattr(smiles_next_token, "HAVE_CORE_BINDINGS"))
        if CORE_MODULE is None:
            with self.assertRaises(ImportError):
                getattr(
                    smiles_next_token,
                    "enumerate_rooted_connected_nonstereo_smiles_support",
                )
            with self.assertRaises(ImportError):
                getattr(
                    smiles_next_token,
                    "enumerate_rooted_connected_stereo_smiles_support",
                )
            return

        self.assertTrue(callable(smiles_next_token.enumerate_rooted_connected_nonstereo_smiles_support))
        self.assertTrue(callable(smiles_next_token.enumerate_rooted_connected_stereo_smiles_support))

    def test_internal_runtime_bridge_accepts_reference_prepared_graph(self) -> None:
        if CORE_MODULE is None:
            raise unittest.SkipTest("private Rust extension is not installed")
        from smiles_next_token import _runtime

        reference_prepared = prepare_reference_smiles_graph(parse_smiles("CCO"), self.policy)
        prepared = _runtime.prepare_smiles_graph(reference_prepared, self.policy)

        self.assertEqual(reference_prepared.to_dict(), prepared.to_dict())

    def test_top_level_runtime_nonstereo_surface_smoke(self) -> None:
        if CORE_MODULE is None:
            raise unittest.SkipTest("private Rust extension is not installed")
        from smiles_next_token import _runtime

        mol = parse_smiles("CCO")
        expected = _runtime.enumerate_rooted_connected_nonstereo_smiles_support(
            mol,
            0,
            self.policy,
        )

        support = smiles_next_token.enumerate_rooted_connected_nonstereo_smiles_support(
            mol,
            0,
            self.policy,
        )

        self.assertEqual(expected, support)

    def test_top_level_runtime_stereo_surface_smoke(self) -> None:
        if CORE_MODULE is None:
            raise unittest.SkipTest("private Rust extension is not installed")
        from smiles_next_token import _runtime

        mol = parse_smiles("F/C=C\\Cl")
        expected = _runtime.enumerate_rooted_connected_stereo_smiles_support(
            mol,
            0,
            self.policy,
        )
        support = smiles_next_token.enumerate_rooted_connected_stereo_smiles_support(
            mol,
            0,
            self.policy,
        )

        self.assertEqual(expected, support)


if __name__ == "__main__":
    unittest.main()
