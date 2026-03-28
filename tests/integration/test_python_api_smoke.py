from __future__ import annotations

import unittest

import smiles_next_token
from smiles_next_token import rdkit_reference
from tests.helpers.mols import parse_smiles
from tests.helpers.policies import load_connected_nonstereo_policy


class PythonApiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_connected_nonstereo_policy()

    def test_top_level_api_exposes_runtime_surface(self) -> None:
        self.assertTrue(hasattr(smiles_next_token, "prepare_smiles_graph"))
        self.assertTrue(hasattr(smiles_next_token, "make_prepared_graph"))
        self.assertTrue(hasattr(smiles_next_token, "make_nonstereo_walker"))
        self.assertTrue(hasattr(smiles_next_token, "make_stereo_walker"))
        self.assertTrue(hasattr(smiles_next_token, "enumerate_rooted_connected_nonstereo_smiles_support"))
        self.assertTrue(hasattr(smiles_next_token, "enumerate_rooted_connected_stereo_smiles_support"))
        self.assertTrue(hasattr(smiles_next_token, "HAVE_CORE_BINDINGS"))

    def test_rdkit_reference_module_exposes_oracle_surface(self) -> None:
        self.assertTrue(hasattr(rdkit_reference, "ReferencePolicy"))
        self.assertTrue(hasattr(rdkit_reference, "sample_rdkit_random_smiles"))
        self.assertTrue(hasattr(rdkit_reference, "build_core_exact_sets_artifact"))

    def test_top_level_prepare_prefers_runtime_prepared_graph(self) -> None:
        mol = parse_smiles("CCO")
        prepared = smiles_next_token.prepare_smiles_graph(mol, self.policy)

        self.assertEqual("CCO", prepared.identity_smiles)
        self.assertEqual(
            smiles_next_token.prepared_smiles_graph_schema_version(),
            prepared.schema_version,
        )
        if smiles_next_token.HAVE_CORE_BINDINGS:
            self.assertIsInstance(prepared, smiles_next_token.PreparedSmilesGraph)
            self.assertNotIsInstance(prepared, rdkit_reference.PreparedSmilesGraph)
        else:
            self.assertIsInstance(prepared, rdkit_reference.PreparedSmilesGraph)

    def test_top_level_prepare_accepts_reference_prepared_graph(self) -> None:
        reference_prepared = rdkit_reference.prepare_smiles_graph(
            parse_smiles("CCO"),
            self.policy,
        )
        prepared = smiles_next_token.prepare_smiles_graph(reference_prepared, self.policy)

        self.assertEqual(reference_prepared.to_dict(), prepared.to_dict())
        if smiles_next_token.HAVE_CORE_BINDINGS:
            self.assertIsInstance(prepared, smiles_next_token.PreparedSmilesGraph)
        else:
            self.assertIs(prepared, reference_prepared)

    def test_top_level_runtime_nonstereo_surface_smoke(self) -> None:
        mol = parse_smiles("CCO")
        expected = rdkit_reference.enumerate_rooted_nonstereo_smiles_support(
            mol,
            0,
            self.policy,
        )

        support = smiles_next_token.enumerate_rooted_connected_nonstereo_smiles_support(
            mol,
            0,
            self.policy,
        )
        walker = smiles_next_token.make_nonstereo_walker(mol, 0, self.policy)

        self.assertEqual(expected, support)
        self.assertTrue(tuple(walker.next_token_support(walker.initial_state())))

    def test_top_level_runtime_stereo_surface_smoke(self) -> None:
        mol = parse_smiles("F/C=C\\Cl")
        expected = rdkit_reference.enumerate_rooted_connected_stereo_smiles_support(
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

    def test_top_level_stereo_walker_factory_prefers_core(self) -> None:
        if not smiles_next_token.HAVE_CORE_BINDINGS:
            with self.assertRaisesRegex(RuntimeError, "requires the Rust core bindings"):
                smiles_next_token.make_stereo_walker(
                    parse_smiles("F/C=C\\Cl"),
                    0,
                    self.policy,
                )
            return

        walker = smiles_next_token.make_stereo_walker(
            parse_smiles("F/C=C\\Cl"),
            0,
            self.policy,
        )
        self.assertTrue(tuple(walker.next_token_support(walker.initial_state())))


if __name__ == "__main__":
    unittest.main()
