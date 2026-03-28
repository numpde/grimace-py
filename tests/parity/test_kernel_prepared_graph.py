from __future__ import annotations

from copy import deepcopy
import unittest

from smiles_next_token.reference import (
    CONNECTED_STEREO_SURFACE,
    PreparedSmilesGraph as PythonPreparedSmilesGraph,
    prepare_smiles_graph,
)
from tests.helpers.kernel import CORE_MODULE
from tests.helpers.mols import parse_smiles
from tests.helpers.policies import load_connected_nonstereo_policy


class CorePreparedSmilesGraphContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if CORE_MODULE is None:
            raise unittest.SkipTest("private Rust extension is not installed")

        cls.policy = load_connected_nonstereo_policy()

    def test_kernel_prepared_graph_accepts_python_reference_object(self) -> None:
        prepared = prepare_smiles_graph(parse_smiles("Cc1ccccc1"), self.policy)
        kernel_prepared = CORE_MODULE.PreparedSmilesGraph(prepared)

        self.assertEqual(prepared.schema_version, kernel_prepared.schema_version)
        self.assertEqual(prepared.surface_kind, kernel_prepared.surface_kind)
        self.assertEqual(prepared.policy_name, kernel_prepared.policy_name)
        self.assertEqual(prepared.policy_digest, kernel_prepared.policy_digest)
        self.assertEqual(prepared.rdkit_version, kernel_prepared.rdkit_version)
        self.assertEqual(prepared.identity_smiles, kernel_prepared.identity_smiles)
        self.assertEqual(prepared.atom_count, kernel_prepared.atom_count)
        self.assertEqual(prepared.bond_count, kernel_prepared.bond_count)
        self.assertEqual(list(prepared.atom_tokens), kernel_prepared.atom_tokens)
        self.assertEqual(list(prepared.neighbors[0]), kernel_prepared.neighbors_of(0))
        self.assertEqual(prepared.bond_token(0, 1), kernel_prepared.bond_token(0, 1))

    def test_kernel_prepared_graph_dict_roundtrip_is_lossless(self) -> None:
        prepared = prepare_smiles_graph(parse_smiles("O=[Ti]=O"), self.policy)
        kernel_prepared = CORE_MODULE.PreparedSmilesGraph(prepared.to_dict())
        rebuilt = PythonPreparedSmilesGraph.from_dict(kernel_prepared.to_dict())

        self.assertEqual(prepared, rebuilt)

    def test_kernel_prepared_graph_dict_roundtrip_is_lossless_for_stereo_surface(self) -> None:
        prepared = prepare_smiles_graph(
            parse_smiles("F/C=C\\Cl"),
            self.policy,
            surface_kind=CONNECTED_STEREO_SURFACE,
        )
        kernel_prepared = CORE_MODULE.PreparedSmilesGraph(prepared.to_dict())
        rebuilt = PythonPreparedSmilesGraph.from_dict(kernel_prepared.to_dict())

        self.assertEqual(prepared, rebuilt)

    def test_kernel_prepared_graph_validates_policy_provenance(self) -> None:
        prepared = prepare_smiles_graph(parse_smiles("CC#N"), self.policy)
        kernel_prepared = CORE_MODULE.PreparedSmilesGraph(prepared)

        kernel_prepared.validate_policy(prepared.policy_name, prepared.policy_digest)
        with self.assertRaisesRegex(ValueError, "does not match the provided policy"):
            kernel_prepared.validate_policy(prepared.policy_name, "badc0de")

    def test_kernel_rooted_support_rejects_out_of_range_root(self) -> None:
        prepared = prepare_smiles_graph(parse_smiles("CC"), self.policy)
        kernel_prepared = CORE_MODULE.PreparedSmilesGraph(prepared)

        with self.assertRaisesRegex(IndexError, "root_idx out of range"):
            kernel_prepared.enumerate_rooted_connected_nonstereo_support(-1)
        with self.assertRaisesRegex(IndexError, "root_idx out of range"):
            kernel_prepared.enumerate_rooted_connected_nonstereo_support(2)
        with self.assertRaisesRegex(IndexError, "root_idx out of range"):
            CORE_MODULE.RootedConnectedNonStereoWalker(prepared, -1)
        with self.assertRaisesRegex(IndexError, "root_idx out of range"):
            CORE_MODULE.RootedConnectedNonStereoWalker(prepared, 2)

    def test_kernel_prepared_graph_rejects_malformed_transport_dicts(self) -> None:
        prepared = prepare_smiles_graph(parse_smiles("CCO"), self.policy)
        stereo_prepared = prepare_smiles_graph(
            parse_smiles("F/C=C\\Cl"),
            self.policy,
            surface_kind=CONNECTED_STEREO_SURFACE,
        )
        cases = [
            (
                "schema_version",
                prepared.to_dict(),
                lambda data: data.__setitem__("schema_version", 0),
                "schema version",
            ),
            (
                "unsorted_neighbors",
                prepared.to_dict(),
                lambda data: data["neighbors"].__setitem__(1, [2, 0]),
                "neighbor rows must be sorted",
            ),
            (
                "asymmetric_bond_tokens",
                prepared.to_dict(),
                lambda data: data["neighbor_bond_tokens"].__setitem__(1, ["", "="]),
                "bond tokens must be symmetric",
            ),
            (
                "stereo_missing_metadata",
                stereo_prepared.to_dict(),
                lambda data: data.__setitem__("atom_chiral_tags", []),
                "requires stereo atom metadata",
            ),
        ]

        for label, source, mutate, expected in cases:
            with self.subTest(case=label):
                broken = deepcopy(source)
                mutate(broken)
                with self.assertRaisesRegex(ValueError, expected):
                    CORE_MODULE.PreparedSmilesGraph(broken)


if __name__ == "__main__":
    unittest.main()
