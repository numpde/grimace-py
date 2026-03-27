from __future__ import annotations

import unittest

from rdkit import Chem

from smiles_next_token.reference import (
    CONNECTED_STEREO_SURFACE,
    DEFAULT_RDKIT_RANDOM_CONNECTED_NONSTEREO_POLICY_PATH,
    PreparedSmilesGraph as PythonPreparedSmilesGraph,
    ReferencePolicy,
    load_default_connected_nonstereo_molecule_cases,
    prepare_smiles_graph,
)

try:
    from smiles_next_token import _core
except ImportError:  # pragma: no cover - exercised only when the extension is absent
    _core = None


class CorePreparedSmilesGraphContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if _core is None:
            raise unittest.SkipTest("private Rust extension is not installed")

        cls.policy = ReferencePolicy.from_path(DEFAULT_RDKIT_RANDOM_CONNECTED_NONSTEREO_POLICY_PATH)

    def test_kernel_prepared_graph_accepts_python_reference_object(self) -> None:
        mol = Chem.MolFromSmiles("Cc1ccccc1")
        self.assertIsNotNone(mol)
        assert mol is not None

        prepared = prepare_smiles_graph(mol, self.policy)
        kernel_prepared = _core.PreparedSmilesGraph(prepared)

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
        mol = Chem.MolFromSmiles("O=[Ti]=O")
        self.assertIsNotNone(mol)
        assert mol is not None

        prepared = prepare_smiles_graph(mol, self.policy)
        kernel_prepared = _core.PreparedSmilesGraph(prepared.to_dict())
        rebuilt = PythonPreparedSmilesGraph.from_dict(kernel_prepared.to_dict())

        self.assertEqual(prepared, rebuilt)

    def test_kernel_prepared_graph_dict_roundtrip_is_lossless_for_stereo_surface(self) -> None:
        mol = Chem.MolFromSmiles("F/C=C\\Cl")
        self.assertIsNotNone(mol)
        assert mol is not None

        prepared = prepare_smiles_graph(mol, self.policy, surface_kind=CONNECTED_STEREO_SURFACE)
        kernel_prepared = _core.PreparedSmilesGraph(prepared.to_dict())
        rebuilt = PythonPreparedSmilesGraph.from_dict(kernel_prepared.to_dict())

        self.assertEqual(prepared, rebuilt)

    def test_kernel_prepared_graph_validates_policy_provenance(self) -> None:
        mol = Chem.MolFromSmiles("CC#N")
        self.assertIsNotNone(mol)
        assert mol is not None

        prepared = prepare_smiles_graph(mol, self.policy)
        kernel_prepared = _core.PreparedSmilesGraph(prepared)

        kernel_prepared.validate_policy(prepared.policy_name, prepared.policy_digest)
        with self.assertRaisesRegex(ValueError, "does not match the provided policy"):
            kernel_prepared.validate_policy(prepared.policy_name, "badc0de")

    def test_kernel_rooted_support_rejects_out_of_range_root(self) -> None:
        mol = Chem.MolFromSmiles("CC")
        self.assertIsNotNone(mol)
        assert mol is not None

        prepared = prepare_smiles_graph(mol, self.policy)
        kernel_prepared = _core.PreparedSmilesGraph(prepared)

        with self.assertRaisesRegex(IndexError, "root_idx out of range"):
            kernel_prepared.enumerate_rooted_connected_nonstereo_support(-1)
        with self.assertRaisesRegex(IndexError, "root_idx out of range"):
            kernel_prepared.enumerate_rooted_connected_nonstereo_support(2)
        with self.assertRaisesRegex(IndexError, "root_idx out of range"):
            _core.RootedConnectedNonStereoWalker(prepared, -1)
        with self.assertRaisesRegex(IndexError, "root_idx out of range"):
            _core.RootedConnectedNonStereoWalker(prepared, 2)

    def test_kernel_prepared_graph_roundtrips_dataset_slice(self) -> None:
        cases = load_default_connected_nonstereo_molecule_cases(limit=25, max_smiles_length=20)
        self.assertEqual(25, len(cases))

        for case in cases:
            mol = Chem.MolFromSmiles(case.smiles)
            self.assertIsNotNone(mol)
            assert mol is not None

            with self.subTest(cid=case.cid, smiles=case.smiles):
                prepared = prepare_smiles_graph(mol, self.policy)
                kernel_prepared = _core.PreparedSmilesGraph(prepared)
                self.assertEqual(prepared.to_dict(), kernel_prepared.to_dict())


if __name__ == "__main__":
    unittest.main()
