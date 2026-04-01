from __future__ import annotations

from copy import deepcopy
import unittest

from rdkit import Chem, rdBase

from grimace._reference import (
    CONNECTED_NONSTEREO_SURFACE,
    CONNECTED_STEREO_SURFACE,
    DEFAULT_RDKIT_RANDOM_CONNECTED_NONSTEREO_POLICY_PATH,
    PREPARED_SMILES_GRAPH_SCHEMA_VERSION,
    PreparedSmilesGraph,
    ReferencePolicy,
    prepare_smiles_graph,
)


class PreparedSmilesGraphContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = ReferencePolicy.from_path(DEFAULT_RDKIT_RANDOM_CONNECTED_NONSTEREO_POLICY_PATH)

    def test_prepared_graph_tracks_policy_and_writer_metadata(self) -> None:
        mol = Chem.MolFromSmiles("Cc1ccccc1")
        self.assertIsNotNone(mol)
        assert mol is not None

        prepared = prepare_smiles_graph(mol, self.policy)

        self.assertEqual(PREPARED_SMILES_GRAPH_SCHEMA_VERSION, prepared.schema_version)
        self.assertEqual(CONNECTED_NONSTEREO_SURFACE, prepared.surface_kind)
        self.assertEqual(self.policy.policy_name, prepared.policy_name)
        self.assertEqual(self.policy.digest(), prepared.policy_digest)
        self.assertEqual(rdBase.rdkitVersion, prepared.rdkit_version)
        self.assertEqual(7, prepared.atom_count)
        self.assertEqual(7, prepared.bond_count)

        sampling = self.policy.data["sampling"]
        self.assertEqual(bool(sampling["isomericSmiles"]), prepared.writer_do_isomeric_smiles)
        self.assertEqual(bool(sampling["kekuleSmiles"]), prepared.writer_kekule_smiles)
        self.assertEqual(bool(sampling["allBondsExplicit"]), prepared.writer_all_bonds_explicit)
        self.assertEqual(bool(sampling["allHsExplicit"]), prepared.writer_all_hs_explicit)
        self.assertEqual(bool(sampling["ignoreAtomMapNumbers"]), prepared.writer_ignore_atom_map_numbers)

        identity = self.policy.data["identity_check"]
        self.assertEqual(bool(identity["parse_with_rdkit"]), prepared.identity_parse_with_rdkit)
        self.assertEqual(bool(identity["canonical"]), prepared.identity_canonical)
        self.assertEqual(bool(identity["isomericSmiles"]), prepared.identity_do_isomeric_smiles)
        self.assertEqual(bool(identity["kekuleSmiles"]), prepared.identity_kekule_smiles)
        self.assertEqual(int(identity["rootedAtAtom"]), prepared.identity_rooted_at_atom)
        self.assertEqual(bool(identity["allBondsExplicit"]), prepared.identity_all_bonds_explicit)
        self.assertEqual(bool(identity["allHsExplicit"]), prepared.identity_all_hs_explicit)
        self.assertEqual(bool(identity["doRandom"]), prepared.identity_do_random)
        self.assertEqual(
            bool(identity["ignoreAtomMapNumbers"]),
            prepared.identity_ignore_atom_map_numbers,
        )

    def test_prepared_graph_dict_roundtrip_is_lossless(self) -> None:
        mol = Chem.MolFromSmiles("O=[Ti]=O")
        self.assertIsNotNone(mol)
        assert mol is not None

        prepared = prepare_smiles_graph(mol, self.policy)
        rebuilt = PreparedSmilesGraph.from_dict(prepared.to_dict())

        self.assertEqual(prepared, rebuilt)
        self.assertEqual("[O]", rebuilt.atom_tokens[0])
        self.assertEqual("[Ti]", rebuilt.atom_tokens[1])
        self.assertEqual("=", rebuilt.bond_token(0, 1))
        self.assertEqual("=", rebuilt.bond_token(1, 2))

    def test_prepared_graph_carries_identity_roundtrip_policy(self) -> None:
        mol = Chem.MolFromSmiles("Cc1ccccc1")
        self.assertIsNotNone(mol)
        assert mol is not None

        prepared = prepare_smiles_graph(mol, self.policy)
        rebuilt = PreparedSmilesGraph.from_dict(prepared.to_dict())

        self.assertEqual(prepared.identity_smiles, rebuilt.identity_smiles_for(mol))

    def test_prepared_graph_rejects_unsupported_surface_input(self) -> None:
        mol = Chem.MolFromSmiles("[NH3][Cu]")
        self.assertIsNotNone(mol)
        assert mol is not None

        with self.assertRaisesRegex(ValueError, "Unsupported bond type"):
            prepare_smiles_graph(mol, self.policy)

    def test_prepared_graph_rejects_inconsistent_shape(self) -> None:
        mol = Chem.MolFromSmiles("CC")
        self.assertIsNotNone(mol)
        assert mol is not None

        prepared = prepare_smiles_graph(mol, self.policy)
        broken = deepcopy(prepared.to_dict())
        broken["neighbor_bond_tokens"][0] = []

        with self.assertRaisesRegex(ValueError, "neighbor token row length mismatch"):
            PreparedSmilesGraph.from_dict(broken)

    def test_connected_stereo_prepared_graph_carries_atom_stereo_metadata(self) -> None:
        mol = Chem.MolFromSmiles("F[C@H](Cl)Br")
        self.assertIsNotNone(mol)
        assert mol is not None

        prepared = prepare_smiles_graph(mol, self.policy, surface_kind=CONNECTED_STEREO_SURFACE)
        rebuilt = PreparedSmilesGraph.from_dict(prepared.to_dict())

        self.assertEqual(CONNECTED_STEREO_SURFACE, prepared.surface_kind)
        self.assertEqual(prepared, rebuilt)
        self.assertEqual(
            ("CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CCW", "CHI_UNSPECIFIED", "CHI_UNSPECIFIED"),
            prepared.atom_chiral_tags,
        )
        self.assertEqual((1,), prepared.atom_stereo_neighbor_orders[0])
        self.assertEqual((0, 2, 3), prepared.atom_stereo_neighbor_orders[1])
        self.assertEqual((0, 1, 0, 0), prepared.atom_explicit_h_counts)
        self.assertEqual((0, 0, 0, 0), prepared.atom_implicit_h_counts)

    def test_connected_nonstereo_prepared_graph_drops_atom_stereo_metadata(self) -> None:
        mol = Chem.MolFromSmiles("F[C@H](Cl)Br")
        self.assertIsNotNone(mol)
        assert mol is not None

        prepared = prepare_smiles_graph(mol, self.policy, surface_kind=CONNECTED_NONSTEREO_SURFACE)

        self.assertEqual(CONNECTED_NONSTEREO_SURFACE, prepared.surface_kind)
        self.assertEqual((), prepared.atom_chiral_tags)
        self.assertEqual((), prepared.atom_stereo_neighbor_orders)

    def test_connected_stereo_prepared_graph_carries_bond_stereo_metadata(self) -> None:
        mol = Chem.MolFromSmiles("F/C=C\\Cl")
        self.assertIsNotNone(mol)
        assert mol is not None

        prepared = prepare_smiles_graph(mol, self.policy, surface_kind=CONNECTED_STEREO_SURFACE)
        rebuilt = PreparedSmilesGraph.from_dict(prepared.to_dict())

        self.assertEqual(prepared, rebuilt)
        self.assertEqual(("STEREONONE", "STEREOZ", "STEREONONE"), prepared.bond_stereo_kinds)
        self.assertEqual(((-1, -1), (0, 3), (-1, -1)), prepared.bond_stereo_atoms)
        self.assertEqual(("ENDUPRIGHT", "NONE", "ENDDOWNRIGHT"), prepared.bond_dirs)
        self.assertEqual((0, 1, 2), prepared.bond_begin_atom_indices)
        self.assertEqual((1, 2, 3), prepared.bond_end_atom_indices)
        self.assertEqual("/", prepared.directed_bond_token(0, 1))
        self.assertEqual("\\", prepared.directed_bond_token(1, 0))
        self.assertEqual("\\", prepared.directed_bond_token(2, 3))
        self.assertEqual("/", prepared.directed_bond_token(3, 2))

    def test_connected_nonstereo_prepared_graph_drops_bond_stereo_metadata(self) -> None:
        mol = Chem.MolFromSmiles("F/C=C\\Cl")
        self.assertIsNotNone(mol)
        assert mol is not None

        prepared = prepare_smiles_graph(mol, self.policy, surface_kind=CONNECTED_NONSTEREO_SURFACE)

        self.assertEqual((), prepared.bond_stereo_kinds)
        self.assertEqual((), prepared.bond_stereo_atoms)
        self.assertEqual((), prepared.bond_dirs)

    def test_connected_stereo_surface_still_rejects_unsupported_stereo_families(self) -> None:
        mol = Chem.MolFromSmiles("C/C=C/C")
        self.assertIsNotNone(mol)
        assert mol is not None

        prepared = prepare_smiles_graph(mol, self.policy, surface_kind=CONNECTED_STEREO_SURFACE)
        self.assertEqual(CONNECTED_STEREO_SURFACE, prepared.surface_kind)

        unsupported = Chem.MolFromSmiles("CC")
        self.assertIsNotNone(unsupported)
        assert unsupported is not None
        unsupported_bond = unsupported.GetBondWithIdx(0)
        unsupported_bond.SetStereo(Chem.BondStereo.STEREOATROPCW)

        with self.assertRaisesRegex(ValueError, "Unsupported bond stereo"):
            prepare_smiles_graph(unsupported, self.policy, surface_kind=CONNECTED_STEREO_SURFACE)


if __name__ == "__main__":
    unittest.main()
