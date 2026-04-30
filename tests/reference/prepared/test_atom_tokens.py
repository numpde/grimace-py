from __future__ import annotations

import unittest

from rdkit import Chem

from grimace._reference import (
    PreparedSmilesGraph,
    load_default_connected_nonstereo_molecule_cases,
    prepare_smiles_graph,
)
from grimace._reference.rooted.connected_nonstereo import (
    build_atom_tokens,
    enumerate_rooted_connected_nonstereo_smiles_support,
)
from tests.helpers.mols import parse_smiles
from tests.helpers.policies import load_connected_nonstereo_policy, with_sampling_override


def rdkit_fragment_tokens(mol: Chem.Mol, policy) -> tuple[str, ...]:
    sampling = policy.data["sampling"]
    return tuple(
        Chem.MolFragmentToSmiles(
            mol,
            atomsToUse=[atom_idx],
            bondsToUse=[],
            canonical=False,
            isomericSmiles=bool(sampling["isomericSmiles"]),
            kekuleSmiles=bool(sampling["kekuleSmiles"]),
            allHsExplicit=bool(sampling["allHsExplicit"]),
            allBondsExplicit=bool(sampling["allBondsExplicit"]),
        )
        for atom_idx in range(mol.GetNumAtoms())
    )


class RootedAtomTokenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_connected_nonstereo_policy()

    def test_local_atom_tokens_match_rdkit_on_curated_cases(self) -> None:
        for smiles in [
            "C",
            "[CH3]",
            "[NH3+]",
            "c1ccncc1",
            "[13CH3]C",
            "[CH3:7]C",
            "[O-]",
            "[nH]1cccc1",
            "CC#N",
            "[CH2]C",
            "[Cu+2]",
            "[O]",
            "[Na+]",
            "O=[Ti]=O",
            "O=[Cr](=O)=O",
            "Cl[Fe]Cl",
            "F[Mg]",
            "C[Ge]",
            "C[Al]",
            "O=[Se]=O",
            "C[Si]",
            "O=[P]=O",
        ]:
            with self.subTest(smiles=smiles):
                mol = parse_smiles(smiles)
                prepared = prepare_smiles_graph(mol, self.policy)
                self.assertEqual(rdkit_fragment_tokens(mol, self.policy), prepared.atom_tokens)
                self.assertEqual(prepared.atom_tokens, build_atom_tokens(prepared))

    def test_prepared_graph_supports_dative_bond_tokens(self) -> None:
        prepared = prepare_smiles_graph(parse_smiles("[NH3][Cu]"), self.policy)
        self.assertEqual(("[NH3]", "[Cu]"), prepared.atom_tokens)
        self.assertEqual((("->",), ("<-",)), prepared.neighbor_bond_tokens)

    def test_dummy_neighbors_do_not_force_organic_atom_brackets(self) -> None:
        for smiles, expected_tokens in [
            ("*C", ("*", "C")),
            ("[1*]C", ("[1*]", "C")),
            ("[2*]=O", ("[2*]", "O")),
        ]:
            with self.subTest(smiles=smiles):
                mol = parse_smiles(smiles)
                prepared = prepare_smiles_graph(mol, self.policy)
                self.assertEqual(rdkit_fragment_tokens(mol, self.policy), prepared.atom_tokens)
                self.assertEqual(expected_tokens, prepared.atom_tokens)

    def test_local_atom_tokens_match_rdkit_on_dataset_slice(self) -> None:
        cases = load_default_connected_nonstereo_molecule_cases(limit=120, max_smiles_length=20)
        self.assertEqual(120, len(cases))

        for case in cases:
            mol = parse_smiles(case.smiles)
            with self.subTest(cid=case.cid, smiles=case.smiles):
                prepared = prepare_smiles_graph(mol, self.policy)
                self.assertEqual(rdkit_fragment_tokens(mol, self.policy), prepared.atom_tokens)
                self.assertEqual(prepared.atom_tokens, build_atom_tokens(prepared))

    def test_forced_brackets_include_implicit_hydrogens(self) -> None:
        methane = parse_smiles("C")
        methane.GetAtomWithIdx(0).SetIsotope(13)
        prepared = prepare_smiles_graph(methane, self.policy)
        self.assertEqual(("[13CH4]",), prepared.atom_tokens)

        mapped_methane = parse_smiles("C")
        mapped_methane.GetAtomWithIdx(0).SetAtomMapNum(7)
        prepared = prepare_smiles_graph(mapped_methane, self.policy)
        self.assertEqual(("[CH4:7]",), prepared.atom_tokens)

        benzene = parse_smiles("c1ccccc1")
        benzene.GetAtomWithIdx(0).SetAtomMapNum(7)
        prepared = prepare_smiles_graph(benzene, self.policy)
        self.assertEqual("[cH:7]", prepared.atom_tokens[0])

    def test_isotopes_are_suppressed_without_isomeric_smiles(self) -> None:
        nonisomeric_policy = with_sampling_override(self.policy, isomericSmiles=False)

        methane = parse_smiles("C")
        methane.GetAtomWithIdx(0).SetIsotope(13)
        prepared = prepare_smiles_graph(methane, nonisomeric_policy)
        self.assertEqual(("C",), prepared.atom_tokens)

        neopentane = parse_smiles("C[13C](C)(C)C")
        prepared = prepare_smiles_graph(neopentane, nonisomeric_policy)
        self.assertEqual(("C", "C", "C", "C", "C"), prepared.atom_tokens)

    def test_ignore_atom_map_numbers_does_not_change_emitted_atom_maps(self) -> None:
        ignore_maps_policy = with_sampling_override(self.policy, ignoreAtomMapNumbers=True)
        prepared = prepare_smiles_graph(parse_smiles("[CH3:7]C"), ignore_maps_policy)
        self.assertEqual(("C", "C"), prepared.atom_tokens)

    def test_all_hs_explicit_is_emitted_locally(self) -> None:
        explicit_h_policy = with_sampling_override(self.policy, allHsExplicit=True)

        prepared = prepare_smiles_graph(parse_smiles("C"), explicit_h_policy)
        self.assertEqual(("[CH4]",), prepared.atom_tokens)

        prepared = prepare_smiles_graph(parse_smiles("c1ccncc1"), explicit_h_policy)
        self.assertEqual(
            ("[cH]", "[cH]", "[cH]", "[n]", "[cH]", "[cH]"),
            prepared.atom_tokens,
        )

    def test_all_bonds_explicit_is_emitted_locally(self) -> None:
        explicit_bonds_policy = with_sampling_override(self.policy, allBondsExplicit=True)
        prepared = prepare_smiles_graph(parse_smiles("CC#N"), explicit_bonds_policy)
        self.assertEqual(
            {"C-C#N"},
            enumerate_rooted_connected_nonstereo_smiles_support(prepared, 0),
        )

    def test_prepared_graph_dict_roundtrip_preserves_atom_tokens_on_dataset_slice(self) -> None:
        cases = load_default_connected_nonstereo_molecule_cases(limit=40, max_smiles_length=20)
        self.assertEqual(40, len(cases))

        for case in cases:
            with self.subTest(cid=case.cid, smiles=case.smiles):
                prepared = prepare_smiles_graph(parse_smiles(case.smiles), self.policy)
                rebuilt = PreparedSmilesGraph.from_dict(prepared.to_dict())
                self.assertEqual(prepared.atom_tokens, rebuilt.atom_tokens)
                self.assertEqual(prepared.neighbors, rebuilt.neighbors)
                self.assertEqual(prepared.neighbor_bond_tokens, rebuilt.neighbor_bond_tokens)


if __name__ == "__main__":
    unittest.main()
