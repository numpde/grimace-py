from __future__ import annotations

import unittest

from rdkit import Chem, rdBase

from grimace._reference import (
    enumerate_rooted_connected_stereo_smiles_support,
    validate_rooted_connected_stereo_smiles_support,
)
from grimace._reference.rdkit_random import sample_rdkit_random_smiles_from_root
from tests.helpers.cases import (
    STEREO_ATOM_CURATED_CASES,
    STEREO_BOND_CURATED_CASES,
    load_connected_atom_stereo_cases,
    load_connected_bond_stereo_cases,
    load_connected_multi_atom_stereo_cases,
)
from tests.helpers.mols import parse_smiles
from tests.helpers.policies import load_connected_nonstereo_policy


def _sample_rooted_rdkit_support(
    mol: Chem.Mol,
    policy,
    root_idx: int,
    *,
    draws: int,
) -> set[str]:
    sampling = policy.data["sampling"]
    rdBase.SeedRandomNumberGenerator(int(sampling["seed"]))
    return {
        Chem.MolToSmiles(
            Chem.Mol(mol),
            rootedAtAtom=root_idx,
            canonical=bool(sampling["canonical"]),
            doRandom=bool(sampling["doRandom"]),
            isomericSmiles=bool(sampling["isomericSmiles"]),
            kekuleSmiles=bool(sampling["kekuleSmiles"]),
            allBondsExplicit=bool(sampling["allBondsExplicit"]),
            allHsExplicit=bool(sampling["allHsExplicit"]),
            ignoreAtomMapNumbers=bool(sampling["ignoreAtomMapNumbers"]),
        )
        for _ in range(draws)
    }


def _assert_support_valid(
    test_case: unittest.TestCase,
    mol: Chem.Mol,
    policy,
    root_idx: int,
    support: set[str],
) -> None:
    test_case.assertEqual(
        [],
        validate_rooted_connected_stereo_smiles_support(
            mol,
            root_idx,
            policy,
            support,
        ),
    )


class RootedConnectedStereoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_connected_nonstereo_policy()

    def test_curated_atom_stereo_cases_match_rooted_rdkit_samples(self) -> None:
        for smiles in STEREO_ATOM_CURATED_CASES:
            mol = parse_smiles(smiles)

            for root_idx in range(mol.GetNumAtoms()):
                with self.subTest(smiles=smiles, root_idx=root_idx):
                    observed = enumerate_rooted_connected_stereo_smiles_support(
                        mol,
                        root_idx,
                        self.policy,
                    )
                    sampled = set(sample_rdkit_random_smiles_from_root(mol, self.policy, root_idx))

                    self.assertEqual(sampled, observed)
                    _assert_support_valid(self, mol, self.policy, root_idx, observed)

    def test_dataset_atom_stereo_slice_matches_rooted_rdkit_samples(self) -> None:
        cases = load_connected_atom_stereo_cases(limit=12, max_smiles_length=16)
        self.assertEqual(12, len(cases))

        for cid, smiles in cases:
            mol = parse_smiles(smiles)

            for root_idx in range(mol.GetNumAtoms()):
                with self.subTest(cid=cid, smiles=smiles, root_idx=root_idx):
                    observed = enumerate_rooted_connected_stereo_smiles_support(
                        mol,
                        root_idx,
                        self.policy,
                    )
                    sampled = set(sample_rdkit_random_smiles_from_root(mol, self.policy, root_idx))
                    self.assertEqual(sampled, observed)

    def test_dataset_three_center_atom_stereo_slice_matches_rdkit_at_higher_draw_budget(self) -> None:
        cases = load_connected_multi_atom_stereo_cases(limit=5, max_smiles_length=36)
        self.assertEqual(5, len(cases))

        for cid, smiles, chiral_count in cases:
            mol = parse_smiles(smiles)

            for root_idx in range(mol.GetNumAtoms()):
                with self.subTest(cid=cid, smiles=smiles, chiral_count=chiral_count, root_idx=root_idx):
                    observed = enumerate_rooted_connected_stereo_smiles_support(
                        mol,
                        root_idx,
                        self.policy,
                    )
                    sampled = _sample_rooted_rdkit_support(
                        mol,
                        self.policy,
                        root_idx,
                        draws=5000,
                    )
                    self.assertEqual(sampled, observed)
                    _assert_support_valid(self, mol, self.policy, root_idx, observed)

    def test_curated_bond_stereo_cases_match_rooted_rdkit_samples(self) -> None:
        for smiles in STEREO_BOND_CURATED_CASES:
            mol = parse_smiles(smiles)

            for root_idx in range(mol.GetNumAtoms()):
                with self.subTest(smiles=smiles, root_idx=root_idx):
                    observed = enumerate_rooted_connected_stereo_smiles_support(
                        mol,
                        root_idx,
                        self.policy,
                    )
                    sampled = _sample_rooted_rdkit_support(
                        mol,
                        self.policy,
                        root_idx,
                        draws=5000,
                    )
                    self.assertEqual(sampled, observed)
                    _assert_support_valid(self, mol, self.policy, root_idx, observed)

    def test_dataset_bond_stereo_slice_contains_high_draw_rdkit_support(self) -> None:
        cases = load_connected_bond_stereo_cases(limit=8, max_smiles_length=20)
        self.assertEqual(8, len(cases))

        for cid, smiles in cases:
            mol = parse_smiles(smiles)

            for root_idx in range(mol.GetNumAtoms()):
                with self.subTest(cid=cid, smiles=smiles, root_idx=root_idx):
                    observed = enumerate_rooted_connected_stereo_smiles_support(
                        mol,
                        root_idx,
                        self.policy,
                    )
                    sampled = _sample_rooted_rdkit_support(
                        mol,
                        self.policy,
                        root_idx,
                        draws=5000,
                    )
                    self.assertTrue(sampled.issubset(observed))
                    _assert_support_valid(self, mol, self.policy, root_idx, observed)

    def test_conjugated_bond_stereo_case_matches_rooted_rdkit_samples(self) -> None:
        mol = parse_smiles("C/C=C/C=C/C(=O)O")

        for root_idx in range(mol.GetNumAtoms()):
            with self.subTest(root_idx=root_idx):
                observed = enumerate_rooted_connected_stereo_smiles_support(
                    mol,
                    root_idx,
                    self.policy,
                )
                sampled = _sample_rooted_rdkit_support(
                    mol,
                    self.policy,
                    root_idx,
                    draws=5000,
                )
                self.assertEqual(sampled, observed)
                _assert_support_valid(self, mol, self.policy, root_idx, observed)

    def test_rooted_polyene_bond_stereo_case_matches_high_draw_rdkit_samples(self) -> None:
        mol = parse_smiles("CC1=C(C(CCC1)(C)C)/C=C/C(=C/C=C/C(=C/C(=O)O)/C)/C")
        observed = enumerate_rooted_connected_stereo_smiles_support(
            mol,
            11,
            self.policy,
        )
        sampled = _sample_rooted_rdkit_support(
            mol,
            self.policy,
            11,
            draws=2000,
        )

        self.assertEqual(120, len(sampled))
        self.assertEqual(sampled, observed)
        _assert_support_valid(self, mol, self.policy, 11, observed)

    def test_rooted_polyene_internal_branch_carrier_case_matches_high_draw_rdkit_samples(self) -> None:
        mol = parse_smiles("CC1=C(C(CCC1)(C)C)/C=C/C(=C/C=C/C(=C/C(=O)O)/C)/C")
        observed = enumerate_rooted_connected_stereo_smiles_support(
            mol,
            12,
            self.policy,
        )
        sampled = _sample_rooted_rdkit_support(
            mol,
            self.policy,
            12,
            draws=5000,
        )

        self.assertEqual(80, len(sampled))
        self.assertEqual(sampled, observed)
        _assert_support_valid(self, mol, self.policy, 12, observed)

    def test_branched_and_hetero_bond_stereo_cases_match_rooted_rdkit_samples(self) -> None:
        cases = (
            "C/C=C(\\C)/C(=O)O",
            "C/C(=N\\\\OC(=O)NC)/SC",
        )

        for smiles in cases:
            mol = parse_smiles(smiles)

            for root_idx in range(mol.GetNumAtoms()):
                with self.subTest(smiles=smiles, root_idx=root_idx):
                    observed = enumerate_rooted_connected_stereo_smiles_support(
                        mol,
                        root_idx,
                        self.policy,
                    )
                    sampled = _sample_rooted_rdkit_support(
                        mol,
                        self.policy,
                        root_idx,
                        draws=5000,
                    )
                    self.assertEqual(sampled, observed)

    def test_dataset_regression_ring_opening_does_not_precommit_stereo_carrier(self) -> None:
        mol = parse_smiles("CC\\1=C(C2=C(/C1=C\\C3=CC=C(C=C3)S(=O)C)C=CC(=C2)F)CC(=O)O")
        expected = "CC1=C(CC(=O)O)c2c(ccc(F)c2)/C1=C\\c1ccc(S(=O)C)cc1"

        observed = enumerate_rooted_connected_stereo_smiles_support(
            mol,
            0,
            self.policy,
        )
        self.assertIn(expected, observed)
        _assert_support_valid(self, mol, self.policy, 0, observed)

    def test_dataset_regression_porphyrin_like_fragment_matches_rooted_rdkit(self) -> None:
        mol = parse_smiles(
            "C1=CC=C2/C/3=N/C4=C5C(=C([N-]4)/N=C/6\\[N-]/C(=N\\C7=C8C(=C([N-]7)"
            "/N=C(/C2=C1)\\[N-]3)C=CC=C8)/C9=CC=CC=C69)C=CC=C5"
        )
        expected = (
            "c1ccc2/c3[n-]/c(c2c1)=N\\c1c2c(c([n-]1)/N=c1\\[n-]/c(c4c1cccc4)"
            "=N\\c1c4c(c([n-]1)/N=3)cccc4)cccc2"
        )

        observed = enumerate_rooted_connected_stereo_smiles_support(
            mol,
            0,
            self.policy,
        )
        self.assertIn(expected, observed)

    def test_dataset_regression_steroid_ring_coupled_component_matches_rooted_rdkit_samples(self) -> None:
        mol = parse_smiles(
            "C[C@H](/C=C/[C@H](C)C(C)C)[C@H]1CC[C@@H]\\\\2"
            "[C@@]1(CCC/C2=C\\\\C=C/3\\\\C[C@H](CCC3=C)O)C"
        )
        observed = enumerate_rooted_connected_stereo_smiles_support(
            mol,
            0,
            self.policy,
        )
        # These two rooted outputs differ only in the coupled begin-side token
        # family. The first was observed in RDKit samples and was missing before
        # the coupled-component fix; the second is the prior wrong-family form.
        expected = (
            "C[C@@H]([C@@H]1[C@@]2([C@@H](CC1)/C(=C/C=C1/C[C@@H](O)CCC1=C)CCC2)C)"
            "/C=C/[C@@H](C(C)C)C"
        )
        rejected = (
            "C[C@@H]([C@@H]1[C@@]2([C@@H](CC1)\\C(=C\\C=C1/C[C@@H](O)CCC1=C)CCC2)C)"
            "/C=C/[C@@H](C(C)C)C"
        )

        self.assertIn(expected, observed)
        self.assertNotIn(rejected, observed)
        _assert_support_valid(self, mol, self.policy, 0, observed)

    def test_dataset_regression_sidechain_steroid_rooted_outputs_are_in_support(self) -> None:
        mol = parse_smiles(
            "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]\\\\2"
            "[C@@]1(CCC/C2=C\\\\C=C/3\\\\C[C@H](CCC3=C)O)C"
        )

        for root_idx in (16, 17, 18, 19):
            with self.subTest(root_idx=root_idx):
                expected = Chem.MolToSmiles(
                    Chem.Mol(mol),
                    rootedAtAtom=root_idx,
                    canonical=False,
                    doRandom=False,
                    isomericSmiles=True,
                )
                observed = enumerate_rooted_connected_stereo_smiles_support(
                    mol,
                    root_idx,
                    self.policy,
                )
                self.assertIn(expected, observed)
                _assert_support_valid(self, mol, self.policy, root_idx, observed)

    def test_dataset_regression_terminal_methyl_root_preserves_isolated_stereo_choice(self) -> None:
        mol = parse_smiles("CC1=C(C(CCC1)(C)C)/C=C/C(=C/C=C/C(=C/C(=O)O)/C)/C")
        expected = "CC(/C=C/C=C(/C=C/C1=C(C)CCCC1(C)C)C)=C\\C(=O)O"

        observed = enumerate_rooted_connected_stereo_smiles_support(
            mol,
            20,
            self.policy,
        )
        self.assertIn(expected, observed)

    def test_degree_three_alkene_carrier_selection_matches_rooted_rdkit_samples(self) -> None:
        mol = parse_smiles("F/C(Cl)=C/F")

        for root_idx in range(mol.GetNumAtoms()):
            with self.subTest(root_idx=root_idx):
                observed = enumerate_rooted_connected_stereo_smiles_support(
                    mol,
                    root_idx,
                    self.policy,
                )
                sampled = _sample_rooted_rdkit_support(
                    mol,
                    self.policy,
                    root_idx,
                    draws=5000,
                )
                self.assertEqual(sampled, observed)


if __name__ == "__main__":
    unittest.main()
