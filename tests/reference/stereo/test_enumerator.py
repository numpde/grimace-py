from __future__ import annotations

import unittest

from rdkit import Chem, rdBase

from smiles_next_token.reference import (
    enumerate_rooted_connected_stereo_smiles_support,
    validate_rooted_connected_stereo_smiles_support,
)
from smiles_next_token.reference.rdkit_random import sample_rdkit_random_smiles_from_root
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
