from __future__ import annotations

import unittest

from rdkit import Chem, rdBase

from smiles_next_token.reference import (
    DEFAULT_RDKIT_RANDOM_CONNECTED_NONSTEREO_POLICY_PATH,
    ReferencePolicy,
    enumerate_rooted_connected_stereo_smiles_support,
    load_default_molecule_cases,
    molecule_is_connected,
    validate_rooted_connected_stereo_smiles_support,
)
from smiles_next_token.reference.rdkit_random import sample_rdkit_random_smiles_from_root


def _load_connected_atom_stereo_cases(*, limit: int, max_smiles_length: int) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    for case in load_default_molecule_cases(limit=5000, max_smiles_length=max_smiles_length):
        mol = Chem.MolFromSmiles(case.smiles)
        if mol is None or not molecule_is_connected(mol):
            continue
        if any(str(bond.GetStereo()) != "STEREONONE" or str(bond.GetBondDir()) != "NONE" for bond in mol.GetBonds()):
            continue
        if not any(str(atom.GetChiralTag()) != "CHI_UNSPECIFIED" for atom in mol.GetAtoms()):
            continue
        selected.append((case.cid, case.smiles))
        if len(selected) >= limit:
            break
    return selected


def _load_connected_multi_atom_stereo_cases(*, limit: int, max_smiles_length: int) -> list[tuple[str, str, int]]:
    selected: list[tuple[str, str, int]] = []
    for case in load_default_molecule_cases(limit=50000, max_smiles_length=max_smiles_length):
        mol = Chem.MolFromSmiles(case.smiles)
        if mol is None or not molecule_is_connected(mol):
            continue
        if any(str(bond.GetStereo()) != "STEREONONE" or str(bond.GetBondDir()) != "NONE" for bond in mol.GetBonds()):
            continue
        chiral_count = sum(1 for atom in mol.GetAtoms() if str(atom.GetChiralTag()) != "CHI_UNSPECIFIED")
        if chiral_count < 3:
            continue
        selected.append((case.cid, case.smiles, chiral_count))
        if len(selected) >= limit:
            break
    return selected


def _load_connected_bond_stereo_cases(*, limit: int, max_smiles_length: int) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    for case in load_default_molecule_cases(limit=5000, max_smiles_length=max_smiles_length):
        mol = Chem.MolFromSmiles(case.smiles)
        if mol is None or not molecule_is_connected(mol):
            continue
        if any(atom.GetIsAromatic() for atom in mol.GetAtoms()):
            continue
        if any(bond.IsInRing() for bond in mol.GetBonds()):
            continue
        if any(str(atom.GetChiralTag()) != "CHI_UNSPECIFIED" for atom in mol.GetAtoms()):
            continue
        stereo_bonds = [bond for bond in mol.GetBonds() if str(bond.GetStereo()) != "STEREONONE"]
        if len(stereo_bonds) != 1:
            continue
        stereo_bond = stereo_bonds[0]
        if stereo_bond.GetBeginAtom().GetAtomicNum() != 6 or stereo_bond.GetEndAtom().GetAtomicNum() != 6:
            continue
        begin_idx = stereo_bond.GetBeginAtomIdx()
        end_idx = stereo_bond.GetEndAtomIdx()
        begin_single_substituents = sum(
            1
            for neighbor in mol.GetAtomWithIdx(begin_idx).GetNeighbors()
            if neighbor.GetIdx() != end_idx and mol.GetBondBetweenAtoms(begin_idx, neighbor.GetIdx()).GetBondType() == Chem.BondType.SINGLE
        )
        end_single_substituents = sum(
            1
            for neighbor in mol.GetAtomWithIdx(end_idx).GetNeighbors()
            if neighbor.GetIdx() != begin_idx and mol.GetBondBetweenAtoms(end_idx, neighbor.GetIdx()).GetBondType() == Chem.BondType.SINGLE
        )
        if begin_single_substituents > 1 or end_single_substituents > 1:
            continue
        selected.append((case.cid, case.smiles))
        if len(selected) >= limit:
            break
    return selected


def _sample_rooted_rdkit_support(
    mol: Chem.Mol,
    policy: ReferencePolicy,
    root_idx: int,
    *,
    draws: int,
) -> set[str]:
    sampling = policy.data["sampling"]
    rdBase.SeedRandomNumberGenerator(int(sampling["seed"]))
    observed = {
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
    return observed


class RootedConnectedStereoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = ReferencePolicy.from_path(DEFAULT_RDKIT_RANDOM_CONNECTED_NONSTEREO_POLICY_PATH)

    def test_curated_atom_stereo_cases_match_rooted_rdkit_samples(self) -> None:
        cases = [
            "F[C@H](Cl)Br",
            "F[C@](Cl)(Br)I",
            "C[C@H](O)[C@@H](F)Cl",
        ]

        for smiles in cases:
            mol = Chem.MolFromSmiles(smiles)
            self.assertIsNotNone(mol)
            assert mol is not None

            for root_idx in range(mol.GetNumAtoms()):
                with self.subTest(smiles=smiles, root_idx=root_idx):
                    observed = enumerate_rooted_connected_stereo_smiles_support(
                        mol,
                        root_idx,
                        self.policy,
                    )
                    sampled = set(sample_rdkit_random_smiles_from_root(mol, self.policy, root_idx))

                    self.assertEqual(sampled, observed)
                    self.assertEqual(
                        [],
                        validate_rooted_connected_stereo_smiles_support(
                            mol,
                            root_idx,
                            self.policy,
                            observed,
                        ),
                    )

    def test_dataset_atom_stereo_slice_matches_rooted_rdkit_samples(self) -> None:
        cases = _load_connected_atom_stereo_cases(limit=12, max_smiles_length=16)
        self.assertEqual(12, len(cases))

        for cid, smiles in cases:
            mol = Chem.MolFromSmiles(smiles)
            self.assertIsNotNone(mol)
            assert mol is not None

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
        cases = _load_connected_multi_atom_stereo_cases(limit=5, max_smiles_length=36)
        self.assertEqual(5, len(cases))

        for cid, smiles, chiral_count in cases:
            mol = Chem.MolFromSmiles(smiles)
            self.assertIsNotNone(mol)
            assert mol is not None

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
                    self.assertEqual(
                        [],
                        validate_rooted_connected_stereo_smiles_support(
                            mol,
                            root_idx,
                            self.policy,
                            observed,
                        ),
                    )

    def test_curated_bond_stereo_cases_match_rooted_rdkit_samples(self) -> None:
        cases = [
            "F/C=C\\Cl",
            "F/C=C/C",
            "C(/C=C/Cl)Cl",
            "C(=C/Cl)\\Cl",
            "CC/C=C\\CCO",
            "C/C=C/C=O",
        ]

        for smiles in cases:
            mol = Chem.MolFromSmiles(smiles)
            self.assertIsNotNone(mol)
            assert mol is not None

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
                    self.assertEqual(
                        [],
                        validate_rooted_connected_stereo_smiles_support(
                            mol,
                            root_idx,
                            self.policy,
                            observed,
                        ),
                    )

    def test_dataset_bond_stereo_slice_contains_high_draw_rdkit_support(self) -> None:
        cases = _load_connected_bond_stereo_cases(limit=8, max_smiles_length=20)
        self.assertEqual(8, len(cases))

        for cid, smiles in cases:
            mol = Chem.MolFromSmiles(smiles)
            self.assertIsNotNone(mol)
            assert mol is not None

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
                    self.assertEqual(
                        [],
                        validate_rooted_connected_stereo_smiles_support(
                            mol,
                            root_idx,
                            self.policy,
                            observed,
                        ),
                    )

    def test_conjugated_bond_stereo_case_matches_rooted_rdkit_samples(self) -> None:
        smiles = "C/C=C/C=C/C(=O)O"
        mol = Chem.MolFromSmiles(smiles)
        self.assertIsNotNone(mol)
        assert mol is not None

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
                self.assertEqual(
                    [],
                    validate_rooted_connected_stereo_smiles_support(
                        mol,
                        root_idx,
                        self.policy,
                        observed,
                    ),
                )

    def test_branched_and_hetero_bond_stereo_cases_match_rooted_rdkit_samples(self) -> None:
        cases = [
            "C/C=C(\\C)/C(=O)O",
            "C/C(=N\\\\OC(=O)NC)/SC",
        ]

        for smiles in cases:
            mol = Chem.MolFromSmiles(smiles)
            self.assertIsNotNone(mol)
            assert mol is not None

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
        smiles = "F/C(Cl)=C/F"
        mol = Chem.MolFromSmiles(smiles)
        self.assertIsNotNone(mol)
        assert mol is not None

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


if __name__ == "__main__":
    unittest.main()
