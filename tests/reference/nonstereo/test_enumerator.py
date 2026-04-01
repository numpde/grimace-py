from __future__ import annotations

from copy import deepcopy
import unittest

from rdkit import Chem

from grimace._reference import (
    PreparedSmilesGraph,
    ReferencePolicy,
    enumerate_rooted_nonstereo_smiles_support,
    enumerate_rooted_smiles_support,
    load_default_connected_nonstereo_molecule_cases,
    prepare_smiles_graph,
    sample_rdkit_random_smiles_from_root,
    validate_rooted_nonstereo_smiles_support,
    validate_rooted_smiles_support,
)
from tests.helpers.cases import NONSTEREO_AWKWARD_CASES
from tests.helpers.mols import parse_smiles
from tests.helpers.policies import (
    load_connected_nonstereo_policy,
    load_default_policy,
    with_sampling_override,
)


class RootedEnumeratorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_default_policy()
        cls.connected_nonstereo_policy = load_connected_nonstereo_policy()
        cls.high_budget_connected_nonstereo_policy = with_sampling_override(
            cls.connected_nonstereo_policy,
            draw_budget=2000,
        )
        policy_data = deepcopy(cls.connected_nonstereo_policy.data)
        policy_data["sampling"] = dict(policy_data["sampling"], isomericSmiles=False)
        policy_data["identity_check"] = dict(policy_data["identity_check"], isomericSmiles=False)
        cls.nonstereo_output_policy = ReferencePolicy(data=policy_data)

    def assert_rooted_support_matches_sampled_rdkit(
        self,
        smiles: str,
        root_idx: int,
        *,
        expected_draw_budget_floor: int = 1,
        policy=None,
        enumerator=enumerate_rooted_smiles_support,
        validator=validate_rooted_smiles_support,
    ) -> None:
        mol = parse_smiles(smiles)

        policy = self.policy if policy is None else policy
        prepared = prepare_smiles_graph(mol, policy)
        exact = enumerator(prepared, root_idx)
        sampled = set(sample_rdkit_random_smiles_from_root(mol, policy, root_idx))

        self.assertGreaterEqual(len(sampled), expected_draw_budget_floor)
        self.assertEqual(exact, sampled)
        self.assertEqual([], validator(prepared, root_idx, None, exact))

    def test_toluene_methyl_root(self) -> None:
        self.assert_rooted_support_matches_sampled_rdkit("Cc1ccccc1", 0)

    def test_toluene_ring_root(self) -> None:
        self.assert_rooted_support_matches_sampled_rdkit("Cc1ccccc1", 1)

    def test_cyclohexene_root(self) -> None:
        self.assert_rooted_support_matches_sampled_rdkit("C1CCCC=C1", 0)

    def test_pyridine_root(self) -> None:
        self.assert_rooted_support_matches_sampled_rdkit("c1ccncc1", 0)

    def test_connected_nonstereo_branch_matches_sampled_rdkit(self) -> None:
        self.assert_rooted_support_matches_sampled_rdkit(
            "C1CCCC=C1",
            0,
            policy=self.connected_nonstereo_policy,
            enumerator=enumerate_rooted_nonstereo_smiles_support,
            validator=validate_rooted_nonstereo_smiles_support,
        )

    def test_curated_rooted_exact_supports_are_stable(self) -> None:
        cases = [
            ("Cc1ccccc1", 0, {"Cc1ccccc1"}),
            ("Cc1ccccc1", 1, {"c1(C)ccccc1", "c1(ccccc1)C"}),
            ("C1CCCC=C1", 0, {"C1C=CCCC1", "C1CCCC=C1"}),
            ("c1ccncc1", 0, {"c1ccncc1"}),
            ("O=[Ti]=O", 0, {"[O]=[Ti]=[O]"}),
            ("O=[Ti]=O", 1, {"[Ti](=[O])=[O]"}),
            ("C[Ge]", 0, {"[CH3][Ge]"}),
            ("C[Ge]", 1, {"[Ge][CH3]"}),
        ]

        for smiles, root_idx, expected in cases:
            with self.subTest(smiles=smiles, root_idx=root_idx):
                prepared = prepare_smiles_graph(parse_smiles(smiles), self.connected_nonstereo_policy)
                self.assertEqual(expected, enumerate_rooted_nonstereo_smiles_support(prepared, root_idx))

    def test_connected_nonstereo_branch_matches_sampled_rdkit_for_all_roots_on_dataset_slice(self) -> None:
        cases = load_default_connected_nonstereo_molecule_cases(limit=120, max_smiles_length=14)
        self.assertEqual(120, len(cases))

        for case in cases:
            mol = parse_smiles(case.smiles)
            prepared = prepare_smiles_graph(mol, self.high_budget_connected_nonstereo_policy)
            for root_idx in range(mol.GetNumAtoms()):
                with self.subTest(cid=case.cid, smiles=case.smiles, root_idx=root_idx):
                    exact = enumerate_rooted_nonstereo_smiles_support(prepared, root_idx)
                    sampled = set(
                        sample_rdkit_random_smiles_from_root(
                            mol,
                            self.high_budget_connected_nonstereo_policy,
                            root_idx,
                        )
                    )
                    self.assertEqual(exact, sampled)
                    self.assertEqual(
                        [],
                        validate_rooted_nonstereo_smiles_support(prepared, root_idx, None, exact),
                    )

    def test_connected_nonstereo_branch_roundtrips_for_all_roots_on_larger_dataset_slice(self) -> None:
        cases = load_default_connected_nonstereo_molecule_cases(limit=120, max_smiles_length=20)
        self.assertEqual(120, len(cases))

        for case in cases:
            mol = parse_smiles(case.smiles)
            prepared = prepare_smiles_graph(mol, self.connected_nonstereo_policy)
            rebuilt = PreparedSmilesGraph.from_dict(prepared.to_dict())
            for root_idx in range(mol.GetNumAtoms()):
                with self.subTest(cid=case.cid, smiles=case.smiles, root_idx=root_idx):
                    support = enumerate_rooted_nonstereo_smiles_support(rebuilt, root_idx)
                    self.assertTrue(support)
                    self.assertEqual(
                        [],
                        validate_rooted_nonstereo_smiles_support(rebuilt, root_idx, None, support),
                    )

    def test_connected_nonstereo_branch_matches_sampled_rdkit_on_curated_awkward_cases(self) -> None:
        for smiles in NONSTEREO_AWKWARD_CASES:
            mol = parse_smiles(smiles)
            prepared = prepare_smiles_graph(mol, self.high_budget_connected_nonstereo_policy)
            for root_idx in range(mol.GetNumAtoms()):
                with self.subTest(smiles=smiles, root_idx=root_idx):
                    exact = enumerate_rooted_nonstereo_smiles_support(prepared, root_idx)
                    sampled = set(
                        sample_rdkit_random_smiles_from_root(
                            mol,
                            self.high_budget_connected_nonstereo_policy,
                            root_idx,
                        )
                    )
                    self.assertEqual(exact, sampled)
                    self.assertEqual(
                        [],
                        validate_rooted_nonstereo_smiles_support(prepared, root_idx, None, exact),
                    )

    def test_stereochemistry_is_dropped_for_general_nonstereo_surface(self) -> None:
        mol = parse_smiles("F[C@H](Cl)Br")
        stripped = Chem.Mol(mol)
        Chem.RemoveStereochemistry(stripped)

        self.assertEqual(
            enumerate_rooted_smiles_support(stripped, 0, self.nonstereo_output_policy),
            enumerate_rooted_smiles_support(mol, 0, self.nonstereo_output_policy),
        )

    def test_connected_nonstereo_branch_drops_bond_stereo_input(self) -> None:
        mol = parse_smiles("F/C=C\\Cl")
        stripped = Chem.Mol(mol)
        Chem.RemoveStereochemistry(stripped)

        self.assertEqual(
            enumerate_rooted_nonstereo_smiles_support(stripped, 1, self.nonstereo_output_policy),
            enumerate_rooted_nonstereo_smiles_support(mol, 1, self.nonstereo_output_policy),
        )

    def test_connected_nonstereo_branch_rejects_unsupported_bond_types(self) -> None:
        mol = parse_smiles("[NH3][Cu]")
        with self.assertRaisesRegex(ValueError, "Unsupported bond type"):
            enumerate_rooted_nonstereo_smiles_support(mol, 0, self.connected_nonstereo_policy)


if __name__ == "__main__":
    unittest.main()
