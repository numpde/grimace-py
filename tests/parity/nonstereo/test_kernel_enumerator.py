from __future__ import annotations

import unittest

from smiles_next_token.reference import (
    enumerate_rooted_nonstereo_smiles_support,
    load_default_connected_nonstereo_molecule_cases,
    prepare_smiles_graph,
    validate_rooted_nonstereo_smiles_support,
)
from tests.helpers.cases import NONSTEREO_AWKWARD_CASES, NONSTEREO_CURATED_ROOT_CASES
from tests.helpers.kernel import CORE_MODULE
from tests.helpers.mols import parse_smiles
from tests.helpers.policies import load_connected_nonstereo_policy


class CoreRootedEnumeratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if CORE_MODULE is None:
            raise unittest.SkipTest("private Rust extension is not installed")
        cls.policy = load_connected_nonstereo_policy()

    def test_kernel_matches_python_reference_on_curated_cases(self) -> None:
        for smiles, root_idx in NONSTEREO_CURATED_ROOT_CASES:
            with self.subTest(smiles=smiles, root_idx=root_idx):
                prepared = prepare_smiles_graph(parse_smiles(smiles), self.policy)
                kernel_prepared = CORE_MODULE.PreparedSmilesGraph(prepared)

                python_support = enumerate_rooted_nonstereo_smiles_support(prepared, root_idx)
                kernel_support = set(
                    kernel_prepared.enumerate_rooted_connected_nonstereo_support(root_idx)
                )

                self.assertEqual(python_support, kernel_support)
                self.assertEqual(
                    [],
                    validate_rooted_nonstereo_smiles_support(prepared, root_idx, None, kernel_support),
                )

    def test_kernel_matches_python_reference_for_all_roots_on_dataset_slice(self) -> None:
        cases = load_default_connected_nonstereo_molecule_cases(limit=30, max_smiles_length=14)
        self.assertEqual(30, len(cases))

        for case in cases:
            prepared = prepare_smiles_graph(parse_smiles(case.smiles), self.policy)
            kernel_prepared = CORE_MODULE.PreparedSmilesGraph(prepared)
            for root_idx in range(prepared.atom_count):
                with self.subTest(cid=case.cid, smiles=case.smiles, root_idx=root_idx):
                    python_support = enumerate_rooted_nonstereo_smiles_support(prepared, root_idx)
                    kernel_support = set(
                        kernel_prepared.enumerate_rooted_connected_nonstereo_support(root_idx)
                    )

                    self.assertEqual(python_support, kernel_support)
                    self.assertEqual(
                        [],
                        validate_rooted_nonstereo_smiles_support(prepared, root_idx, None, kernel_support),
                    )

    def test_kernel_matches_python_reference_on_curated_awkward_cases(self) -> None:
        for smiles in NONSTEREO_AWKWARD_CASES:
            prepared = prepare_smiles_graph(parse_smiles(smiles), self.policy)
            kernel_prepared = CORE_MODULE.PreparedSmilesGraph(prepared)
            for root_idx in range(prepared.atom_count):
                with self.subTest(smiles=smiles, root_idx=root_idx):
                    python_support = enumerate_rooted_nonstereo_smiles_support(prepared, root_idx)
                    kernel_support = set(
                        kernel_prepared.enumerate_rooted_connected_nonstereo_support(root_idx)
                    )

                    self.assertEqual(python_support, kernel_support)
                    self.assertEqual(
                        [],
                        validate_rooted_nonstereo_smiles_support(prepared, root_idx, None, kernel_support),
                    )


if __name__ == "__main__":
    unittest.main()
