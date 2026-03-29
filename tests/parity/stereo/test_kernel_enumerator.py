from __future__ import annotations

import unittest

from grimace._reference import (
    CONNECTED_STEREO_SURFACE,
    enumerate_rooted_connected_stereo_smiles_support,
    prepare_smiles_graph,
    validate_rooted_connected_stereo_smiles_support,
)
from tests.helpers.cases import (
    STEREO_CURATED_CASES,
    load_connected_atom_stereo_cases,
    load_connected_bond_stereo_cases,
    load_connected_multi_atom_stereo_cases,
)
from tests.helpers.kernel import CORE_MODULE
from tests.helpers.mols import parse_smiles
from tests.helpers.policies import load_connected_nonstereo_policy


class CoreRootedConnectedStereoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if CORE_MODULE is None:
            raise unittest.SkipTest("private Rust extension is not installed")
        cls.policy = load_connected_nonstereo_policy()

    def test_kernel_matches_python_reference_on_curated_stereo_cases(self) -> None:
        for smiles in STEREO_CURATED_CASES:
            prepared = prepare_smiles_graph(
                parse_smiles(smiles),
                self.policy,
                surface_kind=CONNECTED_STEREO_SURFACE,
            )
            kernel_prepared = CORE_MODULE.PreparedSmilesGraph(prepared)

            for root_idx in range(prepared.atom_count):
                with self.subTest(smiles=smiles, root_idx=root_idx):
                    python_support = enumerate_rooted_connected_stereo_smiles_support(prepared, root_idx)
                    kernel_support = set(
                        kernel_prepared.enumerate_rooted_connected_stereo_support(root_idx)
                    )
                    self.assertEqual(python_support, kernel_support)
                    self.assertEqual(
                        [],
                        validate_rooted_connected_stereo_smiles_support(
                            prepared,
                            root_idx,
                            None,
                            kernel_support,
                        ),
                    )

    def test_kernel_matches_python_reference_on_representative_stereo_slice(self) -> None:
        cases: list[tuple[str, str, str]] = []
        cases.extend(
            (cid, smiles, "atom")
            for cid, smiles in load_connected_atom_stereo_cases(limit=1, max_smiles_length=16)
        )
        cases.extend(
            (cid, smiles, "multi_atom")
            for cid, smiles, _ in load_connected_multi_atom_stereo_cases(limit=1, max_smiles_length=28)
        )
        cases.extend(
            (cid, smiles, "bond")
            for cid, smiles in load_connected_bond_stereo_cases(limit=1, max_smiles_length=18)
        )
        self.assertEqual(3, len(cases))

        for cid, smiles, category in cases:
            prepared = prepare_smiles_graph(
                parse_smiles(smiles),
                self.policy,
                surface_kind=CONNECTED_STEREO_SURFACE,
            )
            kernel_prepared = CORE_MODULE.PreparedSmilesGraph(prepared)
            for root_idx in range(prepared.atom_count):
                with self.subTest(cid=cid, smiles=smiles, category=category, root_idx=root_idx):
                    python_support = enumerate_rooted_connected_stereo_smiles_support(
                        prepared,
                        root_idx,
                    )
                    kernel_support = set(
                        kernel_prepared.enumerate_rooted_connected_stereo_support(root_idx)
                    )
                    self.assertEqual(python_support, kernel_support)

if __name__ == "__main__":
    unittest.main()
