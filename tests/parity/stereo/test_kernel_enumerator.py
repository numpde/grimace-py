from __future__ import annotations

import unittest

from rdkit import Chem

from smiles_next_token.reference import (
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

    def test_kernel_outputs_canonicalize_on_connected_stereo_case_set(self) -> None:
        cases: list[tuple[str, str, str]] = []
        cases.extend(
            (cid, smiles, "atom")
            for cid, smiles in load_connected_atom_stereo_cases(limit=1, max_smiles_length=18)
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

        total_generated = 0
        for cid, smiles, category in cases:
            prepared = prepare_smiles_graph(
                parse_smiles(smiles),
                self.policy,
                surface_kind=CONNECTED_STEREO_SURFACE,
            )
            kernel_prepared = CORE_MODULE.PreparedSmilesGraph(prepared)
            generated: set[str] = set()
            for root_idx in range(prepared.atom_count):
                generated.update(kernel_prepared.enumerate_rooted_connected_stereo_support(root_idx))

            with self.subTest(cid=cid, smiles=smiles, category=category):
                self.assertTrue(generated)
                total_generated += len(generated)
                self.assertEqual(
                    [],
                    validate_rooted_connected_stereo_smiles_support(
                        prepared,
                        0,
                        None,
                        generated,
                    ),
                )
                canonicalized = set()
                for output_smiles in generated:
                    parsed = Chem.MolFromSmiles(output_smiles)
                    self.assertIsNotNone(parsed, msg=output_smiles)
                    assert parsed is not None
                    canonicalized.add(prepared.identity_smiles_for(parsed))
                self.assertEqual({prepared.identity_smiles}, canonicalized)

        self.assertGreaterEqual(total_generated, 12)


if __name__ == "__main__":
    unittest.main()
