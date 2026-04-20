from __future__ import annotations

import unittest

from rdkit import Chem, rdBase

from tests.helpers.mols import parse_smiles
from tests.helpers.rdkit_writer_cases import ROOTED_RANDOM_CASES
from tests.rdkit_serialization._support import (
    assert_rooted_random_case_in_grimace_support,
    grimace_support,
    sample_rdkit_random_support,
)


class RDKITRootedRandomWriterTests(unittest.TestCase):
    """RDKit writer tests mapped onto Grimace support semantics.

    Source in the local RDKit fork:
    - Code/GraphMol/SmilesParse/test.cpp:testdoRandomSmileGeneration()
    """

    def test_rdkit_rooted_random_generation_cases_are_in_grimace_support(self) -> None:
        for case in ROOTED_RANDOM_CASES:
            assert_rooted_random_case_in_grimace_support(self, case)

    def test_rdkit_random_start_behavior_is_reflected_in_unrooted_support(self) -> None:
        case = ROOTED_RANDOM_CASES[0]
        mol = parse_smiles(case.smiles)

        rdBase.SeedRandomNumberGenerator(0xF00D)
        rdkit_seen = {
            Chem.MolToSmiles(
                Chem.Mol(mol),
                isomericSmiles=True,
                kekuleSmiles=False,
                rootedAtAtom=-1,
                canonical=False,
                doRandom=True,
            )
            for _ in range(100)
        }
        rdkit_starts = {smiles[0] for smiles in rdkit_seen}

        support = grimace_support(
            mol,
            rooted_at_atom=None,
            isomeric_smiles=True,
        )
        grimace_starts = {smiles[0] for smiles in support}

        self.assertTrue({"C", "c"} <= grimace_starts)
        self.assertTrue("n" in grimace_starts or "O" in grimace_starts)
        self.assertTrue(rdkit_starts <= grimace_starts)

    def test_rooted_polyene_bond_stereo_case_matches_high_draw_rdkit_support(self) -> None:
        mol = parse_smiles("CC1=C(C(CCC1)(C)C)/C=C/C(=C/C=C/C(=C/C(=O)O)/C)/C")
        expected = sample_rdkit_random_support(
            mol,
            root_idx=11,
            isomeric_smiles=True,
            draw_budget=2000,
        )

        self.assertEqual(120, len(expected))
        self.assertEqual(
            expected,
            grimace_support(
                mol,
                rooted_at_atom=11,
                isomeric_smiles=True,
            ),
        )

    def test_rooted_polyene_internal_branch_carrier_case_matches_high_draw_rdkit_support(self) -> None:
        mol = parse_smiles("CC1=C(C(CCC1)(C)C)/C=C/C(=C/C=C/C(=C/C(=O)O)/C)/C")
        expected = sample_rdkit_random_support(
            mol,
            root_idx=12,
            isomeric_smiles=True,
            draw_budget=5000,
        )

        self.assertEqual(80, len(expected))
        self.assertEqual(
            expected,
            grimace_support(
                mol,
                rooted_at_atom=12,
                isomeric_smiles=True,
            ),
        )

    def test_rooted_steroid_ring_coupled_stereo_case_matches_high_draw_rdkit_support(self) -> None:
        mol = parse_smiles(
            "C[C@H](/C=C/[C@H](C)C(C)C)[C@H]1CC[C@@H]\\\\2"
            "[C@@]1(CCC/C2=C\\\\C=C/3\\\\C[C@H](CCC3=C)O)C"
        )
        support = grimace_support(
            mol,
            rooted_at_atom=0,
            isomeric_smiles=True,
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

        self.assertIn(expected, support)
        self.assertNotIn(rejected, support)

    def test_rooted_sidechain_steroid_outputs_are_in_support(self) -> None:
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
                support = grimace_support(
                    mol,
                    rooted_at_atom=root_idx,
                    isomeric_smiles=True,
                )
                self.assertIn(expected, support)


if __name__ == "__main__":
    unittest.main()
