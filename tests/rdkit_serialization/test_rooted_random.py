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


if __name__ == "__main__":
    unittest.main()
