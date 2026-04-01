from __future__ import annotations

import unittest

from rdkit import Chem, rdBase

from tests.helpers.mols import parse_smiles
from tests.helpers.rdkit_writer_cases import ROOTED_RANDOM_CASES
from tests.rdkit_serialization._support import grimace_support


class RDKITRootedRandomWriterTests(unittest.TestCase):
    """RDKit writer tests mapped onto Grimace support semantics.

    Source in the local RDKit fork:
    - Code/GraphMol/SmilesParse/test.cpp:testdoRandomSmileGeneration()
    """

    def test_rdkit_rooted_random_generation_cases_are_in_grimace_support(self) -> None:
        for case in ROOTED_RANDOM_CASES:
            mol = parse_smiles(case.smiles)
            self.assertEqual(mol.GetNumAtoms(), len(case.rooted_outputs))

            for root_idx, expected in enumerate(case.rooted_outputs):
                with self.subTest(smiles=case.smiles, root_idx=root_idx):
                    rdkit_rooted = Chem.MolToSmiles(
                        mol,
                        isomericSmiles=True,
                        kekuleSmiles=False,
                        rootedAtAtom=root_idx,
                        canonical=False,
                        doRandom=False,
                    )
                    self.assertEqual(expected, rdkit_rooted)

                    support = grimace_support(
                        mol,
                        rooted_at_atom=root_idx,
                        isomeric_smiles=True,
                    )
                    self.assertIn(expected, support)
                    self.assertGreaterEqual(len(support), 3)

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


if __name__ == "__main__":
    unittest.main()
