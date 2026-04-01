from __future__ import annotations

import unittest

import grimace
from tests.helpers.mols import parse_smiles


class RDKITBondTokenWriterTests(unittest.TestCase):
    """RDKit-grounded bond-token serialization behavior.

    These cases are public-surface regressions discovered against RDKit output:
    - explicit dative bond arrows
    - explicit '-' between aromatic atoms when RDKit writes it
    """

    def test_dative_bond_serialization_matches_rdkit(self) -> None:
        mol = parse_smiles("[NH3][Cu]")
        expected_by_root = {
            0: {"[NH3]->[Cu]"},
            1: {"[Cu]<-[NH3]"},
        }

        for isomeric_smiles in (False, True):
            for root_idx, expected in expected_by_root.items():
                with self.subTest(isomeric_smiles=isomeric_smiles, root_idx=root_idx):
                    actual = set(
                        grimace.MolToSmilesEnum(
                            mol,
                            isomericSmiles=isomeric_smiles,
                            rootedAtAtom=root_idx,
                            canonical=False,
                            doRandom=True,
                        )
                    )
                    self.assertEqual(expected, actual)

    def test_aromatic_bridge_single_bond_matches_rdkit(self) -> None:
        mol = parse_smiles("C1=CC=C(C=C1)N2C=C(C=N2)C=O")
        expected_by_root = {
            0: {
                "c1ccc(-n2cc(C=O)cn2)cc1",
                "c1ccc(-n2cc(cn2)C=O)cc1",
                "c1ccc(-n2ncc(C=O)c2)cc1",
                "c1ccc(-n2ncc(c2)C=O)cc1",
                "c1ccc(cc1)-n1cc(C=O)cn1",
                "c1ccc(cc1)-n1cc(cn1)C=O",
                "c1ccc(cc1)-n1ncc(C=O)c1",
                "c1ccc(cc1)-n1ncc(c1)C=O",
            }
        }

        for isomeric_smiles in (False, True):
            for root_idx, expected in expected_by_root.items():
                with self.subTest(isomeric_smiles=isomeric_smiles, root_idx=root_idx):
                    actual = set(
                        grimace.MolToSmilesEnum(
                            mol,
                            isomericSmiles=isomeric_smiles,
                            rootedAtAtom=root_idx,
                            canonical=False,
                            doRandom=True,
                        )
                    )
                    self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
