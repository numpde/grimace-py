from __future__ import annotations

import unittest

from rdkit import Chem

from tests.helpers.mols import parse_smiles
from tests.helpers.rdkit_writer_cases import CANONICAL_CHIRALITY_CASES
from tests.rdkit_serialization._support import grimace_support


class RDKITChiralityWriterTests(unittest.TestCase):
    """RDKit chirality-writing tests mapped onto Grimace support membership.

    Source in the local RDKit fork:
    - Code/GraphMol/Wrap/rough_test.py:test31ChiralitySmiles()
    """

    def test_rdkit_canonical_chirality_serializations_are_members_of_grimace_support(self) -> None:
        for case in CANONICAL_CHIRALITY_CASES:
            mol = parse_smiles(case.smiles)
            with self.subTest(smiles=case.smiles):
                rdkit_canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
                self.assertEqual(case.expected, rdkit_canonical)

                support = grimace_support(
                    mol,
                    rooted_at_atom=None,
                    isomeric_smiles=True,
                )
                self.assertIn(case.expected, support)


if __name__ == "__main__":
    unittest.main()
