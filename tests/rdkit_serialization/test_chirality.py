from __future__ import annotations

import unittest

from tests.helpers.rdkit_writer_cases import CANONICAL_CHIRALITY_CASES
from tests.rdkit_serialization._support import assert_exact_writer_case_in_grimace_support


class RDKITChiralityWriterTests(unittest.TestCase):
    """RDKit chirality-writing tests mapped onto Grimace support membership.

    Source in the local RDKit fork:
    - Code/GraphMol/Wrap/rough_test.py:test31ChiralitySmiles()
    """

    def test_rdkit_canonical_chirality_serializations_are_members_of_grimace_support(self) -> None:
        for case in CANONICAL_CHIRALITY_CASES:
            with self.subTest(smiles=case.smiles):
                assert_exact_writer_case_in_grimace_support(self, case)


if __name__ == "__main__":
    unittest.main()
