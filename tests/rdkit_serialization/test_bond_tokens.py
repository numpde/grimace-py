from __future__ import annotations

import unittest

from tests.helpers.rdkit_writer_cases import BOND_TOKEN_CASES
from tests.rdkit_serialization._support import assert_exact_support_case_equals_grimace_support


class RDKITBondTokenWriterTests(unittest.TestCase):
    """RDKit-grounded bond-token serialization behavior.

    These cases are public-surface regressions discovered against RDKit output:
    - explicit dative bond arrows
    - explicit '-' between aromatic atoms when RDKit writes it
    """

    def test_bond_token_cases_match_rdkit(self) -> None:
        for case in BOND_TOKEN_CASES:
            for isomeric_smiles in (False, True):
                assert_exact_support_case_equals_grimace_support(
                    self,
                    case,
                    isomeric_smiles=isomeric_smiles,
                )


if __name__ == "__main__":
    unittest.main()
