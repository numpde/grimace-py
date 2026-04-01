from __future__ import annotations

import unittest

from tests.helpers.rdkit_writer_cases import DATASET_REGRESSION_CASES
from tests.rdkit_serialization._support import assert_exact_writer_case_in_grimace_support


class RDKITDatasetRegressionTests(unittest.TestCase):
    """Dataset-derived RDKit writer regressions on Grimace's public surface."""

    def test_dataset_regression_cases_are_members_of_grimace_support(self) -> None:
        for case in DATASET_REGRESSION_CASES:
            with self.subTest(
                smiles=case.smiles,
                expected=case.expected,
                rooted_at_atom=case.rooted_at_atom,
                isomeric_smiles=case.isomeric_smiles,
            ):
                assert_exact_writer_case_in_grimace_support(self, case)


if __name__ == "__main__":
    unittest.main()
