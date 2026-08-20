from __future__ import annotations

import unittest

from rdkit import rdBase

from tests.helpers.rdkit_writer_membership import load_pinned_writer_membership_cases
from tests.rdkit_serialization._support import assert_exact_writer_case_in_grimace_support


class RDKITWriterMembershipTests(unittest.TestCase):
    """Pinned RDKit deterministic writer outputs represented in Grimace support."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.cases = load_pinned_writer_membership_cases(rdBase.rdkitVersion)
        except FileNotFoundError:
            raise unittest.SkipTest(
                f"no pinned writer-membership corpus for RDKit {rdBase.rdkitVersion}"
            )

    def test_rdkit_deterministic_writer_outputs_are_members_of_grimace_support(self) -> None:
        for case in self.cases:
            with self.subTest(
                case_id=case.case_id,
                source=case.source,
                smiles=case.smiles,
                expected=case.expected,
                rooted_at_atom=case.rooted_at_atom,
                isomeric_smiles=case.isomeric_smiles,
                rdkit_canonical=case.rdkit_canonical,
                kekule_smiles=case.kekule_smiles,
                all_bonds_explicit=case.all_bonds_explicit,
                all_hs_explicit=case.all_hs_explicit,
                ignore_atom_map_numbers=case.ignore_atom_map_numbers,
            ):
                assert_exact_writer_case_in_grimace_support(self, case)


if __name__ == "__main__":
    unittest.main()
