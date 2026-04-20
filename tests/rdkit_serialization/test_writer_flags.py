from __future__ import annotations

import unittest

from tests.helpers.rdkit_writer_cases import ExactWriterCase, WRITER_FLAG_CASES
from tests.rdkit_serialization._support import assert_exact_writer_case_in_grimace_support


class RDKITWriterFlagTests(unittest.TestCase):
    """RDKit writer-flag expectations mapped onto Grimace support membership.

    Sources in the local RDKit fork:
    - Code/GraphMol/Wrap/rough_test.py:test75AllBondsExplicit()
    - Code/GraphMol/Wrap/rough_test.py:testIgnoreAtomMapNumbers()
    - Code/GraphMol/Wrap/rough_test.py:testIssue266()
    - Code/GraphMol/JavaWrappers/gmwrapper/src-test/org/RDKit/SmilesDetailsTests.java
      - testRootedAt()
      - testBug1719046()
      - testBug1842174()
    - Code/GraphMol/SmilesParse/test.cpp:testGithub1219()
    """

    def test_rdkit_writer_flag_expectations_are_members_of_grimace_support(self) -> None:
        for case in WRITER_FLAG_CASES:
            with self.subTest(
                smiles=case.smiles,
                expected=case.expected,
                rooted_at_atom=case.rooted_at_atom,
                isomeric_smiles=case.isomeric_smiles,
                kekule_smiles=case.kekule_smiles,
                all_bonds_explicit=case.all_bonds_explicit,
                all_hs_explicit=case.all_hs_explicit,
                ignore_atom_map_numbers=case.ignore_atom_map_numbers,
            ):
                assert_exact_writer_case_in_grimace_support(self, case)

    def test_all_bonds_explicit_nonisomeric_bond_stereo_exact_writer_is_member_of_support(self) -> None:
        case = ExactWriterCase(
            smiles="F/C=C\\Cl",
            expected="F/C=C\\Cl",
            isomeric_smiles=False,
            rooted_at_atom=0,
            rdkit_canonical=False,
            all_bonds_explicit=True,
        )

        assert_exact_writer_case_in_grimace_support(self, case)

    def test_aromatic_selenium_exact_writer_is_member_of_support(self) -> None:
        case = ExactWriterCase(
            smiles="C1=CC=C(C=C1)N2C(=O)C3=CC=CC=C3[Se]2",
            expected="O=c1c2ccccc2[se]n1-c1ccccc1",
            isomeric_smiles=True,
        )

        assert_exact_writer_case_in_grimace_support(self, case)


if __name__ == "__main__":
    unittest.main()
