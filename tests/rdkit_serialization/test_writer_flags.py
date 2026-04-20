from __future__ import annotations

import unittest

from rdkit import Chem

from tests.helpers.rdkit_writer_cases import ExactWriterCase, WRITER_FLAG_CASES
from tests.rdkit_serialization._support import (
    assert_exact_writer_case_in_grimace_support,
    grimace_support,
    sample_rdkit_random_support,
)


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

    def test_coupled_diphenyl_diene_exact_writer_is_member_of_support(self) -> None:
        mol = Chem.MolFromSmiles("C/C=C(/C(=C/C)/c1ccccc1)\\c1ccccc1")
        expected = sample_rdkit_random_support(
            mol,
            root_idx=None,
            isomeric_smiles=True,
            draw_budget=20_000,
        )
        support = grimace_support(
            mol,
            rooted_at_atom=None,
            isomeric_smiles=True,
        )
        self.assertEqual(expected, support)

    def test_rooted_tetrasubstituted_alkene_explicit_writer_is_member_of_support(self) -> None:
        cases = (
            ExactWriterCase(
                smiles="C/C(=C(/C)\\c1ccccc1)/c1ccccc1",
                expected=(
                    "[C](\\[CH3])(=[C](/[CH3])-[c]1:[cH]:[cH]:[cH]:[cH]:[cH]:1)"
                    "-[c]1:[cH]:[cH]:[cH]:[cH]:[cH]:1"
                ),
                isomeric_smiles=True,
                rooted_at_atom=1,
                rdkit_canonical=False,
                all_bonds_explicit=True,
                all_hs_explicit=True,
            ),
            ExactWriterCase(
                smiles="C/C(=C(/C)\\c1ccccc1)/c1ccccc1",
                expected=(
                    "[C](=[C](/[CH3])-[c]1:[cH]:[cH]:[cH]:[cH]:[cH]:1)(\\[CH3])"
                    "-[c]1:[cH]:[cH]:[cH]:[cH]:[cH]:1"
                ),
                isomeric_smiles=True,
                rooted_at_atom=2,
                rdkit_canonical=False,
                all_bonds_explicit=True,
                all_hs_explicit=True,
            ),
        )

        for case in cases:
            with self.subTest(rooted_at_atom=case.rooted_at_atom, expected=case.expected):
                assert_exact_writer_case_in_grimace_support(self, case)


if __name__ == "__main__":
    unittest.main()
