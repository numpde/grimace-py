from __future__ import annotations

import unittest

from rdkit import Chem

from tests.helpers.mols import parse_smiles
from tests.helpers.rdkit_writer_cases import WRITER_FLAG_CASES
from tests.rdkit_serialization._support import grimace_support


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

    @staticmethod
    def _rdkit_smiles(case) -> str:
        mol = parse_smiles(case.smiles)
        kwargs = dict(
            isomericSmiles=case.isomeric_smiles,
            canonical=True,
            kekuleSmiles=case.kekule_smiles,
            allBondsExplicit=case.all_bonds_explicit,
            allHsExplicit=case.all_hs_explicit,
            ignoreAtomMapNumbers=case.ignore_atom_map_numbers,
        )
        if case.rooted_at_atom is not None:
            kwargs["rootedAtAtom"] = case.rooted_at_atom
        return Chem.MolToSmiles(mol, **kwargs)

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
                rdkit_out = self._rdkit_smiles(case)
                self.assertEqual(case.expected, rdkit_out)

                support = grimace_support(
                    parse_smiles(case.smiles),
                    rooted_at_atom=case.rooted_at_atom,
                    isomeric_smiles=case.isomeric_smiles,
                    kekule_smiles=case.kekule_smiles,
                    all_bonds_explicit=case.all_bonds_explicit,
                    all_hs_explicit=case.all_hs_explicit,
                    ignore_atom_map_numbers=case.ignore_atom_map_numbers,
                )
                self.assertIn(case.expected, support)


if __name__ == "__main__":
    unittest.main()
