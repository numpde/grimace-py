from __future__ import annotations

import unittest

from rdkit import Chem

from tests.helpers.mols import parse_smiles
from tests.helpers.rdkit_disconnected_sampling import load_disconnected_root_zero_smiles
from tests.rdkit_serialization._support import assert_grimace_support_matches_rdkit_sampling


class RDKITDisconnectedWriterTests(unittest.TestCase):
    """RDKit disconnected-writer behavior applied to Grimace's public surface.

    These are RDKit-grounded public-surface checks rather than upstream copied
    single-function tests. The aim is to preserve RDKit's fragment-order and
    rooted-fragment behavior for disconnected molecules.
    """

    def test_root_omitted_support_matches_rdkit_sampling(self) -> None:
        for smiles, isomeric_smiles in (
            ("O.CCO", True),
            ("[Na+].C#N", False),
            ("[Na+].C#N", True),
        ):
            mol = parse_smiles(smiles)
            with self.subTest(smiles=smiles, isomeric_smiles=isomeric_smiles):
                assert_grimace_support_matches_rdkit_sampling(
                    self,
                    mol=mol,
                    rooted_at_atom=None,
                    isomeric_smiles=isomeric_smiles,
                    draw_budget=512,
                )

    def test_disconnected_cyanide_salt_matches_rdkit_across_all_roots(self) -> None:
        mol = parse_smiles("[Na+].C#N")

        for isomeric_smiles in (False, True):
            for root_idx in range(mol.GetNumAtoms()):
                with self.subTest(isomeric_smiles=isomeric_smiles, root_idx=root_idx):
                    assert_grimace_support_matches_rdkit_sampling(
                        self,
                        mol=mol,
                        rooted_at_atom=root_idx,
                        isomeric_smiles=isomeric_smiles,
                        draw_budget=256,
                    )

    def test_disconnected_root_zero_suite_matches_rdkit_sampling(self) -> None:
        root_zero_cases = load_disconnected_root_zero_smiles()
        self.assertEqual(30, len(root_zero_cases))

        for smiles in root_zero_cases:
            mol = parse_smiles(smiles)
            self.assertGreater(len(Chem.GetMolFrags(mol)), 1)
            for isomeric_smiles in (False, True):
                with self.subTest(smiles=smiles, isomeric_smiles=isomeric_smiles):
                    assert_grimace_support_matches_rdkit_sampling(
                        self,
                        mol=mol,
                        rooted_at_atom=0,
                        isomeric_smiles=isomeric_smiles,
                        draw_budget=256,
                    )


if __name__ == "__main__":
    unittest.main()
