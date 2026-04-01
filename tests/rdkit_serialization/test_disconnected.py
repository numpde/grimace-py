from __future__ import annotations

import unittest

from rdkit import Chem

import grimace
from tests.helpers.mols import parse_smiles
from tests.helpers.rdkit_writer_cases import DISCONNECTED_ROOT_ZERO_CASES
from tests.rdkit_serialization._support import sample_rdkit_random_support


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
                expected = sample_rdkit_random_support(
                    mol,
                    root_idx=None,
                    isomeric_smiles=isomeric_smiles,
                    draw_budget=512,
                )
                actual = set(
                    grimace.MolToSmilesEnum(
                        mol,
                        isomericSmiles=isomeric_smiles,
                        canonical=False,
                        doRandom=True,
                    )
                )
                self.assertEqual(expected, actual)

    def test_disconnected_cyanide_salt_matches_rdkit_across_all_roots(self) -> None:
        mol = parse_smiles("[Na+].C#N")

        for isomeric_smiles in (False, True):
            for root_idx in range(mol.GetNumAtoms()):
                with self.subTest(isomeric_smiles=isomeric_smiles, root_idx=root_idx):
                    expected = sample_rdkit_random_support(
                        mol,
                        root_idx=root_idx,
                        isomeric_smiles=isomeric_smiles,
                        draw_budget=256,
                    )
                    actual = set(
                        grimace.MolToSmilesEnum(
                            mol,
                            rootedAtAtom=root_idx,
                            isomericSmiles=isomeric_smiles,
                            canonical=False,
                            doRandom=True,
                        )
                    )
                    self.assertEqual(expected, actual)

    def test_disconnected_root_zero_suite_matches_rdkit_sampling(self) -> None:
        self.assertEqual(30, len(DISCONNECTED_ROOT_ZERO_CASES))

        for smiles in DISCONNECTED_ROOT_ZERO_CASES:
            mol = parse_smiles(smiles)
            self.assertGreater(len(Chem.GetMolFrags(mol)), 1)
            for isomeric_smiles in (False, True):
                with self.subTest(smiles=smiles, isomeric_smiles=isomeric_smiles):
                    expected = sample_rdkit_random_support(
                        mol,
                        root_idx=0,
                        isomeric_smiles=isomeric_smiles,
                        draw_budget=256,
                    )
                    actual = set(
                        grimace.MolToSmilesEnum(
                            mol,
                            rootedAtAtom=0,
                            isomericSmiles=isomeric_smiles,
                            canonical=False,
                            doRandom=True,
                        )
                    )
                    self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
