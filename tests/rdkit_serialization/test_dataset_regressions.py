from __future__ import annotations

import unittest

from rdkit import Chem

from tests.helpers.mols import parse_smiles
from tests.rdkit_serialization._support import grimace_support


class RDKITDatasetRegressionTests(unittest.TestCase):
    """Dataset-derived RDKit writer regressions on Grimace's public surface."""

    def test_cid_1548887_ring_opening_must_not_precommit_stereo_carrier(self) -> None:
        # Discovered by scanning the local top_100000 dataset for RDKit strings
        # that fall outside Grimace's exact rooted support.
        smiles = "CC\\1=C(C2=C(/C1=C\\C3=CC=C(C=C3)S(=O)C)C=CC(=C2)F)CC(=O)O"
        expected = "CC1=C(CC(=O)O)c2c(ccc(F)c2)/C1=C\\c1ccc(S(=O)C)cc1"
        mol = parse_smiles(smiles)

        rdkit_rooted = Chem.MolToSmiles(
            Chem.Mol(mol),
            isomericSmiles=True,
            canonical=False,
            doRandom=False,
            rootedAtAtom=0,
        )
        self.assertEqual(expected, rdkit_rooted)

        support = grimace_support(
            mol,
            rooted_at_atom=0,
            isomeric_smiles=True,
        )
        self.assertIn(expected, support)

    def test_cid_54609463_rooted_porhyrin_like_metal_complex_matches_rdkit(self) -> None:
        smiles = (
            "C1=CC=C2/C/3=N/C4=C5C(=C([N-]4)/N=C/6\\[N-]/C(=N\\C7=C8C(=C([N-]7)"
            "/N=C(/C2=C1)\\[N-]3)C=CC=C8)/C9=CC=CC=C69)C=CC=C5.[Cu]"
        )
        expected = (
            "c1ccc2/c3[n-]/c(c2c1)=N\\c1c2c(c([n-]1)/N=c1\\[n-]/c(c4c1cccc4)"
            "=N\\c1c4c(c([n-]1)/N=3)cccc4)cccc2.[Cu]"
        )
        mol = parse_smiles(smiles)

        rdkit_rooted = Chem.MolToSmiles(
            Chem.Mol(mol),
            isomericSmiles=True,
            canonical=False,
            doRandom=False,
            rootedAtAtom=0,
        )
        self.assertEqual(expected, rdkit_rooted)

        support = grimace_support(
            mol,
            rooted_at_atom=0,
            isomeric_smiles=True,
        )
        self.assertIn(expected, support)


if __name__ == "__main__":
    unittest.main()
