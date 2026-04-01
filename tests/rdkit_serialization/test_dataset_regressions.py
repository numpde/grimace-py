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


if __name__ == "__main__":
    unittest.main()
