from __future__ import annotations

import unittest

from rdkit import Chem

from tests.rdkit_serialization._support import (
    grimace_support,
    sample_rdkit_random_support,
)


class RDKITWriterFlagTests(unittest.TestCase):
    """RDKit writer-flag expectations mapped onto Grimace support membership.

    Deterministic RDKit writer-membership cases live in
    `tests/fixtures/rdkit_writer_membership/`.
    """

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


if __name__ == "__main__":
    unittest.main()
