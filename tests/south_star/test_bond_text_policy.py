from __future__ import annotations

import unittest

from rdkit import Chem

from grimace._south_star.bond_text import bond_text_obligation_for_supported_bond
from grimace._south_star.bond_text import bond_text_for_supported_bond
from tests.helpers.south_star_semantic_oracle import parse_smiles


class SouthStarBondTextPolicyTests(unittest.TestCase):
    def test_supported_bond_text_uses_obligation_boundary(self) -> None:
        cases = (
            ("CC", Chem.BondType.SINGLE, "", "elided_single_bond"),
            ("C=C", Chem.BondType.DOUBLE, "=", "explicit_double_bond"),
            ("C#N", Chem.BondType.TRIPLE, "#", "explicit_triple_bond"),
        )

        for smiles, expected_type, expected_text, expected_family in cases:
            bond = parse_smiles(smiles).GetBondWithIdx(0)
            obligation = bond_text_obligation_for_supported_bond(bond)

            with self.subTest(smiles=smiles):
                self.assertEqual(expected_type, obligation.bond_type)
                self.assertEqual(expected_text, obligation.emitted_text)
                self.assertEqual(expected_family, obligation.token_family)
                self.assertEqual(expected_text, bond_text_for_supported_bond(bond))

    def test_unsupported_bond_text_remains_fail_fast(self) -> None:
        bond = parse_smiles("C$C").GetBondWithIdx(0)

        with self.assertRaisesRegex(NotImplementedError, "QUADRUPLE"):
            bond_text_obligation_for_supported_bond(bond)


if __name__ == "__main__":
    unittest.main()
