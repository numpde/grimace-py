from __future__ import annotations

import unittest

from grimace._south_star.atom_text import SOUTH_STAR_BRACKET_ATOM_TEXT_TOKENS
from grimace._south_star.atom_text import SOUTH_STAR_ORGANIC_ATOM_TEXT_TOKENS
from grimace._south_star.atom_text import atom_text_for_supported_atom
from grimace._south_star.atom_text import south_star_atom_text_fields
from grimace._south_star.atom_text import unsupported_atom_text_reasons
from tests.helpers.south_star_grammar_conformance import (
    south_star_grammar_conformance,
)
from tests.helpers.south_star_semantic_oracle import parse_smiles


class SouthStarAtomTextPolicyTests(unittest.TestCase):
    def test_policy_tokens_are_grammar_tokens(self) -> None:
        for token in SOUTH_STAR_ORGANIC_ATOM_TEXT_TOKENS:
            with self.subTest(token=token):
                self.assertTrue(south_star_grammar_conformance(token).passed)
        for token in SOUTH_STAR_BRACKET_ATOM_TEXT_TOKENS:
            with self.subTest(token=token):
                self.assertTrue(south_star_grammar_conformance(token).passed)

    def test_supported_atom_renderer_uses_policy_boundary(self) -> None:
        mol = parse_smiles("[H][H]")

        self.assertEqual("[H]", atom_text_for_supported_atom(mol.GetAtomWithIdx(0)))

    def test_policy_names_deferred_bracket_modifiers(self) -> None:
        cases = (
            ("[2H][H]", "unsupported_atom_isotope"),
            ("[H+]", "unsupported_atom_charge"),
            ("[H]", "unsupported_radical_atom"),
            ("[CH3:1]C", "unsupported_atom_map"),
        )

        for smiles, expected_category in cases:
            mol = parse_smiles(smiles)
            fields = south_star_atom_text_fields(mol.GetAtomWithIdx(0))
            categories = {
                reason.category for reason in unsupported_atom_text_reasons(fields)
            }

            with self.subTest(smiles=smiles):
                self.assertIn(expected_category, categories)


if __name__ == "__main__":
    unittest.main()
