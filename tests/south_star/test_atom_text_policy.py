from __future__ import annotations

import unittest

from grimace._south_star.atom_text import SOUTH_STAR_BRACKET_ATOM_TEXT_TOKENS
from grimace._south_star.atom_text import SOUTH_STAR_ORGANIC_ATOM_TEXT_TOKENS
from grimace._south_star.atom_text import atom_text_modifier_obligations
from grimace._south_star.atom_text import atom_text_for_supported_atom
from grimace._south_star.atom_text import atom_text_obligation_for_supported_atom
from grimace._south_star.atom_text import south_star_atom_text_fields
from grimace._south_star.atom_text import tetrahedral_atom_text_obligation
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
        obligation = atom_text_obligation_for_supported_atom(mol.GetAtomWithIdx(0))

        self.assertEqual("[H]", atom_text_for_supported_atom(mol.GetAtomWithIdx(0)))
        self.assertEqual("[H]", obligation.emitted_text)
        self.assertEqual("bracket_atom", obligation.token_family)
        self.assertEqual(("element_requires_bracket",), obligation.bracket_obligations)
        self.assertTrue(obligation.uses_brackets)

    def test_organic_subset_renderer_has_no_bracket_obligation(self) -> None:
        mol = parse_smiles("CC")
        obligation = atom_text_obligation_for_supported_atom(mol.GetAtomWithIdx(0))

        self.assertEqual("C", obligation.emitted_text)
        self.assertEqual("organic_subset", obligation.token_family)
        self.assertEqual((), obligation.bracket_obligations)
        self.assertFalse(obligation.uses_brackets)

    def test_tetrahedral_renderer_uses_atom_text_obligation_boundary(self) -> None:
        mol = parse_smiles("C[C@H](F)Cl")
        obligation = tetrahedral_atom_text_obligation(
            mol.GetAtomWithIdx(1),
            stereo_token="@",
            implicit_hydrogen_count=1,
        )

        self.assertEqual("[C@H]", obligation.emitted_text)
        self.assertEqual("bracket_atom", obligation.token_family)
        self.assertEqual(
            ("tetrahedral_stereo_token", "implicit_hydrogen_text"),
            obligation.bracket_obligations,
        )
        self.assertTrue(obligation.uses_brackets)

    def test_policy_names_deferred_bracket_modifiers(self) -> None:
        cases = (
            (
                "[2H][H]",
                "isotope",
                "isotope",
                2,
                "unsupported_atom_isotope",
            ),
            (
                "[H+]",
                "charge",
                "formal_charge",
                1,
                "unsupported_atom_charge",
            ),
            (
                "[H]",
                "radical",
                "radical_electron_count",
                1,
                "unsupported_radical_atom",
            ),
            (
                "[CH3:1]C",
                "atom_map",
                "atom_map_number",
                1,
                "unsupported_atom_map",
            ),
        )

        for (
            smiles,
            expected_modifier,
            expected_field,
            expected_value,
            expected_category,
        ) in cases:
            mol = parse_smiles(smiles)
            fields = south_star_atom_text_fields(mol.GetAtomWithIdx(0))
            obligations = atom_text_modifier_obligations(fields)
            categories = {
                reason.category for reason in unsupported_atom_text_reasons(fields)
            }

            with self.subTest(smiles=smiles):
                self.assertEqual(1, len(obligations))
                obligation = obligations[0]
                self.assertIs(fields, obligation.fields)
                self.assertEqual(fields.atom_idx, obligation.atom_idx)
                self.assertEqual(expected_modifier, obligation.modifier_name)
                self.assertEqual(expected_field, obligation.field_name)
                self.assertEqual(expected_value, obligation.value)
                self.assertEqual(expected_category, obligation.unsupported_category)
                self.assertEqual(
                    "bracket_atom_modifier_renderer",
                    obligation.renderer_requirement,
                )
                self.assertEqual(
                    expected_category,
                    obligation.unsupported_reason.category,
                )
                self.assertIn(expected_category, categories)

    def test_supported_atom_text_has_no_modifier_obligation(self) -> None:
        mol = parse_smiles("CC")
        fields = south_star_atom_text_fields(mol.GetAtomWithIdx(0))

        self.assertEqual((), atom_text_modifier_obligations(fields))

    def test_modifier_obligations_do_not_enable_renderer_support(self) -> None:
        cases = (
            ("[2H][H]", "unsupported_atom_isotope"),
            ("[H+]", "unsupported_atom_charge"),
            ("[H]", "unsupported_radical_atom"),
            ("[CH3:1]C", "unsupported_atom_map"),
        )

        for smiles, expected_category in cases:
            mol = parse_smiles(smiles)

            with self.subTest(smiles=smiles):
                with self.assertRaisesRegex(NotImplementedError, expected_category):
                    atom_text_obligation_for_supported_atom(mol.GetAtomWithIdx(0))


if __name__ == "__main__":
    unittest.main()
