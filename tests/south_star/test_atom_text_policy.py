from __future__ import annotations

import unittest

from grimace._south_star.atom_text import SOUTH_STAR_AROMATIC_ATOM_TEXT_TOKENS
from grimace._south_star.atom_text import (
    SOUTH_STAR_BRACKET_AROMATIC_ATOM_TEXT_TOKENS,
)
from grimace._south_star.atom_text import (
    SOUTH_STAR_BRACKET_ONLY_AROMATIC_ATOM_TEXT_TOKENS,
)
from grimace._south_star.atom_text import SOUTH_STAR_BRACKET_ONLY_ATOM_TEXT_TOKENS
from grimace._south_star.atom_text import SOUTH_STAR_BRACKET_ATOM_TEXT_TOKENS
from grimace._south_star.atom_text import SOUTH_STAR_ORGANIC_ATOM_TEXT_TOKENS
from grimace._south_star.atom_text import atom_text_modifier_obligations
from grimace._south_star.atom_text import atom_text_for_supported_atom
from grimace._south_star.atom_text import atom_text_obligation_for_supported_atom
from grimace._south_star.atom_text import is_south_star_bracket_atom_text_token
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
        for token in SOUTH_STAR_AROMATIC_ATOM_TEXT_TOKENS:
            with self.subTest(token=token):
                self.assertTrue(south_star_grammar_conformance(token).passed)
        for token in SOUTH_STAR_BRACKET_ATOM_TEXT_TOKENS:
            with self.subTest(token=token):
                self.assertTrue(south_star_grammar_conformance(token).passed)
        for token in SOUTH_STAR_BRACKET_AROMATIC_ATOM_TEXT_TOKENS:
            with self.subTest(token=token):
                self.assertTrue(south_star_grammar_conformance(token).passed)

    def test_supported_atom_renderer_uses_policy_boundary(self) -> None:
        mol = parse_smiles("[H][H]")
        obligation = atom_text_obligation_for_supported_atom(mol.GetAtomWithIdx(0))

        self.assertEqual("[H]", atom_text_for_supported_atom(mol.GetAtomWithIdx(0)))
        self.assertEqual("[H]", obligation.emitted_text)
        self.assertEqual("bracket_atom", obligation.token_family)
        self.assertEqual(
            ("bracket_atom", "element_requires_bracket"),
            obligation.bracket_obligations,
        )
        self.assertTrue(obligation.uses_brackets)

    def test_organic_subset_renderer_has_no_bracket_obligation(self) -> None:
        mol = parse_smiles("CC")
        obligation = atom_text_obligation_for_supported_atom(mol.GetAtomWithIdx(0))

        self.assertEqual("C", obligation.emitted_text)
        self.assertEqual("organic_subset", obligation.token_family)
        self.assertEqual((), obligation.bracket_obligations)
        self.assertFalse(obligation.uses_brackets)

    def test_non_organic_symbol_renderer_requires_bracket_text(self) -> None:
        cases = (
            ("[SiH3]C", "[SiH3]", ("non_organic_symbol_requires_bracket",)),
            (
                "[SeH]",
                "[SeH]",
                (
                    "non_organic_symbol_requires_bracket",
                    "radical_valence_semantics",
                ),
            ),
        )

        for smiles, expected_text, required_obligations in cases:
            atom = parse_smiles(smiles).GetAtomWithIdx(0)
            obligation = atom_text_obligation_for_supported_atom(atom)

            with self.subTest(smiles=smiles):
                self.assertIn(atom.GetSymbol(), SOUTH_STAR_BRACKET_ONLY_ATOM_TEXT_TOKENS)
                self.assertEqual(expected_text, obligation.emitted_text)
                self.assertEqual("bracket_atom", obligation.token_family)
                for required_obligation in required_obligations:
                    self.assertIn(
                        required_obligation,
                        obligation.bracket_obligations,
                    )
                self.assertTrue(obligation.uses_brackets)

    def test_aromatic_subset_renderer_uses_lowercase_atom_text(self) -> None:
        mol = parse_smiles("c1ccccc1")
        obligation = atom_text_obligation_for_supported_atom(mol.GetAtomWithIdx(0))

        self.assertEqual("c", obligation.emitted_text)
        self.assertEqual("aromatic_subset", obligation.token_family)
        self.assertEqual((), obligation.bracket_obligations)
        self.assertFalse(obligation.uses_brackets)

    def test_bracket_aromatic_atom_text_tokens_are_named_policy(self) -> None:
        accepted_tokens = ("[nH]", "[15nH]", "[n:7]", "[nH+]", "[n+]", "[se]")
        rejected_tokens = ("[n@H]", "[n@@H]", "[Na+]")

        for token in accepted_tokens:
            with self.subTest(token=token):
                self.assertTrue(is_south_star_bracket_atom_text_token(token))
                self.assertTrue(south_star_grammar_conformance(token).passed)
                if token == "[se]":
                    self.assertIn("se", SOUTH_STAR_BRACKET_ONLY_AROMATIC_ATOM_TEXT_TOKENS)
        for token in rejected_tokens:
            with self.subTest(token=token):
                self.assertFalse(is_south_star_bracket_atom_text_token(token))
                self.assertFalse(south_star_grammar_conformance(token).passed)

    def test_bracket_aromatic_renderer_uses_structured_fields(self) -> None:
        cases = (
            (
                "c1cc[nH]c1",
                3,
                "[nH]",
                ("bracket_aromatic_atom", "explicit_hydrogen_count"),
            ),
            (
                "c1cc[15nH]c1",
                3,
                "[15nH]",
                (
                    "bracket_aromatic_atom",
                    "isotope_prefix",
                    "explicit_hydrogen_count",
                ),
            ),
            (
                "[nH:7]1cccc1",
                0,
                "[nH:7]",
                (
                    "bracket_aromatic_atom",
                    "explicit_hydrogen_count",
                    "atom_map_suffix",
                ),
            ),
            (
                "c1cc[n:7]cc1",
                3,
                "[n:7]",
                ("bracket_aromatic_atom", "atom_map_suffix"),
            ),
            (
                "c1cc[nH+]cc1",
                3,
                "[nH+]",
                (
                    "bracket_aromatic_atom",
                    "explicit_hydrogen_count",
                    "charge_suffix",
                ),
            ),
            (
                "[se]1cccc1",
                0,
                "[se]",
                ("bracket_aromatic_atom",),
            ),
        )

        for smiles, atom_idx, expected_text, expected_obligations in cases:
            atom = parse_smiles(smiles).GetAtomWithIdx(atom_idx)
            obligation = atom_text_obligation_for_supported_atom(atom)

            with self.subTest(smiles=smiles):
                self.assertEqual(expected_text, obligation.emitted_text)
                self.assertEqual("bracket_aromatic_atom", obligation.token_family)
                self.assertEqual(expected_obligations, obligation.bracket_obligations)
                self.assertTrue(obligation.uses_brackets)
                self.assertTrue(is_south_star_bracket_atom_text_token(expected_text))

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

    def test_policy_names_renderer_capable_bracket_modifiers(self) -> None:
        cases = (
            (
                "[2H][H]",
                ("isotope",),
                ("isotope",),
                (2,),
                (None,),
                ("bracket_atom_isotope_prefix",),
            ),
            (
                "[H+]",
                ("charge",),
                ("formal_charge",),
                (1,),
                (None,),
                ("bracket_atom_charge_suffix",),
            ),
            (
                "[Cl-]",
                ("charge",),
                ("formal_charge",),
                (-1,),
                (None,),
                ("bracket_atom_charge_suffix",),
            ),
            (
                "[CH3:1]C",
                ("atom_map",),
                ("atom_map_number",),
                (1,),
                (None,),
                ("bracket_atom_map_suffix",),
            ),
            (
                "[13CH3:7]C",
                ("isotope", "atom_map"),
                ("isotope", "atom_map_number"),
                (13, 7),
                (None, None),
                ("bracket_atom_isotope_prefix", "bracket_atom_map_suffix"),
            ),
            (
                "[15NH3+]C",
                ("isotope", "charge"),
                ("isotope", "formal_charge"),
                (15, 1),
                (None, None),
                ("bracket_atom_isotope_prefix", "bracket_atom_charge_suffix"),
            ),
            (
                "[CH3]",
                ("radical",),
                ("radical_electron_count",),
                (1,),
                (None,),
                ("bracket_atom_radical_valence_semantics",),
            ),
        )

        for (
            smiles,
            expected_modifiers,
            expected_fields,
            expected_values,
            expected_categories,
            expected_renderer_requirements,
        ) in cases:
            mol = parse_smiles(smiles)
            fields = south_star_atom_text_fields(mol.GetAtomWithIdx(0))
            obligations = atom_text_modifier_obligations(fields)
            categories = {
                reason.category for reason in unsupported_atom_text_reasons(fields)
            }

            with self.subTest(smiles=smiles):
                self.assertEqual(len(expected_modifiers), len(obligations))
                for idx, obligation in enumerate(obligations):
                    self.assertIs(fields, obligation.fields)
                    self.assertEqual(fields.atom_idx, obligation.atom_idx)
                    self.assertEqual(expected_modifiers[idx], obligation.modifier_name)
                    self.assertEqual(expected_fields[idx], obligation.field_name)
                    self.assertEqual(expected_values[idx], obligation.value)
                    self.assertEqual(
                        expected_categories[idx], obligation.unsupported_category
                    )
                    self.assertEqual(
                        expected_renderer_requirements[idx],
                        obligation.renderer_requirement,
                    )
                    self.assertNotIn(expected_categories[idx], categories)

    def test_radical_modifier_is_renderer_capable_semantic_field(self) -> None:
        fields = south_star_atom_text_fields(parse_smiles("[CH3]").GetAtomWithIdx(0))
        obligations = atom_text_modifier_obligations(fields)
        categories = {
            reason.category for reason in unsupported_atom_text_reasons(fields)
        }

        self.assertEqual(1, len(obligations))
        obligation = obligations[0]
        self.assertEqual("radical", obligation.modifier_name)
        self.assertIsNone(obligation.unsupported_category)
        self.assertEqual(
            "bracket_atom_radical_valence_semantics",
            obligation.renderer_requirement,
        )
        self.assertNotIn("unsupported_radical_atom", categories)

    def test_bracket_modifier_renderer_uses_structured_fields(self) -> None:
        cases = (
            ("[2H][H]", "[2H]", ("bracket_atom", "isotope_prefix")),
            ("[H+]", "[H+]", ("bracket_atom", "charge_suffix")),
            ("[Cl-]", "[Cl-]", ("bracket_atom", "charge_suffix")),
            (
                "[NH4+]",
                "[NH4+]",
                ("bracket_atom", "explicit_hydrogen_count", "charge_suffix"),
            ),
            (
                "[CH3:1]C",
                "[CH3:1]",
                ("bracket_atom", "explicit_hydrogen_count", "atom_map_suffix"),
            ),
            (
                "[13CH3:7]C",
                "[13CH3:7]",
                (
                    "bracket_atom",
                    "isotope_prefix",
                    "explicit_hydrogen_count",
                    "atom_map_suffix",
                ),
            ),
            (
                "[15NH3+]C",
                "[15NH3+]",
                (
                    "bracket_atom",
                    "isotope_prefix",
                    "explicit_hydrogen_count",
                    "charge_suffix",
                ),
            ),
            (
                "[CH3]",
                "[CH3]",
                (
                    "bracket_atom",
                    "explicit_hydrogen_count",
                    "radical_valence_semantics",
                ),
            ),
            ("[O]", "[O]", ("bracket_atom", "radical_valence_semantics")),
            (
                "[SiH3]C",
                "[SiH3]",
                (
                    "bracket_atom",
                    "non_organic_symbol_requires_bracket",
                    "explicit_hydrogen_count",
                ),
            ),
            (
                "[SeH]",
                "[SeH]",
                (
                    "bracket_atom",
                    "non_organic_symbol_requires_bracket",
                    "explicit_hydrogen_count",
                    "radical_valence_semantics",
                ),
            ),
        )

        for smiles, expected_text, expected_obligations in cases:
            atom = parse_smiles(smiles).GetAtomWithIdx(0)
            obligation = atom_text_obligation_for_supported_atom(atom)

            with self.subTest(smiles=smiles):
                self.assertEqual(expected_text, obligation.emitted_text)
                self.assertEqual(expected_obligations, obligation.bracket_obligations)
                self.assertTrue(obligation.uses_brackets)
                self.assertTrue(is_south_star_bracket_atom_text_token(expected_text))

    def test_supported_atom_text_has_no_modifier_obligation(self) -> None:
        mol = parse_smiles("CC")
        fields = south_star_atom_text_fields(mol.GetAtomWithIdx(0))

        self.assertEqual((), atom_text_modifier_obligations(fields))

    def test_radical_renderer_uses_bracket_valence_semantics(self) -> None:
        obligation = atom_text_obligation_for_supported_atom(
            parse_smiles("[H]").GetAtomWithIdx(0)
        )

        self.assertEqual("[H]", obligation.emitted_text)
        self.assertEqual(
            ("bracket_atom", "element_requires_bracket", "radical_valence_semantics"),
            obligation.bracket_obligations,
        )


if __name__ == "__main__":
    unittest.main()
