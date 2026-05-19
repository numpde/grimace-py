from __future__ import annotations

import unittest

import grimace
from tests.helpers.mols import parse_smiles
from tests.helpers.public_runtime import supported_public_kwargs


class SmilesDeviationTests(unittest.TestCase):
    def _deviation(self, smiles: str, candidate: object, **kwargs: object):
        return grimace.MolToSmilesDeviation(
            parse_smiles(smiles),
            candidate,
            **supported_public_kwargs(**kwargs),
        )

    def _assert_deviation(
        self,
        deviation: object,
        *,
        reason: str,
        char_index: int,
        token_index: int | None,
        offset_in_token: int | None,
        accepted_text: str,
        rejected_text: str,
        legal_next_tokens: tuple[str, ...],
    ) -> None:
        self.assertIsInstance(deviation, grimace.SmilesDeviation)
        self.assertEqual(reason, deviation.reason)
        self.assertEqual(char_index, deviation.char_index)
        self.assertEqual(token_index, deviation.token_index)
        self.assertEqual(offset_in_token, deviation.offset_in_token)
        self.assertEqual(accepted_text, deviation.accepted_text)
        self.assertEqual(rejected_text, deviation.rejected_text)
        self.assertEqual(legal_next_tokens, deviation.legal_next_tokens)

    def test_exact_serialization_returns_none(self) -> None:
        self.assertIsNone(
            self._deviation("CCO", "CCO", rootedAtAtom=0, isomericSmiles=False)
        )

    def test_exact_external_token_serialization_returns_none(self) -> None:
        self.assertIsNone(
            self._deviation(
                "CCO",
                ("C", "C", "O"),
                rootedAtAtom=0,
                isomericSmiles=False,
            )
        )

    def test_empty_string_reports_initial_choices(self) -> None:
        self._assert_deviation(
            self._deviation("CCO", "", rootedAtAtom=0, isomericSmiles=False),
            reason="incomplete",
            char_index=0,
            token_index=None,
            offset_in_token=None,
            accepted_text="",
            rejected_text="",
            legal_next_tokens=("C",),
        )

    def test_empty_external_token_sequence_reports_initial_choices(self) -> None:
        self._assert_deviation(
            self._deviation("CCO", (), rootedAtAtom=0, isomericSmiles=False),
            reason="incomplete",
            char_index=0,
            token_index=None,
            offset_in_token=None,
            accepted_text="",
            rejected_text="",
            legal_next_tokens=("C",),
        )

    def test_string_candidate_reports_unexpected_character_after_longest_prefix(self) -> None:
        self._assert_deviation(
            self._deviation(
                "CCO",
                "CCN",
                rootedAtAtom=0,
                isomericSmiles=False,
            ),
            reason="unexpected_text",
            char_index=2,
            token_index=None,
            offset_in_token=None,
            accepted_text="CC",
            rejected_text="N",
            legal_next_tokens=("O",),
        )

    def test_unrooted_string_candidate_accepts_multiple_roots(self) -> None:
        self.assertIsNone(
            self._deviation("CCO", "CCO", rootedAtAtom=-1, isomericSmiles=False)
        )
        self.assertIsNone(
            self._deviation("CCO", "OCC", rootedAtAtom=-1, isomericSmiles=False)
        )

    def test_unrooted_string_candidate_reports_merged_initial_choices(self) -> None:
        self._assert_deviation(
            self._deviation("CCO", "N", rootedAtAtom=-1, isomericSmiles=False),
            reason="unexpected_text",
            char_index=0,
            token_index=None,
            offset_in_token=None,
            accepted_text="",
            rejected_text="N",
            legal_next_tokens=("C", "O"),
        )

    def test_unrooted_string_candidate_reports_merged_later_choices(self) -> None:
        self._assert_deviation(
            self._deviation("CCO", "CO", rootedAtAtom=-1, isomericSmiles=False),
            reason="unexpected_text",
            char_index=1,
            token_index=None,
            offset_in_token=None,
            accepted_text="C",
            rejected_text="O",
            legal_next_tokens=("(", "C"),
        )

    def test_string_candidate_can_be_resegmented_over_grimace_tokens(self) -> None:
        self._assert_deviation(
            self._deviation(
                "CCO",
                "CCl",
                rootedAtAtom=0,
                isomericSmiles=False,
            ),
            reason="unexpected_text",
            char_index=2,
            token_index=None,
            offset_in_token=None,
            accepted_text="CC",
            rejected_text="l",
            legal_next_tokens=("O",),
        )

    def test_external_tokens_are_atomic_observations(self) -> None:
        self._assert_deviation(
            self._deviation(
                "CCO",
                ("C", "Cl"),
                rootedAtAtom=0,
                isomericSmiles=False,
            ),
            reason="unexpected_token",
            char_index=1,
            token_index=1,
            offset_in_token=0,
            accepted_text="C",
            rejected_text="Cl",
            legal_next_tokens=("C",),
        )

    def test_external_token_sequence_accepts_lists_as_well_as_tuples(self) -> None:
        self.assertIsNone(
            self._deviation(
                "CCO",
                ["C", "C", "O"],
                rootedAtAtom=0,
                isomericSmiles=False,
            )
        )

    def test_external_tokens_can_match_multi_character_grimace_tokens(self) -> None:
        self.assertIsNone(
            self._deviation(
                "CCl",
                ("C", "Cl"),
                rootedAtAtom=0,
                isomericSmiles=False,
            )
        )

    def test_external_tokens_do_not_split_multi_character_grimace_tokens(self) -> None:
        self._assert_deviation(
            self._deviation(
                "CCl",
                ("C", "C", "l"),
                rootedAtAtom=0,
                isomericSmiles=False,
            ),
            reason="unexpected_token",
            char_index=1,
            token_index=1,
            offset_in_token=0,
            accepted_text="C",
            rejected_text="Cl",
            legal_next_tokens=("Cl",),
        )

    def test_string_candidate_reports_extra_text_after_terminal_serialization(self) -> None:
        self._assert_deviation(
            self._deviation(
                "CCl",
                "CCla",
                rootedAtAtom=0,
                isomericSmiles=False,
            ),
            reason="unexpected_text",
            char_index=3,
            token_index=None,
            offset_in_token=None,
            accepted_text="CCl",
            rejected_text="a",
            legal_next_tokens=(),
        )

    def test_external_token_sequence_does_not_split_a_grimace_token(self) -> None:
        self._assert_deviation(
            self._deviation(
                "[Na+]",
                ("[", "Na", "+]"),
                rootedAtAtom=0,
                isomericSmiles=False,
            ),
            reason="unexpected_token",
            char_index=0,
            token_index=0,
            offset_in_token=0,
            accepted_text="",
            rejected_text="[Na+]",
            legal_next_tokens=("[Na+]",),
        )

    def test_string_candidate_can_partially_match_a_grimace_token(self) -> None:
        self._assert_deviation(
            self._deviation(
                "[Na+]",
                "[Na",
                rootedAtAtom=0,
                isomericSmiles=False,
            ),
            reason="incomplete",
            char_index=3,
            token_index=None,
            offset_in_token=None,
            accepted_text="[Na",
            rejected_text="",
            legal_next_tokens=("[Na+]",),
        )

    def test_branch_string_candidate_reports_all_legal_branch_entries(self) -> None:
        self._assert_deviation(
            self._deviation("CC(C)O", "CC(", rootedAtAtom=0, isomericSmiles=False),
            reason="incomplete",
            char_index=3,
            token_index=None,
            offset_in_token=None,
            accepted_text="CC(",
            rejected_text="",
            legal_next_tokens=("C", "O"),
        )

    def test_branch_string_candidate_rejects_atom_where_branch_is_required(self) -> None:
        self._assert_deviation(
            self._deviation("CC(C)O", "CCN", rootedAtAtom=0, isomericSmiles=False),
            reason="unexpected_text",
            char_index=2,
            token_index=None,
            offset_in_token=None,
            accepted_text="CC",
            rejected_text="N",
            legal_next_tokens=("(",),
        )

    def test_branch_string_candidate_accepts_alternate_branch_order(self) -> None:
        self.assertIsNone(
            self._deviation(
                "CC(C)O",
                "CC(O)C",
                rootedAtAtom=0,
                isomericSmiles=False,
            )
        )

    def test_branch_external_tokens_report_all_legal_branch_entries(self) -> None:
        self._assert_deviation(
            self._deviation(
                "CC(C)O",
                ("C", "C", "("),
                rootedAtAtom=0,
                isomericSmiles=False,
            ),
            reason="incomplete",
            char_index=3,
            token_index=2,
            offset_in_token=1,
            accepted_text="CC(",
            rejected_text="",
            legal_next_tokens=("C", "O"),
        )

    def test_ring_string_candidate_reports_missing_ring_closure_digit(self) -> None:
        self._assert_deviation(
            self._deviation("C1CC1", "C1CC", rootedAtAtom=0, isomericSmiles=False),
            reason="incomplete",
            char_index=4,
            token_index=None,
            offset_in_token=None,
            accepted_text="C1CC",
            rejected_text="",
            legal_next_tokens=("1",),
        )

    def test_ring_string_candidate_rejects_wrong_atom_before_closure(self) -> None:
        self._assert_deviation(
            self._deviation("C1CC1", "C1CO", rootedAtAtom=0, isomericSmiles=False),
            reason="unexpected_text",
            char_index=3,
            token_index=None,
            offset_in_token=None,
            accepted_text="C1C",
            rejected_text="O",
            legal_next_tokens=("C",),
        )

    def test_ring_external_tokens_do_not_coalesce_adjacent_atoms(self) -> None:
        self._assert_deviation(
            self._deviation(
                "C1CC1",
                ("C", "1", "CC", "1"),
                rootedAtAtom=0,
                isomericSmiles=False,
            ),
            reason="unexpected_token",
            char_index=2,
            token_index=2,
            offset_in_token=0,
            accepted_text="C1",
            rejected_text="CC1",
            legal_next_tokens=("C",),
        )

    def test_incomplete_external_token_sequence_reports_boundary_location(self) -> None:
        self._assert_deviation(
            self._deviation(
                "CCO",
                ("C", "C"),
                rootedAtAtom=0,
                isomericSmiles=False,
            ),
            reason="incomplete",
            char_index=2,
            token_index=1,
            offset_in_token=1,
            accepted_text="CC",
            rejected_text="",
            legal_next_tokens=("O",),
        )

    def test_incomplete_external_token_inside_grimace_token_reports_unexpected_token(self) -> None:
        self._assert_deviation(
            self._deviation(
                "[Na+]",
                ("[", "Na"),
                rootedAtAtom=0,
                isomericSmiles=False,
            ),
            reason="unexpected_token",
            char_index=0,
            token_index=0,
            offset_in_token=0,
            accepted_text="",
            rejected_text="[Na",
            legal_next_tokens=("[Na+]",),
        )

    def test_disconnected_string_candidate_requires_dot_between_fragments(self) -> None:
        self._assert_deviation(
            self._deviation("[Na+].CC", "[Na+]CC", rootedAtAtom=0, isomericSmiles=False),
            reason="unexpected_text",
            char_index=5,
            token_index=None,
            offset_in_token=None,
            accepted_text="[Na+]",
            rejected_text="CC",
            legal_next_tokens=(".",),
        )

    def test_disconnected_external_tokens_accept_fragment_separator(self) -> None:
        self.assertIsNone(
            self._deviation(
                "[Na+].CC",
                ("[Na+]", ".", "C", "C"),
                rootedAtAtom=0,
                isomericSmiles=False,
            )
        )

    def test_disconnected_external_tokens_require_fragment_separator(self) -> None:
        self._assert_deviation(
            self._deviation(
                "[Na+].CC",
                ("[Na+]", "C", "C"),
                rootedAtAtom=0,
                isomericSmiles=False,
            ),
            reason="unexpected_token",
            char_index=5,
            token_index=1,
            offset_in_token=0,
            accepted_text="[Na+]",
            rejected_text="CC",
            legal_next_tokens=(".",),
        )

    def test_explicit_bond_flag_changes_string_legal_next_token(self) -> None:
        self._assert_deviation(
            self._deviation(
                "CCO",
                "CCO",
                rootedAtAtom=0,
                isomericSmiles=False,
                allBondsExplicit=True,
            ),
            reason="unexpected_text",
            char_index=1,
            token_index=None,
            offset_in_token=None,
            accepted_text="C",
            rejected_text="CO",
            legal_next_tokens=("-",),
        )

    def test_explicit_bond_external_tokens_accept_bond_tokens(self) -> None:
        self.assertIsNone(
            self._deviation(
                "CCO",
                ("C", "-", "C", "-", "O"),
                rootedAtAtom=0,
                isomericSmiles=False,
                allBondsExplicit=True,
            )
        )

    def test_candidate_token_sequence_must_contain_strings(self) -> None:
        with self.assertRaisesRegex(TypeError, "contain strings"):
            self._deviation(
                "CCO",
                ("C", object(), "O"),
                rootedAtAtom=0,
                isomericSmiles=False,
            )

    def test_empty_external_tokens_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty strings"):
            self._deviation(
                "CCO",
                ("C", "", "CO"),
                rootedAtAtom=0,
                isomericSmiles=False,
            )


if __name__ == "__main__":
    unittest.main()
