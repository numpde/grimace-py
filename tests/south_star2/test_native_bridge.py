"""Exercise owned native successors across the private Python boundary."""

from __future__ import annotations

import unittest

from grimace._core import _SouthStar2PreparedWriter
from grimace._south_star2.preparation import prepare_molecule
from rdkit import Chem


class NativeBridgeTests(unittest.TestCase):
    def test_equal_text_keeps_distinct_owned_root_successors(self) -> None:
        prepared = _SouthStar2PreparedWriter(
            ["C", "C", "O"], [(0, 1, ""), (1, 2, "")], "generic_any_order"
        )
        initial = prepared.initial_state()
        carbons = [state for text, state in initial.choices() if text == "C"]
        self.assertEqual(len(carbons), 2)
        next_texts = [tuple(text for text, _ in state.choices()) for state in carbons]
        self.assertNotEqual(next_texts[0], next_texts[1])
        del prepared
        self.assertEqual(len(initial.choices()), 3)
        self.assertFalse(initial.is_accepted)
        for state in carbons:
            self.assertTrue(state.choices())

    def test_empty_graph_is_accepted(self) -> None:
        initial = _SouthStar2PreparedWriter([], [], "generic_any_order").initial_state()
        self.assertTrue(initial.is_accepted)
        self.assertEqual(initial.choices(), [])

    def test_text_prefix_keeps_equal_text_roots_and_unions_next_tokens(self) -> None:
        prepared = _SouthStar2PreparedWriter(
            ["C", "C", "O", "C"],
            [(0, 1, ""), (1, 2, ""), (2, 3, "")],
            "generic_any_order",
        )
        source = prepared.initial_prefix()
        first_choices = source.choices()
        carbon_prefix = next(prefix for text, prefix in first_choices if text == "C")
        carbon_roots = [state for text, state in prepared.initial_state().choices() if text == "C"]
        expected_next = {
            text for state in carbon_roots for text, _ in state.choices()
        }

        self.assertEqual(source.frontier_size, 1)
        self.assertEqual(carbon_prefix.frontier_size, 3)
        self.assertEqual(
            {text for text, _ in carbon_prefix.choices()}, expected_next
        )
        self.assertEqual(source.frontier_size, 1)

        sibling = next(prefix for text, prefix in first_choices if text == "O")
        del prepared, first_choices
        self.assertEqual(carbon_prefix.frontier_size, 3)
        self.assertEqual(sibling.frontier_size, 1)
        self.assertFalse(carbon_prefix.is_accepted)

    def test_text_prefix_width_counts_retained_states(self) -> None:
        prepared = _SouthStar2PreparedWriter(
            ["C", "C", "C"], [], "generic_any_order"
        )
        prefix = next(
            candidate
            for text, candidate in prepared.initial_prefix().choices()
            if text == "C"
        )
        for token in (".", "C", ".", "C"):
            prefix = next(
                candidate
                for text, candidate in prefix.choices()
                if text == token
            )
        self.assertEqual(prefix.frontier_size, 6)
        self.assertTrue(prefix.is_accepted)
        self.assertEqual(prefix.choices(), [])

    def test_bad_bond_endpoints_fail_during_preparation(self) -> None:
        for bonds, reason in (
            ([(0, 2, "")], "unknown"),
            ([(0, 0, "")], "self-bond"),
            ([(0, 1, ""), (1, 0, "")], "duplicate bond"),
        ):
            with self.subTest(bonds=bonds):
                with self.assertRaises(ValueError) as caught:
                    _SouthStar2PreparedWriter(["C", "O"], bonds, "generic_any_order")
                self.assertIn("Cannot prepare bond", str(caught.exception))
                self.assertIn(reason, str(caught.exception))

    def test_bond_spelling_is_not_guessed(self) -> None:
        with self.assertRaises(ValueError) as caught:
            _SouthStar2PreparedWriter(
                ["C", "O"], [(0, 1, "~")], "generic_any_order"
            )
        self.assertIn("bond 0", str(caught.exception))
        self.assertIn('"~"', str(caught.exception))

    def test_empty_atom_text_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as caught:
            _SouthStar2PreparedWriter([""], [], "generic_any_order")
        self.assertIn("Cannot prepare the writer", str(caught.exception))
        self.assertIn("must not be empty", str(caught.exception))

    def test_component_order_is_required_and_validated(self) -> None:
        with self.assertRaises(TypeError):
            _SouthStar2PreparedWriter(["C"], [])
        with self.assertRaisesRegex(ValueError, "writer_regime"):
            _SouthStar2PreparedWriter(["C"], [], "sometimes")

    def test_raw_any_order_keeps_both_disconnected_permutations(self) -> None:
        prepared = _SouthStar2PreparedWriter(
            ["C", "O"], [], "generic_any_order"
        )
        self.assertEqual(
            {text for text, _ in prepared.initial_state().choices()}, {"C", "O"}
        )
        self.assertEqual(
            {text for text, _ in prepared.initial_prefix().choices()}, {"C", "O"}
        )
        accepted = set()
        pending = [(prepared.initial_state(), "")]
        while pending:
            state, text = pending.pop()
            if state.is_accepted:
                accepted.add(text)
            else:
                pending.extend(
                    (successor, text + token) for token, successor in state.choices()
                )
        self.assertEqual(accepted, {"C.O", "O.C"})

    def test_input_order_prefix_retains_both_carbon_roots_through_period(self) -> None:
        molecule = Chem.MolFromSmiles("CCO.O")
        assert molecule is not None
        prepared = prepare_molecule(molecule).build_writer()
        prefix = next(
            candidate
            for text, candidate in prepared.initial_prefix().choices()
            if text == "C"
        )
        self.assertEqual(prefix.frontier_size, 2)
        initial = dict(prepared.initial_prefix().choices())
        self.assertEqual(set(initial), {"C", "O"})
        self.assertEqual(initial["C"].frontier_size, 2)
        self.assertEqual(initial["O"].frontier_size, 1)

        accepted = set()
        pending = [(prefix, "C")]
        while pending:
            current, text = pending.pop()
            if current.is_accepted:
                accepted.add(text)
                continue
            choices = current.choices()
            for token, successor in choices:
                if token == ".":
                    self.assertTrue(text.startswith("C"))
                    self.assertEqual(
                        {next_text for next_text, _ in successor.choices()}, {"O"}
                    )
                pending.append((successor, text + token))
        self.assertTrue(accepted)
        self.assertTrue(all(text.endswith(".O") for text in accepted))

    def test_generic_input_order_keeps_all_roots_in_earliest_component(self) -> None:
        prepared = _SouthStar2PreparedWriter(
            ["C", "C", "O", "O"],
            [(0, 1, ""), (1, 2, "")],
            "generic_input_order",
        )
        initial = dict(prepared.initial_prefix().choices())
        self.assertEqual(initial["C"].frontier_size, 2)
        self.assertEqual(initial["O"].frontier_size, 1)


if __name__ == "__main__":
    unittest.main()
