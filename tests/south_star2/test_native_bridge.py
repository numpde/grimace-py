"""Exercise owned native successors across the private Python boundary."""

from __future__ import annotations

import unittest

from grimace._core import _SouthStar2PreparedWriter


class NativeBridgeTests(unittest.TestCase):
    def test_equal_text_keeps_distinct_owned_root_successors(self) -> None:
        prepared = _SouthStar2PreparedWriter(
            ["C", "C", "O"], [(0, 1, ""), (1, 2, "")]
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
        initial = _SouthStar2PreparedWriter([], []).initial_state()
        self.assertTrue(initial.is_accepted)
        self.assertEqual(initial.choices(), [])

    def test_bad_bond_endpoints_fail_during_preparation(self) -> None:
        for bonds, reason in (
            ([(0, 2, "")], "unknown"),
            ([(0, 0, "")], "self-bond"),
            ([(0, 1, ""), (1, 0, "")], "duplicate bond"),
        ):
            with self.subTest(bonds=bonds):
                with self.assertRaises(ValueError) as caught:
                    _SouthStar2PreparedWriter(["C", "O"], bonds)
                self.assertIn("Cannot prepare bond", str(caught.exception))
                self.assertIn(reason, str(caught.exception))

    def test_bond_spelling_is_not_guessed(self) -> None:
        with self.assertRaises(ValueError) as caught:
            _SouthStar2PreparedWriter(["C", "O"], [(0, 1, "~")])
        self.assertIn("bond 0", str(caught.exception))
        self.assertIn('"~"', str(caught.exception))

    def test_empty_atom_text_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as caught:
            _SouthStar2PreparedWriter([""], [])
        self.assertIn("Cannot prepare the writer", str(caught.exception))
        self.assertIn("must not be empty", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
