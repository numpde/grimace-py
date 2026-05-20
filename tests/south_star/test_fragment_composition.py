from __future__ import annotations

import unittest

from grimace._south_star.fragments import compose_disconnected_fragment_supports
from grimace._south_star.fragments import SouthStarFragmentSupport


class SouthStarFragmentCompositionTests(unittest.TestCase):
    def test_all_fragment_order_policy_joins_rendered_fragment_supports(self) -> None:
        result = compose_disconnected_fragment_supports(
            (
                SouthStarFragmentSupport(fragment_id="alkene", outputs=("F/C=C/Cl",)),
                SouthStarFragmentSupport(fragment_id="oxygen", outputs=("O",)),
            )
        )

        self.assertEqual(
            (
                "F/C=C/Cl.O",
                "O.F/C=C/Cl",
            ),
            result.outputs,
        )
        self.assertEqual(2, result.fragment_count)
        self.assertEqual((1, 1), result.fragment_output_counts)
        self.assertEqual("all_fragment_orders", result.fragment_order_policy)
        self.assertEqual(2, result.fragment_order_count)
        self.assertEqual(2, result.estimated_product_size)

    def test_composition_multiplies_local_supports_by_fragment_orders(self) -> None:
        result = compose_disconnected_fragment_supports(
            (
                SouthStarFragmentSupport(
                    fragment_id="left",
                    outputs=("F/C=C/Cl", "F\\C=C\\Cl"),
                ),
                SouthStarFragmentSupport(
                    fragment_id="right",
                    outputs=("Br/C=C/I", "Br\\C=C\\I"),
                ),
                SouthStarFragmentSupport(fragment_id="oxygen", outputs=("O",)),
            )
        )

        self.assertEqual(24, result.estimated_product_size)
        self.assertEqual(24, len(result.outputs))
        self.assertIn("F/C=C/Cl.Br/C=C/I.O", result.outputs)
        self.assertIn("O.Br\\C=C\\I.F\\C=C\\Cl", result.outputs)

    def test_fragment_outputs_must_be_connected_strings(self) -> None:
        with self.assertRaisesRegex(ValueError, "already disconnected"):
            SouthStarFragmentSupport(fragment_id="bad", outputs=("CC.O",))

    def test_duplicate_fragment_outputs_fail_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate outputs"):
            SouthStarFragmentSupport(fragment_id="bad", outputs=("CC", "CC"))


if __name__ == "__main__":
    unittest.main()
