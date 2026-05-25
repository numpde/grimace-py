"""Tests for reversible online residual constraints."""

from __future__ import annotations

import unittest

from grimace._south_star1.facts import DirectionalValue
from grimace._south_star1.facts import SiteStatus
from grimace._south_star1.facts import TetraValue
from grimace._south_star1.ids import OccurrenceId
from grimace._south_star1.policy import DirectionMark
from grimace._south_star1.policy import TetraToken
from grimace._south_star1.residual_constraints import DirectionalCarrierResidual
from grimace._south_star1.residual_constraints import DirectionalResidualFactor
from grimace._south_star1.residual_constraints import ResidualStore
from grimace._south_star1.residual_constraints import TetraResidualFactor
from grimace._south_star1.residual_constraints import VarId
from grimace._south_star1.residual_constraints import direction_var
from grimace._south_star1.residual_constraints import tetra_var


class ResidualConstraintTest(unittest.TestCase):
    def test_tetra_specified_factor_forces_one_token_for_local_order(self) -> None:
        factor = _tetra_factor(
            target=TetraValue.PLUS,
            local_order=(0, 1, 2, 3),
        )

        self.assertEqual(factor.allowed_tokens(), frozenset((TetraToken.AT,)))
        self.assertTrue(factor.assign(tetra_var(0), TetraToken.AT))
        self.assertTrue(factor.close())

    def test_tetra_swap_flips_forced_token(self) -> None:
        factor = _tetra_factor(
            target=TetraValue.PLUS,
            local_order=(1, 0, 2, 3),
        )

        self.assertEqual(factor.allowed_tokens(), frozenset((TetraToken.ATAT,)))
        self.assertFalse(factor.assign(tetra_var(0), TetraToken.AT))
        self.assertTrue(factor.assign(tetra_var(0), TetraToken.ATAT))

    def test_tetra_unspecified_rejects_at_tokens(self) -> None:
        factor = TetraResidualFactor(
            scope=(tetra_var(0),),
            status=SiteStatus.UNSPECIFIED,
            target=TetraValue.NONE,
            reference_order=_occurrences(0, 1, 2, 3),
            local_order=_occurrences(0, 1, 2, 3),
        )

        self.assertFalse(factor.assign(tetra_var(0), TetraToken.AT))
        self.assertTrue(factor.assign(tetra_var(0), TetraToken.NONE))
        self.assertTrue(factor.close())

    def test_directional_factor_accepts_exact_specified_pair(self) -> None:
        factor = _directional_factor(DirectionalValue.OPPOSITE)

        self.assertTrue(factor.assign(direction_var(1), DirectionMark.FWD))
        self.assertTrue(factor.assign(direction_var(2), DirectionMark.REV))
        self.assertEqual(factor.value(), DirectionalValue.OPPOSITE)
        self.assertTrue(factor.close())

    def test_directional_factor_rejects_same_endpoint_inconsistent_signs(
        self,
    ) -> None:
        left_a = direction_var("left-a")
        left_b = direction_var("left-b")
        right = direction_var("right")
        factor = DirectionalResidualFactor(
            scope=(left_a, left_b, right),
            status=SiteStatus.SPECIFIED,
            target=DirectionalValue.TOGETHER,
            carrier_models={
                left_a: DirectionalCarrierResidual(left_a, "left", 1, 1),
                left_b: DirectionalCarrierResidual(left_b, "left", 1, 1),
                right: DirectionalCarrierResidual(right, "right", 1, 1),
            },
        )

        self.assertTrue(factor.assign(left_a, DirectionMark.FWD))
        self.assertFalse(factor.assign(left_b, DirectionMark.REV))

    def test_directional_factor_returns_none_for_one_sided_marks(self) -> None:
        factor = _directional_factor(DirectionalValue.OPPOSITE)

        self.assertTrue(factor.assign(direction_var(1), DirectionMark.FWD))

        self.assertEqual(factor.value(), DirectionalValue.NONE)

    def test_directional_unspecified_rejects_accidental_two_sided_stereo(
        self,
    ) -> None:
        factor = _directional_factor(
            DirectionalValue.NONE,
            status=SiteStatus.UNSPECIFIED,
        )

        self.assertTrue(factor.assign(direction_var(1), DirectionMark.FWD))
        self.assertTrue(factor.assign(direction_var(2), DirectionMark.FWD))
        self.assertEqual(factor.value(), DirectionalValue.TOGETHER)
        self.assertFalse(factor.close())

    def test_checkpoint_rollback_restores_store_and_factor_state(self) -> None:
        store = ResidualStore()
        left = direction_var(1)
        right = direction_var(2)
        store.add_var(left, (DirectionMark.ABSENT, DirectionMark.FWD, DirectionMark.REV))
        store.add_var(right, (DirectionMark.ABSENT, DirectionMark.FWD, DirectionMark.REV))
        factor_id = store.add_factor(_directional_factor(DirectionalValue.OPPOSITE))
        checkpoint = store.checkpoint()

        self.assertTrue(store.assign(left, DirectionMark.FWD))
        self.assertTrue(store.assign(right, DirectionMark.REV))
        self.assertTrue(store.close_factor(factor_id))

        store.rollback(checkpoint)
        self.assertIsNone(store.assignment(left))
        self.assertIsNone(store.assignment(right))
        self.assertFalse(store.close_factor(factor_id))
        self.assertTrue(store.assign(left, DirectionMark.FWD))
        self.assertTrue(store.assign(right, DirectionMark.REV))
        self.assertTrue(store.close_factor(factor_id))

    def test_residual_store_value_snapshot_is_canonical_by_var_order(self) -> None:
        left = ResidualStore()
        right = ResidualStore()
        first = VarId("test", (1,))
        second = VarId("test", (2,))

        for var in (second, first):
            left.add_var(var, ("a", "b"))
        for var in (first, second):
            right.add_var(var, ("a", "b"))
        self.assertTrue(left.assign(first, "a"))
        self.assertTrue(left.assign(second, "b"))
        self.assertTrue(right.assign(second, "b"))
        self.assertTrue(right.assign(first, "a"))

        self.assertEqual(left.value_snapshot(), right.value_snapshot())


def _tetra_factor(
    *,
    target: TetraValue,
    local_order: tuple[int, ...],
) -> TetraResidualFactor:
    return TetraResidualFactor(
        scope=(tetra_var(0),),
        status=SiteStatus.SPECIFIED,
        target=target,
        reference_order=_occurrences(0, 1, 2, 3),
        local_order=_occurrences(*local_order),
    )


def _directional_factor(
    target: DirectionalValue,
    *,
    status: SiteStatus = SiteStatus.SPECIFIED,
) -> DirectionalResidualFactor:
    left = direction_var(1)
    right = direction_var(2)
    return DirectionalResidualFactor(
        scope=(left, right),
        status=status,
        target=target,
        carrier_models={
            left: DirectionalCarrierResidual(left, "left", 1, 1),
            right: DirectionalCarrierResidual(right, "right", 1, 1),
        },
    )


def _occurrences(*values: int) -> tuple[OccurrenceId, ...]:
    return tuple(OccurrenceId(value) for value in values)


if __name__ == "__main__":
    unittest.main()
