"""Tests for reversible online residual constraints."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import unittest

from grimace._south_star1.facts import DirectionalValue
from grimace._south_star1.facts import SiteStatus
from grimace._south_star1.facts import TetraValue
from grimace._south_star1.ids import OccurrenceId
from grimace._south_star1.policy import DirectionMark
from grimace._south_star1.policy import TetraToken
from grimace._south_star1.residual_constraints import DirectionalCarrierResidual
from grimace._south_star1.residual_constraints import DirectionalResidualFactor
from grimace._south_star1.residual_constraints import DirectionalResidualFactorValueSnapshot
from grimace._south_star1.residual_constraints import ResidualConstraintComponentSnapshot
from grimace._south_star1.residual_constraints import ResidualStore
from grimace._south_star1.residual_constraints import ResidualStoreValueSnapshot
from grimace._south_star1.residual_constraints import TetraResidualFactor
from grimace._south_star1.residual_constraints import TetraResidualFactorValueSnapshot
from grimace._south_star1.residual_constraints import VarId
from grimace._south_star1.residual_constraints import add_factor_checked
from grimace._south_star1.residual_constraints import direction_var
from grimace._south_star1.residual_constraints import residual_store_assignments_have_support
from grimace._south_star1.residual_constraints import residual_store_constraint_components
from grimace._south_star1.residual_constraints import residual_store_projected_values
from grimace._south_star1.residual_constraints import tetra_var


@dataclass(frozen=True, slots=True)
class _DummyFactorSnapshot:
    scope: tuple[VarId, ...]


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

    def test_residual_snapshot_rejects_duplicate_domains(self) -> None:
        var = tetra_var(("center", 0))
        snapshot = ResidualStoreValueSnapshot(
            domains=(
                (var, (TetraToken.AT,)),
                (var, (TetraToken.ATAT,)),
            ),
            assignments=(),
            factors=(),
        )

        with self.assertRaises(ValueError):
            ResidualStore.from_value_snapshot(snapshot)
        with self.assertRaises(ValueError):
            residual_store_constraint_components(snapshot)
        with self.assertRaises(ValueError):
            residual_store_projected_values(snapshot, var)
        with self.assertRaises(ValueError):
            residual_store_assignments_have_support(snapshot, ())

    def test_residual_snapshot_rejects_empty_domain(self) -> None:
        var = tetra_var(("center", 0))
        snapshot = ResidualStoreValueSnapshot(
            domains=((var, ()),),
            assignments=(),
            factors=(),
        )

        with self.assertRaises(ValueError):
            ResidualStore.from_value_snapshot(snapshot)
        with self.assertRaises(ValueError):
            residual_store_constraint_components(snapshot)
        with self.assertRaises(ValueError):
            residual_store_projected_values(snapshot, var)
        with self.assertRaises(ValueError):
            residual_store_assignments_have_support(snapshot, ())

    def test_residual_constraint_components_empty_snapshot(self) -> None:
        snapshot = ResidualStoreValueSnapshot(domains=(), assignments=(), factors=())

        self.assertEqual(residual_store_constraint_components(snapshot), ())

    def test_residual_constraint_components_include_isolated_variables(self) -> None:
        first = VarId("test", (1,))
        second = VarId("test", (2,))
        snapshot = ResidualStoreValueSnapshot(
            domains=((first, ("a",)), (second, ("b",))),
            assignments=(),
            factors=(),
        )

        self.assertEqual(
            residual_store_constraint_components(snapshot),
            (
                ResidualConstraintComponentSnapshot(
                    variables=(first,),
                    factor_indexes=(),
                    assigned_variables=(),
                ),
                ResidualConstraintComponentSnapshot(
                    variables=(second,),
                    factor_indexes=(),
                    assigned_variables=(),
                ),
            ),
        )

    def test_residual_constraint_components_merge_factor_scopes(self) -> None:
        first = VarId("test", (1,))
        second = VarId("test", (2,))
        third = VarId("test", (3,))
        snapshot = ResidualStoreValueSnapshot(
            domains=((first, ("a",)), (second, ("b",)), (third, ("c",))),
            assignments=((second, "b"),),
            factors=(
                _DummyFactorSnapshot(scope=(first, second)),
                _DummyFactorSnapshot(scope=(third,)),
            ),
        )

        self.assertEqual(
            residual_store_constraint_components(snapshot),
            (
                ResidualConstraintComponentSnapshot(
                    variables=(first, second),
                    factor_indexes=(0,),
                    assigned_variables=(second,),
                ),
                ResidualConstraintComponentSnapshot(
                    variables=(third,),
                    factor_indexes=(1,),
                    assigned_variables=(),
                ),
            ),
        )

    def test_residual_constraint_components_reject_unknown_variables(self) -> None:
        known = VarId("test", (1,))
        unknown = VarId("test", (2,))
        factor_snapshot = ResidualStoreValueSnapshot(
            domains=((known, ("a",)),),
            assignments=(),
            factors=(_DummyFactorSnapshot(scope=(known, unknown)),),
        )
        assignment_snapshot = ResidualStoreValueSnapshot(
            domains=((known, ("a",)),),
            assignments=((unknown, "b"),),
            factors=(),
        )

        with self.assertRaises(ValueError):
            residual_store_constraint_components(factor_snapshot)
        with self.assertRaises(ValueError):
            residual_store_constraint_components(assignment_snapshot)

    def test_residual_projected_values_isolated_variable_returns_domain(self) -> None:
        var = tetra_var(("test", 0))
        snapshot = ResidualStoreValueSnapshot(
            domains=((var, (TetraToken.AT, TetraToken.ATAT)),),
            assignments=(),
            factors=(),
        )

        self.assertEqual(
            residual_store_projected_values(snapshot, var),
            (TetraToken.AT, TetraToken.ATAT),
        )

    def test_residual_projected_values_assigned_variable_returns_assignment(self) -> None:
        var = tetra_var(("test", 0))
        snapshot = ResidualStoreValueSnapshot(
            domains=((var, (TetraToken.AT, TetraToken.ATAT)),),
            assignments=((var, TetraToken.AT),),
            factors=(),
        )

        self.assertEqual(
            residual_store_projected_values(snapshot, var),
            (TetraToken.AT,),
        )

    def test_residual_projected_values_filters_unary_tetra_factor(self) -> None:
        store = ResidualStore()
        var = tetra_var(("test", 0))
        store.add_var(var, (TetraToken.AT, TetraToken.ATAT))
        factor = TetraResidualFactor(
            scope=(var,),
            status=SiteStatus.SPECIFIED,
            target=TetraValue.PLUS,
            reference_order=_occurrences(0, 1, 2, 3),
            local_order=_occurrences(0, 1, 2, 3),
        )

        self.assertTrue(add_factor_checked(store, factor))
        self.assertEqual(
            residual_store_projected_values(store.value_snapshot(), var),
            (TetraToken.AT,),
        )

    def test_residual_projected_values_detects_no_coupled_directional_completion(self) -> None:
        store = ResidualStore()
        left = direction_var(("left", 0))
        right = direction_var(("right", 0))
        store.add_var(left, (DirectionMark.FWD,))
        store.add_var(right, (DirectionMark.ABSENT,))
        factor = DirectionalResidualFactor(
            scope=(left, right),
            status=SiteStatus.SPECIFIED,
            target=DirectionalValue.OPPOSITE,
            carrier_models={
                left: DirectionalCarrierResidual(left, "left", 1, 1),
                right: DirectionalCarrierResidual(right, "right", 1, 1),
            },
        )

        self.assertTrue(add_factor_checked(store, factor))
        self.assertTrue(store.assign(left, DirectionMark.FWD))
        self.assertEqual(
            residual_store_projected_values(store.value_snapshot(), left),
            (),
        )

    def test_residual_projected_values_rejects_unknown_variable(self) -> None:
        snapshot = ResidualStoreValueSnapshot(domains=(), assignments=(), factors=())

        with self.assertRaises(ValueError):
            residual_store_projected_values(snapshot, tetra_var(("missing", 0)))

    def test_residual_assignment_support_empty_snapshot(self) -> None:
        snapshot = ResidualStoreValueSnapshot(domains=(), assignments=(), factors=())

        self.assertTrue(residual_store_assignments_have_support(snapshot, ()))

    def test_residual_assignment_support_rejects_unknown_variable(self) -> None:
        snapshot = ResidualStoreValueSnapshot(domains=(), assignments=(), factors=())

        with self.assertRaises(ValueError):
            residual_store_assignments_have_support(
                snapshot,
                ((tetra_var(("missing", 0)), TetraToken.AT),),
            )

    def test_residual_assignment_support_rejects_out_of_domain_value(self) -> None:
        var = tetra_var(("test", 0))
        snapshot = ResidualStoreValueSnapshot(
            domains=((var, (TetraToken.AT,)),),
            assignments=(),
            factors=(),
        )

        self.assertFalse(
            residual_store_assignments_have_support(
                snapshot,
                ((var, TetraToken.ATAT),),
            )
        )

    def test_residual_assignment_support_handles_duplicate_assignments(self) -> None:
        var = tetra_var(("test", 0))
        snapshot = ResidualStoreValueSnapshot(
            domains=((var, (TetraToken.AT, TetraToken.ATAT)),),
            assignments=(),
            factors=(),
        )

        self.assertTrue(
            residual_store_assignments_have_support(
                snapshot,
                ((var, TetraToken.AT), (var, TetraToken.AT)),
            )
        )
        self.assertFalse(
            residual_store_assignments_have_support(
                snapshot,
                ((var, TetraToken.AT), (var, TetraToken.ATAT)),
            )
        )

    def test_residual_assignment_support_rejects_existing_assignment_conflict(self) -> None:
        var = tetra_var(("test", 0))
        snapshot = ResidualStoreValueSnapshot(
            domains=((var, (TetraToken.AT, TetraToken.ATAT)),),
            assignments=((var, TetraToken.AT),),
            factors=(),
        )

        self.assertFalse(
            residual_store_assignments_have_support(
                snapshot,
                ((var, TetraToken.ATAT),),
            )
        )

    def test_residual_assignment_support_filters_unary_tetra_factor(self) -> None:
        store = ResidualStore()
        var = tetra_var(("test", 0))
        store.add_var(var, (TetraToken.AT, TetraToken.ATAT))
        self.assertTrue(
            add_factor_checked(
                store,
                TetraResidualFactor(
                    scope=(var,),
                    status=SiteStatus.SPECIFIED,
                    target=TetraValue.PLUS,
                    reference_order=_occurrences(0, 1, 2, 3),
                    local_order=_occurrences(0, 1, 2, 3),
                ),
            )
        )

        self.assertTrue(
            residual_store_assignments_have_support(
                store.value_snapshot(),
                ((var, TetraToken.AT),),
            )
        )
        self.assertFalse(
            residual_store_assignments_have_support(
                store.value_snapshot(),
                ((var, TetraToken.ATAT),),
            )
        )

    def test_residual_assignment_support_detects_no_coupled_directional_completion(self) -> None:
        store = ResidualStore()
        left = direction_var(("left", 0))
        right = direction_var(("right", 0))
        store.add_var(left, (DirectionMark.FWD,))
        store.add_var(right, (DirectionMark.ABSENT,))
        factor = DirectionalResidualFactor(
            scope=(left, right),
            status=SiteStatus.SPECIFIED,
            target=DirectionalValue.OPPOSITE,
            carrier_models={
                left: DirectionalCarrierResidual(left, "left", 1, 1),
                right: DirectionalCarrierResidual(right, "right", 1, 1),
            },
        )

        self.assertTrue(add_factor_checked(store, factor))
        self.assertFalse(
            residual_store_assignments_have_support(
                store.value_snapshot(),
                ((left, DirectionMark.FWD),),
            )
        )

    def test_residual_assignment_support_conjoins_independent_components(self) -> None:
        store = ResidualStore()
        tetra = tetra_var(("test", 0))
        direction = direction_var(("direction", 0))
        store.add_var(tetra, (TetraToken.AT, TetraToken.ATAT))
        store.add_var(direction, (DirectionMark.FWD, DirectionMark.REV))
        self.assertTrue(
            add_factor_checked(
                store,
                TetraResidualFactor(
                    scope=(tetra,),
                    status=SiteStatus.SPECIFIED,
                    target=TetraValue.PLUS,
                    reference_order=_occurrences(0, 1, 2, 3),
                    local_order=_occurrences(0, 1, 2, 3),
                ),
            )
        )
        snapshot = store.value_snapshot()

        self.assertTrue(
            residual_store_assignments_have_support(
                snapshot,
                ((tetra, TetraToken.AT), (direction, DirectionMark.REV)),
            )
        )
        self.assertFalse(
            residual_store_assignments_have_support(
                snapshot,
                ((tetra, TetraToken.ATAT), (direction, DirectionMark.REV)),
            )
        )

    def test_assigned_tetra_residual_snapshot_round_trips(self) -> None:
        store = ResidualStore()
        var = tetra_var(("test", 0))
        store.add_var(var, (TetraToken.AT, TetraToken.ATAT))
        self.assertTrue(
            add_factor_checked(
                store,
                TetraResidualFactor(
                    scope=(var,),
                    status=SiteStatus.SPECIFIED,
                    target=TetraValue.PLUS,
                    reference_order=_occurrences(0, 1, 2, 3),
                    local_order=_occurrences(0, 1, 2, 3),
                ),
            )
        )
        self.assertTrue(store.assign(var, TetraToken.AT))
        snapshot = store.value_snapshot()

        restored = ResidualStore.from_value_snapshot(snapshot)

        self.assertEqual(restored.value_snapshot(), snapshot)

    def test_tetra_factor_assignment_missing_from_top_level_rejects(self) -> None:
        var = tetra_var(("test", 0))
        snapshot = ResidualStoreValueSnapshot(
            domains=((var, (TetraToken.AT, TetraToken.ATAT)),),
            assignments=(),
            factors=(
                TetraResidualFactorValueSnapshot(
                    scope=(var,),
                    status=SiteStatus.SPECIFIED,
                    target=TetraValue.PLUS,
                    reference_order=_occurrences(0, 1, 2, 3),
                    local_order=_occurrences(0, 1, 2, 3),
                    assigned=TetraToken.AT,
                ),
            ),
        )

        with self.assertRaises(ValueError):
            ResidualStore.from_value_snapshot(snapshot)
        with self.assertRaises(ValueError):
            residual_store_assignments_have_support(snapshot, ())

    def test_top_level_tetra_assignment_missing_from_factor_rejects(self) -> None:
        store = ResidualStore()
        var = tetra_var(("test", 0))
        store.add_var(var, (TetraToken.AT, TetraToken.ATAT))
        self.assertTrue(
            add_factor_checked(
                store,
                TetraResidualFactor(
                    scope=(var,),
                    status=SiteStatus.SPECIFIED,
                    target=TetraValue.PLUS,
                    reference_order=_occurrences(0, 1, 2, 3),
                    local_order=_occurrences(0, 1, 2, 3),
                ),
            )
        )
        snapshot = store.value_snapshot()
        tampered = replace(
            snapshot,
            assignments=((var, TetraToken.AT),),
        )

        with self.assertRaises(ValueError):
            ResidualStore.from_value_snapshot(tampered)
        with self.assertRaises(ValueError):
            residual_store_assignments_have_support(tampered, ())

    def test_directional_factor_mark_disagreement_rejects(self) -> None:
        left = direction_var(("left", 0))
        right = direction_var(("right", 0))
        snapshot = ResidualStoreValueSnapshot(
            domains=(
                (left, (DirectionMark.ABSENT, DirectionMark.FWD, DirectionMark.REV)),
                (right, (DirectionMark.ABSENT, DirectionMark.FWD, DirectionMark.REV)),
            ),
            assignments=((left, DirectionMark.FWD),),
            factors=(
                DirectionalResidualFactorValueSnapshot(
                    scope=(left, right),
                    status=SiteStatus.SPECIFIED,
                    target=DirectionalValue.OPPOSITE,
                    carrier_models=(
                        (
                            left,
                            DirectionalCarrierResidual(left, "left", 1, 1),
                        ),
                        (
                            right,
                            DirectionalCarrierResidual(right, "right", 1, 1),
                        ),
                    ),
                    marks=((left, DirectionMark.REV),),
                ),
            ),
        )

        with self.assertRaises(ValueError):
            ResidualStore.from_value_snapshot(snapshot)
        with self.assertRaises(ValueError):
            residual_store_assignments_have_support(snapshot, ())


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
