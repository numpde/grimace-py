"""Tests for reversible online residual constraints."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import inspect
import unittest

import grimace._south_star1.residual_constraints as residual_constraints_module
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
from grimace._south_star1.residual_constraints import ResidualFactor
from grimace._south_star1.residual_constraints import ResidualFactorKey
from grimace._south_star1.residual_constraints import ResidualPropagationKind
from grimace._south_star1.residual_constraints import ResidualStore
from grimace._south_star1.residual_constraints import ResidualStoreValueSnapshot
from grimace._south_star1.residual_constraints import TetraResidualFactor
from grimace._south_star1.residual_constraints import TetraResidualFactorValueSnapshot
from grimace._south_star1.residual_constraints import VarId
from grimace._south_star1.residual_constraints import add_factor_and_propagate
from grimace._south_star1.residual_constraints import add_factors_and_propagate
from grimace._south_star1.residual_constraints import direction_var
from grimace._south_star1.residual_constraints import residual_store_constraint_components
from grimace._south_star1.residual_constraints import residual_store_projected_values
from grimace._south_star1.residual_constraints import tetra_var


@dataclass(frozen=True, slots=True)
class _DummyFactorSnapshot:
    scope: tuple[VarId, ...]
    key: ResidualFactorKey = ResidualFactorKey("dummy", ())


@dataclass(frozen=True, slots=True)
class _AcceptAllFactor:
    key: ResidualFactorKey
    scope: tuple[VarId, ...]

    def accepts(self, row: tuple[object, ...]) -> bool:
        return len(row) == len(self.scope)

    def value_snapshot(self) -> _DummyFactorSnapshot:
        return _DummyFactorSnapshot(scope=self.scope, key=self.key)


class ResidualConstraintTest(unittest.TestCase):
    def test_tetra_factor_relation_accepts_only_required_token(self) -> None:
        factor = _tetra_factor(
            target=TetraValue.PLUS,
            local_order=(0, 1, 2, 3),
        )

        self.assertEqual(factor.allowed_tokens(), frozenset((TetraToken.AT,)))
        self.assertTrue(factor.accepts((TetraToken.AT,)))
        self.assertFalse(factor.accepts((TetraToken.ATAT,)))

    def test_tetra_unspecified_relation_accepts_none_only(self) -> None:
        factor = TetraResidualFactor(
            scope=(tetra_var(0),),
            status=SiteStatus.UNSPECIFIED,
            target=TetraValue.NONE,
            reference_order=_occurrences(0, 1, 2, 3),
            local_order=_occurrences(0, 1, 2, 3),
        )

        self.assertTrue(factor.accepts((TetraToken.NONE,)))
        self.assertFalse(factor.accepts((TetraToken.AT,)))

    def test_directional_factor_relation_accepts_exact_pair(self) -> None:
        factor = _directional_factor(DirectionalValue.OPPOSITE)

        self.assertTrue(
            factor.accepts((DirectionMark.FWD, DirectionMark.REV))
        )
        self.assertFalse(
            factor.accepts((DirectionMark.FWD, DirectionMark.FWD))
        )

    def test_directional_relation_rejects_same_endpoint_inconsistent_signs(
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

        self.assertFalse(
            factor.accepts(
                (
                    DirectionMark.FWD,
                    DirectionMark.REV,
                    DirectionMark.FWD,
                )
            )
        )

    def test_tetra_factor_propagation_reduces_domain(self) -> None:
        store = ResidualStore()
        var = tetra_var(("test", 0))
        store.add_var(var, (TetraToken.AT, TetraToken.ATAT))

        result = add_factor_and_propagate(
            store,
            TetraResidualFactor(
                scope=(var,),
                status=SiteStatus.SPECIFIED,
                target=TetraValue.PLUS,
                reference_order=_occurrences(0, 1, 2, 3),
                local_order=_occurrences(0, 1, 2, 3),
            ),
        )

        self.assertIs(result.kind, ResidualPropagationKind.CERTIFIED_CONSISTENT)
        self.assertEqual(store.domain(var), (TetraToken.AT,))
        self.assertIsNone(store.assignment(var))

    def test_directional_propagation_reduces_coupled_domain(self) -> None:
        store = ResidualStore()
        left = direction_var(("left", 0))
        right = direction_var(("right", 0))
        domain = (DirectionMark.FWD, DirectionMark.REV)
        store.add_var(left, domain)
        store.add_var(right, domain)
        self.assertIs(
            add_factor_and_propagate(
                store,
                _directional_factor_between(
                    left,
                    right,
                    DirectionalValue.TOGETHER,
                ),
            ).kind,
            ResidualPropagationKind.CERTIFIED_CONSISTENT,
        )

        result = store.restrict_to_value(left, DirectionMark.FWD)

        self.assertIs(result.kind, ResidualPropagationKind.CERTIFIED_CONSISTENT)
        self.assertEqual(store.domain(left), (DirectionMark.FWD,))
        self.assertEqual(store.domain(right), (DirectionMark.FWD,))
        self.assertIs(store.assignment(left), DirectionMark.FWD)
        self.assertIsNone(store.assignment(right))

    def test_domain_intersection_propagates_without_assignment(self) -> None:
        store = ResidualStore()
        left = direction_var(("left", 0))
        right = direction_var(("right", 0))
        domain = (DirectionMark.FWD, DirectionMark.REV)
        store.add_var(left, domain)
        store.add_var(right, domain)
        self.assertIs(
            add_factor_and_propagate(
                store,
                _directional_factor_between(
                    left,
                    right,
                    DirectionalValue.OPPOSITE,
                ),
            ).kind,
            ResidualPropagationKind.CERTIFIED_CONSISTENT,
        )

        result = store.intersect_domain_and_propagate(
            left,
            (DirectionMark.FWD,),
        )

        self.assertIs(result.kind, ResidualPropagationKind.CERTIFIED_CONSISTENT)
        self.assertEqual(store.domain(left), (DirectionMark.FWD,))
        self.assertEqual(store.domain(right), (DirectionMark.REV,))
        self.assertIsNone(store.assignment(left))
        self.assertIsNone(store.assignment(right))

    def test_propagation_through_two_shared_factors(self) -> None:
        store = ResidualStore()
        a = direction_var(("a",))
        b = direction_var(("b",))
        c = direction_var(("c",))
        domain = (DirectionMark.FWD, DirectionMark.REV)
        for var in (a, b, c):
            store.add_var(var, domain)
        self.assertIs(
            add_factor_and_propagate(
                store,
                _directional_factor_between(a, b, DirectionalValue.TOGETHER),
            ).kind,
            ResidualPropagationKind.CERTIFIED_CONSISTENT,
        )
        self.assertIs(
            add_factor_and_propagate(
                store,
                _directional_factor_between(b, c, DirectionalValue.OPPOSITE),
            ).kind,
            ResidualPropagationKind.CERTIFIED_CONSISTENT,
        )

        result = store.restrict_to_value(a, DirectionMark.FWD)

        self.assertIs(result.kind, ResidualPropagationKind.CERTIFIED_CONSISTENT)
        self.assertEqual(store.domain(a), (DirectionMark.FWD,))
        self.assertEqual(store.domain(b), (DirectionMark.FWD,))
        self.assertEqual(store.domain(c), (DirectionMark.REV,))
        self.assertIs(store.assignment(a), DirectionMark.FWD)
        self.assertIsNone(store.assignment(b))
        self.assertIsNone(store.assignment(c))

    def test_component_locality(self) -> None:
        store = ResidualStore()
        domain = (DirectionMark.FWD, DirectionMark.REV)
        pairs: list[tuple[VarId, VarId]] = []
        for index in range(20):
            left = direction_var(("left", index))
            right = direction_var(("right", index))
            pairs.append((left, right))
            store.add_var(left, domain)
            store.add_var(right, domain)
            result = add_factor_and_propagate(
                store,
                _directional_factor_between(
                    left,
                    right,
                    DirectionalValue.TOGETHER,
                ),
            )
            self.assertIs(result.kind, ResidualPropagationKind.CERTIFIED_CONSISTENT)

        result = store.restrict_to_value(pairs[7][0], DirectionMark.FWD)

        self.assertIs(result.kind, ResidualPropagationKind.CERTIFIED_CONSISTENT)
        self.assertEqual(
            result.stats.component_factor_keys,
            (
                _directional_factor_between(
                    pairs[7][0],
                    pairs[7][1],
                    DirectionalValue.TOGETHER,
                ).key,
            ),
        )
        self.assertEqual(len(result.stats.component_variables), 2)
        for index, (left, right) in enumerate(pairs):
            if index == 7:
                continue
            self.assertEqual(store.domain(left), domain)
            self.assertEqual(store.domain(right), domain)

    def test_bulk_factor_addition_preserves_independent_tetra_components(self) -> None:
        store = ResidualStore()
        left = tetra_var(("left",))
        right = tetra_var(("right",))
        domain = (TetraToken.AT, TetraToken.ATAT)
        store.add_var(left, domain)
        store.add_var(right, domain)

        result = add_factors_and_propagate(
            store,
            (
                _tetra_factor_for_var(left, target=TetraValue.PLUS),
                _tetra_factor_for_var(right, target=TetraValue.PLUS),
            ),
        )

        self.assertIs(result.kind, ResidualPropagationKind.CERTIFIED_CONSISTENT)
        self.assertEqual(store.domain(left), (TetraToken.AT,))
        self.assertEqual(store.domain(right), (TetraToken.AT,))
        self.assertEqual(len(result.stats.component_variables), 2)
        self.assertEqual(len(result.stats.component_factor_keys), 2)

    def test_bulk_factor_addition_preserves_tetra_and_directional_components(self) -> None:
        store = ResidualStore()
        tetra = tetra_var(("tetra",))
        left = direction_var(("left",))
        right = direction_var(("right",))
        store.add_var(tetra, (TetraToken.AT, TetraToken.ATAT))
        for var in (left, right):
            store.add_var(var, (DirectionMark.FWD, DirectionMark.REV))

        result = add_factors_and_propagate(
            store,
            (
                _tetra_factor_for_var(tetra, target=TetraValue.PLUS),
                _directional_factor_between(
                    left,
                    right,
                    DirectionalValue.TOGETHER,
                ),
            ),
        )

        self.assertIs(result.kind, ResidualPropagationKind.CERTIFIED_CONSISTENT)
        self.assertEqual(store.domain(tetra), (TetraToken.AT,))
        self.assertEqual(store.domain(left), (DirectionMark.FWD, DirectionMark.REV))
        self.assertEqual(store.domain(right), (DirectionMark.FWD, DirectionMark.REV))

    def test_atomic_restrictions_preserve_independent_components(self) -> None:
        store = ResidualStore()
        first_left = direction_var(("first-left",))
        first_right = direction_var(("first-right",))
        second_left = direction_var(("second-left",))
        second_right = direction_var(("second-right",))
        domain = (DirectionMark.FWD, DirectionMark.REV)
        for var in (first_left, first_right, second_left, second_right):
            store.add_var(var, domain)
        self.assertIs(
            add_factors_and_propagate(
                store,
                (
                    _directional_factor_between(
                        first_left,
                        first_right,
                        DirectionalValue.TOGETHER,
                    ),
                    _directional_factor_between(
                        second_left,
                        second_right,
                        DirectionalValue.OPPOSITE,
                    ),
                ),
            ).kind,
            ResidualPropagationKind.CERTIFIED_CONSISTENT,
        )

        result = store.restrict_many_and_propagate(
            (
                (first_left, DirectionMark.FWD),
                (second_left, DirectionMark.FWD),
            )
        )

        self.assertIs(result.kind, ResidualPropagationKind.CERTIFIED_CONSISTENT)
        self.assertEqual(store.domain(first_right), (DirectionMark.FWD,))
        self.assertEqual(store.domain(second_right), (DirectionMark.REV,))

    def test_bulk_factor_addition_rolls_back_when_one_component_contradicts(self) -> None:
        store = ResidualStore()
        good = tetra_var(("good",))
        left = direction_var(("left",))
        right = direction_var(("right",))
        store.add_var(good, (TetraToken.AT, TetraToken.ATAT))
        store.add_var(left, (DirectionMark.FWD,))
        store.add_var(right, (DirectionMark.ABSENT,))
        before = store.value_snapshot()

        result = add_factors_and_propagate(
            store,
            (
                _tetra_factor_for_var(good, target=TetraValue.PLUS),
                _directional_factor_between(
                    left,
                    right,
                    DirectionalValue.OPPOSITE,
                ),
            ),
        )

        self.assertIs(result.kind, ResidualPropagationKind.CONTRADICTION)
        self.assertEqual(store.value_snapshot(), before)

    def test_propagate_all_components_reports_contradiction_before_uncertified(self) -> None:
        store = ResidualStore()
        uncertain_vars = tuple(VarId("wide", (index,)) for index in range(5))
        for var in uncertain_vars:
            store.add_var(var, ("x", "y"))
        store.add_factor(
            _AcceptAllFactor(
                key=ResidualFactorKey("wide", ()),
                scope=uncertain_vars,
            )
        )

        left = direction_var(("left",))
        right = direction_var(("right",))
        store.add_var(left, (DirectionMark.FWD,))
        store.add_var(right, (DirectionMark.ABSENT,))
        store.add_factor(
            _directional_factor_between(
                left,
                right,
                DirectionalValue.OPPOSITE,
            )
        )

        self.assertIs(
            store.propagate_all_components().kind,
            ResidualPropagationKind.CONTRADICTION,
        )

    def test_factor_addition_rolls_back_on_contradiction(self) -> None:
        store = ResidualStore()
        left = direction_var(("left",))
        right = direction_var(("right",))
        store.add_var(left, (DirectionMark.FWD,))
        store.add_var(right, (DirectionMark.ABSENT,))
        before = store.value_snapshot()

        result = add_factor_and_propagate(
            store,
            _directional_factor_between(left, right, DirectionalValue.OPPOSITE),
        )

        self.assertIs(result.kind, ResidualPropagationKind.CONTRADICTION)
        self.assertEqual(store.value_snapshot(), before)

    def test_value_restriction_rolls_back_on_contradiction(self) -> None:
        store = ResidualStore()
        left = direction_var(("left",))
        right = direction_var(("right",))
        domain = (DirectionMark.FWD, DirectionMark.REV)
        store.add_var(left, domain)
        store.add_var(right, domain)
        self.assertIs(
            add_factor_and_propagate(
                store,
                _directional_factor_between(
                    left,
                    right,
                    DirectionalValue.TOGETHER,
                ),
            ).kind,
            ResidualPropagationKind.CERTIFIED_CONSISTENT,
        )
        before = store.value_snapshot()

        result = store.restrict_to_value(left, DirectionMark.ABSENT)

        self.assertIs(result.kind, ResidualPropagationKind.CONTRADICTION)
        self.assertEqual(store.value_snapshot(), before)

    def test_unresolved_cyclic_components_fail_closed(self) -> None:
        store = ResidualStore()
        a = direction_var(("a",))
        b = direction_var(("b",))
        c = direction_var(("c",))
        domain = (DirectionMark.FWD, DirectionMark.REV)
        for var in (a, b, c):
            store.add_var(var, domain)

        for left, right in ((a, b), (b, c)):
            self.assertIs(
                add_factor_and_propagate(
                    store,
                    _directional_factor_between(
                        left,
                        right,
                        DirectionalValue.TOGETHER,
                    ),
                ).kind,
                ResidualPropagationKind.CERTIFIED_CONSISTENT,
            )
        before = store.value_snapshot()

        result = add_factor_and_propagate(
            store,
            _directional_factor_between(c, a, DirectionalValue.TOGETHER),
        )

        self.assertIs(
            result.kind,
            ResidualPropagationKind.LOCALLY_CONSISTENT_UNCERTIFIED,
        )
        self.assertNotEqual(store.value_snapshot(), before)

    def test_singleton_cyclic_component_is_accepted(self) -> None:
        store = ResidualStore()
        a = direction_var(("a",))
        b = direction_var(("b",))
        c = direction_var(("c",))
        for var in (a, b, c):
            store.add_var(var, (DirectionMark.FWD,))

        for left, right in ((a, b), (b, c), (c, a)):
            result = add_factor_and_propagate(
                store,
                _directional_factor_between(
                    left,
                    right,
                    DirectionalValue.TOGETHER,
                ),
            )
            self.assertIs(result.kind, ResidualPropagationKind.CERTIFIED_CONSISTENT)

    def test_residual_store_value_snapshot_is_canonical_by_var_order(self) -> None:
        left = ResidualStore()
        right = ResidualStore()
        first = VarId("test", (1,))
        second = VarId("test", (2,))

        for var in (second, first):
            left.add_var(var, ("a", "b"))
        for var in (first, second):
            right.add_var(var, ("a", "b"))
        self.assertIs(
            left.restrict_many_and_propagate(((first, "a"),)).kind,
            ResidualPropagationKind.CERTIFIED_CONSISTENT,
        )
        self.assertIs(
            left.restrict_many_and_propagate(((second, "b"),)).kind,
            ResidualPropagationKind.CERTIFIED_CONSISTENT,
        )
        self.assertIs(
            right.restrict_many_and_propagate(((second, "b"),)).kind,
            ResidualPropagationKind.CERTIFIED_CONSISTENT,
        )
        self.assertIs(
            right.restrict_many_and_propagate(((first, "a"),)).kind,
            ResidualPropagationKind.CERTIFIED_CONSISTENT,
        )

        self.assertEqual(left.value_snapshot(), right.value_snapshot())

    def test_residual_store_rejects_duplicate_domain_values(self) -> None:
        var = tetra_var(("center", 0))
        store = ResidualStore()

        with self.assertRaises(ValueError):
            store.add_var(var, (TetraToken.AT, TetraToken.AT))

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
            _snapshot_accepts_restrictions(snapshot, ())

    def test_residual_snapshot_rejects_duplicate_domain_values(self) -> None:
        var = tetra_var(("center", 0))
        snapshot = ResidualStoreValueSnapshot(
            domains=((var, (TetraToken.AT, TetraToken.AT)),),
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
            _snapshot_accepts_restrictions(snapshot, ())

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
            _snapshot_accepts_restrictions(snapshot, ())

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
                    factor_keys=(),
                    assigned_variables=(),
                ),
                ResidualConstraintComponentSnapshot(
                    variables=(second,),
                    factor_keys=(),
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
                _DummyFactorSnapshot(
                    scope=(first, second),
                    key=ResidualFactorKey("dummy", (0,)),
                ),
                _DummyFactorSnapshot(
                    scope=(third,),
                    key=ResidualFactorKey("dummy", (1,)),
                ),
            ),
        )

        self.assertEqual(
            residual_store_constraint_components(snapshot),
            (
                ResidualConstraintComponentSnapshot(
                    variables=(first, second),
                    factor_keys=(ResidualFactorKey("dummy", (0,)),),
                    assigned_variables=(second,),
                ),
                ResidualConstraintComponentSnapshot(
                    variables=(third,),
                    factor_keys=(ResidualFactorKey("dummy", (1,)),),
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
            domains=((var, (TetraToken.AT,)),),
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
        result = add_factor_and_propagate(
            store,
            TetraResidualFactor(
                scope=(var,),
                status=SiteStatus.SPECIFIED,
                target=TetraValue.PLUS,
                reference_order=_occurrences(0, 1, 2, 3),
                local_order=_occurrences(0, 1, 2, 3),
            ),
        )

        self.assertIs(result.kind, ResidualPropagationKind.CERTIFIED_CONSISTENT)
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
        result = add_factor_and_propagate(
            store,
            _directional_factor_between(left, right, DirectionalValue.OPPOSITE),
        )

        self.assertIs(result.kind, ResidualPropagationKind.CONTRADICTION)
        self.assertEqual(
            residual_store_projected_values(store.value_snapshot(), left),
            (DirectionMark.FWD,),
        )

    def test_residual_projected_values_rejects_unknown_variable(self) -> None:
        snapshot = ResidualStoreValueSnapshot(domains=(), assignments=(), factors=())

        with self.assertRaises(ValueError):
            residual_store_projected_values(snapshot, tetra_var(("missing", 0)))

    def test_residual_assignment_support_empty_snapshot(self) -> None:
        snapshot = ResidualStoreValueSnapshot(domains=(), assignments=(), factors=())

        self.assertTrue(_snapshot_accepts_restrictions(snapshot, ()))

    def test_residual_assignment_support_rejects_unknown_variable(self) -> None:
        snapshot = ResidualStoreValueSnapshot(domains=(), assignments=(), factors=())

        with self.assertRaises(ValueError):
            _snapshot_accepts_restrictions(
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
            _snapshot_accepts_restrictions(
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
            _snapshot_accepts_restrictions(
                snapshot,
                ((var, TetraToken.AT), (var, TetraToken.AT)),
            )
        )
        self.assertFalse(
            _snapshot_accepts_restrictions(
                snapshot,
                ((var, TetraToken.AT), (var, TetraToken.ATAT)),
            )
        )

    def test_residual_assignment_support_rejects_existing_assignment_conflict(self) -> None:
        var = tetra_var(("test", 0))
        snapshot = ResidualStoreValueSnapshot(
            domains=((var, (TetraToken.AT,)),),
            assignments=((var, TetraToken.AT),),
            factors=(),
        )

        self.assertFalse(
            _snapshot_accepts_restrictions(
                snapshot,
                ((var, TetraToken.ATAT),),
            )
        )

    def test_residual_assignment_support_filters_unary_tetra_factor(self) -> None:
        store = ResidualStore()
        var = tetra_var(("test", 0))
        store.add_var(var, (TetraToken.AT, TetraToken.ATAT))
        self.assertIs(
            add_factor_and_propagate(
                store,
                TetraResidualFactor(
                    scope=(var,),
                    status=SiteStatus.SPECIFIED,
                    target=TetraValue.PLUS,
                    reference_order=_occurrences(0, 1, 2, 3),
                    local_order=_occurrences(0, 1, 2, 3),
                ),
            ).kind,
            ResidualPropagationKind.CERTIFIED_CONSISTENT,
        )

        self.assertTrue(
            _snapshot_accepts_restrictions(
                store.value_snapshot(),
                ((var, TetraToken.AT),),
            )
        )
        self.assertFalse(
            _snapshot_accepts_restrictions(
                store.value_snapshot(),
                ((var, TetraToken.ATAT),),
            )
        )

    def test_residual_assignment_support_conjoins_independent_components(self) -> None:
        store = ResidualStore()
        tetra = tetra_var(("test", 0))
        direction = direction_var(("direction", 0))
        store.add_var(tetra, (TetraToken.AT, TetraToken.ATAT))
        store.add_var(direction, (DirectionMark.FWD, DirectionMark.REV))
        self.assertIs(
            add_factor_and_propagate(
                store,
                TetraResidualFactor(
                    scope=(tetra,),
                    status=SiteStatus.SPECIFIED,
                    target=TetraValue.PLUS,
                    reference_order=_occurrences(0, 1, 2, 3),
                    local_order=_occurrences(0, 1, 2, 3),
                ),
            ).kind,
            ResidualPropagationKind.CERTIFIED_CONSISTENT,
        )
        snapshot = store.value_snapshot()

        self.assertTrue(
            _snapshot_accepts_restrictions(
                snapshot,
                ((tetra, TetraToken.AT), (direction, DirectionMark.REV)),
            )
        )
        self.assertFalse(
            _snapshot_accepts_restrictions(
                snapshot,
                ((tetra, TetraToken.ATAT), (direction, DirectionMark.REV)),
            )
        )

    def test_assigned_tetra_residual_snapshot_round_trips(self) -> None:
        store = ResidualStore()
        var = tetra_var(("test", 0))
        store.add_var(var, (TetraToken.AT, TetraToken.ATAT))
        self.assertIs(
            add_factor_and_propagate(
                store,
                TetraResidualFactor(
                    scope=(var,),
                    status=SiteStatus.SPECIFIED,
                    target=TetraValue.PLUS,
                    reference_order=_occurrences(0, 1, 2, 3),
                    local_order=_occurrences(0, 1, 2, 3),
                ),
            ).kind,
            ResidualPropagationKind.CERTIFIED_CONSISTENT,
        )
        self.assertIs(
            store.restrict_to_value(var, TetraToken.AT).kind,
            ResidualPropagationKind.CERTIFIED_CONSISTENT,
        )
        snapshot = store.value_snapshot()

        restored = ResidualStore.from_value_snapshot(snapshot)

        self.assertEqual(restored.value_snapshot(), snapshot)

    def test_factor_snapshots_contain_no_assignment_or_marks(self) -> None:
        tetra_snapshot = TetraResidualFactor(
            scope=(tetra_var(("test", 0)),),
            status=SiteStatus.SPECIFIED,
            target=TetraValue.PLUS,
            reference_order=_occurrences(0, 1, 2, 3),
            local_order=_occurrences(0, 1, 2, 3),
        ).value_snapshot()
        directional_snapshot = _directional_factor(
            DirectionalValue.TOGETHER,
        ).value_snapshot()

        self.assertFalse(hasattr(tetra_snapshot, "assigned"))
        self.assertFalse(hasattr(directional_snapshot, "marks"))

    def test_explicit_assignments_require_singleton_domains(self) -> None:
        var = tetra_var(("test", 0))
        snapshot = ResidualStoreValueSnapshot(
            domains=((var, (TetraToken.AT, TetraToken.ATAT)),),
            assignments=((var, TetraToken.AT),),
            factors=(),
        )

        with self.assertRaises(ValueError):
            ResidualStore.from_value_snapshot(snapshot)

    def test_inferred_singleton_domain_is_not_assignment(self) -> None:
        store = ResidualStore()
        var = tetra_var(("test", 0))
        store.add_var(var, (TetraToken.AT, TetraToken.ATAT))

        self.assertIs(
            add_factor_and_propagate(
                store,
                TetraResidualFactor(
                    scope=(var,),
                    status=SiteStatus.SPECIFIED,
                    target=TetraValue.PLUS,
                    reference_order=_occurrences(0, 1, 2, 3),
                    local_order=_occurrences(0, 1, 2, 3),
                ),
            ).kind,
            ResidualPropagationKind.CERTIFIED_CONSISTENT,
        )

        snapshot = store.value_snapshot()
        self.assertEqual(snapshot.domains, ((var, (TetraToken.AT,)),))
        self.assertEqual(snapshot.assignments, ())

    def test_snapshot_reconstruction_rebuilds_factor_indexes(self) -> None:
        store = ResidualStore()
        left = direction_var(("left",))
        right = direction_var(("right",))
        for var in (left, right):
            store.add_var(var, (DirectionMark.FWD, DirectionMark.REV))
        self.assertIs(
            add_factor_and_propagate(
                store,
                _directional_factor_between(
                    left,
                    right,
                    DirectionalValue.TOGETHER,
                ),
            ).kind,
            ResidualPropagationKind.CERTIFIED_CONSISTENT,
        )
        snapshot = store.value_snapshot()
        restored = ResidualStore.from_value_snapshot(snapshot)

        result = restored.restrict_to_value(left, DirectionMark.FWD)

        self.assertIs(result.kind, ResidualPropagationKind.CERTIFIED_CONSISTENT)
        self.assertEqual(restored.domain(right), (DirectionMark.FWD,))

    def test_oracle_matches_unary_tetra_projection(self) -> None:
        store = ResidualStore()
        var = tetra_var(("test", 0))
        domain = (TetraToken.AT, TetraToken.ATAT)
        store.add_var(var, domain)
        factor = TetraResidualFactor(
            scope=(var,),
            status=SiteStatus.SPECIFIED,
            target=TetraValue.PLUS,
            reference_order=_occurrences(0, 1, 2, 3),
            local_order=_occurrences(0, 1, 2, 3),
        )
        self.assertIs(
            add_factor_and_propagate(store, factor).kind,
            ResidualPropagationKind.CERTIFIED_CONSISTENT,
        )

        self.assertEqual(
            store.domain(var),
            _oracle_projection(
                var,
                (var,),
                {var: domain},
                (factor,),
            ),
        )

    def test_oracle_matches_two_factor_chain_projection(self) -> None:
        store = ResidualStore()
        a = direction_var(("a",))
        b = direction_var(("b",))
        c = direction_var(("c",))
        domain = (DirectionMark.FWD, DirectionMark.REV)
        for var in (a, b, c):
            store.add_var(var, domain)
        factors = (
            _directional_factor_between(a, b, DirectionalValue.TOGETHER),
            _directional_factor_between(b, c, DirectionalValue.OPPOSITE),
        )
        for factor in factors:
            self.assertIs(
                add_factor_and_propagate(store, factor).kind,
                ResidualPropagationKind.CERTIFIED_CONSISTENT,
            )
        self.assertIs(
            store.restrict_to_value(a, DirectionMark.FWD).kind,
            ResidualPropagationKind.CERTIFIED_CONSISTENT,
        )
        domains = {
            a: (DirectionMark.FWD,),
            b: domain,
            c: domain,
        }

        for var in (a, b, c):
            self.assertEqual(
                store.domain(var),
                _oracle_projection(var, (a, b, c), domains, factors),
            )

    def test_oracle_matches_independent_components(self) -> None:
        store = ResidualStore()
        left = direction_var(("left",))
        right = direction_var(("right",))
        tetra = tetra_var(("tetra",))
        direction_domain = (DirectionMark.FWD, DirectionMark.REV)
        tetra_domain = (TetraToken.AT, TetraToken.ATAT)
        store.add_var(left, direction_domain)
        store.add_var(right, direction_domain)
        store.add_var(tetra, tetra_domain)
        factors = (
            _directional_factor_between(
                left,
                right,
                DirectionalValue.TOGETHER,
            ),
            TetraResidualFactor(
                scope=(tetra,),
                status=SiteStatus.SPECIFIED,
                target=TetraValue.PLUS,
                reference_order=_occurrences(0, 1, 2, 3),
                local_order=_occurrences(0, 1, 2, 3),
            ),
        )
        for factor in factors:
            self.assertIs(
                add_factor_and_propagate(store, factor).kind,
                ResidualPropagationKind.CERTIFIED_CONSISTENT,
            )

        self.assertEqual(
            store.domain(tetra),
            _oracle_projection(
                tetra,
                (tetra,),
                {tetra: tetra_domain},
                (factors[1],),
            ),
        )
        self.assertEqual(
            store.domain(left),
            _oracle_projection(
                left,
                (left, right),
                {left: direction_domain, right: direction_domain},
                (factors[0],),
            ),
        )

    def test_production_residual_module_has_no_component_assignment_search(
        self,
    ) -> None:
        source = inspect.getsource(residual_constraints_module)
        self.assertNotIn("_residual_component_has_solution", source)


def _oracle_solutions(
    variables: tuple[VarId, ...],
    domains: dict[VarId, tuple[object, ...]],
    factors: tuple[ResidualFactor, ...],
) -> tuple[dict[VarId, object], ...]:
    solutions: list[dict[VarId, object]] = []

    for values in product(*(domains[var] for var in variables)):
        assignment = dict(zip(variables, values))
        if all(
            factor.accepts(
                tuple(assignment[var] for var in factor.scope)
            )
            for factor in factors
        ):
            solutions.append(assignment)

    return tuple(solutions)


def _oracle_projection(
    var: VarId,
    variables: tuple[VarId, ...],
    domains: dict[VarId, tuple[object, ...]],
    factors: tuple[ResidualFactor, ...],
) -> tuple[object, ...]:
    solutions = _oracle_solutions(variables, domains, factors)
    return tuple(
        value
        for value in domains[var]
        if any(solution[var] == value for solution in solutions)
    )


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


def _tetra_factor_for_var(
    var: VarId,
    *,
    target: TetraValue,
) -> TetraResidualFactor:
    return TetraResidualFactor(
        scope=(var,),
        status=SiteStatus.SPECIFIED,
        target=target,
        reference_order=_occurrences(0, 1, 2, 3),
        local_order=_occurrences(0, 1, 2, 3),
    )


def _directional_factor(
    target: DirectionalValue,
    *,
    status: SiteStatus = SiteStatus.SPECIFIED,
) -> DirectionalResidualFactor:
    left = direction_var(1)
    right = direction_var(2)
    return _directional_factor_between(left, right, target, status=status)


def _directional_factor_between(
    left: VarId,
    right: VarId,
    target: DirectionalValue,
    *,
    status: SiteStatus = SiteStatus.SPECIFIED,
) -> DirectionalResidualFactor:
    return DirectionalResidualFactor(
        scope=(left, right),
        status=status,
        target=target,
        carrier_models={
            left: DirectionalCarrierResidual(left, "left", 1, 1),
            right: DirectionalCarrierResidual(right, "right", 1, 1),
        },
    )


def _snapshot_accepts_restrictions(
    snapshot: ResidualStoreValueSnapshot,
    restrictions: tuple[tuple[VarId, object], ...],
) -> bool:
    store = ResidualStore.from_value_snapshot(snapshot)
    result = store.restrict_many_and_propagate(restrictions)
    if result.kind is ResidualPropagationKind.CONTRADICTION:
        return False
    if result.kind is ResidualPropagationKind.LOCALLY_CONSISTENT_UNCERTIFIED:
        raise AssertionError("test restriction remained uncertified")
    result = store.propagate_all_components()
    if result.kind is ResidualPropagationKind.CONTRADICTION:
        return False
    if result.kind is ResidualPropagationKind.LOCALLY_CONSISTENT_UNCERTIFIED:
        raise AssertionError("test snapshot remained uncertified")
    return True


def _occurrences(*values: int) -> tuple[OccurrenceId, ...]:
    return tuple(OccurrenceId(value) for value in values)


if __name__ == "__main__":
    unittest.main()
