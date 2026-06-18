"""Reversible residual constraints for online South Star enumeration."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from itertools import product
from math import prod
from typing import Literal
from typing import Protocol

from .facts import DirectionalValue
from .facts import SiteStatus
from .facts import TetraValue
from .ids import OccurrenceId
from .policy import DirectionMark
from .policy import TetraToken


_UNASSIGNED = object()
_INVALID = object()
_MAX_RESIDUAL_FACTOR_SCOPE = 4
_MAX_RESIDUAL_FACTOR_CANDIDATE_ROWS = 81


@dataclass(frozen=True, slots=True)
class VarId:
    kind: str
    key: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class ResidualVariable:
    id: VarId
    domain: tuple[object, ...]


class ResidualFactor(Protocol):
    scope: tuple[VarId, ...]

    def accepts(self, row: tuple[object, ...]) -> bool: ...
    def value_snapshot(self) -> object: ...


@dataclass(frozen=True, slots=True)
class TetraResidualFactorValueSnapshot:
    scope: tuple[VarId, ...]
    status: SiteStatus
    target: TetraValue
    reference_order: tuple[OccurrenceId, ...]
    local_order: tuple[OccurrenceId, ...]


@dataclass(frozen=True, slots=True)
class DirectionalResidualFactorValueSnapshot:
    scope: tuple[VarId, ...]
    status: SiteStatus
    target: DirectionalValue
    carrier_models: tuple[tuple[VarId, DirectionalCarrierResidual], ...]


@dataclass(frozen=True, slots=True)
class ResidualStoreValueSnapshot:
    domains: tuple[tuple[VarId, tuple[object, ...]], ...]
    assignments: tuple[tuple[VarId, object], ...]
    factors: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class ResidualConstraintComponentSnapshot:
    variables: tuple[VarId, ...]
    factor_indexes: tuple[int, ...]
    assigned_variables: tuple[VarId, ...]


class ResidualPropagationKind(Enum):
    CONSISTENT = "consistent"
    CONTRADICTION = "contradiction"
    UNSUPPORTED_COMPLEXITY = "unsupported_complexity"


@dataclass(frozen=True, slots=True)
class ResidualPropagationStats:
    component_variables: tuple[VarId, ...] = ()
    component_factor_indexes: tuple[int, ...] = ()
    checked_candidate_rows: int = 0
    largest_factor_scope: int = 0
    largest_candidate_row_count: int = 0


@dataclass(frozen=True, slots=True)
class ResidualPropagationResult:
    kind: ResidualPropagationKind
    stats: ResidualPropagationStats


class ResidualPropagationComplexityError(RuntimeError):
    def __init__(self, stats: ResidualPropagationStats) -> None:
        super().__init__(
            "residual propagation exceeded the supported complexity envelope"
        )
        self.stats = stats


@dataclass(frozen=True, slots=True)
class _ResidualLiveComponent:
    variables: tuple[VarId, ...]
    factor_indexes: tuple[int, ...]


class ResidualStore:
    """Trail-based reversible store for online DFS branches."""

    def __init__(self) -> None:
        self._domains: dict[VarId, tuple[object, ...]] = {}
        self._assignments: dict[VarId, object] = {}
        self._factors: list[ResidualFactor] = []
        self._factor_by_id: dict[int, ResidualFactor] = {}
        self._factors_by_var: dict[VarId, list[int]] = {}
        self._trail: list[tuple[object, ...]] = []

    def add_var(self, var: VarId, domain: tuple[object, ...]) -> None:
        if var in self._domains:
            raise ValueError(f"duplicate residual variable: {var!r}")
        if not domain:
            raise ValueError(f"residual variable has empty domain: {var!r}")
        if _domain_has_duplicate_value(domain):
            raise ValueError(f"residual variable has duplicate domain value: {var!r}")
        self._domains[var] = domain

    def add_factor(self, factor: ResidualFactor) -> int:
        if len(set(factor.scope)) != len(factor.scope):
            raise ValueError("residual factor scope contains duplicates")
        for var in factor.scope:
            if var not in self._domains:
                raise ValueError(f"factor references unknown variable: {var!r}")
        factor_id = len(self._factors)
        self._factors.append(factor)
        self._factor_by_id[factor_id] = factor
        for var in factor.scope:
            self._factors_by_var.setdefault(var, []).append(factor_id)
        self._trail.append(("factor_add", factor_id))
        return factor_id

    def assign(self, var: VarId, value: object) -> bool:
        result = self.restrict_to_value(var, value)
        if result.kind is ResidualPropagationKind.UNSUPPORTED_COMPLEXITY:
            raise ResidualPropagationComplexityError(result.stats)
        return result.kind is ResidualPropagationKind.CONSISTENT

    def restrict_to_value(
        self,
        var: VarId,
        value: object,
    ) -> ResidualPropagationResult:
        if var not in self._domains:
            raise ValueError(f"unknown residual variable: {var!r}")
        if value not in self._domains[var]:
            return ResidualPropagationResult(
                ResidualPropagationKind.CONTRADICTION,
                ResidualPropagationStats(component_variables=(var,)),
            )
        existing = self._assignments.get(var, _UNASSIGNED)
        if existing is not _UNASSIGNED:
            return ResidualPropagationResult(
                (
                    ResidualPropagationKind.CONSISTENT
                    if existing == value
                    else ResidualPropagationKind.CONTRADICTION
                ),
                ResidualPropagationStats(component_variables=(var,)),
            )

        checkpoint = self.checkpoint()
        self._assignments[var] = value
        self._trail.append(("assignment", var))
        self._replace_domain(var, (value,))

        component = self._component_from_variables((var,))
        result = self._propagate_component(component)
        if result.kind is not ResidualPropagationKind.CONSISTENT:
            self.rollback(checkpoint)
        return result

    def checkpoint(self) -> int:
        return len(self._trail)

    def rollback(self, checkpoint: int) -> None:
        if checkpoint < 0 or checkpoint > len(self._trail):
            raise ValueError(f"invalid residual checkpoint: {checkpoint!r}")
        while len(self._trail) > checkpoint:
            entry = self._trail.pop()
            if entry[0] == "assignment":
                _, var = entry
                del self._assignments[var]
                continue
            if entry[0] == "domain":
                _, var, old_domain = entry
                self._domains[var] = old_domain
                continue
            if entry[0] == "factor_add":
                _, factor_id = entry
                _remove_factor(self, factor_id)
                continue
            raise AssertionError(f"unknown residual trail entry: {entry!r}")

    def contains_var(self, var: VarId) -> bool:
        return var in self._domains

    def domain(self, var: VarId) -> tuple[object, ...]:
        try:
            return self._domains[var]
        except KeyError as exc:
            raise ValueError(f"unknown residual variable: {var!r}") from exc

    def assignment(self, var: VarId) -> object | None:
        return self._assignments.get(var)

    def value_snapshot(self) -> ResidualStoreValueSnapshot:
        return ResidualStoreValueSnapshot(
            domains=tuple(
                sorted(self._domains.items(), key=lambda item: _var_sort_key(item[0]))
            ),
            assignments=tuple(
                sorted(self._assignments.items(), key=lambda item: _var_sort_key(item[0]))
            ),
            factors=tuple(factor.value_snapshot() for factor in self._factors),
        )

    @classmethod
    def from_value_snapshot(
        cls,
        snapshot: ResidualStoreValueSnapshot,
    ) -> "ResidualStore":
        _validate_residual_snapshot_assignment_consistency(snapshot)
        store = cls()
        store._domains = dict(snapshot.domains)
        store._assignments = dict(snapshot.assignments)
        store._factors = [
            _factor_from_value_snapshot(factor_snapshot)
            for factor_snapshot in snapshot.factors
        ]
        store._factor_by_id = {
            factor_id: factor
            for factor_id, factor in enumerate(store._factors)
        }
        store._factors_by_var = {}
        for factor_id, factor in enumerate(store._factors):
            for var in factor.scope:
                if var not in store._domains:
                    raise ValueError(f"factor snapshot references unknown variable: {var!r}")
                store._factors_by_var.setdefault(var, []).append(factor_id)
        store._trail = []
        return store

    def _replace_domain(
        self,
        var: VarId,
        domain: tuple[object, ...],
    ) -> None:
        if not domain:
            raise ValueError("cannot install an empty residual domain")
        old_domain = self._domains[var]
        if domain == old_domain:
            return
        self._trail.append(("domain", var, old_domain))
        self._domains[var] = domain

    def _component_from_variables(
        self,
        seed_variables: tuple[VarId, ...],
    ) -> _ResidualLiveComponent:
        pending_variables = list(seed_variables)
        seen_variables: set[VarId] = set()
        seen_factors: set[int] = set()

        while pending_variables:
            var = pending_variables.pop()
            if var in seen_variables:
                continue
            if var not in self._domains:
                raise ValueError(f"unknown residual variable: {var!r}")

            seen_variables.add(var)
            for factor_id in self._factors_by_var.get(var, ()):
                if factor_id in seen_factors:
                    continue
                seen_factors.add(factor_id)
                pending_variables.extend(self._factor_by_id[factor_id].scope)

        return _ResidualLiveComponent(
            variables=tuple(sorted(seen_variables, key=_var_sort_key)),
            factor_indexes=tuple(sorted(seen_factors)),
        )

    def _supported_factor_rows(
        self,
        factor: ResidualFactor,
    ) -> tuple[tuple[tuple[object, ...], ...], int] | None:
        if len(factor.scope) > _MAX_RESIDUAL_FACTOR_SCOPE:
            return None

        domains = tuple(self._domains[var] for var in factor.scope)
        candidate_count = prod(len(domain) for domain in domains)
        if candidate_count > _MAX_RESIDUAL_FACTOR_CANDIDATE_ROWS:
            return None

        rows = tuple(
            row
            for row in product(*domains)
            if factor.accepts(row)
        )
        return rows, candidate_count

    def _propagate_component(
        self,
        component: _ResidualLiveComponent,
    ) -> ResidualPropagationResult:
        queue = deque(component.factor_indexes)
        queued = set(component.factor_indexes)
        checked_rows = 0
        largest_scope = 0
        largest_candidate_count = 0

        while queue:
            factor_id = queue.popleft()
            queued.remove(factor_id)
            factor = self._factor_by_id[factor_id]

            largest_scope = max(largest_scope, len(factor.scope))
            supported = self._supported_factor_rows(factor)
            if supported is None:
                return ResidualPropagationResult(
                    ResidualPropagationKind.UNSUPPORTED_COMPLEXITY,
                    ResidualPropagationStats(
                        component.variables,
                        component.factor_indexes,
                        checked_rows,
                        largest_scope,
                        largest_candidate_count,
                    ),
                )

            rows, candidate_count = supported
            checked_rows += candidate_count
            largest_candidate_count = max(
                largest_candidate_count,
                candidate_count,
            )

            if not rows:
                return ResidualPropagationResult(
                    ResidualPropagationKind.CONTRADICTION,
                    ResidualPropagationStats(
                        component.variables,
                        component.factor_indexes,
                        checked_rows,
                        largest_scope,
                        largest_candidate_count,
                    ),
                )

            for position, var in enumerate(factor.scope):
                old_domain = self._domains[var]
                supported_values = tuple(row[position] for row in rows)
                new_domain = tuple(
                    value
                    for value in old_domain
                    if any(value == supported for supported in supported_values)
                )

                if not new_domain:
                    return ResidualPropagationResult(
                        ResidualPropagationKind.CONTRADICTION,
                        ResidualPropagationStats(
                            component.variables,
                            component.factor_indexes,
                            checked_rows,
                            largest_scope,
                            largest_candidate_count,
                        ),
                    )

                if new_domain == old_domain:
                    continue

                self._replace_domain(var, new_domain)
                for neighbour_id in self._factors_by_var.get(var, ()):
                    if neighbour_id != factor_id and neighbour_id not in queued:
                        queue.append(neighbour_id)
                        queued.add(neighbour_id)

        unresolved = any(
            len(self._domains[var]) > 1
            for var in component.variables
        )
        if unresolved and not self._component_incidence_is_acyclic(component):
            return ResidualPropagationResult(
                ResidualPropagationKind.UNSUPPORTED_COMPLEXITY,
                ResidualPropagationStats(
                    component.variables,
                    component.factor_indexes,
                    checked_rows,
                    largest_scope,
                    largest_candidate_count,
                ),
            )

        return ResidualPropagationResult(
            ResidualPropagationKind.CONSISTENT,
            ResidualPropagationStats(
                component.variables,
                component.factor_indexes,
                checked_rows,
                largest_scope,
                largest_candidate_count,
            ),
        )

    def _component_incidence_is_acyclic(
        self,
        component: _ResidualLiveComponent,
    ) -> bool:
        node_count = len(component.variables) + len(component.factor_indexes)
        edge_count = sum(
            len(self._factor_by_id[factor_id].scope)
            for factor_id in component.factor_indexes
        )
        return edge_count == max(0, node_count - 1)


def add_factor_and_propagate(
    store: ResidualStore,
    factor: ResidualFactor,
) -> ResidualPropagationResult:
    checkpoint = store.checkpoint()
    try:
        factor_id = store.add_factor(factor)
        if factor.scope:
            component = store._component_from_variables(factor.scope)
        else:
            component = _ResidualLiveComponent(
                variables=(),
                factor_indexes=(factor_id,),
            )
        result = store._propagate_component(component)
        if result.kind is not ResidualPropagationKind.CONSISTENT:
            store.rollback(checkpoint)
        return result
    except Exception:
        store.rollback(checkpoint)
        raise


def _residual_snapshot_domain_map(
    snapshot: ResidualStoreValueSnapshot,
) -> dict[VarId, tuple[object, ...]]:
    domains = dict(snapshot.domains)
    if len(domains) != len(snapshot.domains):
        raise ValueError("duplicate residual snapshot domain")
    for var, domain in snapshot.domains:
        if not domain:
            raise ValueError(f"empty residual snapshot domain: {var!r}")
        if _domain_has_duplicate_value(domain):
            raise ValueError(f"duplicate residual snapshot domain value: {var!r}")
    return domains


def _domain_has_duplicate_value(domain: tuple[object, ...]) -> bool:
    for index, value in enumerate(domain):
        for other in domain[index + 1 :]:
            if value == other:
                return True
    return False


def residual_store_constraint_components(
    snapshot: ResidualStoreValueSnapshot,
) -> tuple[ResidualConstraintComponentSnapshot, ...]:
    domains = _residual_snapshot_domain_map(snapshot)
    domain_vars = tuple(domains)
    known = frozenset(domains)
    assignment_vars = frozenset(var for var, _ in snapshot.assignments)
    unknown_assignments = assignment_vars - known
    if unknown_assignments:
        raise ValueError(
            "residual assignment references unknown variable: "
            f"{tuple(sorted(unknown_assignments, key=_var_sort_key))!r}"
        )

    parent = {var: var for var in domain_vars}

    def find(var: VarId) -> VarId:
        while parent[var] != var:
            parent[var] = parent[parent[var]]
            var = parent[var]
        return var

    def union(left: VarId, right: VarId) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    factor_scopes: list[tuple[VarId, ...]] = []
    zero_scope_factors: list[int] = []
    for index, factor_snapshot in enumerate(snapshot.factors):
        scope = tuple(getattr(factor_snapshot, "scope", ()))
        unknown_scope = frozenset(scope) - known
        if unknown_scope:
            raise ValueError(
                "residual factor references unknown variable: "
                f"{tuple(sorted(unknown_scope, key=_var_sort_key))!r}"
            )
        factor_scopes.append(scope)
        if not scope:
            zero_scope_factors.append(index)
            continue
        first = scope[0]
        for var in scope[1:]:
            union(first, var)

    variables_by_root: dict[VarId, list[VarId]] = {}
    for var in domain_vars:
        variables_by_root.setdefault(find(var), []).append(var)
    factor_indexes_by_root: dict[VarId, list[int]] = {
        root: [] for root in variables_by_root
    }
    for index, scope in enumerate(factor_scopes):
        if not scope:
            continue
        factor_indexes_by_root.setdefault(find(scope[0]), []).append(index)

    components: list[ResidualConstraintComponentSnapshot] = []
    for root, variables in variables_by_root.items():
        sorted_variables = tuple(sorted(variables, key=_var_sort_key))
        assigned = tuple(var for var in sorted_variables if var in assignment_vars)
        components.append(
            ResidualConstraintComponentSnapshot(
                variables=sorted_variables,
                factor_indexes=tuple(factor_indexes_by_root.get(root, ())),
                assigned_variables=assigned,
            )
        )
    for index in zero_scope_factors:
        components.append(
            ResidualConstraintComponentSnapshot(
                variables=(),
                factor_indexes=(index,),
                assigned_variables=(),
            )
        )

    return tuple(
        sorted(
            components,
            key=lambda component: (
                0 if component.variables else 1,
                _var_sort_key(component.variables[0])
                if component.variables
                else ("", ()),
                component.factor_indexes,
            ),
        )
    )


def residual_store_projected_values(
    snapshot: ResidualStoreValueSnapshot,
    var: VarId,
) -> tuple[object, ...]:
    store = ResidualStore.from_value_snapshot(snapshot)
    if not store.contains_var(var):
        raise ValueError(f"unknown residual variable: {var!r}")
    component = store._component_from_variables((var,))
    result = store._propagate_component(component)

    if result.kind is ResidualPropagationKind.CONTRADICTION:
        return ()
    if result.kind is ResidualPropagationKind.UNSUPPORTED_COMPLEXITY:
        raise ResidualPropagationComplexityError(result.stats)

    return store.domain(var)


def residual_store_assignments_have_support(
    snapshot: ResidualStoreValueSnapshot,
    assignments: tuple[tuple[VarId, object], ...],
) -> bool:
    store = ResidualStore.from_value_snapshot(snapshot)
    for var, value in assignments:
        if not store.contains_var(var):
            raise ValueError(f"unknown residual variable: {var!r}")
        result = store.restrict_to_value(var, value)
        if result.kind is ResidualPropagationKind.CONTRADICTION:
            return False
        if result.kind is ResidualPropagationKind.UNSUPPORTED_COMPLEXITY:
            raise ResidualPropagationComplexityError(result.stats)
    for component in residual_store_constraint_components(store.value_snapshot()):
        if component.variables:
            live_component = store._component_from_variables(component.variables)
        else:
            live_component = _ResidualLiveComponent(
                variables=(),
                factor_indexes=component.factor_indexes,
            )
        result = store._propagate_component(live_component)
        if result.kind is ResidualPropagationKind.CONTRADICTION:
            return False
        if result.kind is ResidualPropagationKind.UNSUPPORTED_COMPLEXITY:
            raise ResidualPropagationComplexityError(result.stats)
    return True


def _validate_residual_snapshot_assignment_consistency(
    snapshot: ResidualStoreValueSnapshot,
) -> None:
    domains = _residual_snapshot_domain_map(snapshot)
    assignments = dict(snapshot.assignments)
    if len(assignments) != len(snapshot.assignments):
        raise ValueError("duplicate residual snapshot assignment")

    for var, value in snapshot.assignments:
        if var not in domains:
            raise ValueError(
                f"residual assignment references unknown variable: {var!r}"
            )
        if domains[var] != (value,):
            raise ValueError(
                "explicit residual assignment must have singleton domain: "
                f"{var!r}={value!r}, domain={domains[var]!r}"
            )
        if value not in domains[var]:
            raise ValueError(
                f"residual assignment value outside domain: {var!r}={value!r}"
            )

    for factor_snapshot in snapshot.factors:
        scope = tuple(getattr(factor_snapshot, "scope", ()))
        for var in scope:
            if var not in domains:
                raise ValueError(
                    f"factor snapshot references unknown variable: {var!r}"
                )

        if isinstance(factor_snapshot, TetraResidualFactorValueSnapshot):
            if len(scope) != 1:
                raise ValueError(
                    "tetra residual factor snapshot must have unary scope"
                )
            continue

        if isinstance(factor_snapshot, DirectionalResidualFactorValueSnapshot):
            continue

        raise ValueError(f"unknown residual factor snapshot: {factor_snapshot!r}")


def _remove_factor(store: ResidualStore, factor_id: int) -> None:
    if factor_id != len(store._factors) - 1:
        raise AssertionError("residual factor rollback is not LIFO")
    factor = store._factors[factor_id]
    if factor_id in store._factor_by_id:
        del store._factor_by_id[factor_id]
    store._factors.pop()
    for var in factor.scope:
        factors = store._factors_by_var.get(var)
        if factors is None:
            continue
        store._factors_by_var[var] = [item for item in factors if item != factor_id]
        if not store._factors_by_var[var]:
            del store._factors_by_var[var]


@dataclass(frozen=True, slots=True)
class TetraResidualFactor:
    scope: tuple[VarId, ...]
    status: SiteStatus
    target: TetraValue
    reference_order: tuple[OccurrenceId, ...]
    local_order: tuple[OccurrenceId, ...]

    def __post_init__(self) -> None:
        if len(self.scope) != 1:
            raise ValueError("tetra residual factor must have unary scope")

    @property
    def token_var(self) -> VarId:
        return self.scope[0]

    def accepts(self, row: tuple[object, ...]) -> bool:
        return len(row) == 1 and row[0] in self.allowed_tokens()

    def value_snapshot(self) -> TetraResidualFactorValueSnapshot:
        return TetraResidualFactorValueSnapshot(
            scope=self.scope,
            status=self.status,
            target=self.target,
            reference_order=self.reference_order,
            local_order=self.local_order,
        )

    def allowed_tokens(self) -> frozenset[TetraToken]:
        if self.status is SiteStatus.UNSPECIFIED:
            return frozenset((TetraToken.NONE,))
        if self.target is TetraValue.NONE:
            return frozenset()
        if set(self.local_order) != set(self.reference_order):
            return frozenset()
        if len(self.local_order) != len(self.reference_order):
            return frozenset()
        is_even = _is_even_permutation(
            tuple(self.reference_order.index(item) for item in self.local_order)
        )
        if self.target is TetraValue.PLUS:
            return frozenset((TetraToken.AT if is_even else TetraToken.ATAT,))
        if self.target is TetraValue.MINUS:
            return frozenset((TetraToken.ATAT if is_even else TetraToken.AT,))
        return frozenset()


@dataclass(frozen=True, slots=True)
class DirectionalCarrierResidual:
    var: VarId
    side: Literal["left", "right"]
    orientation: Literal[-1, 1]
    ligand_factor: Literal[-1, 1] = 1


@dataclass(frozen=True, slots=True)
class DirectionalResidualFactor:
    scope: tuple[VarId, ...]
    status: SiteStatus
    target: DirectionalValue
    carrier_models: Mapping[VarId, DirectionalCarrierResidual]

    def __post_init__(self) -> None:
        if len(set(self.scope)) != len(self.scope):
            raise ValueError("directional residual scope contains duplicates")
        if set(self.scope) != set(self.carrier_models):
            raise ValueError("directional residual scope/model mismatch")

    def accepts(self, row: tuple[object, ...]) -> bool:
        if len(row) != len(self.scope):
            return False
        if any(mark not in _DIRECTION_MARKS for mark in row):
            return False

        marks = dict(zip(self.scope, row))
        value = _directional_value(marks, self.carrier_models)
        if value is _INVALID:
            return False

        if self.status is SiteStatus.UNSPECIFIED:
            return value is DirectionalValue.NONE

        return value is self.target

    def value_snapshot(self) -> DirectionalResidualFactorValueSnapshot:
        return DirectionalResidualFactorValueSnapshot(
            scope=self.scope,
            status=self.status,
            target=self.target,
            carrier_models=tuple(
                sorted(self.carrier_models.items(), key=lambda item: _var_sort_key(item[0]))
            ),
        )


def tetra_var(center: object) -> VarId:
    return VarId("tetra", (center,))


def direction_var(carrier: object) -> VarId:
    return VarId("direction", (carrier,))


def _var_sort_key(var: VarId) -> tuple[str, tuple[str, ...]]:
    return (var.kind, tuple(repr(item) for item in var.key))


def _directional_value(
    marks: Mapping[VarId, object],
    carrier_models: Mapping[VarId, DirectionalCarrierResidual],
) -> DirectionalValue | object:
    left: list[int] = []
    right: list[int] = []
    for var, mark in marks.items():
        if mark is DirectionMark.ABSENT:
            continue
        model = carrier_models[var]
        normalized = _mark_sign(mark) * model.orientation * model.ligand_factor
        if model.side == "left":
            left.append(normalized)
        else:
            right.append(normalized)

    if len(set(left)) > 1 or len(set(right)) > 1:
        return _INVALID
    if not left and not right:
        return DirectionalValue.NONE
    if not left or not right:
        return DirectionalValue.NONE
    return DirectionalValue.TOGETHER if left[0] == right[0] else DirectionalValue.OPPOSITE


def _mark_sign(mark: object) -> int:
    if mark is DirectionMark.FWD:
        return 1
    if mark is DirectionMark.REV:
        return -1
    raise ValueError(f"direction mark has no sign: {mark!r}")


def _is_even_permutation(indices: tuple[int, ...]) -> bool:
    inversions = 0
    for left, value in enumerate(indices):
        for other in indices[left + 1 :]:
            if value > other:
                inversions += 1
    return inversions % 2 == 0


def _factor_from_value_snapshot(snapshot: object) -> ResidualFactor:
    if isinstance(snapshot, TetraResidualFactorValueSnapshot):
        return TetraResidualFactor(
            scope=snapshot.scope,
            status=snapshot.status,
            target=snapshot.target,
            reference_order=snapshot.reference_order,
            local_order=snapshot.local_order,
        )
    if isinstance(snapshot, DirectionalResidualFactorValueSnapshot):
        return DirectionalResidualFactor(
            scope=snapshot.scope,
            status=snapshot.status,
            target=snapshot.target,
            carrier_models=dict(snapshot.carrier_models),
        )
    raise ValueError(f"unknown residual factor snapshot: {snapshot!r}")


_TETRA_TOKENS = frozenset((TetraToken.NONE, TetraToken.AT, TetraToken.ATAT))
_DIRECTION_MARKS = frozenset((DirectionMark.ABSENT, DirectionMark.FWD, DirectionMark.REV))


__all__ = (
    "DirectionalCarrierResidual",
    "DirectionalResidualFactor",
    "DirectionalResidualFactorValueSnapshot",
    "ResidualConstraintComponentSnapshot",
    "ResidualFactor",
    "ResidualPropagationComplexityError",
    "ResidualPropagationKind",
    "ResidualPropagationResult",
    "ResidualPropagationStats",
    "ResidualStore",
    "ResidualStoreValueSnapshot",
    "ResidualVariable",
    "TetraResidualFactor",
    "TetraResidualFactorValueSnapshot",
    "VarId",
    "add_factor_and_propagate",
    "direction_var",
    "residual_store_assignments_have_support",
    "residual_store_constraint_components",
    "residual_store_projected_values",
    "tetra_var",
)
