"""Reversible residual constraints for online South Star enumeration."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from itertools import product
from math import prod
from typing import Literal
from typing import Protocol

from .facts import DirectionalValue
from .facts import SiteStatus
from .facts import TetraValue
from .ids import OccurrenceId
from .ids import BondId
from .ids import SiteId
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


class TetraLocalParity(Enum):
    EVEN = "even"
    ODD = "odd"


class DirectionalNormalizedSign(Enum):
    ABSENT = "absent"
    POSITIVE = "positive"
    NEGATIVE = "negative"


@dataclass(frozen=True, slots=True)
class ResidualFactorKey:
    kind: str
    key: tuple[object, ...]


class ResidualFactor(Protocol):
    key: ResidualFactorKey
    scope: tuple[VarId, ...]

    def accepts(self, row: tuple[object, ...]) -> bool: ...
    def value_snapshot(self) -> object: ...


@dataclass(frozen=True, slots=True)
class TetraResidualFactorValueSnapshot:
    key: ResidualFactorKey
    scope: tuple[VarId, ...]
    status: SiteStatus
    target: TetraValue
    reference_order: tuple[OccurrenceId, ...]
    local_order: tuple[OccurrenceId, ...]


@dataclass(frozen=True, slots=True)
class DirectionalResidualFactorValueSnapshot:
    key: ResidualFactorKey
    scope: tuple[VarId, ...]
    status: SiteStatus
    target: DirectionalValue
    carrier_models: tuple[tuple[VarId, DirectionalCarrierResidual], ...]


@dataclass(frozen=True, slots=True)
class TetraTokenParityFactorValueSnapshot:
    key: ResidualFactorKey
    scope: tuple[VarId, VarId]
    status: SiteStatus
    target: TetraValue


@dataclass(frozen=True, slots=True)
class DirectionalSiteCarrierModel:
    site: SiteId
    bond: BondId
    side: Literal["left", "right"]
    endpoint_orientation_factor: Literal[-1, 1]
    ligand_factor: Literal[-1, 1] = 1


@dataclass(frozen=True, slots=True)
class DirectionalSiteFactorValueSnapshot:
    key: ResidualFactorKey
    scope: tuple[VarId, ...]
    sides: tuple[tuple[VarId, Literal["left", "right"]], ...]
    status: SiteStatus
    target: DirectionalValue


@dataclass(frozen=True, slots=True)
class DirectionalBondEmissionFactorValueSnapshot:
    key: ResidualFactorKey
    scope: tuple[VarId, ...]
    models: tuple[DirectionalSiteCarrierModel, ...]
    allowed_marks: tuple[DirectionMark, ...]


@dataclass(frozen=True, slots=True)
class ResidualStoreValueSnapshot:
    domains: tuple[tuple[VarId, tuple[object, ...]], ...]
    assignments: tuple[tuple[VarId, object], ...]
    factors: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class ResidualConstraintComponentSnapshot:
    variables: tuple[VarId, ...]
    factor_keys: tuple[ResidualFactorKey, ...]
    assigned_variables: tuple[VarId, ...]

    @property
    def factor_indexes(self) -> tuple[int, ...]:
        return tuple(range(len(self.factor_keys)))


class ResidualPropagationKind(Enum):
    CERTIFIED_CONSISTENT = "certified_consistent"
    CONTRADICTION = "contradiction"
    LOCALLY_CONSISTENT_UNCERTIFIED = "locally_consistent_uncertified"
    CONSISTENT = "certified_consistent"
    UNSUPPORTED_COMPLEXITY = "locally_consistent_uncertified"


@dataclass(frozen=True, slots=True)
class ResidualPropagationStats:
    component_variables: tuple[VarId, ...] = ()
    component_factor_keys: tuple[ResidualFactorKey, ...] = ()
    checked_candidate_rows: int = 0
    largest_factor_scope: int = 0
    largest_candidate_row_count: int = 0

    @property
    def component_factor_indexes(self) -> tuple[int, ...]:
        return tuple(range(len(self.component_factor_keys)))


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
    factor_keys: tuple[ResidualFactorKey, ...]

    @property
    def factor_indexes(self) -> tuple[int, ...]:
        return tuple(range(len(self.factor_keys)))


class ResidualStore:
    """Trail-based reversible store for online DFS branches."""

    def __init__(self) -> None:
        self._domains: dict[VarId, tuple[object, ...]] = {}
        self._assignments: dict[VarId, object] = {}
        self._factors: dict[ResidualFactorKey, ResidualFactor] = {}
        self._factors_by_var: dict[VarId, set[ResidualFactorKey]] = {}
        self._trail: list[tuple[object, ...]] = []

    def add_var(self, var: VarId, domain: tuple[object, ...]) -> None:
        if var in self._domains:
            raise ValueError(f"duplicate residual variable: {var!r}")
        if not domain:
            raise ValueError(f"residual variable has empty domain: {var!r}")
        if _domain_has_duplicate_value(domain):
            raise ValueError(f"residual variable has duplicate domain value: {var!r}")
        self._domains[var] = domain

    def add_factor(self, factor: ResidualFactor) -> ResidualFactorKey:
        if len(set(factor.scope)) != len(factor.scope):
            raise ValueError("residual factor scope contains duplicates")
        if factor.key in self._factors:
            raise ValueError(f"duplicate residual factor key: {factor.key!r}")
        for var in factor.scope:
            if var not in self._domains:
                raise ValueError(f"factor references unknown variable: {var!r}")
        factor_key = factor.key
        self._factors[factor_key] = factor
        for var in factor.scope:
            self._factors_by_var.setdefault(var, set()).add(factor_key)
        self._trail.append(("factor_add", factor_key))
        return factor_key

    def assign(self, var: VarId, value: object) -> bool:
        result = self.restrict_many_and_propagate(((var, value),))
        if result.kind is ResidualPropagationKind.LOCALLY_CONSISTENT_UNCERTIFIED:
            raise ResidualPropagationComplexityError(result.stats)
        return result.kind is ResidualPropagationKind.CERTIFIED_CONSISTENT

    def restrict_to_value(
        self,
        var: VarId,
        value: object,
    ) -> ResidualPropagationResult:
        checkpoint = self.checkpoint()
        result = self.restrict_many_and_propagate(((var, value),))
        if result.kind is not ResidualPropagationKind.CERTIFIED_CONSISTENT:
            self.rollback(checkpoint)
        return result

    def restrict_many_and_propagate(
        self,
        restrictions: tuple[tuple[VarId, object], ...],
    ) -> ResidualPropagationResult:
        normalized: dict[VarId, object] = {}
        for var, value in restrictions:
            if var in normalized and normalized[var] != value:
                return ResidualPropagationResult(
                    ResidualPropagationKind.CONTRADICTION,
                    ResidualPropagationStats(component_variables=(var,)),
                )
            normalized[var] = value

        if not normalized:
            return self.propagate_all_components()

        affected: list[VarId] = []
        for var, value in normalized.items():
            result = self._install_restriction(var, value)
            if result is not None:
                return result
            affected.append(var)

        component = self._component_from_variables(tuple(affected))
        return self._propagate_component(component)

    def _install_restriction(
        self,
        var: VarId,
        value: object,
    ) -> ResidualPropagationResult | None:
        if var not in self._domains:
            raise ValueError(f"unknown residual variable: {var!r}")
        if value not in self._domains[var]:
            return ResidualPropagationResult(
                ResidualPropagationKind.CONTRADICTION,
                ResidualPropagationStats(component_variables=(var,)),
            )
        existing = self._assignments.get(var, _UNASSIGNED)
        if existing is not _UNASSIGNED:
            if existing == value:
                return None
            return ResidualPropagationResult(
                ResidualPropagationKind.CONTRADICTION,
                ResidualPropagationStats(component_variables=(var,)),
            )

        self._assignments[var] = value
        self._trail.append(("assignment", var))
        self._replace_domain(var, (value,))
        return None

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
                _, factor_key = entry
                self._remove_factor(factor_key)
                continue
            if entry[0] == "factor_remove":
                _, factor = entry
                self._install_factor_untrailed(factor)
                continue
            if entry[0] == "var_remove":
                _, var, domain, assigned = entry
                self._domains[var] = domain
                if assigned is not _UNASSIGNED:
                    self._assignments[var] = assigned
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
            factors=tuple(
                self._factors[key].value_snapshot()
                for key in sorted(self._factors, key=_factor_key_sort_key)
            ),
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
        store._factors = {}
        store._factors_by_var = {}
        for factor_snapshot in snapshot.factors:
            factor = _factor_from_value_snapshot(factor_snapshot)
            store._install_factor_untrailed(factor)
        store._trail = []
        return store

    def _install_factor_untrailed(self, factor: ResidualFactor) -> None:
        if factor.key in self._factors:
            raise ValueError(f"duplicate residual factor key: {factor.key!r}")
        for var in factor.scope:
            if var not in self._domains:
                raise ValueError(f"factor snapshot references unknown variable: {var!r}")
        self._factors[factor.key] = factor
        for var in factor.scope:
            self._factors_by_var.setdefault(var, set()).add(factor.key)

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
        seen_factors: set[ResidualFactorKey] = set()

        while pending_variables:
            var = pending_variables.pop()
            if var in seen_variables:
                continue
            if var not in self._domains:
                raise ValueError(f"unknown residual variable: {var!r}")

            seen_variables.add(var)
            for factor_key in self._factors_by_var.get(var, ()):
                if factor_key in seen_factors:
                    continue
                seen_factors.add(factor_key)
                pending_variables.extend(self._factors[factor_key].scope)

        return _ResidualLiveComponent(
            variables=tuple(sorted(seen_variables, key=_var_sort_key)),
            factor_keys=tuple(sorted(seen_factors, key=_factor_key_sort_key)),
        )

    def _component_from_factor_key(
        self,
        factor_key: ResidualFactorKey,
    ) -> _ResidualLiveComponent:
        factor = self._factors[factor_key]
        if factor.scope:
            return self._component_from_variables(factor.scope)
        return _ResidualLiveComponent(variables=(), factor_keys=(factor_key,))

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
        queue = deque(component.factor_keys)
        queued = set(component.factor_keys)
        checked_rows = 0
        largest_scope = 0
        largest_candidate_count = 0

        while queue:
            factor_key = queue.popleft()
            queued.remove(factor_key)
            factor = self._factors[factor_key]

            largest_scope = max(largest_scope, len(factor.scope))
            supported = self._supported_factor_rows(factor)
            if supported is None:
                return ResidualPropagationResult(
                    ResidualPropagationKind.LOCALLY_CONSISTENT_UNCERTIFIED,
                    ResidualPropagationStats(
                        component.variables,
                        component.factor_keys,
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
                        component.factor_keys,
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
                            component.factor_keys,
                            checked_rows,
                            largest_scope,
                            largest_candidate_count,
                        ),
                    )

                if new_domain == old_domain:
                    continue

                self._replace_domain(var, new_domain)
                for neighbour_key in self._factors_by_var.get(var, ()):
                    if neighbour_key != factor_key and neighbour_key not in queued:
                        queue.append(neighbour_key)
                        queued.add(neighbour_key)

        unresolved = any(
            len(self._domains[var]) > 1
            for var in component.variables
        )
        if unresolved and not self._component_incidence_is_acyclic(component):
            return ResidualPropagationResult(
                ResidualPropagationKind.LOCALLY_CONSISTENT_UNCERTIFIED,
                ResidualPropagationStats(
                    component.variables,
                    component.factor_keys,
                    checked_rows,
                    largest_scope,
                    largest_candidate_count,
                ),
            )

        return ResidualPropagationResult(
            ResidualPropagationKind.CERTIFIED_CONSISTENT,
            ResidualPropagationStats(
                component.variables,
                component.factor_keys,
                checked_rows,
                largest_scope,
                largest_candidate_count,
            ),
        )

    def _component_incidence_is_acyclic(
        self,
        component: _ResidualLiveComponent,
    ) -> bool:
        node_count = len(component.variables) + len(component.factor_keys)
        edge_count = sum(
            len(self._factors[factor_key].scope)
            for factor_key in component.factor_keys
        )
        return edge_count == max(0, node_count - 1)

    def propagate_all_components(self) -> ResidualPropagationResult:
        seen_variables: set[VarId] = set()
        checked_rows = 0
        largest_scope = 0
        largest_candidate_count = 0
        for var in tuple(sorted(self._domains, key=_var_sort_key)):
            if var in seen_variables:
                continue
            component = self._component_from_variables((var,))
            seen_variables.update(component.variables)
            result = self._propagate_component(component)
            checked_rows += result.stats.checked_candidate_rows
            largest_scope = max(largest_scope, result.stats.largest_factor_scope)
            largest_candidate_count = max(
                largest_candidate_count,
                result.stats.largest_candidate_row_count,
            )
            if result.kind is not ResidualPropagationKind.CERTIFIED_CONSISTENT:
                return result
        for factor_key, factor in tuple(self._factors.items()):
            if factor.scope:
                continue
            result = self._propagate_component(
                _ResidualLiveComponent(variables=(), factor_keys=(factor_key,))
            )
            checked_rows += result.stats.checked_candidate_rows
            if result.kind is not ResidualPropagationKind.CERTIFIED_CONSISTENT:
                return result
        return ResidualPropagationResult(
            ResidualPropagationKind.CERTIFIED_CONSISTENT,
            ResidualPropagationStats(
                tuple(sorted(self._domains, key=_var_sort_key)),
                tuple(sorted(self._factors, key=_factor_key_sort_key)),
                checked_rows,
                largest_scope,
                largest_candidate_count,
            ),
        )

    def discharge_satisfied_factors(
        self,
        factor_keys: tuple[ResidualFactorKey, ...],
    ) -> None:
        for factor_key in factor_keys:
            factor = self._factors.get(factor_key)
            if factor is None:
                raise ValueError(f"unknown residual factor: {factor_key!r}")
            row = []
            for var in factor.scope:
                domain = self._domains[var]
                if len(domain) != 1:
                    raise ValueError(
                        "cannot discharge residual factor with non-singleton "
                        f"domain: {factor_key!r}, {var!r}"
                    )
                row.append(domain[0])
            if not factor.accepts(tuple(row)):
                raise ValueError(
                    f"cannot discharge unsatisfied residual factor: {factor_key!r}"
                )
            self._trail.append(("factor_remove", factor))
            self._remove_factor(factor_key)
            for var in factor.scope:
                if self._factors_by_var.get(var):
                    continue
                domain = self._domains.pop(var)
                assigned = self._assignments.pop(var, _UNASSIGNED)
                self._trail.append(("var_remove", var, domain, assigned))

    def _remove_factor(self, factor_key: ResidualFactorKey) -> None:
        factor = self._factors.pop(factor_key)
        for var in factor.scope:
            factors = self._factors_by_var.get(var)
            if factors is None:
                continue
            factors.discard(factor_key)
            if not factors:
                del self._factors_by_var[var]


def add_factor_and_propagate(
    store: ResidualStore,
    factor: ResidualFactor,
) -> ResidualPropagationResult:
    checkpoint = store.checkpoint()
    try:
        factor_key = store.add_factor(factor)
        component = store._component_from_factor_key(factor_key)
        result = store._propagate_component(component)
        if result.kind is ResidualPropagationKind.CONTRADICTION:
            store.rollback(checkpoint)
        return result
    except Exception:
        store.rollback(checkpoint)
        raise


def add_factors_and_propagate(
    store: ResidualStore,
    factors: tuple[ResidualFactor, ...],
) -> ResidualPropagationResult:
    checkpoint = store.checkpoint()
    try:
        factor_keys = tuple(store.add_factor(factor) for factor in factors)
        affected_variables = tuple(
            var
            for factor in factors
            for var in factor.scope
        )
        if affected_variables:
            component = store._component_from_variables(affected_variables)
        else:
            component = _ResidualLiveComponent(
                variables=(),
                factor_keys=factor_keys,
            )
        result = store._propagate_component(component)
        if result.kind is ResidualPropagationKind.CONTRADICTION:
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

    factor_items: list[tuple[ResidualFactorKey, tuple[VarId, ...]]] = []
    zero_scope_factors: list[ResidualFactorKey] = []
    for index, factor_snapshot in enumerate(snapshot.factors):
        scope = tuple(getattr(factor_snapshot, "scope", ()))
        factor_key = getattr(
            factor_snapshot,
            "key",
            ResidualFactorKey("snapshot_factor", (index,)),
        )
        unknown_scope = frozenset(scope) - known
        if unknown_scope:
            raise ValueError(
                "residual factor references unknown variable: "
                f"{tuple(sorted(unknown_scope, key=_var_sort_key))!r}"
            )
        factor_items.append((factor_key, scope))
        if not scope:
            zero_scope_factors.append(factor_key)
            continue
        first = scope[0]
        for var in scope[1:]:
            union(first, var)

    variables_by_root: dict[VarId, list[VarId]] = {}
    for var in domain_vars:
        variables_by_root.setdefault(find(var), []).append(var)
    factor_keys_by_root: dict[VarId, list[ResidualFactorKey]] = {
        root: [] for root in variables_by_root
    }
    for factor_key, scope in factor_items:
        if not scope:
            continue
        factor_keys_by_root.setdefault(find(scope[0]), []).append(factor_key)

    components: list[ResidualConstraintComponentSnapshot] = []
    for root, variables in variables_by_root.items():
        sorted_variables = tuple(sorted(variables, key=_var_sort_key))
        assigned = tuple(var for var in sorted_variables if var in assignment_vars)
        components.append(
            ResidualConstraintComponentSnapshot(
                variables=sorted_variables,
                factor_keys=tuple(
                    sorted(
                        factor_keys_by_root.get(root, ()),
                        key=_factor_key_sort_key,
                    )
                ),
                assigned_variables=assigned,
            )
        )
    for factor_key in zero_scope_factors:
        components.append(
            ResidualConstraintComponentSnapshot(
                variables=(),
                factor_keys=(factor_key,),
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
                component.factor_keys,
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
                factor_keys=component.factor_keys,
            )
        result = store._propagate_component(live_component)
        if result.kind is ResidualPropagationKind.CONTRADICTION:
            return False
        if result.kind is ResidualPropagationKind.LOCALLY_CONSISTENT_UNCERTIFIED:
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

        if isinstance(factor_snapshot, TetraTokenParityFactorValueSnapshot):
            if len(scope) != 2:
                raise ValueError(
                    "tetra token/parity factor snapshot must have binary scope"
                )
            continue

        if isinstance(
            factor_snapshot,
            (
                DirectionalSiteFactorValueSnapshot,
                DirectionalBondEmissionFactorValueSnapshot,
            ),
        ):
            continue

        raise ValueError(f"unknown residual factor snapshot: {factor_snapshot!r}")


@dataclass(frozen=True, slots=True)
class TetraResidualFactor:
    scope: tuple[VarId, ...]
    status: SiteStatus
    target: TetraValue
    reference_order: tuple[OccurrenceId, ...]
    local_order: tuple[OccurrenceId, ...]
    key: ResidualFactorKey = field(
        default_factory=lambda: ResidualFactorKey("tetra", ()),
    )

    def __post_init__(self) -> None:
        if len(self.scope) != 1:
            raise ValueError("tetra residual factor must have unary scope")
        if self.key == ResidualFactorKey("tetra", ()):
            object.__setattr__(
                self,
                "key",
                ResidualFactorKey(
                    "tetra",
                    (
                        self.scope,
                        self.status.value,
                        self.target.value,
                        tuple(int(item) for item in self.reference_order),
                        tuple(int(item) for item in self.local_order),
                    ),
                ),
            )

    @property
    def token_var(self) -> VarId:
        return self.scope[0]

    def accepts(self, row: tuple[object, ...]) -> bool:
        return len(row) == 1 and row[0] in self.allowed_tokens()

    def value_snapshot(self) -> TetraResidualFactorValueSnapshot:
        return TetraResidualFactorValueSnapshot(
            key=self.key,
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
    key: ResidualFactorKey = field(
        default_factory=lambda: ResidualFactorKey("directional", ()),
    )

    def __post_init__(self) -> None:
        if len(set(self.scope)) != len(self.scope):
            raise ValueError("directional residual scope contains duplicates")
        if set(self.scope) != set(self.carrier_models):
            raise ValueError("directional residual scope/model mismatch")
        if self.key == ResidualFactorKey("directional", ()):
            object.__setattr__(
                self,
                "key",
                ResidualFactorKey(
                    "directional",
                    (
                        self.scope,
                        self.status.value,
                        self.target.value,
                        tuple(
                            (
                                var,
                                model.side,
                                model.orientation,
                                model.ligand_factor,
                            )
                            for var, model in sorted(
                                self.carrier_models.items(),
                                key=lambda item: _var_sort_key(item[0]),
                            )
                        ),
                    ),
                ),
            )

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
            key=self.key,
            scope=self.scope,
            status=self.status,
            target=self.target,
            carrier_models=tuple(
                sorted(self.carrier_models.items(), key=lambda item: _var_sort_key(item[0]))
            ),
        )


@dataclass(frozen=True, slots=True)
class TetraTokenParityFactor:
    key: ResidualFactorKey
    scope: tuple[VarId, VarId]
    status: SiteStatus
    target: TetraValue

    def __post_init__(self) -> None:
        if len(self.scope) != 2:
            raise ValueError("tetra token/parity factor must have binary scope")

    def accepts(self, row: tuple[object, ...]) -> bool:
        if len(row) != 2:
            return False
        token, parity = row
        if token not in _TETRA_TOKENS:
            return False
        if parity not in (TetraLocalParity.EVEN, TetraLocalParity.ODD):
            return False

        if self.status is SiteStatus.UNSPECIFIED:
            return token is TetraToken.NONE

        if self.target is TetraValue.PLUS:
            return (
                (token is TetraToken.AT and parity is TetraLocalParity.EVEN)
                or (token is TetraToken.ATAT and parity is TetraLocalParity.ODD)
            )

        if self.target is TetraValue.MINUS:
            return (
                (token is TetraToken.ATAT and parity is TetraLocalParity.EVEN)
                or (token is TetraToken.AT and parity is TetraLocalParity.ODD)
            )

        return False

    def value_snapshot(self) -> TetraTokenParityFactorValueSnapshot:
        return TetraTokenParityFactorValueSnapshot(
            key=self.key,
            scope=self.scope,
            status=self.status,
            target=self.target,
        )


@dataclass(frozen=True, slots=True)
class DirectionalSiteFactor:
    key: ResidualFactorKey
    scope: tuple[VarId, ...]
    sides: tuple[tuple[VarId, Literal["left", "right"]], ...]
    status: SiteStatus
    target: DirectionalValue

    def __post_init__(self) -> None:
        if len(set(self.scope)) != len(self.scope):
            raise ValueError("directional site scope contains duplicates")
        if set(var for var, _ in self.sides) != set(self.scope):
            raise ValueError("directional site scope/side mismatch")

    def accepts(self, row: tuple[object, ...]) -> bool:
        if len(row) != len(self.scope):
            return False
        if any(sign not in _DIRECTIONAL_NORMALIZED_SIGNS for sign in row):
            return False

        side_by_var = dict(self.sides)
        left: list[int] = []
        right: list[int] = []
        for var, sign in zip(self.scope, row):
            if sign is DirectionalNormalizedSign.ABSENT:
                continue
            numeric = _normalized_sign_value(sign)
            if side_by_var[var] == "left":
                left.append(numeric)
            else:
                right.append(numeric)

        if len(set(left)) > 1 or len(set(right)) > 1:
            return False
        if not left and not right:
            value = DirectionalValue.NONE
        elif not left or not right:
            value = DirectionalValue.NONE
        else:
            value = (
                DirectionalValue.TOGETHER
                if left[0] == right[0]
                else DirectionalValue.OPPOSITE
            )

        if self.status is SiteStatus.UNSPECIFIED:
            return value is DirectionalValue.NONE

        return value is self.target

    def value_snapshot(self) -> DirectionalSiteFactorValueSnapshot:
        return DirectionalSiteFactorValueSnapshot(
            key=self.key,
            scope=self.scope,
            sides=tuple(sorted(self.sides, key=lambda item: _var_sort_key(item[0]))),
            status=self.status,
            target=self.target,
        )


@dataclass(frozen=True, slots=True)
class DirectionalBondEmissionFactor:
    key: ResidualFactorKey
    scope: tuple[VarId, ...]
    models: tuple[DirectionalSiteCarrierModel, ...]
    allowed_marks: tuple[DirectionMark, ...]

    def __post_init__(self) -> None:
        if len(self.scope) != len(self.models):
            raise ValueError("directional bond-emission scope/model mismatch")
        if len(set(self.scope)) != len(self.scope):
            raise ValueError("directional bond-emission scope contains duplicates")

    def accepts(self, row: tuple[object, ...]) -> bool:
        if len(row) != len(self.scope):
            return False
        if any(sign not in _DIRECTIONAL_NORMALIZED_SIGNS for sign in row):
            return False
        return any(
            row == tuple(
                normalized_sign_from_mark(
                    mark=mark,
                    canonical_orientation=orientation,
                    model=model,
                )
                for model in self.models
            )
            for orientation in (-1, 1)
            for mark in self.allowed_marks
        )

    def value_snapshot(self) -> DirectionalBondEmissionFactorValueSnapshot:
        return DirectionalBondEmissionFactorValueSnapshot(
            key=self.key,
            scope=self.scope,
            models=self.models,
            allowed_marks=self.allowed_marks,
        )


def tetra_var(center: object) -> VarId:
    return VarId("tetra", (center,))


def direction_var(carrier: object) -> VarId:
    return VarId("direction", (carrier,))


def tetra_token_var(site: SiteId) -> VarId:
    return VarId("tetra_token", (int(site),))


def tetra_parity_var(site: SiteId) -> VarId:
    return VarId("tetra_local_parity", (int(site),))


def directional_site_carrier_var(
    site: SiteId,
    bond: BondId,
) -> VarId:
    return VarId("directional_site_carrier", (int(site), int(bond)))


def normalized_sign_from_mark(
    *,
    mark: DirectionMark,
    canonical_orientation: Literal[-1, 1],
    model: DirectionalSiteCarrierModel,
) -> DirectionalNormalizedSign:
    if mark is DirectionMark.ABSENT:
        return DirectionalNormalizedSign.ABSENT
    sign = (
        _mark_sign(mark)
        * canonical_orientation
        * model.endpoint_orientation_factor
        * model.ligand_factor
    )
    return (
        DirectionalNormalizedSign.POSITIVE
        if sign == 1
        else DirectionalNormalizedSign.NEGATIVE
    )


def _var_sort_key(var: VarId) -> tuple[str, tuple[str, ...]]:
    return (var.kind, tuple(repr(item) for item in var.key))


def _factor_key_sort_key(
    key: ResidualFactorKey,
) -> tuple[str, tuple[str, ...]]:
    return (key.kind, tuple(repr(item) for item in key.key))


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


def _normalized_sign_value(sign: DirectionalNormalizedSign) -> int:
    if sign is DirectionalNormalizedSign.POSITIVE:
        return 1
    if sign is DirectionalNormalizedSign.NEGATIVE:
        return -1
    raise ValueError(f"normalized sign has no numeric value: {sign!r}")


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
            key=snapshot.key,
        )
    if isinstance(snapshot, DirectionalResidualFactorValueSnapshot):
        return DirectionalResidualFactor(
            scope=snapshot.scope,
            status=snapshot.status,
            target=snapshot.target,
            carrier_models=dict(snapshot.carrier_models),
            key=snapshot.key,
        )
    if isinstance(snapshot, TetraTokenParityFactorValueSnapshot):
        return TetraTokenParityFactor(
            key=snapshot.key,
            scope=snapshot.scope,
            status=snapshot.status,
            target=snapshot.target,
        )
    if isinstance(snapshot, DirectionalSiteFactorValueSnapshot):
        return DirectionalSiteFactor(
            key=snapshot.key,
            scope=snapshot.scope,
            sides=snapshot.sides,
            status=snapshot.status,
            target=snapshot.target,
        )
    if isinstance(snapshot, DirectionalBondEmissionFactorValueSnapshot):
        return DirectionalBondEmissionFactor(
            key=snapshot.key,
            scope=snapshot.scope,
            models=snapshot.models,
            allowed_marks=snapshot.allowed_marks,
        )
    raise ValueError(f"unknown residual factor snapshot: {snapshot!r}")


_TETRA_TOKENS = frozenset((TetraToken.NONE, TetraToken.AT, TetraToken.ATAT))
_DIRECTION_MARKS = frozenset((DirectionMark.ABSENT, DirectionMark.FWD, DirectionMark.REV))
_DIRECTIONAL_NORMALIZED_SIGNS = frozenset(
    (
        DirectionalNormalizedSign.ABSENT,
        DirectionalNormalizedSign.POSITIVE,
        DirectionalNormalizedSign.NEGATIVE,
    )
)


__all__ = (
    "DirectionalCarrierResidual",
    "DirectionalBondEmissionFactor",
    "DirectionalBondEmissionFactorValueSnapshot",
    "DirectionalNormalizedSign",
    "DirectionalResidualFactor",
    "DirectionalResidualFactorValueSnapshot",
    "DirectionalSiteCarrierModel",
    "DirectionalSiteFactor",
    "DirectionalSiteFactorValueSnapshot",
    "ResidualConstraintComponentSnapshot",
    "ResidualFactorKey",
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
    "TetraLocalParity",
    "TetraTokenParityFactor",
    "TetraTokenParityFactorValueSnapshot",
    "VarId",
    "add_factor_and_propagate",
    "add_factors_and_propagate",
    "direction_var",
    "directional_site_carrier_var",
    "normalized_sign_from_mark",
    "residual_store_assignments_have_support",
    "residual_store_constraint_components",
    "residual_store_projected_values",
    "tetra_var",
    "tetra_parity_var",
    "tetra_token_var",
)
