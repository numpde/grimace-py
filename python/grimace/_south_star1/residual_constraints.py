"""Reversible residual constraints for online South Star enumeration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
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

    def assign(self, var: VarId, value: object) -> bool: ...
    def close(self) -> bool: ...
    def checkpoint(self) -> object: ...
    def rollback(self, token: object) -> None: ...
    def value_snapshot(self) -> object: ...


@dataclass(frozen=True, slots=True)
class TetraResidualFactorValueSnapshot:
    scope: tuple[VarId, ...]
    status: SiteStatus
    target: TetraValue
    reference_order: tuple[OccurrenceId, ...]
    local_order: tuple[OccurrenceId, ...]
    assigned: object


@dataclass(frozen=True, slots=True)
class DirectionalResidualFactorValueSnapshot:
    scope: tuple[VarId, ...]
    status: SiteStatus
    target: DirectionalValue
    carrier_models: tuple[tuple[VarId, DirectionalCarrierResidual], ...]
    marks: tuple[tuple[VarId, object], ...]


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


class ResidualStore:
    """Trail-based reversible store for online DFS branches."""

    def __init__(self) -> None:
        self._domains: dict[VarId, tuple[object, ...]] = {}
        self._assignments: dict[VarId, object] = {}
        self._factors: list[ResidualFactor] = []
        self._factor_by_id: dict[int, ResidualFactor] = {}
        self._factors_by_var: dict[VarId, list[ResidualFactor]] = {}
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
        for var in factor.scope:
            if var not in self._domains:
                raise ValueError(f"factor references unknown variable: {var!r}")
        factor_token = factor.checkpoint()
        factor_id = len(self._factors)
        self._factors.append(factor)
        self._factor_by_id[factor_id] = factor
        for var in factor.scope:
            self._factors_by_var.setdefault(var, []).append(factor)
        self._trail.append(("factor_add", factor_id, factor, factor_token))
        return factor_id

    def assign(self, var: VarId, value: object) -> bool:
        if var not in self._domains:
            raise ValueError(f"unknown residual variable: {var!r}")
        if value not in self._domains[var]:
            return False
        existing = self._assignments.get(var, _UNASSIGNED)
        if existing is not _UNASSIGNED:
            return existing == value

        checkpoint = self.checkpoint()
        self._assignments[var] = value
        self._trail.append(("assignment", var))
        for factor in self._factors_by_var.get(var, ()):
            token = factor.checkpoint()
            self._trail.append(("factor", factor, token))
            if not factor.assign(var, value):
                self.rollback(checkpoint)
                return False
        return True

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
            if entry[0] == "factor":
                _, factor, token = entry
                factor.rollback(token)
                continue
            if entry[0] == "factor_add":
                _, factor_id, factor, factor_token = entry
                factor.rollback(factor_token)
                _remove_factor(self, factor, factor_id)
                continue
            raise AssertionError(f"unknown residual trail entry: {entry!r}")

    def close_factor(self, factor_id: object) -> bool:
        if isinstance(factor_id, int):
            factor = self._factor_by_id.get(factor_id)
        else:
            factor = factor_id if factor_id in self._factors else None
        if factor is None:
            raise ValueError(f"unknown residual factor: {factor_id!r}")
        return factor.close()

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
        for factor in store._factors:
            for var in factor.scope:
                if var not in store._domains:
                    raise ValueError(f"factor snapshot references unknown variable: {var!r}")
                store._factors_by_var.setdefault(var, []).append(factor)
        store._trail = []
        return store


def add_factor_checked(store: ResidualStore, factor: ResidualFactor) -> bool:
    checkpoint = store.checkpoint()
    try:
        store.add_factor(factor)
        for var in factor.scope:
            assigned = store._assignments.get(var, _UNASSIGNED)
            if assigned is _UNASSIGNED:
                continue
            if not factor.assign(var, assigned):
                store.rollback(checkpoint)
                return False
        return True
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
    domains = _residual_snapshot_domain_map(snapshot)
    if var not in domains:
        raise ValueError(f"unknown residual variable: {var!r}")
    component = next(
        component
        for component in residual_store_constraint_components(snapshot)
        if var in component.variables
    )
    assigned = dict(snapshot.assignments)
    candidates = (assigned[var],) if var in assigned else domains[var]
    return tuple(
        value
        for value in candidates
        if _residual_component_has_solution(
            snapshot,
            component,
            ((var, value),),
        )
    )


def residual_store_assignments_have_support(
    snapshot: ResidualStoreValueSnapshot,
    assignments: tuple[tuple[VarId, object], ...],
) -> bool:
    domains = _residual_snapshot_domain_map(snapshot)
    required: dict[VarId, object] = {}
    for var, value in assignments:
        if var not in domains:
            raise ValueError(f"unknown residual variable: {var!r}")
        if value not in domains[var]:
            return False
        existing = required.get(var, _UNASSIGNED)
        if existing is not _UNASSIGNED and existing != value:
            return False
        required[var] = value

    existing_assignments = dict(snapshot.assignments)
    for var, value in required.items():
        existing = existing_assignments.get(var, _UNASSIGNED)
        if existing is not _UNASSIGNED and existing != value:
            return False

    for component in residual_store_constraint_components(snapshot):
        component_required = tuple(
            (var, value)
            for var, value in required.items()
            if var in component.variables
        )
        if not _residual_component_has_solution(
            snapshot,
            component,
            component_required,
        ):
            return False
    return True


def _residual_component_has_solution(
    snapshot: ResidualStoreValueSnapshot,
    component: ResidualConstraintComponentSnapshot,
    required_assignments: tuple[tuple[VarId, object], ...],
) -> bool:
    domains = _residual_snapshot_domain_map(snapshot)
    fixed = dict(snapshot.assignments)
    for var, value in required_assignments:
        if var not in domains:
            raise ValueError(f"unknown residual variable: {var!r}")
        if value not in domains[var]:
            return False
        existing = fixed.get(var, _UNASSIGNED)
        if existing is not _UNASSIGNED and existing != value:
            return False
        fixed[var] = value

    store = ResidualStore.from_value_snapshot(snapshot)
    ordered_choices = tuple(
        (var, (fixed[var],) if var in fixed else domains[var])
        for var in component.variables
    )

    def search(index: int) -> bool:
        if index == len(ordered_choices):
            return all(
                store.close_factor(factor_index)
                for factor_index in component.factor_indexes
            )
        current_var, values = ordered_choices[index]
        for value in values:
            checkpoint = store.checkpoint()
            try:
                if store.assign(current_var, value) and search(index + 1):
                    return True
            finally:
                store.rollback(checkpoint)
        return False

    return search(0)


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
            var = scope[0]
            factor_value = factor_snapshot.assigned
            top_value = assignments.get(var, _UNASSIGNED)
            if factor_value is _UNASSIGNED:
                if top_value is not _UNASSIGNED:
                    raise ValueError(
                        "tetra factor missing assigned value for assigned "
                        f"variable: {var!r}"
                    )
                continue
            if factor_value not in domains[var]:
                raise ValueError(
                    "tetra factor assigned value outside domain: "
                    f"{var!r}={factor_value!r}"
                )
            if top_value is _UNASSIGNED:
                raise ValueError(
                    f"tetra factor assignment missing from residual snapshot: {var!r}"
                )
            if top_value != factor_value:
                raise ValueError(
                    "tetra factor assignment disagrees with residual "
                    f"snapshot: {var!r}"
                )
            continue

        if isinstance(factor_snapshot, DirectionalResidualFactorValueSnapshot):
            marks = dict(factor_snapshot.marks)
            if len(marks) != len(factor_snapshot.marks):
                raise ValueError("duplicate directional residual factor mark")
            for var, value in factor_snapshot.marks:
                if var not in scope:
                    raise ValueError(
                        "directional factor mark references out-of-scope "
                        f"variable: {var!r}"
                    )
                if var not in domains:
                    raise ValueError(
                        "directional factor mark references unknown "
                        f"variable: {var!r}"
                    )
                if value not in domains[var]:
                    raise ValueError(
                        "directional factor mark outside domain: "
                        f"{var!r}={value!r}"
                    )
            for var in scope:
                top_value = assignments.get(var, _UNASSIGNED)
                mark_value = marks.get(var, _UNASSIGNED)
                if top_value is _UNASSIGNED and mark_value is _UNASSIGNED:
                    continue
                if top_value is _UNASSIGNED:
                    raise ValueError(
                        "directional factor mark missing from residual "
                        f"snapshot: {var!r}"
                    )
                if mark_value is _UNASSIGNED:
                    raise ValueError(
                        "directional factor missing mark for assigned "
                        f"variable: {var!r}"
                    )
                if top_value != mark_value:
                    raise ValueError(
                        "directional factor mark disagrees with residual "
                        f"snapshot: {var!r}"
                    )
            continue

        raise ValueError(f"unknown residual factor snapshot: {factor_snapshot!r}")


def _remove_factor(
    store: ResidualStore,
    factor: ResidualFactor,
    factor_id: int,
) -> None:
    if factor_id in store._factor_by_id:
        del store._factor_by_id[factor_id]
    if factor_id == len(store._factors) - 1 and store._factors[factor_id] is factor:
        store._factors.pop()
    elif factor in store._factors:
        store._factors.remove(factor)
    for var in factor.scope:
        factors = store._factors_by_var.get(var)
        if factors is None:
            continue
        store._factors_by_var[var] = [item for item in factors if item is not factor]
        if not store._factors_by_var[var]:
            del store._factors_by_var[var]


@dataclass(frozen=True, slots=True)
class TetraResidualFactor:
    scope: tuple[VarId, ...]
    status: SiteStatus
    target: TetraValue
    reference_order: tuple[OccurrenceId, ...]
    local_order: tuple[OccurrenceId, ...]
    _assigned: object = field(default=_UNASSIGNED, init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.scope) != 1:
            raise ValueError("tetra residual factor must have unary scope")
        object.__setattr__(self, "_assigned", _UNASSIGNED)

    @property
    def token_var(self) -> VarId:
        return self.scope[0]

    def assign(self, var: VarId, value: object) -> bool:
        if var != self.token_var:
            return True
        if value not in _TETRA_TOKENS:
            return False
        existing = self._assigned
        if existing is not _UNASSIGNED:
            return existing is value
        if value not in self.allowed_tokens():
            return False
        object.__setattr__(self, "_assigned", value)
        return True

    def close(self) -> bool:
        return self._assigned is not _UNASSIGNED and self._assigned in self.allowed_tokens()

    def checkpoint(self) -> object:
        return self._assigned

    def rollback(self, token: object) -> None:
        object.__setattr__(self, "_assigned", token)

    def value_snapshot(self) -> TetraResidualFactorValueSnapshot:
        return TetraResidualFactorValueSnapshot(
            scope=self.scope,
            status=self.status,
            target=self.target,
            reference_order=self.reference_order,
            local_order=self.local_order,
            assigned=self._assigned,
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
    _marks: dict[VarId, object] = field(
        default_factory=dict,
        init=False,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if set(self.scope) != set(self.carrier_models):
            raise ValueError("directional residual scope/model mismatch")
        object.__setattr__(self, "_marks", {})

    def assign(self, var: VarId, value: object) -> bool:
        if var not in self.carrier_models:
            return True
        if value not in _DIRECTION_MARKS:
            return False
        marks = dict(self._marks)
        existing = marks.get(var, _UNASSIGNED)
        if existing is not _UNASSIGNED:
            return existing is value
        marks[var] = value
        if _directional_value(marks, self.carrier_models) is _INVALID:
            return False
        object.__setattr__(self, "_marks", marks)
        return True

    def close(self) -> bool:
        if set(self._marks) != set(self.scope):
            return False
        value = self.value()
        if value is _INVALID:
            return False
        if self.status is SiteStatus.UNSPECIFIED:
            return value is DirectionalValue.NONE
        return value is self.target

    def checkpoint(self) -> object:
        return dict(self._marks)

    def rollback(self, token: object) -> None:
        object.__setattr__(self, "_marks", dict(token))

    def value(self) -> DirectionalValue | object:
        return _directional_value(self._marks, self.carrier_models)

    def value_snapshot(self) -> DirectionalResidualFactorValueSnapshot:
        return DirectionalResidualFactorValueSnapshot(
            scope=self.scope,
            status=self.status,
            target=self.target,
            carrier_models=tuple(
                sorted(self.carrier_models.items(), key=lambda item: _var_sort_key(item[0]))
            ),
            marks=tuple(sorted(self._marks.items(), key=lambda item: _var_sort_key(item[0]))),
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
        factor = TetraResidualFactor(
            scope=snapshot.scope,
            status=snapshot.status,
            target=snapshot.target,
            reference_order=snapshot.reference_order,
            local_order=snapshot.local_order,
        )
        factor.rollback(snapshot.assigned)
        return factor
    if isinstance(snapshot, DirectionalResidualFactorValueSnapshot):
        factor = DirectionalResidualFactor(
            scope=snapshot.scope,
            status=snapshot.status,
            target=snapshot.target,
            carrier_models=dict(snapshot.carrier_models),
        )
        factor.rollback(dict(snapshot.marks))
        return factor
    raise ValueError(f"unknown residual factor snapshot: {snapshot!r}")


_TETRA_TOKENS = frozenset((TetraToken.NONE, TetraToken.AT, TetraToken.ATAT))
_DIRECTION_MARKS = frozenset((DirectionMark.ABSENT, DirectionMark.FWD, DirectionMark.REV))


__all__ = (
    "DirectionalCarrierResidual",
    "DirectionalResidualFactor",
    "DirectionalResidualFactorValueSnapshot",
    "ResidualConstraintComponentSnapshot",
    "ResidualFactor",
    "ResidualStore",
    "ResidualStoreValueSnapshot",
    "ResidualVariable",
    "TetraResidualFactor",
    "TetraResidualFactorValueSnapshot",
    "VarId",
    "add_factor_checked",
    "direction_var",
    "residual_store_assignments_have_support",
    "residual_store_constraint_components",
    "residual_store_projected_values",
    "tetra_var",
)
