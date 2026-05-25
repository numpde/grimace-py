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
        self._domains[var] = domain

    def add_factor(self, factor: ResidualFactor) -> int:
        factor_id = len(self._factors)
        self._factors.append(factor)
        self._factor_by_id[factor_id] = factor
        for var in factor.scope:
            if var not in self._domains:
                raise ValueError(f"factor references unknown variable: {var!r}")
            self._factors_by_var.setdefault(var, []).append(factor)
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
            domains=tuple(self._domains.items()),
            assignments=tuple(self._assignments.items()),
            factors=tuple(factor.value_snapshot() for factor in self._factors),
        )

    @classmethod
    def from_value_snapshot(
        cls,
        snapshot: ResidualStoreValueSnapshot,
    ) -> "ResidualStore":
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
                sorted(self.carrier_models.items(), key=lambda item: repr(item[0]))
            ),
            marks=tuple(sorted(self._marks.items(), key=lambda item: repr(item[0]))),
        )


def tetra_var(center: object) -> VarId:
    return VarId("tetra", (center,))


def direction_var(carrier: object) -> VarId:
    return VarId("direction", (carrier,))


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
    "ResidualFactor",
    "ResidualStore",
    "ResidualStoreValueSnapshot",
    "ResidualVariable",
    "TetraResidualFactor",
    "TetraResidualFactorValueSnapshot",
    "VarId",
    "direction_var",
    "tetra_var",
)
