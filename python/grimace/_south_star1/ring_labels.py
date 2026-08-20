"""Bounded ring-label relations for the private proof kernel."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from .ids import BondId
from .ids import RingEndpointId
from .policy import RingLabel
from .policy import SmilesPolicy
from .slots import SlotBundle


@dataclass(frozen=True, slots=True)
class RingLabelInterval:
    bond: BondId
    label: RingLabel
    endpoint_1: RingEndpointId
    endpoint_2: RingEndpointId
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _RingInterval:
    bond: BondId
    endpoint_1: RingEndpointId
    endpoint_2: RingEndpointId
    start: int
    end: int


def enumerate_ring_label_assignments(
    *,
    slots: SlotBundle,
    policy: SmilesPolicy,
) -> Iterator[dict[RingEndpointId, RingLabel]]:
    """Enumerate bounded ring-label assignments for one slot bundle.

    The validator is the source of truth for endpoint coverage, same-label
    pairing, label-domain membership, interval non-overlap, and least-free
    normalization. This generator mirrors that policy by construction and
    validates each yielded complete assignment.
    """

    if not slots.ring_endpoints:
        yield {}
        return

    intervals = _ring_intervals(slots)
    labels = policy.ring_labels

    out: dict[RingEndpointId, RingLabel] = {}
    chosen: list[tuple[int, int, RingLabel]] = []

    def active_labels_at(position: int) -> set[RingLabel]:
        return {
            label
            for start, end, label in chosen
            if start < position < end
        }

    def rec(index: int) -> Iterator[dict[RingEndpointId, RingLabel]]:
        if index == len(intervals):
            validate_bounded_ring_labels(policy, slots, out)
            yield dict(out)
            return

        interval = intervals[index]
        active = active_labels_at(interval.start)
        candidates = tuple(
            label
            for label in labels
            if label not in active
        )

        if policy.least_free_ring_labels:
            if not candidates:
                return
            candidates = (min(candidates, key=lambda label: label.value),)

        for label in candidates:
            out[interval.endpoint_1] = label
            out[interval.endpoint_2] = label
            chosen.append((interval.start, interval.end, label))

            yield from rec(index + 1)

            chosen.pop()
            del out[interval.endpoint_1]
            del out[interval.endpoint_2]

    yield from rec(0)


def validate_bounded_ring_labels(
    policy: SmilesPolicy,
    slots: SlotBundle,
    labels: dict[RingEndpointId, RingLabel],
) -> tuple[RingLabelInterval, ...]:
    endpoint_ids = {endpoint.id for endpoint in slots.ring_endpoints}
    if set(labels) != endpoint_ids:
        missing = endpoint_ids - set(labels)
        extra = set(labels) - endpoint_ids
        raise ValueError(
            "ring label endpoint coverage mismatch: "
            f"missing={missing!r}, extra={extra!r}"
        )

    policy_labels = set(policy.ring_labels)
    out_of_domain = {
        label for label in labels.values() if label not in policy_labels
    }
    if out_of_domain:
        raise ValueError(f"ring labels outside policy domain: {out_of_domain!r}")

    endpoints_by_bond: dict[BondId, list[tuple[RingEndpointId, int]]] = {}
    for endpoint in slots.ring_endpoints:
        endpoints_by_bond.setdefault(endpoint.bond, []).append(
            (endpoint.id, endpoint.syntax_position)
        )

    intervals: list[RingLabelInterval] = []
    for bond, endpoints in endpoints_by_bond.items():
        if len(endpoints) != 2:
            raise ValueError(f"ring bond {bond!r} does not have two endpoints")
        (endpoint_1, position_1), (endpoint_2, position_2) = endpoints
        label = labels[endpoint_1]
        if labels[endpoint_2] != label:
            raise ValueError(f"ring bond {bond!r} endpoints use different labels")
        start, end = sorted((position_1, position_2))
        intervals.append(
            RingLabelInterval(
                bond=bond,
                label=label,
                endpoint_1=endpoint_1,
                endpoint_2=endpoint_2,
                start=start,
                end=end,
            )
        )

    sorted_intervals = tuple(
        sorted(intervals, key=lambda interval: (interval.start, int(interval.bond)))
    )
    _validate_label_reuse(sorted_intervals)
    if policy.least_free_ring_labels:
        _validate_least_free_labels(policy, sorted_intervals)
    return sorted_intervals


def _validate_label_reuse(intervals: tuple[RingLabelInterval, ...]) -> None:
    for i, left in enumerate(intervals):
        for right in intervals[i + 1 :]:
            if left.label != right.label:
                continue
            if left.start < right.end and right.start < left.end:
                raise ValueError(
                    f"ring label {left.label!r} has overlapping intervals "
                    f"for bonds {left.bond!r} and {right.bond!r}"
                )


def _ring_intervals(slots: SlotBundle) -> tuple[_RingInterval, ...]:
    by_bond: dict[BondId, list[object]] = {}

    for endpoint in slots.ring_endpoints:
        by_bond.setdefault(endpoint.bond, []).append(endpoint)

    intervals: list[_RingInterval] = []

    for bond, endpoints in by_bond.items():
        if len(endpoints) != 2:
            raise ValueError(
                f"ring bond {bond!r} has {len(endpoints)} endpoint slots, not two"
            )

        endpoint_1, endpoint_2 = sorted(
            endpoints,
            key=lambda endpoint: endpoint.syntax_position,
        )

        intervals.append(
            _RingInterval(
                bond=bond,
                endpoint_1=endpoint_1.id,
                endpoint_2=endpoint_2.id,
                start=endpoint_1.syntax_position,
                end=endpoint_2.syntax_position,
            )
        )

    return tuple(
        sorted(
            intervals,
            key=lambda interval: (interval.start, int(interval.bond)),
        )
    )


def _validate_least_free_labels(
    policy: SmilesPolicy,
    intervals: tuple[RingLabelInterval, ...],
) -> None:
    for interval in intervals:
        active = {
            other.label
            for other in intervals
            if other.start < interval.start < other.end
        }
        candidates = tuple(
            label
            for label in policy.ring_labels
            if label not in active
        )
        if not candidates:
            raise ValueError(
                f"no free ring label for interval on bond {interval.bond!r}"
            )

        expected = min(candidates, key=lambda label: label.value)
        if interval.label != expected:
            raise ValueError(
                f"ring bond {interval.bond!r} violates least-free label policy: "
                f"expected {expected!r}, got {interval.label!r}"
            )


__all__ = (
    "enumerate_ring_label_assignments",
    "RingLabelInterval",
    "validate_bounded_ring_labels",
)
