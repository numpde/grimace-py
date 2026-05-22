"""Bounded ring-label relations for the private proof kernel."""

from __future__ import annotations

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

    _validate_label_reuse(intervals)
    return tuple(sorted(intervals, key=lambda interval: interval.start))


def _validate_label_reuse(intervals: list[RingLabelInterval]) -> None:
    for i, left in enumerate(intervals):
        for right in intervals[i + 1 :]:
            if left.label != right.label:
                continue
            if left.start < right.end and right.start < left.end:
                raise ValueError(
                    f"ring label {left.label!r} has overlapping intervals "
                    f"for bonds {left.bond!r} and {right.bond!r}"
                )


__all__ = (
    "RingLabelInterval",
    "validate_bounded_ring_labels",
)
