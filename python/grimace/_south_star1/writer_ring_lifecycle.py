"""Ring-label lifecycle event derivation for writer transition streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .writer_events import WriterEvent
from .writer_events import WriterRingEndpointEmitted
from .writer_events import WriterRingEndpointPaired
from .writer_events import WriterRingLabelAllocated
from .writer_events import WriterRingLabelReleased
from .writer_state import WriterClosureLabel


WriterRingLabelAllocationSource = Literal["fresh", "reused"]


@dataclass(frozen=True, slots=True)
class WriterRingLifecycleTransitionViolation:
    kind: str
    label: WriterClosureLabel | None = None
    message: str = ""


def writer_events_with_ring_label_lifecycle(
    *,
    source_state: object,
    events: tuple[WriterEvent, ...],
) -> tuple[WriterEvent, ...]:
    """Return ``events`` with ring-label lifecycle evidence made explicit."""

    allocated = _event_labels(events, WriterRingLabelAllocated)
    released = _event_labels(events, WriterRingLabelReleased)
    result: list[WriterEvent] = []

    for event in events:
        if isinstance(event, WriterRingEndpointEmitted):
            if event.label not in allocated:
                result.append(
                    WriterRingLabelAllocated(
                        label=event.label,
                        source=writer_ring_label_allocation_source(
                            source_state=source_state,
                            label=event.label,
                        ),
                    )
                )
            result.append(event)
        elif isinstance(event, WriterRingEndpointPaired):
            result.append(event)
            if event.label not in released:
                result.append(WriterRingLabelReleased(label=event.label))
        else:
            result.append(event)

    return tuple(result)


def writer_ring_label_allocation_source(
    *,
    source_state: object,
    label: WriterClosureLabel,
) -> WriterRingLabelAllocationSource:
    return "reused" if label in _labels(source_state, "reusable") else "fresh"


def validate_writer_ring_lifecycle_transition(
    *,
    source_state: object,
    successor_state: object,
    events: tuple[object, ...],
) -> None:
    violations = writer_ring_lifecycle_transition_violations(
        source_state=source_state,
        successor_state=successor_state,
        events=events,
    )
    if not violations:
        return

    first = violations[0]
    detail = f": {first.message}" if first.message else ""
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer ring lifecycle transition violation: {first.kind}{detail}",
    )


def writer_ring_lifecycle_transition_violations(
    *,
    source_state: object,
    successor_state: object,
    events: tuple[object, ...],
) -> tuple[WriterRingLifecycleTransitionViolation, ...]:
    indexed = tuple(enumerate(events))
    opens = _indexed(indexed, WriterRingEndpointEmitted)
    pairs = _indexed(indexed, WriterRingEndpointPaired)
    allocations = _indexed(indexed, WriterRingLabelAllocated)
    releases = _indexed(indexed, WriterRingLabelReleased)
    violations: list[WriterRingLifecycleTransitionViolation] = []

    for index, event in opens:
        prior = _matching(allocations, event.label, before=index)
        _require_one(
            violations,
            prior,
            event.label,
            missing_kind="missing_open_label_allocation",
            duplicate_kind="duplicate_open_label_allocation",
        )
        if len(prior) == 1:
            expected = writer_ring_label_allocation_source(
                source_state=source_state,
                label=event.label,
            )
            _require(
                violations,
                prior[0].source == expected,
                "allocation_source_mismatch",
                event.label,
                message=f"expected {expected!r}, got {prior[0].source!r}",
            )
        _require_open_state(violations, source_state, successor_state, event)

    for index, event in pairs:
        following = _matching(releases, event.label, after=index)
        _require_one(
            violations,
            following,
            event.label,
            missing_kind="missing_paired_label_release",
            duplicate_kind="duplicate_paired_label_release",
        )
        _require_pair_state(violations, source_state, successor_state, event)

    _require_order(
        violations,
        lifecycle_events=allocations,
        transition_events=opens,
        missing_kind="allocation_without_open_endpoint",
        order_kind="allocation_after_open_endpoint",
        ordered=lambda lifecycle_index, transition_index: (
            lifecycle_index < transition_index
        ),
    )
    _require_order(
        violations,
        lifecycle_events=releases,
        transition_events=pairs,
        missing_kind="release_without_paired_endpoint",
        order_kind="release_before_paired_endpoint",
        ordered=lambda lifecycle_index, transition_index: (
            transition_index < lifecycle_index
        ),
    )
    return tuple(violations)


def _require_open_state(
    violations: list[WriterRingLifecycleTransitionViolation],
    source_state: object,
    successor_state: object,
    event: WriterRingEndpointEmitted,
) -> None:
    _require(
        violations,
        event.label not in _labels(source_state, "allocated"),
        "open_source_label_already_allocated",
        event.label,
    )
    _require(
        violations,
        _has_open_endpoint(successor_state, event),
        "successor_open_endpoint_missing",
        event.label,
    )
    _require(
        violations,
        event.label in _labels(successor_state, "allocated"),
        "successor_open_label_not_allocated",
        event.label,
    )
    _require(
        violations,
        event.label not in _labels(successor_state, "reusable"),
        "successor_open_label_still_reusable",
        event.label,
    )


def _require_pair_state(
    violations: list[WriterRingLifecycleTransitionViolation],
    source_state: object,
    successor_state: object,
    event: WriterRingEndpointPaired,
) -> None:
    _require(
        violations,
        _has_open_endpoint_for_pair(source_state, event),
        "source_pair_open_endpoint_missing",
        event.label,
    )
    _require(
        violations,
        event.label in _labels(source_state, "allocated"),
        "pair_source_label_not_allocated",
        event.label,
    )
    _require(
        violations,
        not _has_open_endpoint_for_bond(successor_state, event.bond),
        "successor_pair_open_endpoint_retained",
        event.label,
    )
    _require(
        violations,
        _has_closed_closure(successor_state, event),
        "successor_closed_closure_missing",
        event.label,
    )
    _require(
        violations,
        event.label not in _labels(successor_state, "allocated"),
        "successor_paired_label_still_allocated",
        event.label,
    )
    _require(
        violations,
        event.label in _labels(successor_state, "reusable"),
        "successor_paired_label_not_reusable",
        event.label,
    )


def _require_one(
    violations: list[WriterRingLifecycleTransitionViolation],
    matches: tuple[object, ...],
    label: WriterClosureLabel,
    *,
    missing_kind: str,
    duplicate_kind: str,
) -> None:
    if not matches:
        _require(violations, False, missing_kind, label)
    elif len(matches) > 1:
        _require(violations, False, duplicate_kind, label)


def _require(
    violations: list[WriterRingLifecycleTransitionViolation],
    condition: bool,
    kind: str,
    label: WriterClosureLabel,
    *,
    message: str = "",
) -> None:
    if not condition:
        violations.append(_violation(kind, label, message))


def _require_order(
    violations: list[WriterRingLifecycleTransitionViolation],
    *,
    lifecycle_events,
    transition_events,
    missing_kind: str,
    order_kind: str,
    ordered,
) -> None:
    for lifecycle_index, lifecycle_event in lifecycle_events:
        positions = tuple(
            transition_index
            for transition_index, transition_event in transition_events
            if transition_event.label == lifecycle_event.label
        )
        if not positions:
            _require(violations, False, missing_kind, lifecycle_event.label)
        elif not any(ordered(lifecycle_index, position) for position in positions):
            _require(violations, False, order_kind, lifecycle_event.label)


def _matching(
    indexed_events,
    label: WriterClosureLabel,
    *,
    before: int | None = None,
    after: int | None = None,
) -> tuple[object, ...]:
    return tuple(
        event
        for index, event in indexed_events
        if event.label == label
        and (before is None or index < before)
        and (after is None or index > after)
    )


def _indexed(indexed_events, event_type):
    return tuple(
        (index, event)
        for index, event in indexed_events
        if isinstance(event, event_type)
    )


def _event_labels(events, event_type) -> frozenset[WriterClosureLabel]:
    return frozenset(event.label for event in events if isinstance(event, event_type))


def _violation(
    kind: str,
    label: WriterClosureLabel | None,
    message: str,
) -> WriterRingLifecycleTransitionViolation:
    return WriterRingLifecycleTransitionViolation(
        kind=kind,
        label=label,
        message=message,
    )


def _labels(state: object, label_state_field: str) -> tuple[WriterClosureLabel, ...]:
    label_state = getattr(getattr(state, "ring_state", None), "label_state", None)
    if label_state is None:
        return ()
    return tuple(getattr(label_state, label_state_field, ()))


def _open_endpoints(state: object) -> tuple[object, ...]:
    return tuple(getattr(getattr(state, "ring_state", None), "open_endpoints", ()))


def _closed_closures(state: object) -> tuple[object, ...]:
    return tuple(getattr(getattr(state, "ring_state", None), "closed_closures", ()))


def _has_open_endpoint_for_bond(state: object, bond) -> bool:
    return any(endpoint.bond == bond for endpoint in _open_endpoints(state))


def _has_open_endpoint(state: object, event: WriterRingEndpointEmitted) -> bool:
    return any(
        endpoint.bond == event.bond
        and endpoint.first_atom == event.endpoint_atom
        and endpoint.second_atom == event.partner_atom
        and endpoint.label == event.label
        and endpoint.first_endpoint_text == event.endpoint_text
        and endpoint.first_endpoint_bond_text == event.bond_text
        and endpoint.first_endpoint_direction_mark == event.direction_mark
        for endpoint in _open_endpoints(state)
    )


def _has_open_endpoint_for_pair(
    state: object,
    event: WriterRingEndpointPaired,
) -> bool:
    return any(
        endpoint.bond == event.bond
        and endpoint.first_atom == event.partner_atom
        and endpoint.second_atom == event.endpoint_atom
        and endpoint.label == event.label
        and endpoint.first_endpoint_bond_text == event.first_endpoint_bond_text
        and endpoint.first_endpoint_direction_mark == event.first_endpoint_direction_mark
        for endpoint in _open_endpoints(state)
    )


def _has_closed_closure(state: object, event: WriterRingEndpointPaired) -> bool:
    return any(
        closure.bond == event.bond
        and closure.first_atom == event.partner_atom
        and closure.second_atom == event.endpoint_atom
        and closure.label == event.label
        and closure.second_endpoint_text == event.endpoint_text
        and closure.first_endpoint_bond_text == event.first_endpoint_bond_text
        and closure.second_endpoint_bond_text == event.bond_text
        and closure.first_endpoint_direction_mark == event.first_endpoint_direction_mark
        and closure.second_endpoint_direction_mark == event.direction_mark
        for closure in _closed_closures(state)
    )


__all__ = (
    "WriterRingLabelAllocationSource",
    "WriterRingLifecycleTransitionViolation",
    "validate_writer_ring_lifecycle_transition",
    "writer_events_with_ring_label_lifecycle",
    "writer_ring_label_allocation_source",
    "writer_ring_lifecycle_transition_violations",
)
