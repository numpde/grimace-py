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

    return (
        *_open_violations(source_state, successor_state, opens, allocations),
        *_pair_violations(source_state, successor_state, pairs, releases),
        *_ordering_violations(
            lifecycle_events=allocations,
            transition_events=opens,
            missing_kind="allocation_without_open_endpoint",
            order_kind="allocation_after_open_endpoint",
            ordered=lambda lifecycle_index, transition_index: (
                lifecycle_index < transition_index
            ),
        ),
        *_ordering_violations(
            lifecycle_events=releases,
            transition_events=pairs,
            missing_kind="release_without_paired_endpoint",
            order_kind="release_before_paired_endpoint",
            ordered=lambda lifecycle_index, transition_index: (
                transition_index < lifecycle_index
            ),
        ),
    )


def _open_violations(
    source_state: object,
    successor_state: object,
    opens,
    allocations,
) -> tuple[WriterRingLifecycleTransitionViolation, ...]:
    result: list[WriterRingLifecycleTransitionViolation] = []
    for index, event in opens:
        prior = _matching(allocations, event.label, before=index)
        result.extend(
            _one_event_violations(
                prior,
                label=event.label,
                missing_kind="missing_open_label_allocation",
                duplicate_kind="duplicate_open_label_allocation",
            )
        )
        if len(prior) == 1:
            expected = writer_ring_label_allocation_source(
                source_state=source_state,
                label=event.label,
            )
            result.extend(
                _when_false(
                    prior[0].source == expected,
                    "allocation_source_mismatch",
                    event.label,
                    f"expected {expected!r}, got {prior[0].source!r}",
                )
            )
        result.extend(
            _state_violations(
                event.label,
                checks=(
                    (
                        event.label not in _labels(source_state, "allocated"),
                        "open_source_label_already_allocated",
                        "source label is already allocated",
                    ),
                    (
                        _has_open_endpoint(successor_state, event),
                        "successor_open_endpoint_missing",
                        "successor lacks the emitted open endpoint",
                    ),
                    (
                        event.label in _labels(successor_state, "allocated"),
                        "successor_open_label_not_allocated",
                        "successor did not allocate the opened label",
                    ),
                    (
                        event.label not in _labels(successor_state, "reusable"),
                        "successor_open_label_still_reusable",
                        "successor still marks opened label reusable",
                    ),
                ),
            )
        )
    return tuple(result)


def _pair_violations(
    source_state: object,
    successor_state: object,
    pairs,
    releases,
) -> tuple[WriterRingLifecycleTransitionViolation, ...]:
    result: list[WriterRingLifecycleTransitionViolation] = []
    for index, event in pairs:
        following = _matching(releases, event.label, after=index)
        result.extend(
            _one_event_violations(
                following,
                label=event.label,
                missing_kind="missing_paired_label_release",
                duplicate_kind="duplicate_paired_label_release",
            )
        )
        result.extend(
            _state_violations(
                event.label,
                checks=(
                    (
                        _has_open_endpoint_for_pair(source_state, event),
                        "source_pair_open_endpoint_missing",
                        "source lacks the paired open endpoint",
                    ),
                    (
                        event.label in _labels(source_state, "allocated"),
                        "pair_source_label_not_allocated",
                        "source label is not allocated",
                    ),
                    (
                        not _has_open_endpoint_for_bond(successor_state, event.bond),
                        "successor_pair_open_endpoint_retained",
                        "successor retained an open endpoint for the paired bond",
                    ),
                    (
                        _has_closed_closure(successor_state, event),
                        "successor_closed_closure_missing",
                        "successor lacks the emitted closed closure",
                    ),
                    (
                        event.label not in _labels(successor_state, "allocated"),
                        "successor_paired_label_still_allocated",
                        "successor still marks paired label allocated",
                    ),
                    (
                        event.label in _labels(successor_state, "reusable"),
                        "successor_paired_label_not_reusable",
                        "successor did not release paired label to reusable",
                    ),
                ),
            )
        )
    return tuple(result)


def _one_event_violations(
    matches: tuple[object, ...],
    *,
    label: WriterClosureLabel,
    missing_kind: str,
    duplicate_kind: str,
) -> tuple[WriterRingLifecycleTransitionViolation, ...]:
    if not matches:
        return (_violation(missing_kind, label, "missing lifecycle evidence"),)
    if len(matches) > 1:
        return (_violation(duplicate_kind, label, "duplicate lifecycle evidence"),)
    return ()


def _state_violations(
    label: WriterClosureLabel,
    *,
    checks: tuple[tuple[bool, str, str], ...],
) -> tuple[WriterRingLifecycleTransitionViolation, ...]:
    return tuple(
        _violation(kind, label, message)
        for condition, kind, message in checks
        if not condition
    )


def _when_false(
    condition: bool,
    kind: str,
    label: WriterClosureLabel,
    message: str,
) -> tuple[WriterRingLifecycleTransitionViolation, ...]:
    return () if condition else (_violation(kind, label, message),)


def _ordering_violations(
    *,
    lifecycle_events,
    transition_events,
    missing_kind: str,
    order_kind: str,
    ordered,
) -> tuple[WriterRingLifecycleTransitionViolation, ...]:
    result: list[WriterRingLifecycleTransitionViolation] = []
    for lifecycle_index, lifecycle_event in lifecycle_events:
        positions = tuple(
            transition_index
            for transition_index, transition_event in transition_events
            if transition_event.label == lifecycle_event.label
        )
        if not positions:
            result.append(
                _violation(
                    missing_kind,
                    lifecycle_event.label,
                    "missing matching endpoint",
                )
            )
        elif not any(ordered(lifecycle_index, position) for position in positions):
            result.append(
                _violation(
                    order_kind,
                    lifecycle_event.label,
                    "wrong lifecycle event order",
                )
            )
    return tuple(result)


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
