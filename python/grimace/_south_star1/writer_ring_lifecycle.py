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
    """Return ``events`` with ring-label lifecycle evidence made explicit.

    The helper is intentionally idempotent. Raw writer transitions can start
    carrying these events directly without causing branch-runtime duplication.
    """

    allocated_labels = _event_labels(events, WriterRingLabelAllocated)
    released_labels = _event_labels(events, WriterRingLabelReleased)
    result: list[WriterEvent] = []

    for event in events:
        if isinstance(event, WriterRingEndpointEmitted):
            if event.label not in allocated_labels:
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
            continue

        if isinstance(event, WriterRingEndpointPaired):
            result.append(event)
            if event.label not in released_labels:
                result.append(WriterRingLabelReleased(label=event.label))
            continue

        result.append(event)

    return tuple(result)


def writer_ring_label_allocation_source(
    *,
    source_state: object,
    label: WriterClosureLabel,
) -> WriterRingLabelAllocationSource:
    reusable_labels = _source_state_reusable_labels(source_state)
    return "reused" if label in reusable_labels else "fresh"


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
    violations: list[WriterRingLifecycleTransitionViolation] = []
    open_events = _indexed_events(events, WriterRingEndpointEmitted)
    pair_events = _indexed_events(events, WriterRingEndpointPaired)
    allocation_events = _indexed_events(events, WriterRingLabelAllocated)
    release_events = _indexed_events(events, WriterRingLabelReleased)

    for index, allocation in allocation_events:
        matching_positions = tuple(
            endpoint_index
            for endpoint_index, endpoint in open_events
            if endpoint.label == allocation.label
        )
        if not matching_positions:
            violations.append(
                _violation(
                    "allocation_without_open_endpoint",
                    allocation.label,
                    "label allocation has no matching emitted open endpoint",
                )
            )
            continue
        if all(endpoint_index < index for endpoint_index in matching_positions):
            violations.append(
                _violation(
                    "allocation_after_open_endpoint",
                    allocation.label,
                    "label allocation appears after its open endpoint",
                )
            )

    for index, release in release_events:
        matching_positions = tuple(
            pair_index
            for pair_index, pair in pair_events
            if pair.label == release.label
        )
        if not matching_positions:
            violations.append(
                _violation(
                    "release_without_paired_endpoint",
                    release.label,
                    "label release has no matching paired endpoint",
                )
            )
            continue
        if all(pair_index > index for pair_index in matching_positions):
            violations.append(
                _violation(
                    "release_before_paired_endpoint",
                    release.label,
                    "label release appears before its paired endpoint",
                )
            )

    for index, event in open_events:
        allocations = tuple(
            allocation
            for allocation_index, allocation in allocation_events
            if allocation.label == event.label and allocation_index < index
        )
        if not allocations:
            violations.append(
                _violation(
                    "missing_open_label_allocation",
                    event.label,
                    "open endpoint is not preceded by label allocation evidence",
                )
            )
        elif len(allocations) > 1:
            violations.append(
                _violation(
                    "duplicate_open_label_allocation",
                    event.label,
                    "open endpoint has multiple prior label allocations",
                )
            )
        else:
            expected_source = writer_ring_label_allocation_source(
                source_state=source_state,
                label=event.label,
            )
            if allocations[0].source != expected_source:
                violations.append(
                    _violation(
                        "allocation_source_mismatch",
                        event.label,
                        (
                            "allocation source does not match source-state "
                            f"label lifecycle: expected {expected_source!r}, "
                            f"got {allocations[0].source!r}"
                        ),
                    )
                )

        if event.label in _allocated_labels(source_state):
            violations.append(
                _violation(
                    "open_source_label_already_allocated",
                    event.label,
                    "open endpoint allocates a label already allocated in source state",
                )
            )
        if not _successor_has_open_endpoint(successor_state, event):
            violations.append(
                _violation(
                    "successor_open_endpoint_missing",
                    event.label,
                    "successor state does not contain the emitted open endpoint",
                )
            )
        if event.label not in _allocated_labels(successor_state):
            violations.append(
                _violation(
                    "successor_open_label_not_allocated",
                    event.label,
                    "successor state does not mark the opened label allocated",
                )
            )
        if event.label in _reusable_labels(successor_state):
            violations.append(
                _violation(
                    "successor_open_label_still_reusable",
                    event.label,
                    "successor state still marks the opened label reusable",
                )
            )

    for index, event in pair_events:
        releases = tuple(
            release
            for release_index, release in release_events
            if release.label == event.label and release_index > index
        )
        if not releases:
            violations.append(
                _violation(
                    "missing_paired_label_release",
                    event.label,
                    "paired endpoint is not followed by label release evidence",
                )
            )
        elif len(releases) > 1:
            violations.append(
                _violation(
                    "duplicate_paired_label_release",
                    event.label,
                    "paired endpoint has multiple following label releases",
                )
            )

        if not _source_has_open_endpoint_for_pair(source_state, event):
            violations.append(
                _violation(
                    "source_pair_open_endpoint_missing",
                    event.label,
                    "source state does not contain the paired open endpoint",
                )
            )
        if event.label not in _allocated_labels(source_state):
            violations.append(
                _violation(
                    "pair_source_label_not_allocated",
                    event.label,
                    "paired endpoint source state does not mark the label allocated",
                )
            )
        if _successor_has_open_endpoint_for_bond(successor_state, event.bond):
            violations.append(
                _violation(
                    "successor_pair_open_endpoint_retained",
                    event.label,
                    "successor state still contains an open endpoint for paired bond",
                )
            )
        if not _successor_has_closed_closure(successor_state, event):
            violations.append(
                _violation(
                    "successor_closed_closure_missing",
                    event.label,
                    "successor state does not contain the paired closed closure",
                )
            )
        if event.label in _allocated_labels(successor_state):
            violations.append(
                _violation(
                    "successor_paired_label_still_allocated",
                    event.label,
                    "successor state still marks the paired label allocated",
                )
            )
        if event.label not in _reusable_labels(successor_state):
            violations.append(
                _violation(
                    "successor_paired_label_not_reusable",
                    event.label,
                    "successor state does not mark the paired label reusable",
                )
            )

    return tuple(violations)


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


def _indexed_events(events, event_type):
    return tuple(
        (index, event)
        for index, event in enumerate(events)
        if isinstance(event, event_type)
    )


def _event_labels(events, event_type) -> frozenset[WriterClosureLabel]:
    return frozenset(
        event.label
        for event in events
        if isinstance(event, event_type)
    )


def _source_state_reusable_labels(
    source_state: object,
) -> tuple[WriterClosureLabel, ...]:
    return _reusable_labels(source_state)


def _ring_state(state: object):
    return getattr(state, "ring_state", None)


def _label_state(state: object):
    ring_state = _ring_state(state)
    if ring_state is None:
        return None
    return getattr(ring_state, "label_state", None)


def _allocated_labels(state: object) -> tuple[WriterClosureLabel, ...]:
    label_state = _label_state(state)
    if label_state is None:
        return ()
    return tuple(getattr(label_state, "allocated", ()))


def _reusable_labels(state: object) -> tuple[WriterClosureLabel, ...]:
    label_state = _label_state(state)
    if label_state is None:
        return ()
    return tuple(getattr(label_state, "reusable", ()))


def _open_endpoints(state: object) -> tuple[object, ...]:
    ring_state = _ring_state(state)
    if ring_state is None:
        return ()
    return tuple(getattr(ring_state, "open_endpoints", ()))


def _closed_closures(state: object) -> tuple[object, ...]:
    ring_state = _ring_state(state)
    if ring_state is None:
        return ()
    return tuple(getattr(ring_state, "closed_closures", ()))


def _successor_has_open_endpoint(
    state: object,
    event: WriterRingEndpointEmitted,
) -> bool:
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


def _source_has_open_endpoint_for_pair(
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


def _successor_has_open_endpoint_for_bond(state: object, bond) -> bool:
    return any(endpoint.bond == bond for endpoint in _open_endpoints(state))


def _successor_has_closed_closure(
    state: object,
    event: WriterRingEndpointPaired,
) -> bool:
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
