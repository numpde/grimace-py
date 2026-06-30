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
    opens = _indexed_events(indexed, WriterRingEndpointEmitted)
    pairs = _indexed_events(indexed, WriterRingEndpointPaired)
    allocations = _indexed_events(indexed, WriterRingLabelAllocated)
    releases = _indexed_events(indexed, WriterRingLabelReleased)

    violations: list[WriterRingLifecycleTransitionViolation] = []
    for index, event in opens:
        matches = _matching_before(allocations, index, event.label)
        violations.extend(_single_lifecycle_event_violations(
            matches,
            missing_kind="missing_open_label_allocation",
            duplicate_kind="duplicate_open_label_allocation",
            label=event.label,
            missing_message="open endpoint is not preceded by label allocation evidence",
            duplicate_message="open endpoint has multiple prior label allocations",
        ))
        if len(matches) == 1:
            expected = writer_ring_label_allocation_source(
                source_state=source_state,
                label=event.label,
            )
            if matches[0].source != expected:
                violations.append(_violation(
                    "allocation_source_mismatch",
                    event.label,
                    (
                        "allocation source does not match source-state label "
                        f"lifecycle: expected {expected!r}, got {matches[0].source!r}"
                    ),
                ))
        violations.extend(_open_state_violations(source_state, successor_state, event))

    for index, event in pairs:
        matches = _matching_after(releases, index, event.label)
        violations.extend(_single_lifecycle_event_violations(
            matches,
            missing_kind="missing_paired_label_release",
            duplicate_kind="duplicate_paired_label_release",
            label=event.label,
            missing_message="paired endpoint is not followed by label release evidence",
            duplicate_message="paired endpoint has multiple following label releases",
        ))
        violations.extend(_pair_state_violations(source_state, successor_state, event))

    violations.extend(_unmatched_lifecycle_evidence_violations(
        lifecycle_events=allocations,
        transition_events=opens,
        missing_kind="allocation_without_open_endpoint",
        late_kind="allocation_after_open_endpoint",
        label_word="allocation",
        transition_word="open endpoint",
        relation=lambda lifecycle_index, transition_index: lifecycle_index < transition_index,
    ))
    violations.extend(_unmatched_lifecycle_evidence_violations(
        lifecycle_events=releases,
        transition_events=pairs,
        missing_kind="release_without_paired_endpoint",
        late_kind="release_before_paired_endpoint",
        label_word="release",
        transition_word="paired endpoint",
        relation=lambda lifecycle_index, transition_index: transition_index < lifecycle_index,
    ))
    return tuple(violations)


def _open_state_violations(
    source_state: object,
    successor_state: object,
    event: WriterRingEndpointEmitted,
) -> tuple[WriterRingLifecycleTransitionViolation, ...]:
    checks = (
        (
            event.label not in _labels(source_state, "allocated"),
            "open_source_label_already_allocated",
            "open endpoint allocates a label already allocated in source state",
        ),
        (
            _has_open_endpoint(successor_state, event),
            "successor_open_endpoint_missing",
            "successor state does not contain the emitted open endpoint",
        ),
        (
            event.label in _labels(successor_state, "allocated"),
            "successor_open_label_not_allocated",
            "successor state does not mark the opened label allocated",
        ),
        (
            event.label not in _labels(successor_state, "reusable"),
            "successor_open_label_still_reusable",
            "successor state still marks the opened label reusable",
        ),
    )
    return tuple(_violation(kind, event.label, message) for ok, kind, message in checks if not ok)


def _pair_state_violations(
    source_state: object,
    successor_state: object,
    event: WriterRingEndpointPaired,
) -> tuple[WriterRingLifecycleTransitionViolation, ...]:
    checks = (
        (
            _has_open_endpoint_for_pair(source_state, event),
            "source_pair_open_endpoint_missing",
            "source state does not contain the paired open endpoint",
        ),
        (
            event.label in _labels(source_state, "allocated"),
            "pair_source_label_not_allocated",
            "paired endpoint source state does not mark the label allocated",
        ),
        (
            not any(endpoint.bond == event.bond for endpoint in _open_endpoints(successor_state)),
            "successor_pair_open_endpoint_retained",
            "successor state still contains an open endpoint for paired bond",
        ),
        (
            _has_closed_closure(successor_state, event),
            "successor_closed_closure_missing",
            "successor state does not contain the paired closed closure",
        ),
        (
            event.label not in _labels(successor_state, "allocated"),
            "successor_paired_label_still_allocated",
            "successor state still marks the paired label allocated",
        ),
        (
            event.label in _labels(successor_state, "reusable"),
            "successor_paired_label_not_reusable",
            "successor state does not mark the paired label reusable",
        ),
    )
    return tuple(_violation(kind, event.label, message) for ok, kind, message in checks if not ok)


def _single_lifecycle_event_violations(
    matches: tuple[object, ...],
    *,
    missing_kind: str,
    duplicate_kind: str,
    label: WriterClosureLabel,
    missing_message: str,
    duplicate_message: str,
) -> tuple[WriterRingLifecycleTransitionViolation, ...]:
    if not matches:
        return (_violation(missing_kind, label, missing_message),)
    if len(matches) > 1:
        return (_violation(duplicate_kind, label, duplicate_message),)
    return ()


def _unmatched_lifecycle_evidence_violations(
    *,
    lifecycle_events,
    transition_events,
    missing_kind: str,
    late_kind: str,
    label_word: str,
    transition_word: str,
    relation,
) -> tuple[WriterRingLifecycleTransitionViolation, ...]:
    violations: list[WriterRingLifecycleTransitionViolation] = []
    for lifecycle_index, lifecycle_event in lifecycle_events:
        candidates = tuple(
            transition_index
            for transition_index, transition_event in transition_events
            if transition_event.label == lifecycle_event.label
        )
        if not candidates:
            violations.append(_violation(
                missing_kind,
                lifecycle_event.label,
                f"label {label_word} has no matching emitted {transition_word}",
            ))
        elif not any(relation(lifecycle_index, transition_index) for transition_index in candidates):
            violations.append(_violation(
                late_kind,
                lifecycle_event.label,
                f"label {label_word} is not ordered with its {transition_word}",
            ))
    return tuple(violations)


def _matching_before(indexed_events, index: int, label: WriterClosureLabel) -> tuple[object, ...]:
    return tuple(event for event_index, event in indexed_events if event_index < index and event.label == label)


def _matching_after(indexed_events, index: int, label: WriterClosureLabel) -> tuple[object, ...]:
    return tuple(event for event_index, event in indexed_events if event_index > index and event.label == label)


def _indexed_events(indexed_events, event_type):
    return tuple((index, event) for index, event in indexed_events if isinstance(event, event_type))


def _event_labels(events, event_type) -> frozenset[WriterClosureLabel]:
    return frozenset(event.label for event in events if isinstance(event, event_type))


def _violation(
    kind: str,
    label: WriterClosureLabel | None,
    message: str,
) -> WriterRingLifecycleTransitionViolation:
    return WriterRingLifecycleTransitionViolation(kind=kind, label=label, message=message)


def _labels(state: object, label_state_field: str) -> tuple[WriterClosureLabel, ...]:
    label_state = getattr(getattr(state, "ring_state", None), "label_state", None)
    if label_state is None:
        return ()
    return tuple(getattr(label_state, label_state_field, ()))


def _open_endpoints(state: object) -> tuple[object, ...]:
    return tuple(getattr(getattr(state, "ring_state", None), "open_endpoints", ()))


def _closed_closures(state: object) -> tuple[object, ...]:
    return tuple(getattr(getattr(state, "ring_state", None), "closed_closures", ()))


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


def _has_open_endpoint_for_pair(state: object, event: WriterRingEndpointPaired) -> bool:
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
