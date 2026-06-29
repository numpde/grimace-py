"""Ring-label lifecycle event derivation for writer transition streams."""

from __future__ import annotations

from typing import Literal

from .writer_events import WriterEvent
from .writer_events import WriterRingEndpointEmitted
from .writer_events import WriterRingEndpointPaired
from .writer_events import WriterRingLabelAllocated
from .writer_events import WriterRingLabelReleased
from .writer_state import WriterClosureLabel


WriterRingLabelAllocationSource = Literal["fresh", "reused"]


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


def _event_labels(events, event_type) -> frozenset[WriterClosureLabel]:
    return frozenset(
        event.label
        for event in events
        if isinstance(event, event_type)
    )


def _source_state_reusable_labels(
    source_state: object,
) -> tuple[WriterClosureLabel, ...]:
    ring_state = getattr(source_state, "ring_state", None)
    if ring_state is None:
        return ()

    label_state = getattr(ring_state, "label_state", None)
    if label_state is None:
        return ()

    return tuple(getattr(label_state, "reusable", ()))


__all__ = (
    "WriterRingLabelAllocationSource",
    "writer_events_with_ring_label_lifecycle",
    "writer_ring_label_allocation_source",
)
