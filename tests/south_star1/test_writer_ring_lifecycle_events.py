"""Ring-label lifecycle event derivation tests."""

from __future__ import annotations

import unittest
from collections import deque
from dataclasses import dataclass

from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.writer_events import WriterRingEndpointEmitted
from grimace._south_star1.writer_events import WriterRingEndpointPaired
from grimace._south_star1.writer_events import WriterRingLabelAllocated
from grimace._south_star1.writer_events import WriterRingLabelReleased
from grimace._south_star1.writer_frontier import WriterFrontierCursor
from grimace._south_star1.writer_frontier import _checked_writer_frontier_schedule_outcome
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_ring_lifecycle import writer_events_with_ring_label_lifecycle
from grimace._south_star1.writer_ring_lifecycle import writer_ring_label_allocation_source
from grimace._south_star1.writer_state import WriterClosureLabel
from grimace._south_star1.writer_state import WriterRingLabelState
from grimace._south_star1.writer_state import WriterRingState
from tests.south_star1.helpers import cyclopropane_facts


class WriterRingLifecycleEventsTest(unittest.TestCase):
    def test_open_endpoint_event_allocates_fresh_label_when_not_reusable(self) -> None:
        label = _label(1)
        opened = _endpoint_emitted(label)

        events = writer_events_with_ring_label_lifecycle(
            source_state=_source_state(),
            events=(opened,),
        )

        self.assertEqual(
            events,
            (
                WriterRingLabelAllocated(label=label, source="fresh"),
                opened,
            ),
        )

    def test_open_endpoint_event_allocates_reused_label_when_reusable(self) -> None:
        label = _label(1)
        opened = _endpoint_emitted(label)

        events = writer_events_with_ring_label_lifecycle(
            source_state=_source_state(reusable=(label,)),
            events=(opened,),
        )

        self.assertEqual(
            events,
            (
                WriterRingLabelAllocated(label=label, source="reused"),
                opened,
            ),
        )
        self.assertEqual(
            writer_ring_label_allocation_source(
                source_state=_source_state(reusable=(label,)),
                label=label,
            ),
            "reused",
        )

    def test_pair_endpoint_event_releases_label_to_reusable(self) -> None:
        label = _label(1)
        paired = _endpoint_paired(label)

        events = writer_events_with_ring_label_lifecycle(
            source_state=_source_state(),
            events=(paired,),
        )

        self.assertEqual(
            events,
            (
                paired,
                WriterRingLabelReleased(label=label),
            ),
        )

    def test_lifecycle_derivation_is_idempotent_for_predecorated_events(self) -> None:
        label = _label(1)
        opened = _endpoint_emitted(label)
        paired = _endpoint_paired(label)
        predecorated = (
            WriterRingLabelAllocated(label=label, source="reused"),
            opened,
            paired,
            WriterRingLabelReleased(label=label),
        )

        events = writer_events_with_ring_label_lifecycle(
            source_state=_source_state(reusable=(label,)),
            events=predecorated,
        )

        self.assertEqual(events, predecorated)

    def test_raw_frontier_support_transitions_carry_lifecycle_events(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            cyclopropane_facts(),
            writer_surface=SouthStarWriterSurface(),
        )
        initial = initial_writer_frontier_cursor(prepared, _writer_options())

        opened, opened_successor = _find_raw_frontier_transition(
            prepared,
            initial,
            "open_closure_endpoint",
        )
        opened_allocated = _single_event(opened.events, WriterRingLabelAllocated)
        opened_event = _single_event(opened.events, WriterRingEndpointEmitted)

        self.assertEqual(opened.events[0], opened_allocated)
        self.assertEqual(opened.events[1], opened_event)
        self.assertEqual(opened_allocated.source, "fresh")
        self.assertEqual(opened_allocated.label, opened_event.label)

        paired, _paired_successor = _find_raw_frontier_transition(
            prepared,
            WriterFrontierCursor(weighted_states=((opened_successor, 1),)),
            "pair_closure_endpoint",
        )
        paired_event = _single_event(paired.events, WriterRingEndpointPaired)
        paired_released = _single_event(paired.events, WriterRingLabelReleased)

        self.assertEqual(paired.events[-2], paired_event)
        self.assertEqual(paired.events[-1], paired_released)
        self.assertEqual(paired_released.label, opened_allocated.label)
        self.assertEqual(paired_released.label, paired_event.label)


@dataclass(frozen=True, slots=True)
class _SourceState:
    ring_state: WriterRingState


def _source_state(
    *,
    reusable: tuple[WriterClosureLabel, ...] = (),
) -> _SourceState:
    return _SourceState(
        ring_state=WriterRingState(
            label_state=WriterRingLabelState(reusable=reusable),
        ),
    )


def _label(value: int) -> WriterClosureLabel:
    return WriterClosureLabel(value=value, text=str(value))


def _endpoint_emitted(label: WriterClosureLabel) -> WriterRingEndpointEmitted:
    return WriterRingEndpointEmitted(
        bond=BondId(0),
        endpoint_atom=AtomId(0),
        partner_atom=AtomId(1),
        label=label,
        endpoint_text=label.text,
        bond_text="",
    )


def _endpoint_paired(label: WriterClosureLabel) -> WriterRingEndpointPaired:
    return WriterRingEndpointPaired(
        bond=BondId(0),
        endpoint_atom=AtomId(1),
        partner_atom=AtomId(0),
        label=label,
        endpoint_text=label.text,
        bond_text="",
    )


def _single_event(events, event_type):
    matches = tuple(event for event in events if isinstance(event, event_type))
    if len(matches) != 1:
        raise AssertionError(
            f"expected exactly one {event_type.__name__}, got {len(matches)}",
        )
    return matches[0]


def _find_raw_frontier_transition(
    prepared,
    cursor: WriterFrontierCursor,
    kind_value: str,
):
    pending = deque((cursor,))
    seen = set()

    while pending and len(seen) < 512:
        current = pending.popleft()
        if current in seen:
            continue
        seen.add(current)

        outcome = _checked_writer_frontier_schedule_outcome(prepared, current)
        for support in outcome.next_token_supports:
            transition = support.schedule_support.transition
            if getattr(transition.kind, "value", None) == kind_value:
                return transition, support.successor_key
        pending.extend(
            WriterFrontierCursor(weighted_states=((support.successor_key, 1),))
            for support in outcome.next_token_supports
        )

    raise AssertionError(f"did not find raw writer transition kind {kind_value!r}")


def _writer_options() -> SouthStarRuntimeOptions:
    return SouthStarRuntimeOptions(
        serialization_language=SerializationLanguageMode.WRITER_SHAPED,
    )


if __name__ == "__main__":
    unittest.main()
