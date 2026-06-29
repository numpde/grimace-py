"""Ring-label lifecycle event derivation tests."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.writer_events import WriterRingEndpointEmitted
from grimace._south_star1.writer_events import WriterRingEndpointPaired
from grimace._south_star1.writer_events import WriterRingLabelAllocated
from grimace._south_star1.writer_events import WriterRingLabelReleased
from grimace._south_star1.writer_ring_lifecycle import writer_events_with_ring_label_lifecycle
from grimace._south_star1.writer_ring_lifecycle import writer_ring_label_allocation_source
from grimace._south_star1.writer_state import WriterClosureLabel
from grimace._south_star1.writer_state import WriterRingLabelState
from grimace._south_star1.writer_state import WriterRingState


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


if __name__ == "__main__":
    unittest.main()
