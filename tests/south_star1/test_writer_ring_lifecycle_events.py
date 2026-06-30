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
from grimace._south_star1.writer_ring_lifecycle import writer_ring_lifecycle_transition_violations
from grimace._south_star1.writer_runtime import initial_writer_runtime_state
from grimace._south_star1.writer_runtime import writer_runtime_branch_transitions
from grimace._south_star1.writer_state import WriterClosedClosure
from grimace._south_star1.writer_state import WriterClosureLabel
from grimace._south_star1.writer_state import WriterOpenClosureEndpoint
from grimace._south_star1.writer_state import WriterRingLabelState
from grimace._south_star1.writer_state import WriterRingState
from tests.south_star1.helpers import cyclopropane_facts


class WriterRingLifecycleEventsTest(unittest.TestCase):
    def test_open_endpoint_event_allocates_fresh_label_when_not_reusable(self) -> None:
        label = _label(1)
        opened = _endpoint_emitted(label)

        events = writer_events_with_ring_label_lifecycle(
            source_state=_state(),
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
            source_state=_state(reusable=(label,)),
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
                source_state=_state(reusable=(label,)),
                label=label,
            ),
            "reused",
        )

    def test_pair_endpoint_event_releases_label_to_reusable(self) -> None:
        label = _label(1)
        paired = _endpoint_paired(label)

        events = writer_events_with_ring_label_lifecycle(
            source_state=_state(),
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
            source_state=_state(reusable=(label,)),
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
        self.assertEqual(
            writer_ring_lifecycle_transition_violations(
                source_state=initial.weighted_states[0][0],
                successor_state=opened_successor,
                events=opened.events,
            ),
            (),
        )

        paired, paired_successor = _find_raw_frontier_transition(
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
        self.assertEqual(
            writer_ring_lifecycle_transition_violations(
                source_state=opened_successor,
                successor_state=paired_successor,
                events=paired.events,
            ),
            (),
        )

    def test_runtime_branch_transition_preserves_raw_transition_events(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            cyclopropane_facts(),
            writer_surface=SouthStarWriterSurface(),
        )
        runtime_initial = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )

        raw_transition, raw_successor = _find_raw_frontier_transition(
            prepared,
            runtime_initial.snapshot.cursor,
            "open_closure_endpoint",
        )
        runtime_branch = _find_runtime_branch_transition(
            prepared,
            runtime_initial,
            "open_closure_endpoint",
        )

        self.assertEqual(runtime_branch.successor_state, raw_successor)
        self.assertEqual(runtime_branch.events, raw_transition.events)
        self.assertEqual(
            _single_event(runtime_branch.events, WriterRingLabelAllocated).source,
            "fresh",
        )

    def test_transition_validator_reports_missing_open_allocation(self) -> None:
        label = _label(1)
        violations = writer_ring_lifecycle_transition_violations(
            source_state=_state(),
            successor_state=_state(
                allocated=(label,),
                open_endpoints=(_open_endpoint(label),),
            ),
            events=(_endpoint_emitted(label),),
        )

        self.assertIn(
            "missing_open_label_allocation",
            _violation_kinds(violations),
        )

    def test_transition_validator_reports_allocation_source_mismatch(self) -> None:
        label = _label(1)
        violations = writer_ring_lifecycle_transition_violations(
            source_state=_state(reusable=(label,)),
            successor_state=_state(
                allocated=(label,),
                open_endpoints=(_open_endpoint(label),),
            ),
            events=(
                WriterRingLabelAllocated(label=label, source="fresh"),
                _endpoint_emitted(label),
            ),
        )

        self.assertIn(
            "allocation_source_mismatch",
            _violation_kinds(violations),
        )

    def test_transition_validator_reports_missing_pair_release(self) -> None:
        label = _label(1)
        violations = writer_ring_lifecycle_transition_violations(
            source_state=_state(
                allocated=(label,),
                open_endpoints=(_open_endpoint(label),),
            ),
            successor_state=_state(
                reusable=(label,),
                closed_closures=(_closed_closure(label),),
            ),
            events=(_endpoint_paired(label),),
        )

        self.assertIn(
            "missing_paired_label_release",
            _violation_kinds(violations),
        )


@dataclass(frozen=True, slots=True)
class _SourceState:
    ring_state: WriterRingState


def _state(
    *,
    allocated: tuple[WriterClosureLabel, ...] = (),
    reusable: tuple[WriterClosureLabel, ...] = (),
    open_endpoints: tuple[WriterOpenClosureEndpoint, ...] = (),
    closed_closures: tuple[WriterClosedClosure, ...] = (),
) -> _SourceState:
    return _SourceState(
        ring_state=WriterRingState(
            open_endpoints=open_endpoints,
            closed_closures=closed_closures,
            label_state=WriterRingLabelState(
                allocated=allocated,
                reusable=reusable,
            ),
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


def _open_endpoint(label: WriterClosureLabel) -> WriterOpenClosureEndpoint:
    return WriterOpenClosureEndpoint(
        bond=BondId(0),
        first_atom=AtomId(0),
        second_atom=AtomId(1),
        label=label,
        first_endpoint_text=label.text,
        first_endpoint_bond_text="",
    )


def _closed_closure(label: WriterClosureLabel) -> WriterClosedClosure:
    return WriterClosedClosure(
        bond=BondId(0),
        first_atom=AtomId(0),
        second_atom=AtomId(1),
        label=label,
        first_endpoint_text=label.text,
        second_endpoint_text=label.text,
        first_endpoint_bond_text="",
        second_endpoint_bond_text="",
    )


def _single_event(events, event_type):
    matches = tuple(event for event in events if isinstance(event, event_type))
    if len(matches) != 1:
        raise AssertionError(
            f"expected exactly one {event_type.__name__}, got {len(matches)}",
        )
    return matches[0]


def _violation_kinds(violations) -> tuple[str, ...]:
    return tuple(violation.kind for violation in violations)


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


def _find_runtime_branch_transition(
    prepared,
    state,
    kind_value: str,
):
    pending = deque((state,))
    seen = set()

    while pending and len(seen) < 512:
        current = pending.popleft()
        cursor = current.snapshot.cursor
        if cursor in seen:
            continue
        seen.add(cursor)

        branches = writer_runtime_branch_transitions(
            prepared=prepared,
            state=current,
            include_counts=False,
        )
        for branch in branches.transitions:
            if getattr(branch.transition_kind, "value", None) == kind_value:
                return branch
        pending.extend(branch.next_state for branch in branches.transitions)

    raise AssertionError(f"did not find writer branch transition kind {kind_value!r}")


def _writer_options() -> SouthStarRuntimeOptions:
    return SouthStarRuntimeOptions(
        serialization_language=SerializationLanguageMode.WRITER_SHAPED,
    )


if __name__ == "__main__":
    unittest.main()
