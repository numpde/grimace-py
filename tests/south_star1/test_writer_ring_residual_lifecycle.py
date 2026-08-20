"""Ring residual lifecycle coverage through branch-preserving runtime."""

from __future__ import annotations

from tests.south_star1.writer_test_context import initial_writer_snapshot
from tests.south_star1.writer_test_context import prepare_writer_facts
from tests.south_star1.writer_test_context import writer_runtime_options

import unittest
from collections import deque

from grimace._south_star1.facts import ComponentFacts
from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import ComponentId
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.writer_events import WriterRingEndpointEmitted
from grimace._south_star1.writer_events import WriterRingEndpointPaired
from grimace._south_star1.writer_events import WriterRingLabelAllocated
from grimace._south_star1.writer_events import WriterRingLabelReleased
from grimace._south_star1.writer_runtime import WriterRuntimeState
from grimace._south_star1.writer_runtime import count_writer_runtime_branch_completions
from grimace._south_star1.writer_runtime import initial_writer_runtime_state
from grimace._south_star1.writer_runtime import writer_runtime_branch_transitions
from tests.south_star1.helpers import atom
from tests.south_star1.helpers import cyclopropane_facts
from tests.south_star1.helpers import single_bond


class WriterRingResidualLifecycleTest(unittest.TestCase):
    def test_ring_open_and_pair_are_branch_runtime_lifecycle_steps(self) -> None:
        prepared = prepare_writer_facts(cyclopropane_facts())
        initial = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=writer_runtime_options(),
        )

        opened = _find_branch_transition(
            prepared,
            initial,
            "open_closure_endpoint",
        )
        opened_event = _single_event(opened.events, WriterRingEndpointEmitted)
        opened_label_event = _single_event(opened.events, WriterRingLabelAllocated)
        opened_state = _single_state_key(opened.next_state)

        self.assertEqual(len(opened_state.ring_state.open_endpoints), 1)
        self.assertEqual(opened_state.ring_state.closed_closures, ())
        opened_endpoint = opened_state.ring_state.open_endpoints[0]
        self.assertEqual(opened_endpoint.bond, opened_event.bond)
        self.assertEqual(opened_endpoint.first_atom, opened_event.endpoint_atom)
        self.assertEqual(opened_endpoint.second_atom, opened_event.partner_atom)
        self.assertEqual(opened_endpoint.label, opened_event.label)
        self.assertEqual(opened_endpoint.label, opened_label_event.label)
        self.assertEqual(opened_endpoint.first_endpoint_text, opened_event.endpoint_text)
        self.assertEqual(opened_endpoint.first_endpoint_bond_text, opened_event.bond_text)
        self.assertEqual(
            opened_endpoint.first_endpoint_direction_mark,
            opened_event.direction_mark,
        )
        self.assertEqual(opened_label_event.source, "fresh")
        self.assertIn(opened_endpoint.label, opened_state.ring_state.label_state.allocated)
        self.assertNotIn(opened_endpoint.label, opened_state.ring_state.label_state.reusable)
        self.assertGreater(
            count_writer_runtime_branch_completions(
                prepared=prepared,
                state=opened.next_state,
            ),
            0,
        )

        paired = _find_branch_transition(
            prepared,
            opened.next_state,
            "pair_closure_endpoint",
        )
        paired_event = _single_event(paired.events, WriterRingEndpointPaired)
        paired_label_event = _single_event(paired.events, WriterRingLabelReleased)
        paired_state = _single_state_key(paired.next_state)

        self.assertEqual(paired_state.ring_state.open_endpoints, ())
        self.assertFalse(
            any(
                endpoint.bond == opened_endpoint.bond
                for endpoint in paired_state.ring_state.open_endpoints
            )
        )
        self.assertEqual(len(paired_state.ring_state.closed_closures), 1)
        closed_closure = paired_state.ring_state.closed_closures[0]
        self.assertEqual(closed_closure.bond, paired_event.bond)
        self.assertEqual(closed_closure.bond, opened_endpoint.bond)
        self.assertEqual(closed_closure.first_atom, opened_endpoint.first_atom)
        self.assertEqual(closed_closure.second_atom, opened_endpoint.second_atom)
        self.assertEqual(closed_closure.second_atom, paired_event.endpoint_atom)
        self.assertEqual(closed_closure.first_atom, paired_event.partner_atom)
        self.assertEqual(closed_closure.label, opened_endpoint.label)
        self.assertEqual(closed_closure.label, paired_event.label)
        self.assertEqual(closed_closure.label, paired_label_event.label)
        self.assertEqual(
            closed_closure.first_endpoint_text,
            opened_endpoint.first_endpoint_text,
        )
        self.assertEqual(
            closed_closure.second_endpoint_text,
            paired_event.endpoint_text,
        )
        self.assertEqual(
            closed_closure.first_endpoint_bond_text,
            paired_event.first_endpoint_bond_text,
        )
        self.assertEqual(
            closed_closure.first_endpoint_bond_text,
            opened_endpoint.first_endpoint_bond_text,
        )
        self.assertEqual(
            closed_closure.second_endpoint_bond_text,
            paired_event.bond_text,
        )
        self.assertEqual(
            closed_closure.first_endpoint_direction_mark,
            paired_event.first_endpoint_direction_mark,
        )
        self.assertEqual(
            closed_closure.first_endpoint_direction_mark,
            opened_endpoint.first_endpoint_direction_mark,
        )
        self.assertEqual(
            closed_closure.second_endpoint_direction_mark,
            paired_event.direction_mark,
        )
        self.assertEqual(paired_label_event.destination, "reusable")
        self.assertNotIn(
            closed_closure.label,
            paired_state.ring_state.label_state.allocated,
        )
        self.assertIn(
            closed_closure.label,
            paired_state.ring_state.label_state.reusable,
        )
        self.assertTrue(
            any(
                evidence.relation_kind == "closure_endpoint"
                for evidence in paired.finite_relation_work_evidence
            )
        )
        self.assertGreater(
            count_writer_runtime_branch_completions(
                prepared=prepared,
                state=paired.next_state,
            ),
            0,
        )

    def test_released_ring_label_is_reused_by_later_component_closure(self) -> None:
        prepared = prepare_writer_facts(_two_independent_cyclopropane_components_facts())
        initial = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=writer_runtime_options(),
        )

        first_open = _find_branch_transition(
            prepared,
            initial,
            "open_closure_endpoint",
        )
        first_allocated = _single_event(
            first_open.events,
            WriterRingLabelAllocated,
        )
        first_opened_state = _single_state_key(first_open.next_state)
        first_endpoint = first_opened_state.ring_state.open_endpoints[0]
        self.assertEqual(first_allocated.source, "fresh")
        self.assertEqual(first_endpoint.label, first_allocated.label)

        first_pair = _find_branch_transition(
            prepared,
            first_open.next_state,
            "pair_closure_endpoint",
        )
        first_released = _single_event(
            first_pair.events,
            WriterRingLabelReleased,
        )
        first_paired_state = _single_state_key(first_pair.next_state)
        first_closure = first_paired_state.ring_state.closed_closures[0]

        self.assertEqual(first_released.label, first_allocated.label)
        self.assertEqual(first_paired_state.ring_state.open_endpoints, ())
        self.assertEqual(len(first_paired_state.ring_state.closed_closures), 1)
        self.assertNotIn(
            first_released.label,
            first_paired_state.ring_state.label_state.allocated,
        )
        self.assertIn(
            first_released.label,
            first_paired_state.ring_state.label_state.reusable,
        )
        self.assertGreater(
            count_writer_runtime_branch_completions(
                prepared=prepared,
                state=first_pair.next_state,
            ),
            0,
        )

        second_open = _find_branch_transition(
            prepared,
            first_pair.next_state,
            "open_closure_endpoint",
            predicate=lambda branch: _has_label_allocation_source(
                branch,
                "reused",
            ),
        )
        second_allocated = _single_event(
            second_open.events,
            WriterRingLabelAllocated,
        )
        second_opened_state = _single_state_key(second_open.next_state)
        second_endpoint = second_opened_state.ring_state.open_endpoints[0]

        self.assertEqual(second_allocated.source, "reused")
        self.assertEqual(second_allocated.label, first_released.label)
        self.assertEqual(second_endpoint.label, first_released.label)
        self.assertNotEqual(second_endpoint.bond, first_closure.bond)
        self.assertEqual(len(second_opened_state.ring_state.closed_closures), 1)
        self.assertIn(
            second_allocated.label,
            second_opened_state.ring_state.label_state.allocated,
        )
        self.assertNotIn(
            second_allocated.label,
            second_opened_state.ring_state.label_state.reusable,
        )

        second_pair = _find_branch_transition(
            prepared,
            second_open.next_state,
            "pair_closure_endpoint",
        )
        second_paired = _single_event(
            second_pair.events,
            WriterRingEndpointPaired,
        )
        second_released = _single_event(
            second_pair.events,
            WriterRingLabelReleased,
        )
        second_paired_state = _single_state_key(second_pair.next_state)

        self.assertEqual(second_paired.label, first_released.label)
        self.assertEqual(second_released.label, first_released.label)
        self.assertEqual(second_released.destination, "reusable")
        self.assertEqual(second_paired_state.ring_state.open_endpoints, ())
        self.assertEqual(len(second_paired_state.ring_state.closed_closures), 2)
        self.assertNotIn(
            first_released.label,
            second_paired_state.ring_state.label_state.allocated,
        )
        self.assertIn(
            first_released.label,
            second_paired_state.ring_state.label_state.reusable,
        )
        self.assertTrue(
            any(
                evidence.relation_kind == "closure_endpoint"
                for evidence in second_pair.finite_relation_work_evidence
            )
        )
        self.assertGreater(
            count_writer_runtime_branch_completions(
                prepared=prepared,
                state=second_pair.next_state,
            ),
            0,
        )


def _single_event(events, event_type):
    matches = tuple(event for event in events if isinstance(event, event_type))
    if len(matches) != 1:
        raise AssertionError(
            f"expected exactly one {event_type.__name__}, got {len(matches)}",
        )
    return matches[0]


def _single_state_key(state: WriterRuntimeState):
    weighted_states = state.snapshot.cursor.weighted_states
    if len(weighted_states) != 1:
        raise AssertionError(
            f"expected one branch successor state, got {len(weighted_states)}",
        )
    state_key, weight = weighted_states[0]
    if weight != 1:
        raise AssertionError(f"expected unit branch successor weight, got {weight}")
    return state_key


def _find_branch_transition(
    prepared,
    state,
    kind_value: str,
    *,
    predicate=None,
):
    pending = deque((state,))
    seen = set()

    while pending and len(seen) < 4096:
        current = pending.popleft()
        cursor = current.snapshot.cursor
        if cursor in seen:
            continue
        seen.add(cursor)

        branches = writer_runtime_branch_transitions(
            prepared=prepared,
            state=current,
            include_counts=True,
        )
        for branch in branches.transitions:
            if getattr(branch.transition_kind, "value", None) != kind_value:
                continue
            if predicate is None or predicate(branch):
                return branch
        pending.extend(branch.next_state for branch in branches.transitions)

    raise AssertionError(f"did not find writer branch transition kind {kind_value!r}")


def _has_label_allocation_source(branch, source: str) -> bool:
    return any(
        isinstance(event, WriterRingLabelAllocated) and event.source == source
        for event in branch.events
    )


def _two_independent_cyclopropane_components_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=tuple(atom(index, "C") for index in range(6)),
        bonds=(
            single_bond(0, 0, 1),
            single_bond(1, 1, 2),
            single_bond(2, 2, 0),
            single_bond(3, 3, 4),
            single_bond(4, 4, 5),
            single_bond(5, 5, 3),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1), AtomId(2)),
                bonds=(BondId(0), BondId(1), BondId(2)),
            ),
            ComponentFacts(
                id=ComponentId(1),
                atoms=(AtomId(3), AtomId(4), AtomId(5)),
                bonds=(BondId(3), BondId(4), BondId(5)),
            ),
        ),
    )


if __name__ == "__main__":
    unittest.main()
