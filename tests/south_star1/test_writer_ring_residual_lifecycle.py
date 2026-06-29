"""Ring residual lifecycle coverage through branch-preserving runtime."""

from __future__ import annotations

import unittest
from collections import deque

from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.writer_events import WriterRingEndpointEmitted
from grimace._south_star1.writer_events import WriterRingEndpointPaired
from grimace._south_star1.writer_runtime import WriterRuntimeState
from grimace._south_star1.writer_runtime import count_writer_runtime_branch_completions
from grimace._south_star1.writer_runtime import initial_writer_runtime_state
from grimace._south_star1.writer_runtime import writer_runtime_branch_transitions
from tests.south_star1.helpers import cyclopropane_facts


class WriterRingResidualLifecycleTest(unittest.TestCase):
    def test_ring_open_and_pair_are_branch_runtime_lifecycle_steps(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        initial = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )

        opened = _find_branch_transition(
            prepared,
            initial,
            "open_closure_endpoint",
        )
        opened_event = _single_event(opened.events, WriterRingEndpointEmitted)
        opened_state = _single_state_key(opened.next_state)

        self.assertEqual(len(opened_state.ring_state.open_endpoints), 1)
        self.assertEqual(opened_state.ring_state.closed_closures, ())
        self.assertEqual(
            opened_state.ring_state.open_endpoints[0].bond,
            opened_event.bond,
        )
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
        paired_state = _single_state_key(paired.next_state)

        self.assertEqual(paired_state.ring_state.open_endpoints, ())
        self.assertEqual(len(paired_state.ring_state.closed_closures), 1)
        self.assertEqual(
            paired_state.ring_state.closed_closures[0].bond,
            paired_event.bond,
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


def _find_branch_transition(prepared, state, kind_value: str):
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
            include_counts=True,
        )
        for branch in branches.transitions:
            if getattr(branch.transition_kind, "value", None) == kind_value:
                return branch
        pending.extend(branch.next_state for branch in branches.transitions)

    raise AssertionError(f"did not find writer branch transition kind {kind_value!r}")


def _prepare(facts):
    return prepare_south_star_mol_from_facts(
        facts,
        writer_surface=SouthStarWriterSurface(),
    )


def _writer_options() -> SouthStarRuntimeOptions:
    return SouthStarRuntimeOptions(
        serialization_language=SerializationLanguageMode.WRITER_SHAPED,
    )


if __name__ == "__main__":
    unittest.main()
