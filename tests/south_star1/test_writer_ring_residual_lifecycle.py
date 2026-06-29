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
        self.assertTrue(
            any(isinstance(event, WriterRingEndpointEmitted) for event in opened.events)
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
        self.assertTrue(
            any(isinstance(event, WriterRingEndpointPaired) for event in paired.events)
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
