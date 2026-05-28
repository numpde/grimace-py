"""Writer transition semantic-event coverage tests."""

from __future__ import annotations

import unittest

from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.writer_events import WriterAtomEmitted
from grimace._south_star1.writer_events import WriterBondEmitted
from grimace._south_star1.writer_events import WriterBranchClosed
from grimace._south_star1.writer_events import WriterBranchOpened
from grimace._south_star1.writer_events import WriterComponentBoundaryEmitted
from grimace._south_star1.writer_events import WriterLocalOrderClosed
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_state import writer_state_from_key
from grimace._south_star1.writer_state import writer_state_key
from grimace._south_star1.writer_transitions import legal_writer_transitions
from tests.south_star1.helpers import cco_facts
from tests.south_star1.test_writer_state_kernel import disconnected_co_facts


class WriterEventsTest(unittest.TestCase):
    def test_all_transition_kinds_emit_typed_events(self) -> None:
        prepared = _prepare(cco_facts())
        cursor = initial_writer_frontier_cursor(
            prepared,
            _writer_options(rooted_at_atom=1),
        )
        seen_types = set()
        stack = [key for key, _ in cursor.weighted_states]
        visited = set()
        while stack:
            key = stack.pop()
            if key in visited:
                continue
            visited.add(key)
            for transition in legal_writer_transitions(
                prepared,
                writer_state_from_key(key),
            ):
                self.assertTrue(transition.events)
                seen_types.update(type(event) for event in transition.events)
                stack.append(writer_state_key(transition.successor))

        self.assertIn(WriterAtomEmitted, seen_types)
        self.assertIn(WriterBondEmitted, seen_types)
        self.assertIn(WriterBranchOpened, seen_types)
        self.assertIn(WriterBranchClosed, seen_types)
        self.assertIn(WriterLocalOrderClosed, seen_types)

    def test_component_boundary_is_a_typed_event(self) -> None:
        prepared = _prepare(disconnected_co_facts())
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        seen_types = set()
        stack = [key for key, _ in cursor.weighted_states]
        visited = set()
        while stack:
            key = stack.pop()
            if key in visited:
                continue
            visited.add(key)
            for transition in legal_writer_transitions(
                prepared,
                writer_state_from_key(key),
            ):
                seen_types.update(type(event) for event in transition.events)
                stack.append(writer_state_key(transition.successor))

        self.assertIn(WriterComponentBoundaryEmitted, seen_types)


def _prepare(facts):
    return prepare_south_star_mol_from_facts(
        facts,
        writer_surface=SouthStarWriterSurface(),
    )


def _writer_options(*, rooted_at_atom: int = -1) -> SouthStarRuntimeOptions:
    return SouthStarRuntimeOptions(
        rooted_at_atom=rooted_at_atom,
        serialization_language=SerializationLanguageMode.WRITER_SHAPED,
    )


if __name__ == "__main__":
    unittest.main()
