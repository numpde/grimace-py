"""Tests for the public branch-preserving writer runtime surface."""

from __future__ import annotations

import unittest
from collections import Counter

from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.writer_frontier import WriterFrontierCursor
from grimace._south_star1.writer_frontier import _checked_writer_frontier_branch_supports
from grimace._south_star1.writer_frontier import _checked_writer_frontier_schedule_outcome
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_runtime import initial_writer_runtime_state
from grimace._south_star1.writer_runtime import writer_runtime_branch_transitions
from grimace._south_star1.writer_runtime import writer_runtime_choices
from tests.south_star1.helpers import cco_facts


class WriterBranchRuntimeTest(unittest.TestCase):
    def test_branch_surface_projects_to_current_text_choices(self) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )

        branches = writer_runtime_branch_transitions(
            prepared=prepared,
            state=state,
            include_counts=True,
        )
        choices = writer_runtime_choices(prepared=prepared, state=state)

        self.assertEqual(branches.choices, choices)
        self.assertEqual(
            tuple(branch.branch_ordinal for branch in branches.transitions),
            tuple(range(len(branches.transitions))),
        )

        text_counts = Counter(branch.emitted_text for branch in branches.transitions)
        self.assertGreater(max(text_counts.values()), 1)

        for choice in choices.choices:
            weighted_successors: Counter = Counter()
            for branch in branches.transitions:
                if branch.emitted_text == choice.emitted_text:
                    weighted_successors[branch.successor_state] += branch.parent_weight
                    self.assertEqual(
                        branch.next_state.snapshot.cursor,
                        WriterFrontierCursor(
                            weighted_states=((branch.successor_state, 1),)
                        ),
                    )
            self.assertEqual(
                choice.successor,
                WriterFrontierCursor(
                    weighted_states=tuple(weighted_successors.items())
                ),
            )

    def test_checked_frontier_branch_supports_preserve_raw_supports(self) -> None:
        prepared = _prepare(cco_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())

        schedule = _checked_writer_frontier_schedule_outcome(prepared, initial)
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            initial,
            include_counts=False,
        )

        self.assertEqual(len(batch.supports), len(schedule.next_token_supports))
        self.assertTrue(batch.supports)

        for ordinal, (projected, raw) in enumerate(
            zip(batch.supports, schedule.next_token_supports)
        ):
            transition = raw.schedule_support.transition

            self.assertEqual(projected.branch_ordinal, ordinal)
            self.assertEqual(projected.emitted_text, raw.emitted_text)
            self.assertEqual(projected.source_state, raw.state_key)
            self.assertEqual(projected.successor_state, raw.successor_key)
            self.assertEqual(projected.parent_weight, raw.parent_weight)
            self.assertEqual(projected.transition_kind, transition.kind)
            self.assertEqual(projected.events, transition.events)
            self.assertEqual(projected.evidence, transition.evidence)
            self.assertEqual(
                projected.execution_capabilities,
                frozenset(raw.execution_capabilities),
            )
            self.assertEqual(
                projected.residual_work_evidence,
                tuple(raw.residual_work_evidence),
            )
            self.assertEqual(
                projected.finite_relation_work_evidence,
                tuple(raw.finite_relation_work_evidence),
            )


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
