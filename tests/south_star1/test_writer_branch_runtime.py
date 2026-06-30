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
from grimace._south_star1.writer_frontier import _count_checked_writer_frontier_branch_completions
from grimace._south_star1.writer_frontier import _writer_frontier_diagnostics
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_runtime import count_writer_runtime_branch_completions
from grimace._south_star1.writer_runtime import initial_writer_runtime_state
from grimace._south_star1.writer_runtime import writer_runtime_branch_transitions
from grimace._south_star1.writer_runtime import writer_runtime_choice_transitions
from grimace._south_star1.writer_runtime import writer_runtime_choices
from grimace._south_star1.writer_runtime import writer_runtime_diagnostics
from grimace._south_star1.writer_snapshot import _writer_search_snapshot_after_checked_branch_support
from grimace._south_star1.writer_snapshot import _writer_search_snapshot_after_checked_choice
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

    def test_runtime_branch_completion_count_is_frontier_owned(self) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )

        self.assertEqual(
            count_writer_runtime_branch_completions(
                prepared=prepared,
                state=state,
            ),
            _count_checked_writer_frontier_branch_completions(
                prepared,
                state.snapshot.cursor,
            ),
        )

    def test_runtime_diagnostics_is_frontier_owned(self) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )

        runtime = writer_runtime_diagnostics(
            prepared=prepared,
            state=state,
        )
        frontier = _writer_frontier_diagnostics(
            prepared,
            state.snapshot.cursor,
        )

        self.assertEqual(runtime.blocked, frontier.blocked)
        self.assertEqual(
            runtime.graph_policy_blockers,
            frontier.graph_policy_blockers,
        )
        self.assertEqual(
            runtime.stereo_policy_blockers,
            frontier.stereo_policy_blockers,
        )
        self.assertEqual(
            runtime.execution_capabilities,
            frontier.execution_capabilities,
        )
        self.assertEqual(
            runtime.terminal_execution_capabilities,
            frontier.terminal_execution_capabilities,
        )
        self.assertEqual(
            runtime.unsupported_execution_capabilities,
            frontier.unsupported_execution_capabilities,
        )
        self.assertEqual(
            runtime.unsupported_terminal_execution_capabilities,
            frontier.unsupported_terminal_execution_capabilities,
        )
        self.assertEqual(
            runtime.residual_work_evidence,
            frontier.residual_work_evidence,
        )
        self.assertEqual(
            runtime.terminal_residual_work_evidence,
            frontier.terminal_residual_work_evidence,
        )
        self.assertEqual(
            runtime.finite_relation_work_evidence,
            frontier.finite_relation_work_evidence,
        )
        self.assertEqual(
            runtime.graph_obligation_work_evidence,
            frontier.graph_obligation_work_evidence,
        )
        self.assertEqual(
            runtime.residual_work_envelope_violations,
            frontier.residual_work_envelope_violations,
        )
        self.assertEqual(
            runtime.terminal_residual_work_envelope_violations,
            frontier.terminal_residual_work_envelope_violations,
        )
        self.assertEqual(
            runtime.finite_relation_work_envelope_violations,
            frontier.finite_relation_work_envelope_violations,
        )
        self.assertEqual(
            runtime.graph_obligation_work_envelope_violations,
            frontier.graph_obligation_work_envelope_violations,
        )
        self.assertEqual(runtime.choice_texts, frontier.choice_texts)
        self.assertEqual(runtime.has_eos, frontier.has_eos)

    def test_branch_runtime_next_state_uses_snapshot_branch_packaging(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            state.snapshot.cursor,
            include_counts=False,
        )
        branches = writer_runtime_branch_transitions(
            prepared=prepared,
            state=state,
            include_counts=False,
        )

        for support, branch in zip(batch.supports, branches.transitions):
            expected = _writer_search_snapshot_after_checked_branch_support(
                state.snapshot,
                prepared=prepared,
                support=support,
            )
            self.assertEqual(support.successor_cursor, expected.cursor)
            self.assertEqual(branch.next_state.snapshot, expected)

    def test_choice_runtime_next_state_uses_snapshot_choice_packaging(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )
        choices = writer_runtime_choice_transitions(
            prepared=prepared,
            state=state,
        )

        for transition in choices.transitions:
            expected = _writer_search_snapshot_after_checked_choice(
                state.snapshot,
                prepared=prepared,
                choice=transition.choice,
            )
            self.assertEqual(transition.next_state.snapshot, expected)


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
