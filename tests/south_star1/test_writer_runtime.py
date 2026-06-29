"""Writer-shaped runtime facade tests."""

from __future__ import annotations

import unittest

from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import enumerate_prepared_writer_shaped_support
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.writer_runtime import _writer_runtime_branch_transition_batch
from grimace._south_star1.writer_runtime import advance_writer_runtime_state
from grimace._south_star1.writer_runtime import count_writer_runtime_branch_completions
from grimace._south_star1.writer_runtime import count_writer_runtime_completions
from grimace._south_star1.writer_runtime import count_writer_runtime_support
from grimace._south_star1.writer_runtime import initial_writer_runtime_state
from grimace._south_star1.writer_runtime import iter_writer_runtime_support
from grimace._south_star1.writer_runtime import writer_runtime_choice_transitions
from grimace._south_star1.writer_runtime import writer_runtime_choices
from grimace._south_star1.writer_runtime import writer_runtime_diagnostics
from grimace._south_star1.writer_runtime import writer_runtime_has_eos
from grimace._south_star1.writer_runtime import writer_runtime_state_from_snapshot
from grimace._south_star1.writer_runtime import writer_runtime_terminal
from grimace._south_star1.writer_snapshot import advance_writer_frontier_snapshot
from grimace._south_star1.writer_snapshot import resume_writer_frontier_choices_from_snapshot
from tests.south_star1.helpers import cco_facts


class WriterRuntimeFacadeTest(unittest.TestCase):
    def test_initial_runtime_support_matches_existing_writer_support_image(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options()

        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=options,
        )
        support = enumerate_prepared_writer_shaped_support(
            prepared=prepared,
            runtime_options=options,
        )

        self.assertEqual(
            tuple(iter_writer_runtime_support(prepared=prepared, state=state)),
            support.strings,
        )
        self.assertEqual(
            count_writer_runtime_support(prepared=prepared, state=state),
            support.distinct_count,
        )
        self.assertEqual(
            count_writer_runtime_completions(prepared=prepared, state=state),
            support.witness_count,
        )
        self.assertEqual(
            count_writer_runtime_branch_completions(
                prepared=prepared,
                state=state,
            ),
            support.witness_count,
        )

    def test_branch_transition_batch_sits_below_text_projection(self) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )

        branch_batch = _writer_runtime_branch_transition_batch(
            prepared=prepared,
            state=state,
            include_counts=True,
        )
        choices = writer_runtime_choices(prepared=prepared, state=state)
        branch_texts = tuple(
            branch.emitted_text
            for branch in branch_batch.branch_transitions
        )

        self.assertEqual(branch_batch.choices, choices)
        self.assertEqual(
            tuple(branch.branch_ordinal for branch in branch_batch.branch_transitions),
            tuple(range(len(branch_batch.branch_transitions))),
        )
        self.assertEqual(
            sum(branch.parent_weight for branch in branch_batch.branch_transitions),
            sum(choice.immediate_multiplicity for choice in choices.choices),
        )
        self.assertEqual(
            sorted(set(branch_texts)),
            sorted(choice.emitted_text for choice in choices.choices),
        )
        self.assertGreater(
            max(branch_texts.count(text) for text in set(branch_texts)),
            1,
        )

    def test_diagnostics_observe_live_frontier_without_classifying_support(self) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )

        diagnostics = writer_runtime_diagnostics(prepared=prepared, state=state)
        choices = writer_runtime_choices(prepared=prepared, state=state)

        self.assertFalse(diagnostics.blocked)
        self.assertFalse(diagnostics.has_policy_blockers)
        self.assertFalse(diagnostics.has_unsupported_execution_capabilities)
        self.assertFalse(diagnostics.has_work_envelope_violations)
        self.assertEqual((), diagnostics.work_envelope_violations)
        self.assertEqual(
            diagnostics.choice_texts,
            tuple(choice.emitted_text for choice in choices.choices),
        )
        self.assertEqual(
            diagnostics.has_eos,
            choices.terminal is not None,
        )

    def test_choice_transitions_package_checked_successors(self) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )

        transitions = writer_runtime_choice_transitions(
            prepared=prepared,
            state=state,
        )
        choices = writer_runtime_choices(prepared=prepared, state=state)

        self.assertEqual(transitions.choices, choices)
        self.assertEqual(transitions.terminal, choices.terminal)
        self.assertEqual(transitions.has_eos, choices.terminal is not None)
        self.assertEqual(
            transitions.support_count,
            sum(choice.support_count or 0 for choice in choices.choices),
        )
        self.assertEqual(
            transitions.completion_count,
            sum(choice.completion_count or 0 for choice in choices.choices),
        )
        self.assertEqual(
            tuple(transition.choice for transition in transitions.transitions),
            choices.choices,
        )
        for transition in transitions.transitions:
            self.assertEqual(
                transition.next_state.snapshot,
                advance_writer_frontier_snapshot(
                    state.snapshot,
                    prepared=prepared,
                    emitted_text=transition.choice.emitted_text,
                ),
            )

    def test_choices_and_advance_delegate_to_checked_snapshot_path(self) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )

        choices = writer_runtime_choices(prepared=prepared, state=state)
        snapshot_choices = resume_writer_frontier_choices_from_snapshot(
            state.snapshot,
            prepared=prepared,
        )
        self.assertEqual(choices, snapshot_choices)

        emitted_text = choices.choices[0].emitted_text
        advanced = advance_writer_runtime_state(
            prepared=prepared,
            state=state,
            emitted_text=emitted_text,
        )
        expected_snapshot = advance_writer_frontier_snapshot(
            state.snapshot,
            prepared=prepared,
            emitted_text=emitted_text,
        )

        self.assertEqual(advanced.snapshot, expected_snapshot)

    def test_resume_runtime_state_from_snapshot_preserves_behavior(self) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )

        resumed = writer_runtime_state_from_snapshot(
            state.snapshot,
            prepared=prepared,
        )

        self.assertEqual(
            writer_runtime_choices(prepared=prepared, state=resumed),
            writer_runtime_choices(prepared=prepared, state=state),
        )
        self.assertEqual(
            count_writer_runtime_support(prepared=prepared, state=resumed),
            count_writer_runtime_support(prepared=prepared, state=state),
        )
        self.assertEqual(
            count_writer_runtime_branch_completions(
                prepared=prepared,
                state=resumed,
            ),
            count_writer_runtime_branch_completions(
                prepared=prepared,
                state=state,
            ),
        )

    def test_terminal_eos_after_complete_runtime_string(self) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )
        target = next(iter_writer_runtime_support(prepared=prepared, state=state))

        state = _advance_runtime_along_string(prepared, state, target)

        self.assertTrue(writer_runtime_has_eos(prepared=prepared, state=state))
        terminal = writer_runtime_terminal(prepared=prepared, state=state)
        self.assertIsNotNone(terminal)
        assert terminal is not None
        self.assertGreaterEqual(terminal.support_count, 1)
        self.assertGreaterEqual(terminal.completion_count, 1)


def _advance_runtime_along_string(prepared, state, text: str):
    remaining = text
    while remaining:
        choices = writer_runtime_choices(prepared=prepared, state=state).choices
        matches = tuple(
            choice
            for choice in choices
            if remaining.startswith(choice.emitted_text)
        )
        if not matches:
            raise AssertionError(f"no writer runtime choice can consume {remaining!r}")

        # Prefer the longest token so multi-character tokens such as "Cl" are
        # replayed as one live transition rather than as a misleading prefix.
        choice = max(matches, key=lambda item: len(item.emitted_text))
        state = advance_writer_runtime_state(
            prepared=prepared,
            state=state,
            emitted_text=choice.emitted_text,
        )
        remaining = remaining[len(choice.emitted_text) :]
    return state


def _prepare(facts):
    return prepare_south_star_mol_from_facts(
        facts,
        writer_surface=SouthStarWriterSurface(),
    )


def _writer_options() -> SouthStarRuntimeOptions:
    return SouthStarRuntimeOptions(
        serialization_language=SerializationLanguageMode.WRITER_SHAPED,
    )
