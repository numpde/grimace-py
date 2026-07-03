"""Writer-shaped runtime facade tests."""

from __future__ import annotations

import unittest
from collections import Counter
from dataclasses import replace

from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import enumerate_prepared_writer_shaped_support
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.writer_branch_certificates import (
    writer_checked_terminal_support_certificate,
)
from grimace._south_star1.writer_frontier import WriterFrontierCursor
from grimace._south_star1.writer_frontier import _checked_writer_frontier_branch_supports
from grimace._south_star1.writer_projection_certificates import (
    writer_terminal_projection_certificate,
)
from grimace._south_star1.writer_stereo import EMPTY_RESIDUAL_SNAPSHOT
from grimace._south_star1.writer_terminal_certificates import (
    WriterTerminalCertificateKind,
)
from grimace._south_star1.writer_terminal_certificates import (
    writer_terminal_certificates,
)
from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.writer_runtime import advance_writer_runtime_state
from grimace._south_star1.writer_runtime import count_writer_runtime_branch_completions
from grimace._south_star1.writer_runtime import count_writer_runtime_completions
from grimace._south_star1.writer_runtime import count_writer_runtime_support
from grimace._south_star1.writer_runtime import initial_writer_runtime_state
from grimace._south_star1.writer_runtime import iter_writer_runtime_certified_support
from grimace._south_star1.writer_runtime import iter_writer_runtime_support
from grimace._south_star1.writer_runtime import writer_runtime_support_image_certificate
from grimace._south_star1.writer_runtime import writer_runtime_branch_transitions
from grimace._south_star1.writer_runtime import writer_runtime_choice_transitions
from grimace._south_star1.writer_runtime import writer_runtime_choices
from grimace._south_star1.writer_runtime import writer_runtime_diagnostics
from grimace._south_star1.writer_runtime import writer_runtime_has_eos
from grimace._south_star1.writer_runtime import writer_runtime_state_from_snapshot
from grimace._south_star1.writer_runtime import writer_runtime_terminal
from grimace._south_star1.writer_snapshot import advance_writer_frontier_snapshot
from grimace._south_star1.writer_snapshot import resume_writer_frontier_choices_from_snapshot
from grimace._south_star1.writer_support_certificates import (
    writer_support_string_certificate,
)
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
        certified = tuple(
            iter_writer_runtime_certified_support(
                prepared=prepared,
                state=state,
            )
        )
        image_certificate = writer_runtime_support_image_certificate(
            prepared=prepared,
            state=state,
            witness_count=count_writer_runtime_completions(
                prepared=prepared,
                state=state,
            ),
        )
        self.assertEqual(tuple(item.string for item in certified), support.strings)
        self.assertTrue(all(item.certificate for item in certified))
        self.assertEqual(image_certificate.strings, support.strings)
        self.assertEqual(image_certificate.distinct_count, support.distinct_count)
        self.assertEqual(image_certificate.witness_count, support.witness_count)
        self.assertEqual(
            image_certificate.string_certificates,
            tuple(item.certificate for item in certified),
        )
        for item in certified:
            certificate = item.certificate
            self.assertEqual(item.string, certificate.string)
            self.assertEqual(item.string, "".join(certificate.emitted_texts))
            self.assertIsNotNone(certificate.terminal_projection_certificate)
            self.assertTrue(certificate.terminal_certificates)
            self.assertEqual(
                certificate.replay_certificate.final_snapshot,
                certificate.final_snapshot,
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

        branch_batch = writer_runtime_branch_transitions(
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
        for choice in choices.choices:
            weighted_successors = Counter(
                {
                    branch.successor_state: 0
                    for branch in branch_batch.branch_transitions
                    if branch.emitted_text == choice.emitted_text
                }
            )
            for branch in branch_batch.branch_transitions:
                if branch.emitted_text == choice.emitted_text:
                    weighted_successors[branch.successor_state] += branch.parent_weight
            self.assertEqual(
                choice.successor,
                WriterFrontierCursor(
                    weighted_states=tuple(weighted_successors.items())
                ),
            )
            self.assertEqual(
                choice.immediate_multiplicity,
                sum(weighted_successors.values()),
            )

    def test_branch_transitions_package_provenance_successors(self) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )

        branch_transitions = writer_runtime_branch_transitions(
            prepared=prepared,
            state=state,
            include_counts=True,
        )
        branch_text_counts = Counter(
            transition.emitted_text
            for transition in branch_transitions.transitions
        )

        self.assertEqual(
            branch_transitions.branch_transitions,
            branch_transitions.transitions,
        )
        self.assertGreater(max(branch_text_counts.values()), 1)
        for transition in branch_transitions.transitions:
            self.assertEqual(
                transition.next_state.snapshot.cursor,
                WriterFrontierCursor(
                    weighted_states=((transition.successor_state, 1),)
                ),
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
            self.assertIsNotNone(transition.snapshot_step_certificate)
            self.assertEqual(
                transition.snapshot_step_certificate.emitted_text,
                transition.choice.emitted_text,
            )
            self.assertEqual(
                transition.snapshot_step_certificate.source_snapshot,
                state.snapshot,
            )
            self.assertEqual(
                transition.snapshot_step_certificate.advanced_snapshot,
                transition.next_state.snapshot,
            )
            self.assertEqual(
                transition.snapshot_step_certificate.successor_cursor,
                transition.choice.successor,
            )
            self.assertTrue(
                transition.snapshot_step_certificate.branch_certificates
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

        branches = writer_runtime_branch_transitions(
            prepared=prepared,
            state=state,
            include_counts=False,
        )
        frontier = _checked_writer_frontier_branch_supports(
            prepared,
            state.snapshot.cursor,
            include_counts=False,
        )
        self.assertIsNotNone(branches.terminal)
        self.assertTrue(branches.terminal_supports)
        self.assertEqual(
            branches.terminal.multiplicity,
            sum(support.parent_weight for support in branches.terminal_supports),
        )

        support = branches.terminal_supports[0]
        graph = _single_terminal_certificate(
            support,
            WriterTerminalCertificateKind.GRAPH_COMPLETE,
        )
        self.assertTrue(graph.graph_completion_status.complete)
        self.assertEqual(graph.graph_completion_status.unresolved_kinds, ())
        self.assertEqual(graph.graph_completion_status.unresolved_bonds, ())

        stereo = _single_terminal_certificate(
            support,
            WriterTerminalCertificateKind.STEREO_TERMINALIZED,
        )
        self.assertEqual(
            support.finalized_state.stereo_state.residual_snapshot,
            EMPTY_RESIDUAL_SNAPSHOT,
        )
        self.assertEqual(
            stereo.terminal_stereo_lifecycle_evidence,
            support.terminal_stereo_lifecycle_evidence,
        )

        finalized = _single_terminal_certificate(
            support,
            WriterTerminalCertificateKind.FINALIZED_STATE,
        )
        self.assertEqual(finalized.finalized_state, support.finalized_state)
        self.assertIsNotNone(support.checked_terminal_certificate)
        self.assertEqual(
            tuple(
                support.checked_terminal_certificate
                for support in branches.terminal_supports
            ),
            tuple(
                support.checked_terminal_certificate
                for support in frontier.terminal_supports
            ),
        )
        self.assertIsNotNone(branches.terminal_projection_certificate)
        self.assertEqual(
            branches.terminal_projection_certificate,
            frontier.terminal_projection_certificate,
        )
        self.assertEqual(
            branches.terminal_projection_certificate.terminal,
            branches.terminal,
        )
        self.assertEqual(
            branches.terminal_projection_certificate.terminal_certificates,
            tuple(
                support.checked_terminal_certificate
                for support in branches.terminal_supports
            ),
        )

    def test_initial_runtime_state_has_no_terminal_supports(self) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )
        branches = writer_runtime_branch_transitions(
            prepared=prepared,
            state=state,
            include_counts=False,
        )

        self.assertIsNone(branches.terminal)
        self.assertEqual(branches.terminal_supports, ())

    def test_terminal_certificate_rejects_nonterminal_graph(self) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )
        source = state.snapshot.cursor.weighted_states[0][0]

        with self.assertRaisesRegex(
            SouthStarError,
            "terminal graph is not complete",
        ):
            writer_terminal_certificates(
                prepared=prepared,
                source_state=source,
                finalized_state=source,
                graph_obligation_work_evidence=(),
                terminal_stereo_lifecycle_evidence=(),
                terminal_execution_capabilities=frozenset(),
                terminal_residual_work_evidence=(),
            )

    def test_checked_terminal_certificate_rejects_zero_weight(self) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )
        state = advance_writer_runtime_state(
            prepared=prepared,
            state=state,
            emitted_text="C",
        )
        state = advance_writer_runtime_state(
            prepared=prepared,
            state=state,
            emitted_text="C",
        )
        state = advance_writer_runtime_state(
            prepared=prepared,
            state=state,
            emitted_text="O",
        )
        branches = writer_runtime_branch_transitions(
            prepared=prepared,
            state=state,
            include_counts=False,
        )
        support = branches.terminal_supports[0]

        with self.assertRaisesRegex(SouthStarError, "nonpositive_parent_weight"):
            writer_checked_terminal_support_certificate(
                source_state=support.source_state,
                finalized_state=support.finalized_state,
                parent_weight=0,
                terminal_execution_capabilities=(
                    support.terminal_execution_capabilities
                ),
                terminal_residual_work_evidence=(
                    support.terminal_residual_work_evidence
                ),
                terminal_stereo_lifecycle_evidence=(
                    support.terminal_stereo_lifecycle_evidence
                ),
                graph_obligation_work_evidence=(
                    support.graph_obligation_work_evidence
                ),
                terminal_certificates=support.terminal_certificates,
            )

    def test_terminal_projection_certificate_rejects_missing_support(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )
        state = advance_writer_runtime_state(
            prepared=prepared,
            state=state,
            emitted_text="C",
        )
        state = advance_writer_runtime_state(
            prepared=prepared,
            state=state,
            emitted_text="C",
        )
        state = advance_writer_runtime_state(
            prepared=prepared,
            state=state,
            emitted_text="O",
        )
        branches = writer_runtime_branch_transitions(
            prepared=prepared,
            state=state,
            include_counts=False,
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "terminal_lacks_terminal_supports",
        ):
            writer_terminal_projection_certificate(
                terminal=branches.terminal,
                terminal_supports=(),
            )

    def test_terminal_projection_certificate_rejects_multiplicity_mismatch(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )
        state = advance_writer_runtime_state(
            prepared=prepared,
            state=state,
            emitted_text="C",
        )
        state = advance_writer_runtime_state(
            prepared=prepared,
            state=state,
            emitted_text="C",
        )
        state = advance_writer_runtime_state(
            prepared=prepared,
            state=state,
            emitted_text="O",
        )
        branches = writer_runtime_branch_transitions(
            prepared=prepared,
            state=state,
            include_counts=False,
        )
        terminal = branches.terminal

        with self.assertRaisesRegex(
            SouthStarError,
            "terminal_multiplicity_mismatch",
        ):
            writer_terminal_projection_certificate(
                terminal=terminal.__class__(
                    support_count=terminal.support_count,
                    completion_count=terminal.completion_count,
                    multiplicity=terminal.multiplicity + 1,
                    finalized_cursor=terminal.finalized_cursor,
                ),
                terminal_supports=branches.terminal_supports,
            )

    def test_support_string_certificate_rejects_malformed_inputs(self) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )
        item = next(
            iter_writer_runtime_certified_support(
                prepared=prepared,
                state=state,
            )
        )
        certificate = item.certificate

        with self.assertRaisesRegex(
            SouthStarError,
            "string_emitted_texts_mismatch",
        ):
            writer_support_string_certificate(
                source_snapshot=state.snapshot,
                string=certificate.string + "x",
                emitted_texts=certificate.emitted_texts,
                replay_certificate=certificate.replay_certificate,
                terminal_projection_certificate=(
                    certificate.terminal_projection_certificate
                ),
            )

        with self.assertRaisesRegex(
            SouthStarError,
            "replay_source_snapshot_mismatch",
        ):
            writer_support_string_certificate(
                source_snapshot=certificate.final_snapshot,
                string=certificate.string,
                emitted_texts=certificate.emitted_texts,
                replay_certificate=certificate.replay_certificate,
                terminal_projection_certificate=(
                    certificate.terminal_projection_certificate
                ),
            )

        with self.assertRaisesRegex(
            SouthStarError,
            "missing_terminal_projection_certificate",
        ):
            writer_support_string_certificate(
                source_snapshot=state.snapshot,
                string=certificate.string,
                emitted_texts=certificate.emitted_texts,
                replay_certificate=certificate.replay_certificate,
                terminal_projection_certificate=None,
            )

        with self.assertRaisesRegex(
            SouthStarError,
            "terminal_projection_lacks_certificates",
        ):
            writer_support_string_certificate(
                source_snapshot=state.snapshot,
                string=certificate.string,
                emitted_texts=certificate.emitted_texts,
                replay_certificate=certificate.replay_certificate,
                terminal_projection_certificate=replace(
                    certificate.terminal_projection_certificate,
                    terminal_certificates=(),
                ),
            )


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


def _single_terminal_certificate(
    support,
    kind: WriterTerminalCertificateKind,
):
    matches = tuple(
        certificate
        for certificate in support.terminal_certificates
        if certificate.kind is kind
    )
    if len(matches) != 1:
        raise AssertionError(
            f"expected exactly one terminal certificate {kind!r}"
        )
    return matches[0]


def _prepare(facts):
    return prepare_south_star_mol_from_facts(
        facts,
        writer_surface=SouthStarWriterSurface(),
    )


def _writer_options() -> SouthStarRuntimeOptions:
    return SouthStarRuntimeOptions(
        serialization_language=SerializationLanguageMode.WRITER_SHAPED,
    )
