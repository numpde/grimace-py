"""Writer-shaped runtime facade tests."""

from __future__ import annotations

import unittest
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from grimace._south_star1 import writer_frontier as writer_frontier_module
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import enumerate_prepared_writer_shaped_support
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.writer_branch_certificates import (
    writer_checked_terminal_support_certificate,
)
from grimace._south_star1.writer_capabilities import _WriterExecutionCapabilityKind
from grimace._south_star1.writer_blocked_frontier_certificates import (
    writer_blocked_frontier_certificate,
)
from grimace._south_star1.writer_frontier_certificates import (
    writer_checked_frontier_certificate,
)
from grimace._south_star1.writer_count_certificates import (
    WriterBranchCompletionTermCertificate,
)
from grimace._south_star1.writer_count_certificates import (
    writer_branch_completion_term_certificate,
)
from grimace._south_star1.writer_count_certificates import (
    writer_cursor_completion_count_certificate,
)
from grimace._south_star1.writer_count_certificates import writer_state_completion_count_certificate
from grimace._south_star1.writer_choice_count_certificates import (
    writer_text_choice_count_certificate,
)
from grimace._south_star1.writer_choice_count_certificates import (
    writer_terminal_choice_count_certificate,
)
from grimace._south_star1.writer_diagnostic_certificates import (
    WriterDiagnosticsCertificate,
    writer_diagnostics_certificate,
)
from grimace._south_star1.writer_frontier import WriterFrontierCursor
from grimace._south_star1.writer_frontier import _checked_writer_frontier_branch_supports
from grimace._south_star1.writer_frontier import _checked_writer_frontier_product
from grimace._south_star1.writer_frontier import _checked_writer_frontier_count_certificate
from grimace._south_star1.writer_frontier import (
    _iter_checked_writer_frontier_certified_support_strings,
)
from grimace._south_star1.writer_frontier import (
    _writer_frontier_raw_successors_for_streaming,
)
from grimace._south_star1.writer_frontier import count_writer_cursor_completions
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_frontier import writer_frontier_choices
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
from grimace._south_star1.writer_runtime import writer_runtime_branch_completion_count_certificate
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
from grimace._south_star1.writer_runtime import writer_runtime_support_count_certificate
from grimace._south_star1.writer_snapshot import advance_writer_frontier_snapshot
from grimace._south_star1.writer_snapshot import resume_writer_frontier_choices_from_snapshot
from grimace._south_star1.writer_snapshot_certificates import (
    writer_snapshot_step_certificate,
)
from grimace._south_star1.writer_support_count_certificates import (
    WriterTextSupportCountCertificate,
)
from grimace._south_star1.writer_support_count_certificates import (
    writer_text_choice_support_count_term_certificate,
)
from grimace._south_star1.writer_support_count_certificates import (
    writer_text_state_support_count_certificate,
)
from grimace._south_star1.writer_support_count_certificates import (
    writer_text_support_count_certificate,
)
from grimace._south_star1.writer_support_certificates import (
    writer_frontier_support_string_certificate,
)
from grimace._south_star1.writer_support_certificates import (
    writer_support_image_certificate,
)
from grimace._south_star1.writer_support_certificates import (
    writer_support_string_certificate,
)
from tests.south_star1.helpers import cco_facts
from tests.south_star1.helpers import tetrahedral_facts


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
        self.assertIsNotNone(image_certificate.witness_count_certificate)
        assert image_certificate.witness_count_certificate is not None
        self.assertEqual(
            image_certificate.witness_count,
            image_certificate.witness_count_certificate.completion_count,
        )
        self.assertIsNotNone(image_certificate.support_count_certificate)
        assert image_certificate.support_count_certificate is not None
        self.assertEqual(
            image_certificate.distinct_count,
            image_certificate.support_count_certificate.support_count,
        )
        count_certificate = (
            writer_runtime_branch_completion_count_certificate(
                prepared=prepared,
                state=state,
            )
        )
        self.assertEqual(
            count_writer_runtime_branch_completions(
                prepared=prepared,
                state=state,
            ),
            count_certificate.completion_count,
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

    def test_count_writer_runtime_support_is_certificate_backed(self) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )
        count_certificate = writer_runtime_support_count_certificate(
            prepared=prepared,
            state=state,
        )
        self.assertIsInstance(
            count_certificate,
            WriterTextSupportCountCertificate,
        )
        self.assertEqual(
            count_writer_runtime_support(
                prepared=prepared,
                state=state,
            ),
            count_certificate.support_count,
        )

    def test_writer_frontier_choices_are_product_backed(self) -> None:
        prepared = _prepare(cco_facts())
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())

        product = _checked_writer_frontier_product(prepared, cursor)

        self.assertEqual(writer_frontier_choices(prepared, cursor), product.choices)
        self.assertIsNotNone(product.checked_frontier_certificate)

    def test_raw_streaming_successors_follow_projection_certificates(self) -> None:
        prepared = _prepare(cco_facts())
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())

        product = _checked_writer_frontier_product(
            prepared,
            cursor,
            include_counts=False,
        )

        self.assertEqual(
            _writer_frontier_raw_successors_for_streaming(prepared, cursor),
            tuple(
                (certificate.emitted_text, certificate.successor_cursor)
                for certificate in product.text_choice_projection_certificates
            ),
        )

    def test_count_writer_cursor_completions_is_certificate_backed(self) -> None:
        prepared = _prepare(cco_facts())
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        certificate = _checked_writer_frontier_count_certificate(
            prepared=prepared,
            cursor=cursor,
        )

        self.assertEqual(
            count_writer_cursor_completions(prepared, cursor),
            certificate.completion_count,
        )

    def test_support_string_certificate_carries_projection_chain(self) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )

        certified = tuple(
            iter_writer_runtime_certified_support(
                prepared=prepared,
                state=state,
            )
        )

        self.assertTrue(certified)
        for item in certified:
            certificate = item.certificate
            self.assertEqual(item.string, certificate.string)
            self.assertEqual(item.string, "".join(certificate.emitted_texts))
            self.assertEqual(
                tuple(
                    projection.emitted_text
                    for projection in certificate.text_projection_certificates
                ),
                certificate.emitted_texts,
            )
            self.assertIsNotNone(certificate.terminal_projection_certificate)

    def test_frontier_support_string_certificate_rejects_projection_source_mismatch(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        certified = tuple(
            _iter_checked_writer_frontier_certified_support_strings(
                prepared,
                cursor,
            )
        )
        certificate = next(
            item.certificate
            for item in certified
            if item.certificate.text_projection_certificates
        )
        projection = certificate.text_projection_certificates[0]
        bad_projection = replace(
            projection,
            source_cursor=WriterFrontierCursor(weighted_states=()),
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "frontier_projection_source_cursor_mismatch",
        ):
            writer_frontier_support_string_certificate(
                source_cursor=cursor,
                string=certificate.string,
                emitted_texts=certificate.emitted_texts,
                text_projection_certificates=(
                    bad_projection,
                    *certificate.text_projection_certificates[1:],
                ),
                terminal_projection_certificate=(
                    certificate.terminal_projection_certificate
                ),
            )

    def test_frontier_support_string_certificate_rejects_terminal_source_mismatch(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        certified = tuple(
            _iter_checked_writer_frontier_certified_support_strings(
                prepared,
                cursor,
            )
        )
        certificate = next(item.certificate for item in certified)
        bad_terminal = replace(
            certificate.terminal_projection_certificate,
            source_cursor=WriterFrontierCursor(weighted_states=()),
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "frontier_terminal_projection_source_cursor_mismatch",
        ):
            writer_frontier_support_string_certificate(
                source_cursor=certificate.source_cursor,
                string=certificate.string,
                emitted_texts=certificate.emitted_texts,
                text_projection_certificates=(
                    certificate.text_projection_certificates
                ),
                terminal_projection_certificate=bad_terminal,
            )

    def test_snapshot_step_certificate_rejects_projection_source_mismatch(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )
        transition = writer_runtime_choice_transitions(
            prepared=prepared,
            state=state,
        ).transitions[0]
        step = transition.snapshot_step_certificate
        bad_projection = replace(
            step.text_projection_certificate,
            source_cursor=WriterFrontierCursor(weighted_states=()),
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "projection_source_cursor_mismatch",
        ):
            writer_snapshot_step_certificate(
                source_snapshot=state.snapshot,
                emitted_text=step.emitted_text,
                frontier_projection_certificate=replace(
                    step.frontier_projection_certificate,
                    text_choice_projection_certificates=(bad_projection,),
                ),
                text_projection_certificate=bad_projection,
                advanced_snapshot=step.advanced_snapshot,
            )

    def test_snapshot_step_rejects_projection_not_in_frontier_projection(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )
        transition = writer_runtime_choice_transitions(
            prepared=prepared,
            state=state,
        ).transitions[0]
        step = transition.snapshot_step_certificate

        with self.assertRaisesRegex(
            SouthStarError,
            "text_projection_not_in_frontier_projection",
        ):
            writer_snapshot_step_certificate(
                source_snapshot=state.snapshot,
                emitted_text=step.emitted_text,
                frontier_projection_certificate=(
                    step.frontier_projection_certificate
                ),
                text_projection_certificate=replace(
                    step.text_projection_certificate
                ),
                advanced_snapshot=step.advanced_snapshot,
            )

    def test_support_string_certificate_rejects_projection_chain_mismatch(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )
        certificate = next(
            item.certificate
            for item in iter_writer_runtime_certified_support(
                prepared=prepared,
                state=state,
            )
            if len(item.certificate.text_projection_certificates) > 1
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "support_string_replay_projection_chain_mismatch",
        ):
            writer_support_string_certificate(
                source_snapshot=state.snapshot,
                string=certificate.string,
                emitted_texts=certificate.emitted_texts,
                replay_certificate=certificate.replay_certificate,
                terminal_frontier_projection_certificate=(
                    certificate.terminal_frontier_projection_certificate
                ),
                terminal_projection_certificate=(
                    certificate.terminal_projection_certificate
                ),
                text_projection_certificates=(
                    replace(certificate.text_projection_certificates[0]),
                    *certificate.text_projection_certificates[1:],
                ),
            )

    def test_support_string_rejects_detached_terminal_projection(self) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )
        certificate = next(
            iter_writer_runtime_certified_support(
                prepared=prepared,
                state=state,
            )
        ).certificate

        with self.assertRaisesRegex(
            SouthStarError,
            "terminal_projection_not_in_frontier_projection",
        ):
            writer_support_string_certificate(
                source_snapshot=state.snapshot,
                string=certificate.string,
                emitted_texts=certificate.emitted_texts,
                replay_certificate=certificate.replay_certificate,
                terminal_frontier_projection_certificate=(
                    certificate.terminal_frontier_projection_certificate
                ),
                terminal_projection_certificate=replace(
                    certificate.terminal_projection_certificate
                ),
                text_projection_certificates=(
                    certificate.text_projection_certificates
                ),
            )

    def test_support_image_certificate_rejects_foreign_string_certificate(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )
        foreign = next(
            iter_writer_runtime_certified_support(
                prepared=prepared,
                state=state,
            )
        ).certificate
        advanced = advance_writer_runtime_state(
            prepared=prepared,
            state=state,
            emitted_text=writer_runtime_choices(
                prepared=prepared,
                state=state,
            ).choices[0].emitted_text,
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "string_certificate_source_snapshot_mismatch",
        ):
            writer_support_image_certificate(
                source_snapshot=advanced.snapshot,
                string_certificates=(foreign,),
                witness_count=1,
            )

    def test_count_certificate_matches_counted_completions(self) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )
        count_certificate = writer_runtime_branch_completion_count_certificate(
            prepared=prepared,
            state=state,
        )
        self.assertEqual(
            count_writer_runtime_branch_completions(
                prepared=prepared,
                state=state,
            ),
            count_certificate.completion_count,
        )

    def test_count_certificate_state_term_invariants(self) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )
        count_certificate = writer_runtime_branch_completion_count_certificate(
            prepared=prepared,
            state=state,
        )

        state_key, _, state_count = count_certificate.state_count_certificates[0]
        self.assertEqual(
            state_count.completion_count,
            state_count.terminal_count
            + sum(term.successor_count for term in state_count.branch_terms),
        )
        if state_count.terminal_projection_certificate is not None:
            self.assertEqual(
                state_count.terminal_projection_certificate.terminal.completion_count,
                state_count.terminal_count,
            )
        self.assertTrue(
            all(
                isinstance(term, WriterBranchCompletionTermCertificate)
                for term in state_count.branch_terms
            )
        )

        if state_count.branch_terms:
            term = state_count.branch_terms[0]
            self.assertEqual(
                term.branch_certificate.successor_state,
                term.successor_count_certificate.state_count_certificates[0][0],
            )

    def test_count_certificate_rejects_malformed_state_data(self) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )
        support_cert = writer_runtime_branch_completion_count_certificate(
            prepared=prepared,
            state=state,
        )
        state_key, _, state_count = support_cert.state_count_certificates[0]
        self.assertIsNotNone(state_count)

        with self.assertRaisesRegex(
            SouthStarError,
            "terminal_count_without_terminal_projection",
        ):
            writer_state_completion_count_certificate(
                state_key=state_key,
                terminal_projection_certificate=None,
                terminal_count=1,
                branch_terms=state_count.branch_terms,
            )

        with self.assertRaises(SouthStarError):
            writer_cursor_completion_count_certificate(
                cursor=state.snapshot.cursor,
                state_count_certificates=((state_key, 0, state_count),),
            )

        with self.assertRaisesRegex(
            SouthStarError,
            "cursor_weighted_states_mismatch",
        ):
            writer_cursor_completion_count_certificate(
                cursor=state.snapshot.cursor,
                state_count_certificates=((state_key, 2, state_count),),
            )

        if not state_count.branch_terms:
            self.skipTest("fixture has no branch terms for completion count")
        with self.assertRaisesRegex(
            SouthStarError,
            "branch_term_successor_count_mismatch",
        ):
            writer_state_completion_count_certificate(
                state_key=state_key,
                terminal_projection_certificate=(
                    state_count.terminal_projection_certificate
                ),
                terminal_count=state_count.terminal_count,
                branch_terms=(
                    replace(
                        state_count.branch_terms[0],
                        successor_count=(
                            state_count.branch_terms[0].successor_count + 1
                        ),
                    ),
                    *state_count.branch_terms[1:],
                ),
            )

    def test_support_count_certificates_reject_malformed(self) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )
        support_count_certificate = writer_runtime_support_count_certificate(
            prepared=prepared,
            state=state,
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "state_support_count_cursor_mismatch",
        ):
            writer_text_support_count_certificate(
                source_snapshot=state.snapshot,
                cursor=WriterFrontierCursor(
                    weighted_states=(
                        (state.snapshot.cursor.weighted_states[0][0], 1),
                    )
                ),
                state_support_count_certificate=(
                    support_count_certificate.state_support_count_certificate
                ),
            )

        if not support_count_certificate.state_support_count_certificate.choice_terms:
            self.skipTest("fixture has no choice terms for support count")
        first_term = (
            support_count_certificate.state_support_count_certificate.choice_terms[0]
        )
        with self.assertRaisesRegex(
            SouthStarError,
            "projection_successor_cursor_mismatch",
        ):
            writer_text_choice_support_count_term_certificate(
                text_projection_certificate=(
                    first_term.text_projection_certificate
                ),
                successor_support_count_certificate=SimpleNamespace(
                    cursor=WriterFrontierCursor(
                        weighted_states=((state.snapshot.cursor.weighted_states[0][0], 2),)
                    ),
                    support_count=first_term.support_count,
                ),
            )

        with self.assertRaisesRegex(
            SouthStarError,
            "choice_term_support_count_mismatch",
        ):
            writer_text_state_support_count_certificate(
                cursor=support_count_certificate.cursor,
                terminal_projection_certificate=(
                    support_count_certificate.state_support_count_certificate
                    .terminal_projection_certificate
                ),
                terminal_count=(
                    support_count_certificate.state_support_count_certificate
                    .terminal_count
                ),
                choice_terms=(
                    replace(
                        first_term,
                        support_count=first_term.support_count + 1,
                    ),
                    *support_count_certificate.state_support_count_certificate
                    .choice_terms[1:],
                ),
            )

        if (
            support_count_certificate.state_support_count_certificate.terminal_count
            == 0
        ):
            self.skipTest("fixture has no terminal support at this state")
        with self.assertRaisesRegex(
            SouthStarError,
            "terminal_count_mismatch",
        ):
            writer_text_state_support_count_certificate(
                cursor=support_count_certificate.cursor,
                terminal_projection_certificate=(
                    support_count_certificate.state_support_count_certificate
                    .terminal_projection_certificate
                ),
                terminal_count=0,
                choice_terms=(
                    support_count_certificate.state_support_count_certificate
                    .choice_terms
                ),
            )

    def test_branch_certificate_term_matches_singleton_successor_cursor(self) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )
        support_batch = _checked_writer_frontier_branch_supports(
            prepared,
            state.snapshot.cursor,
            include_counts=False,
        )
        support = support_batch.supports[0]
        count_certificate = writer_runtime_branch_completion_count_certificate(
            prepared=prepared,
            state=state,
        )
        state_key, _, state_count = count_certificate.state_count_certificates[0]
        with self.assertRaisesRegex(
            SouthStarError,
            "branch_successor_cursor_not_singleton",
        ):
            writer_branch_completion_term_certificate(
                branch_certificate=support.checked_branch_certificate,
                successor_count_certificate=SimpleNamespace(
                cursor=WriterFrontierCursor(
                    weighted_states=((state_key, 1), (state_key, 1))
                ),
                state_count_certificates=((state_key, 1, state_count),),
            ),
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "branch_successor_cursor_not_singleton",
        ):
            writer_branch_completion_term_certificate(
                branch_certificate=support.checked_branch_certificate,
                successor_count_certificate=SimpleNamespace(
                    cursor=WriterFrontierCursor(
                        weighted_states=((state_key, 2),)
                    ),
                    state_count_certificates=((state_key, 2, state_count),),
                ),
            )

    def test_diagnostics_certificate_blocks_blocked_state(self) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )

        blocked_diagnostics = SimpleNamespace(
            blocked=True,
            graph_policy_blockers=("blocked",),
            stereo_policy_blockers=(),
            execution_capabilities=frozenset(),
            terminal_execution_capabilities=frozenset(),
            unsupported_execution_capabilities=frozenset(),
            unsupported_terminal_execution_capabilities=frozenset(),
            residual_work_evidence=(),
            terminal_residual_work_evidence=(),
            finite_relation_work_evidence=(),
            graph_obligation_work_evidence=(),
            residual_work_envelope_violations=(),
            terminal_residual_work_envelope_violations=(),
            finite_relation_work_envelope_violations=(),
            graph_obligation_work_envelope_violations=(),
            choice_texts=(),
            has_eos=False,
        )
        branch_batch = SimpleNamespace(
            supports=(),
            terminal_supports=(),
            text_choice_projection_certificates=(),
            terminal_projection_certificate=None,
        )
        certificate = writer_diagnostics_certificate(
            cursor=state.snapshot.cursor,
            diagnostics=blocked_diagnostics,
            branch_batch=branch_batch,
        )

        self.assertTrue(certificate.blocked)
        self.assertFalse(certificate.text_choice_projection_certificates)
        self.assertFalse(certificate.terminal_projection_certificate)
        self.assertTrue(certificate.graph_policy_blocker_certificates)

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
        self.assertIsNotNone(diagnostics.diagnostic_certificate)
        self.assertIsNotNone(diagnostics.checked_frontier_certificate)
        self.assertIsNone(diagnostics.blocked_frontier_certificate)
        self.assertEqual(
            diagnostics.diagnostic_certificate.text_choice_projection_certificates,
            _checked_writer_frontier_product(
                prepared,
                state.snapshot.cursor,
            ).text_choice_projection_certificates,
        )
        self.assertEqual(
            diagnostics.diagnostic_certificate.blocked,
            diagnostics.blocked,
        )
        self.assertEqual(
            diagnostics.diagnostic_certificate.execution_capabilities,
            diagnostics.execution_capabilities,
        )
        self.assertEqual(
            diagnostics.diagnostic_certificate.terminal_execution_capabilities,
            diagnostics.terminal_execution_capabilities,
        )

    def test_blocked_frontier_certificate_binds_blocked_diagnostics(self) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )
        blocked_diagnostics = SimpleNamespace(
            blocked=True,
            graph_policy_blockers=("blocked",),
            stereo_policy_blockers=(),
            execution_capabilities=frozenset(),
            terminal_execution_capabilities=frozenset(),
            unsupported_execution_capabilities=frozenset(),
            unsupported_terminal_execution_capabilities=frozenset(),
            residual_work_evidence=(),
            terminal_residual_work_evidence=(),
            finite_relation_work_evidence=(),
            graph_obligation_work_evidence=(),
            residual_work_envelope_violations=(),
            terminal_residual_work_envelope_violations=(),
            finite_relation_work_envelope_violations=(),
            graph_obligation_work_envelope_violations=(),
            choice_texts=(),
            has_eos=False,
        )
        empty_batch = SimpleNamespace(
            supports=(),
            terminal_supports=(),
            text_choice_projection_certificates=(),
            terminal_projection_certificate=None,
        )
        diagnostic_certificate = writer_diagnostics_certificate(
            cursor=state.snapshot.cursor,
            diagnostics=blocked_diagnostics,
            branch_batch=empty_batch,
        )
        blocked_certificate = writer_blocked_frontier_certificate(
            cursor=state.snapshot.cursor,
            diagnostic_certificate=diagnostic_certificate,
        )

        self.assertTrue(blocked_certificate.blocked)
        self.assertIs(
            blocked_certificate.diagnostic_certificate,
            diagnostic_certificate,
        )
        self.assertTrue(
            blocked_certificate.graph_policy_blocker_certificates
        )

    def test_diagnostics_returns_blocked_product_for_unsupported_capability(
        self,
    ) -> None:
        prepared = _prepare(tetrahedral_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )
        product = _checked_writer_frontier_product(
            prepared,
            state.snapshot.cursor,
        )
        capability = next(
            capability
            for support in product.branch_supports
            for capability in support.execution_capabilities
        )

        def unsupported(capabilities):
            return frozenset(
                item for item in capabilities if item is capability
            )

        with patch.object(
            writer_frontier_module,
            "_unsupported_public_writer_execution_capabilities",
            unsupported,
        ):
            diagnostics = writer_runtime_diagnostics(
                prepared=prepared,
                state=state,
            )

        self.assertTrue(diagnostics.blocked)
        self.assertIsNone(diagnostics.checked_frontier_certificate)
        self.assertIsNotNone(diagnostics.blocked_frontier_certificate)
        self.assertEqual(diagnostics.choice_texts, ())
        self.assertFalse(diagnostics.has_eos)
        self.assertIn(
            capability,
            diagnostics.unsupported_execution_capabilities,
        )
        cert = diagnostics.diagnostic_certificate
        self.assertIsNotNone(cert)
        self.assertFalse(cert.text_choice_projection_certificates)
        self.assertIsNone(cert.terminal_projection_certificate)
        self.assertFalse(cert.branch_certificates)
        self.assertFalse(cert.terminal_certificates)
        self.assertIsNone(cert.count_certificate)
        self.assertTrue(
            cert.unsupported_execution_capability_certificates
        )
        self.assertTrue(
            cert
            .unsupported_execution_capability_certificates[0]
            .source_evidence
        )

    def test_runtime_choices_still_raise_for_unsupported_capability(
        self,
    ) -> None:
        prepared = _prepare(tetrahedral_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )
        product = _checked_writer_frontier_product(
            prepared,
            state.snapshot.cursor,
        )
        capability = next(
            capability
            for support in product.branch_supports
            for capability in support.execution_capabilities
        )

        def unsupported(capabilities):
            return frozenset(
                item for item in capabilities if item is capability
            )

        with patch.object(
            writer_frontier_module,
            "_unsupported_public_writer_execution_capabilities",
            unsupported,
        ):
            with self.assertRaisesRegex(
                SouthStarError,
                "unsupported South Star execution capability",
            ):
                writer_runtime_choices(prepared=prepared, state=state)

    def test_diagnostics_returns_blocked_product_for_work_envelope_violation(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )

        def fake_violation(evidence):
            return SimpleNamespace(evidence=evidence)

        with patch.object(
            writer_frontier_module,
            "writer_graph_obligation_work_envelope_violation",
            fake_violation,
        ):
            diagnostics = writer_runtime_diagnostics(
                prepared=prepared,
                state=state,
            )

        self.assertTrue(diagnostics.blocked)
        self.assertIsNone(diagnostics.checked_frontier_certificate)
        self.assertIsNotNone(diagnostics.blocked_frontier_certificate)
        self.assertEqual(diagnostics.choice_texts, ())
        self.assertFalse(diagnostics.has_eos)
        self.assertTrue(
            diagnostics.graph_obligation_work_envelope_violations
        )

    def test_diagnostics_certificate_rejects_malformed_inputs(self) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )
        branch_batch = _checked_writer_frontier_branch_supports(
            prepared,
            state.snapshot.cursor,
            include_counts=False,
        )

        base = SimpleNamespace(
            blocked=False,
            graph_policy_blockers=(),
            stereo_policy_blockers=(),
            execution_capabilities=frozenset(),
            terminal_execution_capabilities=frozenset(),
            unsupported_execution_capabilities=frozenset(),
            unsupported_terminal_execution_capabilities=frozenset(),
            residual_work_evidence=(),
            terminal_residual_work_evidence=(),
            finite_relation_work_evidence=(),
            graph_obligation_work_evidence=(),
            residual_work_envelope_violations=(),
            terminal_residual_work_envelope_violations=(),
            finite_relation_work_envelope_violations=(),
            graph_obligation_work_envelope_violations=(),
            choice_texts=("bad",),
            has_eos=False,
        )
        with self.assertRaisesRegex(
            SouthStarError,
            "choice_texts_mismatch",
        ):
            writer_diagnostics_certificate(
                cursor=state.snapshot.cursor,
                diagnostics=base,
                branch_batch=branch_batch,
            )

        bad_projection = replace(
            branch_batch.text_choice_projection_certificates[0],
            source_cursor=WriterFrontierCursor(weighted_states=()),
        )
        bad_batch = replace(
            branch_batch,
            text_choice_projection_certificates=(
                bad_projection,
                *branch_batch.text_choice_projection_certificates[1:],
            ),
        )
        with self.assertRaisesRegex(
            SouthStarError,
            "projection_source_cursor_mismatch",
        ):
            writer_diagnostics_certificate(
                cursor=state.snapshot.cursor,
                diagnostics=SimpleNamespace(**{
                    **vars(base),
                    "choice_texts": tuple(
                        cert.emitted_text
                        for cert in bad_batch.text_choice_projection_certificates
                    ),
                    "has_eos": bool(
                        bad_batch.terminal_projection_certificate is not None
                    ),
                }),
                branch_batch=bad_batch,
            )

        with self.assertRaisesRegex(
            SouthStarError,
            "unsupported_capability_lacks_source_evidence",
        ):
            writer_diagnostics_certificate(
                cursor=state.snapshot.cursor,
                diagnostics=SimpleNamespace(**{
                    **vars(base),
                    "blocked": True,
                    "choice_texts": (),
                    "has_eos": False,
                    "unsupported_execution_capabilities": frozenset((
                        _WriterExecutionCapabilityKind.TREE_CHILD_ENTRY,
                    )),
                }),
                branch_batch=SimpleNamespace(
                    supports=(),
                    terminal_supports=(),
                    text_choice_projection_certificates=(),
                    terminal_projection_certificate=None,
                ),
            )

        with self.assertRaisesRegex(
            SouthStarError,
            "has_eos_mismatch",
        ):
            writer_diagnostics_certificate(
                cursor=state.snapshot.cursor,
                diagnostics=SimpleNamespace(**{
                    **vars(base),
                    "choice_texts": tuple(
                        cert.emitted_text
                        for cert in branch_batch.text_choice_projection_certificates
                    ),
                    "has_eos": True,
                }),
                branch_batch=branch_batch,
            )

        with self.assertRaisesRegex(
            SouthStarError,
            "residual_work_violation_evidence_missing",
        ):
            writer_diagnostics_certificate(
                cursor=state.snapshot.cursor,
                diagnostics=SimpleNamespace(**{
                    **vars(base),
                    "choice_texts": tuple(
                        cert.emitted_text
                        for cert in branch_batch.text_choice_projection_certificates
                    ),
                    "has_eos": False,
                    "residual_work_envelope_violations": (
                        SimpleNamespace(evidence=object()),
                    ),
                }),
                branch_batch=branch_batch,
            )

        with self.assertRaisesRegex(
            SouthStarError,
            "count_certificate_cursor_mismatch",
        ):
            writer_diagnostics_certificate(
                cursor=state.snapshot.cursor,
                diagnostics=SimpleNamespace(**{
                    **vars(base),
                    "choice_texts": tuple(
                        cert.emitted_text
                        for cert in branch_batch.text_choice_projection_certificates
                    ),
                    "has_eos": bool(
                        branch_batch.terminal_projection_certificate is not None
                    ),
                }),
                branch_batch=branch_batch,
                count_certificate=replace(
                    writer_runtime_branch_completion_count_certificate(
                        prepared=prepared,
                        state=state,
                    ),
                    cursor=WriterFrontierCursor(weighted_states=()),
                ),
            )

    def test_blocked_frontier_certificate_rejects_positive_payload(self) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )
        product = _checked_writer_frontier_product(
            prepared,
            state.snapshot.cursor,
        )
        diagnostic_certificate = WriterDiagnosticsCertificate(
            cursor=state.snapshot.cursor,
            blocked=True,
            graph_policy_blocker_certificates=("blocked",),
            stereo_policy_blocker_certificates=(),
            execution_capabilities=frozenset(),
            terminal_execution_capabilities=frozenset(),
            unsupported_execution_capability_certificates=(),
            unsupported_terminal_execution_capability_certificates=(),
            work_envelope_violation_certificates=(),
            text_choice_projection_certificates=(
                product.text_choice_projection_certificates
            ),
            terminal_projection_certificate=None,
            branch_certificates=(),
            terminal_certificates=(),
            count_certificate=None,
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "blocked_frontier_has_text_projections",
        ):
            writer_blocked_frontier_certificate(
                cursor=state.snapshot.cursor,
                diagnostic_certificate=diagnostic_certificate,
            )

    def test_blocked_frontier_certificate_rejects_missing_negative_evidence(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )
        diagnostic_certificate = WriterDiagnosticsCertificate(
            cursor=state.snapshot.cursor,
            blocked=True,
            graph_policy_blocker_certificates=(),
            stereo_policy_blocker_certificates=(),
            execution_capabilities=frozenset(),
            terminal_execution_capabilities=frozenset(),
            unsupported_execution_capability_certificates=(),
            unsupported_terminal_execution_capability_certificates=(),
            work_envelope_violation_certificates=(),
            text_choice_projection_certificates=(),
            terminal_projection_certificate=None,
            branch_certificates=(),
            terminal_certificates=(),
            count_certificate=None,
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "blocked_frontier_lacks_negative_evidence",
        ):
            writer_blocked_frontier_certificate(
                cursor=state.snapshot.cursor,
                diagnostic_certificate=diagnostic_certificate,
            )

    def test_diagnostics_certificate_counts_linked(self) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )
        branch_batch = _checked_writer_frontier_branch_supports(
            prepared,
            state.snapshot.cursor,
            include_counts=False,
        )
        diagnostics = writer_runtime_diagnostics(
            prepared=prepared,
            state=state,
        )
        cert = diagnostics.diagnostic_certificate
        self.assertIsNotNone(cert)
        self.assertIsNotNone(cert.count_certificate)
        self.assertEqual(
            writer_runtime_branch_completion_count_certificate(
                prepared=prepared,
                state=state,
            ).completion_count,
            cert.count_certificate.completion_count,
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
            transitions.support_count_certificate.support_count
            if transitions.support_count_certificate is not None
            else sum(
                choice.support_count or 0 for choice in choices.choices
            ),
        )
        self.assertEqual(
            transitions.completion_count,
            transitions.count_certificate.completion_count
            if transitions.count_certificate is not None
            else sum(
                choice.completion_count or 0 for choice in choices.choices
            ),
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

    def test_choice_count_certificates_are_projection_backed(self) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )
        product = _checked_writer_frontier_product(
            prepared,
            state.snapshot.cursor,
        )

        self.assertEqual(
            len(product.text_choice_count_certificates),
            len(product.text_choice_projection_certificates),
        )

        for projection, count_certificate in zip(
            product.text_choice_projection_certificates,
            product.text_choice_count_certificates,
        ):
            self.assertIs(
                count_certificate.text_projection_certificate,
                projection,
            )
            self.assertEqual(
                projection.successor_cursor,
                count_certificate.support_count_certificate.cursor,
            )
            self.assertEqual(
                projection.successor_cursor,
                count_certificate.completion_count_certificate.cursor,
            )
            if projection.support_count is not None:
                self.assertEqual(
                    projection.support_count,
                    count_certificate.support_count_certificate.support_count,
                )
            if projection.completion_count is not None:
                self.assertEqual(
                    projection.completion_count,
                    count_certificate.completion_count_certificate.completion_count,
                )

        if product.terminal_projection_certificate is not None:
            self.assertIsNotNone(product.terminal_choice_count_certificate)
            self.assertIs(
                product.terminal_choice_count_certificate.terminal_projection_certificate,  # type: ignore[union-attr]
                product.terminal_projection_certificate,
            )

        if product.terminal_choice_count_certificate is not None:
            self.assertEqual(
                product.terminal_choice_count_certificate.support_count,
                product.terminal_projection_certificate.support_count,
            )

    def test_choice_count_certificate_rejects_malformed(self) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )
        product = _checked_writer_frontier_product(
            prepared,
            state.snapshot.cursor,
        )
        if not product.text_choice_projection_certificates:
            self.skipTest("fixture has no text choices")

        projection = product.text_choice_projection_certificates[0]
        successor_cursor = projection.successor_cursor
        with self.assertRaisesRegex(
            SouthStarError,
            "support_count_successor_cursor_mismatch",
        ):
            writer_text_choice_count_certificate(
                text_projection_certificate=projection,
                support_count_certificate=SimpleNamespace(
                    cursor=WriterFrontierCursor(
                        weighted_states=((state.snapshot.cursor.weighted_states[0][0], 1),)
                    ),
                    support_count=projection.support_count or 0,
                ),
                completion_count_certificate=SimpleNamespace(
                    cursor=successor_cursor,
                    completion_count=projection.completion_count or 0,
                ),
            )

        with self.assertRaisesRegex(
            SouthStarError,
            "completion_count_successor_cursor_mismatch",
        ):
            writer_text_choice_count_certificate(
                text_projection_certificate=projection,
                support_count_certificate=SimpleNamespace(
                    cursor=successor_cursor,
                    support_count=projection.support_count or 0,
                ),
                completion_count_certificate=SimpleNamespace(
                    cursor=WriterFrontierCursor(
                        weighted_states=((state.snapshot.cursor.weighted_states[0][0], 1),)
                    ),
                    completion_count=projection.completion_count or 0,
                ),
            )

        with self.assertRaisesRegex(
            SouthStarError,
            "choice_count_certificate_support_count_mismatch",
        ):
            writer_checked_frontier_certificate(
                projection_certificate=product.projection_certificate,
                text_choice_count_certificates=(
                    replace(
                        product.text_choice_count_certificates[0],
                        support_count=(
                            product.text_choice_count_certificates[0]
                            .support_count
                            + 1
                        ),
                    ),
                    *product.text_choice_count_certificates[1:],
                ),
                terminal_choice_count_certificate=(
                    product.terminal_choice_count_certificate
                ),
                support_count_certificate=product.support_count_certificate,
                count_certificate=product.count_certificate,
            )

        with self.assertRaisesRegex(
            SouthStarError,
            "support_count_state_certificate_total_mismatch",
        ):
            writer_checked_frontier_certificate(
                projection_certificate=product.projection_certificate,
                text_choice_count_certificates=(
                    product.text_choice_count_certificates
                ),
                terminal_choice_count_certificate=(
                    product.terminal_choice_count_certificate
                ),
                support_count_certificate=replace(
                    product.support_count_certificate,
                    state_support_count_certificate=replace(
                        product.support_count_certificate
                        .state_support_count_certificate,
                        support_count=(
                            product.support_count_certificate
                            .state_support_count_certificate
                            .support_count
                            + 1
                        ),
                    ),
                ),
                count_certificate=product.count_certificate,
            )

        if product.terminal_projection_certificate is None:
            self.skipTest("fixture has no terminal projection")
        with self.assertRaisesRegex(
            SouthStarError,
            "terminal_projection_lacks_terminal",
        ):
            writer_terminal_choice_count_certificate(
                terminal_projection_certificate=SimpleNamespace(
                    terminal=None,
                )
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
                source_cursor=state.snapshot.cursor,
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
                source_cursor=state.snapshot.cursor,
                terminal=terminal.__class__(
                    support_count=terminal.support_count,
                    completion_count=terminal.completion_count,
                    multiplicity=terminal.multiplicity + 1,
                    finalized_cursor=terminal.finalized_cursor,
                ),
                terminal_supports=branches.terminal_supports,
            )

    def test_checked_frontier_rejects_terminal_projection_source_mismatch(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )
        for emitted_text in ("C", "C", "O"):
            state = advance_writer_runtime_state(
                prepared=prepared,
                state=state,
                emitted_text=emitted_text,
            )
        product = _checked_writer_frontier_product(
            prepared,
            state.snapshot.cursor,
        )
        if product.terminal_projection_certificate is None:
            self.skipTest("fixture state has no terminal projection")
        if product.terminal_choice_count_certificate is None:
            self.skipTest("fixture state has no terminal choice count")

        with self.assertRaisesRegex(
            SouthStarError,
            "terminal_choice_count_support_count_mismatch",
        ):
            writer_checked_frontier_certificate(
                projection_certificate=product.projection_certificate,
                text_choice_count_certificates=(
                    product.text_choice_count_certificates
                ),
                terminal_choice_count_certificate=replace(
                    product.terminal_choice_count_certificate,
                    support_count=(
                        product.terminal_choice_count_certificate
                        .support_count
                        + 1
                    ),
                ),
                support_count_certificate=None,
                count_certificate=product.count_certificate,
            )

        bad_terminal = replace(
            product.terminal_projection_certificate,
            source_cursor=WriterFrontierCursor(weighted_states=()),
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "terminal_projection_source_cursor_mismatch",
        ):
            writer_checked_frontier_certificate(
                cursor=state.snapshot.cursor,
                choices=product.choices,
                branch_supports=product.branch_supports,
                terminal_supports=product.terminal_supports,
                text_choice_projection_certificates=(
                    product.text_choice_projection_certificates
                ),
                text_choice_count_certificates=(
                    product.text_choice_count_certificates
                ),
                terminal_choice_count_certificate=(
                    product.terminal_choice_count_certificate
                ),
                support_count_certificate=product.support_count_certificate,
                terminal_projection_certificate=bad_terminal,
                count_certificate=product.count_certificate,
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
                terminal_frontier_projection_certificate=(
                    certificate.terminal_frontier_projection_certificate
                ),
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
                terminal_frontier_projection_certificate=(
                    certificate.terminal_frontier_projection_certificate
                ),
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
                terminal_frontier_projection_certificate=(
                    certificate.terminal_frontier_projection_certificate
                ),
                terminal_projection_certificate=None,
            )

        with self.assertRaisesRegex(
            SouthStarError,
            "terminal_projection_lacks_certificates",
        ):
            bad_terminal_projection = replace(
                certificate.terminal_projection_certificate,
                terminal_certificates=(),
            )
            writer_support_string_certificate(
                source_snapshot=state.snapshot,
                string=certificate.string,
                emitted_texts=certificate.emitted_texts,
                replay_certificate=certificate.replay_certificate,
                terminal_frontier_projection_certificate=replace(
                    certificate.terminal_frontier_projection_certificate,
                    terminal_projection_certificate=bad_terminal_projection,
                ),
                terminal_projection_certificate=bad_terminal_projection,
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
