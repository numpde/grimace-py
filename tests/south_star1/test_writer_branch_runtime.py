"""Tests for the public branch-preserving writer runtime surface."""

from __future__ import annotations

import inspect
import unittest
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import grimace._south_star1.writer_frontier as writer_frontier_module
import grimace._south_star1.writer_state_delta_certificates as writer_state_delta_certificates
import grimace._south_star1.writer_transitions as writer_transitions
from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.errors import SouthStarErrorKind
from grimace._south_star1.facts import BondOrder
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
from grimace._south_star1.writer_events import WriterAtomEmitted
from grimace._south_star1.writer_events import WriterBondEmitted
from grimace._south_star1.writer_events import WriterLocalOrderClosed
from grimace._south_star1.writer_capabilities import _WriterExecutionCapabilityKind
from grimace._south_star1.writer_branch_certificates import (
    writer_checked_branch_support_certificate,
)
from grimace._south_star1.writer_capability_certificates import (
    WriterCapabilityCoverageCertificate,
    WriterCapabilityCertificateKind,
)
from grimace._south_star1.writer_closure_candidate_branch_certificates import (
    WriterClosureCandidateBranchCertificateKind,
)
from grimace._south_star1.writer_closure_candidate_branch_certificates import (
    writer_closure_candidate_branch_certificates,
)
from grimace._south_star1.writer_closure_candidate_lifecycle import (
    WriterClosureCandidateLifecycleOutcomeKind,
)
from grimace._south_star1.writer_closure_candidate_lifecycle import (
    validate_writer_closure_candidate_lifecycle_transition,
)
from grimace._south_star1.writer_closure_candidate_lifecycle import (
    writer_closure_candidate_lifecycle_transition_violations,
)
from grimace._south_star1.writer_execution_evidence import (
    writer_graph_obligation_work_envelope_violation,
)
from grimace._south_star1.writer_frontier import WriterFrontierCursor
from grimace._south_star1.writer_frontier import WriterFrontierTerminal
from grimace._south_star1.writer_frontier import _checked_writer_frontier_branch_supports
from grimace._south_star1.writer_frontier import _checked_writer_frontier_product
from grimace._south_star1.writer_frontier import _checked_writer_frontier_schedule_outcome
from grimace._south_star1.writer_frontier import _count_checked_writer_frontier_branch_completions
from grimace._south_star1.writer_frontier import _writer_frontier_diagnostics
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_graph_obligations import WriterBoundaryOwnerKind
from grimace._south_star1.writer_graph_obligations import (
    WriterClosureCandidateResolutionKind,
)
from grimace._south_star1.writer_graph_obligations import WriterControlLiveAtomRole
from grimace._south_star1.writer_graph_obligations import WriterEdgeObligationKind
from grimace._south_star1.writer_graph_obligations import WriterResidualAttachmentActionKind
from grimace._south_star1.writer_graph_obligations import build_writer_graph_obligation_context
from grimace._south_star1.writer_graph_obligations import writer_control_live_roles_by_atom
from grimace._south_star1.writer_graph_obligations import writer_graph_obligation_work_evidence
from grimace._south_star1.writer_projection_certificates import (
    writer_text_choice_projection_certificates,
)
from grimace._south_star1.writer_projection_certificates import (
    writer_terminal_projection_certificate,
)
from grimace._south_star1.writer_projection_certificates import (
    WriterTextChoiceProjectionCertificate,
)
from grimace._south_star1.writer_frontier_certificates import (
    writer_checked_frontier_certificate,
)
from grimace._south_star1.writer_frontier_certificates import (
    writer_frontier_projection_certificate,
)
from grimace._south_star1.writer_count_certificates import (
    writer_cursor_completion_count_certificate,
)
from grimace._south_star1.writer_count_certificates import (
    writer_state_completion_count_certificate,
)
from grimace._south_star1.writer_residual_attachment_branch_certificates import (
    WriterResidualAttachmentBranchCertificateKind,
)
from grimace._south_star1.writer_residual_attachment_branch_certificates import (
    writer_residual_attachment_branch_certificates,
)
from grimace._south_star1.writer_residual_attachment_lifecycle import (
    WriterResidualAttachmentLifecycleOutcomeKind,
)
from grimace._south_star1.writer_residual_attachment_lifecycle import (
    validate_writer_residual_attachment_lifecycle_transition,
)
from grimace._south_star1.writer_stereo_branch_certificates import (
    WriterStereoBranchCertificateKind,
)
from grimace._south_star1.writer_stereo_branch_certificates import (
    writer_stereo_branch_certificates,
)
from grimace._south_star1.writer_state_delta_certificates import (
    writer_branch_successor_state_certificate,
)
from grimace._south_star1.writer_runtime import count_writer_runtime_branch_completions
from grimace._south_star1.writer_runtime import initial_writer_runtime_state
from grimace._south_star1.writer_runtime import writer_runtime_branch_transitions
from grimace._south_star1.writer_runtime import writer_runtime_choice_transitions
from grimace._south_star1.writer_runtime import writer_runtime_choices
from grimace._south_star1.writer_runtime import writer_runtime_diagnostics
from grimace._south_star1.writer_runtime import (
    writer_runtime_branch_completion_count_certificate,
)
from grimace._south_star1.writer_snapshot import _writer_search_snapshot_after_checked_branch_support
from grimace._south_star1.writer_snapshot import _writer_search_snapshot_after_checked_choice
from grimace._south_star1.writer_state import ComponentCursor
from grimace._south_star1.writer_state import ObligationState
from grimace._south_star1.writer_state import PendingWriterEntry
from grimace._south_star1.writer_state import WriterAtomFrame
from grimace._south_star1.writer_state import WriterBranchFrame
from grimace._south_star1.writer_state import WriterClosureLabel
from grimace._south_star1.writer_state import WriterPolicyState
from grimace._south_star1.writer_state import WriterPolicyStateKey
from grimace._south_star1.writer_state import WriterRingState
from grimace._south_star1.writer_state import WriterState
from grimace._south_star1.writer_state import WriterStereoState
from grimace._south_star1.writer_state import writer_state_key
from tests.south_star1.helpers import cco_facts
from tests.south_star1.helpers import cyclopropane_facts
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import tetrahedral_facts
from tests.south_star1.helpers import atom
from tests.south_star1.helpers import bond
from tests.south_star1.helpers import single_bond


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

    def test_runtime_branch_transition_certificate_carries_support_identity(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )

        transitions = writer_runtime_branch_transitions(
            prepared=prepared,
            state=state,
            include_counts=False,
        )

        self.assertTrue(transitions.transitions)
        for transition in transitions.transitions:
            certificate = transition.checked_branch_certificate
            self.assertIsNotNone(certificate)
            self.assertEqual(certificate.parent_weight, transition.parent_weight)
            self.assertEqual(certificate.branch_ordinal, transition.branch_ordinal)
            self.assertEqual(certificate.source_state, transition.source_state)
            self.assertEqual(
                certificate.successor_state,
                transition.successor_state,
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
            self.assertEqual(
                projected.graph_obligation_work_evidence,
                tuple(
                    evidence
                    for state_outcome in schedule.state_outcomes
                    if state_outcome.state_key == raw.state_key
                    for evidence in state_outcome.graph_obligation_work_evidence
                ),
            )
            self.assertEqual(projected.graph_action_surface, raw.graph_action_surface)
            self.assertEqual(projected.policy_family, raw.policy_family)
            self.assertEqual(
                projected.closure_candidate_resolution_evidence,
                (),
            )
            self.assertEqual(
                projected.closure_candidate_lifecycle_evidence,
                (),
            )
            self.assertEqual(
                projected.closure_candidate_branch_certificates,
                (),
            )
            self.assertEqual(projected.residual_attachment_policy_evidence, ())
            self.assertIsNotNone(projected.checked_branch_certificate)
            self.assertEqual(
                projected.checked_branch_certificate.source_state,
                projected.source_state,
            )
            self.assertEqual(
                projected.checked_branch_certificate.successor_state,
                projected.successor_state,
            )
            self.assertEqual(
                projected.checked_branch_certificate.events,
                projected.events,
            )

    def test_checked_frontier_batch_has_checked_frontier_certificate(self) -> None:
        prepared = _prepare(cco_facts())
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            cursor,
        )
        cert = batch.checked_frontier_certificate
        self.assertIsNotNone(cert)
        assert cert is not None

        self.assertEqual(cert.cursor, cursor)
        self.assertEqual(cert.choices, batch.choices)
        self.assertEqual(
            cert.branch_certificates,
            tuple(
                support.checked_branch_certificate
                for support in batch.supports
            ),
        )
        self.assertEqual(
            cert.terminal_certificates,
            tuple(
                support.checked_terminal_certificate
                for support in batch.terminal_supports
            ),
        )
        self.assertEqual(
            cert.text_choice_projection_certificates,
            batch.text_choice_projection_certificates,
        )
        self.assertEqual(
            cert.terminal_projection_certificate,
            batch.terminal_projection_certificate,
        )
        self.assertIs(
            cert.projection_certificate,
            batch.projection_certificate,
        )
        self.assertIs(cert.count_certificate, batch.count_certificate)
        self.assertIsNone(cert.diagnostic_certificate)

    def test_uncounted_product_has_projection_certificate_without_counts(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        product = _checked_writer_frontier_product(
            prepared,
            cursor,
            include_counts=False,
            include_frontier_certificate=False,
            include_count_certificate=False,
        )

        self.assertIsNotNone(product.projection_certificate)
        self.assertIsNone(product.count_certificate)
        self.assertIsNone(product.checked_frontier_certificate)
        self.assertEqual(
            product.projection_certificate.text_choice_projection_certificates,
            product.text_choice_projection_certificates,
        )

    def test_checked_frontier_batch_lacks_certificate_when_counts_disabled(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            cursor,
            include_counts=False,
            include_frontier_certificate=False,
        )
        self.assertIsNone(batch.checked_frontier_certificate)

    def test_checked_frontier_certificate_partitions_text_projection(
        self,
    ) -> None:
        prepared = _prepare(_two_live_closure_candidate_facts())
        cursor = WriterFrontierCursor(
            weighted_states=(
                (
                    writer_state_key(
                        _multi_live_branch_return_closure_candidate_state()
                    ),
                    1,
                ),
            )
        )
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            cursor,
        )
        cert = batch.checked_frontier_certificate
        self.assertIsNotNone(cert)
        assert cert is not None

        branch_certificates = tuple(
            support.checked_branch_certificate
            for support in batch.supports
            if support.checked_branch_certificate is not None
        )
        projected = tuple(
            branch_certificate
            for projection in cert.text_choice_projection_certificates
            for branch_certificate in projection.branch_certificates
        )
        self.assertEqual(
            Counter(id(certificate) for certificate in projected),
            Counter(id(certificate) for certificate in branch_certificates),
        )

        same_text_choices = tuple(
            cert for cert in batch.text_choice_projection_certificates
            if len(cert.branch_certificates) > 1
        )
        self.assertTrue(
            any(
                len(proj.branch_certificates) > 1
                for proj in same_text_choices
            )
        )

    def test_checked_frontier_certificate_count_certificate_matches_runtime_count(
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
        )
        cert = batch.count_certificate
        self.assertIsNotNone(cert)
        assert cert is not None
        self.assertEqual(
            count_writer_runtime_branch_completions(
                prepared=prepared,
                state=state,
            ),
            cert.completion_count,
        )
        self.assertEqual(cert.cursor, state.snapshot.cursor)

    def test_checked_frontier_product_equals_compatibility_batch(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        product = _checked_writer_frontier_product(
            prepared,
            cursor,
            include_counts=False,
        )
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            cursor,
            include_counts=False,
        )

        self.assertEqual(product.branch_batch, batch)
        self.assertIsNone(product.count_certificate)
        self.assertIsNone(product.checked_frontier_certificate)

    def test_runtime_choice_and_product_align(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )
        product = _checked_writer_frontier_product(
            prepared,
            state.snapshot.cursor,
            include_counts=True,
        )
        runtime = writer_runtime_branch_transitions(
            prepared=prepared,
            state=state,
            include_counts=True,
        )

        self.assertEqual(product.choices, runtime.choices)
        self.assertEqual(product.count_certificate, runtime.count_certificate)
        self.assertEqual(
            product.text_choice_projection_certificates,
            runtime.text_choice_projection_certificates,
        )
        self.assertEqual(
            product.terminal_projection_certificate,
            runtime.terminal_projection_certificate,
        )
        self.assertEqual(
            product.checked_frontier_certificate,
            runtime.checked_frontier_certificate,
        )

    def test_checked_frontier_count_certificate_drives_count_function(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        product = _checked_writer_frontier_product(
            prepared,
            cursor,
            include_counts=True,
        )

        self.assertEqual(
            _count_checked_writer_frontier_branch_completions(
                prepared,
                cursor,
            ),
            product.count_certificate.completion_count,
        )

    def test_runtime_branch_transitions_preserve_checked_frontier_certificate(self) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )
        runtime = writer_runtime_branch_transitions(
            prepared=prepared,
            state=state,
            include_counts=True,
        )
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            state.snapshot.cursor,
        )
        self.assertIsNotNone(batch.checked_frontier_certificate)
        self.assertEqual(
            runtime.checked_frontier_certificate,
            batch.checked_frontier_certificate,
        )

    def test_checked_frontier_certificate_rejects_missing_branch_certificate(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            cursor,
        )
        malformed_supports = (
            replace(batch.supports[0], checked_branch_certificate=None),
            *batch.supports[1:],
        )
        with self.assertRaisesRegex(
            SouthStarError,
            "missing_branch_certificate",
        ):
            writer_checked_frontier_certificate(
                cursor=cursor,
                choices=batch.choices,
                branch_supports=malformed_supports,
                terminal_supports=batch.terminal_supports,
                text_choice_projection_certificates=(
                    batch.text_choice_projection_certificates
                ),
                terminal_projection_certificate=(
                    batch.terminal_projection_certificate
                ),
                count_certificate=batch.count_certificate,
            )

    def test_checked_frontier_certificate_rejects_terminal_choice_without_projection(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            cursor,
        )

        fake_choice = batch.choices.__class__(
            terminal=WriterFrontierTerminal(
                1,
                1,
                1,
                WriterFrontierCursor(weighted_states=()),
            ),
            choices=(),
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "terminal_choice_lacks_projection_certificate",
        ):
            writer_checked_frontier_certificate(
                cursor=cursor,
                choices=fake_choice,
                branch_supports=(),
                terminal_supports=(),
                text_choice_projection_certificates=(),
                terminal_projection_certificate=None,
                count_certificate=writer_cursor_completion_count_certificate(
                    cursor=WriterFrontierCursor(weighted_states=((batch.supports[0].source_state, 1),)),
                    state_count_certificates=(
                        (
                            batch.supports[0].source_state,
                            1,
                            writer_state_completion_count_certificate(
                                state_key=batch.supports[0].source_state,
                                terminal_projection_certificate=None,
                                terminal_count=0,
                                branch_terms=(),
                            ),
                        ),
                    ),
                ),
            )

    def test_checked_frontier_certificate_rejects_count_cursor_mismatch(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            cursor,
            include_counts=True,
        )

        wrong_cursor = WriterFrontierCursor(
            weighted_states=((batch.supports[0].source_state, 2),)
        )
        wrong_count = writer_cursor_completion_count_certificate(
            cursor=wrong_cursor,
            state_count_certificates=(
                (
                    batch.supports[0].source_state,
                    2,
                    writer_state_completion_count_certificate(
                        state_key=batch.supports[0].source_state,
                        terminal_projection_certificate=None,
                        terminal_count=0,
                        branch_terms=(),
                    ),
                ),
            ),
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "count_certificate_cursor_mismatch",
        ):
            writer_checked_frontier_certificate(
                cursor=cursor,
                choices=batch.choices,
                branch_supports=batch.supports,
                terminal_supports=batch.terminal_supports,
                text_choice_projection_certificates=(
                    batch.text_choice_projection_certificates
                ),
                terminal_projection_certificate=
                batch.terminal_projection_certificate,
                count_certificate=wrong_count,
            )

    def test_checked_frontier_certificate_rejects_projection_partition_mismatch(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        batch = _checked_writer_frontier_branch_supports(prepared, cursor)
        if not batch.text_choice_projection_certificates:
            self.skipTest("no text choices available in this frontier")

        first = batch.text_choice_projection_certificates[0]
        if not first.branch_certificates:
            self.skipTest("expected at least one branch certificate")

        malformed_projection = WriterTextChoiceProjectionCertificate(
            source_cursor=first.source_cursor,
            emitted_text=first.emitted_text,
            choice=first.choice,
            branch_certificates=first.branch_certificates[:-1],
            successor_cursor=first.successor_cursor,
            immediate_multiplicity=first.immediate_multiplicity,
            support_count=first.support_count,
            completion_count=first.completion_count,
        )
        malformed_projections = (
            malformed_projection,
            *batch.text_choice_projection_certificates[1:],
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "projection_branch_certificate_partition_mismatch",
        ):
            writer_checked_frontier_certificate(
                cursor=cursor,
                choices=batch.choices,
                branch_supports=batch.supports,
                terminal_supports=batch.terminal_supports,
                text_choice_projection_certificates=malformed_projections,
                terminal_projection_certificate=(
                    batch.terminal_projection_certificate
                ),
                count_certificate=batch.count_certificate,
            )

    def test_frontier_projection_rejects_branch_certificate_without_successor_proof(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            cursor,
            include_counts=False,
        )
        if not batch.supports:
            self.skipTest("fixture has no branch supports")

        support = batch.supports[0]
        bad_certificate = SimpleNamespace(
            source_state=support.source_state,
            successor_state=support.successor_state,
            emitted_text=support.emitted_text,
        )
        bad_support = replace(
            support,
            checked_branch_certificate=bad_certificate,
        )
        bad_projection = replace(
            batch.text_choice_projection_certificates[0],
            branch_certificates=(bad_certificate,),
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "branch_certificate_lacks_successor_state_certificate",
        ):
            writer_frontier_projection_certificate(
                cursor=cursor,
                choices=batch.choices,
                branch_supports=(bad_support, *batch.supports[1:]),
                terminal_supports=batch.terminal_supports,
                text_choice_projection_certificates=(
                    bad_projection,
                    *batch.text_choice_projection_certificates[1:],
                ),
                terminal_projection_certificate=(
                    batch.terminal_projection_certificate
                ),
            )

    def test_checked_branch_supports_have_aggregate_certificates(self) -> None:
        for facts in (
            cco_facts(),
            cyclopropane_facts(),
            tetrahedral_facts(),
            directional_facts(),
        ):
            prepared = _prepare(facts)
            initial = initial_writer_frontier_cursor(
                prepared,
                _writer_options(),
            )
            batch = _checked_writer_frontier_branch_supports(
                prepared,
                initial,
                include_counts=False,
            )

            self.assertTrue(batch.supports)
            for support in batch.supports:
                certificate = support.checked_branch_certificate
                self.assertIsNotNone(certificate)
                self.assertEqual(certificate.source_state, support.source_state)
                self.assertEqual(
                    certificate.successor_state,
                    support.successor_state,
                )
                self.assertEqual(
                    certificate.execution_capabilities,
                    support.execution_capabilities,
                )
                self.assertEqual(
                    certificate.closure_candidate_branch_certificates,
                    support.closure_candidate_branch_certificates,
                )
                self.assertEqual(
                    certificate.residual_attachment_branch_certificates,
                    support.residual_attachment_branch_certificates,
                )
                self.assertEqual(
                    certificate.stereo_branch_certificates,
                    support.stereo_branch_certificates,
                )
                self.assertIsNotNone(support.successor_state_certificate)
                self.assertIs(
                    certificate.successor_state_certificate,
                    support.successor_state_certificate,
                )
                self.assertEqual(
                    support.successor_state_certificate.source_state,
                    support.source_state,
                )
                self.assertEqual(
                    support.successor_state_certificate.successor_state,
                    support.successor_state,
                )

    def test_checked_branch_certificate_rejects_successor_state_mismatch(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _checked_writer_frontier_branch_supports(
            prepared,
            initial,
            include_counts=False,
        ).supports[0]

        with self.assertRaisesRegex(
            SouthStarError,
            "successor_certificate_successor_mismatch",
        ):
            writer_checked_branch_support_certificate(
                **_checked_branch_certificate_kwargs(
                    support,
                    successor_state_certificate=replace(
                        support.successor_state_certificate,
                        successor_state=support.source_state,
                    ),
                )
            )

    def test_checked_branch_certificate_rejects_nonpositive_parent_weight(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _checked_writer_frontier_branch_supports(
            prepared,
            initial,
            include_counts=False,
        ).supports[0]

        with self.assertRaisesRegex(
            SouthStarError,
            "nonpositive_parent_weight",
        ):
            writer_checked_branch_support_certificate(
                **_checked_branch_certificate_kwargs(
                    support,
                    parent_weight=0,
                )
            )

    def test_checked_branch_certificate_rejects_negative_branch_ordinal(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _checked_writer_frontier_branch_supports(
            prepared,
            initial,
            include_counts=False,
        ).supports[0]

        with self.assertRaisesRegex(
            SouthStarError,
            "negative_branch_ordinal",
        ):
            writer_checked_branch_support_certificate(
                **_checked_branch_certificate_kwargs(
                    support,
                    branch_ordinal=-1,
                )
            )

    def test_checked_branch_certificate_rejects_successor_graph_obligation_evidence_mismatch(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _checked_writer_frontier_branch_supports(
            prepared,
            initial,
            include_counts=False,
        ).supports[0]

        with self.assertRaisesRegex(
            SouthStarError,
            "successor_certificate_graph_obligation_evidence_mismatch",
        ):
            writer_checked_branch_support_certificate(
                **_checked_branch_certificate_kwargs(
                    support,
                    successor_state_certificate=replace(
                        support.successor_state_certificate,
                        graph_obligation_work_evidence=(object(),),
                    ),
                )
            )

    def test_checked_branch_certificate_rejects_successor_residual_work_mismatch(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _checked_writer_frontier_branch_supports(
            prepared,
            initial,
            include_counts=False,
        ).supports[0]

        with self.assertRaisesRegex(
            SouthStarError,
            "successor_certificate_residual_work_evidence_mismatch",
        ):
            writer_checked_branch_support_certificate(
                **_checked_branch_certificate_kwargs(
                    support,
                    successor_state_certificate=replace(
                        support.successor_state_certificate,
                        residual_work_evidence=(object(),),
                    ),
                )
            )

    def test_checked_branch_certificate_rejects_successor_finite_relation_evidence_mismatch(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _checked_writer_frontier_branch_supports(
            prepared,
            initial,
            include_counts=False,
        ).supports[0]

        with self.assertRaisesRegex(
            SouthStarError,
            "successor_certificate_finite_relation_evidence_mismatch",
        ):
            writer_checked_branch_support_certificate(
                **_checked_branch_certificate_kwargs(
                    support,
                    successor_state_certificate=replace(
                        support.successor_state_certificate,
                        finite_relation_work_evidence=(object(),),
                    ),
                )
            )

    def test_checked_branch_certificate_rejects_successor_closure_lifecycle_mismatch(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _checked_writer_frontier_branch_supports(
            prepared,
            initial,
            include_counts=False,
        ).supports[0]

        with self.assertRaisesRegex(
            SouthStarError,
            "successor_certificate_closure_lifecycle_evidence_mismatch",
        ):
            writer_checked_branch_support_certificate(
                **_checked_branch_certificate_kwargs(
                    support,
                    successor_state_certificate=replace(
                        support.successor_state_certificate,
                        closure_candidate_lifecycle_evidence=(object(),),
                    ),
                )
            )

    def test_checked_branch_certificate_rejects_successor_residual_attachment_lifecycle_mismatch(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _checked_writer_frontier_branch_supports(
            prepared,
            initial,
            include_counts=False,
        ).supports[0]

        with self.assertRaisesRegex(
            SouthStarError,
            (
                "successor_certificate_residual_attachment_lifecycle_evidence"
                "_mismatch"
            ),
        ):
            writer_checked_branch_support_certificate(
                **_checked_branch_certificate_kwargs(
                    support,
                    successor_state_certificate=replace(
                        support.successor_state_certificate,
                        residual_attachment_lifecycle_evidence=(object(),),
                    ),
                )
            )

    def test_checked_branch_certificate_rejects_successor_stereo_lifecycle_mismatch(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _checked_writer_frontier_branch_supports(
            prepared,
            initial,
            include_counts=False,
        ).supports[0]

        with self.assertRaisesRegex(
            SouthStarError,
            "successor_certificate_stereo_lifecycle_evidence_mismatch",
        ):
            writer_checked_branch_support_certificate(
                **_checked_branch_certificate_kwargs(
                    support,
                    successor_state_certificate=replace(
                        support.successor_state_certificate,
                        stereo_lifecycle_evidence=(object(),),
                    ),
                )
            )

    def test_successor_state_certificate_evidence_matches_branch_support(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            initial,
            include_counts=False,
        )

        self.assertTrue(batch.supports)
        for support in batch.supports:
            certificate = support.successor_state_certificate
            self.assertEqual(
                certificate.graph_obligation_work_evidence,
                support.graph_obligation_work_evidence,
            )
            self.assertEqual(
                certificate.residual_work_evidence,
                support.residual_work_evidence,
            )
            self.assertEqual(
                certificate.finite_relation_work_evidence,
                support.finite_relation_work_evidence,
            )
            self.assertEqual(
                certificate.closure_candidate_lifecycle_evidence,
                support.closure_candidate_lifecycle_evidence,
            )
            self.assertEqual(
                certificate.residual_attachment_lifecycle_evidence,
                support.residual_attachment_lifecycle_evidence,
            )
            self.assertEqual(
                certificate.stereo_lifecycle_evidence,
                support.stereo_lifecycle_evidence,
            )

    def test_checked_branch_certificate_identity_matches_branch_support(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            initial,
            include_counts=False,
        )

        self.assertTrue(batch.supports)
        for support in batch.supports:
            certificate = support.checked_branch_certificate
            self.assertEqual(certificate.parent_weight, support.parent_weight)
            self.assertEqual(certificate.branch_ordinal, support.branch_ordinal)

    def test_closure_candidate_lifecycle_replay_present_when_evidence_exists(
        self,
    ) -> None:
        prepared = _prepare(cyclopropane_facts())
        cursor = WriterFrontierCursor(
            weighted_states=(
                (
                    writer_state_key(
                        _branch_return_closure_candidate_state(
                            live_partner=True,
                        )
                    ),
                    1,
                ),
            )
        )
        support = _find_checked_branch_support(
            prepared,
            cursor,
            lambda support: bool(
                support.closure_candidate_lifecycle_evidence
            ),
        )

        replay = (
            support.successor_state_certificate
            .closure_candidate_lifecycle_replay_certificate
        )

        self.assertIsNotNone(replay)
        self.assertTrue(replay.replay_complete)
        self.assertEqual(
            replay.lifecycle_evidence,
            support.closure_candidate_lifecycle_evidence,
        )
        self.assertEqual(
            len(replay.replay_terms),
            len(support.closure_candidate_lifecycle_evidence),
        )

    def test_closure_lifecycle_replay_rejects_stale_evidence(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        cursor = WriterFrontierCursor(
            weighted_states=(
                (
                    writer_state_key(
                        _branch_return_closure_candidate_state(
                            live_partner=True,
                        )
                    ),
                    1,
                ),
            )
        )
        support = _find_checked_branch_support(
            prepared,
            cursor,
            lambda support: bool(
                support.closure_candidate_lifecycle_evidence
            ),
        )
        certificate = support.successor_state_certificate
        bad_certificate = replace(
            certificate,
            closure_candidate_lifecycle_replay_certificate=replace(
                certificate.closure_candidate_lifecycle_replay_certificate,
                lifecycle_evidence=(),
            ),
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "closure_lifecycle_replay_evidence_mismatch",
        ):
            (
                writer_state_delta_certificates
                .validate_writer_branch_successor_state_certificate(
                    bad_certificate
                )
            )

    def test_closure_lifecycle_replay_rejects_wrong_kind(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        cursor = WriterFrontierCursor(
            weighted_states=(
                (
                    writer_state_key(
                        _branch_return_closure_candidate_state(
                            live_partner=True,
                        )
                    ),
                    1,
                ),
            )
        )
        support = _find_checked_branch_support(
            prepared,
            cursor,
            lambda support: (
                support.successor_state_certificate
                .closure_candidate_lifecycle_replay_certificate
                is not None
            ),
        )
        certificate = support.successor_state_certificate
        bad_certificate = replace(
            certificate,
            closure_candidate_lifecycle_replay_certificate=replace(
                certificate.closure_candidate_lifecycle_replay_certificate,
                kind=(
                    writer_state_delta_certificates
                    .WriterClosureCandidateLifecycleReplayKind
                    .EVIDENCE_BOUND_INCOMPLETE
                ),
            ),
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "closure_lifecycle_replay_kind_mismatch",
        ):
            (
                writer_state_delta_certificates
                .validate_writer_branch_successor_state_certificate(
                    bad_certificate
                )
            )

    def test_residual_attachment_lifecycle_replay_present_when_evidence_exists(
        self,
    ) -> None:
        prepared = _prepare(cyclopropane_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _find_checked_branch_support(
            prepared,
            initial,
            lambda support: bool(
                support.residual_attachment_lifecycle_evidence
            ),
        )

        replay = (
            support.successor_state_certificate
            .residual_attachment_lifecycle_replay_certificate
        )

        self.assertIsNotNone(replay)
        self.assertTrue(replay.replay_complete)
        self.assertEqual(
            replay.lifecycle_evidence,
            support.residual_attachment_lifecycle_evidence,
        )
        self.assertEqual(
            len(replay.replay_terms),
            len(support.residual_attachment_lifecycle_evidence),
        )

    def test_residual_attachment_lifecycle_replay_rejects_stale_evidence(
        self,
    ) -> None:
        prepared = _prepare(cyclopropane_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _find_checked_branch_support(
            prepared,
            initial,
            lambda support: bool(
                support.residual_attachment_lifecycle_evidence
            ),
        )
        certificate = support.successor_state_certificate
        bad_certificate = replace(
            certificate,
            residual_attachment_lifecycle_replay_certificate=replace(
                (
                    certificate
                    .residual_attachment_lifecycle_replay_certificate
                ),
                lifecycle_evidence=(),
            ),
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "residual_attachment_lifecycle_replay_evidence_mismatch",
        ):
            (
                writer_state_delta_certificates
                .validate_writer_branch_successor_state_certificate(
                    bad_certificate
                )
            )

    def test_residual_attachment_lifecycle_replay_rejects_wrong_kind(
        self,
    ) -> None:
        prepared = _prepare(cyclopropane_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _find_checked_branch_support(
            prepared,
            initial,
            lambda support: (
                support.successor_state_certificate
                .residual_attachment_lifecycle_replay_certificate
                is not None
            ),
        )
        certificate = support.successor_state_certificate
        bad_certificate = replace(
            certificate,
            residual_attachment_lifecycle_replay_certificate=replace(
                (
                    certificate
                    .residual_attachment_lifecycle_replay_certificate
                ),
                kind=(
                    writer_state_delta_certificates
                    .WriterResidualAttachmentLifecycleReplayKind
                    .EVIDENCE_BOUND_INCOMPLETE
                ),
            ),
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "residual_attachment_lifecycle_replay_kind_mismatch",
        ):
            (
                writer_state_delta_certificates
                .validate_writer_branch_successor_state_certificate(
                    bad_certificate
                )
            )

    def test_residual_attachment_replay_rejects_successor_boundary_mismatch(
        self,
    ) -> None:
        prepared = _prepare(cyclopropane_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _find_checked_branch_support(
            prepared,
            initial,
            lambda support: bool(
                support.residual_attachment_lifecycle_evidence
            ),
        )
        evidence = support.residual_attachment_lifecycle_evidence[0]
        bad_successor_attachment = replace(
            evidence.successor_attachment,
            boundary=evidence.source_attachment.boundary,
        )
        bad_evidence = replace(
            evidence,
            successor_attachment=bad_successor_attachment,
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "residual_attachment_successor_boundary_mismatch",
        ):
            writer_branch_successor_state_certificate(
                **_successor_state_certificate_kwargs(
                    support,
                    residual_attachment_lifecycle_evidence=(bad_evidence,),
                )
            )

    def test_residual_attachment_replay_rejects_deficit_mismatch(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _find_checked_branch_support(
            prepared,
            initial,
            lambda support: bool(
                support.residual_attachment_lifecycle_evidence
            ),
        )
        evidence = support.residual_attachment_lifecycle_evidence[0]
        bad_evidence = replace(
            evidence,
            successor_closure_deficit=evidence.source_closure_deficit,
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "residual_attachment_deficit_delta_mismatch",
        ):
            writer_branch_successor_state_certificate(
                **_successor_state_certificate_kwargs(
                    support,
                    residual_attachment_lifecycle_evidence=(bad_evidence,),
                )
            )

    def test_checked_branch_certificate_rejects_stale_field_delta(self) -> None:
        prepared = _prepare(cco_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _find_checked_branch_support(
            prepared,
            initial,
            lambda support: (
                support.successor_state.active != support.source_state.active
            ),
        )
        bad_successor_certificate = replace(
            support.successor_state_certificate,
            active_delta=replace(
                support.successor_state_certificate.active_delta,
                successor_value=support.source_state.active,
                changed=False,
            ),
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "active_delta_",
        ):
            writer_checked_branch_support_certificate(
                **_checked_branch_certificate_kwargs(
                    support,
                    successor_state_certificate=bad_successor_certificate,
                )
            )

    def test_successor_state_certificate_constructor_calls_validator(self) -> None:
        prepared = _prepare(cco_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _checked_writer_frontier_branch_supports(
            prepared,
            initial,
            include_counts=False,
        ).supports[0]

        with patch.object(
            writer_state_delta_certificates,
            "validate_writer_branch_successor_state_certificate",
            side_effect=SouthStarError(
                SouthStarErrorKind.INTERNAL_INVARIANT,
                "sentinel",
            ),
        ):
            with self.assertRaisesRegex(SouthStarError, "sentinel"):
                writer_branch_successor_state_certificate(
                    **_successor_state_certificate_kwargs(support)
                )

    def test_checked_branch_certificate_rejects_stale_replay_certificate(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _find_checked_branch_support(
            prepared,
            initial,
            lambda support: (
                support.successor_state_certificate.graph_replay_certificate
                is not None
            ),
        )
        bad_successor_certificate = replace(
            support.successor_state_certificate,
            graph_replay_certificate=replace(
                support.successor_state_certificate.graph_replay_certificate,
                actual_successor_state=support.source_state,
            ),
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "graph_replay_successor_mismatch",
        ):
            writer_checked_branch_support_certificate(
                **_checked_branch_certificate_kwargs(
                    support,
                    successor_state_certificate=bad_successor_certificate,
                )
            )

    def test_successor_state_certificate_rejects_stale_graph_replay_projection(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _find_checked_branch_support(
            prepared,
            initial,
            lambda support: (
                support.successor_state_certificate.graph_replay_certificate
                is not None
            ),
        )
        certificate = support.successor_state_certificate
        bad_successor_certificate = replace(
            certificate,
            graph_replay_certificate=replace(
                certificate.graph_replay_certificate,
                expected_successor_projection=SimpleNamespace(
                    visited_atoms=certificate.source_state.visited_atoms,
                    written_bonds=certificate.successor_state.written_bonds,
                    active=certificate.successor_state.active,
                    branch_stack=certificate.successor_state.branch_stack,
                    component_cursor=certificate.successor_state.component_cursor,
                    obligations=certificate.successor_state.obligations,
                ),
            ),
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "graph_replay_projection_visited_atoms_mismatch",
        ):
            (
                writer_state_delta_certificates
                .validate_writer_branch_successor_state_certificate(
                    bad_successor_certificate,
                )
            )

    def test_graph_replay_rejects_missing_atom_event_for_visited_delta(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _find_checked_branch_support(
            prepared,
            initial,
            lambda support: bool(
                support.successor_state.visited_atoms
                - support.source_state.visited_atoms
            ),
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "graph_replay_visited_atoms_mismatch",
        ):
            writer_branch_successor_state_certificate(
                **_successor_state_certificate_kwargs(
                    support,
                    events=tuple(
                        event
                        for event in support.events
                        if not isinstance(event, WriterAtomEmitted)
                    ),
                )
            )

    def test_graph_replay_rejects_active_frame_mismatch(self) -> None:
        prepared = _prepare(cco_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _find_checked_branch_support(
            prepared,
            initial,
            lambda support: (
                support.successor_state.active != support.source_state.active
            ),
        )
        bad_successor = replace(
            support.successor_state,
            active=support.source_state.active,
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "graph_replay_active_mismatch",
        ):
            writer_branch_successor_state_certificate(
                **_successor_state_certificate_kwargs(
                    support,
                    successor_state=bad_successor,
                )
            )

    def test_graph_replay_rejects_branch_stack_mismatch(self) -> None:
        prepared = _prepare(cco_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _find_checked_branch_support(
            prepared,
            initial,
            lambda support: (
                support.successor_state.branch_stack
                != support.source_state.branch_stack
            ),
        )
        bad_successor = replace(
            support.successor_state,
            branch_stack=support.source_state.branch_stack,
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "graph_replay_branch_stack_mismatch",
        ):
            writer_branch_successor_state_certificate(
                **_successor_state_certificate_kwargs(
                    support,
                    successor_state=bad_successor,
                )
            )

    def test_graph_replay_does_not_treat_bond_text_event_as_written_bond_authority(
        self,
    ) -> None:
        prepared = _prepare(directional_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _find_checked_branch_support(
            prepared,
            initial,
            lambda support: any(
                isinstance(event, WriterBondEmitted) for event in support.events
            ),
        )
        bond_event = next(
            event
            for event in support.events
            if isinstance(event, WriterBondEmitted)
        )
        bad_successor = replace(
            support.successor_state,
            written_bonds=frozenset(
                (*support.successor_state.written_bonds, bond_event.bond)
            ),
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "graph_replay_written_bonds_mismatch",
        ):
            writer_branch_successor_state_certificate(
                **_successor_state_certificate_kwargs(
                    support,
                    successor_state=bad_successor,
                )
            )

    def test_obligation_replay_is_complete_for_pending_entry_creation(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _find_checked_branch_support(
            prepared,
            initial,
            lambda support: (
                support.source_state.obligations.pending_entry is None
                and support.successor_state.obligations.pending_entry is not None
            ),
        )
        obligation = (
            support.successor_state_certificate
            .graph_replay_certificate
            .obligation_replay_certificate
        )

        self.assertIsNotNone(obligation)
        self.assertIs(
            obligation.kind,
            (
                writer_state_delta_certificates
                .WriterGraphObligationReplayKind
                .PENDING_ENTRY_CREATED
            ),
        )
        self.assertTrue(obligation.replay_complete)
        self.assertEqual(
            obligation.expected_successor_obligations,
            obligation.actual_successor_obligations,
        )

    def test_obligation_replay_is_complete_for_pending_entry_discharge(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _find_checked_branch_support(
            prepared,
            initial,
            lambda support: (
                support.source_state.obligations.pending_entry is not None
                and support.successor_state.obligations.pending_entry is None
                and any(
                    isinstance(event, WriterAtomEmitted)
                    for event in support.events
                )
            ),
        )
        obligation = (
            support.successor_state_certificate
            .graph_replay_certificate
            .obligation_replay_certificate
        )

        self.assertIsNotNone(obligation)
        self.assertIs(
            obligation.kind,
            (
                writer_state_delta_certificates
                .WriterGraphObligationReplayKind
                .PENDING_ENTRY_DISCHARGED
            ),
        )
        self.assertTrue(obligation.replay_complete)
        self.assertEqual(
            obligation.expected_successor_obligations,
            obligation.actual_successor_obligations,
        )

    def test_obligation_replay_rejects_wrong_pending_child_event(self) -> None:
        prepared = _prepare(cco_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _find_checked_branch_support(
            prepared,
            initial,
            lambda support: (
                support.source_state.obligations.pending_entry is not None
                and any(
                    isinstance(event, WriterAtomEmitted)
                    for event in support.events
                )
            ),
        )
        event = next(
            event
            for event in support.events
            if isinstance(event, WriterAtomEmitted)
        )
        bad_event = replace(event, atom=AtomId(int(event.atom) + 100))

        with self.assertRaisesRegex(
            SouthStarError,
            "obligation_replay_pending_child_mismatch",
        ):
            writer_branch_successor_state_certificate(
                **_successor_state_certificate_kwargs(
                    support,
                    events=_replace_event_identity(
                        support.events,
                        event,
                        bad_event,
                    ),
                )
            )

    def test_directional_graph_obligation_replay_has_no_incomplete_cases(
        self,
    ) -> None:
        prepared = _prepare(directional_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        seen: set[WriterFrontierCursor] = set()
        pending = [initial]
        while pending and len(seen) < 1000:
            cursor = pending.pop(0)
            if cursor in seen:
                continue
            seen.add(cursor)
            batch = _checked_writer_frontier_branch_supports(
                prepared,
                cursor,
                include_counts=False,
            )
            for support in batch.supports:
                replay = (
                    support.successor_state_certificate
                    .graph_replay_certificate
                    .obligation_replay_certificate
                )
                self.assertIsNot(
                    replay.kind,
                    (
                        writer_state_delta_certificates
                        .WriterGraphObligationReplayKind
                        .EVIDENCE_BOUND_INCOMPLETE
                    ),
                )
                pending.append(support.successor_cursor)

    def test_obligation_replay_rejects_stale_expected_successor(self) -> None:
        prepared = _prepare(cco_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _find_checked_branch_support(
            prepared,
            initial,
            lambda support: (
                support.source_state.obligations
                != support.successor_state.obligations
                and (
                    support.successor_state_certificate
                    .graph_replay_certificate
                    .obligation_replay_certificate
                    .replay_complete
                )
            ),
        )
        certificate = support.successor_state_certificate
        graph = certificate.graph_replay_certificate
        obligation = graph.obligation_replay_certificate
        bad_certificate = replace(
            certificate,
            graph_replay_certificate=replace(
                graph,
                obligation_replay_certificate=replace(
                    obligation,
                    expected_successor_obligations=(
                        certificate.source_state.obligations
                    ),
                ),
            ),
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "obligation_replay_expected_successor_mismatch",
        ):
            (
                writer_state_delta_certificates
                .validate_writer_branch_successor_state_certificate(
                    bad_certificate,
                )
            )

    def test_obligation_replay_rejects_false_completion_kind(self) -> None:
        prepared = _prepare(cco_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _find_checked_branch_support(
            prepared,
            initial,
            lambda support: (
                support.successor_state_certificate.graph_replay_certificate
                is not None
                and (
                    support.successor_state_certificate
                    .graph_replay_certificate
                    .obligation_replay_certificate
                    is not None
                )
            ),
        )
        certificate = support.successor_state_certificate
        graph = certificate.graph_replay_certificate
        obligation = graph.obligation_replay_certificate
        bad_certificate = replace(
            certificate,
            graph_replay_certificate=replace(
                graph,
                obligation_replay_certificate=replace(
                    obligation,
                    kind=(
                        writer_state_delta_certificates
                        .WriterGraphObligationReplayKind
                        .EVIDENCE_BOUND_INCOMPLETE
                    ),
                    replay_complete=True,
                ),
            ),
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "obligation_replay_false_completion",
        ):
            (
                writer_state_delta_certificates
                .validate_writer_branch_successor_state_certificate(
                    bad_certificate,
                )
            )

    def test_obligation_replay_rejects_stale_event_view(self) -> None:
        prepared = _prepare(cco_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _find_checked_branch_support(
            prepared,
            initial,
            lambda support: (
                support.successor_state_certificate.graph_replay_certificate
                is not None
                and (
                    support.successor_state_certificate
                    .graph_replay_certificate
                    .obligation_replay_certificate
                    is not None
                )
            ),
        )
        certificate = support.successor_state_certificate
        graph = certificate.graph_replay_certificate
        obligation = graph.obligation_replay_certificate
        bad_certificate = replace(
            certificate,
            graph_replay_certificate=replace(
                graph,
                obligation_replay_certificate=replace(
                    obligation,
                    event_view=(
                        writer_state_delta_certificates.WriterEventDeltaView(
                            atom_events=(),
                            bond_events=(),
                            branch_open_events=(),
                            branch_close_events=(),
                            component_boundary_events=(),
                            local_order_events=(),
                            ring_label_allocated_events=(),
                            ring_label_released_events=(),
                            ring_endpoint_emitted_events=(),
                            ring_endpoint_paired_events=(),
                        )
                    ),
                ),
            ),
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "obligation_replay_event_view_mismatch",
        ):
            (
                writer_state_delta_certificates
                .validate_writer_branch_successor_state_certificate(
                    bad_certificate,
                )
            )

    def test_event_delta_view_ignores_class_name_spoofing(self) -> None:
        spoof = type(
            "WriterAtomEmitted",
            (),
            {"atom": AtomId(0), "text": "C"},
        )()

        view = writer_state_delta_certificates.writer_event_delta_view((spoof,))

        self.assertEqual(view.atom_events, ())

    def test_successor_state_certificate_rejects_nonmonotone_visited_atoms(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _find_checked_branch_support(
            prepared,
            initial,
            lambda support: bool(support.source_state.visited_atoms),
        )
        bad_successor = replace(
            support.successor_state,
            visited_atoms=frozenset(),
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "visited_atoms_not_monotone",
        ):
            writer_branch_successor_state_certificate(
                **_successor_state_certificate_kwargs(
                    support,
                    successor_state=bad_successor,
                )
            )

    def test_policy_delta_rejects_missing_event_payload(self) -> None:
        prepared = _prepare(cco_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _checked_writer_frontier_branch_supports(
            prepared,
            initial,
            include_counts=False,
        ).supports[0]
        atom = next(iter(support.successor_state.visited_atoms))
        bad_successor = replace(
            support.successor_state,
            policy_state=WriterPolicyStateKey(
                atom_text=((atom, "synthetic"),),
                bond_text=support.successor_state.policy_state.bond_text,
            ),
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "atom_policy_delta_lacks_event",
        ):
            writer_branch_successor_state_certificate(
                **_successor_state_certificate_kwargs(
                    support,
                    successor_state=bad_successor,
                    events=(),
                )
            )

    def test_successor_state_certificate_rejects_ring_delta_without_event(
        self,
    ) -> None:
        prepared = _prepare(cyclopropane_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _find_checked_branch_support(
            prepared,
            initial,
            lambda support: (
                support.successor_state.ring_state
                != support.source_state.ring_state
            ),
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "ring_delta_without_ring_lifecycle_event",
        ):
            writer_branch_successor_state_certificate(
                source_state=support.source_state,
                successor_state=support.successor_state,
                emitted_text=support.emitted_text,
                transition_kind=support.transition_kind,
                graph_action_surface=support.graph_action_surface,
                policy_family=support.policy_family,
                events=tuple(
                    event
                    for event in support.events
                    if not event.__class__.__name__.startswith("WriterRing")
                ),
                transition_evidence=support.evidence,
                graph_obligation_work_evidence=(
                    support.graph_obligation_work_evidence
                ),
                residual_work_evidence=support.residual_work_evidence,
                finite_relation_work_evidence=(
                    support.finite_relation_work_evidence
                ),
                closure_candidate_lifecycle_evidence=(
                    support.closure_candidate_lifecycle_evidence
                ),
                residual_attachment_lifecycle_evidence=(
                    support.residual_attachment_lifecycle_evidence
                ),
                stereo_lifecycle_evidence=support.stereo_lifecycle_evidence,
            )

    def test_ring_replay_rejects_wrong_endpoint_payload(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _find_checked_branch_support(
            prepared,
            initial,
            lambda support: any(
                isinstance(event, WriterRingEndpointEmitted)
                for event in support.events
            ),
        )
        event = next(
            event
            for event in support.events
            if isinstance(event, WriterRingEndpointEmitted)
        )
        bad_event = replace(event, endpoint_text="BAD")

        with self.assertRaisesRegex(
            SouthStarError,
            "ring_replay_added_open_mismatch",
        ):
            writer_branch_successor_state_certificate(
                **_successor_state_certificate_kwargs(
                    support,
                    events=_replace_event_identity(
                        support.events,
                        event,
                        bad_event,
                    ),
                )
            )

    def test_ring_pair_replay_requires_matching_open_endpoint(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _find_checked_branch_support(
            prepared,
            initial,
            lambda support: any(
                isinstance(event, WriterRingEndpointPaired)
                for event in support.events
            ),
        )
        event = next(
            event
            for event in support.events
            if isinstance(event, WriterRingEndpointPaired)
        )
        bad_event = replace(
            event,
            label=WriterClosureLabel(99, "%99"),
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "ring_replay_pair_lacks_matching_open_endpoint",
        ):
            writer_branch_successor_state_certificate(
                **_successor_state_certificate_kwargs(
                    support,
                    events=_replace_event_identity(
                        support.events,
                        event,
                        bad_event,
                    ),
                )
            )

    def test_ring_replay_rejects_wrong_removed_open_endpoint(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _find_checked_branch_support(
            prepared,
            initial,
            lambda support: any(
                isinstance(event, WriterRingEndpointPaired)
                for event in support.events
            ),
        )
        bad_successor = replace(
            support.successor_state,
            ring_state=replace(
                support.successor_state.ring_state,
                open_endpoints=support.source_state.ring_state.open_endpoints,
            ),
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "ring_replay_successor_state_mismatch",
        ):
            writer_branch_successor_state_certificate(
                **_successor_state_certificate_kwargs(
                    support,
                    successor_state=bad_successor,
                )
            )

    def test_ring_replay_rejects_label_state_mismatch(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _find_checked_branch_support(
            prepared,
            initial,
            lambda support: (
                support.successor_state.ring_state.label_state
                != support.source_state.ring_state.label_state
            ),
        )
        bad_successor = replace(
            support.successor_state,
            ring_state=replace(
                support.successor_state.ring_state,
                label_state=support.source_state.ring_state.label_state,
            ),
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "ring_replay_successor_state_mismatch",
        ):
            writer_branch_successor_state_certificate(
                **_successor_state_certificate_kwargs(
                    support,
                    successor_state=bad_successor,
                )
            )

    def test_ring_replay_rejects_stale_auxiliary_label_state(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _find_checked_branch_support(
            prepared,
            initial,
            lambda support: (
                support.successor_state_certificate.ring_replay_certificate
                is not None
                and (
                    support.successor_state.ring_state.label_state
                    != support.source_state.ring_state.label_state
                )
            ),
        )
        certificate = support.successor_state_certificate
        bad_certificate = replace(
            certificate,
            ring_replay_certificate=replace(
                certificate.ring_replay_certificate,
                replayed_label_state=certificate.source_state.ring_state.label_state,
            ),
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "ring_replay_label_state_mismatch",
        ):
            (
                writer_state_delta_certificates
                .validate_writer_branch_successor_state_certificate(
                    bad_certificate,
                )
            )

    def test_successor_state_certificate_rejects_stereo_delta_without_lifecycle(
        self,
    ) -> None:
        prepared = _prepare(tetrahedral_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _find_checked_branch_support(
            prepared,
            initial,
            lambda support: (
                support.successor_state.stereo_state
                != support.source_state.stereo_state
            ),
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "stereo_delta_without_lifecycle_evidence",
        ):
            writer_branch_successor_state_certificate(
                source_state=support.source_state,
                successor_state=support.successor_state,
                emitted_text=support.emitted_text,
                transition_kind=support.transition_kind,
                graph_action_surface=support.graph_action_surface,
                policy_family=support.policy_family,
                events=support.events,
                transition_evidence=support.evidence,
                graph_obligation_work_evidence=(
                    support.graph_obligation_work_evidence
                ),
                residual_work_evidence=support.residual_work_evidence,
                finite_relation_work_evidence=(
                    support.finite_relation_work_evidence
                ),
                closure_candidate_lifecycle_evidence=(
                    support.closure_candidate_lifecycle_evidence
                ),
                residual_attachment_lifecycle_evidence=(
                    support.residual_attachment_lifecycle_evidence
                ),
                stereo_lifecycle_evidence=(),
            )

    def test_stereo_replay_has_complete_lifecycle_chain_when_evidence_matches(
        self,
    ) -> None:
        prepared = _prepare(tetrahedral_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _find_checked_branch_support(
            prepared,
            initial,
            lambda support: (
                support.successor_state_certificate.stereo_replay_certificate
                is not None
                and (
                    support.successor_state_certificate
                    .stereo_replay_certificate
                    .replay_complete
                )
            ),
        )
        stereo = support.successor_state_certificate.stereo_replay_certificate

        self.assertIs(
            stereo.kind,
            (
                writer_state_delta_certificates.WriterStereoReplayKind
                .LIFECYCLE_CHAIN_COMPLETE
            ),
        )
        self.assertIsNotNone(stereo.lifecycle_chain_certificate)
        self.assertTrue(stereo.lifecycle_chain_certificate.replay_complete)
        self.assertEqual(
            stereo.expected_successor_stereo_state,
            stereo.actual_successor_stereo_state,
        )

    def test_stereo_replay_rejects_missing_residual_work_item(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _find_checked_branch_support(
            prepared,
            initial,
            lambda support: any(
                getattr(evidence, "residual_work_evidence", ())
                for evidence in support.stereo_lifecycle_evidence
            ),
        )
        lifecycle_work = tuple(
            item
            for evidence in support.stereo_lifecycle_evidence
            for item in getattr(evidence, "residual_work_evidence", ())
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "stereo_delta_work_evidence_missing|stereo_replay_work_evidence_missing",
        ):
            writer_branch_successor_state_certificate(
                **_successor_state_certificate_kwargs(
                    support,
                    residual_work_evidence=tuple(
                        item
                        for item in support.residual_work_evidence
                        if item is not lifecycle_work[0]
                    ),
                )
            )

    def test_stereo_replay_rejects_false_completion(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _find_checked_branch_support(
            prepared,
            initial,
            lambda support: (
                support.successor_state_certificate.stereo_replay_certificate
                is not None
            ),
        )
        certificate = support.successor_state_certificate
        bad_certificate = replace(
            certificate,
            stereo_replay_certificate=replace(
                certificate.stereo_replay_certificate,
                kind=(
                    writer_state_delta_certificates.WriterStereoReplayKind
                    .EVIDENCE_BOUND_INCOMPLETE
                ),
                replay_complete=True,
            ),
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "stereo_replay_false_completion",
        ):
            (
                writer_state_delta_certificates
                .validate_writer_branch_successor_state_certificate(
                    bad_certificate,
                )
            )

    def test_stereo_lifecycle_chain_rejects_successor_snapshot_mismatch(
        self,
    ) -> None:
        prepared = _prepare(tetrahedral_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _find_checked_branch_support(
            prepared,
            initial,
            lambda support: (
                support.successor_state_certificate.stereo_replay_certificate
                is not None
                and (
                    support.successor_state_certificate
                    .stereo_replay_certificate
                    .lifecycle_chain_certificate
                    is not None
                )
            ),
        )
        certificate = support.successor_state_certificate
        chain = certificate.stereo_replay_certificate.lifecycle_chain_certificate
        bad_certificate = replace(
            certificate,
            stereo_replay_certificate=replace(
                certificate.stereo_replay_certificate,
                lifecycle_chain_certificate=replace(
                    chain,
                    actual_successor_residual_snapshot=(
                        certificate.source_state.stereo_state.residual_snapshot
                    ),
                ),
            ),
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "stereo_lifecycle_chain_successor_mismatch",
        ):
            (
                writer_state_delta_certificates
                .validate_writer_branch_successor_state_certificate(
                    bad_certificate,
                )
            )

    def test_runtime_branch_transitions_expose_successor_state_certificate(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )

        transitions = writer_runtime_branch_transitions(
            prepared=prepared,
            state=state,
            include_counts=False,
        )

        self.assertTrue(transitions.transitions)
        for transition in transitions.transitions:
            certificate = transition.successor_state_certificate
            self.assertIsNotNone(certificate)
            self.assertEqual(certificate.source_state, transition.source_state)
            self.assertEqual(
                certificate.successor_state,
                transition.successor_state,
            )
            self.assertTrue(
                certificate.graph_delta_certificate is not None
                or certificate.policy_delta_certificate is not None
                or certificate.ring_delta_certificate is not None
                or certificate.stereo_delta_certificate is not None
            )
            if certificate.graph_delta_certificate is not None:
                self.assertIsNotNone(certificate.graph_replay_certificate)
            if certificate.policy_delta_certificate is not None:
                self.assertIsNotNone(certificate.policy_replay_certificate)
            if certificate.ring_delta_certificate is not None:
                self.assertIsNotNone(certificate.ring_replay_certificate)
            if certificate.stereo_delta_certificate is not None:
                self.assertIsNotNone(certificate.stereo_replay_certificate)

    def test_projection_certificate_exposes_branch_successor_proofs(self) -> None:
        prepared = _prepare(cco_facts())
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        product = _checked_writer_frontier_product(
            prepared,
            cursor,
            include_counts=False,
            include_frontier_certificate=False,
            include_count_certificate=False,
        )

        self.assertIsNotNone(product.projection_certificate)
        for projection in (
            product.projection_certificate.text_choice_projection_certificates
        ):
            for branch_certificate in projection.branch_certificates:
                successor = branch_certificate.successor_state_certificate
                self.assertIsNotNone(successor)
                self.assertEqual(
                    successor.source_state,
                    branch_certificate.source_state,
                )
                self.assertEqual(
                    successor.successor_state,
                    branch_certificate.successor_state,
                )

    def test_text_projection_immediate_multiplicity_matches_branch_certificate_weights(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        product = _checked_writer_frontier_product(
            prepared,
            cursor,
            include_counts=False,
            include_frontier_certificate=False,
            include_count_certificate=False,
        )

        for projection in product.text_choice_projection_certificates:
            self.assertEqual(
                projection.immediate_multiplicity,
                sum(
                    certificate.parent_weight
                    for certificate in projection.branch_certificates
                ),
            )

    def test_text_projection_successor_cursor_is_branch_certificate_derived(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        product = _checked_writer_frontier_product(
            prepared,
            cursor,
            include_counts=False,
            include_frontier_certificate=False,
            include_count_certificate=False,
        )

        for projection in product.text_choice_projection_certificates:
            weighted = Counter()
            for certificate in projection.branch_certificates:
                weighted[certificate.successor_state] += (
                    certificate.parent_weight
                )
            expected = projection.successor_cursor.__class__(
                weighted_states=tuple(weighted.items())
            )
            self.assertEqual(projection.successor_cursor, expected)

    def test_text_projection_rejects_branch_certificate_parent_weight_mismatch(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            cursor,
            include_counts=False,
        )
        support = batch.supports[0]
        bad_certificate = replace(
            support.checked_branch_certificate,
            parent_weight=support.parent_weight + 1,
        )
        bad_support = replace(
            support,
            checked_branch_certificate=bad_certificate,
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "branch_certificate_parent_weight_mismatch",
        ):
            writer_text_choice_projection_certificates(
                source_cursor=cursor,
                choices=batch.choices,
                branch_supports=(bad_support, *batch.supports[1:]),
            )

    def test_text_projection_rejects_branch_certificate_successor_mismatch(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            cursor,
            include_counts=False,
        )
        support = batch.supports[0]
        bad_certificate = replace(
            support.checked_branch_certificate,
            successor_state=support.source_state,
        )
        bad_support = replace(
            support,
            checked_branch_certificate=bad_certificate,
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "branch_certificate_successor_mismatch",
        ):
            writer_text_choice_projection_certificates(
                source_cursor=cursor,
                choices=batch.choices,
                branch_supports=(bad_support, *batch.supports[1:]),
            )

    def test_terminal_projection_multiplicity_matches_terminal_certificate_weights(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        product = _first_terminal_frontier_product(prepared)
        terminal_projection = product.terminal_projection_certificate
        self.assertIsNotNone(terminal_projection)

        self.assertEqual(
            terminal_projection.multiplicity,
            sum(
                certificate.parent_weight
                for certificate in terminal_projection.terminal_certificates
            ),
        )

    def test_terminal_projection_finalized_cursor_is_certificate_derived(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        product = _first_terminal_frontier_product(prepared)
        terminal_projection = product.terminal_projection_certificate
        self.assertIsNotNone(terminal_projection)

        weighted = Counter()
        for certificate in terminal_projection.terminal_certificates:
            weighted[certificate.finalized_state] += certificate.parent_weight
        expected = terminal_projection.finalized_cursor.__class__(
            weighted_states=tuple(weighted.items())
        )
        self.assertEqual(terminal_projection.finalized_cursor, expected)

    def test_terminal_projection_rejects_certificate_parent_weight_mismatch(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        product = _first_terminal_frontier_product(prepared)
        support = product.terminal_supports[0]
        bad_certificate = replace(
            support.checked_terminal_certificate,
            parent_weight=support.parent_weight + 1,
        )
        bad_support = replace(
            support,
            checked_terminal_certificate=bad_certificate,
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "terminal_certificate_parent_weight_mismatch",
        ):
            writer_terminal_projection_certificate(
                source_cursor=product.cursor,
                terminal=product.choices.terminal,
                terminal_supports=(bad_support, *product.terminal_supports[1:]),
            )

    def test_terminal_projection_rejects_certificate_finalized_mismatch(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        product = _first_terminal_frontier_product(prepared)
        support = product.terminal_supports[0]
        bad_certificate = replace(
            support.checked_terminal_certificate,
            finalized_state=support.source_state,
        )
        bad_support = replace(
            support,
            checked_terminal_certificate=bad_certificate,
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "terminal_certificate_finalized_mismatch",
        ):
            writer_terminal_projection_certificate(
                source_cursor=product.cursor,
                terminal=product.choices.terminal,
                terminal_supports=(bad_support, *product.terminal_supports[1:]),
            )

    def test_frontier_projection_rejects_terminal_certificate_parent_weight_mismatch(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        product = _first_terminal_frontier_product(prepared)
        support = product.terminal_supports[0]
        bad_certificate = replace(
            support.checked_terminal_certificate,
            parent_weight=support.parent_weight + 1,
        )
        bad_support = replace(
            support,
            checked_terminal_certificate=bad_certificate,
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "terminal_certificate_parent_weight_mismatch",
        ):
            writer_frontier_projection_certificate(
                cursor=product.cursor,
                choices=product.choices,
                branch_supports=product.branch_supports,
                terminal_supports=(bad_support, *product.terminal_supports[1:]),
                text_choice_projection_certificates=(
                    product.text_choice_projection_certificates
                ),
                terminal_projection_certificate=(
                    product.terminal_projection_certificate
                ),
            )

    def test_frontier_projection_rejects_branch_certificate_ordinal_mismatch(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        product = _checked_writer_frontier_product(
            prepared,
            cursor,
            include_counts=False,
            include_frontier_certificate=False,
            include_count_certificate=False,
        )
        support = product.branch_supports[0]
        bad_certificate = replace(
            support.checked_branch_certificate,
            branch_ordinal=support.branch_ordinal + 100,
        )
        bad_support = replace(
            support,
            checked_branch_certificate=bad_certificate,
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "branch_certificate_ordinal_mismatch",
        ):
            writer_frontier_projection_certificate(
                cursor=product.cursor,
                choices=product.choices,
                branch_supports=(bad_support, *product.branch_supports[1:]),
                terminal_supports=product.terminal_supports,
                text_choice_projection_certificates=(
                    product.text_choice_projection_certificates
                ),
                terminal_projection_certificate=(
                    product.terminal_projection_certificate
                ),
            )

    def test_checked_branch_supports_have_capability_coverage(self) -> None:
        prepared = _prepare(cco_facts())
        initial = initial_writer_frontier_cursor(
            prepared,
            _writer_options(),
        )
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            initial,
            include_counts=False,
        )

        for support in batch.supports:
            coverage = support.capability_coverage_certificate
            self.assertIsNotNone(coverage)
            self.assertIsNotNone(coverage.execution_capabilities)
            self.assertEqual(
                coverage.execution_capabilities,
                support.execution_capabilities,
            )
            self.assertEqual(
                coverage.covered_capabilities,
                support.execution_capabilities,
            )

            for capability in support.execution_capabilities:
                self.assertIn(
                    capability,
                    tuple(c.capability for c in coverage.capability_certificates),
                )

    def test_checked_branch_supporter_coverage_requires_matching_capability_certificates(
        self,
    ) -> None:
        prepared = _prepare(directional_facts())
        initial = initial_writer_frontier_cursor(
            prepared,
            _writer_options(),
        )
        support = _find_checked_branch_support(
            prepared,
            initial,
            lambda support: bool(support.execution_capabilities),
        )

        # Remove one capability without removing its corresponding coverage
        # certificate to verify the aggregate certificate rejects the mismatch.
        reduced_capabilities = set(support.execution_capabilities)
        removed_capability = next(iter(reduced_capabilities))
        reduced_capabilities.remove(removed_capability)

        with self.assertRaisesRegex(
            SouthStarError,
            "capability_coverage_execution_mismatch",
        ):
            writer_checked_branch_support_certificate(
                source_state=support.source_state,
                successor_state=support.successor_state,
                emitted_text=support.emitted_text,
                parent_weight=support.parent_weight,
                branch_ordinal=support.branch_ordinal,
                transition_kind=support.transition_kind,
                graph_action_surface=support.graph_action_surface,
                policy_family=support.policy_family,
                events=support.events,
                transition_evidence=support.evidence,
                execution_capabilities=frozenset(
                    capability for capability in reduced_capabilities
                ),
                graph_obligation_work_evidence=(
                    support.graph_obligation_work_evidence
                ),
                residual_work_evidence=support.residual_work_evidence,
                finite_relation_work_evidence=(
                    support.finite_relation_work_evidence
                ),
                closure_candidate_resolution_evidence=(
                    support.closure_candidate_resolution_evidence
                ),
                closure_candidate_lifecycle_evidence=(
                    support.closure_candidate_lifecycle_evidence
                ),
                closure_candidate_branch_certificates=(
                    support.closure_candidate_branch_certificates
                ),
                residual_attachment_lifecycle_evidence=(
                    support.residual_attachment_lifecycle_evidence
                ),
                residual_attachment_branch_certificates=(
                    support.residual_attachment_branch_certificates
                ),
                stereo_lifecycle_evidence=support.stereo_lifecycle_evidence,
                stereo_branch_certificates=support.stereo_branch_certificates,
                residual_attachment_policy_evidence=(
                    support.residual_attachment_policy_evidence
                ),
                capability_coverage_certificate=(
                    support.capability_coverage_certificate
                ),
                successor_state_certificate=(
                    support.successor_state_certificate
                ),
            )

        # Remove the graph-capability coverage while leaving execution unchanged.
        with self.assertRaisesRegex(
            SouthStarError,
            "capability_coverage_incomplete",
        ):
            writer_checked_branch_support_certificate(
                source_state=support.source_state,
                successor_state=support.successor_state,
                emitted_text=support.emitted_text,
                parent_weight=support.parent_weight,
                branch_ordinal=support.branch_ordinal,
                transition_kind=support.transition_kind,
                graph_action_surface=support.graph_action_surface,
                policy_family=support.policy_family,
                events=support.events,
                transition_evidence=support.evidence,
                execution_capabilities=support.execution_capabilities,
                graph_obligation_work_evidence=(
                    support.graph_obligation_work_evidence
                ),
                residual_work_evidence=support.residual_work_evidence,
                finite_relation_work_evidence=(
                    support.finite_relation_work_evidence
                ),
                closure_candidate_resolution_evidence=(
                    support.closure_candidate_resolution_evidence
                ),
                closure_candidate_lifecycle_evidence=(
                    support.closure_candidate_lifecycle_evidence
                ),
                closure_candidate_branch_certificates=(
                    support.closure_candidate_branch_certificates
                ),
                residual_attachment_lifecycle_evidence=(
                    support.residual_attachment_lifecycle_evidence
                ),
                residual_attachment_branch_certificates=(
                    support.residual_attachment_branch_certificates
                ),
                stereo_lifecycle_evidence=support.stereo_lifecycle_evidence,
                stereo_branch_certificates=support.stereo_branch_certificates,
                residual_attachment_policy_evidence=(
                    support.residual_attachment_policy_evidence
                ),
                capability_coverage_certificate=(
                    WriterCapabilityCoverageCertificate(
                        execution_capabilities=support.execution_capabilities,
                        capability_certificates=(),
                    )
                ),
                successor_state_certificate=(
                    support.successor_state_certificate
                ),
            )

    def test_checked_text_choices_have_projection_certificates(self) -> None:
        for facts in (
            cco_facts(),
            cyclopropane_facts(),
            tetrahedral_facts(),
            directional_facts(),
        ):
            prepared = _prepare(facts)
            initial = initial_writer_frontier_cursor(
                prepared,
                _writer_options(),
            )
            batch = _checked_writer_frontier_branch_supports(
                prepared,
                initial,
                include_counts=True,
            )

            self.assertEqual(
                len(batch.text_choice_projection_certificates),
                len(batch.choices.choices),
            )
            for certificate in batch.text_choice_projection_certificates:
                matching = tuple(
                    support
                    for support in batch.supports
                    if support.emitted_text == certificate.emitted_text
                )
                expected = WriterFrontierCursor(
                    weighted_states=tuple(
                        (support.successor_state, support.parent_weight)
                        for support in matching
                    )
                )

                self.assertEqual(
                    certificate.choice.emitted_text,
                    certificate.emitted_text,
                )
                self.assertEqual(certificate.successor_cursor, expected)
                self.assertEqual(certificate.choice.successor, expected)
                self.assertEqual(
                    certificate.immediate_multiplicity,
                    sum(support.parent_weight for support in matching),
                )
                self.assertEqual(
                    certificate.branch_certificates,
                    tuple(
                        support.checked_branch_certificate
                        for support in matching
                    ),
                )

    def test_open_ring_endpoint_owned_residual_resolution_reaches_branch_support(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        source_key = initial.weighted_states[0][0]
        key = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=source_key.active.atom,
            attachment_id=7,
        )
        support = writer_frontier_module._WriterFrontierNextTokenSupport(
            state_key=source_key,
            parent_weight=1,
            schedule_support=SimpleNamespace(
                emitted_text="C",
                graph_action_surface=SimpleNamespace(
                    residual_attachment_policy_key=key,
                ),
                policy_family=(
                    writer_transitions
                    ._WriterGraphPolicyActionFamily
                    .CYCLIC_TREE_ENTRY
                ),
                execution_capabilities=frozenset(),
                residual_work_evidence=(),
                finite_relation_work_evidence=(),
                transition=SimpleNamespace(
                    kind=(
                        writer_transitions
                        ._WriterGraphPolicyActionFamily
                        .CYCLIC_TREE_ENTRY
                    ),
                    events=(),
                    evidence=None,
                ),
            ),
            successor_key=source_key,
        )
        policy_group = writer_transitions._WriterResidualAttachmentPolicyGroup(
            key=key,
            surfaces=(
                writer_transitions._WriterScheduledGraphActionSurface(
                    kind=(
                        writer_transitions
                        ._WriterScheduledActionKind
                        .OPEN_CLOSURE_ENDPOINT
                    ),
                    active_atom=key.active_atom,
                    attachment_id=key.attachment_id,
                    attachment_action_kind=(
                        WriterResidualAttachmentActionKind
                        .CLOSURE_OPEN_READY
                    ),
                    owner_kind=WriterBoundaryOwnerKind.OPEN_RING_ENDPOINT,
                ),
                writer_transitions._WriterScheduledGraphActionSurface(
                    kind=(
                        writer_transitions
                        ._WriterScheduledActionKind
                        .ENTER_INLINE_CHILD
                    ),
                    active_atom=key.active_atom,
                    attachment_id=key.attachment_id,
                    attachment_action_kind=(
                        WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
                    ),
                    owner_kind=WriterBoundaryOwnerKind.OPEN_RING_ENDPOINT,
                ),
            ),
        )
        evidence_group = (
            writer_frontier_module
            ._WriterFrontierResidualAttachmentEvidenceGroup(
                key=key,
                resolved_policy_groups=(policy_group,),
                support_dead_closure_open_vs_cyclic_tree_entry_policy_groups=(
                    policy_group,
                ),
                selected_support_groups=(
                    writer_frontier_module
                    ._WriterFrontierResidualAttachmentSupportGroup(
                        key=key,
                        supports=(support,),
                    ),
                ),
            )
        )
        branch_support = (
            writer_frontier_module
            ._writer_frontier_branch_support_from_next_token_support(
                prepared=prepared,
                branch_ordinal=0,
                support=support,
                schedule_outcome=SimpleNamespace(
                    residual_attachment_evidence_groups=(evidence_group,),
                ),
            )
        )

        self.assertIn(
            (
                _WriterExecutionCapabilityKind
                .OPEN_RING_ENDPOINT_RESIDUAL_ATTACHMENT_RESOLUTION
            ),
            branch_support.execution_capabilities,
        )
        self.assertEqual(
            branch_support.residual_attachment_policy_evidence,
            (evidence_group,),
        )
        self.assertTrue(
            any(
                (
                    group.has_dead_closure_open_resolved_cyclic_tree_entry_support
                    and group.has_open_ring_endpoint_owner_scope_evidence
                )
                for group in branch_support.residual_attachment_policy_evidence
            )
        )
        self.assertGreater(
            _count_checked_writer_frontier_branch_completions(
                prepared,
                branch_support.successor_cursor,
            ),
            0,
        )

    def test_graph_evidence_classifies_live_branch_return_closure_candidate(
        self,
    ) -> None:
        prepared = _prepare(cyclopropane_facts())
        key = writer_state_key(
            _branch_return_closure_candidate_state(live_partner=True)
        )
        context = build_writer_graph_obligation_context(prepared, key)

        evidence = writer_graph_obligation_work_evidence(
            operation="test",
            prepared=prepared,
            key=key,
            context=context,
        )

        self.assertEqual(evidence.closure_candidate_count, 1)
        self.assertEqual(
            evidence.live_branch_return_closure_candidate_count,
            1,
        )
        self.assertEqual(
            evidence.deferred_branch_return_closure_candidate_count,
            0,
        )
        self.assertEqual(
            evidence.deferred_control_live_closure_candidate_count,
            0,
        )
        self.assertEqual(evidence.unsupported_closure_candidate_count, 0)
        self.assertIsNone(
            writer_graph_obligation_work_envelope_violation(evidence)
        )

    def test_graph_evidence_classifies_frozen_closure_candidate(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        key = writer_state_key(
            _branch_return_closure_candidate_state(live_partner=False)
        )
        context = build_writer_graph_obligation_context(prepared, key)

        evidence = writer_graph_obligation_work_evidence(
            operation="test",
            prepared=prepared,
            key=key,
            context=context,
        )

        self.assertEqual(evidence.closure_candidate_count, 1)
        self.assertEqual(
            evidence.live_branch_return_closure_candidate_count,
            0,
        )
        self.assertEqual(
            evidence.deferred_branch_return_closure_candidate_count,
            0,
        )
        self.assertEqual(
            evidence.deferred_control_live_closure_candidate_count,
            0,
        )
        self.assertEqual(evidence.unsupported_closure_candidate_count, 1)

        violation = writer_graph_obligation_work_envelope_violation(evidence)
        self.assertIsNotNone(violation)
        assert violation is not None
        self.assertEqual(
            violation.metric,
            "unsupported_closure_candidate_count",
        )

    def test_live_branch_return_closure_candidate_reaches_branch_support(
        self,
    ) -> None:
        prepared = _prepare(cyclopropane_facts())
        cursor = WriterFrontierCursor(
            weighted_states=(
                (
                    writer_state_key(
                        _branch_return_closure_candidate_state(
                            live_partner=True,
                        )
                    ),
                    1,
                ),
            )
        )

        batch = _checked_writer_frontier_branch_supports(
            prepared,
            cursor,
            include_counts=False,
        )
        support = next(
            support
            for support in batch.supports
            if (
                support.transition_kind
                is writer_transitions.WriterTransitionKind.OPEN_CLOSURE_ENDPOINT
            )
        )

        self.assertIn(
            (
                _WriterExecutionCapabilityKind
                .LIVE_BRANCH_RETURN_CLOSURE_CANDIDATE_OPEN
            ),
            support.execution_capabilities,
        )
        context = build_writer_graph_obligation_context(
            prepared,
            support.source_state,
        )
        evidence = writer_graph_obligation_work_evidence(
            operation="test",
            prepared=prepared,
            key=support.source_state,
            context=context,
        )
        self.assertEqual(
            evidence.live_branch_return_closure_candidate_count,
            1,
        )
        self.assertEqual(
            evidence.deferred_branch_return_closure_candidate_count,
            0,
        )
        self.assertEqual(
            evidence.deferred_control_live_closure_candidate_count,
            0,
        )
        self.assertEqual(evidence.unsupported_closure_candidate_count, 0)
        self.assertIsInstance(support.events[0], WriterRingLabelAllocated)
        self.assertIsInstance(support.events[1], WriterRingEndpointEmitted)
        self.assertEqual(
            support.successor_cursor.weighted_states,
            ((support.successor_state, 1),),
        )
        self.assertEqual(len(support.closure_candidate_resolution_evidence), 1)
        resolution = support.closure_candidate_resolution_evidence[0]
        self.assertIs(
            resolution.resolution_kind,
            WriterClosureCandidateResolutionKind.LIVE_BRANCH_RETURN,
        )
        self.assertEqual(resolution.bond, support.graph_action_surface.bond)
        self.assertEqual(
            resolution.first_atom,
            support.graph_action_surface.boundary_atom,
        )
        self.assertEqual(
            resolution.second_atom,
            support.graph_action_surface.partner_atom,
        )
        self.assertGreater(
            _count_checked_writer_frontier_branch_completions(
                prepared,
                support.successor_cursor,
            ),
            0,
        )

    def test_live_candidate_lifecycle_opens_exact_bond(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        cursor = WriterFrontierCursor(
            weighted_states=(
                (
                    writer_state_key(
                        _branch_return_closure_candidate_state(
                            live_partner=True,
                        )
                    ),
                    1,
                ),
            )
        )
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            cursor,
            include_counts=False,
        )
        support = next(
            support
            for support in batch.supports
            if (
                support.transition_kind
                is writer_transitions.WriterTransitionKind.OPEN_CLOSURE_ENDPOINT
            )
        )

        validate_writer_closure_candidate_lifecycle_transition(
            prepared=prepared,
            source_state=support.source_state,
            successor_state=support.successor_state,
            transition_kind=support.transition_kind,
            graph_action_surface=support.graph_action_surface,
        )
        evidence = support.closure_candidate_lifecycle_evidence
        self.assertEqual(len(evidence), 1)
        self.assertIs(
            evidence[0].outcome_kind,
            WriterClosureCandidateLifecycleOutcomeKind.OPENED,
        )
        self.assertEqual(evidence[0].bond, support.graph_action_surface.bond)
        self.assertIs(
            evidence[0].successor_obligation_kind,
            WriterEdgeObligationKind.OPEN_CLOSURE_ENDPOINT,
        )
        certificate = _single_closure_candidate_branch_certificate(
            support,
            WriterClosureCandidateBranchCertificateKind.LIVE_BRANCH_RETURN_OPENED,
        )
        self.assertEqual(certificate.bond, support.graph_action_surface.bond)
        self.assertIs(
            certificate.lifecycle_evidence.outcome_kind,
            WriterClosureCandidateLifecycleOutcomeKind.OPENED,
        )
        self.assertIsInstance(support.events[0], WriterRingLabelAllocated)
        self.assertIsInstance(support.events[1], WriterRingEndpointEmitted)

    def test_frozen_closure_candidate_still_blocks(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        cursor = WriterFrontierCursor(
            weighted_states=(
                (
                    writer_state_key(
                        _branch_return_closure_candidate_state(
                            live_partner=False,
                        )
                    ),
                    1,
                ),
            )
        )

        with self.assertRaises(SouthStarError) as caught:
            _checked_writer_frontier_branch_supports(
                prepared,
                cursor,
                include_counts=False,
            )

        self.assertIs(caught.exception.kind, SouthStarErrorKind.UNSUPPORTED_POLICY)
        self.assertIn("closure-candidate", str(caught.exception))

    def test_deferred_branch_return_closure_candidate_allows_branch_close(
        self,
    ) -> None:
        prepared = _prepare(cyclopropane_facts())
        cursor = WriterFrontierCursor(
            weighted_states=(
                (
                    writer_state_key(
                        _deferred_branch_return_closure_candidate_state(
                            frozen_endpoint=False,
                        )
                    ),
                    1,
                ),
            )
        )
        key = cursor.weighted_states[0][0]
        context = build_writer_graph_obligation_context(prepared, key)
        evidence = writer_graph_obligation_work_evidence(
            operation="test",
            prepared=prepared,
            key=key,
            context=context,
        )

        self.assertEqual(evidence.closure_candidate_count, 1)
        self.assertEqual(
            evidence.live_branch_return_closure_candidate_count,
            0,
        )
        self.assertEqual(
            evidence.deferred_branch_return_closure_candidate_count,
            1,
        )
        self.assertEqual(
            evidence.deferred_control_live_closure_candidate_count,
            0,
        )
        self.assertEqual(evidence.unsupported_closure_candidate_count, 0)
        self.assertIsNone(
            writer_graph_obligation_work_envelope_violation(evidence)
        )

        batch = _checked_writer_frontier_branch_supports(
            prepared,
            cursor,
            include_counts=False,
        )
        support = next(
            support
            for support in batch.supports
            if (
                support.transition_kind
                is writer_transitions.WriterTransitionKind.CLOSE_BRANCH
            )
        )

        self.assertIn(
            (
                _WriterExecutionCapabilityKind
                .DEFERRED_BRANCH_RETURN_CLOSURE_CANDIDATE
            ),
            support.execution_capabilities,
        )
        self.assertEqual(
            support.graph_obligation_work_evidence[0]
            .deferred_branch_return_closure_candidate_count,
            1,
        )

        next_batch = _checked_writer_frontier_branch_supports(
            prepared,
            support.successor_cursor,
            include_counts=False,
        )
        live_support = next(
            support
            for support in next_batch.supports
            if (
                support.transition_kind
                is writer_transitions.WriterTransitionKind.OPEN_CLOSURE_ENDPOINT
            )
        )
        self.assertIn(
            (
                _WriterExecutionCapabilityKind
                .LIVE_BRANCH_RETURN_CLOSURE_CANDIDATE_OPEN
            ),
            live_support.execution_capabilities,
        )

    def test_multi_live_closure_candidates_preserve_branch_identity(
        self,
    ) -> None:
        prepared = _prepare(_two_live_closure_candidate_facts())
        cursor = WriterFrontierCursor(
            weighted_states=(
                (
                    writer_state_key(
                        _multi_live_branch_return_closure_candidate_state()
                    ),
                    1,
                ),
            )
        )
        key = cursor.weighted_states[0][0]
        context = build_writer_graph_obligation_context(prepared, key)
        evidence = writer_graph_obligation_work_evidence(
            operation="writer graph obligation context",
            prepared=prepared,
            key=key,
            context=context,
        )

        self.assertEqual(evidence.closure_candidate_count, 2)
        self.assertEqual(
            evidence.live_branch_return_closure_candidate_count,
            2,
        )
        self.assertEqual(
            evidence.deferred_branch_return_closure_candidate_count,
            0,
        )
        self.assertEqual(
            evidence.deferred_control_live_closure_candidate_count,
            0,
        )
        self.assertEqual(evidence.unsupported_closure_candidate_count, 0)
        self.assertIsNone(
            writer_graph_obligation_work_envelope_violation(evidence)
        )

        batch = _checked_writer_frontier_branch_supports(
            prepared,
            cursor,
            include_counts=False,
        )
        open_supports = tuple(
            support
            for support in batch.supports
            if (
                support.transition_kind
                is writer_transitions.WriterTransitionKind.OPEN_CLOSURE_ENDPOINT
                and (
                    _WriterExecutionCapabilityKind
                    .LIVE_BRANCH_RETURN_CLOSURE_CANDIDATE_OPEN
                )
                in support.execution_capabilities
            )
        )

        self.assertGreaterEqual(len(open_supports), 2)
        self.assertLess(
            len({support.emitted_text for support in open_supports}),
            len(open_supports),
        )
        self.assertEqual(
            len({support.successor_state for support in open_supports}),
            len(open_supports),
        )
        self.assertEqual(
            len(
                {
                    support.closure_candidate_resolution_evidence[0].bond
                    for support in open_supports
                }
            ),
            len(open_supports),
        )
        self.assertEqual(
            len(
                {
                    certificate.bond
                    for support in open_supports
                    for certificate in support.closure_candidate_branch_certificates
                    if certificate.kind
                    is (
                        WriterClosureCandidateBranchCertificateKind
                        .LIVE_BRANCH_RETURN_OPENED
                    )
                }
            ),
            len(open_supports),
        )
        self.assertEqual(
            len(
                {
                    support.checked_branch_certificate.successor_state
                    for support in open_supports
                }
            ),
            len(open_supports),
        )
        choice_certificate = next(
            certificate
            for certificate in batch.text_choice_projection_certificates
            if certificate.emitted_text == open_supports[0].emitted_text
        )
        same_text_supports = tuple(
            support
            for support in open_supports
            if support.emitted_text == choice_certificate.emitted_text
        )
        self.assertGreater(len(choice_certificate.branch_certificates), 1)
        self.assertEqual(
            choice_certificate.branch_certificates,
            tuple(
                support.checked_branch_certificate
                for support in same_text_supports
            ),
        )
        self.assertEqual(
            choice_certificate.immediate_multiplicity,
            sum(support.parent_weight for support in same_text_supports),
        )

        for support in open_supports:
            self.assertEqual(
                support.graph_obligation_work_evidence,
                (evidence,),
            )
            self.assertIs(
                support.graph_action_surface.closure_open_source_kind,
                (
                    writer_transitions
                    ._WriterClosureOpenObligationSourceKind
                    .LIVE_BRANCH_RETURN_CLOSURE_CANDIDATE
                ),
            )
            self.assertIs(
                support.policy_family,
                writer_transitions._WriterGraphPolicyActionFamily.CLOSURE_OPEN,
            )
            self.assertEqual(
                len(support.closure_candidate_resolution_evidence),
                1,
            )
            resolution = support.closure_candidate_resolution_evidence[0]
            self.assertEqual(resolution.bond, support.graph_action_surface.bond)
            self.assertEqual(
                resolution.first_atom,
                support.graph_action_surface.boundary_atom,
            )
            self.assertEqual(
                resolution.second_atom,
                support.graph_action_surface.partner_atom,
            )
            self.assertIsInstance(support.events[0], WriterRingLabelAllocated)
            self.assertIsInstance(support.events[1], WriterRingEndpointEmitted)
            opened_lifecycle_evidence = tuple(
                evidence
                for evidence in support.closure_candidate_lifecycle_evidence
                if (
                    evidence.outcome_kind
                    is WriterClosureCandidateLifecycleOutcomeKind.OPENED
                )
            )
            self.assertEqual(len(opened_lifecycle_evidence), 1)
            self.assertEqual(
                opened_lifecycle_evidence[0].bond,
                resolution.bond,
            )
            certificate = _single_closure_candidate_branch_certificate(
                support,
                (
                    WriterClosureCandidateBranchCertificateKind
                    .LIVE_BRANCH_RETURN_OPENED
                ),
            )
            self.assertEqual(certificate.bond, resolution.bond)

        text = open_supports[0].emitted_text
        same_text_supports = tuple(
            support for support in open_supports if support.emitted_text == text
        )
        choice = next(
            choice for choice in batch.choices.choices
            if choice.emitted_text == text
        )
        expected_successors: Counter = Counter()
        for support in same_text_supports:
            expected_successors[support.successor_state] += support.parent_weight

        self.assertEqual(
            choice.successor,
            WriterFrontierCursor(
                weighted_states=tuple(expected_successors.items())
            ),
        )

    def test_mixed_live_and_unsupported_closure_candidates_still_blocks(
        self,
    ) -> None:
        prepared = _prepare(_two_live_closure_candidate_facts())
        cursor = WriterFrontierCursor(
            weighted_states=(
                (
                    writer_state_key(
                        _mixed_live_and_unsupported_closure_candidate_state()
                    ),
                    1,
                ),
            )
        )
        key = cursor.weighted_states[0][0]
        context = build_writer_graph_obligation_context(prepared, key)
        evidence = writer_graph_obligation_work_evidence(
            operation="test",
            prepared=prepared,
            key=key,
            context=context,
        )

        self.assertEqual(evidence.closure_candidate_count, 2)
        self.assertEqual(
            evidence.live_branch_return_closure_candidate_count,
            1,
        )
        self.assertEqual(
            evidence.deferred_control_live_closure_candidate_count,
            0,
        )
        self.assertEqual(evidence.unsupported_closure_candidate_count, 1)
        violation = writer_graph_obligation_work_envelope_violation(evidence)
        self.assertIsNotNone(violation)
        assert violation is not None
        self.assertEqual(
            violation.metric,
            "unsupported_closure_candidate_count",
        )

        with self.assertRaises(SouthStarError) as caught:
            _checked_writer_frontier_branch_supports(
                prepared,
                cursor,
                include_counts=False,
            )

        self.assertIs(caught.exception.kind, SouthStarErrorKind.UNSUPPORTED_POLICY)

    def test_control_live_roles_include_pending_parent_and_branch_return(
        self,
    ) -> None:
        key = writer_state_key(
            _pending_parent_branch_return_closure_candidate_state(
                frozen_endpoint=False,
            )
        )

        roles = writer_control_live_roles_by_atom(key)

        self.assertIn(
            WriterControlLiveAtomRole.PENDING_PARENT,
            roles[AtomId(3)],
        )
        self.assertIn(
            WriterControlLiveAtomRole.BRANCH_RETURN,
            roles[AtomId(0)],
        )

    def test_deferred_control_live_closure_candidate_allows_pending_step(
        self,
    ) -> None:
        prepared = _prepare(_pending_control_live_closure_candidate_facts())
        cursor = WriterFrontierCursor(
            weighted_states=(
                (
                    writer_state_key(
                        _pending_parent_branch_return_closure_candidate_state(
                            frozen_endpoint=False,
                        )
                    ),
                    1,
                ),
            )
        )
        key = cursor.weighted_states[0][0]
        context = build_writer_graph_obligation_context(prepared, key)
        evidence = writer_graph_obligation_work_evidence(
            operation="writer graph obligation context",
            prepared=prepared,
            key=key,
            context=context,
        )

        self.assertEqual(evidence.closure_candidate_count, 1)
        self.assertEqual(
            evidence.live_branch_return_closure_candidate_count,
            0,
        )
        self.assertEqual(
            evidence.deferred_branch_return_closure_candidate_count,
            0,
        )
        self.assertEqual(
            evidence.deferred_control_live_closure_candidate_count,
            1,
        )
        self.assertEqual(evidence.unsupported_closure_candidate_count, 0)
        self.assertIsNone(
            writer_graph_obligation_work_envelope_violation(evidence)
        )

        batch = _checked_writer_frontier_branch_supports(
            prepared,
            cursor,
            include_counts=False,
        )
        self.assertFalse(
            any(
                support.transition_kind
                is writer_transitions.WriterTransitionKind.OPEN_CLOSURE_ENDPOINT
                for support in batch.supports
            )
        )
        support = next(
            support
            for support in batch.supports
            if (
                _WriterExecutionCapabilityKind
                .DEFERRED_CONTROL_LIVE_CLOSURE_CANDIDATE
            )
            in support.execution_capabilities
        )

        self.assertIs(
            support.transition_kind,
            writer_transitions.WriterTransitionKind.ENTER_CHILD_BOND,
        )
        self.assertEqual(
            support.graph_obligation_work_evidence[0]
            .deferred_control_live_closure_candidate_count,
            1,
        )
        self.assertEqual(
            len(support.closure_candidate_resolution_evidence),
            1,
        )
        self.assertIs(
            (
                support.closure_candidate_resolution_evidence[0]
                .resolution_kind
            ),
            WriterClosureCandidateResolutionKind.DEFERRED_CONTROL_LIVE,
        )

        successor_context = build_writer_graph_obligation_context(
            prepared,
            support.successor_state,
        )
        successor_evidence = writer_graph_obligation_work_evidence(
            operation="writer graph obligation context",
            prepared=prepared,
            key=support.successor_state,
            context=successor_context,
        )
        self.assertEqual(
            successor_evidence.deferred_control_live_closure_candidate_count,
            1,
        )
        self.assertEqual(
            successor_evidence.unsupported_closure_candidate_count,
            0,
        )

    def test_deferred_control_live_candidate_lifecycle_survives_pending_step(
        self,
    ) -> None:
        prepared = _prepare(_pending_control_live_closure_candidate_facts())
        cursor = WriterFrontierCursor(
            weighted_states=(
                (
                    writer_state_key(
                        _pending_parent_branch_return_closure_candidate_state(
                            frozen_endpoint=False,
                        )
                    ),
                    1,
                ),
            )
        )
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            cursor,
            include_counts=False,
        )
        support = next(
            support
            for support in batch.supports
            if (
                _WriterExecutionCapabilityKind
                .DEFERRED_CONTROL_LIVE_CLOSURE_CANDIDATE
            )
            in support.execution_capabilities
        )

        validate_writer_closure_candidate_lifecycle_transition(
            prepared=prepared,
            source_state=support.source_state,
            successor_state=support.successor_state,
            transition_kind=support.transition_kind,
            graph_action_surface=support.graph_action_surface,
        )
        evidence = support.closure_candidate_lifecycle_evidence
        self.assertEqual(len(evidence), 1)
        self.assertIs(
            evidence[0].outcome_kind,
            WriterClosureCandidateLifecycleOutcomeKind.RETAINED_SUPPORTED,
        )
        self.assertIn(
            evidence[0].successor_resolution.resolution_kind,
            {
                WriterClosureCandidateResolutionKind.DEFERRED_CONTROL_LIVE,
                WriterClosureCandidateResolutionKind.DEFERRED_BRANCH_RETURN,
                WriterClosureCandidateResolutionKind.LIVE_BRANCH_RETURN,
            },
        )
        certificate = _single_closure_candidate_branch_certificate(
            support,
            (
                WriterClosureCandidateBranchCertificateKind
                .DEFERRED_CONTROL_LIVE_RETAINED
            ),
        )
        self.assertIs(
            certificate.lifecycle_evidence.outcome_kind,
            WriterClosureCandidateLifecycleOutcomeKind.RETAINED_SUPPORTED,
        )
        self.assertIsNot(
            support.transition_kind,
            writer_transitions.WriterTransitionKind.OPEN_CLOSURE_ENDPOINT,
        )

    def test_deferred_branch_return_candidate_gets_retained_certificate(
        self,
    ) -> None:
        prepared = _prepare(cyclopropane_facts())
        cursor = WriterFrontierCursor(
            weighted_states=(
                (
                    writer_state_key(
                        _deferred_branch_return_closure_candidate_state(
                            frozen_endpoint=False,
                        )
                    ),
                    1,
                ),
            )
        )
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            cursor,
            include_counts=False,
        )
        support = next(
            support
            for support in batch.supports
            if (
                _WriterExecutionCapabilityKind
                .DEFERRED_BRANCH_RETURN_CLOSURE_CANDIDATE
            )
            in support.execution_capabilities
        )

        certificate = _single_closure_candidate_branch_certificate(
            support,
            (
                WriterClosureCandidateBranchCertificateKind
                .DEFERRED_BRANCH_RETURN_RETAINED
            ),
        )
        self.assertIs(
            certificate.lifecycle_evidence.outcome_kind,
            WriterClosureCandidateLifecycleOutcomeKind.RETAINED_SUPPORTED,
        )

    def test_deferred_candidate_lifecycle_rejects_unsupported_successor(
        self,
    ) -> None:
        prepared = _prepare(_pending_control_live_closure_candidate_facts())
        source = writer_state_key(
            _pending_parent_branch_return_closure_candidate_state(
                frozen_endpoint=False,
            )
        )
        successor = writer_state_key(
            _pending_parent_branch_return_closure_candidate_state(
                frozen_endpoint=True,
            )
        )

        violations = writer_closure_candidate_lifecycle_transition_violations(
            prepared=prepared,
            source_state=source,
            successor_state=successor,
            transition_kind=writer_transitions.WriterTransitionKind.ENTER_CHILD_BOND,
            graph_action_surface=None,
        )

        self.assertIn(
            "supported_candidate_became_unsupported",
            violations,
        )

    def test_live_candidate_lifecycle_rejects_claimed_open_without_open_state(
        self,
    ) -> None:
        prepared = _prepare(cyclopropane_facts())
        state = writer_state_key(
            _branch_return_closure_candidate_state(live_partner=True)
        )

        violations = writer_closure_candidate_lifecycle_transition_violations(
            prepared=prepared,
            source_state=state,
            successor_state=state,
            transition_kind=(
                writer_transitions.WriterTransitionKind.OPEN_CLOSURE_ENDPOINT
            ),
            graph_action_surface=SimpleNamespace(
                bond=BondId(0),
                boundary_atom=AtomId(0),
                partner_atom=AtomId(1),
            ),
        )

        self.assertIn(
            "opened_candidate_lacks_successor_open_endpoint",
            violations,
        )

    def test_live_open_certificate_requires_opened_lifecycle(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        cursor = WriterFrontierCursor(
            weighted_states=(
                (
                    writer_state_key(
                        _branch_return_closure_candidate_state(
                            live_partner=True,
                        )
                    ),
                    1,
                ),
            )
        )
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            cursor,
            include_counts=False,
        )
        support = next(
            support
            for support in batch.supports
            if (
                _WriterExecutionCapabilityKind
                .LIVE_BRANCH_RETURN_CLOSURE_CANDIDATE_OPEN
            )
            in support.execution_capabilities
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "capability_lacks_exact_lifecycle",
        ):
            writer_closure_candidate_branch_certificates(
                execution_capabilities=support.execution_capabilities,
                transition_kind=support.transition_kind,
                graph_action_surface=support.graph_action_surface,
                graph_obligation_work_evidence=(
                    support.graph_obligation_work_evidence
                ),
                closure_candidate_resolution_evidence=(
                    support.closure_candidate_resolution_evidence
                ),
                closure_candidate_lifecycle_evidence=(),
                events=support.events,
            )

    def test_deferred_certificate_requires_retained_lifecycle(self) -> None:
        prepared = _prepare(_pending_control_live_closure_candidate_facts())
        cursor = WriterFrontierCursor(
            weighted_states=(
                (
                    writer_state_key(
                        _pending_parent_branch_return_closure_candidate_state(
                            frozen_endpoint=False,
                        )
                    ),
                    1,
                ),
            )
        )
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            cursor,
            include_counts=False,
        )
        support = next(
            support
            for support in batch.supports
            if (
                _WriterExecutionCapabilityKind
                .DEFERRED_CONTROL_LIVE_CLOSURE_CANDIDATE
            )
            in support.execution_capabilities
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "deferred_capability_lacks_retained_lifecycle",
        ):
            writer_closure_candidate_branch_certificates(
                execution_capabilities=support.execution_capabilities,
                transition_kind=support.transition_kind,
                graph_action_surface=support.graph_action_surface,
                graph_obligation_work_evidence=(
                    support.graph_obligation_work_evidence
                ),
                closure_candidate_resolution_evidence=(
                    support.closure_candidate_resolution_evidence
                ),
                closure_candidate_lifecycle_evidence=(),
                events=support.events,
            )

    def test_control_live_candidate_with_frozen_endpoint_still_blocks(
        self,
    ) -> None:
        prepared = _prepare(_pending_control_live_closure_candidate_facts())
        cursor = WriterFrontierCursor(
            weighted_states=(
                (
                    writer_state_key(
                        _pending_parent_branch_return_closure_candidate_state(
                            frozen_endpoint=True,
                        )
                    ),
                    1,
                ),
            )
        )
        key = cursor.weighted_states[0][0]
        context = build_writer_graph_obligation_context(prepared, key)
        evidence = writer_graph_obligation_work_evidence(
            operation="test",
            prepared=prepared,
            key=key,
            context=context,
        )

        self.assertEqual(
            evidence.deferred_control_live_closure_candidate_count,
            0,
        )
        self.assertEqual(evidence.unsupported_closure_candidate_count, 1)
        violation = writer_graph_obligation_work_envelope_violation(evidence)
        self.assertIsNotNone(violation)
        assert violation is not None
        self.assertEqual(
            violation.metric,
            "unsupported_closure_candidate_count",
        )

        with self.assertRaises(SouthStarError) as caught:
            _checked_writer_frontier_branch_supports(
                prepared,
                cursor,
                include_counts=False,
            )

        self.assertIs(caught.exception.kind, SouthStarErrorKind.UNSUPPORTED_POLICY)

    def test_deferred_candidate_with_frozen_endpoint_still_blocks(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        cursor = WriterFrontierCursor(
            weighted_states=(
                (
                    writer_state_key(
                        _deferred_branch_return_closure_candidate_state(
                            frozen_endpoint=True,
                        )
                    ),
                    1,
                ),
            )
        )
        key = cursor.weighted_states[0][0]
        context = build_writer_graph_obligation_context(prepared, key)
        evidence = writer_graph_obligation_work_evidence(
            operation="test",
            prepared=prepared,
            key=key,
            context=context,
        )

        self.assertEqual(evidence.unsupported_closure_candidate_count, 1)
        violation = writer_graph_obligation_work_envelope_violation(evidence)
        self.assertIsNotNone(violation)
        assert violation is not None
        self.assertEqual(
            violation.metric,
            "unsupported_closure_candidate_count",
        )

        with self.assertRaises(SouthStarError) as caught:
            _checked_writer_frontier_branch_supports(
                prepared,
                cursor,
                include_counts=False,
            )

        self.assertIs(caught.exception.kind, SouthStarErrorKind.UNSUPPORTED_POLICY)

    def test_residual_closure_open_obligation_requires_source_shape(self) -> None:
        with self.assertRaises(SouthStarError):
            writer_transitions._WriterClosureOpenObligation(
                bond=BondId(0),
                first_atom=AtomId(0),
                second_atom=AtomId(1),
                attachment_id=1,
                attachment_action_kind=(
                    WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
                ),
                owner_kind=None,
                source_attachment=None,
            )

    def test_residual_attachment_closure_open_has_lifecycle_evidence(
        self,
    ) -> None:
        prepared = _prepare(cyclopropane_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _find_checked_branch_support(
            prepared,
            initial,
            lambda support: (
                support.transition_kind
                is writer_transitions.WriterTransitionKind.OPEN_CLOSURE_ENDPOINT
                and getattr(
                    support.graph_action_surface,
                    "attachment_id",
                    None,
                )
                is not None
            ),
        )

        evidence = support.residual_attachment_lifecycle_evidence
        self.assertEqual(len(evidence), 1)
        self.assertIs(
            evidence[0].outcome_kind,
            (
                WriterResidualAttachmentLifecycleOutcomeKind
                .CLOSURE_OPEN_DISCHARGED
            ),
        )
        self.assertEqual(evidence[0].bond, support.graph_action_surface.bond)
        self.assertEqual(
            evidence[0].source_closure_deficit,
            evidence[0].successor_closure_deficit + 1,
        )
        self.assertEqual(
            evidence[0].removed_boundary_bonds,
            (support.graph_action_surface.bond,),
        )

    def test_closure_candidate_open_has_no_residual_attachment_lifecycle(
        self,
    ) -> None:
        prepared = _prepare(cyclopropane_facts())
        cursor = WriterFrontierCursor(
            weighted_states=(
                (
                    writer_state_key(
                        _branch_return_closure_candidate_state(
                            live_partner=True,
                        )
                    ),
                    1,
                ),
            )
        )
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            cursor,
            include_counts=False,
        )
        support = next(
            support
            for support in batch.supports
            if (
                _WriterExecutionCapabilityKind
                .LIVE_BRANCH_RETURN_CLOSURE_CANDIDATE_OPEN
            )
            in support.execution_capabilities
        )

        self.assertEqual(support.residual_attachment_lifecycle_evidence, ())
        self.assertEqual(support.residual_attachment_branch_certificates, ())

    def test_coupled_cyclic_attachment_capability_is_certified(self) -> None:
        capability = (
            _WriterExecutionCapabilityKind
            .COUPLED_CYCLIC_ATTACHMENT_RESTRICTION
        )
        prepared = _prepare(_fused_rank_two_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _find_checked_branch_support(
            prepared,
            initial,
            lambda support: capability in support.execution_capabilities,
        )

        certificate = _single_residual_attachment_branch_certificate(
            support,
            (
                WriterResidualAttachmentBranchCertificateKind
                .COUPLED_CYCLIC_ATTACHMENT_DISCHARGED
            ),
        )
        self.assertEqual(certificate.bond, support.graph_action_surface.bond)
        self.assertEqual(
            certificate.lifecycle_evidence.source_closure_deficit,
            2,
        )
        self.assertEqual(
            certificate.lifecycle_evidence.successor_closure_deficit,
            1,
        )
        self.assertEqual(
            len(certificate.lifecycle_evidence.source_attachment.block_ids),
            1,
        )

    def test_residual_attachment_lifecycle_rejects_malformed_successor(
        self,
    ) -> None:
        prepared = _prepare(cyclopropane_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _find_checked_branch_support(
            prepared,
            initial,
            lambda support: (
                support.transition_kind
                is writer_transitions.WriterTransitionKind.OPEN_CLOSURE_ENDPOINT
                and getattr(
                    support.graph_action_surface,
                    "attachment_id",
                    None,
                )
                is not None
            ),
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "successor_boundary_mismatch",
        ):
            validate_writer_residual_attachment_lifecycle_transition(
                prepared=prepared,
                source_state=support.source_state,
                successor_state=support.source_state,
                graph_action_surface=support.graph_action_surface,
            )

    def test_coupled_certificate_requires_lifecycle(self) -> None:
        with self.assertRaisesRegex(
            SouthStarError,
            "coupled_capability_lacks_lifecycle",
        ):
            writer_residual_attachment_branch_certificates(
                execution_capabilities=frozenset(
                    {
                        (
                            _WriterExecutionCapabilityKind
                            .COUPLED_CYCLIC_ATTACHMENT_RESTRICTION
                        )
                    }
                ),
                graph_action_surface=SimpleNamespace(),
                residual_attachment_lifecycle_evidence=(),
            )

    def test_tetra_atom_token_capability_is_stereo_certified(self) -> None:
        capability = _WriterExecutionCapabilityKind.TETRA_TOKEN_RESTRICTION
        prepared = _prepare(tetrahedral_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _find_checked_branch_support(
            prepared,
            initial,
            lambda support: capability in support.execution_capabilities,
        )

        self.assertTrue(support.stereo_lifecycle_evidence)
        certificate = _single_stereo_branch_certificate(
            support,
            WriterStereoBranchCertificateKind.TETRA_TOKEN_RESTRICTED,
        )
        self.assertIsInstance(certificate.event, WriterAtomEmitted)
        self.assertIn(capability, certificate.lifecycle_evidence.capabilities)
        self.assertTrue(certificate.residual_work_evidence)
        self.assertEqual(
            certificate.residual_work_evidence[0].operation,
            "tetrahedral atom-token restriction",
        )

    def test_tetra_local_order_capability_is_stereo_certified(self) -> None:
        capability = (
            _WriterExecutionCapabilityKind.TETRA_LOCAL_ORDER_RESTRICTION
        )
        prepared = _prepare(tetrahedral_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _find_checked_branch_support(
            prepared,
            initial,
            lambda support: capability in support.execution_capabilities,
        )

        certificate = _single_stereo_branch_certificate(
            support,
            WriterStereoBranchCertificateKind.TETRA_LOCAL_ORDER_RESTRICTED,
        )
        self.assertIsInstance(certificate.event, WriterLocalOrderClosed)
        self.assertIn(capability, certificate.lifecycle_evidence.capabilities)
        self.assertEqual(
            certificate.residual_work_evidence[0].operation,
            "tetrahedral local-order factor closure",
        )

    def test_directional_carrier_capability_is_stereo_certified(self) -> None:
        capability = (
            _WriterExecutionCapabilityKind.DIRECTIONAL_CARRIER_RESTRICTION
        )
        prepared = _prepare(directional_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _find_checked_branch_support(
            prepared,
            initial,
            lambda support: capability in support.execution_capabilities,
        )

        certificate = _single_stereo_branch_certificate(
            support,
            WriterStereoBranchCertificateKind.DIRECTIONAL_CARRIER_RESTRICTED,
        )
        self.assertIsInstance(certificate.event, WriterBondEmitted)
        self.assertIn(capability, certificate.lifecycle_evidence.capabilities)
        self.assertEqual(
            certificate.residual_work_evidence[0].operation,
            "directional carrier-mark restriction",
        )

    def test_residual_factor_discharge_capability_is_stereo_certified(
        self,
    ) -> None:
        capability = _WriterExecutionCapabilityKind.RESIDUAL_FACTOR_DISCHARGE
        prepared = _prepare(directional_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _find_checked_branch_support(
            prepared,
            initial,
            lambda support: capability in support.execution_capabilities,
        )

        certificate = _single_stereo_branch_certificate(
            support,
            WriterStereoBranchCertificateKind.RESIDUAL_FACTOR_DISCHARGED,
        )
        self.assertIn(capability, certificate.lifecycle_evidence.capabilities)
        self.assertNotEqual(
            certificate.lifecycle_evidence.source_residual_snapshot,
            certificate.lifecycle_evidence.successor_residual_snapshot,
        )

    def test_stereo_certificate_requires_matching_lifecycle(self) -> None:
        with self.assertRaisesRegex(
            SouthStarError,
            "tetra_token_restriction_lacks_exact_lifecycle",
        ):
            writer_stereo_branch_certificates(
                execution_capabilities=frozenset(
                    {_WriterExecutionCapabilityKind.TETRA_TOKEN_RESTRICTION}
                ),
                stereo_lifecycle_evidence=(),
                events=(),
            )

    def test_checked_branch_certificate_rejects_missing_stereo_capability(
        self,
    ) -> None:
        capability = (
            _WriterExecutionCapabilityKind.DIRECTIONAL_CARRIER_RESTRICTION
        )
        prepared = _prepare(directional_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _find_checked_branch_support(
            prepared,
            initial,
            lambda support: capability in support.execution_capabilities,
        )

        with self.assertRaisesRegex(
            SouthStarError,
            "capability_coverage_execution_mismatch",
        ):
            writer_checked_branch_support_certificate(
                source_state=support.source_state,
                successor_state=support.successor_state,
                emitted_text=support.emitted_text,
                parent_weight=support.parent_weight,
                branch_ordinal=support.branch_ordinal,
                transition_kind=support.transition_kind,
                graph_action_surface=support.graph_action_surface,
                policy_family=support.policy_family,
                events=support.events,
                transition_evidence=support.evidence,
                execution_capabilities=frozenset(
                    capability
                    for capability in support.execution_capabilities
                    if capability
                    is not (
                        _WriterExecutionCapabilityKind
                        .DIRECTIONAL_CARRIER_RESTRICTION
                    )
                ),
                graph_obligation_work_evidence=(
                    support.graph_obligation_work_evidence
                ),
                residual_work_evidence=support.residual_work_evidence,
                finite_relation_work_evidence=(
                    support.finite_relation_work_evidence
                ),
                closure_candidate_resolution_evidence=(
                    support.closure_candidate_resolution_evidence
                ),
                closure_candidate_lifecycle_evidence=(
                    support.closure_candidate_lifecycle_evidence
                ),
                closure_candidate_branch_certificates=(
                    support.closure_candidate_branch_certificates
                ),
                residual_attachment_lifecycle_evidence=(
                    support.residual_attachment_lifecycle_evidence
                ),
                residual_attachment_branch_certificates=(
                    support.residual_attachment_branch_certificates
                ),
                stereo_lifecycle_evidence=support.stereo_lifecycle_evidence,
                stereo_branch_certificates=support.stereo_branch_certificates,
                residual_attachment_policy_evidence=(
                    support.residual_attachment_policy_evidence
                ),
                capability_coverage_certificate=(
                    support.capability_coverage_certificate
                ),
                successor_state_certificate=(
                    support.successor_state_certificate
                ),
            )

    def test_checked_branch_certificate_rejects_policy_mismatch(self) -> None:
        prepared = _prepare(cco_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        support = _checked_writer_frontier_branch_supports(
            prepared,
            initial,
            include_counts=False,
        ).supports[0]

        with self.assertRaisesRegex(
            SouthStarError,
            "graph_action_surface_policy_family_mismatch",
        ):
            writer_checked_branch_support_certificate(
                source_state=support.source_state,
                successor_state=support.successor_state,
                emitted_text=support.emitted_text,
                parent_weight=support.parent_weight,
                branch_ordinal=support.branch_ordinal,
                transition_kind=support.transition_kind,
                graph_action_surface=SimpleNamespace(policy_family=object()),
                policy_family=support.policy_family,
                events=support.events,
                transition_evidence=support.evidence,
                execution_capabilities=support.execution_capabilities,
                graph_obligation_work_evidence=(
                    support.graph_obligation_work_evidence
                ),
                residual_work_evidence=support.residual_work_evidence,
                finite_relation_work_evidence=(
                    support.finite_relation_work_evidence
                ),
                closure_candidate_resolution_evidence=(
                    support.closure_candidate_resolution_evidence
                ),
                closure_candidate_lifecycle_evidence=(
                    support.closure_candidate_lifecycle_evidence
                ),
                closure_candidate_branch_certificates=(
                    support.closure_candidate_branch_certificates
                ),
                residual_attachment_lifecycle_evidence=(
                    support.residual_attachment_lifecycle_evidence
                ),
                residual_attachment_branch_certificates=(
                    support.residual_attachment_branch_certificates
                ),
                stereo_lifecycle_evidence=support.stereo_lifecycle_evidence,
                stereo_branch_certificates=support.stereo_branch_certificates,
                residual_attachment_policy_evidence=(
                    support.residual_attachment_policy_evidence
                ),
                capability_coverage_certificate=(
                    support.capability_coverage_certificate
                ),
                successor_state_certificate=(
                    support.successor_state_certificate
                ),
            )

    def test_text_projection_certificate_rejects_missing_support(self) -> None:
        prepared = _prepare(cco_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            initial,
            include_counts=False,
        )
        choice = batch.choices.choices[0]

        with self.assertRaisesRegex(
            SouthStarError,
            "choice_lacks_branch_support",
        ):
            writer_text_choice_projection_certificates(
                source_cursor=initial,
                choices=SimpleNamespace(
                    choices=(
                        SimpleNamespace(
                            emitted_text="missing",
                            successor=choice.successor,
                            immediate_multiplicity=(
                                choice.immediate_multiplicity
                            ),
                            support_count=choice.support_count,
                            completion_count=choice.completion_count,
                        ),
                    )
                ),
                branch_supports=batch.supports,
            )

    def test_text_projection_certificate_rejects_successor_mismatch(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            initial,
            include_counts=False,
        )
        choice = batch.choices.choices[0]

        with self.assertRaisesRegex(
            SouthStarError,
            "choice_successor_cursor_mismatch",
        ):
            writer_text_choice_projection_certificates(
                source_cursor=initial,
                choices=SimpleNamespace(
                    choices=(
                        SimpleNamespace(
                            emitted_text=choice.emitted_text,
                            successor=WriterFrontierCursor(
                                weighted_states=()
                            ),
                            immediate_multiplicity=(
                                choice.immediate_multiplicity
                            ),
                            support_count=choice.support_count,
                            completion_count=choice.completion_count,
                        ),
                    )
                ),
                branch_supports=batch.supports,
            )

    def test_text_projection_certificate_rejects_multiplicity_mismatch(
        self,
    ) -> None:
        prepared = _prepare(cco_facts())
        initial = initial_writer_frontier_cursor(prepared, _writer_options())
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            initial,
            include_counts=False,
        )
        choice = batch.choices.choices[0]

        with self.assertRaisesRegex(
            SouthStarError,
            "choice_immediate_multiplicity_mismatch",
        ):
            writer_text_choice_projection_certificates(
                source_cursor=initial,
                choices=SimpleNamespace(
                    choices=(
                        SimpleNamespace(
                            emitted_text=choice.emitted_text,
                            successor=choice.successor,
                            immediate_multiplicity=(
                                choice.immediate_multiplicity + 1
                            ),
                            support_count=choice.support_count,
                            completion_count=choice.completion_count,
                        ),
                    )
                ),
                branch_supports=batch.supports,
            )

    def test_closure_candidate_liveness_is_graph_owned(self) -> None:
        source = inspect.getsource(writer_transitions)

        self.assertIn(
            "writer_live_branch_return_closure_candidate_resolutions",
            source,
        )
        self.assertNotIn(
            "def _live_branch_return_closure_candidate_partner",
            source,
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

    def test_runtime_branch_completion_count_certificate_is_frontier_owned(self) -> None:
        prepared = _prepare(cco_facts())
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        )

        facade = count_writer_runtime_branch_completions(
            prepared=prepared,
            state=state,
        )
        certificate = writer_runtime_branch_completion_count_certificate(
            prepared=prepared,
            state=state,
        )
        self.assertEqual(facade, certificate.completion_count)

        state_key, _, state_count = (
            certificate.state_count_certificates[0]
        )
        self.assertEqual(
            state_count.completion_count,
            state_count.terminal_count
            + sum(term.successor_count for term in state_count.branch_terms),
        )

        for term in state_count.branch_terms:
            self.assertIsNotNone(term.branch_certificate)
            self.assertIsNotNone(term.successor_count_certificate)
            self.assertIsInstance(term.successor_count, int)

        with self.assertRaises(SouthStarError):
            writer_cursor_completion_count_certificate(
                cursor=state.snapshot.cursor,
                state_count_certificates=((state_key, 0, state_count),),
            )

        frontier = _checked_writer_frontier_branch_supports(
            prepared,
            state.snapshot.cursor,
            include_counts=False,
        )
        if frontier.terminal_projection_certificate is not None:
            terminal_count = frontier.terminal_projection_certificate.terminal.completion_count
        else:
            terminal_count = 0

        reconstructed = writer_state_completion_count_certificate(
            state_key=state_key,
            terminal_projection_certificate=frontier.terminal_projection_certificate,
            terminal_count=terminal_count,
            branch_terms=state_count.branch_terms,
        )
        self.assertEqual(state_count.completion_count, reconstructed.completion_count)

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
        self.assertEqual(
            branches.text_choice_projection_certificates,
            batch.text_choice_projection_certificates,
        )
        self.assertEqual(
            branches.terminal_projection_certificate,
            batch.terminal_projection_certificate,
        )

        for support, branch in zip(batch.supports, branches.transitions):
            expected = _writer_search_snapshot_after_checked_branch_support(
                state.snapshot,
                prepared=prepared,
                support=support,
            )
            self.assertEqual(support.successor_cursor, expected.cursor)
            self.assertEqual(branch.next_state.snapshot, expected)
            self.assertEqual(
                branch.graph_obligation_work_evidence,
                support.graph_obligation_work_evidence,
            )
            self.assertEqual(
                branch.graph_action_surface,
                support.graph_action_surface,
            )
            self.assertEqual(branch.policy_family, support.policy_family)
            self.assertEqual(
                branch.closure_candidate_resolution_evidence,
                support.closure_candidate_resolution_evidence,
            )
            self.assertEqual(
                branch.closure_candidate_lifecycle_evidence,
                support.closure_candidate_lifecycle_evidence,
            )
            self.assertEqual(
                branch.closure_candidate_branch_certificates,
                support.closure_candidate_branch_certificates,
            )
            self.assertEqual(
                branch.residual_attachment_lifecycle_evidence,
                support.residual_attachment_lifecycle_evidence,
            )
            self.assertEqual(
                branch.residual_attachment_branch_certificates,
                support.residual_attachment_branch_certificates,
            )
            self.assertEqual(
                branch.stereo_lifecycle_evidence,
                support.stereo_lifecycle_evidence,
            )
            self.assertEqual(
                branch.stereo_branch_certificates,
                support.stereo_branch_certificates,
            )
            self.assertEqual(
                branch.residual_attachment_policy_evidence,
                support.residual_attachment_policy_evidence,
            )
            self.assertEqual(
                branch.checked_branch_certificate,
                support.checked_branch_certificate,
            )

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


def _single_closure_candidate_branch_certificate(
    support,
    kind: WriterClosureCandidateBranchCertificateKind,
):
    matches = tuple(
        certificate
        for certificate in support.closure_candidate_branch_certificates
        if certificate.kind is kind
    )
    if len(matches) != 1:
        raise AssertionError(
            f"expected exactly one closure-candidate certificate {kind!r}"
        )
    return matches[0]


def _single_residual_attachment_branch_certificate(
    support,
    kind: WriterResidualAttachmentBranchCertificateKind,
):
    matches = tuple(
        certificate
        for certificate in support.residual_attachment_branch_certificates
        if certificate.kind is kind
    )
    if len(matches) != 1:
        raise AssertionError(
            f"expected exactly one residual-attachment certificate {kind!r}"
        )
    return matches[0]


def _single_stereo_branch_certificate(
    support,
    kind: WriterStereoBranchCertificateKind,
):
    matches = tuple(
        certificate
        for certificate in support.stereo_branch_certificates
        if certificate.kind is kind
    )
    if len(matches) != 1:
        raise AssertionError(
            f"expected exactly one stereo certificate {kind!r}"
        )
    return matches[0]


def _find_checked_branch_support(
    prepared,
    cursor: WriterFrontierCursor,
    predicate,
):
    pending = [cursor]
    seen: set[WriterFrontierCursor] = set()
    while pending and len(seen) < 1000:
        current = pending.pop(0)
        if current in seen:
            continue
        seen.add(current)
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            current,
            include_counts=False,
        )
        for support in batch.supports:
            if predicate(support):
                return support
            pending.append(support.successor_cursor)

    raise AssertionError("expected checked branch support was not found")


def _first_terminal_frontier_product(prepared):
    pending = [initial_writer_frontier_cursor(prepared, _writer_options())]
    seen: set[WriterFrontierCursor] = set()
    while pending and len(seen) < 1000:
        current = pending.pop(0)
        if current in seen:
            continue
        seen.add(current)
        product = _checked_writer_frontier_product(
            prepared,
            current,
            include_counts=False,
            include_frontier_certificate=False,
            include_count_certificate=False,
        )
        if product.terminal_supports:
            return product
        pending.extend(choice.successor for choice in product.choices.choices)

    raise AssertionError("expected terminal frontier product was not found")


def _checked_branch_certificate_kwargs(support, **overrides):
    kwargs = dict(
        source_state=support.source_state,
        successor_state=support.successor_state,
        emitted_text=support.emitted_text,
        parent_weight=support.parent_weight,
        branch_ordinal=support.branch_ordinal,
        transition_kind=support.transition_kind,
        graph_action_surface=support.graph_action_surface,
        policy_family=support.policy_family,
        events=support.events,
        transition_evidence=support.evidence,
        execution_capabilities=support.execution_capabilities,
        graph_obligation_work_evidence=(
            support.graph_obligation_work_evidence
        ),
        residual_work_evidence=support.residual_work_evidence,
        finite_relation_work_evidence=support.finite_relation_work_evidence,
        closure_candidate_resolution_evidence=(
            support.closure_candidate_resolution_evidence
        ),
        closure_candidate_lifecycle_evidence=(
            support.closure_candidate_lifecycle_evidence
        ),
        closure_candidate_branch_certificates=(
            support.closure_candidate_branch_certificates
        ),
        residual_attachment_lifecycle_evidence=(
            support.residual_attachment_lifecycle_evidence
        ),
        residual_attachment_branch_certificates=(
            support.residual_attachment_branch_certificates
        ),
        stereo_lifecycle_evidence=support.stereo_lifecycle_evidence,
        stereo_branch_certificates=support.stereo_branch_certificates,
        residual_attachment_policy_evidence=(
            support.residual_attachment_policy_evidence
        ),
        capability_coverage_certificate=(
            support.capability_coverage_certificate
        ),
        successor_state_certificate=support.successor_state_certificate,
    )
    kwargs.update(overrides)
    return kwargs


def _successor_state_certificate_kwargs(support, **overrides):
    kwargs = dict(
        source_state=support.source_state,
        successor_state=support.successor_state,
        emitted_text=support.emitted_text,
        transition_kind=support.transition_kind,
        graph_action_surface=support.graph_action_surface,
        policy_family=support.policy_family,
        events=support.events,
        transition_evidence=support.evidence,
        graph_obligation_work_evidence=(
            support.graph_obligation_work_evidence
        ),
        residual_work_evidence=support.residual_work_evidence,
        finite_relation_work_evidence=support.finite_relation_work_evidence,
        closure_candidate_lifecycle_evidence=(
            support.closure_candidate_lifecycle_evidence
        ),
        residual_attachment_lifecycle_evidence=(
            support.residual_attachment_lifecycle_evidence
        ),
        stereo_lifecycle_evidence=support.stereo_lifecycle_evidence,
    )
    kwargs.update(overrides)
    return kwargs


def _replace_event_identity(events, old_event, new_event):
    return tuple(
        new_event if event is old_event else event
        for event in events
    )


def _writer_options() -> SouthStarRuntimeOptions:
    return SouthStarRuntimeOptions(
        serialization_language=SerializationLanguageMode.WRITER_SHAPED,
    )


def _branch_return_closure_candidate_state(
    *,
    live_partner: bool,
) -> WriterState:
    return_atom = WriterAtomFrame(
        atom=AtomId(1),
        parent=None,
        incoming_bond=None,
        atom_emitted=True,
    )
    return WriterState(
        component_cursor=ComponentCursor(
            component_index=0,
            component_roots=(AtomId(1),),
        ),
        active=WriterAtomFrame(
            atom=AtomId(0),
            parent=AtomId(2),
            incoming_bond=BondId(2),
            atom_emitted=True,
        ),
        branch_stack=(
            (WriterBranchFrame(return_atom=return_atom),)
            if live_partner
            else ()
        ),
        visited_atoms=frozenset((AtomId(0), AtomId(1), AtomId(2))),
        written_bonds=frozenset((BondId(1), BondId(2))),
        obligations=ObligationState(),
        ring_state=WriterRingState(),
        stereo_state=WriterStereoState(),
        policy_state=WriterPolicyState(),
    )


def _deferred_branch_return_closure_candidate_state(
    *,
    frozen_endpoint: bool,
) -> WriterState:
    return_frames = (
        WriterBranchFrame(
            return_atom=WriterAtomFrame(
                atom=AtomId(1),
                parent=None,
                incoming_bond=None,
                atom_emitted=True,
            )
        ),
        WriterBranchFrame(
            return_atom=WriterAtomFrame(
                atom=AtomId(0),
                parent=AtomId(2),
                incoming_bond=BondId(2),
                atom_emitted=True,
            )
        ),
    )
    if frozen_endpoint:
        return_frames = return_frames[1:]

    return WriterState(
        component_cursor=ComponentCursor(
            component_index=0,
            component_roots=(AtomId(1),),
        ),
        active=WriterAtomFrame(
            atom=AtomId(2),
            parent=AtomId(0),
            incoming_bond=BondId(2),
            atom_emitted=True,
        ),
        branch_stack=return_frames,
        visited_atoms=frozenset((AtomId(0), AtomId(1), AtomId(2))),
        written_bonds=frozenset((BondId(1), BondId(2))),
        obligations=ObligationState(),
        ring_state=WriterRingState(),
        stereo_state=WriterStereoState(),
        policy_state=WriterPolicyState(),
    )


def _two_live_closure_candidate_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=tuple(atom(index, "C") for index in range(4)),
        bonds=(
            single_bond(0, 3, 0),
            single_bond(1, 0, 1),
            single_bond(2, 1, 2),
            single_bond(3, 2, 0),
            single_bond(4, 2, 3),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1), AtomId(2), AtomId(3)),
                bonds=(BondId(0), BondId(1), BondId(2), BondId(3), BondId(4)),
            ),
        ),
    )


def _pending_control_live_closure_candidate_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=tuple(atom(index, "C") for index in range(4)),
        bonds=(
            single_bond(0, 3, 2),
            bond(1, 3, 1, BondOrder.DOUBLE),
            single_bond(2, 3, 0),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1), AtomId(2), AtomId(3)),
                bonds=(BondId(0), BondId(1), BondId(2)),
            ),
        ),
    )


def _fused_rank_two_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=tuple(atom(index, "C") for index in range(4)),
        bonds=(
            single_bond(0, 0, 1),
            single_bond(1, 1, 2),
            single_bond(2, 2, 0),
            single_bond(3, 0, 3),
            single_bond(4, 3, 1),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=tuple(AtomId(index) for index in range(4)),
                bonds=tuple(BondId(index) for index in range(5)),
            ),
        ),
    )


def _multi_live_branch_return_closure_candidate_state() -> WriterState:
    return _multi_closure_candidate_state(
        branch_return_atoms=(AtomId(3), AtomId(0)),
    )


def _mixed_live_and_unsupported_closure_candidate_state() -> WriterState:
    return _multi_closure_candidate_state(
        branch_return_atoms=(AtomId(0),),
    )


def _pending_parent_branch_return_closure_candidate_state(
    *,
    frozen_endpoint: bool,
) -> WriterState:
    return WriterState(
        component_cursor=ComponentCursor(
            component_index=0,
            component_roots=(AtomId(3),),
        ),
        active=WriterAtomFrame(
            atom=AtomId(2),
            parent=AtomId(3),
            incoming_bond=BondId(0),
            atom_emitted=True,
        ),
        branch_stack=(
            ()
            if frozen_endpoint
            else (
                WriterBranchFrame(
                    return_atom=WriterAtomFrame(
                        atom=AtomId(0),
                        parent=None,
                        incoming_bond=None,
                        atom_emitted=True,
                    )
                ),
            )
        ),
        visited_atoms=frozenset((AtomId(0), AtomId(2), AtomId(3))),
        written_bonds=frozenset((BondId(0),)),
        obligations=ObligationState(
            pending_entry=PendingWriterEntry(
                parent=AtomId(3),
                child=AtomId(1),
                bond=BondId(1),
                branch=False,
            )
        ),
        ring_state=WriterRingState(),
        stereo_state=WriterStereoState(),
        policy_state=WriterPolicyState(),
    )


def _multi_closure_candidate_state(
    *,
    branch_return_atoms: tuple[AtomId, ...],
) -> WriterState:
    return WriterState(
        component_cursor=ComponentCursor(
            component_index=0,
            component_roots=(AtomId(3),),
        ),
        active=WriterAtomFrame(
            atom=AtomId(2),
            parent=AtomId(1),
            incoming_bond=BondId(2),
            atom_emitted=True,
        ),
        branch_stack=tuple(
            WriterBranchFrame(
                return_atom=WriterAtomFrame(
                    atom=atom_id,
                    parent=(AtomId(3) if atom_id == AtomId(0) else None),
                    incoming_bond=(BondId(0) if atom_id == AtomId(0) else None),
                    atom_emitted=True,
                )
            )
            for atom_id in branch_return_atoms
        ),
        visited_atoms=frozenset(
            (AtomId(0), AtomId(1), AtomId(2), AtomId(3))
        ),
        written_bonds=frozenset((BondId(0), BondId(1), BondId(2))),
        obligations=ObligationState(),
        ring_state=WriterRingState(),
        stereo_state=WriterStereoState(),
        policy_state=WriterPolicyState(),
    )


if __name__ == "__main__":
    unittest.main()
