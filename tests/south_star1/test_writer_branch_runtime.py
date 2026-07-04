"""Tests for the public branch-preserving writer runtime surface."""

from __future__ import annotations

import inspect
import unittest
from collections import Counter
from types import SimpleNamespace

import grimace._south_star1.writer_frontier as writer_frontier_module
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
from grimace._south_star1.writer_frontier import _checked_writer_frontier_branch_supports
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
from grimace._south_star1.writer_state import WriterPolicyState
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
