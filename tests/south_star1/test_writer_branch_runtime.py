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
from grimace._south_star1.writer_capabilities import _WriterExecutionCapabilityKind
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
from grimace._south_star1.writer_runtime import count_writer_runtime_branch_completions
from grimace._south_star1.writer_runtime import initial_writer_runtime_state
from grimace._south_star1.writer_runtime import writer_runtime_branch_transitions
from grimace._south_star1.writer_runtime import writer_runtime_choice_transitions
from grimace._south_star1.writer_runtime import writer_runtime_choices
from grimace._south_star1.writer_runtime import writer_runtime_diagnostics
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
            self.assertEqual(projected.residual_attachment_policy_evidence, ())

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
                    evidence.bond
                    for support in open_supports
                    for evidence in support.closure_candidate_lifecycle_evidence
                    if (
                        evidence.outcome_kind
                        is WriterClosureCandidateLifecycleOutcomeKind.OPENED
                    )
                }
            ),
            len(open_supports),
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
                branch.residual_attachment_policy_evidence,
                support.residual_attachment_policy_evidence,
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
