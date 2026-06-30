"""Tests for the public branch-preserving writer runtime surface."""

from __future__ import annotations

import unittest
from collections import Counter
from types import SimpleNamespace

import grimace._south_star1.writer_frontier as writer_frontier_module
import grimace._south_star1.writer_transitions as writer_transitions
from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.errors import SouthStarErrorKind
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.writer_events import WriterRingEndpointEmitted
from grimace._south_star1.writer_events import WriterRingLabelAllocated
from grimace._south_star1.writer_capabilities import _WriterExecutionCapabilityKind
from grimace._south_star1.writer_frontier import WriterFrontierCursor
from grimace._south_star1.writer_frontier import _checked_writer_frontier_branch_supports
from grimace._south_star1.writer_frontier import _checked_writer_frontier_schedule_outcome
from grimace._south_star1.writer_frontier import _count_checked_writer_frontier_branch_completions
from grimace._south_star1.writer_frontier import _writer_frontier_diagnostics
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_graph_obligations import WriterBoundaryOwnerKind
from grimace._south_star1.writer_graph_obligations import WriterResidualAttachmentActionKind
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
from grimace._south_star1.writer_state import WriterAtomFrame
from grimace._south_star1.writer_state import WriterBranchFrame
from grimace._south_star1.writer_state import WriterPolicyState
from grimace._south_star1.writer_state import WriterRingState
from grimace._south_star1.writer_state import WriterState
from grimace._south_star1.writer_state import WriterStereoState
from grimace._south_star1.writer_state import writer_state_key
from tests.south_star1.helpers import cco_facts
from tests.south_star1.helpers import cyclopropane_facts


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

    def test_live_branch_return_closure_candidate_partner(self) -> None:
        state = _branch_return_closure_candidate_state(live_partner=True)

        self.assertEqual(
            writer_transitions._live_branch_return_closure_candidate_partner(
                state,
                active_atom=AtomId(0),
                left=AtomId(0),
                right=AtomId(1),
            ),
            AtomId(1),
        )

    def test_frozen_closure_candidate_partner_is_not_live(self) -> None:
        state = _branch_return_closure_candidate_state(live_partner=False)

        self.assertIsNone(
            writer_transitions._live_branch_return_closure_candidate_partner(
                state,
                active_atom=AtomId(0),
                left=AtomId(0),
                right=AtomId(1),
            )
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
        self.assertIsInstance(support.events[0], WriterRingLabelAllocated)
        self.assertIsInstance(support.events[1], WriterRingEndpointEmitted)
        self.assertEqual(
            support.successor_cursor.weighted_states,
            ((support.successor_state, 1),),
        )
        self.assertGreater(
            _count_checked_writer_frontier_branch_completions(
                prepared,
                support.successor_cursor,
            ),
            0,
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


if __name__ == "__main__":
    unittest.main()
