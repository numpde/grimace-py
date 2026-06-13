"""Tests for the writer-shaped state/frontier MVP."""

from __future__ import annotations

import ast
import contextlib
import inspect
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import grimace._south_star1.writer_frontier as writer_frontier_module
import grimace._south_star1.writer_snapshot as writer_snapshot
import grimace._south_star1.writer_state as writer_state_module
import grimace._south_star1.writer_transitions as writer_transitions
from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.errors import SouthStarErrorKind
from grimace._south_star1.facts import BondOrder
from grimace._south_star1.facts import ComponentFacts
from grimace._south_star1.facts import LigandKind
from grimace._south_star1.facts import LigandOccurrence
from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import ComponentId
from grimace._south_star1.ids import OccurrenceId
from grimace._south_star1.policy import AnnotationMode
from grimace._south_star1.policy import AtomTextChoice
from grimace._south_star1.policy import AtomTextDomain
from grimace._south_star1.policy import RingLabel
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.policy import SmilesPolicy
from grimace._south_star1.policy import TetraToken
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import enumerate_prepared_stereo_support
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.writer_frontier import WriterFrontierCursor
from grimace._south_star1.writer_frontier import count_writer_cursor_completions
from grimace._south_star1.writer_frontier import count_writer_frontier_support
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_frontier import initial_writer_transition_frontier_cursor
from grimace._south_star1.writer_frontier import iter_writer_frontier_support
from grimace._south_star1.writer_frontier import writer_frontier_choices
from grimace._south_star1.writer_graph_obligations import WriterBoundaryOwnerKind
from grimace._south_star1.writer_graph_obligations import WriterEdgeObligationKind
from grimace._south_star1.writer_graph_obligations import WriterResidualAttachmentActionKind
from grimace._south_star1.writer_state import ComponentCursor
from grimace._south_star1.writer_state import ObligationState
from grimace._south_star1.writer_state import PendingWriterEntry
from grimace._south_star1.writer_state import WriterAtomFrame
from grimace._south_star1.writer_state import WriterClosedClosure
from grimace._south_star1.writer_state import WriterClosureLabel
from grimace._south_star1.writer_state import WriterOpenClosureEndpoint
from grimace._south_star1.writer_state import WriterPolicyState
from grimace._south_star1.writer_state import WriterRingLabelState
from grimace._south_star1.writer_state import WriterRingState
from grimace._south_star1.writer_state import WriterState
from grimace._south_star1.writer_state import WriterStateKey
from grimace._south_star1.writer_state import writer_state_from_key
from grimace._south_star1.writer_state import writer_state_key
from grimace._south_star1.writer_state import writer_state_key_sort_tuple
from grimace._south_star1.writer_stereo import empty_writer_stereo_state
from tests.south_star1.helpers import atom
from tests.south_star1.helpers import bond
from tests.south_star1.helpers import cco_facts
from tests.south_star1.helpers import cyclopropane_facts
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import single_bond
from tests.south_star1.helpers import tetrahedral_facts


REPO_ROOT = Path(__file__).resolve().parents[2]
SOUTH_STAR1_ROOT = REPO_ROOT / "python" / "grimace" / "_south_star1"


class WriterStateKernelTest(unittest.TestCase):
    def _closure_policy_for_outcome(
        self,
        active_atom: AtomId,
    ) -> tuple[
        writer_transitions._WriterActiveEmittedGraphPolicyDecision,
        writer_transitions._WriterActiveEmittedScheduleDecision,
    ]:
        closure_decision, _, _ = self._test_closure_endpoint_decision(
            active_atom=active_atom,
            pair_survives=True,
            open_survives=False,
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .CLOSURE_ENDPOINT
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
        )
        active_decision = writer_transitions._active_emitted_closure_decision(
            closure_decision,
            graph_policy_decision=policy,
        )

        return policy, active_decision

    def _blocked_child_active_emitted_outcome(
        self,
        active_atom: AtomId,
    ) -> writer_transitions._WriterActiveEmittedScheduleOutcome:
        child_blocker = writer_transitions._WriterChildObligationBlocker(
            kind=(
                writer_transitions._WriterChildObligationBlockerKind
                .MULTI_INCIDENCE_RESIDUAL_ATTACHMENT
            ),
            atom=active_atom,
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(child_blocker,),
            child_obligations=(),
            scheduled_actions=(),
        )
        empty_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=empty_batch,
            open_batch=empty_batch,
            surviving_emissions=(),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .BLOCKED_CHILD
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )

        return writer_transitions._WriterActiveEmittedScheduleOutcome(
            kind=writer_transitions._WriterActiveEmittedScheduleOutcomeKind.BLOCKED,
            graph_policy_decision=policy,
        )

    def _dead_closure_open_active_emitted_outcome(
        self,
        active_atom: AtomId,
    ) -> tuple[
        writer_transitions._WriterActiveEmittedGraphPolicyDecision,
        writer_transitions._WriterActiveEmittedScheduleOutcome,
    ]:
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        open_emission = writer_transitions._WriterScheduledActionEmission(
            action=open_action,
            transitions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(open_emission,),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD_AFTER_DEAD_CLOSURE_OPEN
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(child_action,),
            emissions=(),
            surviving_emissions=(),
        )
        active_decision = writer_transitions._active_emitted_child_decision(
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
            child_batch=child_batch,
            graph_policy_decision=policy,
        )
        active_outcome = writer_transitions._WriterActiveEmittedScheduleOutcome(
            kind=writer_transitions._WriterActiveEmittedScheduleOutcomeKind.SCHEDULED,
            graph_policy_decision=policy,
            schedule_decision=active_decision,
        )

        return policy, active_outcome

    def _cyclic_active_child_scheduled_outcome(
        self,
        active_atom: AtomId,
    ) -> writer_transitions._WriterActiveEmittedScheduleOutcome:
        closure_decision, _, _ = self._test_closure_endpoint_decision(
            active_atom=active_atom,
            pair_survives=False,
            open_survives=False,
        )
        cyclic_action = self._test_child_action(
            active_atom,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(cyclic_action.child_obligation,),
            scheduled_actions=(cyclic_action,),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD_AFTER_DEAD_CLOSURE_OPEN
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )
        cyclic_emission = writer_transitions._WriterScheduledActionEmission(
            action=cyclic_action,
            transitions=(object(),),  # type: ignore[arg-type]
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(cyclic_action,),
            emissions=(cyclic_emission,),
            surviving_emissions=(cyclic_emission,),
        )
        decision = writer_transitions._active_emitted_child_decision(
            closure_decision,
            child_surface,
            child_batch,
            graph_policy_decision=policy,
        )

        return writer_transitions._WriterActiveEmittedScheduleOutcome(
            kind=writer_transitions._WriterActiveEmittedScheduleOutcomeKind.SCHEDULED,
            graph_policy_decision=policy,
            schedule_decision=decision,
        )

    def _dead_closure_open_graph_policy_decision_for_owner_scope(
        self,
        *,
        closure_owner_kind: WriterBoundaryOwnerKind | None,
        child_owner_kind: WriterBoundaryOwnerKind | None,
    ) -> writer_transitions._WriterActiveEmittedGraphPolicyDecision:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=closure_owner_kind,  # type: ignore[arg-type]
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        open_emission = writer_transitions._WriterScheduledActionEmission(
            action=open_action,
            transitions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(open_emission,),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=child_owner_kind,
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            return_value=child_surface,
        ):
            return writer_transitions._active_emitted_graph_policy_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

    def _test_choice_snapshot_entry(
        self,
        emitted_text: str,
        *,
        successor_atom: AtomId,
        parent_weight: int = 1,
    ) -> writer_frontier_module._WriterFrontierChoiceSnapshotEntry:
        parent_key = writer_state_key(_raw_initial_state(AtomId(0)))
        successor_key = writer_state_key(_raw_initial_state(successor_atom))
        family = writer_transitions._WriterGraphPolicyActionFamily.ACYCLIC_TREE_ENTRY
        entry = writer_frontier_module._WriterFrontierNextTokenEntry(
            emitted_text=emitted_text,
            supports=(
                writer_frontier_module._WriterFrontierNextTokenSupport(
                    state_key=parent_key,
                    parent_weight=parent_weight,
                    schedule_support=SimpleNamespace(
                        emitted_text=emitted_text,
                        graph_action_surface=object(),
                        policy_family=family,
                    ),  # type: ignore[arg-type]
                    successor_key=successor_key,
                ),
            ),
        )

        return writer_frontier_module._WriterFrontierChoiceSnapshotEntry(
            next_token_entry=entry,
            successor=WriterFrontierCursor(
                weighted_states=tuple(entry.weighted_successors.items())
            ),
        )

    def _test_frontier_next_token_support(
        self,
        *,
        emitted_text: str,
        successor_atom: AtomId,
        policy_family: writer_transitions._WriterGraphPolicyActionFamily,
        residual_key: (
            writer_transitions._WriterResidualAttachmentPolicyKey | None
        ) = None,
        parent_weight: int = 1,
    ) -> writer_frontier_module._WriterFrontierNextTokenSupport:
        return writer_frontier_module._WriterFrontierNextTokenSupport(
            state_key=writer_state_key(_raw_initial_state(AtomId(0))),
            parent_weight=parent_weight,
            schedule_support=SimpleNamespace(
                emitted_text=emitted_text,
                graph_action_surface=SimpleNamespace(
                    residual_attachment_policy_key=residual_key,
                ),
                policy_family=policy_family,
            ),  # type: ignore[arg-type]
            successor_key=writer_state_key(_raw_initial_state(successor_atom)),
        )

    def _test_residual_policy_group(
        self,
        key: writer_transitions._WriterResidualAttachmentPolicyKey,
    ) -> writer_transitions._WriterResidualAttachmentPolicyGroup:
        return writer_transitions._WriterResidualAttachmentPolicyGroup(
            key=key,
            surfaces=(),
        )

    def _test_residual_policy_group_with_owner_scope(
        self,
        key: writer_transitions._WriterResidualAttachmentPolicyKey,
        *,
        closure_owner_kind: WriterBoundaryOwnerKind | None,
        child_owner_kind: WriterBoundaryOwnerKind | None,
    ) -> writer_transitions._WriterResidualAttachmentPolicyGroup:
        closure_open = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.OPEN_CLOSURE_ENDPOINT,
            active_atom=key.active_atom,
            attachment_id=key.attachment_id,
            owner_kind=closure_owner_kind,
        )
        cyclic_tree = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.ENTER_INLINE_CHILD,
            active_atom=key.active_atom,
            attachment_id=key.attachment_id,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
            owner_kind=child_owner_kind,
        )

        return writer_transitions._WriterResidualAttachmentPolicyGroup(
            key=key,
            surfaces=(closure_open, cyclic_tree),
        )

    def _test_closure_pair_action(
        self,
        active_atom: AtomId,
    ) -> writer_transitions._WriterScheduledAction:
        label = WriterClosureLabel(value=1, text="1")
        endpoint = WriterOpenClosureEndpoint(
            bond=BondId(1),
            first_atom=AtomId(2),
            second_atom=active_atom,
            label=label,
            first_endpoint_text="1",
            first_endpoint_bond_text="",
        )
        closure = WriterClosedClosure(
            bond=BondId(1),
            first_atom=AtomId(2),
            second_atom=active_atom,
            label=label,
            first_endpoint_text="1",
            second_endpoint_text="1",
            first_endpoint_bond_text="",
            second_endpoint_bond_text="",
        )

        return writer_transitions._pair_closure_endpoint_action(
            active_atom,
            writer_transitions._WriterClosurePairObligation(
                endpoint=endpoint,
                closure=closure,
            ),
        )

    def _test_closure_open_action(
        self,
        active_atom: AtomId,
        *,
        attachment_id: int = 7,
    ) -> writer_transitions._WriterScheduledAction:
        label = WriterClosureLabel(value=2, text="2")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(2),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=attachment_id,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )

        return writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )

    def _test_closure_endpoint_decision(
        self,
        *,
        active_atom: AtomId = AtomId(0),
        pair_survives: bool,
        open_survives: bool,
    ) -> tuple[
        writer_transitions._WriterClosureEndpointScheduleDecision,
        writer_transitions._WriterScheduledActionEmission,
        writer_transitions._WriterScheduledActionEmission,
    ]:
        pair_action = self._test_closure_pair_action(active_atom)
        open_action = self._test_closure_open_action(active_atom)
        pair_emission = writer_transitions._WriterScheduledActionEmission(
            action=pair_action,
            transitions=(object(),) if pair_survives else (),  # type: ignore[arg-type]
        )
        open_emission = writer_transitions._WriterScheduledActionEmission(
            action=open_action,
            transitions=(object(),) if open_survives else (),  # type: ignore[arg-type]
        )
        pair_surviving = (pair_emission,) if pair_survives else ()
        open_surviving = (open_emission,) if open_survives else ()
        decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(pair_action,),
                emissions=(pair_emission,),
                surviving_emissions=pair_surviving,
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(open_emission,),
                surviving_emissions=open_surviving,
            ),
            surviving_emissions=(
                *pair_surviving,
                *open_surviving,
            ),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(pair_action,),
                open_actions=(open_action,),
            ),
        )

        return decision, pair_emission, open_emission

    def _test_closure_endpoint_decision_for_actions(
        self,
        active_atom: AtomId,
        *,
        pair_actions: tuple[writer_transitions._WriterScheduledAction, ...] = (),
        open_actions: tuple[writer_transitions._WriterScheduledAction, ...] = (),
    ) -> writer_transitions._WriterClosureEndpointScheduleDecision:
        pair_emissions = tuple(
            writer_transitions._WriterScheduledActionEmission(
                action=action,
                transitions=(),
            )
            for action in pair_actions
        )
        open_emissions = tuple(
            writer_transitions._WriterScheduledActionEmission(
                action=action,
                transitions=(),
            )
            for action in open_actions
        )

        return writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=pair_actions,
                emissions=pair_emissions,
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=open_actions,
                emissions=open_emissions,
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=pair_actions,
                open_actions=open_actions,
            ),
        )

    def _test_residual_cyclic_policy_inputs(
        self,
        *,
        active_atom: AtomId = AtomId(0),
        attachment_id: int = 11,
        closure_owner_kind: WriterBoundaryOwnerKind | None = (
            WriterBoundaryOwnerKind.ACTIVE_ATOM
        ),
        child_owner_kind: WriterBoundaryOwnerKind | None = (
            WriterBoundaryOwnerKind.ACTIVE_ATOM
        ),
        include_closure_emission: bool = True,
        child_attachment_action_kind: WriterResidualAttachmentActionKind = (
            WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
        ),
    ) -> tuple[
        writer_transitions._WriterClosureEndpointScheduleDecision,
        writer_transitions._WriterActiveChildScheduleSurface,
    ]:
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=attachment_id,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=closure_owner_kind,  # type: ignore[arg-type]
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        open_emissions = (
            writer_transitions._WriterScheduledActionEmission(
                action=open_action,
                transitions=(),
            ),
        ) if include_closure_emission else ()
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=open_emissions,
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=child_owner_kind,
            attachment_id=attachment_id,
            attachment_action_kind=child_attachment_action_kind,
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )

        return closure_decision, child_surface

    def _test_child_action(
        self,
        active_atom: AtomId,
        *,
        attachment_id: int = 7,
        attachment_action_kind: WriterResidualAttachmentActionKind | None = None,
    ) -> writer_transitions._WriterScheduledAction:
        child = writer_transitions._WriterChildObligation(
            bond=BondId(1),
            child=AtomId(2),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=attachment_id,
            attachment_action_kind=attachment_action_kind,
        )

        return writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )

    def _test_choice_snapshot_entry_from_supports(
        self,
        emitted_text: str,
        supports: tuple[
            writer_frontier_module._WriterFrontierNextTokenSupport,
            ...,
        ],
    ) -> writer_frontier_module._WriterFrontierChoiceSnapshotEntry:
        entry = writer_frontier_module._WriterFrontierNextTokenEntry(
            emitted_text=emitted_text,
            supports=supports,
        )

        return writer_frontier_module._WriterFrontierChoiceSnapshotEntry(
            next_token_entry=entry,
            successor=WriterFrontierCursor(
                weighted_states=tuple(entry.weighted_successors.items())
            ),
        )

    def _test_frontier_choice_snapshot(
        self,
        choices: tuple[
            writer_frontier_module._WriterFrontierChoiceSnapshotEntry,
            ...,
        ],
    ) -> writer_frontier_module._WriterFrontierChoiceSnapshot:
        return writer_frontier_module._WriterFrontierChoiceSnapshot(
            schedule_outcome=writer_frontier_module._WriterFrontierScheduleOutcome(
                state_outcomes=(),
                terminal_by_key=Counter(),
                grouped_by_text={},
                weighted_by_text={},
            ),
            terminal=None,
            choices=choices,
        )

    def _test_frontier_choice_snapshot_with_active_policy(
        self,
        active_outcome: writer_transitions._WriterActiveEmittedScheduleOutcome,
    ) -> writer_frontier_module._WriterFrontierChoiceSnapshot:
        top_outcome = writer_transitions._WriterTopLevelScheduleOutcome(
            kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.SCHEDULED,
            schedule_decision=writer_transitions._top_level_active_emitted_decision(
                active_outcome.schedule_decision,
            ),
            active_emitted_outcome=active_outcome,
        )
        state_outcome = writer_frontier_module._WriterFrontierStateScheduleOutcome(
            state_key=writer_state_key(_raw_initial_state(AtomId(0))),
            parent_weight=1,
            finalized_state_key=None,
            schedule_outcome=top_outcome,
        )
        schedule_outcome = writer_frontier_module._WriterFrontierScheduleOutcome(
            state_outcomes=(state_outcome,),
            terminal_by_key=Counter(),
            grouped_by_text={},
            weighted_by_text={},
        )

        return writer_frontier_module._WriterFrontierChoiceSnapshot(
            schedule_outcome=schedule_outcome,
            terminal=None,
            choices=(),
        )

    def _test_active_outcome_for_graph_policy(
        self,
        policy: writer_transitions._WriterActiveEmittedGraphPolicyDecision,
    ) -> writer_transitions._WriterActiveEmittedScheduleOutcome:
        if policy.graph_policy_blocked:
            return writer_transitions._WriterActiveEmittedScheduleOutcome(
                kind=(
                    writer_transitions
                    ._WriterActiveEmittedScheduleOutcomeKind
                    .BLOCKED
                ),
                graph_policy_decision=policy,
            )

        if (
            policy.kind
            is (
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .CLOSURE_ENDPOINT
            )
        ):
            active_decision = writer_transitions._active_emitted_closure_decision(
                policy.closure_endpoint_decision,
                graph_policy_decision=policy,
            )
        else:
            child_surface = policy.child_schedule_surface
            if child_surface is None:
                raise AssertionError("test policy is missing child surface")

            child_batch = (
                writer_transitions._WriterScheduledActionEmissionBatch(
                    actions=child_surface.scheduled_actions,
                    emissions=(),
                    surviving_emissions=(),
                )
            )
            active_decision = writer_transitions._active_emitted_child_decision(
                closure_endpoint_decision=policy.closure_endpoint_decision,
                child_schedule_surface=child_surface,
                child_batch=child_batch,
                graph_policy_decision=policy,
            )

        return writer_transitions._WriterActiveEmittedScheduleOutcome(
            kind=(
                writer_transitions
                ._WriterActiveEmittedScheduleOutcomeKind
                .SCHEDULED
            ),
            graph_policy_decision=policy,
            schedule_decision=active_decision,
        )

    def _test_frontier_schedule_outcome_for_graph_policies(
        self,
        policies: tuple[
            writer_transitions._WriterActiveEmittedGraphPolicyDecision,
            ...,
        ],
        *,
        include_non_policy_state: bool = False,
    ) -> writer_frontier_module._WriterFrontierScheduleOutcome:
        state_outcomes = []

        for index, policy in enumerate(policies):
            active_outcome = self._test_active_outcome_for_graph_policy(policy)

            if active_outcome.kind is (
                writer_transitions
                ._WriterActiveEmittedScheduleOutcomeKind
                .SCHEDULED
            ):
                top_outcome = writer_transitions._WriterTopLevelScheduleOutcome(
                    kind=(
                        writer_transitions
                        ._WriterTopLevelScheduleOutcomeKind
                        .SCHEDULED
                    ),
                    schedule_decision=(
                        writer_transitions
                        ._top_level_active_emitted_decision(
                            active_outcome.schedule_decision,
                        )
                    ),
                    active_emitted_outcome=active_outcome,
                )
            else:
                top_outcome = writer_transitions._WriterTopLevelScheduleOutcome(
                    kind=(
                        writer_transitions
                        ._WriterTopLevelScheduleOutcomeKind
                        .BLOCKED
                    ),
                    active_emitted_outcome=active_outcome,
                )

            state_outcomes.append(
                writer_frontier_module._WriterFrontierStateScheduleOutcome(
                    state_key=writer_state_key(_raw_initial_state(AtomId(index))),
                    parent_weight=1,
                    finalized_state_key=None,
                    schedule_outcome=top_outcome,
                )
            )

        if include_non_policy_state:
            action = writer_transitions._emit_root_atom_action(AtomId(9))
            batch = writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(action,),
                emissions=(),
                surviving_emissions=(),
            )
            top_outcome = writer_transitions._WriterTopLevelScheduleOutcome(
                kind=(
                    writer_transitions
                    ._WriterTopLevelScheduleOutcomeKind
                    .SCHEDULED
                ),
                schedule_decision=(
                    writer_transitions._top_level_actions_decision(batch)
                ),
            )
            state_outcomes.append(
                writer_frontier_module._WriterFrontierStateScheduleOutcome(
                    state_key=writer_state_key(_raw_initial_state(AtomId(9))),
                    parent_weight=1,
                    finalized_state_key=None,
                    schedule_outcome=top_outcome,
                )
            )

        return writer_frontier_module._WriterFrontierScheduleOutcome(
            state_outcomes=tuple(state_outcomes),
            terminal_by_key=Counter(),
            grouped_by_text={},
            weighted_by_text={},
        )

    def _test_blocked_frontier_choice_snapshot(
        self,
        cursor: WriterFrontierCursor,
    ) -> writer_frontier_module._WriterFrontierChoiceSnapshot:
        active_outcome = self._blocked_child_active_emitted_outcome(AtomId(0))
        blocked_top_level_outcome = (
            writer_transitions._WriterTopLevelScheduleOutcome(
                kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.BLOCKED,
                active_emitted_outcome=active_outcome,
            )
        )
        state_outcome = writer_frontier_module._WriterFrontierStateScheduleOutcome(
            state_key=cursor.weighted_states[0][0],
            parent_weight=cursor.weighted_states[0][1],
            finalized_state_key=None,
            schedule_outcome=blocked_top_level_outcome,
        )
        blocked_outcome = writer_frontier_module._WriterFrontierScheduleOutcome(
            state_outcomes=(state_outcome,),
            terminal_by_key=Counter(),
            grouped_by_text={},
            weighted_by_text={},
        )

        return writer_frontier_module._WriterFrontierChoiceSnapshot(
            schedule_outcome=blocked_outcome,
            terminal=None,
            choices=(),
        )

    def _test_writer_snapshot_advance_step(
        self,
        source_snapshot: writer_snapshot.WriterSearchSnapshot,
        *,
        emitted_text: str,
        prepared,
    ) -> writer_snapshot._WriterSnapshotAdvanceOutcome:
        choice_snapshot = writer_snapshot._writer_frontier_choice_snapshot_from_snapshot(
            source_snapshot,
            prepared=prepared,
            include_counts=False,
        )
        choice = choice_snapshot.choices[0]
        advanced_snapshot = (
            writer_snapshot
            ._writer_search_snapshot_with_cursor_after_emitted_text(
                source_snapshot,
                prepared=prepared,
                cursor=choice.successor,
            )
        )

        return writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceOutcomeKind.ADVANCED,
            source_snapshot=source_snapshot,
            emitted_text=emitted_text,
            choice_snapshot=choice_snapshot,
            choice=choice,
            advanced_snapshot=advanced_snapshot,
        )

    def test_writer_shaped_acyclic_support_uses_writer_frontier(self) -> None:
        prepared = _prepare(cco_facts())

        with _forbidden_exhaustive_routes():
            support = enumerate_prepared_stereo_support(
                prepared=prepared,
                runtime_options=_writer_options(),
            )

        self.assertEqual(
            support.strings,
            ("C(C)O", "C(O)C", "CCO", "OCC"),
        )
        self.assertEqual(support.distinct_count, 4)
        self.assertEqual(support.witness_count, 4)

    def test_writer_frontier_groups_same_emitted_text(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())

        choices = writer_frontier_choices(prepared, cursor)

        self.assertIsNone(choices.terminal)
        self.assertEqual(tuple(choice.emitted_text for choice in choices.choices), ("C",))
        self.assertEqual(choices.choices[0].immediate_multiplicity, 2)
        self.assertEqual(choices.choices[0].support_count, 1)
        self.assertEqual(choices.choices[0].completion_count, 2)
        self.assertEqual(len(choices.choices[0].successor.support_state.states), 2)
        self.assertEqual(
            sum(weight for _, weight in choices.choices[0].successor.weighted_states),
            2,
        )

    def test_writer_frontier_choices_use_schedule_outcome_not_legal_next_token_helper(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())

        with patch.object(
            writer_frontier_module,
            "legal_writer_transitions",
            side_effect=AssertionError("writer_frontier used flattened transitions"),
            create=True,
        ), patch(
            "grimace._south_star1.writer_transitions.legal_writer_transitions",
            side_effect=AssertionError(
                "writer_frontier used public flattened transitions"
            ),
        ), patch(
            "grimace._south_star1.writer_transitions._legal_writer_next_token_frontier",
            side_effect=AssertionError(
                "writer_frontier used legal next-token frontier helper"
            ),
        ), patch(
            "grimace._south_star1.writer_frontier._legal_writer_schedule_outcome",
            wraps=writer_frontier_module._legal_writer_schedule_outcome,
        ) as schedule_outcome:
            choices = writer_frontier_choices(prepared, cursor)

        self.assertEqual(tuple(choice.emitted_text for choice in choices.choices), ("C",))
        self.assertGreater(schedule_outcome.call_count, 0)

    def test_count_writer_cursor_completions_use_schedule_outcome_not_legal_next_token_helper(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())

        with patch(
            "grimace._south_star1.writer_transitions._legal_writer_next_token_frontier",
            side_effect=AssertionError(
                "completion count used legal next-token frontier helper"
            ),
        ), patch(
            "grimace._south_star1.writer_frontier._legal_writer_schedule_outcome",
            wraps=writer_frontier_module._legal_writer_schedule_outcome,
        ) as schedule_outcome:
            count = count_writer_cursor_completions(prepared, cursor)

        self.assertEqual(count, 2)
        self.assertGreater(schedule_outcome.call_count, 0)

    def test_writer_frontier_residual_attachment_support_groups_preserve_order(self) -> None:
        key_7 = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=7,
        )
        key_8 = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=8,
        )
        support_a = self._test_frontier_next_token_support(
            emitted_text="C",
            successor_atom=AtomId(1),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CYCLIC_TREE_ENTRY
            ),
            residual_key=key_7,
        )
        support_b = self._test_frontier_next_token_support(
            emitted_text="",
            successor_atom=AtomId(2),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .FINISH_ACTIVE
            ),
        )
        support_c = self._test_frontier_next_token_support(
            emitted_text="1",
            successor_atom=AtomId(3),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CLOSURE_OPEN
            ),
            residual_key=key_7,
        )
        support_d = self._test_frontier_next_token_support(
            emitted_text="N",
            successor_atom=AtomId(4),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .ACYCLIC_TREE_ENTRY
            ),
            residual_key=key_8,
        )

        groups = (
            writer_frontier_module
            ._writer_frontier_residual_attachment_support_groups_from_supports(
                (support_a, support_b, support_c, support_d)
            )
        )

        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0].key, key_7)
        self.assertEqual(groups[0].supports, (support_a, support_c))
        self.assertEqual(groups[0].cyclic_tree_entry_supports, (support_a,))
        self.assertEqual(groups[0].closure_open_supports, (support_c,))
        self.assertEqual(groups[0].tree_entry_supports, (support_a,))
        self.assertEqual(
            groups[0].policy_families,
            (
                writer_transitions._WriterGraphPolicyActionFamily.CYCLIC_TREE_ENTRY,
                writer_transitions._WriterGraphPolicyActionFamily.CLOSURE_OPEN,
            ),
        )
        self.assertEqual(groups[1].key, key_8)
        self.assertEqual(groups[1].supports, (support_d,))
        self.assertEqual(groups[1].acyclic_tree_entry_supports, (support_d,))
        self.assertEqual(groups[1].tree_entry_supports, (support_d,))

    def test_writer_frontier_next_token_entry_exposes_residual_attachment_support_groups(self) -> None:
        key = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=7,
        )
        support_a = self._test_frontier_next_token_support(
            emitted_text="C",
            successor_atom=AtomId(1),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CYCLIC_TREE_ENTRY
            ),
            residual_key=key,
        )
        support_b = self._test_frontier_next_token_support(
            emitted_text="C",
            successor_atom=AtomId(2),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CLOSURE_OPEN
            ),
            residual_key=key,
        )
        entry = writer_frontier_module._WriterFrontierNextTokenEntry(
            emitted_text="C",
            supports=(support_a, support_b),
        )

        self.assertEqual(
            entry.residual_attachment_support_groups[0].supports,
            entry.supports,
        )
        self.assertEqual(
            entry.policy_families,
            (support_a.policy_family, support_b.policy_family),
        )

    def test_writer_frontier_choice_snapshot_entry_exposes_residual_attachment_support_groups(self) -> None:
        key = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=7,
        )
        support = self._test_frontier_next_token_support(
            emitted_text="C",
            successor_atom=AtomId(1),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CYCLIC_TREE_ENTRY
            ),
            residual_key=key,
        )
        next_token_entry = writer_frontier_module._WriterFrontierNextTokenEntry(
            emitted_text="C",
            supports=(support,),
        )
        choice_entry = writer_frontier_module._WriterFrontierChoiceSnapshotEntry(
            next_token_entry=next_token_entry,
            successor=WriterFrontierCursor(
                weighted_states=tuple(
                    next_token_entry.weighted_successors.items()
                )
            ),
        )

        self.assertEqual(
            choice_entry.residual_attachment_support_groups,
            next_token_entry.residual_attachment_support_groups,
        )

    def test_writer_frontier_residual_attachment_evidence_groups_merge_policy_and_support_evidence(self) -> None:
        key_7 = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=7,
        )
        key_8 = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=8,
        )
        resolved_7 = self._test_residual_policy_group(key_7)
        support_dead_7 = self._test_residual_policy_group(key_7)
        unresolved_8 = self._test_residual_policy_group(key_8)
        cyclic_support_7 = self._test_frontier_next_token_support(
            emitted_text="C",
            successor_atom=AtomId(1),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CYCLIC_TREE_ENTRY
            ),
            residual_key=key_7,
        )
        closure_support_8 = self._test_frontier_next_token_support(
            emitted_text="1",
            successor_atom=AtomId(2),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CLOSURE_OPEN
            ),
            residual_key=key_8,
        )
        support_group_7 = (
            writer_frontier_module._WriterFrontierResidualAttachmentSupportGroup(
                key=key_7,
                supports=(cyclic_support_7,),
            )
        )
        support_group_8 = (
            writer_frontier_module._WriterFrontierResidualAttachmentSupportGroup(
                key=key_8,
                supports=(closure_support_8,),
            )
        )

        groups = (
            writer_frontier_module
            ._writer_frontier_residual_attachment_evidence_groups(
                resolved_policy_groups=(resolved_7,),
                support_dead_closure_open_vs_cyclic_tree_entry_policy_groups=(
                    support_dead_7,
                ),
                unsupported_owner_scope_policy_groups=(),
                unresolved_policy_groups=(unresolved_8,),
                selected_support_groups=(support_group_7, support_group_8),
            )
        )

        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0].key, key_7)
        self.assertEqual(groups[0].resolved_policy_groups, (resolved_7,))
        self.assertEqual(
            (
                groups[0]
                .support_dead_closure_open_vs_cyclic_tree_entry_policy_groups
            ),
            (support_dead_7,),
        )
        self.assertEqual(groups[0].selected_support_groups, (support_group_7,))
        self.assertEqual(
            groups[0].selected_cyclic_tree_entry_supports,
            (cyclic_support_7,),
        )
        self.assertEqual(
            groups[0].selected_policy_families,
            (
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CYCLIC_TREE_ENTRY,
            ),
        )
        self.assertTrue(groups[0].has_resolved_policy_evidence)
        self.assertTrue(groups[0].has_support_dead_closure_open_evidence)
        self.assertTrue(groups[0].has_selected_support_evidence)
        self.assertFalse(groups[0].has_unresolved_policy_evidence)
        self.assertEqual(groups[1].key, key_8)
        self.assertEqual(groups[1].unresolved_policy_groups, (unresolved_8,))
        self.assertEqual(
            groups[1].selected_closure_open_supports,
            (closure_support_8,),
        )
        self.assertTrue(groups[1].has_unresolved_policy_evidence)

    def test_writer_frontier_residual_attachment_evidence_groups_include_support_only_keys(self) -> None:
        key = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=7,
        )
        support = self._test_frontier_next_token_support(
            emitted_text="C",
            successor_atom=AtomId(1),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .ACYCLIC_TREE_ENTRY
            ),
            residual_key=key,
        )
        support_group = (
            writer_frontier_module._WriterFrontierResidualAttachmentSupportGroup(
                key=key,
                supports=(support,),
            )
        )

        groups = (
            writer_frontier_module
            ._writer_frontier_residual_attachment_evidence_groups(
                resolved_policy_groups=(),
                support_dead_closure_open_vs_cyclic_tree_entry_policy_groups=(),
                unsupported_owner_scope_policy_groups=(),
                unresolved_policy_groups=(),
                selected_support_groups=(support_group,),
            )
        )

        self.assertEqual(tuple(group.key for group in groups), (key,))
        self.assertTrue(groups[0].has_selected_support_evidence)
        self.assertFalse(groups[0].has_resolved_policy_evidence)
        self.assertFalse(groups[0].has_unresolved_policy_evidence)

    def test_writer_frontier_residual_attachment_evidence_groups_preserve_first_seen_key_order(self) -> None:
        key_7 = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=7,
        )
        key_8 = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=8,
        )
        key_9 = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=9,
        )
        support_9 = self._test_frontier_next_token_support(
            emitted_text="N",
            successor_atom=AtomId(9),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .ACYCLIC_TREE_ENTRY
            ),
            residual_key=key_9,
        )
        support_7 = self._test_frontier_next_token_support(
            emitted_text="C",
            successor_atom=AtomId(7),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CLOSURE_OPEN
            ),
            residual_key=key_7,
        )

        groups = (
            writer_frontier_module
            ._writer_frontier_residual_attachment_evidence_groups(
                resolved_policy_groups=(
                    self._test_residual_policy_group(key_8),
                ),
                support_dead_closure_open_vs_cyclic_tree_entry_policy_groups=(),
                unsupported_owner_scope_policy_groups=(),
                unresolved_policy_groups=(
                    self._test_residual_policy_group(key_7),
                ),
                selected_support_groups=(
                    writer_frontier_module
                    ._WriterFrontierResidualAttachmentSupportGroup(
                        key=key_9,
                        supports=(support_9,),
                    ),
                    writer_frontier_module
                    ._WriterFrontierResidualAttachmentSupportGroup(
                        key=key_7,
                        supports=(support_7,),
                    ),
                ),
            )
        )

        self.assertEqual(tuple(group.key for group in groups), (key_8, key_7, key_9))

    def test_writer_frontier_residual_attachment_evidence_group_exposes_residual_cyclic_decisions(self) -> None:
        support_dead_closure, support_dead_child = (
            self._test_residual_cyclic_policy_inputs()
        )
        support_dead_decision = (
            writer_transitions._residual_cyclic_policy_decision(
                support_dead_closure,
                support_dead_child,
            )
        )
        missing_closure, missing_child = self._test_residual_cyclic_policy_inputs(
            include_closure_emission=False,
        )
        missing_decision = writer_transitions._residual_cyclic_policy_decision(
            missing_closure,
            missing_child,
        )
        unsupported_closure, unsupported_child = (
            self._test_residual_cyclic_policy_inputs(
                closure_owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
                child_owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
            )
        )
        unsupported_decision = (
            writer_transitions._residual_cyclic_policy_decision(
                unsupported_closure,
                unsupported_child,
            )
        )
        support_dead_group = (
            writer_frontier_module
            ._WriterFrontierResidualAttachmentEvidenceGroup(
                key=support_dead_decision.choice_groups[0].key,
                residual_cyclic_policy_decisions=(support_dead_decision,),
            )
        )
        mixed_group = (
            writer_frontier_module
            ._WriterFrontierResidualAttachmentEvidenceGroup(
                key=missing_decision.choice_groups[0].key,
                residual_cyclic_policy_decisions=(
                    missing_decision,
                    unsupported_decision,
                ),
            )
        )

        self.assertEqual(
            support_dead_group.residual_cyclic_policy_decisions,
            (support_dead_decision,),
        )
        self.assertEqual(
            support_dead_group.residual_cyclic_policy_kinds,
            (
                writer_transitions
                ._WriterResidualCyclicPolicyDecisionKind
                .ACTIVE_CHILD_AFTER_DEAD_CLOSURE_OPEN,
            ),
        )
        self.assertTrue(
            support_dead_group.has_residual_cyclic_policy_evidence
        )
        self.assertTrue(
            support_dead_group.has_residual_cyclic_support_dead_resolution
        )
        self.assertTrue(
            mixed_group.has_residual_cyclic_missing_evidence_blocker
        )
        self.assertTrue(
            mixed_group.has_residual_cyclic_unsupported_owner_scope_blocker
        )

    def test_writer_frontier_choice_residual_attachment_evidence_groups_join_matching_policy_evidence(self) -> None:
        key_7 = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=7,
        )
        key_8 = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=8,
        )
        resolved_7 = self._test_residual_policy_group(key_7)
        support_dead_7 = self._test_residual_policy_group(key_7)
        unresolved_8 = self._test_residual_policy_group(key_8)
        support_7 = self._test_frontier_next_token_support(
            emitted_text="C",
            successor_atom=AtomId(1),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CYCLIC_TREE_ENTRY
            ),
            residual_key=key_7,
        )
        choice = self._test_choice_snapshot_entry_from_supports(
            "C",
            (support_7,),
        )
        schedule_outcome = SimpleNamespace(
            resolved_residual_attachment_policy_groups=(resolved_7,),
            support_dead_closure_open_vs_cyclic_tree_entry_groups=(
                support_dead_7,
            ),
            unsupported_owner_scope_residual_attachment_policy_groups=(),
            unresolved_residual_attachment_policy_groups=(unresolved_8,),
        )

        groups = (
            writer_frontier_module
            ._writer_frontier_choice_residual_attachment_evidence_groups(
                choice=choice,
                schedule_outcome=schedule_outcome,  # type: ignore[arg-type]
            )
        )

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].key, key_7)
        self.assertEqual(groups[0].resolved_policy_groups, (resolved_7,))
        self.assertEqual(
            (
                groups[0]
                .support_dead_closure_open_vs_cyclic_tree_entry_policy_groups
            ),
            (support_dead_7,),
        )
        self.assertEqual(groups[0].unresolved_policy_groups, ())
        self.assertEqual(
            groups[0].selected_support_groups,
            choice.residual_attachment_support_groups,
        )

    def test_writer_frontier_schedule_outcome_evidence_groups_include_matching_residual_cyclic_decisions(self) -> None:
        key_7 = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=7,
        )
        support_7 = self._test_frontier_next_token_support(
            emitted_text="C",
            successor_atom=AtomId(1),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CYCLIC_TREE_ENTRY
            ),
            residual_key=key_7,
        )
        support_group_7 = (
            writer_frontier_module._WriterFrontierResidualAttachmentSupportGroup(
                key=key_7,
                supports=(support_7,),
            )
        )
        closure_decision, child_surface = (
            self._test_residual_cyclic_policy_inputs(attachment_id=7)
        )
        residual_decision = writer_transitions._residual_cyclic_policy_decision(
            closure_decision,
            child_surface,
        )
        none_closure, none_child = self._test_residual_cyclic_policy_inputs(
            attachment_id=8,
            child_attachment_action_kind=(
                WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY
            ),
        )
        none_decision = writer_transitions._residual_cyclic_policy_decision(
            none_closure,
            none_child,
        )

        groups = (
            writer_frontier_module
            ._writer_frontier_residual_attachment_evidence_groups(
                residual_cyclic_policy_decisions=(
                    residual_decision,
                    none_decision,
                ),
                resolved_policy_groups=(),
                support_dead_closure_open_vs_cyclic_tree_entry_policy_groups=(),
                unsupported_owner_scope_policy_groups=(),
                unresolved_policy_groups=(),
                selected_support_groups=(support_group_7,),
            )
        )

        self.assertEqual(tuple(group.key for group in groups), (key_7,))
        self.assertEqual(
            groups[0].residual_cyclic_policy_decisions,
            (residual_decision,),
        )

    def test_writer_frontier_choice_residual_attachment_evidence_groups_include_matching_residual_cyclic_decisions(self) -> None:
        key_7 = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=7,
        )
        support_7 = self._test_frontier_next_token_support(
            emitted_text="C",
            successor_atom=AtomId(1),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CYCLIC_TREE_ENTRY
            ),
            residual_key=key_7,
        )
        choice = self._test_choice_snapshot_entry_from_supports(
            "C",
            (support_7,),
        )
        closure_7, child_7 = self._test_residual_cyclic_policy_inputs(
            attachment_id=7,
        )
        residual_7 = writer_transitions._residual_cyclic_policy_decision(
            closure_7,
            child_7,
        )
        closure_8, child_8 = self._test_residual_cyclic_policy_inputs(
            attachment_id=8,
        )
        residual_8 = writer_transitions._residual_cyclic_policy_decision(
            closure_8,
            child_8,
        )
        policy_7 = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD_AFTER_DEAD_CLOSURE_OPEN
            ),
            active_atom=AtomId(0),
            closure_endpoint_decision=closure_7,
            child_schedule_surface=child_7,
            residual_cyclic_policy_decision=residual_7,
        )
        policy_8 = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD_AFTER_DEAD_CLOSURE_OPEN
            ),
            active_atom=AtomId(0),
            closure_endpoint_decision=closure_8,
            child_schedule_surface=child_8,
            residual_cyclic_policy_decision=residual_8,
        )
        schedule_outcome = (
            self._test_frontier_schedule_outcome_for_graph_policies(
                (policy_7, policy_8),
            )
        )
        evidence_groups = (
            writer_frontier_module
            ._writer_frontier_choice_residual_attachment_evidence_groups(
                choice=choice,
                schedule_outcome=schedule_outcome,
            )
        )
        evidence = (
            writer_frontier_module
            ._WriterFrontierChoiceResidualAttachmentEvidence(
                choice=choice,
                residual_attachment_evidence_groups=evidence_groups,
            )
        )

        self.assertEqual(
            evidence.residual_cyclic_policy_decisions,
            (residual_7,),
        )
        self.assertEqual(
            evidence.residual_cyclic_policy_kinds,
            (residual_7.kind,),
        )

    def test_writer_frontier_choice_residual_attachment_evidence_groups_include_support_only_keys(self) -> None:
        key = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=7,
        )
        support = self._test_frontier_next_token_support(
            emitted_text="C",
            successor_atom=AtomId(1),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CYCLIC_TREE_ENTRY
            ),
            residual_key=key,
        )
        choice = self._test_choice_snapshot_entry_from_supports("C", (support,))
        schedule_outcome = SimpleNamespace(
            resolved_residual_attachment_policy_groups=(),
            support_dead_closure_open_vs_cyclic_tree_entry_groups=(),
            unsupported_owner_scope_residual_attachment_policy_groups=(),
            unresolved_residual_attachment_policy_groups=(),
        )

        groups = (
            writer_frontier_module
            ._writer_frontier_choice_residual_attachment_evidence_groups(
                choice=choice,
                schedule_outcome=schedule_outcome,  # type: ignore[arg-type]
            )
        )

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].key, key)
        self.assertEqual(
            groups[0].selected_support_groups,
            choice.residual_attachment_support_groups,
        )
        self.assertEqual(groups[0].resolved_policy_groups, ())
        self.assertEqual(groups[0].unresolved_policy_groups, ())

    def test_writer_frontier_choice_residual_attachment_evidence_exposes_choice_fields(self) -> None:
        key = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=7,
        )
        support = self._test_frontier_next_token_support(
            emitted_text="C",
            successor_atom=AtomId(1),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CYCLIC_TREE_ENTRY
            ),
            residual_key=key,
        )
        choice = self._test_choice_snapshot_entry_from_supports("C", (support,))
        group = writer_frontier_module._WriterFrontierResidualAttachmentEvidenceGroup(
            key=key,
            selected_support_groups=choice.residual_attachment_support_groups,
        )
        evidence = (
            writer_frontier_module
            ._WriterFrontierChoiceResidualAttachmentEvidence(
                choice=choice,
                residual_attachment_evidence_groups=(group,),
            )
        )

        self.assertEqual(evidence.emitted_text, choice.emitted_text)
        self.assertIs(evidence.successor, choice.successor)
        self.assertEqual(evidence.supports, choice.supports)
        self.assertEqual(evidence.public_choice, choice.to_public_choice())
        self.assertEqual(evidence.residual_attachment_policy_keys, (key,))
        self.assertEqual(evidence.selected_supports, (support,))
        self.assertEqual(
            evidence.selected_policy_families,
            (
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CYCLIC_TREE_ENTRY,
            ),
        )
        self.assertTrue(evidence.has_residual_attachment_evidence)

        empty_evidence = (
            writer_frontier_module
            ._WriterFrontierChoiceResidualAttachmentEvidence(
                choice=choice,
                residual_attachment_evidence_groups=(),
            )
        )

        self.assertFalse(empty_evidence.has_residual_attachment_evidence)
        self.assertEqual(empty_evidence.selected_supports, ())
        self.assertEqual(empty_evidence.selected_policy_families, ())

    def test_writer_frontier_residual_attachment_evidence_group_classifies_selected_support_families(self) -> None:
        key = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=7,
        )
        closure = self._test_frontier_next_token_support(
            emitted_text="1",
            successor_atom=AtomId(1),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CLOSURE_OPEN
            ),
            residual_key=key,
        )
        cyclic = self._test_frontier_next_token_support(
            emitted_text="C",
            successor_atom=AtomId(2),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CYCLIC_TREE_ENTRY
            ),
            residual_key=key,
        )
        acyclic = self._test_frontier_next_token_support(
            emitted_text="N",
            successor_atom=AtomId(3),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .ACYCLIC_TREE_ENTRY
            ),
            residual_key=key,
        )

        cases = (
            (
                (closure,),
                (True, False, False, False),
            ),
            (
                (cyclic,),
                (False, True, False, True),
            ),
            (
                (acyclic,),
                (False, False, True, True),
            ),
            (
                (),
                (False, False, False, False),
            ),
        )

        for supports, expected in cases:
            with self.subTest(supports=supports):
                selected_support_groups = ()
                if supports:
                    selected_support_groups = (
                        writer_frontier_module
                        ._WriterFrontierResidualAttachmentSupportGroup(
                            key=key,
                            supports=supports,
                        ),
                    )
                group = (
                    writer_frontier_module
                    ._WriterFrontierResidualAttachmentEvidenceGroup(
                        key=key,
                        selected_support_groups=selected_support_groups,
                    )
                )

                self.assertEqual(
                    (
                        group.has_selected_closure_open_supports,
                        group.has_selected_cyclic_tree_entry_supports,
                        group.has_selected_acyclic_tree_entry_supports,
                        group.has_selected_tree_entry_supports,
                    ),
                    expected,
                )

    def test_writer_frontier_residual_attachment_evidence_group_detects_dead_closure_resolved_cyclic_support(self) -> None:
        key = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=7,
        )
        resolved = self._test_residual_policy_group(key)
        support_dead = self._test_residual_policy_group(key)
        cyclic = self._test_frontier_next_token_support(
            emitted_text="C",
            successor_atom=AtomId(1),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CYCLIC_TREE_ENTRY
            ),
            residual_key=key,
        )
        cyclic_support_group = (
            writer_frontier_module._WriterFrontierResidualAttachmentSupportGroup(
                key=key,
                supports=(cyclic,),
            )
        )
        resolved_cyclic = (
            writer_frontier_module
            ._WriterFrontierResidualAttachmentEvidenceGroup(
                key=key,
                resolved_policy_groups=(resolved,),
                support_dead_closure_open_vs_cyclic_tree_entry_policy_groups=(
                    support_dead,
                ),
                selected_support_groups=(cyclic_support_group,),
            )
        )

        self.assertTrue(
            resolved_cyclic.has_dead_closure_open_resolution_evidence
        )
        self.assertTrue(
            (
                resolved_cyclic
                .has_dead_closure_open_resolved_cyclic_tree_entry_support
            )
        )

        support_dead_without_cyclic = (
            writer_frontier_module
            ._WriterFrontierResidualAttachmentEvidenceGroup(
                key=key,
                resolved_policy_groups=(resolved,),
                support_dead_closure_open_vs_cyclic_tree_entry_policy_groups=(
                    support_dead,
                ),
            )
        )
        cyclic_without_support_dead = (
            writer_frontier_module
            ._WriterFrontierResidualAttachmentEvidenceGroup(
                key=key,
                selected_support_groups=(cyclic_support_group,),
            )
        )
        unresolved = (
            writer_frontier_module
            ._WriterFrontierResidualAttachmentEvidenceGroup(
                key=key,
                unresolved_policy_groups=(self._test_residual_policy_group(key),),
                selected_support_groups=(cyclic_support_group,),
            )
        )
        unsupported_owner = (
            writer_frontier_module
            ._WriterFrontierResidualAttachmentEvidenceGroup(
                key=key,
                unsupported_owner_scope_policy_groups=(
                    self._test_residual_policy_group(key),
                ),
                selected_support_groups=(cyclic_support_group,),
            )
        )

        self.assertTrue(
            support_dead_without_cyclic
            .has_dead_closure_open_resolution_evidence
        )
        self.assertFalse(
            (
                support_dead_without_cyclic
                .has_dead_closure_open_resolved_cyclic_tree_entry_support
            )
        )
        self.assertTrue(
            cyclic_without_support_dead.has_selected_cyclic_tree_entry_supports
        )
        self.assertFalse(
            cyclic_without_support_dead
            .has_dead_closure_open_resolution_evidence
        )
        self.assertFalse(
            (
                cyclic_without_support_dead
                .has_dead_closure_open_resolved_cyclic_tree_entry_support
            )
        )
        self.assertFalse(unresolved.has_dead_closure_open_resolution_evidence)
        self.assertFalse(
            unresolved.has_dead_closure_open_resolved_cyclic_tree_entry_support
        )
        self.assertFalse(
            unsupported_owner.has_dead_closure_open_resolution_evidence
        )

    def test_writer_frontier_residual_attachment_evidence_group_exposes_owner_scope_kinds(self) -> None:
        key = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=7,
        )
        active = self._test_residual_policy_group_with_owner_scope(
            key,
            closure_owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            child_owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        branch_return = self._test_residual_policy_group_with_owner_scope(
            key,
            closure_owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
            child_owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
        )
        missing = self._test_residual_policy_group_with_owner_scope(
            key,
            closure_owner_kind=None,
            child_owner_kind=None,
        )
        group = (
            writer_frontier_module
            ._WriterFrontierResidualAttachmentEvidenceGroup(
                key=key,
                resolved_policy_groups=(active,),
                support_dead_closure_open_vs_cyclic_tree_entry_policy_groups=(
                    active,
                ),
                unsupported_owner_scope_policy_groups=(branch_return,),
                unresolved_policy_groups=(missing,),
            )
        )

        self.assertEqual(
            group.resolved_policy_owner_scope_kinds,
            (
                writer_transitions
                ._WriterResidualAttachmentOwnerScopeKind
                .ACTIVE_ATOM,
            ),
        )
        self.assertEqual(
            (
                group
                .support_dead_closure_open_vs_cyclic_tree_entry_policy_owner_scope_kinds
            ),
            (
                writer_transitions
                ._WriterResidualAttachmentOwnerScopeKind
                .ACTIVE_ATOM,
            ),
        )
        self.assertEqual(
            group.unsupported_owner_scope_kinds,
            (
                writer_transitions
                ._WriterResidualAttachmentOwnerScopeKind
                .BRANCH_RETURN,
            ),
        )
        self.assertEqual(
            group.unresolved_policy_owner_scope_kinds,
            (
                writer_transitions
                ._WriterResidualAttachmentOwnerScopeKind
                .MISSING,
            ),
        )
        self.assertEqual(
            group.policy_owner_scope_kinds,
            (
                writer_transitions
                ._WriterResidualAttachmentOwnerScopeKind
                .ACTIVE_ATOM,
                writer_transitions
                ._WriterResidualAttachmentOwnerScopeKind
                .ACTIVE_ATOM,
                writer_transitions
                ._WriterResidualAttachmentOwnerScopeKind
                .BRANCH_RETURN,
                writer_transitions
                ._WriterResidualAttachmentOwnerScopeKind
                .MISSING,
            ),
        )
        self.assertTrue(group.has_active_atom_owner_scope_evidence)
        self.assertTrue(group.has_branch_return_owner_scope_evidence)
        self.assertTrue(group.has_missing_owner_scope_evidence)
        self.assertFalse(group.has_mixed_owner_scope_evidence)

    def test_writer_frontier_residual_attachment_evidence_group_reports_unsupported_owner_scope_predicates(self) -> None:
        key = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=7,
        )
        cases = (
            (
                WriterBoundaryOwnerKind.BRANCH_RETURN,
                WriterBoundaryOwnerKind.BRANCH_RETURN,
                writer_transitions._WriterResidualAttachmentOwnerScopeKind.BRANCH_RETURN,
                "has_branch_return_owner_scope_evidence",
            ),
            (
                WriterBoundaryOwnerKind.PENDING_PARENT,
                WriterBoundaryOwnerKind.PENDING_PARENT,
                writer_transitions._WriterResidualAttachmentOwnerScopeKind.PENDING_PARENT,
                "has_pending_parent_owner_scope_evidence",
            ),
            (
                WriterBoundaryOwnerKind.OPEN_RING_ENDPOINT,
                WriterBoundaryOwnerKind.OPEN_RING_ENDPOINT,
                (
                    writer_transitions
                    ._WriterResidualAttachmentOwnerScopeKind
                    .OPEN_RING_ENDPOINT
                ),
                "has_open_ring_endpoint_owner_scope_evidence",
            ),
            (
                WriterBoundaryOwnerKind.UNOWNED,
                WriterBoundaryOwnerKind.UNOWNED,
                writer_transitions._WriterResidualAttachmentOwnerScopeKind.UNOWNED,
                "has_unowned_owner_scope_evidence",
            ),
            (
                None,
                None,
                writer_transitions._WriterResidualAttachmentOwnerScopeKind.MISSING,
                "has_missing_owner_scope_evidence",
            ),
            (
                WriterBoundaryOwnerKind.ACTIVE_ATOM,
                WriterBoundaryOwnerKind.BRANCH_RETURN,
                writer_transitions._WriterResidualAttachmentOwnerScopeKind.MIXED,
                "has_mixed_owner_scope_evidence",
            ),
        )

        for closure_owner, child_owner, expected, predicate in cases:
            with self.subTest(scope=expected):
                policy_group = (
                    self._test_residual_policy_group_with_owner_scope(
                        key,
                        closure_owner_kind=closure_owner,
                        child_owner_kind=child_owner,
                    )
                )
                group = (
                    writer_frontier_module
                    ._WriterFrontierResidualAttachmentEvidenceGroup(
                        key=key,
                        unsupported_owner_scope_policy_groups=(policy_group,),
                    )
                )

                self.assertEqual(
                    group.unsupported_owner_scope_kinds,
                    (expected,),
                )
                self.assertTrue(getattr(group, predicate))

    def test_writer_frontier_choice_residual_attachment_evidence_exposes_dead_closure_resolved_cyclic_groups(self) -> None:
        key_7 = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=7,
        )
        key_8 = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=8,
        )
        support = self._test_frontier_next_token_support(
            emitted_text="C",
            successor_atom=AtomId(1),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CYCLIC_TREE_ENTRY
            ),
            residual_key=key_7,
        )
        choice = self._test_choice_snapshot_entry_from_supports("C", (support,))
        group_7 = (
            writer_frontier_module
            ._WriterFrontierResidualAttachmentEvidenceGroup(
                key=key_7,
                resolved_policy_groups=(self._test_residual_policy_group(key_7),),
                support_dead_closure_open_vs_cyclic_tree_entry_policy_groups=(
                    self._test_residual_policy_group(key_7),
                ),
                selected_support_groups=(
                    writer_frontier_module
                    ._WriterFrontierResidualAttachmentSupportGroup(
                        key=key_7,
                        supports=(support,),
                    ),
                ),
            )
        )
        group_8 = (
            writer_frontier_module
            ._WriterFrontierResidualAttachmentEvidenceGroup(
                key=key_8,
                unresolved_policy_groups=(self._test_residual_policy_group(key_8),),
            )
        )
        evidence = (
            writer_frontier_module
            ._WriterFrontierChoiceResidualAttachmentEvidence(
                choice=choice,
                residual_attachment_evidence_groups=(group_7, group_8),
            )
        )

        self.assertEqual(
            (
                evidence
                .dead_closure_open_resolved_cyclic_tree_entry_groups
            ),
            (group_7,),
        )
        self.assertTrue(
            (
                evidence
                .has_dead_closure_open_resolved_cyclic_tree_entry_support
            )
        )

        empty = (
            writer_frontier_module
            ._WriterFrontierChoiceResidualAttachmentEvidence(
                choice=choice,
                residual_attachment_evidence_groups=(group_8,),
            )
        )

        self.assertEqual(
            empty.dead_closure_open_resolved_cyclic_tree_entry_groups,
            (),
        )
        self.assertFalse(
            empty.has_dead_closure_open_resolved_cyclic_tree_entry_support
        )

    def test_writer_frontier_choice_residual_attachment_evidence_exposes_owner_scope_kinds(self) -> None:
        key_7 = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=7,
        )
        key_8 = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=8,
        )
        support = self._test_frontier_next_token_support(
            emitted_text="C",
            successor_atom=AtomId(1),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CYCLIC_TREE_ENTRY
            ),
            residual_key=key_7,
        )
        choice = self._test_choice_snapshot_entry_from_supports("C", (support,))
        branch_return = self._test_residual_policy_group_with_owner_scope(
            key_7,
            closure_owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
            child_owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
        )
        active = self._test_residual_policy_group_with_owner_scope(
            key_8,
            closure_owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            child_owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        unsupported_group = (
            writer_frontier_module
            ._WriterFrontierResidualAttachmentEvidenceGroup(
                key=key_7,
                unsupported_owner_scope_policy_groups=(branch_return,),
            )
        )
        supported_group = (
            writer_frontier_module
            ._WriterFrontierResidualAttachmentEvidenceGroup(
                key=key_8,
                resolved_policy_groups=(active,),
            )
        )
        evidence = (
            writer_frontier_module
            ._WriterFrontierChoiceResidualAttachmentEvidence(
                choice=choice,
                residual_attachment_evidence_groups=(
                    unsupported_group,
                    supported_group,
                ),
            )
        )

        self.assertEqual(
            evidence.unsupported_owner_scope_kinds,
            (
                writer_transitions
                ._WriterResidualAttachmentOwnerScopeKind
                .BRANCH_RETURN,
            ),
        )
        self.assertEqual(
            evidence.policy_owner_scope_kinds,
            (
                writer_transitions
                ._WriterResidualAttachmentOwnerScopeKind
                .BRANCH_RETURN,
                writer_transitions
                ._WriterResidualAttachmentOwnerScopeKind
                .ACTIVE_ATOM,
            ),
        )
        self.assertTrue(evidence.has_unsupported_owner_scope_evidence)

        empty = (
            writer_frontier_module
            ._WriterFrontierChoiceResidualAttachmentEvidence(
                choice=choice,
                residual_attachment_evidence_groups=(supported_group,),
            )
        )

        self.assertEqual(empty.unsupported_owner_scope_kinds, ())
        self.assertFalse(empty.has_unsupported_owner_scope_evidence)

    def test_writer_frontier_schedule_outcome_records_scheduled_state_outcomes(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())

        outcome = writer_frontier_module._writer_frontier_schedule_outcome(
            prepared,
            cursor,
        )

        self.assertFalse(outcome.blocked)
        self.assertEqual(outcome.graph_policy_blockers, ())
        self.assertEqual(len(outcome.state_outcomes), len(cursor.weighted_states))
        self.assertEqual(
            outcome.grouped_transitions.grouped_by_text,
            outcome.grouped_by_text,
        )
        self.assertEqual(
            outcome.grouped_transitions.weighted_by_text,
            outcome.weighted_by_text,
        )
        self.assertEqual(tuple(sorted(outcome.grouped_by_text)), ("C",))

    def test_writer_frontier_state_schedule_outcome_exposes_graph_policy_decision(self) -> None:
        state_key = writer_state_key(_raw_initial_state(AtomId(0)))
        policy, active_outcome = self._dead_closure_open_active_emitted_outcome(
            AtomId(0),
        )
        active_top_outcome = writer_transitions._WriterTopLevelScheduleOutcome(
            kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.SCHEDULED,
            schedule_decision=writer_transitions._top_level_active_emitted_decision(
                active_outcome.schedule_decision,
            ),
            active_emitted_outcome=active_outcome,
        )

        state_outcome = writer_frontier_module._WriterFrontierStateScheduleOutcome(
            state_key=state_key,
            parent_weight=1,
            finalized_state_key=None,
            schedule_outcome=active_top_outcome,
        )

        self.assertIs(state_outcome.graph_policy_decision, policy)

        action = writer_transitions._emit_root_atom_action(AtomId(0))
        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(action,),
            emissions=(),
            surviving_emissions=(),
        )
        root_outcome = writer_transitions._WriterTopLevelScheduleOutcome(
            kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.SCHEDULED,
            schedule_decision=writer_transitions._top_level_actions_decision(batch),
        )
        root_state_outcome = (
            writer_frontier_module._WriterFrontierStateScheduleOutcome(
                state_key=state_key,
                parent_weight=1,
                finalized_state_key=None,
                schedule_outcome=root_outcome,
            )
        )

        self.assertIsNone(root_state_outcome.graph_policy_decision)

    def test_writer_frontier_schedule_outcome_aggregates_graph_policy_decisions(self) -> None:
        state_key_1 = writer_state_key(_raw_initial_state(AtomId(0)))
        state_key_2 = writer_state_key(_raw_initial_state(AtomId(1)))
        state_key_3 = writer_state_key(_raw_initial_state(AtomId(2)))
        policy_a, active_outcome_a = (
            self._dead_closure_open_active_emitted_outcome(AtomId(0))
        )
        policy_b, active_outcome_b = self._closure_policy_for_outcome(AtomId(2))
        top_a = writer_transitions._WriterTopLevelScheduleOutcome(
            kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.SCHEDULED,
            schedule_decision=writer_transitions._top_level_active_emitted_decision(
                active_outcome_a.schedule_decision,
            ),
            active_emitted_outcome=active_outcome_a,
        )
        action = writer_transitions._emit_root_atom_action(AtomId(1))
        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(action,),
            emissions=(),
            surviving_emissions=(),
        )
        top_without_policy = writer_transitions._WriterTopLevelScheduleOutcome(
            kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.SCHEDULED,
            schedule_decision=writer_transitions._top_level_actions_decision(batch),
        )
        active_outcome_b = writer_transitions._WriterActiveEmittedScheduleOutcome(
            kind=writer_transitions._WriterActiveEmittedScheduleOutcomeKind.SCHEDULED,
            graph_policy_decision=policy_b,
            schedule_decision=active_outcome_b,
        )
        top_b = writer_transitions._WriterTopLevelScheduleOutcome(
            kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.SCHEDULED,
            schedule_decision=writer_transitions._top_level_active_emitted_decision(
                active_outcome_b.schedule_decision,
            ),
            active_emitted_outcome=active_outcome_b,
        )
        outcome = writer_frontier_module._WriterFrontierScheduleOutcome(
            state_outcomes=(
                writer_frontier_module._WriterFrontierStateScheduleOutcome(
                    state_key=state_key_1,
                    parent_weight=1,
                    finalized_state_key=None,
                    schedule_outcome=top_a,
                ),
                writer_frontier_module._WriterFrontierStateScheduleOutcome(
                    state_key=state_key_2,
                    parent_weight=1,
                    finalized_state_key=None,
                    schedule_outcome=top_without_policy,
                ),
                writer_frontier_module._WriterFrontierStateScheduleOutcome(
                    state_key=state_key_3,
                    parent_weight=1,
                    finalized_state_key=None,
                    schedule_outcome=top_b,
                ),
            ),
            terminal_by_key=Counter(),
            grouped_by_text={},
            weighted_by_text={},
        )

        self.assertEqual(outcome.graph_policy_decisions, (policy_a, policy_b))
        self.assertEqual(
            outcome.resolved_residual_attachment_policy_groups,
            policy_a.resolved_residual_attachment_policy_groups,
        )
        self.assertEqual(
            outcome.support_dead_closure_open_vs_cyclic_tree_entry_groups,
            policy_a.support_dead_closure_open_vs_cyclic_tree_entry_groups,
        )

    def test_writer_frontier_schedule_outcome_exposes_active_child_selection_kinds(self) -> None:
        state_key_1 = writer_state_key(_raw_initial_state(AtomId(0)))
        state_key_2 = writer_state_key(_raw_initial_state(AtomId(1)))
        state_key_3 = writer_state_key(_raw_initial_state(AtomId(2)))
        cyclic_outcome = self._cyclic_active_child_scheduled_outcome(AtomId(0))
        cyclic_top = writer_transitions._WriterTopLevelScheduleOutcome(
            kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.SCHEDULED,
            schedule_decision=writer_transitions._top_level_active_emitted_decision(
                cyclic_outcome.schedule_decision,
            ),
            active_emitted_outcome=cyclic_outcome,
        )
        root_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(writer_transitions._emit_root_atom_action(AtomId(1)),),
            emissions=(),
            surviving_emissions=(),
        )
        root_top = writer_transitions._WriterTopLevelScheduleOutcome(
            kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.SCHEDULED,
            schedule_decision=writer_transitions._top_level_actions_decision(
                root_batch,
            ),
        )
        closure_decision, _, _ = self._test_closure_endpoint_decision(
            active_atom=AtomId(2),
            pair_survives=False,
            open_survives=False,
        )
        finish_action = writer_transitions._finish_active_action(AtomId(2))
        finish_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=AtomId(2),
            blockers=(),
            child_obligations=(),
            scheduled_actions=(finish_action,),
        )
        finish_policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD
            ),
            active_atom=AtomId(2),
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=finish_surface,
        )
        finish_emission = writer_transitions._WriterScheduledActionEmission(
            action=finish_action,
            transitions=(object(),),  # type: ignore[arg-type]
        )
        finish_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(finish_action,),
            emissions=(finish_emission,),
            surviving_emissions=(finish_emission,),
        )
        finish_decision = writer_transitions._active_emitted_child_decision(
            closure_decision,
            finish_surface,
            finish_batch,
            graph_policy_decision=finish_policy,
        )
        finish_outcome = writer_transitions._WriterActiveEmittedScheduleOutcome(
            kind=writer_transitions._WriterActiveEmittedScheduleOutcomeKind.SCHEDULED,
            graph_policy_decision=finish_policy,
            schedule_decision=finish_decision,
        )
        finish_top = writer_transitions._WriterTopLevelScheduleOutcome(
            kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.SCHEDULED,
            schedule_decision=writer_transitions._top_level_active_emitted_decision(
                finish_decision,
            ),
            active_emitted_outcome=finish_outcome,
        )
        outcome = writer_frontier_module._WriterFrontierScheduleOutcome(
            state_outcomes=(
                writer_frontier_module._WriterFrontierStateScheduleOutcome(
                    state_key=state_key_1,
                    parent_weight=1,
                    finalized_state_key=None,
                    schedule_outcome=cyclic_top,
                ),
                writer_frontier_module._WriterFrontierStateScheduleOutcome(
                    state_key=state_key_2,
                    parent_weight=1,
                    finalized_state_key=None,
                    schedule_outcome=root_top,
                ),
                writer_frontier_module._WriterFrontierStateScheduleOutcome(
                    state_key=state_key_3,
                    parent_weight=1,
                    finalized_state_key=None,
                    schedule_outcome=finish_top,
                ),
            ),
            terminal_by_key=Counter(),
            grouped_by_text={},
            weighted_by_text={},
        )

        self.assertEqual(
            outcome.considered_active_child_selection_kinds,
            (
                writer_transitions._WriterActiveChildSelectionKind.CYCLIC_TREE_ENTRY,
                writer_transitions._WriterActiveChildSelectionKind.FINISH_ACTIVE,
            ),
        )
        self.assertEqual(
            outcome.selected_active_child_selection_kinds,
            (
                writer_transitions._WriterActiveChildSelectionKind.CYCLIC_TREE_ENTRY,
                writer_transitions._WriterActiveChildSelectionKind.FINISH_ACTIVE,
            ),
        )

    def test_writer_frontier_schedule_outcome_exposes_cyclic_tree_entry_surfaces(self) -> None:
        state_key = writer_state_key(_raw_initial_state(AtomId(0)))
        active_outcome = self._cyclic_active_child_scheduled_outcome(AtomId(0))
        top = writer_transitions._WriterTopLevelScheduleOutcome(
            kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.SCHEDULED,
            schedule_decision=writer_transitions._top_level_active_emitted_decision(
                active_outcome.schedule_decision,
            ),
            active_emitted_outcome=active_outcome,
        )
        state_outcome = writer_frontier_module._WriterFrontierStateScheduleOutcome(
            state_key=state_key,
            parent_weight=1,
            finalized_state_key=None,
            schedule_outcome=top,
        )
        outcome = writer_frontier_module._WriterFrontierScheduleOutcome(
            state_outcomes=(state_outcome,),
            terminal_by_key=Counter(),
            grouped_by_text={},
            weighted_by_text={},
        )

        self.assertEqual(
            outcome.considered_cyclic_tree_entry_graph_action_surfaces,
            top.considered_cyclic_tree_entry_graph_action_surfaces,
        )
        self.assertEqual(
            outcome.selected_cyclic_tree_entry_graph_action_surfaces,
            top.selected_cyclic_tree_entry_graph_action_surfaces,
        )

    def test_writer_frontier_schedule_outcome_exposes_closure_endpoint_selection_kinds(self) -> None:
        pair_decision, _, _ = self._test_closure_endpoint_decision(
            pair_survives=True,
            open_survives=False,
        )
        policy_pair = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .CLOSURE_ENDPOINT
            ),
            active_atom=AtomId(0),
            closure_endpoint_decision=pair_decision,
        )
        active_pair_decision = writer_transitions._active_emitted_closure_decision(
            pair_decision,
            graph_policy_decision=policy_pair,
        )
        active_pair_outcome = (
            writer_transitions._WriterActiveEmittedScheduleOutcome(
                kind=(
                    writer_transitions
                    ._WriterActiveEmittedScheduleOutcomeKind
                    .SCHEDULED
                ),
                schedule_decision=active_pair_decision,
                graph_policy_decision=policy_pair,
            )
        )

        open_action = self._test_closure_open_action(AtomId(1))
        open_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(
                    writer_transitions._WriterScheduledActionEmission(
                        action=open_action,
                        transitions=(),
                    ),
                ),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=AtomId(1),
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(3),
            child=AtomId(4),
            boundary_atom=AtomId(1),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=17,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            AtomId(1),
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=AtomId(1),
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )
        policy_open_none = (
            writer_transitions._WriterActiveEmittedGraphPolicyDecision(
                kind=(
                    writer_transitions
                    ._WriterActiveEmittedGraphPolicyDecisionKind
                    .ACTIVE_CHILD
                ),
                active_atom=AtomId(1),
                closure_endpoint_decision=open_decision,
                child_schedule_surface=child_surface,
            )
        )
        active_child_decision = writer_transitions._active_emitted_child_decision(
            closure_endpoint_decision=open_decision,
            child_schedule_surface=child_surface,
            child_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(child_action,),
                emissions=(),
                surviving_emissions=(),
            ),
            graph_policy_decision=policy_open_none,
        )
        active_child_outcome = (
            writer_transitions._WriterActiveEmittedScheduleOutcome(
                kind=(
                    writer_transitions
                    ._WriterActiveEmittedScheduleOutcomeKind
                    .SCHEDULED
                ),
                schedule_decision=active_child_decision,
                graph_policy_decision=policy_open_none,
            )
        )
        top_pair = writer_transitions._WriterTopLevelScheduleOutcome(
            kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.SCHEDULED,
            schedule_decision=writer_transitions._top_level_active_emitted_decision(
                active_pair_outcome.schedule_decision,
            ),
            active_emitted_outcome=active_pair_outcome,
        )
        top_child = writer_transitions._WriterTopLevelScheduleOutcome(
            kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.SCHEDULED,
            schedule_decision=writer_transitions._top_level_active_emitted_decision(
                active_child_outcome.schedule_decision,
            ),
            active_emitted_outcome=active_child_outcome,
        )
        outcome = writer_frontier_module._WriterFrontierScheduleOutcome(
            state_outcomes=(
                writer_frontier_module._WriterFrontierStateScheduleOutcome(
                    state_key=writer_state_key(_raw_initial_state(AtomId(0))),
                    parent_weight=1,
                    finalized_state_key=None,
                    schedule_outcome=top_pair,
                ),
                writer_frontier_module._WriterFrontierStateScheduleOutcome(
                    state_key=writer_state_key(_raw_initial_state(AtomId(1))),
                    parent_weight=1,
                    finalized_state_key=None,
                    schedule_outcome=top_child,
                ),
            ),
            terminal_by_key=Counter(),
            grouped_by_text={},
            weighted_by_text={},
        )

        self.assertEqual(
            outcome.considered_closure_endpoint_selection_kinds,
            (
                writer_transitions
                ._WriterClosureEndpointSelectionKind
                .PAIR_AND_OPEN,
                (
                    writer_transitions
                    ._WriterClosureEndpointSelectionKind
                    .CLOSURE_OPEN
                ),
            ),
        )
        self.assertEqual(
            outcome.selected_closure_endpoint_selection_kinds,
            (
                writer_transitions
                ._WriterClosureEndpointSelectionKind
                .CLOSURE_PAIR,
            ),
        )

    def test_writer_frontier_schedule_outcome_exposes_selected_closure_surfaces(self) -> None:
        closure_decision, pair_emission, open_emission = (
            self._test_closure_endpoint_decision(
                pair_survives=True,
                open_survives=True,
            )
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .CLOSURE_ENDPOINT
            ),
            active_atom=AtomId(0),
            closure_endpoint_decision=closure_decision,
        )
        active_decision = writer_transitions._active_emitted_closure_decision(
            closure_decision,
            graph_policy_decision=policy,
        )
        active_outcome = writer_transitions._WriterActiveEmittedScheduleOutcome(
            kind=(
                writer_transitions
                ._WriterActiveEmittedScheduleOutcomeKind
                .SCHEDULED
            ),
            schedule_decision=active_decision,
            graph_policy_decision=policy,
        )
        top_outcome = writer_transitions._WriterTopLevelScheduleOutcome(
            kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.SCHEDULED,
            schedule_decision=writer_transitions._top_level_active_emitted_decision(
                active_outcome.schedule_decision,
            ),
            active_emitted_outcome=active_outcome,
        )
        outcome = writer_frontier_module._WriterFrontierScheduleOutcome(
            state_outcomes=(
                writer_frontier_module._WriterFrontierStateScheduleOutcome(
                    state_key=writer_state_key(_raw_initial_state(AtomId(0))),
                    parent_weight=1,
                    finalized_state_key=None,
                    schedule_outcome=top_outcome,
                ),
            ),
            terminal_by_key=Counter(),
            grouped_by_text={},
            weighted_by_text={},
        )

        self.assertEqual(
            outcome.selected_closure_pair_graph_action_surfaces,
            (pair_emission.graph_action_surface,),
        )
        self.assertEqual(
            outcome.selected_closure_open_graph_action_surfaces,
            (open_emission.graph_action_surface,),
        )

    def test_writer_frontier_schedule_outcome_exposes_residual_attachment_support_groups(self) -> None:
        key_7 = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=7,
        )
        key_8 = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=8,
        )
        support_a = self._test_frontier_next_token_support(
            emitted_text="C",
            successor_atom=AtomId(1),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CYCLIC_TREE_ENTRY
            ),
            residual_key=key_7,
        )
        support_b = self._test_frontier_next_token_support(
            emitted_text="N",
            successor_atom=AtomId(2),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .ACYCLIC_TREE_ENTRY
            ),
            residual_key=key_8,
        )
        support_c = self._test_frontier_next_token_support(
            emitted_text="N",
            successor_atom=AtomId(3),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CLOSURE_OPEN
            ),
            residual_key=key_7,
        )
        outcome = writer_frontier_module._WriterFrontierScheduleOutcome(
            state_outcomes=(),
            terminal_by_key=Counter(),
            grouped_by_text={},
            weighted_by_text={},
            next_token_frontier=(
                writer_frontier_module._WriterFrontierNextTokenEntry(
                    emitted_text="C",
                    supports=(support_a,),
                ),
                writer_frontier_module._WriterFrontierNextTokenEntry(
                    emitted_text="N",
                    supports=(support_b, support_c),
                ),
            ),
        )

        groups = outcome.residual_attachment_support_groups

        self.assertEqual(tuple(group.key for group in groups), (key_7, key_8))
        self.assertEqual(groups[0].supports, (support_a, support_c))
        self.assertEqual(groups[1].supports, (support_b,))

    def test_writer_frontier_schedule_outcome_exposes_residual_attachment_evidence_groups(self) -> None:
        policy, active_outcome = self._dead_closure_open_active_emitted_outcome(
            AtomId(0),
        )
        policy_group = policy.resolved_residual_attachment_policy_groups[0]
        support = self._test_frontier_next_token_support(
            emitted_text="C",
            successor_atom=AtomId(1),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CYCLIC_TREE_ENTRY
            ),
            residual_key=policy_group.key,
        )
        top_outcome = writer_transitions._WriterTopLevelScheduleOutcome(
            kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.SCHEDULED,
            schedule_decision=writer_transitions._top_level_active_emitted_decision(
                active_outcome.schedule_decision,
            ),
            active_emitted_outcome=active_outcome,
        )
        outcome = writer_frontier_module._WriterFrontierScheduleOutcome(
            state_outcomes=(
                writer_frontier_module._WriterFrontierStateScheduleOutcome(
                    state_key=writer_state_key(_raw_initial_state(AtomId(0))),
                    parent_weight=1,
                    finalized_state_key=None,
                    schedule_outcome=top_outcome,
                ),
            ),
            terminal_by_key=Counter(),
            grouped_by_text={},
            weighted_by_text={},
            next_token_frontier=(
                writer_frontier_module._WriterFrontierNextTokenEntry(
                    emitted_text="C",
                    supports=(support,),
                ),
            ),
        )
        expected = (
            writer_frontier_module
            ._writer_frontier_residual_attachment_evidence_groups(
                resolved_policy_groups=(
                    outcome.resolved_residual_attachment_policy_groups
                ),
                support_dead_closure_open_vs_cyclic_tree_entry_policy_groups=(
                    outcome
                    .support_dead_closure_open_vs_cyclic_tree_entry_groups
                ),
                unsupported_owner_scope_policy_groups=(
                    outcome
                    .unsupported_owner_scope_residual_attachment_policy_groups
                ),
                unresolved_policy_groups=(
                    outcome.unresolved_residual_attachment_policy_groups
                ),
                selected_support_groups=(
                    outcome.residual_attachment_support_groups
                ),
            )
        )

        self.assertEqual(outcome.residual_attachment_evidence_groups, expected)
        self.assertEqual(
            outcome.residual_attachment_evidence_groups[0].key,
            policy_group.key,
        )
        self.assertEqual(
            outcome.residual_attachment_evidence_groups[0].selected_supports,
            (support,),
        )

    def test_writer_frontier_choice_snapshot_exposes_graph_policy_evidence(self) -> None:
        policy, active_outcome = self._dead_closure_open_active_emitted_outcome(
            AtomId(0),
        )
        snapshot = self._test_frontier_choice_snapshot_with_active_policy(
            active_outcome,
        )
        schedule_outcome = snapshot.schedule_outcome

        self.assertEqual(
            snapshot.graph_policy_decisions,
            schedule_outcome.graph_policy_decisions,
        )
        self.assertEqual(snapshot.graph_policy_decisions, (policy,))
        self.assertEqual(
            snapshot.resolved_residual_attachment_policy_groups,
            schedule_outcome.resolved_residual_attachment_policy_groups,
        )
        self.assertEqual(
            snapshot.support_dead_closure_open_vs_cyclic_tree_entry_groups,
            (
                schedule_outcome
                .support_dead_closure_open_vs_cyclic_tree_entry_groups
            ),
        )
        self.assertEqual(
            snapshot.unsupported_owner_scope_residual_attachment_policy_groups,
            (
                schedule_outcome
                .unsupported_owner_scope_residual_attachment_policy_groups
            ),
        )
        self.assertEqual(
            snapshot.unresolved_residual_attachment_policy_groups,
            schedule_outcome.unresolved_residual_attachment_policy_groups,
        )

    def test_writer_frontier_choice_snapshot_exposes_closure_endpoint_selection_evidence(self) -> None:
        closure_decision, _, _ = self._test_closure_endpoint_decision(
            pair_survives=False,
            open_survives=True,
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .CLOSURE_ENDPOINT
            ),
            active_atom=AtomId(0),
            closure_endpoint_decision=closure_decision,
        )
        active_decision = writer_transitions._active_emitted_closure_decision(
            closure_decision,
            graph_policy_decision=policy,
        )
        active_outcome = writer_transitions._WriterActiveEmittedScheduleOutcome(
            kind=(
                writer_transitions
                ._WriterActiveEmittedScheduleOutcomeKind
                .SCHEDULED
            ),
            schedule_decision=active_decision,
            graph_policy_decision=policy,
        )
        snapshot = self._test_frontier_choice_snapshot_with_active_policy(
            active_outcome,
        )
        schedule_outcome = snapshot.schedule_outcome

        self.assertEqual(
            snapshot.considered_closure_endpoint_selection_kinds,
            schedule_outcome.considered_closure_endpoint_selection_kinds,
        )
        self.assertEqual(
            snapshot.selected_closure_endpoint_selection_kinds,
            schedule_outcome.selected_closure_endpoint_selection_kinds,
        )
        self.assertEqual(
            snapshot.selected_closure_open_graph_action_surfaces,
            schedule_outcome.selected_closure_open_graph_action_surfaces,
        )
        self.assertEqual(
            snapshot.selected_closure_pair_graph_action_surfaces,
            schedule_outcome.selected_closure_pair_graph_action_surfaces,
        )

    def test_writer_frontier_choice_snapshot_exposes_active_child_selection_evidence(self) -> None:
        active_outcome = self._cyclic_active_child_scheduled_outcome(AtomId(0))
        snapshot = self._test_frontier_choice_snapshot_with_active_policy(
            active_outcome,
        )
        schedule_outcome = snapshot.schedule_outcome

        self.assertEqual(
            snapshot.considered_active_child_selection_kinds,
            schedule_outcome.considered_active_child_selection_kinds,
        )
        self.assertEqual(
            snapshot.selected_active_child_selection_kinds,
            schedule_outcome.selected_active_child_selection_kinds,
        )
        self.assertEqual(
            snapshot.considered_cyclic_tree_entry_graph_action_surfaces,
            (
                schedule_outcome
                .considered_cyclic_tree_entry_graph_action_surfaces
            ),
        )
        self.assertEqual(
            snapshot.selected_cyclic_tree_entry_graph_action_surfaces,
            (
                schedule_outcome
                .selected_cyclic_tree_entry_graph_action_surfaces
            ),
        )

    def test_writer_frontier_schedule_outcome_exposes_residual_cyclic_policy_decisions(self) -> None:
        none_closure_decision, none_child_surface = (
            self._test_residual_cyclic_policy_inputs(
                child_attachment_action_kind=(
                    WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY
                ),
            )
        )
        none_residual = writer_transitions._residual_cyclic_policy_decision(
            none_closure_decision,
            none_child_surface,
        )
        none_policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD
            ),
            active_atom=AtomId(0),
            closure_endpoint_decision=none_closure_decision,
            child_schedule_surface=none_child_surface,
            residual_cyclic_policy_decision=none_residual,
        )
        dead_closure_decision, dead_child_surface = (
            self._test_residual_cyclic_policy_inputs()
        )
        dead_residual = writer_transitions._residual_cyclic_policy_decision(
            dead_closure_decision,
            dead_child_surface,
        )
        dead_policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD_AFTER_DEAD_CLOSURE_OPEN
            ),
            active_atom=AtomId(0),
            closure_endpoint_decision=dead_closure_decision,
            child_schedule_surface=dead_child_surface,
            residual_cyclic_policy_decision=dead_residual,
        )
        outcome = self._test_frontier_schedule_outcome_for_graph_policies(
            (none_policy, dead_policy),
            include_non_policy_state=True,
        )

        self.assertEqual(
            outcome.residual_cyclic_policy_decisions,
            (none_residual, dead_residual),
        )
        self.assertEqual(
            outcome.residual_cyclic_policy_kinds,
            (
                writer_transitions._WriterResidualCyclicPolicyDecisionKind.NONE,
                (
                    writer_transitions
                    ._WriterResidualCyclicPolicyDecisionKind
                    .ACTIVE_CHILD_AFTER_DEAD_CLOSURE_OPEN
                ),
            ),
        )

    def test_writer_frontier_schedule_outcome_exposes_residual_cyclic_group_buckets(self) -> None:
        unsupported_closure_decision, unsupported_child_surface = (
            self._test_residual_cyclic_policy_inputs(
                closure_owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
                child_owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
            )
        )
        unsupported_residual = (
            writer_transitions._residual_cyclic_policy_decision(
                unsupported_closure_decision,
                unsupported_child_surface,
            )
        )
        unsupported_policy = (
            writer_transitions._WriterActiveEmittedGraphPolicyDecision(
                kind=(
                    writer_transitions
                    ._WriterActiveEmittedGraphPolicyDecisionKind
                    .UNSUPPORTED_OWNER_SCOPE_RESIDUAL_ATTACHMENT_CHOICE
                ),
                active_atom=AtomId(0),
                closure_endpoint_decision=unsupported_closure_decision,
                child_schedule_surface=unsupported_child_surface,
                residual_cyclic_policy_decision=unsupported_residual,
            )
        )
        missing_closure_decision, missing_child_surface = (
            self._test_residual_cyclic_policy_inputs(
                include_closure_emission=False,
            )
        )
        missing_residual = writer_transitions._residual_cyclic_policy_decision(
            missing_closure_decision,
            missing_child_surface,
        )
        missing_policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .UNRESOLVED_RESIDUAL_ATTACHMENT_CHOICE
            ),
            active_atom=AtomId(0),
            closure_endpoint_decision=missing_closure_decision,
            child_schedule_surface=missing_child_surface,
            residual_cyclic_policy_decision=missing_residual,
        )
        support_dead_closure_decision, support_dead_child_surface = (
            self._test_residual_cyclic_policy_inputs()
        )
        support_dead_residual = (
            writer_transitions._residual_cyclic_policy_decision(
                support_dead_closure_decision,
                support_dead_child_surface,
            )
        )
        support_dead_policy = (
            writer_transitions._WriterActiveEmittedGraphPolicyDecision(
                kind=(
                    writer_transitions
                    ._WriterActiveEmittedGraphPolicyDecisionKind
                    .ACTIVE_CHILD_AFTER_DEAD_CLOSURE_OPEN
                ),
                active_atom=AtomId(0),
                closure_endpoint_decision=support_dead_closure_decision,
                child_schedule_surface=support_dead_child_surface,
                residual_cyclic_policy_decision=support_dead_residual,
            )
        )
        outcome = self._test_frontier_schedule_outcome_for_graph_policies(
            (unsupported_policy, missing_policy, support_dead_policy),
        )

        self.assertEqual(
            outcome.residual_cyclic_choice_groups,
            (
                *unsupported_residual.choice_groups,
                *missing_residual.choice_groups,
                *support_dead_residual.choice_groups,
            ),
        )
        self.assertEqual(
            outcome.residual_cyclic_unsupported_owner_scope_groups,
            unsupported_residual.unsupported_owner_scope_groups,
        )
        self.assertEqual(
            outcome.residual_cyclic_missing_evidence_groups,
            missing_residual.missing_evidence_groups,
        )
        self.assertEqual(
            outcome.residual_cyclic_support_dead_groups,
            support_dead_residual.support_dead_groups,
        )

    def test_writer_frontier_choice_snapshot_exposes_residual_cyclic_policy_evidence(self) -> None:
        closure_decision, child_surface = self._test_residual_cyclic_policy_inputs()
        residual_decision = writer_transitions._residual_cyclic_policy_decision(
            closure_decision,
            child_surface,
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD_AFTER_DEAD_CLOSURE_OPEN
            ),
            active_atom=AtomId(0),
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
            residual_cyclic_policy_decision=residual_decision,
        )
        outcome = self._test_frontier_schedule_outcome_for_graph_policies(
            (policy,),
        )
        choice_snapshot = writer_frontier_module._WriterFrontierChoiceSnapshot(
            schedule_outcome=outcome,
            terminal=None,
            choices=(),
        )

        self.assertEqual(
            choice_snapshot.residual_cyclic_policy_decisions,
            outcome.residual_cyclic_policy_decisions,
        )
        self.assertEqual(
            choice_snapshot.residual_cyclic_policy_kinds,
            outcome.residual_cyclic_policy_kinds,
        )
        self.assertEqual(
            choice_snapshot.residual_cyclic_support_dead_groups,
            outcome.residual_cyclic_support_dead_groups,
        )

    def test_writer_frontier_choice_snapshot_exposes_residual_attachment_support_groups(self) -> None:
        key = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=7,
        )
        support = self._test_frontier_next_token_support(
            emitted_text="C",
            successor_atom=AtomId(1),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CYCLIC_TREE_ENTRY
            ),
            residual_key=key,
        )
        schedule_outcome = writer_frontier_module._WriterFrontierScheduleOutcome(
            state_outcomes=(),
            terminal_by_key=Counter(),
            grouped_by_text={},
            weighted_by_text={},
            next_token_frontier=(
                writer_frontier_module._WriterFrontierNextTokenEntry(
                    emitted_text="C",
                    supports=(support,),
                ),
            ),
        )
        choice_snapshot = writer_frontier_module._WriterFrontierChoiceSnapshot(
            schedule_outcome=schedule_outcome,
            terminal=None,
            choices=(),
        )

        self.assertEqual(
            choice_snapshot.residual_attachment_support_groups,
            schedule_outcome.residual_attachment_support_groups,
        )

    def test_writer_frontier_choice_snapshot_exposes_residual_attachment_evidence_groups(self) -> None:
        key = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=7,
        )
        support = self._test_frontier_next_token_support(
            emitted_text="C",
            successor_atom=AtomId(1),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CYCLIC_TREE_ENTRY
            ),
            residual_key=key,
        )
        schedule_outcome = writer_frontier_module._WriterFrontierScheduleOutcome(
            state_outcomes=(),
            terminal_by_key=Counter(),
            grouped_by_text={},
            weighted_by_text={},
            next_token_frontier=(
                writer_frontier_module._WriterFrontierNextTokenEntry(
                    emitted_text="C",
                    supports=(support,),
                ),
            ),
        )
        choice_snapshot = writer_frontier_module._WriterFrontierChoiceSnapshot(
            schedule_outcome=schedule_outcome,
            terminal=None,
            choices=(),
        )

        self.assertEqual(
            choice_snapshot.residual_attachment_evidence_groups,
            schedule_outcome.residual_attachment_evidence_groups,
        )

    def test_writer_frontier_choice_snapshot_exposes_per_choice_residual_attachment_evidence(self) -> None:
        key_c = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=7,
        )
        key_n = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=8,
        )
        support_c = self._test_frontier_next_token_support(
            emitted_text="C",
            successor_atom=AtomId(1),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CYCLIC_TREE_ENTRY
            ),
            residual_key=key_c,
        )
        support_n = self._test_frontier_next_token_support(
            emitted_text="N",
            successor_atom=AtomId(2),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CLOSURE_OPEN
            ),
            residual_key=key_n,
        )
        choice_c = self._test_choice_snapshot_entry_from_supports(
            "C",
            (support_c,),
        )
        choice_n = self._test_choice_snapshot_entry_from_supports(
            "N",
            (support_n,),
        )
        schedule_outcome = writer_frontier_module._WriterFrontierScheduleOutcome(
            state_outcomes=(),
            terminal_by_key=Counter(),
            grouped_by_text={},
            weighted_by_text={},
            next_token_frontier=(
                choice_c.next_token_entry,
                choice_n.next_token_entry,
            ),
        )
        choice_snapshot = writer_frontier_module._WriterFrontierChoiceSnapshot(
            schedule_outcome=schedule_outcome,
            terminal=None,
            choices=(choice_c, choice_n),
        )

        self.assertEqual(
            tuple(
                evidence.choice
                for evidence in (
                    choice_snapshot.choice_residual_attachment_evidence
                )
            ),
            (choice_c, choice_n),
        )
        c_evidence = (
            choice_snapshot
            .choice_residual_attachment_evidence_for_emitted_text("C")
        )
        self.assertIsNotNone(c_evidence)
        self.assertIs(c_evidence.choice, choice_c)
        self.assertIsNone(
            choice_snapshot
            .choice_residual_attachment_evidence_for_emitted_text("missing")
        )

    def test_writer_frontier_choice_snapshot_exposes_unsupported_owner_scope_choice_evidence(self) -> None:
        decision = (
            self._dead_closure_open_graph_policy_decision_for_owner_scope(
                closure_owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
                child_owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
            )
        )
        active_outcome = writer_transitions._WriterActiveEmittedScheduleOutcome(
            kind=writer_transitions._WriterActiveEmittedScheduleOutcomeKind.BLOCKED,
            graph_policy_decision=decision,
        )
        top_outcome = writer_transitions._WriterTopLevelScheduleOutcome(
            kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.BLOCKED,
            active_emitted_outcome=active_outcome,
        )
        group = (
            decision
            .unsupported_owner_scope_residual_attachment_policy_groups[0]
        )
        support_c = self._test_frontier_next_token_support(
            emitted_text="C",
            successor_atom=AtomId(1),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CYCLIC_TREE_ENTRY
            ),
            residual_key=group.key,
        )
        support_n = self._test_frontier_next_token_support(
            emitted_text="N",
            successor_atom=AtomId(2),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .ACYCLIC_TREE_ENTRY
            ),
            residual_key=writer_transitions._WriterResidualAttachmentPolicyKey(
                active_atom=AtomId(0),
                attachment_id=99,
            ),
        )
        choice_c = self._test_choice_snapshot_entry_from_supports(
            "C",
            (support_c,),
        )
        choice_n = self._test_choice_snapshot_entry_from_supports(
            "N",
            (support_n,),
        )
        schedule_outcome = writer_frontier_module._WriterFrontierScheduleOutcome(
            state_outcomes=(
                writer_frontier_module._WriterFrontierStateScheduleOutcome(
                    state_key=writer_state_key(_raw_initial_state(AtomId(0))),
                    parent_weight=1,
                    finalized_state_key=None,
                    schedule_outcome=top_outcome,
                ),
            ),
            terminal_by_key=Counter(),
            grouped_by_text={},
            weighted_by_text={},
            next_token_frontier=(
                choice_c.next_token_entry,
                choice_n.next_token_entry,
            ),
        )
        snapshot = writer_frontier_module._WriterFrontierChoiceSnapshot(
            schedule_outcome=schedule_outcome,
            terminal=None,
            choices=(choice_c, choice_n),
        )
        c_evidence = (
            snapshot.choice_residual_attachment_evidence_for_emitted_text("C")
        )

        self.assertEqual(
            snapshot.unsupported_owner_scope_choice_evidence,
            (c_evidence,),
        )
        self.assertEqual(
            snapshot.unsupported_owner_scope_kinds,
            (
                writer_transitions
                ._WriterResidualAttachmentOwnerScopeKind
                .BRANCH_RETURN,
            ),
        )

    def test_writer_frontier_choice_snapshot_residual_evidence_lookup_rejects_duplicate_emitted_text(self) -> None:
        key = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=7,
        )
        support_a = self._test_frontier_next_token_support(
            emitted_text="C",
            successor_atom=AtomId(1),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CYCLIC_TREE_ENTRY
            ),
            residual_key=key,
        )
        support_b = self._test_frontier_next_token_support(
            emitted_text="C",
            successor_atom=AtomId(2),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CLOSURE_OPEN
            ),
            residual_key=key,
        )
        choice_a = self._test_choice_snapshot_entry_from_supports(
            "C",
            (support_a,),
        )
        choice_b = self._test_choice_snapshot_entry_from_supports(
            "C",
            (support_b,),
        )
        choice_snapshot = writer_frontier_module._WriterFrontierChoiceSnapshot(
            schedule_outcome=writer_frontier_module._WriterFrontierScheduleOutcome(
                state_outcomes=(),
                terminal_by_key=Counter(),
                grouped_by_text={},
                weighted_by_text={},
                next_token_frontier=(
                    choice_a.next_token_entry,
                    choice_b.next_token_entry,
                ),
            ),
            terminal=None,
            choices=(choice_a, choice_b),
        )

        with self.assertRaises(SouthStarError) as raised:
            (
                choice_snapshot
                .choice_residual_attachment_evidence_for_emitted_text("C")
            )

        self.assertIs(
            raised.exception.kind,
            SouthStarErrorKind.INTERNAL_INVARIANT,
        )

    def test_writer_frontier_choice_snapshot_exposes_dead_closure_resolved_cyclic_choice_evidence(self) -> None:
        policy, active_outcome = self._dead_closure_open_active_emitted_outcome(
            AtomId(0),
        )
        policy_group = policy.resolved_residual_attachment_policy_groups[0]
        cyclic_support = self._test_frontier_next_token_support(
            emitted_text="C",
            successor_atom=AtomId(1),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CYCLIC_TREE_ENTRY
            ),
            residual_key=policy_group.key,
        )
        plain_support = self._test_frontier_next_token_support(
            emitted_text="N",
            successor_atom=AtomId(2),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .ACYCLIC_TREE_ENTRY
            ),
            residual_key=(
                writer_transitions._WriterResidualAttachmentPolicyKey(
                    active_atom=AtomId(0),
                    attachment_id=99,
                )
            ),
        )
        choice_c = self._test_choice_snapshot_entry_from_supports(
            "C",
            (cyclic_support,),
        )
        choice_n = self._test_choice_snapshot_entry_from_supports(
            "N",
            (plain_support,),
        )
        top_outcome = writer_transitions._WriterTopLevelScheduleOutcome(
            kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.SCHEDULED,
            schedule_decision=writer_transitions._top_level_active_emitted_decision(
                active_outcome.schedule_decision,
            ),
            active_emitted_outcome=active_outcome,
        )
        schedule_outcome = writer_frontier_module._WriterFrontierScheduleOutcome(
            state_outcomes=(
                writer_frontier_module._WriterFrontierStateScheduleOutcome(
                    state_key=writer_state_key(_raw_initial_state(AtomId(0))),
                    parent_weight=1,
                    finalized_state_key=None,
                    schedule_outcome=top_outcome,
                ),
            ),
            terminal_by_key=Counter(),
            grouped_by_text={},
            weighted_by_text={},
            next_token_frontier=(
                choice_c.next_token_entry,
                choice_n.next_token_entry,
            ),
        )
        snapshot = writer_frontier_module._WriterFrontierChoiceSnapshot(
            schedule_outcome=schedule_outcome,
            terminal=None,
            choices=(choice_c, choice_n),
        )
        c_evidence = (
            snapshot.choice_residual_attachment_evidence_for_emitted_text("C")
        )

        self.assertEqual(
            (
                snapshot
                .dead_closure_open_resolved_cyclic_tree_entry_choice_evidence
            ),
            (c_evidence,),
        )

    def test_writer_frontier_schedule_outcome_records_blocked_state_without_raising(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        base_cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        cursor = WriterFrontierCursor(
            weighted_states=base_cursor.weighted_states[:1],
        )
        active_outcome = self._blocked_child_active_emitted_outcome(AtomId(0))
        blocked_top_level_outcome = (
            writer_transitions._WriterTopLevelScheduleOutcome(
                kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.BLOCKED,
                active_emitted_outcome=active_outcome,
            )
        )

        with patch(
            "grimace._south_star1.writer_frontier._legal_writer_schedule_outcome",
            return_value=blocked_top_level_outcome,
        ):
            outcome = writer_frontier_module._writer_frontier_schedule_outcome(
                prepared,
                cursor,
            )

        self.assertTrue(outcome.blocked)
        self.assertEqual(len(outcome.blocked_state_outcomes), 1)
        self.assertIs(
            outcome.blocked_state_outcomes[0].schedule_outcome,
            blocked_top_level_outcome,
        )
        self.assertEqual(
            outcome.graph_policy_blockers,
            blocked_top_level_outcome.graph_policy_blockers,
        )
        self.assertEqual(outcome.next_token_frontier, ())
        self.assertEqual(outcome.next_token_supports, ())
        self.assertEqual(outcome.grouped_by_text, {})
        self.assertEqual(outcome.weighted_by_text, {})

    def test_writer_frontier_schedule_outcome_can_stop_after_first_blocked_state(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        self.assertGreater(len(cursor.weighted_states), 1)
        active_outcome = self._blocked_child_active_emitted_outcome(AtomId(0))
        blocked_top_level_outcome = (
            writer_transitions._WriterTopLevelScheduleOutcome(
                kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.BLOCKED,
                active_emitted_outcome=active_outcome,
            )
        )

        with patch(
            "grimace._south_star1.writer_frontier._legal_writer_schedule_outcome",
            side_effect=(
                blocked_top_level_outcome,
                AssertionError("second state should not be scheduled"),
            ),
        ):
            outcome = writer_frontier_module._writer_frontier_schedule_outcome(
                prepared,
                cursor,
                stop_after_first_blocked=True,
            )

        self.assertEqual(len(outcome.state_outcomes), 1)
        self.assertTrue(outcome.blocked)

    def test_group_writer_frontier_transitions_raises_from_blocked_frontier_schedule_outcome(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        active_outcome = self._blocked_child_active_emitted_outcome(AtomId(0))
        blocked_top_level_outcome = (
            writer_transitions._WriterTopLevelScheduleOutcome(
                kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.BLOCKED,
                active_emitted_outcome=active_outcome,
            )
        )
        state_outcome = writer_frontier_module._WriterFrontierStateScheduleOutcome(
            state_key=cursor.weighted_states[0][0],
            parent_weight=cursor.weighted_states[0][1],
            finalized_state_key=None,
            schedule_outcome=blocked_top_level_outcome,
        )
        outcome = writer_frontier_module._WriterFrontierScheduleOutcome(
            state_outcomes=(state_outcome,),
            terminal_by_key=Counter(),
            grouped_by_text={},
            weighted_by_text={},
        )

        with patch(
            "grimace._south_star1.writer_frontier._writer_frontier_schedule_outcome",
            return_value=outcome,
        ):
            with self.assertRaises(SouthStarError) as raised:
                writer_frontier_module._group_writer_frontier_transitions(
                    prepared,
                    cursor,
                )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.UNSUPPORTED_POLICY)

    def test_group_writer_frontier_transitions_returns_grouped_transitions_from_frontier_outcome(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        terminal_by_key: Counter[WriterStateKey] = Counter()
        grouped_by_text = {"C": {cursor.weighted_states[0][0]}}
        weighted_by_text = {
            "C": Counter({cursor.weighted_states[0][0]: 2}),
        }
        outcome = writer_frontier_module._WriterFrontierScheduleOutcome(
            state_outcomes=(),
            terminal_by_key=terminal_by_key,
            grouped_by_text=grouped_by_text,
            weighted_by_text=weighted_by_text,
        )

        with patch(
            "grimace._south_star1.writer_frontier._writer_frontier_schedule_outcome",
            return_value=outcome,
        ):
            grouped = writer_frontier_module._group_writer_frontier_transitions(
                prepared,
                cursor,
            )

        self.assertEqual(grouped, outcome.grouped_transitions)

    def test_group_writer_frontier_transitions_returns_checked_outcome_grouped_projection(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        grouped_by_text = {"C": {cursor.weighted_states[0][0]}}
        weighted_by_text = {
            "C": Counter({cursor.weighted_states[0][0]: 2}),
        }
        outcome = writer_frontier_module._WriterFrontierScheduleOutcome(
            state_outcomes=(),
            terminal_by_key=Counter(),
            grouped_by_text=grouped_by_text,
            weighted_by_text=weighted_by_text,
        )

        with patch(
            (
                "grimace._south_star1.writer_frontier"
                "._checked_writer_frontier_schedule_outcome"
            ),
            return_value=outcome,
        ):
            grouped = writer_frontier_module._group_writer_frontier_transitions(
                prepared,
                cursor,
            )

        self.assertEqual(grouped, outcome.grouped_transitions)

    def test_writer_frontier_next_token_entries_group_supports_by_emitted_text(self) -> None:
        parent_key = writer_state_key(_raw_initial_state(AtomId(0)))
        successor_a = writer_state_key(_raw_initial_state(AtomId(1)))
        successor_b = writer_state_key(_raw_initial_state(AtomId(2)))
        cyclic_family = (
            writer_transitions._WriterGraphPolicyActionFamily.CYCLIC_TREE_ENTRY
        )
        closure_family = (
            writer_transitions._WriterGraphPolicyActionFamily.CLOSURE_OPEN
        )
        support_1 = writer_frontier_module._WriterFrontierNextTokenSupport(
            state_key=parent_key,
            parent_weight=2,
            schedule_support=SimpleNamespace(
                emitted_text="C",
                graph_action_surface=object(),
                policy_family=cyclic_family,
            ),  # type: ignore[arg-type]
            successor_key=successor_a,
        )
        support_2 = writer_frontier_module._WriterFrontierNextTokenSupport(
            state_key=parent_key,
            parent_weight=3,
            schedule_support=SimpleNamespace(
                emitted_text="N",
                graph_action_surface=object(),
                policy_family=closure_family,
            ),  # type: ignore[arg-type]
            successor_key=successor_b,
        )
        support_3 = writer_frontier_module._WriterFrontierNextTokenSupport(
            state_key=parent_key,
            parent_weight=5,
            schedule_support=SimpleNamespace(
                emitted_text="C",
                graph_action_surface=object(),
                policy_family=cyclic_family,
            ),  # type: ignore[arg-type]
            successor_key=successor_a,
        )

        entries = (
            writer_frontier_module
            ._writer_frontier_next_token_entries_from_supports(
                (support_1, support_2, support_3)
            )
        )

        self.assertEqual(tuple(entry.emitted_text for entry in entries), ("C", "N"))
        self.assertEqual(entries[0].supports, (support_1, support_3))
        self.assertEqual(entries[1].supports, (support_2,))
        self.assertEqual(entries[0].successor_keys, frozenset({successor_a}))
        self.assertEqual(entries[0].weighted_successors[successor_a], 7)
        self.assertEqual(entries[0].immediate_multiplicity, 7)
        self.assertEqual(entries[0].policy_families, (cyclic_family, cyclic_family))

    def test_successors_from_next_token_frontier_sort_by_emitted_text(self) -> None:
        parent_key = writer_state_key(_raw_initial_state(AtomId(0)))
        c_successor = writer_state_key(_raw_initial_state(AtomId(1)))
        n_successor = writer_state_key(_raw_initial_state(AtomId(2)))
        family = writer_transitions._WriterGraphPolicyActionFamily.ACYCLIC_TREE_ENTRY
        n_support = writer_frontier_module._WriterFrontierNextTokenSupport(
            state_key=parent_key,
            parent_weight=3,
            schedule_support=SimpleNamespace(
                emitted_text="N",
                graph_action_surface=object(),
                policy_family=family,
            ),  # type: ignore[arg-type]
            successor_key=n_successor,
        )
        c_support = writer_frontier_module._WriterFrontierNextTokenSupport(
            state_key=parent_key,
            parent_weight=2,
            schedule_support=SimpleNamespace(
                emitted_text="C",
                graph_action_surface=object(),
                policy_family=family,
            ),  # type: ignore[arg-type]
            successor_key=c_successor,
        )
        n_entry = writer_frontier_module._WriterFrontierNextTokenEntry(
            emitted_text="N",
            supports=(n_support,),
        )
        c_entry = writer_frontier_module._WriterFrontierNextTokenEntry(
            emitted_text="C",
            supports=(c_support,),
        )

        successors = writer_frontier_module._successors_from_next_token_frontier(
            (n_entry, c_entry)
        )

        self.assertEqual(tuple(text for text, _ in successors), ("C", "N"))
        self.assertEqual(
            successors[0][1].weighted_states,
            tuple(c_entry.weighted_successors.items()),
        )
        self.assertEqual(
            successors[1][1].weighted_states,
            tuple(n_entry.weighted_successors.items()),
        )

    def test_writer_frontier_schedule_outcome_grouped_transitions_use_next_token_supports(self) -> None:
        parent_key = writer_state_key(_raw_initial_state(AtomId(0)))
        successor_key = writer_state_key(_raw_initial_state(AtomId(1)))
        support = writer_frontier_module._WriterFrontierNextTokenSupport(
            state_key=parent_key,
            parent_weight=4,
            schedule_support=SimpleNamespace(
                emitted_text="C",
                graph_action_surface=object(),
                policy_family=(
                    writer_transitions
                    ._WriterGraphPolicyActionFamily
                    .ACYCLIC_TREE_ENTRY
                ),
            ),  # type: ignore[arg-type]
            successor_key=successor_key,
        )
        entry = writer_frontier_module._WriterFrontierNextTokenEntry(
            emitted_text="C",
            supports=(support,),
        )
        outcome = writer_frontier_module._WriterFrontierScheduleOutcome(
            state_outcomes=(),
            terminal_by_key=Counter(),
            grouped_by_text={"stale": set()},
            weighted_by_text={"stale": Counter()},
            next_token_frontier=(entry,),
        )

        grouped = outcome.grouped_transitions

        self.assertEqual(grouped.grouped_by_text, {"C": {successor_key}})
        self.assertEqual(
            grouped.weighted_by_text,
            {"C": Counter({successor_key: 4})},
        )

    def test_writer_frontier_schedule_outcome_grouped_transitions_fallback_without_next_token_frontier(self) -> None:
        successor_key = writer_state_key(_raw_initial_state(AtomId(1)))
        outcome = writer_frontier_module._WriterFrontierScheduleOutcome(
            state_outcomes=(),
            terminal_by_key=Counter(),
            grouped_by_text={"C": {successor_key}},
            weighted_by_text={"C": Counter({successor_key: 2})},
        )

        grouped = outcome.grouped_transitions

        self.assertEqual(grouped.grouped_by_text, outcome.grouped_by_text)
        self.assertEqual(grouped.weighted_by_text, outcome.weighted_by_text)

    def test_writer_frontier_schedule_outcome_records_next_token_support_provenance(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())

        outcome = writer_frontier_module._writer_frontier_schedule_outcome(
            prepared,
            cursor,
        )

        self.assertFalse(outcome.blocked)
        self.assertTrue(outcome.next_token_frontier)
        self.assertTrue(outcome.next_token_supports)

        cursor_weight_by_key = dict(cursor.weighted_states)
        selected_schedule_supports = tuple(
            support
            for state_outcome in outcome.state_outcomes
            for entry in (
                state_outcome
                .schedule_outcome
                .selected_next_token_frontier
            )
            for support in entry.supports
        )
        support = outcome.next_token_supports[0]

        self.assertIn(support.state_key, cursor_weight_by_key)
        self.assertEqual(
            support.parent_weight,
            cursor_weight_by_key[support.state_key],
        )
        self.assertIn(support.schedule_support, selected_schedule_supports)
        self.assertEqual(
            support.successor_key,
            writer_state_key(support.schedule_support.transition.successor),
        )
        self.assertEqual(
            support.emitted_text,
            support.schedule_support.emitted_text,
        )
        self.assertEqual(
            outcome.grouped_by_text_from_next_token_frontier,
            outcome.grouped_by_text,
        )
        self.assertEqual(
            outcome.weighted_by_text_from_next_token_frontier,
            outcome.weighted_by_text,
        )
        self.assertEqual(
            outcome.grouped_transitions.grouped_by_text,
            outcome.grouped_by_text_from_next_token_frontier,
        )
        self.assertEqual(
            outcome.grouped_transitions.weighted_by_text,
            outcome.weighted_by_text_from_next_token_frontier,
        )

    def test_writer_frontier_schedule_outcome_terminal_only_has_no_next_token_supports(self) -> None:
        key = writer_state_key(_raw_initial_state(AtomId(0)))
        outcome = writer_frontier_module._WriterFrontierScheduleOutcome(
            state_outcomes=(),
            terminal_by_key=Counter({key: 2}),
            grouped_by_text={},
            weighted_by_text={},
        )

        self.assertEqual(outcome.terminal_by_key, Counter({key: 2}))
        self.assertEqual(outcome.next_token_frontier, ())
        self.assertEqual(outcome.next_token_supports, ())

    def test_writer_frontier_schedule_outcome_grouping_validation_rejects_mismatch(self) -> None:
        parent_key = writer_state_key(_raw_initial_state(AtomId(0)))
        successor_key = writer_state_key(_raw_initial_state(AtomId(1)))
        stale_key = writer_state_key(_raw_initial_state(AtomId(2)))
        support = writer_frontier_module._WriterFrontierNextTokenSupport(
            state_key=parent_key,
            parent_weight=2,
            schedule_support=SimpleNamespace(
                emitted_text="C",
                graph_action_surface=object(),
                policy_family=(
                    writer_transitions
                    ._WriterGraphPolicyActionFamily
                    .ACYCLIC_TREE_ENTRY
                ),
            ),  # type: ignore[arg-type]
            successor_key=successor_key,
        )
        entry = writer_frontier_module._WriterFrontierNextTokenEntry(
            emitted_text="C",
            supports=(support,),
        )
        grouped_mismatch = writer_frontier_module._WriterFrontierScheduleOutcome(
            state_outcomes=(),
            terminal_by_key=Counter(),
            grouped_by_text={"C": {stale_key}},
            weighted_by_text={"C": Counter({successor_key: 2})},
            next_token_frontier=(entry,),
        )

        with self.assertRaises(SouthStarError) as grouped_raised:
            (
                writer_frontier_module
                ._validate_writer_frontier_schedule_outcome_grouping(
                    grouped_mismatch
                )
            )

        self.assertIs(
            grouped_raised.exception.kind,
            SouthStarErrorKind.INTERNAL_INVARIANT,
        )

        weighted_mismatch = writer_frontier_module._WriterFrontierScheduleOutcome(
            state_outcomes=(),
            terminal_by_key=Counter(),
            grouped_by_text={"C": {successor_key}},
            weighted_by_text={"C": Counter({successor_key: 3})},
            next_token_frontier=(entry,),
        )

        with self.assertRaises(SouthStarError) as weighted_raised:
            (
                writer_frontier_module
                ._validate_writer_frontier_schedule_outcome_grouping(
                    weighted_mismatch
                )
            )

        self.assertIs(
            weighted_raised.exception.kind,
            SouthStarErrorKind.INTERNAL_INVARIANT,
        )

    def test_writer_frontier_choice_snapshot_entry_converts_to_public_choice(self) -> None:
        parent_key = writer_state_key(_raw_initial_state(AtomId(0)))
        successor_key = writer_state_key(_raw_initial_state(AtomId(1)))
        family = writer_transitions._WriterGraphPolicyActionFamily.ACYCLIC_TREE_ENTRY
        support = writer_frontier_module._WriterFrontierNextTokenSupport(
            state_key=parent_key,
            parent_weight=3,
            schedule_support=SimpleNamespace(
                emitted_text="C",
                graph_action_surface=object(),
                policy_family=family,
            ),  # type: ignore[arg-type]
            successor_key=successor_key,
        )
        next_token_entry = writer_frontier_module._WriterFrontierNextTokenEntry(
            emitted_text="C",
            supports=(support,),
        )
        successor = WriterFrontierCursor(
            weighted_states=tuple(next_token_entry.weighted_successors.items())
        )
        entry = writer_frontier_module._WriterFrontierChoiceSnapshotEntry(
            next_token_entry=next_token_entry,
            successor=successor,
            support_count=5,
            completion_count=8,
        )

        self.assertEqual(entry.emitted_text, next_token_entry.emitted_text)
        self.assertEqual(
            entry.immediate_multiplicity,
            next_token_entry.immediate_multiplicity,
        )
        self.assertEqual(entry.supports, next_token_entry.supports)
        self.assertEqual(entry.successor_keys, next_token_entry.successor_keys)
        self.assertEqual(
            entry.weighted_successors,
            next_token_entry.weighted_successors,
        )

        public = entry.to_public_choice()

        self.assertEqual(public.emitted_text, entry.emitted_text)
        self.assertIs(public.successor, entry.successor)
        self.assertEqual(
            public.immediate_multiplicity,
            entry.immediate_multiplicity,
        )
        self.assertEqual(public.support_count, entry.support_count)
        self.assertEqual(public.completion_count, entry.completion_count)

    def test_writer_frontier_choice_snapshot_builds_terminal_from_schedule_outcome(self) -> None:
        prepared = _prepare(chain_facts(("C",)))
        final_key = writer_state_key(_raw_initial_state(AtomId(0)))
        outcome = writer_frontier_module._WriterFrontierScheduleOutcome(
            state_outcomes=(),
            terminal_by_key=Counter({final_key: 3}),
            grouped_by_text={},
            weighted_by_text={},
        )

        snapshot = (
            writer_frontier_module
            ._writer_frontier_choice_snapshot_from_schedule_outcome(
                prepared,
                outcome,
            )
        )

        self.assertIsNotNone(snapshot.terminal)
        assert snapshot.terminal is not None
        self.assertEqual(snapshot.terminal.multiplicity, 3)
        self.assertEqual(snapshot.terminal.completion_count, 3)
        self.assertEqual(
            snapshot.terminal.finalized_cursor.weighted_states,
            ((final_key, 3),),
        )
        self.assertEqual(snapshot.public_choices.terminal, snapshot.terminal)

    def test_writer_frontier_choice_snapshot_sorts_choices_by_emitted_text(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        parent_key = writer_state_key(_raw_initial_state(AtomId(0)))
        c_successor = writer_state_key(_raw_initial_state(AtomId(1)))
        n_successor = writer_state_key(_raw_initial_state(AtomId(2)))
        family = writer_transitions._WriterGraphPolicyActionFamily.ACYCLIC_TREE_ENTRY
        n_support = writer_frontier_module._WriterFrontierNextTokenSupport(
            state_key=parent_key,
            parent_weight=3,
            schedule_support=SimpleNamespace(
                emitted_text="N",
                graph_action_surface=object(),
                policy_family=family,
            ),  # type: ignore[arg-type]
            successor_key=n_successor,
        )
        c_support = writer_frontier_module._WriterFrontierNextTokenSupport(
            state_key=parent_key,
            parent_weight=2,
            schedule_support=SimpleNamespace(
                emitted_text="C",
                graph_action_surface=object(),
                policy_family=family,
            ),  # type: ignore[arg-type]
            successor_key=c_successor,
        )
        outcome = writer_frontier_module._WriterFrontierScheduleOutcome(
            state_outcomes=(),
            terminal_by_key=Counter(),
            grouped_by_text={
                "N": {n_successor},
                "C": {c_successor},
            },
            weighted_by_text={
                "N": Counter({n_successor: 3}),
                "C": Counter({c_successor: 2}),
            },
            next_token_frontier=(
                writer_frontier_module._WriterFrontierNextTokenEntry(
                    emitted_text="N",
                    supports=(n_support,),
                ),
                writer_frontier_module._WriterFrontierNextTokenEntry(
                    emitted_text="C",
                    supports=(c_support,),
                ),
            ),
        )

        with patch(
            "grimace._south_star1.writer_frontier._count_writer_frontier_support",
            return_value=1,
        ), patch(
            (
                "grimace._south_star1.writer_frontier"
                "._count_weighted_successor_completions"
            ),
            return_value=2,
        ):
            snapshot = (
                writer_frontier_module
                ._writer_frontier_choice_snapshot_from_schedule_outcome(
                    prepared,
                    outcome,
                )
            )

        self.assertEqual(
            tuple(choice.emitted_text for choice in snapshot.choices),
            ("C", "N"),
        )
        self.assertEqual(
            tuple(choice.emitted_text for choice in snapshot.public_choices.choices),
            ("C", "N"),
        )

    def test_writer_frontier_choice_snapshot_can_omit_counts(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        parent_key = writer_state_key(_raw_initial_state(AtomId(0)))
        successor_key = writer_state_key(_raw_initial_state(AtomId(1)))
        support = writer_frontier_module._WriterFrontierNextTokenSupport(
            state_key=parent_key,
            parent_weight=2,
            schedule_support=SimpleNamespace(
                emitted_text="C",
                graph_action_surface=object(),
                policy_family=(
                    writer_transitions
                    ._WriterGraphPolicyActionFamily
                    .ACYCLIC_TREE_ENTRY
                ),
            ),  # type: ignore[arg-type]
            successor_key=successor_key,
        )
        outcome = writer_frontier_module._WriterFrontierScheduleOutcome(
            state_outcomes=(),
            terminal_by_key=Counter(),
            grouped_by_text={"C": {successor_key}},
            weighted_by_text={"C": Counter({successor_key: 2})},
            next_token_frontier=(
                writer_frontier_module._WriterFrontierNextTokenEntry(
                    emitted_text="C",
                    supports=(support,),
                ),
            ),
        )

        with patch(
            "grimace._south_star1.writer_frontier._count_writer_frontier_support",
            side_effect=AssertionError("snapshot counted support"),
        ), patch(
            (
                "grimace._south_star1.writer_frontier"
                "._count_weighted_successor_completions"
            ),
            side_effect=AssertionError("snapshot counted completions"),
        ):
            snapshot = (
                writer_frontier_module
                ._writer_frontier_choice_snapshot_from_schedule_outcome(
                    prepared,
                    outcome,
                    include_counts=False,
                )
            )

        self.assertTrue(snapshot.choices)
        self.assertTrue(
            all(choice.support_count is None for choice in snapshot.choices)
        )
        self.assertTrue(
            all(choice.completion_count is None for choice in snapshot.choices)
        )

    def test_writer_frontier_choice_snapshot_records_blocked_outcome_without_counting(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        active_outcome = self._blocked_child_active_emitted_outcome(AtomId(0))
        blocked_top_level_outcome = (
            writer_transitions._WriterTopLevelScheduleOutcome(
                kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.BLOCKED,
                active_emitted_outcome=active_outcome,
            )
        )
        state_outcome = writer_frontier_module._WriterFrontierStateScheduleOutcome(
            state_key=writer_state_key(_raw_initial_state(AtomId(0))),
            parent_weight=1,
            finalized_state_key=None,
            schedule_outcome=blocked_top_level_outcome,
        )
        outcome = writer_frontier_module._WriterFrontierScheduleOutcome(
            state_outcomes=(state_outcome,),
            terminal_by_key=Counter(),
            grouped_by_text={},
            weighted_by_text={},
        )

        with patch(
            "grimace._south_star1.writer_frontier._count_writer_frontier_support",
            side_effect=AssertionError("blocked snapshot counted support"),
        ), patch(
            (
                "grimace._south_star1.writer_frontier"
                "._count_weighted_successor_completions"
            ),
            side_effect=AssertionError("blocked snapshot counted completions"),
        ):
            snapshot = (
                writer_frontier_module
                ._writer_frontier_choice_snapshot_from_schedule_outcome(
                    prepared,
                    outcome,
                )
            )

        self.assertTrue(snapshot.blocked)
        self.assertEqual(
            snapshot.graph_policy_blockers,
            outcome.graph_policy_blockers,
        )
        self.assertEqual(snapshot.choices, ())
        self.assertEqual(snapshot.public_choices.choices, ())

    def test_writer_frontier_choice_snapshot_returns_scheduled_snapshot(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())

        snapshot = writer_frontier_module._writer_frontier_choice_snapshot(
            prepared,
            cursor,
        )

        self.assertFalse(snapshot.blocked)
        self.assertEqual(snapshot.graph_policy_blockers, ())
        self.assertFalse(snapshot.schedule_outcome.blocked)
        self.assertTrue(snapshot.choices)
        self.assertEqual(
            snapshot.public_choices,
            writer_frontier_choices(prepared, cursor),
        )

    def test_writer_frontier_choice_snapshot_returns_blocked_snapshot_without_raising(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        active_outcome = self._blocked_child_active_emitted_outcome(AtomId(0))
        blocked_top_level_outcome = (
            writer_transitions._WriterTopLevelScheduleOutcome(
                kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.BLOCKED,
                active_emitted_outcome=active_outcome,
            )
        )
        state_outcome = writer_frontier_module._WriterFrontierStateScheduleOutcome(
            state_key=cursor.weighted_states[0][0],
            parent_weight=cursor.weighted_states[0][1],
            finalized_state_key=None,
            schedule_outcome=blocked_top_level_outcome,
        )
        blocked_outcome = writer_frontier_module._WriterFrontierScheduleOutcome(
            state_outcomes=(state_outcome,),
            terminal_by_key=Counter(),
            grouped_by_text={},
            weighted_by_text={},
        )

        with patch(
            "grimace._south_star1.writer_frontier._writer_frontier_schedule_outcome",
            return_value=blocked_outcome,
        ):
            snapshot = writer_frontier_module._writer_frontier_choice_snapshot(
                prepared,
                cursor,
            )

        self.assertTrue(snapshot.blocked)
        self.assertIs(snapshot.schedule_outcome, blocked_outcome)
        self.assertEqual(
            snapshot.graph_policy_blockers,
            blocked_outcome.graph_policy_blockers,
        )
        self.assertEqual(snapshot.blocked_state_outcomes, (state_outcome,))
        self.assertEqual(snapshot.choices, ())
        self.assertEqual(snapshot.public_choices.choices, ())

    def test_writer_frontier_choice_snapshot_forwards_include_counts(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        outcome = writer_frontier_module._WriterFrontierScheduleOutcome(
            state_outcomes=(),
            terminal_by_key=Counter(),
            grouped_by_text={},
            weighted_by_text={},
        )
        snapshot = writer_frontier_module._WriterFrontierChoiceSnapshot(
            schedule_outcome=outcome,
            terminal=None,
            choices=(),
        )

        with patch(
            "grimace._south_star1.writer_frontier._writer_frontier_schedule_outcome",
            return_value=outcome,
        ), patch(
            (
                "grimace._south_star1.writer_frontier"
                "._writer_frontier_choice_snapshot_from_schedule_outcome"
            ),
            return_value=snapshot,
        ) as build_snapshot:
            writer_frontier_module._writer_frontier_choice_snapshot(
                prepared,
                cursor,
                include_counts=False,
            )
            writer_frontier_module._writer_frontier_choice_snapshot(
                prepared,
                cursor,
                include_counts=True,
            )

        self.assertEqual(
            tuple(call.kwargs["include_counts"] for call in build_snapshot.call_args_list),
            (False, True),
        )

    def test_writer_frontier_choice_snapshot_forwards_stop_after_first_blocked(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        outcome = writer_frontier_module._WriterFrontierScheduleOutcome(
            state_outcomes=(),
            terminal_by_key=Counter(),
            grouped_by_text={},
            weighted_by_text={},
        )

        with patch(
            "grimace._south_star1.writer_frontier._writer_frontier_schedule_outcome",
            return_value=outcome,
        ) as schedule_outcome:
            writer_frontier_module._writer_frontier_choice_snapshot(
                prepared,
                cursor,
                stop_after_first_blocked=True,
            )

        schedule_outcome.assert_called_once_with(
            prepared,
            cursor,
            stop_after_first_blocked=True,
        )

    def test_checked_writer_frontier_choice_snapshot_raises_from_blocked_unchecked_snapshot(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        active_outcome = self._blocked_child_active_emitted_outcome(AtomId(0))
        blocked_top_level_outcome = (
            writer_transitions._WriterTopLevelScheduleOutcome(
                kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.BLOCKED,
                active_emitted_outcome=active_outcome,
            )
        )
        blocked_outcome = writer_frontier_module._WriterFrontierScheduleOutcome(
            state_outcomes=(
                writer_frontier_module._WriterFrontierStateScheduleOutcome(
                    state_key=cursor.weighted_states[0][0],
                    parent_weight=cursor.weighted_states[0][1],
                    finalized_state_key=None,
                    schedule_outcome=blocked_top_level_outcome,
                ),
            ),
            terminal_by_key=Counter(),
            grouped_by_text={},
            weighted_by_text={},
        )
        snapshot = writer_frontier_module._WriterFrontierChoiceSnapshot(
            schedule_outcome=blocked_outcome,
            terminal=None,
            choices=(),
        )

        with patch(
            "grimace._south_star1.writer_frontier._writer_frontier_choice_snapshot",
            return_value=snapshot,
        ) as unchecked_snapshot:
            with self.assertRaises(SouthStarError) as raised:
                writer_frontier_module._checked_writer_frontier_choice_snapshot(
                    prepared,
                    cursor,
                )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.UNSUPPORTED_POLICY)
        unchecked_snapshot.assert_called_once_with(
            prepared,
            cursor,
            include_counts=True,
            stop_after_first_blocked=True,
        )

    def test_checked_writer_frontier_choice_snapshot_returns_scheduled_unchecked_snapshot(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        snapshot = writer_frontier_module._WriterFrontierChoiceSnapshot(
            schedule_outcome=writer_frontier_module._WriterFrontierScheduleOutcome(
                state_outcomes=(),
                terminal_by_key=Counter(),
                grouped_by_text={},
                weighted_by_text={},
            ),
            terminal=None,
            choices=(),
        )

        with patch(
            "grimace._south_star1.writer_frontier._writer_frontier_choice_snapshot",
            return_value=snapshot,
        ):
            result = writer_frontier_module._checked_writer_frontier_choice_snapshot(
                prepared,
                cursor,
            )

        self.assertIs(result, snapshot)

    def test_writer_choice_snapshot_from_snapshot_returns_unchecked_choice_snapshot(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )

        choice_snapshot = (
            writer_snapshot
            ._writer_frontier_choice_snapshot_from_snapshot(
                snapshot,
                prepared=prepared,
                include_counts=False,
            )
        )

        self.assertFalse(choice_snapshot.blocked)
        self.assertTrue(choice_snapshot.choices)
        self.assertTrue(
            all(choice.support_count is None for choice in choice_snapshot.choices)
        )
        self.assertTrue(
            all(choice.completion_count is None for choice in choice_snapshot.choices)
        )

    def test_writer_choice_snapshot_from_snapshot_returns_blocked_snapshot_without_raising(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        active_outcome = self._blocked_child_active_emitted_outcome(AtomId(0))
        blocked_top_level_outcome = (
            writer_transitions._WriterTopLevelScheduleOutcome(
                kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.BLOCKED,
                active_emitted_outcome=active_outcome,
            )
        )
        blocked_outcome = writer_frontier_module._WriterFrontierScheduleOutcome(
            state_outcomes=(
                writer_frontier_module._WriterFrontierStateScheduleOutcome(
                    state_key=cursor.weighted_states[0][0],
                    parent_weight=cursor.weighted_states[0][1],
                    finalized_state_key=None,
                    schedule_outcome=blocked_top_level_outcome,
                ),
            ),
            terminal_by_key=Counter(),
            grouped_by_text={},
            weighted_by_text={},
        )
        blocked_choice_snapshot = writer_frontier_module._WriterFrontierChoiceSnapshot(
            schedule_outcome=blocked_outcome,
            terminal=None,
            choices=(),
        )

        with patch(
            "grimace._south_star1.writer_snapshot._writer_frontier_choice_snapshot",
            return_value=blocked_choice_snapshot,
        ) as build_snapshot:
            result = writer_snapshot._writer_frontier_choice_snapshot_from_snapshot(
                snapshot,
                prepared=prepared,
            )

        self.assertIs(result, blocked_choice_snapshot)
        self.assertTrue(result.blocked)
        build_snapshot.assert_called_once_with(
            prepared,
            cursor,
            include_counts=True,
            stop_after_first_blocked=False,
        )

    def test_checked_writer_choice_snapshot_from_snapshot_raises_from_blocked_snapshot(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )

        with patch(
            (
                "grimace._south_star1.writer_snapshot"
                "._checked_writer_frontier_choice_snapshot"
            ),
            side_effect=SouthStarError(
                SouthStarErrorKind.UNSUPPORTED_POLICY,
                "blocked",
            ),
        ):
            with self.assertRaises(SouthStarError) as raised:
                (
                    writer_snapshot
                    ._checked_writer_frontier_choice_snapshot_from_snapshot(
                        snapshot,
                        prepared=prepared,
                    )
                )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.UNSUPPORTED_POLICY)

    def test_writer_frontier_choices_after_emitted_texts_routes_through_checked_replayed_prefix_snapshot(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        successor_key = cursor.weighted_states[0][0]
        support = writer_frontier_module._WriterFrontierNextTokenSupport(
            state_key=successor_key,
            parent_weight=1,
            schedule_support=SimpleNamespace(
                emitted_text="C",
                graph_action_surface=object(),
                policy_family=(
                    writer_transitions
                    ._WriterGraphPolicyActionFamily
                    .ACYCLIC_TREE_ENTRY
                ),
            ),  # type: ignore[arg-type]
            successor_key=successor_key,
        )
        entry = writer_frontier_module._WriterFrontierNextTokenEntry(
            emitted_text="C",
            supports=(support,),
        )
        choice = writer_frontier_module._WriterFrontierChoiceSnapshotEntry(
            next_token_entry=entry,
            successor=WriterFrontierCursor(weighted_states=((successor_key, 1),)),
            support_count=1,
            completion_count=2,
        )
        choice_snapshot = writer_frontier_module._WriterFrontierChoiceSnapshot(
            schedule_outcome=writer_frontier_module._WriterFrontierScheduleOutcome(
                state_outcomes=(),
                terminal_by_key=Counter(),
                grouped_by_text={"C": {successor_key}},
                weighted_by_text={"C": Counter({successor_key: 1})},
                next_token_frontier=(entry,),
            ),
            terminal=None,
            choices=(choice,),
        )
        first_step = self._test_writer_snapshot_advance_step(
            snapshot,
            emitted_text="C",
            prepared=prepared,
        )
        assert first_step.advanced_snapshot is not None
        sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.ADVANCED,
            source_snapshot=snapshot,
            emitted_texts=("C",),
            step_outcomes=(first_step,),
            current_snapshot=first_step.advanced_snapshot,
        )
        replay = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .CHOICE_SNAPSHOT
            ),
            source_snapshot=snapshot,
            emitted_texts=("C",),
            sequence_outcome=sequence,
            choice_snapshot=choice_snapshot,
        )
        read_outcome = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=writer_snapshot._WriterSnapshotPrefixReadOutcomeKind.READABLE,
            replay_outcome=replay,
            support_count=1,
            completion_count=2,
        )

        with patch(
            (
                "grimace._south_star1.writer_snapshot"
                "._checked_writer_snapshot_prefix_read_outcome_after_emitted_texts"
            ),
            return_value=read_outcome,
        ) as checked_read:
            choices = writer_snapshot._writer_frontier_choices_after_emitted_texts(
                snapshot,
                prepared=prepared,
                emitted_texts=("C",),
            )

        self.assertEqual(choices, choice_snapshot.public_choices)
        checked_read.assert_called_once_with(
            snapshot,
            prepared=prepared,
            emitted_texts=("C",),
            include_counts=True,
        )

    def test_resume_writer_frontier_choices_from_snapshot_routes_through_empty_prefix_choices(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        returned_choices = writer_frontier_choices(prepared, cursor)

        with patch(
            (
                "grimace._south_star1.writer_snapshot"
                "._writer_frontier_choices_after_emitted_texts"
            ),
            return_value=returned_choices,
        ) as choices_after_prefix:
            result = writer_snapshot.resume_writer_frontier_choices_from_snapshot(
                snapshot,
                prepared=prepared,
            )

        self.assertIs(result, returned_choices)
        choices_after_prefix.assert_called_once_with(
            snapshot,
            prepared=prepared,
            emitted_texts=(),
        )

    def test_resume_writer_frontier_choices_from_snapshot_matches_writer_frontier_choices(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )

        self.assertEqual(
            writer_snapshot.resume_writer_frontier_choices_from_snapshot(
                snapshot,
                prepared=prepared,
            ),
            writer_frontier_choices(prepared, cursor),
        )

    def test_writer_frontier_choices_after_empty_emitted_texts_matches_snapshot_resume(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )

        choices = writer_snapshot._writer_frontier_choices_after_emitted_texts(
            snapshot,
            prepared=prepared,
            emitted_texts=(),
        )

        self.assertEqual(
            choices,
            writer_snapshot.resume_writer_frontier_choices_from_snapshot(
                snapshot,
                prepared=prepared,
            ),
        )
        self.assertEqual(choices, writer_frontier_choices(prepared, cursor))

    def test_writer_frontier_choices_after_emitted_texts_matches_advanced_snapshot_choices(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        choices = writer_snapshot.resume_writer_frontier_choices_from_snapshot(
            snapshot,
            prepared=prepared,
        )
        first_text = choices.choices[0].emitted_text
        advanced = writer_snapshot._advance_writer_search_snapshot_by_emitted_text(
            snapshot,
            prepared=prepared,
            emitted_text=first_text,
        )

        self.assertEqual(
            writer_snapshot._writer_frontier_choices_after_emitted_texts(
                snapshot,
                prepared=prepared,
                emitted_texts=(first_text,),
            ),
            writer_snapshot.resume_writer_frontier_choices_from_snapshot(
                advanced,
                prepared=prepared,
            ),
        )

    def test_writer_frontier_choices_after_emitted_texts_raises_for_invalid_replay_token(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )

        with self.assertRaises(SouthStarError) as raised:
            writer_snapshot._writer_frontier_choices_after_emitted_texts(
                snapshot,
                prepared=prepared,
                emitted_texts=("not-a-frontier-token",),
            )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.INVALID_FACTS)

    def test_writer_frontier_choices_after_emitted_texts_raises_from_checked_replay_blockers(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )

        with patch(
            (
                "grimace._south_star1.writer_snapshot"
                "._checked_writer_snapshot_prefix_read_outcome_after_emitted_texts"
            ),
            side_effect=SouthStarError(
                SouthStarErrorKind.UNSUPPORTED_POLICY,
                "blocked",
            ),
        ):
            with self.assertRaises(SouthStarError) as raised:
                writer_snapshot._writer_frontier_choices_after_emitted_texts(
                    snapshot,
                    prepared=prepared,
                    emitted_texts=("C",),
                )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.UNSUPPORTED_POLICY)

    def test_writer_choice_snapshot_count_helpers_reject_missing_counts(self) -> None:
        choice = self._test_choice_snapshot_entry(
            "C",
            successor_atom=AtomId(1),
        )
        choice_snapshot = self._test_frontier_choice_snapshot((choice,))

        with self.assertRaises(SouthStarError) as support_raised:
            writer_snapshot._count_writer_frontier_choice_snapshot_supports(
                choice_snapshot,
            )

        self.assertIs(
            support_raised.exception.kind,
            SouthStarErrorKind.INTERNAL_INVARIANT,
        )

        with self.assertRaises(SouthStarError) as completion_raised:
            writer_snapshot._count_writer_frontier_choice_snapshot_completions(
                choice_snapshot,
            )

        self.assertIs(
            completion_raised.exception.kind,
            SouthStarErrorKind.INTERNAL_INVARIANT,
        )

    def test_count_writer_frontier_support_after_empty_emitted_texts_matches_snapshot_cursor(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )

        self.assertEqual(
            writer_snapshot._count_writer_frontier_support_after_emitted_texts(
                snapshot,
                prepared=prepared,
                emitted_texts=(),
            ),
            count_writer_frontier_support(prepared, cursor.support_state),
        )

    def test_count_writer_completions_after_empty_emitted_texts_matches_snapshot_cursor(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )

        self.assertEqual(
            writer_snapshot._count_writer_completions_after_emitted_texts(
                snapshot,
                prepared=prepared,
                emitted_texts=(),
            ),
            count_writer_cursor_completions(prepared, cursor),
        )

    def test_replayed_prefix_counts_match_advanced_snapshot_cursor(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        choices = writer_snapshot.resume_writer_frontier_choices_from_snapshot(
            snapshot,
            prepared=prepared,
        )
        first_text = choices.choices[0].emitted_text
        advanced = writer_snapshot._advance_writer_search_snapshot_by_emitted_text(
            snapshot,
            prepared=prepared,
            emitted_text=first_text,
        )

        self.assertEqual(
            writer_snapshot._count_writer_frontier_support_after_emitted_texts(
                snapshot,
                prepared=prepared,
                emitted_texts=(first_text,),
            ),
            count_writer_frontier_support(prepared, advanced.cursor.support_state),
        )
        self.assertEqual(
            writer_snapshot._count_writer_completions_after_emitted_texts(
                snapshot,
                prepared=prepared,
                emitted_texts=(first_text,),
            ),
            count_writer_cursor_completions(prepared, advanced.cursor),
        )

    def test_iter_writer_frontier_support_suffixes_after_empty_emitted_texts_matches_cursor_stream(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )

        self.assertEqual(
            tuple(
                writer_snapshot
                ._iter_writer_frontier_support_suffixes_after_emitted_texts(
                    snapshot,
                    prepared=prepared,
                    emitted_texts=(),
                )
            ),
            tuple(iter_writer_frontier_support(prepared, cursor)),
        )

    def test_iter_writer_frontier_support_suffixes_after_emitted_texts_matches_advanced_cursor_stream(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        choices = writer_snapshot.resume_writer_frontier_choices_from_snapshot(
            snapshot,
            prepared=prepared,
        )
        first_text = choices.choices[0].emitted_text
        advanced = writer_snapshot._advance_writer_search_snapshot_by_emitted_text(
            snapshot,
            prepared=prepared,
            emitted_text=first_text,
        )

        self.assertEqual(
            tuple(
                writer_snapshot
                ._iter_writer_frontier_support_suffixes_after_emitted_texts(
                    snapshot,
                    prepared=prepared,
                    emitted_texts=(first_text,),
                )
            ),
            tuple(iter_writer_frontier_support(prepared, advanced.cursor)),
        )

    def test_iter_writer_frontier_support_suffixes_after_emitted_texts_uses_uncounted_snapshot(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )

        with patch(
            (
                "grimace._south_star1.writer_snapshot"
                "._checked_writer_snapshot_prefix_read_outcome_after_emitted_texts"
            ),
            wraps=(
                writer_snapshot
                ._checked_writer_snapshot_prefix_read_outcome_after_emitted_texts
            ),
        ) as checked_read:
            tuple(
                writer_snapshot
                ._iter_writer_frontier_support_suffixes_after_emitted_texts(
                    snapshot,
                    prepared=prepared,
                    emitted_texts=(),
                )
            )

        self.assertTrue(checked_read.call_args_list)
        self.assertTrue(
            all(
                call.kwargs["include_counts"] is False
                for call in checked_read.call_args_list
            )
        )

    def test_replayed_prefix_count_helpers_use_counted_snapshot(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )

        with patch(
            (
                "grimace._south_star1.writer_snapshot"
                "._checked_writer_snapshot_prefix_read_outcome_after_emitted_texts"
            ),
            wraps=(
                writer_snapshot
                ._checked_writer_snapshot_prefix_read_outcome_after_emitted_texts
            ),
        ) as checked_read:
            writer_snapshot._count_writer_frontier_support_after_emitted_texts(
                snapshot,
                prepared=prepared,
                emitted_texts=(),
            )
            writer_snapshot._count_writer_completions_after_emitted_texts(
                snapshot,
                prepared=prepared,
                emitted_texts=(),
            )

        self.assertEqual(len(checked_read.call_args_list), 2)
        self.assertTrue(
            all(
                call.kwargs["include_counts"] is True
                for call in checked_read.call_args_list
            )
        )

    def test_replayed_prefix_support_helpers_raise_for_invalid_emitted_text(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )

        helpers = (
            writer_snapshot._count_writer_frontier_support_after_emitted_texts,
            writer_snapshot._count_writer_completions_after_emitted_texts,
        )

        for helper in helpers:
            with self.subTest(helper=helper.__name__):
                with self.assertRaises(SouthStarError) as raised:
                    helper(
                        snapshot,
                        prepared=prepared,
                        emitted_texts=("not-a-frontier-token",),
                    )

                self.assertIs(
                    raised.exception.kind,
                    SouthStarErrorKind.INVALID_FACTS,
                )

        with self.assertRaises(SouthStarError) as stream_raised:
            tuple(
                writer_snapshot
                ._iter_writer_frontier_support_suffixes_after_emitted_texts(
                    snapshot,
                    prepared=prepared,
                    emitted_texts=("not-a-frontier-token",),
                )
            )

        self.assertIs(stream_raised.exception.kind, SouthStarErrorKind.INVALID_FACTS)

    def test_writer_snapshot_prefix_read_outcome_validates_payload_shape(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        choice_snapshot = writer_snapshot._writer_frontier_choice_snapshot_from_snapshot(
            snapshot,
            prepared=prepared,
        )
        sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.ADVANCED,
            source_snapshot=snapshot,
            emitted_texts=(),
            step_outcomes=(),
            current_snapshot=snapshot,
        )
        replay = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .CHOICE_SNAPSHOT
            ),
            source_snapshot=snapshot,
            emitted_texts=(),
            sequence_outcome=sequence,
            choice_snapshot=choice_snapshot,
        )
        blocked_choice_snapshot = self._test_blocked_frontier_choice_snapshot(cursor)
        final_blocked_replay = (
            writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
                kind=(
                    writer_snapshot
                    ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                    .CHOICE_SNAPSHOT
                ),
                source_snapshot=snapshot,
                emitted_texts=(),
                sequence_outcome=sequence,
                choice_snapshot=blocked_choice_snapshot,
            )
        )
        blocked_step = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceOutcomeKind.BLOCKED,
            source_snapshot=snapshot,
            emitted_text="blocked",
            choice_snapshot=blocked_choice_snapshot,
        )
        blocked_sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.BLOCKED,
            source_snapshot=snapshot,
            emitted_texts=("blocked",),
            step_outcomes=(blocked_step,),
            current_snapshot=snapshot,
        )
        replay_blocked = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .REPLAY_BLOCKED
            ),
            source_snapshot=snapshot,
            emitted_texts=("blocked",),
            sequence_outcome=blocked_sequence,
        )
        invalid_step = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotAdvanceOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_text="bad",
            choice_snapshot=choice_snapshot,
        )
        invalid_sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotAdvanceSequenceOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_texts=("bad",),
            step_outcomes=(invalid_step,),
            current_snapshot=snapshot,
        )
        invalid_replay = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_texts=("bad",),
            sequence_outcome=invalid_sequence,
        )

        readable = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=writer_snapshot._WriterSnapshotPrefixReadOutcomeKind.READABLE,
            replay_outcome=replay,
            support_count=1,
            completion_count=2,
        )
        replay_blocked_outcome = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotPrefixReadOutcomeKind
                .REPLAY_BLOCKED
            ),
            replay_outcome=replay_blocked,
        )
        invalid_outcome = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotPrefixReadOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            replay_outcome=invalid_replay,
        )
        final_blocked = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotPrefixReadOutcomeKind
                .FINAL_FRONTIER_BLOCKED
            ),
            replay_outcome=final_blocked_replay,
        )

        self.assertEqual(readable.public_choices, choice_snapshot.public_choices)
        self.assertTrue(replay_blocked_outcome.blocked)
        self.assertTrue(invalid_outcome.invalid_emitted_text)
        self.assertTrue(final_blocked.blocked)

        invalid_payloads = (
            dict(
                kind=writer_snapshot._WriterSnapshotPrefixReadOutcomeKind.READABLE,
                replay_outcome=replay_blocked,
            ),
            dict(
                kind=(
                    writer_snapshot
                    ._WriterSnapshotPrefixReadOutcomeKind
                    .REPLAY_BLOCKED
                ),
                replay_outcome=replay_blocked,
                support_count=1,
            ),
            dict(
                kind=(
                    writer_snapshot
                    ._WriterSnapshotPrefixReadOutcomeKind
                    .FINAL_FRONTIER_BLOCKED
                ),
                replay_outcome=final_blocked_replay,
                completion_count=1,
            ),
            dict(
                kind=(
                    writer_snapshot
                    ._WriterSnapshotPrefixReadOutcomeKind
                    .INVALID_EMITTED_TEXT
                ),
                replay_outcome=replay,
            ),
        )

        for kwargs in invalid_payloads:
            with self.subTest(kind=kwargs["kind"]):
                with self.assertRaises(SouthStarError) as raised:
                    writer_snapshot._WriterSnapshotPrefixReadOutcome(**kwargs)

                self.assertIs(
                    raised.exception.kind,
                    SouthStarErrorKind.INTERNAL_INVARIANT,
                )

    def test_writer_snapshot_prefix_read_outcome_exposes_graph_policy_evidence(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        policy, active_outcome = self._dead_closure_open_active_emitted_outcome(
            AtomId(0),
        )
        choice_snapshot = self._test_frontier_choice_snapshot_with_active_policy(
            active_outcome,
        )
        sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.ADVANCED,
            source_snapshot=snapshot,
            emitted_texts=(),
            step_outcomes=(),
            current_snapshot=snapshot,
        )
        replay = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .CHOICE_SNAPSHOT
            ),
            source_snapshot=snapshot,
            emitted_texts=(),
            sequence_outcome=sequence,
            choice_snapshot=choice_snapshot,
        )
        prefix_outcome = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=writer_snapshot._WriterSnapshotPrefixReadOutcomeKind.READABLE,
            replay_outcome=replay,
        )

        self.assertEqual(
            prefix_outcome.graph_policy_decisions,
            choice_snapshot.graph_policy_decisions,
        )
        self.assertEqual(prefix_outcome.graph_policy_decisions, (policy,))
        self.assertEqual(
            prefix_outcome.resolved_residual_attachment_policy_groups,
            choice_snapshot.resolved_residual_attachment_policy_groups,
        )
        self.assertTrue(
            prefix_outcome.resolved_residual_attachment_policy_groups
        )
        self.assertEqual(
            (
                prefix_outcome
                .support_dead_closure_open_vs_cyclic_tree_entry_groups
            ),
            (
                choice_snapshot
                .support_dead_closure_open_vs_cyclic_tree_entry_groups
            ),
        )
        self.assertTrue(
            (
                prefix_outcome
                .support_dead_closure_open_vs_cyclic_tree_entry_groups
            )
        )
        self.assertEqual(
            (
                prefix_outcome
                .unsupported_owner_scope_residual_attachment_policy_groups
            ),
            (),
        )
        self.assertEqual(
            prefix_outcome.unresolved_residual_attachment_policy_groups,
            (),
        )

        blocked_choice_snapshot = self._test_blocked_frontier_choice_snapshot(cursor)
        blocked_step = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceOutcomeKind.BLOCKED,
            source_snapshot=snapshot,
            emitted_text="blocked",
            choice_snapshot=blocked_choice_snapshot,
        )
        blocked_sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.BLOCKED,
            source_snapshot=snapshot,
            emitted_texts=("blocked",),
            step_outcomes=(blocked_step,),
            current_snapshot=snapshot,
        )
        replay_blocked = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .REPLAY_BLOCKED
            ),
            source_snapshot=snapshot,
            emitted_texts=("blocked",),
            sequence_outcome=blocked_sequence,
        )
        replay_blocked_outcome = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotPrefixReadOutcomeKind
                .REPLAY_BLOCKED
            ),
            replay_outcome=replay_blocked,
        )
        invalid_step = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotAdvanceOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_text="bad",
            choice_snapshot=choice_snapshot,
        )
        invalid_sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotAdvanceSequenceOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_texts=("bad",),
            step_outcomes=(invalid_step,),
            current_snapshot=snapshot,
        )
        invalid_replay = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_texts=("bad",),
            sequence_outcome=invalid_sequence,
        )
        invalid_outcome = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotPrefixReadOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            replay_outcome=invalid_replay,
        )

        for failed in (replay_blocked_outcome, invalid_outcome):
            with self.subTest(kind=failed.kind):
                self.assertEqual(failed.graph_policy_decisions, ())
                self.assertEqual(
                    failed.resolved_residual_attachment_policy_groups,
                    (),
                )
                self.assertEqual(
                    (
                        failed
                        .support_dead_closure_open_vs_cyclic_tree_entry_groups
                    ),
                    (),
                )

    def test_writer_snapshot_prefix_read_outcome_exposes_closure_endpoint_selection_evidence(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        closure_decision, _, _ = self._test_closure_endpoint_decision(
            pair_survives=False,
            open_survives=True,
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .CLOSURE_ENDPOINT
            ),
            active_atom=AtomId(0),
            closure_endpoint_decision=closure_decision,
        )
        active_decision = writer_transitions._active_emitted_closure_decision(
            closure_decision,
            graph_policy_decision=policy,
        )
        active_outcome = writer_transitions._WriterActiveEmittedScheduleOutcome(
            kind=(
                writer_transitions
                ._WriterActiveEmittedScheduleOutcomeKind
                .SCHEDULED
            ),
            schedule_decision=active_decision,
            graph_policy_decision=policy,
        )
        choice_snapshot = self._test_frontier_choice_snapshot_with_active_policy(
            active_outcome,
        )
        sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.ADVANCED,
            source_snapshot=snapshot,
            emitted_texts=(),
            step_outcomes=(),
            current_snapshot=snapshot,
        )
        replay = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .CHOICE_SNAPSHOT
            ),
            source_snapshot=snapshot,
            emitted_texts=(),
            sequence_outcome=sequence,
            choice_snapshot=choice_snapshot,
        )
        prefix_outcome = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=writer_snapshot._WriterSnapshotPrefixReadOutcomeKind.READABLE,
            replay_outcome=replay,
        )

        self.assertEqual(
            prefix_outcome.considered_closure_endpoint_selection_kinds,
            choice_snapshot.considered_closure_endpoint_selection_kinds,
        )
        self.assertEqual(
            prefix_outcome.selected_closure_endpoint_selection_kinds,
            choice_snapshot.selected_closure_endpoint_selection_kinds,
        )
        self.assertEqual(
            prefix_outcome.selected_closure_open_graph_action_surfaces,
            choice_snapshot.selected_closure_open_graph_action_surfaces,
        )
        self.assertEqual(
            prefix_outcome.selected_closure_pair_graph_action_surfaces,
            choice_snapshot.selected_closure_pair_graph_action_surfaces,
        )

        blocked_choice_snapshot = self._test_blocked_frontier_choice_snapshot(cursor)
        blocked_step = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceOutcomeKind.BLOCKED,
            source_snapshot=snapshot,
            emitted_text="blocked",
            choice_snapshot=blocked_choice_snapshot,
        )
        blocked_sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.BLOCKED,
            source_snapshot=snapshot,
            emitted_texts=("blocked",),
            step_outcomes=(blocked_step,),
            current_snapshot=snapshot,
        )
        replay_blocked = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .REPLAY_BLOCKED
            ),
            source_snapshot=snapshot,
            emitted_texts=("blocked",),
            sequence_outcome=blocked_sequence,
        )
        replay_blocked_outcome = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotPrefixReadOutcomeKind
                .REPLAY_BLOCKED
            ),
            replay_outcome=replay_blocked,
        )
        invalid_step = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotAdvanceOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_text="bad",
            choice_snapshot=choice_snapshot,
        )
        invalid_sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotAdvanceSequenceOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_texts=("bad",),
            step_outcomes=(invalid_step,),
            current_snapshot=snapshot,
        )
        invalid_replay = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_texts=("bad",),
            sequence_outcome=invalid_sequence,
        )
        invalid_outcome = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotPrefixReadOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            replay_outcome=invalid_replay,
        )

        for failed in (replay_blocked_outcome, invalid_outcome):
            with self.subTest(kind=failed.kind):
                self.assertEqual(
                    failed.considered_closure_endpoint_selection_kinds,
                    (),
                )
                self.assertEqual(
                    failed.selected_closure_endpoint_selection_kinds,
                    (),
                )
                self.assertEqual(
                    failed.selected_closure_open_graph_action_surfaces,
                    (),
                )
                self.assertEqual(
                    failed.selected_closure_pair_graph_action_surfaces,
                    (),
                )

    def test_writer_snapshot_prefix_read_outcome_exposes_active_child_selection_evidence(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        choice_snapshot = self._test_frontier_choice_snapshot_with_active_policy(
            self._cyclic_active_child_scheduled_outcome(AtomId(0)),
        )
        sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.ADVANCED,
            source_snapshot=snapshot,
            emitted_texts=(),
            step_outcomes=(),
            current_snapshot=snapshot,
        )
        replay = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .CHOICE_SNAPSHOT
            ),
            source_snapshot=snapshot,
            emitted_texts=(),
            sequence_outcome=sequence,
            choice_snapshot=choice_snapshot,
        )
        prefix_outcome = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=writer_snapshot._WriterSnapshotPrefixReadOutcomeKind.READABLE,
            replay_outcome=replay,
        )

        self.assertEqual(
            prefix_outcome.considered_active_child_selection_kinds,
            choice_snapshot.considered_active_child_selection_kinds,
        )
        self.assertEqual(
            prefix_outcome.selected_active_child_selection_kinds,
            choice_snapshot.selected_active_child_selection_kinds,
        )
        self.assertEqual(
            prefix_outcome.considered_cyclic_tree_entry_graph_action_surfaces,
            (
                choice_snapshot
                .considered_cyclic_tree_entry_graph_action_surfaces
            ),
        )
        self.assertEqual(
            prefix_outcome.selected_cyclic_tree_entry_graph_action_surfaces,
            choice_snapshot.selected_cyclic_tree_entry_graph_action_surfaces,
        )

        blocked_choice_snapshot = self._test_blocked_frontier_choice_snapshot(cursor)
        blocked_step = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceOutcomeKind.BLOCKED,
            source_snapshot=snapshot,
            emitted_text="blocked",
            choice_snapshot=blocked_choice_snapshot,
        )
        blocked_sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.BLOCKED,
            source_snapshot=snapshot,
            emitted_texts=("blocked",),
            step_outcomes=(blocked_step,),
            current_snapshot=snapshot,
        )
        replay_blocked = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .REPLAY_BLOCKED
            ),
            source_snapshot=snapshot,
            emitted_texts=("blocked",),
            sequence_outcome=blocked_sequence,
        )
        replay_blocked_outcome = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotPrefixReadOutcomeKind
                .REPLAY_BLOCKED
            ),
            replay_outcome=replay_blocked,
        )
        invalid_step = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotAdvanceOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_text="bad",
            choice_snapshot=choice_snapshot,
        )
        invalid_sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotAdvanceSequenceOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_texts=("bad",),
            step_outcomes=(invalid_step,),
            current_snapshot=snapshot,
        )
        invalid_replay = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_texts=("bad",),
            sequence_outcome=invalid_sequence,
        )
        invalid_outcome = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotPrefixReadOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            replay_outcome=invalid_replay,
        )

        for failed in (replay_blocked_outcome, invalid_outcome):
            with self.subTest(kind=failed.kind):
                self.assertEqual(
                    failed.considered_active_child_selection_kinds,
                    (),
                )
                self.assertEqual(
                    failed.selected_active_child_selection_kinds,
                    (),
                )
                self.assertEqual(
                    failed.considered_cyclic_tree_entry_graph_action_surfaces,
                    (),
                )
                self.assertEqual(
                    failed.selected_cyclic_tree_entry_graph_action_surfaces,
                    (),
                )

    def test_writer_snapshot_prefix_read_outcome_exposes_residual_cyclic_policy_evidence(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        closure_decision, child_surface = self._test_residual_cyclic_policy_inputs()
        residual_decision = writer_transitions._residual_cyclic_policy_decision(
            closure_decision,
            child_surface,
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD_AFTER_DEAD_CLOSURE_OPEN
            ),
            active_atom=AtomId(0),
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
            residual_cyclic_policy_decision=residual_decision,
        )
        schedule_outcome = (
            self._test_frontier_schedule_outcome_for_graph_policies(
                (policy,),
            )
        )
        choice_snapshot = writer_frontier_module._WriterFrontierChoiceSnapshot(
            schedule_outcome=schedule_outcome,
            terminal=None,
            choices=(),
        )
        sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.ADVANCED,
            source_snapshot=snapshot,
            emitted_texts=(),
            step_outcomes=(),
            current_snapshot=snapshot,
        )
        replay = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .CHOICE_SNAPSHOT
            ),
            source_snapshot=snapshot,
            emitted_texts=(),
            sequence_outcome=sequence,
            choice_snapshot=choice_snapshot,
        )
        prefix_outcome = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=writer_snapshot._WriterSnapshotPrefixReadOutcomeKind.READABLE,
            replay_outcome=replay,
        )

        self.assertEqual(
            prefix_outcome.residual_cyclic_policy_decisions,
            choice_snapshot.residual_cyclic_policy_decisions,
        )
        self.assertEqual(
            prefix_outcome.residual_cyclic_policy_kinds,
            choice_snapshot.residual_cyclic_policy_kinds,
        )
        self.assertEqual(
            prefix_outcome.residual_cyclic_choice_groups,
            choice_snapshot.residual_cyclic_choice_groups,
        )
        self.assertEqual(
            prefix_outcome.residual_cyclic_support_dead_groups,
            choice_snapshot.residual_cyclic_support_dead_groups,
        )

        blocked_choice_snapshot = self._test_blocked_frontier_choice_snapshot(cursor)
        blocked_step = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceOutcomeKind.BLOCKED,
            source_snapshot=snapshot,
            emitted_text="blocked",
            choice_snapshot=blocked_choice_snapshot,
        )
        blocked_sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.BLOCKED,
            source_snapshot=snapshot,
            emitted_texts=("blocked",),
            step_outcomes=(blocked_step,),
            current_snapshot=snapshot,
        )
        replay_blocked = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .REPLAY_BLOCKED
            ),
            source_snapshot=snapshot,
            emitted_texts=("blocked",),
            sequence_outcome=blocked_sequence,
        )
        replay_blocked_outcome = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotPrefixReadOutcomeKind
                .REPLAY_BLOCKED
            ),
            replay_outcome=replay_blocked,
        )
        invalid_step = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotAdvanceOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_text="bad",
            choice_snapshot=choice_snapshot,
        )
        invalid_sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotAdvanceSequenceOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_texts=("bad",),
            step_outcomes=(invalid_step,),
            current_snapshot=snapshot,
        )
        invalid_replay = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_texts=("bad",),
            sequence_outcome=invalid_sequence,
        )
        invalid_outcome = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotPrefixReadOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            replay_outcome=invalid_replay,
        )

        for failed in (replay_blocked_outcome, invalid_outcome):
            with self.subTest(kind=failed.kind):
                self.assertEqual(
                    failed.residual_cyclic_policy_decisions,
                    (),
                )
                self.assertEqual(failed.residual_cyclic_policy_kinds, ())
                self.assertEqual(failed.residual_cyclic_choice_groups, ())
                self.assertEqual(
                    failed.residual_cyclic_support_dead_groups,
                    (),
                )

    def test_writer_snapshot_prefix_read_outcome_exposes_final_choice_residual_cyclic_policy_kinds(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        key = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=7,
        )
        support = self._test_frontier_next_token_support(
            emitted_text="C",
            successor_atom=AtomId(1),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CYCLIC_TREE_ENTRY
            ),
            residual_key=key,
        )
        choice = self._test_choice_snapshot_entry_from_supports(
            "C",
            (support,),
        )
        closure_decision, child_surface = (
            self._test_residual_cyclic_policy_inputs(
                attachment_id=7,
            )
        )
        residual_decision = writer_transitions._residual_cyclic_policy_decision(
            closure_decision,
            child_surface,
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD_AFTER_DEAD_CLOSURE_OPEN
            ),
            active_atom=AtomId(0),
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
            residual_cyclic_policy_decision=residual_decision,
        )
        choice_snapshot = writer_frontier_module._WriterFrontierChoiceSnapshot(
            schedule_outcome=(
                self._test_frontier_schedule_outcome_for_graph_policies(
                    (policy,),
                )
            ),
            terminal=None,
            choices=(choice,),
        )
        sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.ADVANCED,
            source_snapshot=snapshot,
            emitted_texts=(),
            step_outcomes=(),
            current_snapshot=snapshot,
        )
        replay = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .CHOICE_SNAPSHOT
            ),
            source_snapshot=snapshot,
            emitted_texts=(),
            sequence_outcome=sequence,
            choice_snapshot=choice_snapshot,
        )
        prefix_outcome = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=writer_snapshot._WriterSnapshotPrefixReadOutcomeKind.READABLE,
            replay_outcome=replay,
        )

        self.assertEqual(
            prefix_outcome.final_choice_residual_cyclic_policy_kinds,
            (residual_decision.kind,),
        )

        blocked_choice_snapshot = self._test_blocked_frontier_choice_snapshot(cursor)
        blocked_step = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceOutcomeKind.BLOCKED,
            source_snapshot=snapshot,
            emitted_text="blocked",
            choice_snapshot=blocked_choice_snapshot,
        )
        blocked_sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.BLOCKED,
            source_snapshot=snapshot,
            emitted_texts=("blocked",),
            step_outcomes=(blocked_step,),
            current_snapshot=snapshot,
        )
        replay_blocked = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .REPLAY_BLOCKED
            ),
            source_snapshot=snapshot,
            emitted_texts=("blocked",),
            sequence_outcome=blocked_sequence,
        )
        replay_blocked_outcome = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotPrefixReadOutcomeKind
                .REPLAY_BLOCKED
            ),
            replay_outcome=replay_blocked,
        )

        self.assertEqual(
            replay_blocked_outcome.final_choice_residual_cyclic_policy_kinds,
            (),
        )

    def test_writer_snapshot_prefix_read_outcome_exposes_replayed_residual_cyclic_policy_kinds(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        key = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=7,
        )
        support = self._test_frontier_next_token_support(
            emitted_text="C",
            successor_atom=AtomId(1),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CYCLIC_TREE_ENTRY
            ),
            residual_key=key,
        )
        choice = self._test_choice_snapshot_entry_from_supports(
            "C",
            (support,),
        )
        closure_decision, child_surface = (
            self._test_residual_cyclic_policy_inputs(
                attachment_id=7,
            )
        )
        residual_decision = writer_transitions._residual_cyclic_policy_decision(
            closure_decision,
            child_surface,
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD_AFTER_DEAD_CLOSURE_OPEN
            ),
            active_atom=AtomId(0),
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
            residual_cyclic_policy_decision=residual_decision,
        )
        choice_snapshot = writer_frontier_module._WriterFrontierChoiceSnapshot(
            schedule_outcome=(
                self._test_frontier_schedule_outcome_for_graph_policies(
                    (policy,),
                )
            ),
            terminal=None,
            choices=(choice,),
        )
        advance = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceOutcomeKind.ADVANCED,
            source_snapshot=snapshot,
            emitted_text="C",
            choice_snapshot=choice_snapshot,
            choice=choice,
            advanced_snapshot=snapshot,
        )
        sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.ADVANCED,
            source_snapshot=snapshot,
            emitted_texts=("C",),
            step_outcomes=(advance,),
            current_snapshot=snapshot,
        )
        replay = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .CHOICE_SNAPSHOT
            ),
            source_snapshot=snapshot,
            emitted_texts=("C",),
            sequence_outcome=sequence,
            choice_snapshot=choice_snapshot,
        )
        prefix_outcome = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=writer_snapshot._WriterSnapshotPrefixReadOutcomeKind.READABLE,
            replay_outcome=replay,
        )
        empty_sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.ADVANCED,
            source_snapshot=snapshot,
            emitted_texts=(),
            step_outcomes=(),
            current_snapshot=snapshot,
        )
        empty_replay = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .CHOICE_SNAPSHOT
            ),
            source_snapshot=snapshot,
            emitted_texts=(),
            sequence_outcome=empty_sequence,
            choice_snapshot=choice_snapshot,
        )
        empty_prefix = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=writer_snapshot._WriterSnapshotPrefixReadOutcomeKind.READABLE,
            replay_outcome=empty_replay,
        )

        self.assertEqual(
            prefix_outcome.replayed_residual_cyclic_policy_kinds,
            (residual_decision.kind,),
        )
        self.assertEqual(
            empty_prefix.replayed_residual_cyclic_policy_kinds,
            (),
        )

    def test_writer_snapshot_prefix_read_outcome_exposes_dead_closure_residual_cyclic_policy_kind(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        closure_decision, child_surface = self._test_residual_cyclic_policy_inputs()
        residual_decision = writer_transitions._residual_cyclic_policy_decision(
            closure_decision,
            child_surface,
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD_AFTER_DEAD_CLOSURE_OPEN
            ),
            active_atom=AtomId(0),
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
            residual_cyclic_policy_decision=residual_decision,
        )
        choice_snapshot = writer_frontier_module._WriterFrontierChoiceSnapshot(
            schedule_outcome=(
                self._test_frontier_schedule_outcome_for_graph_policies(
                    (policy,),
                )
            ),
            terminal=None,
            choices=(),
        )
        sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.ADVANCED,
            source_snapshot=snapshot,
            emitted_texts=(),
            step_outcomes=(),
            current_snapshot=snapshot,
        )
        replay = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .CHOICE_SNAPSHOT
            ),
            source_snapshot=snapshot,
            emitted_texts=(),
            sequence_outcome=sequence,
            choice_snapshot=choice_snapshot,
        )
        prefix_outcome = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=writer_snapshot._WriterSnapshotPrefixReadOutcomeKind.READABLE,
            replay_outcome=replay,
        )

        self.assertIn(
            (
                writer_transitions
                ._WriterResidualCyclicPolicyDecisionKind
                .ACTIVE_CHILD_AFTER_DEAD_CLOSURE_OPEN
            ),
            prefix_outcome.residual_cyclic_policy_kinds,
        )
        self.assertTrue(prefix_outcome.residual_cyclic_support_dead_groups)

    def test_writer_snapshot_prefix_read_outcome_exposes_residual_attachment_support_groups(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        key = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=7,
        )
        support = self._test_frontier_next_token_support(
            emitted_text="C",
            successor_atom=AtomId(1),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CYCLIC_TREE_ENTRY
            ),
            residual_key=key,
        )
        schedule_outcome = writer_frontier_module._WriterFrontierScheduleOutcome(
            state_outcomes=(),
            terminal_by_key=Counter(),
            grouped_by_text={},
            weighted_by_text={},
            next_token_frontier=(
                writer_frontier_module._WriterFrontierNextTokenEntry(
                    emitted_text="C",
                    supports=(support,),
                ),
            ),
        )
        choice_snapshot = writer_frontier_module._WriterFrontierChoiceSnapshot(
            schedule_outcome=schedule_outcome,
            terminal=None,
            choices=(),
        )
        sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.ADVANCED,
            source_snapshot=snapshot,
            emitted_texts=(),
            step_outcomes=(),
            current_snapshot=snapshot,
        )
        replay = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .CHOICE_SNAPSHOT
            ),
            source_snapshot=snapshot,
            emitted_texts=(),
            sequence_outcome=sequence,
            choice_snapshot=choice_snapshot,
        )
        prefix_outcome = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=writer_snapshot._WriterSnapshotPrefixReadOutcomeKind.READABLE,
            replay_outcome=replay,
        )

        self.assertEqual(
            prefix_outcome.residual_attachment_support_groups,
            choice_snapshot.residual_attachment_support_groups,
        )
        self.assertEqual(
            prefix_outcome.residual_attachment_support_groups[0].supports,
            (support,),
        )

        blocked_choice_snapshot = self._test_blocked_frontier_choice_snapshot(cursor)
        blocked_step = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceOutcomeKind.BLOCKED,
            source_snapshot=snapshot,
            emitted_text="blocked",
            choice_snapshot=blocked_choice_snapshot,
        )
        blocked_sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.BLOCKED,
            source_snapshot=snapshot,
            emitted_texts=("blocked",),
            step_outcomes=(blocked_step,),
            current_snapshot=snapshot,
        )
        replay_blocked = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .REPLAY_BLOCKED
            ),
            source_snapshot=snapshot,
            emitted_texts=("blocked",),
            sequence_outcome=blocked_sequence,
        )
        replay_blocked_outcome = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotPrefixReadOutcomeKind
                .REPLAY_BLOCKED
            ),
            replay_outcome=replay_blocked,
        )
        invalid_step = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotAdvanceOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_text="bad",
            choice_snapshot=choice_snapshot,
        )
        invalid_sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotAdvanceSequenceOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_texts=("bad",),
            step_outcomes=(invalid_step,),
            current_snapshot=snapshot,
        )
        invalid_replay = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_texts=("bad",),
            sequence_outcome=invalid_sequence,
        )
        invalid_outcome = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotPrefixReadOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            replay_outcome=invalid_replay,
        )

        self.assertEqual(
            replay_blocked_outcome.residual_attachment_support_groups,
            (),
        )
        self.assertEqual(invalid_outcome.residual_attachment_support_groups, ())

    def test_writer_snapshot_prefix_read_outcome_exposes_residual_attachment_evidence_groups(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        key = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=7,
        )
        support = self._test_frontier_next_token_support(
            emitted_text="C",
            successor_atom=AtomId(1),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CYCLIC_TREE_ENTRY
            ),
            residual_key=key,
        )
        schedule_outcome = writer_frontier_module._WriterFrontierScheduleOutcome(
            state_outcomes=(),
            terminal_by_key=Counter(),
            grouped_by_text={},
            weighted_by_text={},
            next_token_frontier=(
                writer_frontier_module._WriterFrontierNextTokenEntry(
                    emitted_text="C",
                    supports=(support,),
                ),
            ),
        )
        choice_snapshot = writer_frontier_module._WriterFrontierChoiceSnapshot(
            schedule_outcome=schedule_outcome,
            terminal=None,
            choices=(),
        )
        sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.ADVANCED,
            source_snapshot=snapshot,
            emitted_texts=(),
            step_outcomes=(),
            current_snapshot=snapshot,
        )
        replay = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .CHOICE_SNAPSHOT
            ),
            source_snapshot=snapshot,
            emitted_texts=(),
            sequence_outcome=sequence,
            choice_snapshot=choice_snapshot,
        )
        prefix_outcome = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=writer_snapshot._WriterSnapshotPrefixReadOutcomeKind.READABLE,
            replay_outcome=replay,
        )

        self.assertEqual(
            prefix_outcome.residual_attachment_evidence_groups,
            choice_snapshot.residual_attachment_evidence_groups,
        )
        self.assertEqual(
            prefix_outcome.residual_attachment_evidence_groups[0].selected_supports,
            (support,),
        )

        blocked_choice_snapshot = self._test_blocked_frontier_choice_snapshot(cursor)
        blocked_step = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceOutcomeKind.BLOCKED,
            source_snapshot=snapshot,
            emitted_text="blocked",
            choice_snapshot=blocked_choice_snapshot,
        )
        blocked_sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.BLOCKED,
            source_snapshot=snapshot,
            emitted_texts=("blocked",),
            step_outcomes=(blocked_step,),
            current_snapshot=snapshot,
        )
        replay_blocked = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .REPLAY_BLOCKED
            ),
            source_snapshot=snapshot,
            emitted_texts=("blocked",),
            sequence_outcome=blocked_sequence,
        )
        replay_blocked_outcome = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotPrefixReadOutcomeKind
                .REPLAY_BLOCKED
            ),
            replay_outcome=replay_blocked,
        )
        invalid_step = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotAdvanceOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_text="bad",
            choice_snapshot=choice_snapshot,
        )
        invalid_sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotAdvanceSequenceOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_texts=("bad",),
            step_outcomes=(invalid_step,),
            current_snapshot=snapshot,
        )
        invalid_replay = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_texts=("bad",),
            sequence_outcome=invalid_sequence,
        )
        invalid_outcome = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotPrefixReadOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            replay_outcome=invalid_replay,
        )

        self.assertEqual(
            replay_blocked_outcome.residual_attachment_evidence_groups,
            (),
        )
        self.assertEqual(invalid_outcome.residual_attachment_evidence_groups, ())

    def test_writer_snapshot_prefix_read_outcome_exposes_per_choice_residual_attachment_evidence(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        key = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=7,
        )
        support = self._test_frontier_next_token_support(
            emitted_text="C",
            successor_atom=AtomId(1),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CYCLIC_TREE_ENTRY
            ),
            residual_key=key,
        )
        choice = self._test_choice_snapshot_entry_from_supports("C", (support,))
        choice_snapshot = writer_frontier_module._WriterFrontierChoiceSnapshot(
            schedule_outcome=writer_frontier_module._WriterFrontierScheduleOutcome(
                state_outcomes=(),
                terminal_by_key=Counter(),
                grouped_by_text={},
                weighted_by_text={},
                next_token_frontier=(choice.next_token_entry,),
            ),
            terminal=None,
            choices=(choice,),
        )
        sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.ADVANCED,
            source_snapshot=snapshot,
            emitted_texts=(),
            step_outcomes=(),
            current_snapshot=snapshot,
        )
        replay = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .CHOICE_SNAPSHOT
            ),
            source_snapshot=snapshot,
            emitted_texts=(),
            sequence_outcome=sequence,
            choice_snapshot=choice_snapshot,
        )
        prefix_outcome = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=writer_snapshot._WriterSnapshotPrefixReadOutcomeKind.READABLE,
            replay_outcome=replay,
        )

        self.assertEqual(
            prefix_outcome.choice_residual_attachment_evidence,
            choice_snapshot.choice_residual_attachment_evidence,
        )
        self.assertEqual(
            (
                prefix_outcome
                .choice_residual_attachment_evidence_for_emitted_text("C")
            ),
            (
                choice_snapshot
                .choice_residual_attachment_evidence_for_emitted_text("C")
            ),
        )

        blocked_choice_snapshot = self._test_blocked_frontier_choice_snapshot(cursor)
        blocked_step = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceOutcomeKind.BLOCKED,
            source_snapshot=snapshot,
            emitted_text="blocked",
            choice_snapshot=blocked_choice_snapshot,
        )
        blocked_sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.BLOCKED,
            source_snapshot=snapshot,
            emitted_texts=("blocked",),
            step_outcomes=(blocked_step,),
            current_snapshot=snapshot,
        )
        replay_blocked = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .REPLAY_BLOCKED
            ),
            source_snapshot=snapshot,
            emitted_texts=("blocked",),
            sequence_outcome=blocked_sequence,
        )
        replay_blocked_outcome = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotPrefixReadOutcomeKind
                .REPLAY_BLOCKED
            ),
            replay_outcome=replay_blocked,
        )
        invalid_step = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotAdvanceOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_text="bad",
            choice_snapshot=choice_snapshot,
        )
        invalid_sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotAdvanceSequenceOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_texts=("bad",),
            step_outcomes=(invalid_step,),
            current_snapshot=snapshot,
        )
        invalid_replay = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_texts=("bad",),
            sequence_outcome=invalid_sequence,
        )
        invalid_outcome = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotPrefixReadOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            replay_outcome=invalid_replay,
        )

        self.assertEqual(
            replay_blocked_outcome.choice_residual_attachment_evidence,
            (),
        )
        self.assertIsNone(
            replay_blocked_outcome
            .choice_residual_attachment_evidence_for_emitted_text("C")
        )
        self.assertEqual(
            invalid_outcome.choice_residual_attachment_evidence,
            (),
        )
        self.assertIsNone(
            invalid_outcome
            .choice_residual_attachment_evidence_for_emitted_text("C")
        )

    def test_writer_snapshot_prefix_read_outcome_exposes_owner_scope_evidence(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        key = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=7,
        )
        branch_return_group = (
            self._test_residual_policy_group_with_owner_scope(
                key,
                closure_owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
                child_owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
            )
        )
        support = self._test_frontier_next_token_support(
            emitted_text="C",
            successor_atom=AtomId(1),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CYCLIC_TREE_ENTRY
            ),
            residual_key=key,
        )
        choice = self._test_choice_snapshot_entry_from_supports("C", (support,))
        residual_evidence_group = (
            writer_frontier_module
            ._WriterFrontierResidualAttachmentEvidenceGroup(
                key=key,
                unsupported_owner_scope_policy_groups=(
                    branch_return_group,
                ),
            )
        )
        choice_evidence = (
            writer_frontier_module
            ._WriterFrontierChoiceResidualAttachmentEvidence(
                choice=choice,
                residual_attachment_evidence_groups=(
                    residual_evidence_group,
                ),
            )
        )
        scope = (
            writer_transitions
            ._WriterResidualAttachmentOwnerScopeKind
            .BRANCH_RETURN
        )
        readable_choice_snapshot = SimpleNamespace(
            blocked=False,
            graph_policy_blockers=(),
            unsupported_owner_scope_choice_evidence=(choice_evidence,),
            unsupported_owner_scope_kinds=(scope,),
        )
        sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.ADVANCED,
            source_snapshot=snapshot,
            emitted_texts=(),
            step_outcomes=(),
            current_snapshot=snapshot,
        )
        replay = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .CHOICE_SNAPSHOT
            ),
            source_snapshot=snapshot,
            emitted_texts=(),
            sequence_outcome=sequence,
            choice_snapshot=readable_choice_snapshot,  # type: ignore[arg-type]
        )
        prefix_outcome = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=writer_snapshot._WriterSnapshotPrefixReadOutcomeKind.READABLE,
            replay_outcome=replay,
        )

        self.assertEqual(
            prefix_outcome.final_choice_unsupported_owner_scope_evidence,
            (choice_evidence,),
        )
        self.assertEqual(
            prefix_outcome.final_choice_unsupported_owner_scope_kinds,
            (scope,),
        )
        self.assertEqual(prefix_outcome.blocker_owner_scope_kinds, ())

        blocker = writer_transitions._WriterActiveEmittedGraphPolicyBlocker(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyBlockerKind
                .UNSUPPORTED_OWNER_SCOPE_RESIDUAL_ATTACHMENT_CHOICE
            ),
            residual_group=branch_return_group,
        )
        blocked_choice_snapshot = SimpleNamespace(
            blocked=True,
            graph_policy_blockers=(blocker,),
        )
        blocked_step = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceOutcomeKind.BLOCKED,
            source_snapshot=snapshot,
            emitted_text="blocked",
            choice_snapshot=blocked_choice_snapshot,  # type: ignore[arg-type]
        )
        blocked_sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.BLOCKED,
            source_snapshot=snapshot,
            emitted_texts=("blocked",),
            step_outcomes=(blocked_step,),
            current_snapshot=snapshot,
        )
        blocked_replay = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .REPLAY_BLOCKED
            ),
            source_snapshot=snapshot,
            emitted_texts=("blocked",),
            sequence_outcome=blocked_sequence,
        )
        blocked_prefix = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotPrefixReadOutcomeKind
                .REPLAY_BLOCKED
            ),
            replay_outcome=blocked_replay,
        )

        self.assertEqual(
            blocked_prefix.final_choice_unsupported_owner_scope_evidence,
            (),
        )
        self.assertEqual(
            blocked_prefix.blocker_owner_scope_kinds,
            (scope,),
        )

    def test_writer_snapshot_advance_outcome_exposes_choice_residual_attachment_evidence(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        key = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=7,
        )
        support = self._test_frontier_next_token_support(
            emitted_text="C",
            successor_atom=AtomId(1),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CYCLIC_TREE_ENTRY
            ),
            residual_key=key,
        )
        choice = self._test_choice_snapshot_entry_from_supports("C", (support,))
        choice_snapshot = writer_frontier_module._WriterFrontierChoiceSnapshot(
            schedule_outcome=writer_frontier_module._WriterFrontierScheduleOutcome(
                state_outcomes=(),
                terminal_by_key=Counter(),
                grouped_by_text={},
                weighted_by_text={},
                next_token_frontier=(choice.next_token_entry,),
            ),
            terminal=None,
            choices=(choice,),
        )
        outcome = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceOutcomeKind.ADVANCED,
            source_snapshot=snapshot,
            emitted_text="C",
            choice_snapshot=choice_snapshot,
            choice=choice,
            advanced_snapshot=snapshot,
        )

        self.assertEqual(
            outcome.choice_residual_attachment_evidence,
            (
                outcome.choice_snapshot
                .choice_residual_attachment_evidence_for_emitted_text(
                    outcome.emitted_text
                )
            ),
        )
        self.assertIs(outcome.choice_residual_attachment_evidence.choice, choice)

        blocked_snapshot = self._test_blocked_frontier_choice_snapshot(cursor)
        blocked = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceOutcomeKind.BLOCKED,
            source_snapshot=snapshot,
            emitted_text="blocked",
            choice_snapshot=blocked_snapshot,
        )
        invalid = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotAdvanceOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_text="bad",
            choice_snapshot=choice_snapshot,
        )

        self.assertIsNone(blocked.choice_residual_attachment_evidence)
        self.assertIsNone(invalid.choice_residual_attachment_evidence)

    def test_writer_snapshot_advance_outcome_rejects_missing_choice_residual_evidence(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        support = self._test_frontier_next_token_support(
            emitted_text="C",
            successor_atom=AtomId(1),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CYCLIC_TREE_ENTRY
            ),
        )
        choice = self._test_choice_snapshot_entry_from_supports("C", (support,))
        choice_snapshot = writer_frontier_module._WriterFrontierChoiceSnapshot(
            schedule_outcome=writer_frontier_module._WriterFrontierScheduleOutcome(
                state_outcomes=(),
                terminal_by_key=Counter(),
                grouped_by_text={},
                weighted_by_text={},
            ),
            terminal=None,
            choices=(),
        )
        outcome = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceOutcomeKind.ADVANCED,
            source_snapshot=snapshot,
            emitted_text="C",
            choice_snapshot=choice_snapshot,
            choice=choice,
            advanced_snapshot=snapshot,
        )

        with self.assertRaises(SouthStarError) as raised:
            _ = outcome.choice_residual_attachment_evidence

        self.assertIs(
            raised.exception.kind,
            SouthStarErrorKind.INTERNAL_INVARIANT,
        )

    def test_writer_snapshot_advance_sequence_outcome_exposes_replayed_choice_evidence(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        snapshot_1 = replace(
            snapshot,
            decoder_boundary=writer_snapshot.WriterDecoderBoundary(
                consumed_token_count=1,
            ),
        )
        snapshot_2 = replace(
            snapshot,
            decoder_boundary=writer_snapshot.WriterDecoderBoundary(
                consumed_token_count=2,
            ),
        )
        key_c = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=7,
        )
        key_n = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=8,
        )
        support_c = self._test_frontier_next_token_support(
            emitted_text="C",
            successor_atom=AtomId(1),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CYCLIC_TREE_ENTRY
            ),
            residual_key=key_c,
        )
        support_n = self._test_frontier_next_token_support(
            emitted_text="N",
            successor_atom=AtomId(2),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CLOSURE_OPEN
            ),
            residual_key=key_n,
        )
        choice_c = self._test_choice_snapshot_entry_from_supports(
            "C",
            (support_c,),
        )
        choice_n = self._test_choice_snapshot_entry_from_supports(
            "N",
            (support_n,),
        )
        choice_snapshot_c = writer_frontier_module._WriterFrontierChoiceSnapshot(
            schedule_outcome=writer_frontier_module._WriterFrontierScheduleOutcome(
                state_outcomes=(),
                terminal_by_key=Counter(),
                grouped_by_text={},
                weighted_by_text={},
                next_token_frontier=(choice_c.next_token_entry,),
            ),
            terminal=None,
            choices=(choice_c,),
        )
        choice_snapshot_n = writer_frontier_module._WriterFrontierChoiceSnapshot(
            schedule_outcome=writer_frontier_module._WriterFrontierScheduleOutcome(
                state_outcomes=(),
                terminal_by_key=Counter(),
                grouped_by_text={},
                weighted_by_text={},
                next_token_frontier=(choice_n.next_token_entry,),
            ),
            terminal=None,
            choices=(choice_n,),
        )
        step_c = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceOutcomeKind.ADVANCED,
            source_snapshot=snapshot,
            emitted_text="C",
            choice_snapshot=choice_snapshot_c,
            choice=choice_c,
            advanced_snapshot=snapshot_1,
        )
        step_n = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceOutcomeKind.ADVANCED,
            source_snapshot=snapshot_1,
            emitted_text="N",
            choice_snapshot=choice_snapshot_n,
            choice=choice_n,
            advanced_snapshot=snapshot_2,
        )
        sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.ADVANCED,
            source_snapshot=snapshot,
            emitted_texts=("C", "N"),
            step_outcomes=(step_c, step_n),
            current_snapshot=snapshot_2,
        )

        self.assertEqual(
            sequence.choice_residual_attachment_evidence,
            (
                step_c.choice_residual_attachment_evidence,
                step_n.choice_residual_attachment_evidence,
            ),
        )
        self.assertEqual(
            sequence.selected_supports,
            (
                *step_c.choice_residual_attachment_evidence.selected_supports,
                *step_n.choice_residual_attachment_evidence.selected_supports,
            ),
        )
        self.assertEqual(
            sequence.selected_policy_families,
            (
                *step_c.choice_residual_attachment_evidence.selected_policy_families,
                *step_n.choice_residual_attachment_evidence.selected_policy_families,
            ),
        )

    def test_writer_snapshot_advance_sequence_outcome_exposes_only_consumed_evidence_after_failure(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        snapshot_1 = replace(
            snapshot,
            decoder_boundary=writer_snapshot.WriterDecoderBoundary(
                consumed_token_count=1,
            ),
        )
        key = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=7,
        )
        support = self._test_frontier_next_token_support(
            emitted_text="C",
            successor_atom=AtomId(1),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CYCLIC_TREE_ENTRY
            ),
            residual_key=key,
        )
        choice = self._test_choice_snapshot_entry_from_supports("C", (support,))
        choice_snapshot = writer_frontier_module._WriterFrontierChoiceSnapshot(
            schedule_outcome=writer_frontier_module._WriterFrontierScheduleOutcome(
                state_outcomes=(),
                terminal_by_key=Counter(),
                grouped_by_text={},
                weighted_by_text={},
                next_token_frontier=(choice.next_token_entry,),
            ),
            terminal=None,
            choices=(choice,),
        )
        step = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceOutcomeKind.ADVANCED,
            source_snapshot=snapshot,
            emitted_text="C",
            choice_snapshot=choice_snapshot,
            choice=choice,
            advanced_snapshot=snapshot_1,
        )
        failed = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotAdvanceOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot_1,
            emitted_text="bad",
            choice_snapshot=choice_snapshot,
        )
        sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotAdvanceSequenceOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_texts=("C", "bad", "not-attempted"),
            step_outcomes=(step, failed),
            current_snapshot=snapshot_1,
        )

        self.assertEqual(
            sequence.choice_residual_attachment_evidence,
            (step.choice_residual_attachment_evidence,),
        )
        self.assertEqual(sequence.consumed_emitted_texts, ("C",))
        self.assertEqual(
            sequence.remaining_emitted_texts,
            ("bad", "not-attempted"),
        )

    def test_writer_snapshot_replay_choice_snapshot_outcome_exposes_replayed_token_evidence(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        snapshot_1 = replace(
            snapshot,
            decoder_boundary=writer_snapshot.WriterDecoderBoundary(
                consumed_token_count=1,
            ),
        )
        key = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=7,
        )
        support = self._test_frontier_next_token_support(
            emitted_text="C",
            successor_atom=AtomId(1),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CYCLIC_TREE_ENTRY
            ),
            residual_key=key,
        )
        choice = self._test_choice_snapshot_entry_from_supports("C", (support,))
        choice_snapshot = writer_frontier_module._WriterFrontierChoiceSnapshot(
            schedule_outcome=writer_frontier_module._WriterFrontierScheduleOutcome(
                state_outcomes=(),
                terminal_by_key=Counter(),
                grouped_by_text={},
                weighted_by_text={},
                next_token_frontier=(choice.next_token_entry,),
            ),
            terminal=None,
            choices=(choice,),
        )
        step = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceOutcomeKind.ADVANCED,
            source_snapshot=snapshot,
            emitted_text="C",
            choice_snapshot=choice_snapshot,
            choice=choice,
            advanced_snapshot=snapshot_1,
        )
        sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.ADVANCED,
            source_snapshot=snapshot,
            emitted_texts=("C",),
            step_outcomes=(step,),
            current_snapshot=snapshot_1,
        )
        replay = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .CHOICE_SNAPSHOT
            ),
            source_snapshot=snapshot,
            emitted_texts=("C",),
            sequence_outcome=sequence,
            choice_snapshot=choice_snapshot,
        )

        self.assertEqual(
            replay.replayed_choice_residual_attachment_evidence,
            sequence.choice_residual_attachment_evidence,
        )
        self.assertEqual(
            replay.replayed_residual_attachment_evidence_groups,
            sequence.residual_attachment_evidence_groups,
        )
        self.assertEqual(
            replay.replayed_selected_supports,
            sequence.selected_supports,
        )
        self.assertEqual(
            replay.replayed_selected_policy_families,
            sequence.selected_policy_families,
        )

        failed = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotAdvanceOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot_1,
            emitted_text="bad",
            choice_snapshot=choice_snapshot,
        )
        failed_sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotAdvanceSequenceOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_texts=("C", "bad"),
            step_outcomes=(step, failed),
            current_snapshot=snapshot_1,
        )
        failed_replay = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_texts=("C", "bad"),
            sequence_outcome=failed_sequence,
        )

        self.assertEqual(
            failed_replay.replayed_choice_residual_attachment_evidence,
            (step.choice_residual_attachment_evidence,),
        )
        self.assertEqual(
            failed_replay.replayed_selected_supports,
            step.choice_residual_attachment_evidence.selected_supports,
        )

    def test_writer_snapshot_prefix_read_outcome_separates_replayed_and_final_choice_evidence(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        snapshot_1 = replace(
            snapshot,
            decoder_boundary=writer_snapshot.WriterDecoderBoundary(
                consumed_token_count=1,
            ),
        )
        replay_key = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=7,
        )
        final_key = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom=AtomId(0),
            attachment_id=8,
        )
        replay_support = self._test_frontier_next_token_support(
            emitted_text="C",
            successor_atom=AtomId(1),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CYCLIC_TREE_ENTRY
            ),
            residual_key=replay_key,
        )
        final_support = self._test_frontier_next_token_support(
            emitted_text="N",
            successor_atom=AtomId(2),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CLOSURE_OPEN
            ),
            residual_key=final_key,
        )
        replay_choice = self._test_choice_snapshot_entry_from_supports(
            "C",
            (replay_support,),
        )
        final_choice = self._test_choice_snapshot_entry_from_supports(
            "N",
            (final_support,),
        )
        replay_choice_snapshot = (
            writer_frontier_module._WriterFrontierChoiceSnapshot(
                schedule_outcome=(
                    writer_frontier_module._WriterFrontierScheduleOutcome(
                        state_outcomes=(),
                        terminal_by_key=Counter(),
                        grouped_by_text={},
                        weighted_by_text={},
                        next_token_frontier=(replay_choice.next_token_entry,),
                    )
                ),
                terminal=None,
                choices=(replay_choice,),
            )
        )
        final_choice_snapshot = (
            writer_frontier_module._WriterFrontierChoiceSnapshot(
                schedule_outcome=(
                    writer_frontier_module._WriterFrontierScheduleOutcome(
                        state_outcomes=(),
                        terminal_by_key=Counter(),
                        grouped_by_text={},
                        weighted_by_text={},
                        next_token_frontier=(final_choice.next_token_entry,),
                    )
                ),
                terminal=None,
                choices=(final_choice,),
            )
        )
        step = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceOutcomeKind.ADVANCED,
            source_snapshot=snapshot,
            emitted_text="C",
            choice_snapshot=replay_choice_snapshot,
            choice=replay_choice,
            advanced_snapshot=snapshot_1,
        )
        sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.ADVANCED,
            source_snapshot=snapshot,
            emitted_texts=("C",),
            step_outcomes=(step,),
            current_snapshot=snapshot_1,
        )
        replay = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .CHOICE_SNAPSHOT
            ),
            source_snapshot=snapshot,
            emitted_texts=("C",),
            sequence_outcome=sequence,
            choice_snapshot=final_choice_snapshot,
        )
        prefix_outcome = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=writer_snapshot._WriterSnapshotPrefixReadOutcomeKind.READABLE,
            replay_outcome=replay,
        )

        self.assertEqual(
            prefix_outcome.replayed_choice_residual_attachment_evidence,
            replay.replayed_choice_residual_attachment_evidence,
        )
        self.assertEqual(
            prefix_outcome.choice_residual_attachment_evidence,
            final_choice_snapshot.choice_residual_attachment_evidence,
        )
        self.assertNotEqual(
            prefix_outcome.replayed_choice_residual_attachment_evidence,
            prefix_outcome.choice_residual_attachment_evidence,
        )

    def test_writer_snapshot_empty_replay_has_no_replayed_token_evidence(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.ADVANCED,
            source_snapshot=snapshot,
            emitted_texts=(),
            step_outcomes=(),
            current_snapshot=snapshot,
        )
        replay = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .CHOICE_SNAPSHOT
            ),
            source_snapshot=snapshot,
            emitted_texts=(),
            sequence_outcome=sequence,
            choice_snapshot=self._test_frontier_choice_snapshot(()),
        )
        prefix_outcome = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=writer_snapshot._WriterSnapshotPrefixReadOutcomeKind.READABLE,
            replay_outcome=replay,
        )

        self.assertEqual(
            prefix_outcome.replayed_choice_residual_attachment_evidence,
            (),
        )
        self.assertEqual(
            prefix_outcome.replayed_residual_attachment_evidence_groups,
            (),
        )
        self.assertEqual(prefix_outcome.replayed_selected_supports, ())
        self.assertEqual(prefix_outcome.replayed_selected_policy_families, ())

    def test_writer_snapshot_prefix_read_outcome_exposes_dead_closure_resolved_cyclic_evidence(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        snapshot_1 = replace(
            snapshot,
            decoder_boundary=writer_snapshot.WriterDecoderBoundary(
                consumed_token_count=1,
            ),
        )
        replay_policy, replay_active_outcome = (
            self._dead_closure_open_active_emitted_outcome(AtomId(0))
        )
        final_policy, final_active_outcome = (
            self._dead_closure_open_active_emitted_outcome(AtomId(1))
        )
        replay_key = replay_policy.resolved_residual_attachment_policy_groups[0].key
        final_key = final_policy.resolved_residual_attachment_policy_groups[0].key
        replay_support = self._test_frontier_next_token_support(
            emitted_text="C",
            successor_atom=AtomId(2),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CYCLIC_TREE_ENTRY
            ),
            residual_key=replay_key,
        )
        final_support = self._test_frontier_next_token_support(
            emitted_text="N",
            successor_atom=AtomId(3),
            policy_family=(
                writer_transitions
                ._WriterGraphPolicyActionFamily
                .CYCLIC_TREE_ENTRY
            ),
            residual_key=final_key,
        )
        replay_choice = self._test_choice_snapshot_entry_from_supports(
            "C",
            (replay_support,),
        )
        final_choice = self._test_choice_snapshot_entry_from_supports(
            "N",
            (final_support,),
        )
        replay_top = writer_transitions._WriterTopLevelScheduleOutcome(
            kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.SCHEDULED,
            schedule_decision=writer_transitions._top_level_active_emitted_decision(
                replay_active_outcome.schedule_decision,
            ),
            active_emitted_outcome=replay_active_outcome,
        )
        final_top = writer_transitions._WriterTopLevelScheduleOutcome(
            kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.SCHEDULED,
            schedule_decision=writer_transitions._top_level_active_emitted_decision(
                final_active_outcome.schedule_decision,
            ),
            active_emitted_outcome=final_active_outcome,
        )
        replay_choice_snapshot = (
            writer_frontier_module._WriterFrontierChoiceSnapshot(
                schedule_outcome=(
                    writer_frontier_module._WriterFrontierScheduleOutcome(
                        state_outcomes=(
                            writer_frontier_module
                            ._WriterFrontierStateScheduleOutcome(
                                state_key=writer_state_key(
                                    _raw_initial_state(AtomId(0))
                                ),
                                parent_weight=1,
                                finalized_state_key=None,
                                schedule_outcome=replay_top,
                            ),
                        ),
                        terminal_by_key=Counter(),
                        grouped_by_text={},
                        weighted_by_text={},
                        next_token_frontier=(
                            replay_choice.next_token_entry,
                        ),
                    )
                ),
                terminal=None,
                choices=(replay_choice,),
            )
        )
        final_choice_snapshot = (
            writer_frontier_module._WriterFrontierChoiceSnapshot(
                schedule_outcome=(
                    writer_frontier_module._WriterFrontierScheduleOutcome(
                        state_outcomes=(
                            writer_frontier_module
                            ._WriterFrontierStateScheduleOutcome(
                                state_key=writer_state_key(
                                    _raw_initial_state(AtomId(1))
                                ),
                                parent_weight=1,
                                finalized_state_key=None,
                                schedule_outcome=final_top,
                            ),
                        ),
                        terminal_by_key=Counter(),
                        grouped_by_text={},
                        weighted_by_text={},
                        next_token_frontier=(final_choice.next_token_entry,),
                    )
                ),
                terminal=None,
                choices=(final_choice,),
            )
        )
        step = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceOutcomeKind.ADVANCED,
            source_snapshot=snapshot,
            emitted_text="C",
            choice_snapshot=replay_choice_snapshot,
            choice=replay_choice,
            advanced_snapshot=snapshot_1,
        )
        sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.ADVANCED,
            source_snapshot=snapshot,
            emitted_texts=("C",),
            step_outcomes=(step,),
            current_snapshot=snapshot_1,
        )
        replay = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .CHOICE_SNAPSHOT
            ),
            source_snapshot=snapshot,
            emitted_texts=("C",),
            sequence_outcome=sequence,
            choice_snapshot=final_choice_snapshot,
        )
        prefix_outcome = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=writer_snapshot._WriterSnapshotPrefixReadOutcomeKind.READABLE,
            replay_outcome=replay,
        )
        consumed_evidence = (
            replay_choice_snapshot
            .choice_residual_attachment_evidence_for_emitted_text("C")
        )
        final_evidence = (
            final_choice_snapshot
            .choice_residual_attachment_evidence_for_emitted_text("N")
        )

        self.assertEqual(
            (
                prefix_outcome
                .replayed_dead_closure_open_resolved_cyclic_tree_entry_evidence
            ),
            (consumed_evidence,),
        )
        self.assertEqual(
            (
                prefix_outcome
                .final_choice_dead_closure_open_resolved_cyclic_tree_entry_evidence
            ),
            (final_evidence,),
        )

        empty_sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.ADVANCED,
            source_snapshot=snapshot,
            emitted_texts=(),
            step_outcomes=(),
            current_snapshot=snapshot,
        )
        empty_replay = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .CHOICE_SNAPSHOT
            ),
            source_snapshot=snapshot,
            emitted_texts=(),
            sequence_outcome=empty_sequence,
            choice_snapshot=final_choice_snapshot,
        )
        empty_prefix = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=writer_snapshot._WriterSnapshotPrefixReadOutcomeKind.READABLE,
            replay_outcome=empty_replay,
        )

        self.assertEqual(
            (
                empty_prefix
                .replayed_dead_closure_open_resolved_cyclic_tree_entry_evidence
            ),
            (),
        )

    def test_writer_snapshot_failed_prefix_read_has_no_final_dead_closure_resolved_cyclic_evidence(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        blocked_choice_snapshot = self._test_blocked_frontier_choice_snapshot(cursor)
        blocked_step = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceOutcomeKind.BLOCKED,
            source_snapshot=snapshot,
            emitted_text="blocked",
            choice_snapshot=blocked_choice_snapshot,
        )
        blocked_sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.BLOCKED,
            source_snapshot=snapshot,
            emitted_texts=("blocked",),
            step_outcomes=(blocked_step,),
            current_snapshot=snapshot,
        )
        blocked_replay = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .REPLAY_BLOCKED
            ),
            source_snapshot=snapshot,
            emitted_texts=("blocked",),
            sequence_outcome=blocked_sequence,
        )
        blocked_prefix = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotPrefixReadOutcomeKind
                .REPLAY_BLOCKED
            ),
            replay_outcome=blocked_replay,
        )
        choice_snapshot = self._test_frontier_choice_snapshot(())
        invalid_step = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotAdvanceOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_text="bad",
            choice_snapshot=choice_snapshot,
        )
        invalid_sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotAdvanceSequenceOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_texts=("bad",),
            step_outcomes=(invalid_step,),
            current_snapshot=snapshot,
        )
        invalid_replay = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_texts=("bad",),
            sequence_outcome=invalid_sequence,
        )
        invalid_prefix = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotPrefixReadOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            replay_outcome=invalid_replay,
        )

        self.assertEqual(
            (
                blocked_prefix
                .final_choice_dead_closure_open_resolved_cyclic_tree_entry_evidence
            ),
            (),
        )
        self.assertEqual(
            (
                invalid_prefix
                .final_choice_dead_closure_open_resolved_cyclic_tree_entry_evidence
            ),
            (),
        )

    def test_writer_snapshot_prefix_read_outcome_returns_counted_readable_prefix(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )

        outcome = (
            writer_snapshot
            ._writer_snapshot_prefix_read_outcome_after_emitted_texts(
                snapshot,
                prepared=prepared,
                emitted_texts=(),
                include_counts=True,
            )
        )

        self.assertIs(
            outcome.kind,
            writer_snapshot._WriterSnapshotPrefixReadOutcomeKind.READABLE,
        )
        self.assertEqual(
            outcome.public_choices,
            writer_snapshot.resume_writer_frontier_choices_from_snapshot(
                snapshot,
                prepared=prepared,
            ),
        )
        self.assertEqual(
            outcome.support_count,
            count_writer_frontier_support(prepared, cursor.support_state),
        )
        self.assertEqual(
            outcome.completion_count,
            count_writer_cursor_completions(prepared, cursor),
        )

    def test_writer_snapshot_prefix_read_outcome_can_omit_counts(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )

        with patch(
            (
                "grimace._south_star1.writer_snapshot"
                "._count_writer_frontier_choice_snapshot_supports"
            ),
            side_effect=AssertionError("support count computed"),
        ), patch(
            (
                "grimace._south_star1.writer_snapshot"
                "._count_writer_frontier_choice_snapshot_completions"
            ),
            side_effect=AssertionError("completion count computed"),
        ):
            outcome = (
                writer_snapshot
                ._writer_snapshot_prefix_read_outcome_after_emitted_texts(
                    snapshot,
                    prepared=prepared,
                    emitted_texts=(),
                    include_counts=False,
                )
            )

        self.assertIs(
            outcome.kind,
            writer_snapshot._WriterSnapshotPrefixReadOutcomeKind.READABLE,
        )
        self.assertIsNone(outcome.support_count)
        self.assertIsNone(outcome.completion_count)
        self.assertIsNotNone(outcome.public_choices)

    def test_writer_snapshot_prefix_read_outcome_preserves_failed_replay_outcomes(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        blocked_choice_snapshot = self._test_blocked_frontier_choice_snapshot(cursor)
        blocked_step = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceOutcomeKind.BLOCKED,
            source_snapshot=snapshot,
            emitted_text="blocked",
            choice_snapshot=blocked_choice_snapshot,
        )
        blocked_sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.BLOCKED,
            source_snapshot=snapshot,
            emitted_texts=("blocked",),
            step_outcomes=(blocked_step,),
            current_snapshot=snapshot,
        )
        replay_blocked = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .REPLAY_BLOCKED
            ),
            source_snapshot=snapshot,
            emitted_texts=("blocked",),
            sequence_outcome=blocked_sequence,
        )
        choice_snapshot = writer_snapshot._writer_frontier_choice_snapshot_from_snapshot(
            snapshot,
            prepared=prepared,
        )
        invalid_step = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotAdvanceOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_text="bad",
            choice_snapshot=choice_snapshot,
        )
        invalid_sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotAdvanceSequenceOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_texts=("bad",),
            step_outcomes=(invalid_step,),
            current_snapshot=snapshot,
        )
        invalid_replay = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_texts=("bad",),
            sequence_outcome=invalid_sequence,
        )

        for replay_outcome, kind in (
            (
                replay_blocked,
                writer_snapshot._WriterSnapshotPrefixReadOutcomeKind.REPLAY_BLOCKED,
            ),
            (
                invalid_replay,
                (
                    writer_snapshot
                    ._WriterSnapshotPrefixReadOutcomeKind
                    .INVALID_EMITTED_TEXT
                ),
            ),
        ):
            with self.subTest(kind=kind):
                with patch(
                    (
                        "grimace._south_star1.writer_snapshot"
                        "._writer_frontier_choice_snapshot_after_emitted_texts"
                    ),
                    return_value=replay_outcome,
                ):
                    outcome = (
                        writer_snapshot
                        ._writer_snapshot_prefix_read_outcome_after_emitted_texts(
                            snapshot,
                            prepared=prepared,
                            emitted_texts=replay_outcome.emitted_texts,
                        )
                    )

                self.assertIs(outcome.kind, kind)
                self.assertIsNone(outcome.public_choices)
                self.assertIsNone(outcome.support_count)
                self.assertIsNone(outcome.completion_count)

    def test_writer_snapshot_prefix_read_outcome_preserves_final_blocked_frontier(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.ADVANCED,
            source_snapshot=snapshot,
            emitted_texts=(),
            step_outcomes=(),
            current_snapshot=snapshot,
        )
        blocked_choice_snapshot = self._test_blocked_frontier_choice_snapshot(cursor)
        replay_outcome = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .CHOICE_SNAPSHOT
            ),
            source_snapshot=snapshot,
            emitted_texts=(),
            sequence_outcome=sequence,
            choice_snapshot=blocked_choice_snapshot,
        )

        with patch(
            (
                "grimace._south_star1.writer_snapshot"
                "._writer_frontier_choice_snapshot_after_emitted_texts"
            ),
            return_value=replay_outcome,
        ):
            outcome = (
                writer_snapshot
                ._writer_snapshot_prefix_read_outcome_after_emitted_texts(
                    snapshot,
                    prepared=prepared,
                    emitted_texts=(),
                )
            )

        self.assertIs(
            outcome.kind,
            (
                writer_snapshot
                ._WriterSnapshotPrefixReadOutcomeKind
                .FINAL_FRONTIER_BLOCKED
            ),
        )
        self.assertTrue(outcome.blocked)
        self.assertEqual(
            outcome.graph_policy_blockers,
            blocked_choice_snapshot.graph_policy_blockers,
        )
        self.assertIsNone(outcome.support_count)
        self.assertIsNone(outcome.completion_count)

    def test_checked_writer_snapshot_prefix_read_outcome_returns_readable_outcome(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        outcome = (
            writer_snapshot
            ._writer_snapshot_prefix_read_outcome_after_emitted_texts(
                snapshot,
                prepared=prepared,
                emitted_texts=(),
            )
        )

        with patch(
            (
                "grimace._south_star1.writer_snapshot"
                "._writer_snapshot_prefix_read_outcome_after_emitted_texts"
            ),
            return_value=outcome,
        ):
            result = (
                writer_snapshot
                ._checked_writer_snapshot_prefix_read_outcome_after_emitted_texts(
                    snapshot,
                    prepared=prepared,
                    emitted_texts=(),
                )
            )

        self.assertIs(result, outcome)

    def test_checked_writer_snapshot_prefix_read_outcome_raises_for_failed_or_blocked_outcome(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        blocked_choice_snapshot = self._test_blocked_frontier_choice_snapshot(cursor)
        blocked_step = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceOutcomeKind.BLOCKED,
            source_snapshot=snapshot,
            emitted_text="blocked",
            choice_snapshot=blocked_choice_snapshot,
        )
        blocked_sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.BLOCKED,
            source_snapshot=snapshot,
            emitted_texts=("blocked",),
            step_outcomes=(blocked_step,),
            current_snapshot=snapshot,
        )
        replay_blocked = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .REPLAY_BLOCKED
            ),
            source_snapshot=snapshot,
            emitted_texts=("blocked",),
            sequence_outcome=blocked_sequence,
        )
        replay_blocked_read = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotPrefixReadOutcomeKind
                .REPLAY_BLOCKED
            ),
            replay_outcome=replay_blocked,
        )
        choice_snapshot = writer_snapshot._writer_frontier_choice_snapshot_from_snapshot(
            snapshot,
            prepared=prepared,
        )
        invalid_step = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotAdvanceOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_text="bad",
            choice_snapshot=choice_snapshot,
        )
        invalid_sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotAdvanceSequenceOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_texts=("bad",),
            step_outcomes=(invalid_step,),
            current_snapshot=snapshot,
        )
        invalid_replay = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_texts=("bad",),
            sequence_outcome=invalid_sequence,
        )
        invalid_read = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotPrefixReadOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            replay_outcome=invalid_replay,
        )
        sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.ADVANCED,
            source_snapshot=snapshot,
            emitted_texts=(),
            step_outcomes=(),
            current_snapshot=snapshot,
        )
        final_blocked_replay = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .CHOICE_SNAPSHOT
            ),
            source_snapshot=snapshot,
            emitted_texts=(),
            sequence_outcome=sequence,
            choice_snapshot=blocked_choice_snapshot,
        )
        final_blocked_read = writer_snapshot._WriterSnapshotPrefixReadOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotPrefixReadOutcomeKind
                .FINAL_FRONTIER_BLOCKED
            ),
            replay_outcome=final_blocked_replay,
        )

        cases = (
            (replay_blocked_read, SouthStarErrorKind.UNSUPPORTED_POLICY),
            (invalid_read, SouthStarErrorKind.INVALID_FACTS),
            (final_blocked_read, SouthStarErrorKind.UNSUPPORTED_POLICY),
        )

        for outcome, kind in cases:
            with self.subTest(kind=outcome.kind):
                with patch(
                    (
                        "grimace._south_star1.writer_snapshot"
                        "._writer_snapshot_prefix_read_outcome_after_emitted_texts"
                    ),
                    return_value=outcome,
                ):
                    with self.assertRaises(SouthStarError) as raised:
                        (
                            writer_snapshot
                            ._checked_writer_snapshot_prefix_read_outcome_after_emitted_texts(
                                snapshot,
                                prepared=prepared,
                                emitted_texts=outcome.emitted_texts,
                            )
                        )

                self.assertIs(raised.exception.kind, kind)

    def test_replayed_prefix_helpers_route_through_checked_prefix_read(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        outcome = (
            writer_snapshot
            ._writer_snapshot_prefix_read_outcome_after_emitted_texts(
                snapshot,
                prepared=prepared,
                emitted_texts=(),
                include_counts=True,
            )
        )

        with patch(
            (
                "grimace._south_star1.writer_snapshot"
                "._checked_writer_snapshot_prefix_read_outcome_after_emitted_texts"
            ),
            return_value=outcome,
        ) as checked_read:
            writer_snapshot._writer_frontier_choices_after_emitted_texts(
                snapshot,
                prepared=prepared,
                emitted_texts=(),
            )
            writer_snapshot._count_writer_frontier_support_after_emitted_texts(
                snapshot,
                prepared=prepared,
                emitted_texts=(),
            )
            writer_snapshot._count_writer_completions_after_emitted_texts(
                snapshot,
                prepared=prepared,
                emitted_texts=(),
            )

        self.assertEqual(
            tuple(call.kwargs["include_counts"] for call in checked_read.call_args_list),
            (True, True, True),
        )

        with patch(
            (
                "grimace._south_star1.writer_snapshot"
                "._checked_writer_snapshot_prefix_read_outcome_after_emitted_texts"
            ),
            return_value=outcome,
        ) as checked_read, patch(
            (
                "grimace._south_star1.writer_snapshot"
                "._iter_writer_frontier_support_suffixes_from_choice_snapshot"
            ),
            return_value=iter(("C",)),
        ):
            tuple(
                writer_snapshot
                ._iter_writer_frontier_support_suffixes_after_emitted_texts(
                    snapshot,
                    prepared=prepared,
                    emitted_texts=(),
                )
            )

        checked_read.assert_called_once_with(
            snapshot,
            prepared=prepared,
            emitted_texts=(),
            include_counts=False,
        )

    def test_writer_choice_snapshot_from_snapshot_validates_before_frontier_snapshot(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        other_prepared = _prepare(cco_facts())
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )

        with patch(
            "grimace._south_star1.writer_snapshot._writer_frontier_choice_snapshot",
            side_effect=AssertionError("invalid snapshot reached frontier"),
        ) as build_snapshot:
            with self.assertRaises(SouthStarError) as raised:
                writer_snapshot._writer_frontier_choice_snapshot_from_snapshot(
                    snapshot,
                    prepared=other_prepared,
                )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.INVALID_FACTS)
        build_snapshot.assert_not_called()

    def test_maybe_writer_choice_snapshot_entry_for_emitted_text_returns_none_for_missing_text(self) -> None:
        c_choice = self._test_choice_snapshot_entry(
            "C",
            successor_atom=AtomId(1),
        )
        choice_snapshot = self._test_frontier_choice_snapshot((c_choice,))

        self.assertIs(
            (
                writer_snapshot
                ._maybe_writer_frontier_choice_snapshot_entry_for_emitted_text(
                    choice_snapshot,
                    "C",
                )
            ),
            c_choice,
        )
        self.assertIsNone(
            (
                writer_snapshot
                ._maybe_writer_frontier_choice_snapshot_entry_for_emitted_text(
                    choice_snapshot,
                    "N",
                )
            )
        )

        with self.assertRaises(SouthStarError) as raised:
            writer_snapshot._writer_frontier_choice_snapshot_entry_for_emitted_text(
                choice_snapshot,
                "N",
            )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.INVALID_FACTS)

    def test_maybe_writer_choice_snapshot_entry_for_emitted_text_rejects_duplicate_text(self) -> None:
        first = self._test_choice_snapshot_entry(
            "C",
            successor_atom=AtomId(1),
        )
        second = self._test_choice_snapshot_entry(
            "C",
            successor_atom=AtomId(2),
        )
        choice_snapshot = self._test_frontier_choice_snapshot((first, second))

        with self.assertRaises(SouthStarError) as maybe_raised:
            (
                writer_snapshot
                ._maybe_writer_frontier_choice_snapshot_entry_for_emitted_text(
                    choice_snapshot,
                    "C",
                )
            )

        self.assertIs(
            maybe_raised.exception.kind,
            SouthStarErrorKind.INTERNAL_INVARIANT,
        )

        with self.assertRaises(SouthStarError) as checked_raised:
            writer_snapshot._writer_frontier_choice_snapshot_entry_for_emitted_text(
                choice_snapshot,
                "C",
            )

        self.assertIs(
            checked_raised.exception.kind,
            SouthStarErrorKind.INTERNAL_INVARIANT,
        )

    def test_writer_snapshot_advance_outcome_validates_payload_shape(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        source_snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        choice_snapshot = writer_snapshot._writer_frontier_choice_snapshot_from_snapshot(
            source_snapshot,
            prepared=prepared,
            include_counts=False,
        )
        choice = choice_snapshot.choices[0]
        advanced_snapshot = (
            writer_snapshot
            ._writer_search_snapshot_with_cursor_after_emitted_text(
                source_snapshot,
                prepared=prepared,
                cursor=choice.successor,
            )
        )
        blocked_choice_snapshot = self._test_blocked_frontier_choice_snapshot(cursor)

        advanced = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceOutcomeKind.ADVANCED,
            source_snapshot=source_snapshot,
            emitted_text=choice.emitted_text,
            choice_snapshot=choice_snapshot,
            choice=choice,
            advanced_snapshot=advanced_snapshot,
        )
        blocked = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceOutcomeKind.BLOCKED,
            source_snapshot=source_snapshot,
            emitted_text=choice.emitted_text,
            choice_snapshot=blocked_choice_snapshot,
        )
        invalid = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotAdvanceOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=source_snapshot,
            emitted_text="N",
            choice_snapshot=choice_snapshot,
        )

        self.assertFalse(advanced.blocked)
        self.assertTrue(blocked.blocked)
        self.assertTrue(invalid.invalid_emitted_text)

        invalid_shapes = (
            dict(
                kind=writer_snapshot._WriterSnapshotAdvanceOutcomeKind.ADVANCED,
                source_snapshot=source_snapshot,
                emitted_text=choice.emitted_text,
                choice_snapshot=choice_snapshot,
                choice=choice,
            ),
            dict(
                kind=writer_snapshot._WriterSnapshotAdvanceOutcomeKind.BLOCKED,
                source_snapshot=source_snapshot,
                emitted_text=choice.emitted_text,
                choice_snapshot=blocked_choice_snapshot,
                choice=choice,
            ),
            dict(
                kind=(
                    writer_snapshot
                    ._WriterSnapshotAdvanceOutcomeKind
                    .INVALID_EMITTED_TEXT
                ),
                source_snapshot=source_snapshot,
                emitted_text="N",
                choice_snapshot=blocked_choice_snapshot,
            ),
        )

        for kwargs in invalid_shapes:
            with self.subTest(kind=kwargs["kind"]):
                with self.assertRaises(SouthStarError) as raised:
                    writer_snapshot._WriterSnapshotAdvanceOutcome(**kwargs)

                self.assertIs(
                    raised.exception.kind,
                    SouthStarErrorKind.INTERNAL_INVARIANT,
                )

    def test_writer_snapshot_advance_sequence_outcome_validates_advanced_payload(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        source_snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        first_text = (
            writer_snapshot
            ._writer_frontier_choice_snapshot_from_snapshot(
                source_snapshot,
                prepared=prepared,
                include_counts=False,
            )
            .choices[0]
            .emitted_text
        )
        first = self._test_writer_snapshot_advance_step(
            source_snapshot,
            emitted_text=first_text,
            prepared=prepared,
        )
        assert first.advanced_snapshot is not None
        second_text = (
            writer_snapshot
            ._writer_frontier_choice_snapshot_from_snapshot(
                first.advanced_snapshot,
                prepared=prepared,
                include_counts=False,
            )
            .choices[0]
            .emitted_text
        )
        second = self._test_writer_snapshot_advance_step(
            first.advanced_snapshot,
            emitted_text=second_text,
            prepared=prepared,
        )
        assert second.advanced_snapshot is not None

        empty = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.ADVANCED,
            source_snapshot=source_snapshot,
            emitted_texts=(),
            step_outcomes=(),
            current_snapshot=source_snapshot,
        )
        replayed = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.ADVANCED,
            source_snapshot=source_snapshot,
            emitted_texts=(first_text, second_text),
            step_outcomes=(first, second),
            current_snapshot=second.advanced_snapshot,
        )

        self.assertIs(empty.advanced_snapshot, source_snapshot)
        self.assertEqual(replayed.consumed_emitted_texts, (first_text, second_text))
        self.assertEqual(replayed.remaining_emitted_texts, ())

        invalid_payloads = (
            dict(
                kind=(
                    writer_snapshot
                    ._WriterSnapshotAdvanceSequenceOutcomeKind
                    .ADVANCED
                ),
                source_snapshot=source_snapshot,
                emitted_texts=(first_text,),
                step_outcomes=(),
                current_snapshot=source_snapshot,
            ),
            dict(
                kind=(
                    writer_snapshot
                    ._WriterSnapshotAdvanceSequenceOutcomeKind
                    .ADVANCED
                ),
                source_snapshot=source_snapshot,
                emitted_texts=(first_text,),
                step_outcomes=(first,),
                current_snapshot=source_snapshot,
            ),
            dict(
                kind=(
                    writer_snapshot
                    ._WriterSnapshotAdvanceSequenceOutcomeKind
                    .ADVANCED
                ),
                source_snapshot=source_snapshot,
                emitted_texts=(first_text, "blocked"),
                step_outcomes=(
                    first,
                    writer_snapshot._WriterSnapshotAdvanceOutcome(
                        kind=(
                            writer_snapshot
                            ._WriterSnapshotAdvanceOutcomeKind
                            .BLOCKED
                        ),
                        source_snapshot=first.advanced_snapshot,
                        emitted_text="blocked",
                        choice_snapshot=self._test_blocked_frontier_choice_snapshot(
                            first.advanced_snapshot.cursor,
                        ),
                    ),
                ),
                current_snapshot=first.advanced_snapshot,
            ),
        )

        for kwargs in invalid_payloads:
            with self.subTest(kind=kwargs["kind"]):
                with self.assertRaises(SouthStarError) as raised:
                    writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(**kwargs)

                self.assertIs(
                    raised.exception.kind,
                    SouthStarErrorKind.INTERNAL_INVARIANT,
                )

    def test_writer_snapshot_advance_sequence_outcome_validates_failed_payloads(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        source_snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        first_text = (
            writer_snapshot
            ._writer_frontier_choice_snapshot_from_snapshot(
                source_snapshot,
                prepared=prepared,
                include_counts=False,
            )
            .choices[0]
            .emitted_text
        )
        first = self._test_writer_snapshot_advance_step(
            source_snapshot,
            emitted_text=first_text,
            prepared=prepared,
        )
        assert first.advanced_snapshot is not None
        blocked = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceOutcomeKind.BLOCKED,
            source_snapshot=first.advanced_snapshot,
            emitted_text="blocked",
            choice_snapshot=self._test_blocked_frontier_choice_snapshot(
                first.advanced_snapshot.cursor,
            ),
        )
        invalid = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotAdvanceOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=first.advanced_snapshot,
            emitted_text="bad",
            choice_snapshot=(
                writer_snapshot
                ._writer_frontier_choice_snapshot_from_snapshot(
                    first.advanced_snapshot,
                    prepared=prepared,
                    include_counts=False,
                )
            ),
        )

        blocked_outcome = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.BLOCKED,
            source_snapshot=source_snapshot,
            emitted_texts=(first_text, "blocked", "later"),
            step_outcomes=(first, blocked),
            current_snapshot=first.advanced_snapshot,
        )
        invalid_outcome = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotAdvanceSequenceOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=source_snapshot,
            emitted_texts=(first_text, "bad", "later"),
            step_outcomes=(first, invalid),
            current_snapshot=first.advanced_snapshot,
        )

        self.assertTrue(blocked_outcome.blocked)
        self.assertEqual(blocked_outcome.consumed_emitted_texts, (first_text,))
        self.assertEqual(blocked_outcome.remaining_emitted_texts, ("blocked", "later"))
        self.assertIs(blocked_outcome.failed_outcome, blocked)
        self.assertIsNone(blocked_outcome.advanced_snapshot)
        self.assertTrue(invalid_outcome.invalid_emitted_text)
        self.assertEqual(invalid_outcome.consumed_emitted_texts, (first_text,))
        self.assertEqual(invalid_outcome.remaining_emitted_texts, ("bad", "later"))
        self.assertIs(invalid_outcome.failed_outcome, invalid)

    def test_writer_snapshot_replay_choice_snapshot_outcome_validates_payload_shape(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        source_snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        choice_snapshot = writer_snapshot._writer_frontier_choice_snapshot_from_snapshot(
            source_snapshot,
            prepared=prepared,
            include_counts=False,
        )
        advanced_sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.ADVANCED,
            source_snapshot=source_snapshot,
            emitted_texts=(),
            step_outcomes=(),
            current_snapshot=source_snapshot,
        )
        blocked_step = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceOutcomeKind.BLOCKED,
            source_snapshot=source_snapshot,
            emitted_text="blocked",
            choice_snapshot=self._test_blocked_frontier_choice_snapshot(cursor),
        )
        blocked_sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.BLOCKED,
            source_snapshot=source_snapshot,
            emitted_texts=("blocked",),
            step_outcomes=(blocked_step,),
            current_snapshot=source_snapshot,
        )
        invalid_step = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotAdvanceOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=source_snapshot,
            emitted_text="bad",
            choice_snapshot=choice_snapshot,
        )
        invalid_sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotAdvanceSequenceOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=source_snapshot,
            emitted_texts=("bad",),
            step_outcomes=(invalid_step,),
            current_snapshot=source_snapshot,
        )

        choice_outcome = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .CHOICE_SNAPSHOT
            ),
            source_snapshot=source_snapshot,
            emitted_texts=(),
            sequence_outcome=advanced_sequence,
            choice_snapshot=choice_snapshot,
        )
        blocked_outcome = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .REPLAY_BLOCKED
            ),
            source_snapshot=source_snapshot,
            emitted_texts=("blocked",),
            sequence_outcome=blocked_sequence,
        )
        invalid_outcome = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=source_snapshot,
            emitted_texts=("bad",),
            sequence_outcome=invalid_sequence,
        )

        self.assertTrue(choice_outcome.replay_succeeded)
        self.assertTrue(blocked_outcome.replay_failed)
        self.assertTrue(invalid_outcome.invalid_emitted_text)

        invalid_payloads = (
            dict(
                kind=(
                    writer_snapshot
                    ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                    .CHOICE_SNAPSHOT
                ),
                source_snapshot=source_snapshot,
                emitted_texts=(),
                sequence_outcome=advanced_sequence,
            ),
            dict(
                kind=(
                    writer_snapshot
                    ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                    .CHOICE_SNAPSHOT
                ),
                source_snapshot=source_snapshot,
                emitted_texts=("blocked",),
                sequence_outcome=blocked_sequence,
                choice_snapshot=choice_snapshot,
            ),
            dict(
                kind=(
                    writer_snapshot
                    ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                    .REPLAY_BLOCKED
                ),
                source_snapshot=source_snapshot,
                emitted_texts=("blocked",),
                sequence_outcome=blocked_sequence,
                choice_snapshot=choice_snapshot,
            ),
            dict(
                kind=(
                    writer_snapshot
                    ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                    .INVALID_EMITTED_TEXT
                ),
                source_snapshot=source_snapshot,
                emitted_texts=(),
                sequence_outcome=advanced_sequence,
            ),
        )

        for kwargs in invalid_payloads:
            with self.subTest(kind=kwargs["kind"]):
                with self.assertRaises(SouthStarError) as raised:
                    writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
                        **kwargs
                    )

                self.assertIs(
                    raised.exception.kind,
                    SouthStarErrorKind.INTERNAL_INVARIANT,
                )

    def test_writer_frontier_choice_snapshot_after_empty_emitted_texts_returns_current_frontier(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )

        outcome = writer_snapshot._writer_frontier_choice_snapshot_after_emitted_texts(
            snapshot,
            prepared=prepared,
            emitted_texts=(),
            include_counts=False,
        )
        direct = writer_snapshot._writer_frontier_choice_snapshot_from_snapshot(
            snapshot,
            prepared=prepared,
            include_counts=False,
        )

        self.assertIs(
            outcome.kind,
            (
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .CHOICE_SNAPSHOT
            ),
        )
        self.assertTrue(outcome.replay_succeeded)
        self.assertEqual(outcome.advanced_snapshot, snapshot)
        self.assertIsNotNone(outcome.choice_snapshot)
        self.assertEqual(
            outcome.choice_snapshot.public_choices,
            direct.public_choices,
        )

    def test_writer_frontier_choice_snapshot_after_emitted_texts_returns_final_prefix_frontier(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        first_text = (
            writer_snapshot
            .resume_writer_frontier_choices_from_snapshot(
                snapshot,
                prepared=prepared,
            )
            .choices[0]
            .emitted_text
        )
        advanced = writer_snapshot._advance_writer_search_snapshot_by_emitted_text(
            snapshot,
            prepared=prepared,
            emitted_text=first_text,
        )

        outcome = writer_snapshot._writer_frontier_choice_snapshot_after_emitted_texts(
            snapshot,
            prepared=prepared,
            emitted_texts=(first_text,),
            include_counts=False,
        )
        direct = writer_snapshot._writer_frontier_choice_snapshot_from_snapshot(
            advanced,
            prepared=prepared,
            include_counts=False,
        )

        self.assertIs(
            outcome.kind,
            (
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .CHOICE_SNAPSHOT
            ),
        )
        self.assertEqual(outcome.advanced_snapshot, advanced)
        self.assertIsNotNone(outcome.choice_snapshot)
        self.assertEqual(
            outcome.choice_snapshot.public_choices,
            direct.public_choices,
        )

    def test_writer_frontier_choice_snapshot_after_emitted_texts_returns_invalid_text_outcome(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )

        outcome = writer_snapshot._writer_frontier_choice_snapshot_after_emitted_texts(
            snapshot,
            prepared=prepared,
            emitted_texts=("not-a-frontier-token",),
        )

        self.assertIs(
            outcome.kind,
            (
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
        )
        self.assertTrue(outcome.replay_failed)
        self.assertTrue(outcome.invalid_emitted_text)
        self.assertIsNone(outcome.choice_snapshot)
        self.assertTrue(outcome.sequence_outcome.invalid_emitted_text)

    def test_writer_frontier_choice_snapshot_after_emitted_texts_returns_replay_blocked_outcome(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        blocked_step = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceOutcomeKind.BLOCKED,
            source_snapshot=snapshot,
            emitted_text="blocked",
            choice_snapshot=self._test_blocked_frontier_choice_snapshot(cursor),
        )
        sequence_outcome = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.BLOCKED,
            source_snapshot=snapshot,
            emitted_texts=("blocked",),
            step_outcomes=(blocked_step,),
            current_snapshot=snapshot,
        )

        with patch(
            (
                "grimace._south_star1.writer_snapshot"
                "._writer_snapshot_advance_sequence_outcome_by_emitted_texts"
            ),
            return_value=sequence_outcome,
        ):
            outcome = (
                writer_snapshot
                ._writer_frontier_choice_snapshot_after_emitted_texts(
                    snapshot,
                    prepared=prepared,
                    emitted_texts=("blocked",),
                )
            )

        self.assertIs(
            outcome.kind,
            (
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .REPLAY_BLOCKED
            ),
        )
        self.assertTrue(outcome.replay_failed)
        self.assertIsNone(outcome.choice_snapshot)
        self.assertEqual(
            outcome.graph_policy_blockers,
            sequence_outcome.graph_policy_blockers,
        )

    def test_writer_frontier_choice_snapshot_after_emitted_texts_preserves_final_blocked_frontier(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        sequence_outcome = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.ADVANCED,
            source_snapshot=snapshot,
            emitted_texts=(),
            step_outcomes=(),
            current_snapshot=snapshot,
        )
        blocked_choice_snapshot = self._test_blocked_frontier_choice_snapshot(cursor)

        with patch(
            (
                "grimace._south_star1.writer_snapshot"
                "._writer_snapshot_advance_sequence_outcome_by_emitted_texts"
            ),
            return_value=sequence_outcome,
        ), patch(
            (
                "grimace._south_star1.writer_snapshot"
                "._writer_frontier_choice_snapshot_from_snapshot"
            ),
            return_value=blocked_choice_snapshot,
        ):
            outcome = (
                writer_snapshot
                ._writer_frontier_choice_snapshot_after_emitted_texts(
                    snapshot,
                    prepared=prepared,
                    emitted_texts=(),
                )
            )

        self.assertIs(
            outcome.kind,
            (
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .CHOICE_SNAPSHOT
            ),
        )
        self.assertTrue(outcome.replay_succeeded)
        self.assertIs(outcome.choice_snapshot, blocked_choice_snapshot)
        self.assertTrue(outcome.blocked)
        self.assertEqual(
            outcome.graph_policy_blockers,
            blocked_choice_snapshot.graph_policy_blockers,
        )

    def test_checked_writer_frontier_choice_snapshot_after_emitted_texts_returns_snapshot(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        choice_snapshot = writer_snapshot._writer_frontier_choice_snapshot_from_snapshot(
            snapshot,
            prepared=prepared,
            include_counts=False,
        )
        sequence_outcome = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.ADVANCED,
            source_snapshot=snapshot,
            emitted_texts=(),
            step_outcomes=(),
            current_snapshot=snapshot,
        )
        outcome = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .CHOICE_SNAPSHOT
            ),
            source_snapshot=snapshot,
            emitted_texts=(),
            sequence_outcome=sequence_outcome,
            choice_snapshot=choice_snapshot,
        )

        with patch(
            (
                "grimace._south_star1.writer_snapshot"
                "._writer_frontier_choice_snapshot_after_emitted_texts"
            ),
            return_value=outcome,
        ):
            result = (
                writer_snapshot
                ._checked_writer_frontier_choice_snapshot_after_emitted_texts(
                    snapshot,
                    prepared=prepared,
                    emitted_texts=(),
                    include_counts=False,
                )
            )

        self.assertIs(result, choice_snapshot)

    def test_checked_writer_frontier_choice_snapshot_after_emitted_texts_raises_for_failed_or_blocked_outcome(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        choice_snapshot = writer_snapshot._writer_frontier_choice_snapshot_from_snapshot(
            snapshot,
            prepared=prepared,
            include_counts=False,
        )
        invalid_step = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotAdvanceOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_text="bad",
            choice_snapshot=choice_snapshot,
        )
        invalid_sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotAdvanceSequenceOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_texts=("bad",),
            step_outcomes=(invalid_step,),
            current_snapshot=snapshot,
        )
        invalid_outcome = writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_texts=("bad",),
            sequence_outcome=invalid_sequence,
        )
        blocked_choice_snapshot = self._test_blocked_frontier_choice_snapshot(cursor)
        blocked_step = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceOutcomeKind.BLOCKED,
            source_snapshot=snapshot,
            emitted_text="blocked",
            choice_snapshot=blocked_choice_snapshot,
        )
        blocked_sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.BLOCKED,
            source_snapshot=snapshot,
            emitted_texts=("blocked",),
            step_outcomes=(blocked_step,),
            current_snapshot=snapshot,
        )
        replay_blocked_outcome = (
            writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
                kind=(
                    writer_snapshot
                    ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                    .REPLAY_BLOCKED
                ),
                source_snapshot=snapshot,
                emitted_texts=("blocked",),
                sequence_outcome=blocked_sequence,
            )
        )
        final_blocked_sequence = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.ADVANCED,
            source_snapshot=snapshot,
            emitted_texts=(),
            step_outcomes=(),
            current_snapshot=snapshot,
        )
        final_blocked_outcome = (
            writer_snapshot._WriterSnapshotReplayChoiceSnapshotOutcome(
                kind=(
                    writer_snapshot
                    ._WriterSnapshotReplayChoiceSnapshotOutcomeKind
                    .CHOICE_SNAPSHOT
                ),
                source_snapshot=snapshot,
                emitted_texts=(),
                sequence_outcome=final_blocked_sequence,
                choice_snapshot=blocked_choice_snapshot,
            )
        )

        cases = (
            (invalid_outcome, SouthStarErrorKind.INVALID_FACTS),
            (replay_blocked_outcome, SouthStarErrorKind.UNSUPPORTED_POLICY),
            (final_blocked_outcome, SouthStarErrorKind.UNSUPPORTED_POLICY),
        )

        for outcome, kind in cases:
            with self.subTest(kind=outcome.kind):
                with patch(
                    (
                        "grimace._south_star1.writer_snapshot"
                        "._writer_frontier_choice_snapshot_after_emitted_texts"
                    ),
                    return_value=outcome,
                ):
                    with self.assertRaises(SouthStarError) as raised:
                        (
                            writer_snapshot
                            ._checked_writer_frontier_choice_snapshot_after_emitted_texts(
                                snapshot,
                                prepared=prepared,
                                emitted_texts=outcome.emitted_texts,
                            )
                        )

                self.assertIs(raised.exception.kind, kind)

    def test_writer_frontier_choice_snapshot_after_emitted_texts_forwards_include_counts_to_final_snapshot(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        first_text = (
            writer_snapshot
            ._writer_frontier_choice_snapshot_from_snapshot(
                snapshot,
                prepared=prepared,
                include_counts=False,
            )
            .choices[0]
            .emitted_text
        )
        first_step = self._test_writer_snapshot_advance_step(
            snapshot,
            emitted_text=first_text,
            prepared=prepared,
        )
        assert first_step.advanced_snapshot is not None
        sequence_outcome = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.ADVANCED,
            source_snapshot=snapshot,
            emitted_texts=(first_text,),
            step_outcomes=(first_step,),
            current_snapshot=first_step.advanced_snapshot,
        )
        choice_snapshot = writer_snapshot._writer_frontier_choice_snapshot_from_snapshot(
            first_step.advanced_snapshot,
            prepared=prepared,
            include_counts=False,
        )

        with patch(
            (
                "grimace._south_star1.writer_snapshot"
                "._writer_snapshot_advance_sequence_outcome_by_emitted_texts"
            ),
            return_value=sequence_outcome,
        ), patch(
            (
                "grimace._south_star1.writer_snapshot"
                "._writer_frontier_choice_snapshot_from_snapshot"
            ),
            return_value=choice_snapshot,
        ) as final_snapshot:
            writer_snapshot._writer_frontier_choice_snapshot_after_emitted_texts(
                snapshot,
                prepared=prepared,
                emitted_texts=(first_text,),
                include_counts=False,
            )

        final_snapshot.assert_called_once_with(
            first_step.advanced_snapshot,
            prepared=prepared,
            include_counts=False,
            stop_after_first_blocked=False,
        )

    def test_writer_choice_snapshot_entry_for_emitted_text_returns_match(self) -> None:
        parent_key = writer_state_key(_raw_initial_state(AtomId(0)))
        c_key = writer_state_key(_raw_initial_state(AtomId(1)))
        n_key = writer_state_key(_raw_initial_state(AtomId(2)))
        family = writer_transitions._WriterGraphPolicyActionFamily.ACYCLIC_TREE_ENTRY
        c_entry = writer_frontier_module._WriterFrontierNextTokenEntry(
            emitted_text="C",
            supports=(
                writer_frontier_module._WriterFrontierNextTokenSupport(
                    state_key=parent_key,
                    parent_weight=1,
                    schedule_support=SimpleNamespace(
                        emitted_text="C",
                        graph_action_surface=object(),
                        policy_family=family,
                    ),  # type: ignore[arg-type]
                    successor_key=c_key,
                ),
            ),
        )
        n_entry = writer_frontier_module._WriterFrontierNextTokenEntry(
            emitted_text="N",
            supports=(
                writer_frontier_module._WriterFrontierNextTokenSupport(
                    state_key=parent_key,
                    parent_weight=1,
                    schedule_support=SimpleNamespace(
                        emitted_text="N",
                        graph_action_surface=object(),
                        policy_family=family,
                    ),  # type: ignore[arg-type]
                    successor_key=n_key,
                ),
            ),
        )
        c_choice = writer_frontier_module._WriterFrontierChoiceSnapshotEntry(
            next_token_entry=c_entry,
            successor=WriterFrontierCursor(
                weighted_states=tuple(c_entry.weighted_successors.items())
            ),
        )
        n_choice = writer_frontier_module._WriterFrontierChoiceSnapshotEntry(
            next_token_entry=n_entry,
            successor=WriterFrontierCursor(
                weighted_states=tuple(n_entry.weighted_successors.items())
            ),
        )
        choice_snapshot = writer_frontier_module._WriterFrontierChoiceSnapshot(
            schedule_outcome=writer_frontier_module._WriterFrontierScheduleOutcome(
                state_outcomes=(),
                terminal_by_key=Counter(),
                grouped_by_text={},
                weighted_by_text={},
            ),
            terminal=None,
            choices=(c_choice, n_choice),
        )

        self.assertIs(
            writer_snapshot._writer_frontier_choice_snapshot_entry_for_emitted_text(
                choice_snapshot,
                "C",
            ),
            c_choice,
        )
        self.assertIs(
            writer_snapshot._writer_frontier_choice_snapshot_entry_for_emitted_text(
                choice_snapshot,
                "N",
            ),
            n_choice,
        )

    def test_writer_choice_snapshot_entry_for_emitted_text_rejects_missing_text(self) -> None:
        choice_snapshot = writer_frontier_module._WriterFrontierChoiceSnapshot(
            schedule_outcome=writer_frontier_module._WriterFrontierScheduleOutcome(
                state_outcomes=(),
                terminal_by_key=Counter(),
                grouped_by_text={},
                weighted_by_text={},
            ),
            terminal=None,
            choices=(),
        )

        with self.assertRaises(SouthStarError) as raised:
            writer_snapshot._writer_frontier_choice_snapshot_entry_for_emitted_text(
                choice_snapshot,
                "O",
            )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.INVALID_FACTS)

    def test_writer_choice_snapshot_entry_for_emitted_text_rejects_duplicate_text(self) -> None:
        parent_key = writer_state_key(_raw_initial_state(AtomId(0)))
        successor_key = writer_state_key(_raw_initial_state(AtomId(1)))
        family = writer_transitions._WriterGraphPolicyActionFamily.ACYCLIC_TREE_ENTRY
        entry = writer_frontier_module._WriterFrontierNextTokenEntry(
            emitted_text="C",
            supports=(
                writer_frontier_module._WriterFrontierNextTokenSupport(
                    state_key=parent_key,
                    parent_weight=1,
                    schedule_support=SimpleNamespace(
                        emitted_text="C",
                        graph_action_surface=object(),
                        policy_family=family,
                    ),  # type: ignore[arg-type]
                    successor_key=successor_key,
                ),
            ),
        )
        choice = writer_frontier_module._WriterFrontierChoiceSnapshotEntry(
            next_token_entry=entry,
            successor=WriterFrontierCursor(
                weighted_states=tuple(entry.weighted_successors.items())
            ),
        )
        choice_snapshot = writer_frontier_module._WriterFrontierChoiceSnapshot(
            schedule_outcome=writer_frontier_module._WriterFrontierScheduleOutcome(
                state_outcomes=(),
                terminal_by_key=Counter(),
                grouped_by_text={},
                weighted_by_text={},
            ),
            terminal=None,
            choices=(choice, choice),
        )

        with self.assertRaises(SouthStarError) as raised:
            writer_snapshot._writer_frontier_choice_snapshot_entry_for_emitted_text(
                choice_snapshot,
                "C",
            )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.INTERNAL_INVARIANT)

    def test_writer_search_snapshot_with_cursor_after_emitted_text_updates_cursor_and_boundary(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        choice_snapshot = (
            writer_snapshot
            ._checked_writer_frontier_choice_snapshot_from_snapshot(
                snapshot,
                prepared=prepared,
                include_counts=False,
            )
        )
        choice = choice_snapshot.choices[0]

        advanced = (
            writer_snapshot
            ._writer_search_snapshot_with_cursor_after_emitted_text(
                snapshot,
                prepared=prepared,
                cursor=choice.successor,
            )
        )

        self.assertEqual(advanced.cursor, choice.successor)
        self.assertEqual(
            advanced.decoder_boundary.consumed_token_count,
            snapshot.decoder_boundary.consumed_token_count + 1,
        )
        self.assertEqual(
            advanced.frame_stack,
            (writer_snapshot.WriterFrontierFrame(choice.successor),),
        )
        self.assertEqual(advanced.runtime_options, snapshot.runtime_options)
        self.assertEqual(advanced.prepared_identity, snapshot.prepared_identity)
        writer_snapshot.validate_writer_search_snapshot(
            advanced,
            prepared=prepared,
        )

    def test_writer_snapshot_advance_outcome_returns_advanced_snapshot(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        choice_snapshot = writer_snapshot._writer_frontier_choice_snapshot_from_snapshot(
            snapshot,
            prepared=prepared,
            include_counts=False,
        )
        choice = choice_snapshot.choices[0]

        outcome = writer_snapshot._writer_snapshot_advance_outcome_by_emitted_text(
            snapshot,
            prepared=prepared,
            emitted_text=choice.emitted_text,
        )

        self.assertIs(
            outcome.kind,
            writer_snapshot._WriterSnapshotAdvanceOutcomeKind.ADVANCED,
        )
        self.assertIsNotNone(outcome.choice)
        self.assertEqual(outcome.choice.emitted_text, choice.emitted_text)
        self.assertIsNotNone(outcome.advanced_snapshot)
        self.assertEqual(outcome.advanced_snapshot.cursor, choice.successor)
        self.assertEqual(
            outcome.advanced_snapshot.decoder_boundary.consumed_token_count,
            snapshot.decoder_boundary.consumed_token_count + 1,
        )

    def test_writer_snapshot_advance_outcome_returns_invalid_text_without_raising(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )

        outcome = writer_snapshot._writer_snapshot_advance_outcome_by_emitted_text(
            snapshot,
            prepared=prepared,
            emitted_text="not-a-frontier-token",
        )

        self.assertIs(
            outcome.kind,
            (
                writer_snapshot
                ._WriterSnapshotAdvanceOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
        )
        self.assertTrue(outcome.invalid_emitted_text)
        self.assertIsNone(outcome.advanced_snapshot)
        self.assertIsNone(outcome.choice)

        with self.assertRaises(SouthStarError) as raised:
            writer_snapshot._advance_writer_search_snapshot_by_emitted_text(
                snapshot,
                prepared=prepared,
                emitted_text="not-a-frontier-token",
            )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.INVALID_FACTS)

    def test_writer_snapshot_advance_outcome_returns_blocked_without_raising(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        blocked_choice_snapshot = self._test_blocked_frontier_choice_snapshot(cursor)

        with patch(
            (
                "grimace._south_star1.writer_snapshot"
                "._writer_frontier_choice_snapshot_from_snapshot"
            ),
            return_value=blocked_choice_snapshot,
        ):
            outcome = writer_snapshot._writer_snapshot_advance_outcome_by_emitted_text(
                snapshot,
                prepared=prepared,
                emitted_text="C",
            )

        self.assertIs(
            outcome.kind,
            writer_snapshot._WriterSnapshotAdvanceOutcomeKind.BLOCKED,
        )
        self.assertTrue(outcome.blocked)
        self.assertEqual(
            outcome.graph_policy_blockers,
            blocked_choice_snapshot.graph_policy_blockers,
        )
        self.assertIsNone(outcome.advanced_snapshot)
        self.assertIsNone(outcome.choice)

    def test_writer_snapshot_advance_sequence_outcome_advances_multiple_tokens(self) -> None:
        prepared = _prepare(chain_facts(("C", "C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        first_choices = writer_snapshot.resume_writer_frontier_choices_from_snapshot(
            snapshot,
            prepared=prepared,
        )
        first_text = first_choices.choices[0].emitted_text
        first_advanced = writer_snapshot._advance_writer_search_snapshot_by_emitted_text(
            snapshot,
            prepared=prepared,
            emitted_text=first_text,
        )
        second_choices = writer_snapshot.resume_writer_frontier_choices_from_snapshot(
            first_advanced,
            prepared=prepared,
        )
        second_text = second_choices.choices[0].emitted_text

        outcome = (
            writer_snapshot
            ._writer_snapshot_advance_sequence_outcome_by_emitted_texts(
                snapshot,
                prepared=prepared,
                emitted_texts=(first_text, second_text),
            )
        )
        repeated = writer_snapshot._advance_writer_search_snapshot_by_emitted_text(
            first_advanced,
            prepared=prepared,
            emitted_text=second_text,
        )

        self.assertIs(
            outcome.kind,
            writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.ADVANCED,
        )
        self.assertEqual(len(outcome.step_outcomes), 2)
        self.assertEqual(
            outcome.consumed_emitted_texts,
            (first_text, second_text),
        )
        self.assertEqual(outcome.remaining_emitted_texts, ())
        self.assertIsNotNone(outcome.advanced_snapshot)
        self.assertEqual(
            outcome.advanced_snapshot.decoder_boundary.consumed_token_count,
            snapshot.decoder_boundary.consumed_token_count + 2,
        )
        self.assertEqual(outcome.advanced_snapshot, repeated)

    def test_writer_snapshot_advance_sequence_outcome_stops_on_invalid_token(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        first_text = (
            writer_snapshot
            ._writer_frontier_choice_snapshot_from_snapshot(
                snapshot,
                prepared=prepared,
                include_counts=False,
            )
            .choices[0]
            .emitted_text
        )
        first_advanced = writer_snapshot._advance_writer_search_snapshot_by_emitted_text(
            snapshot,
            prepared=prepared,
            emitted_text=first_text,
        )

        outcome = (
            writer_snapshot
            ._writer_snapshot_advance_sequence_outcome_by_emitted_texts(
                snapshot,
                prepared=prepared,
                emitted_texts=(first_text, "illegal", "not-attempted"),
            )
        )

        self.assertIs(
            outcome.kind,
            (
                writer_snapshot
                ._WriterSnapshotAdvanceSequenceOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
        )
        self.assertEqual(outcome.consumed_emitted_texts, (first_text,))
        self.assertEqual(
            outcome.remaining_emitted_texts,
            ("illegal", "not-attempted"),
        )
        self.assertEqual(outcome.current_snapshot, first_advanced)

    def test_writer_snapshot_advance_sequence_outcome_stops_on_blocked_outcome(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        first = self._test_writer_snapshot_advance_step(
            snapshot,
            emitted_text=(
                writer_snapshot
                ._writer_frontier_choice_snapshot_from_snapshot(
                    snapshot,
                    prepared=prepared,
                    include_counts=False,
                )
                .choices[0]
                .emitted_text
            ),
            prepared=prepared,
        )
        assert first.advanced_snapshot is not None
        blocked = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceOutcomeKind.BLOCKED,
            source_snapshot=first.advanced_snapshot,
            emitted_text="blocked",
            choice_snapshot=self._test_blocked_frontier_choice_snapshot(
                first.advanced_snapshot.cursor,
            ),
        )

        with patch(
            (
                "grimace._south_star1.writer_snapshot"
                "._writer_snapshot_advance_outcome_by_emitted_text"
            ),
            side_effect=(first, blocked, AssertionError("third step attempted")),
        ):
            outcome = (
                writer_snapshot
                ._writer_snapshot_advance_sequence_outcome_by_emitted_texts(
                    snapshot,
                    prepared=prepared,
                    emitted_texts=(
                        first.emitted_text,
                        "blocked",
                        "not-attempted",
                    ),
                )
            )

        self.assertIs(
            outcome.kind,
            writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.BLOCKED,
        )
        self.assertEqual(len(outcome.step_outcomes), 2)
        self.assertEqual(outcome.consumed_emitted_texts, (first.emitted_text,))
        self.assertEqual(
            outcome.remaining_emitted_texts,
            ("blocked", "not-attempted"),
        )

    def test_advance_writer_search_snapshot_by_emitted_text_uses_uncounted_choice_snapshot(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        choice_snapshot = (
            writer_snapshot
            ._checked_writer_frontier_choice_snapshot_from_snapshot(
                snapshot,
                prepared=prepared,
                include_counts=False,
            )
        )
        choice = choice_snapshot.choices[0]

        with patch(
            (
                "grimace._south_star1.writer_snapshot"
                "._writer_frontier_choice_snapshot_from_snapshot"
            ),
            wraps=writer_snapshot._writer_frontier_choice_snapshot_from_snapshot,
        ) as unchecked_snapshot:
            advanced = writer_snapshot._advance_writer_search_snapshot_by_emitted_text(
                snapshot,
                prepared=prepared,
                emitted_text=choice.emitted_text,
            )

        unchecked_snapshot.assert_called_once_with(
            snapshot,
            prepared=prepared,
            include_counts=False,
            stop_after_first_blocked=True,
        )
        self.assertEqual(advanced.cursor, choice.successor)
        self.assertEqual(advanced.decoder_boundary.consumed_token_count, 1)

    def test_advance_writer_search_snapshot_by_emitted_text_rejects_illegal_text(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )

        with self.assertRaises(SouthStarError) as raised:
            writer_snapshot._advance_writer_search_snapshot_by_emitted_text(
                snapshot,
                prepared=prepared,
                emitted_text="not-a-frontier-token",
            )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.INVALID_FACTS)
        self.assertEqual(snapshot.cursor, cursor)
        self.assertEqual(snapshot.decoder_boundary.consumed_token_count, 0)

    def test_advance_writer_search_snapshot_by_emitted_text_raises_from_blocked_choice_snapshot(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        blocked_choice_snapshot = self._test_blocked_frontier_choice_snapshot(cursor)

        with patch(
            (
                "grimace._south_star1.writer_snapshot"
                "._writer_frontier_choice_snapshot_from_snapshot"
            ),
            return_value=blocked_choice_snapshot,
        ):
            with self.assertRaises(SouthStarError) as raised:
                writer_snapshot._advance_writer_search_snapshot_by_emitted_text(
                    snapshot,
                    prepared=prepared,
                    emitted_text="C",
                )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.UNSUPPORTED_POLICY)

    def test_advance_writer_search_snapshot_by_emitted_text_raises_from_blocked_advance_outcome(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        blocked_choice_snapshot = self._test_blocked_frontier_choice_snapshot(cursor)
        outcome = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceOutcomeKind.BLOCKED,
            source_snapshot=snapshot,
            emitted_text="C",
            choice_snapshot=blocked_choice_snapshot,
        )

        with patch(
            (
                "grimace._south_star1.writer_snapshot"
                "._writer_snapshot_advance_outcome_by_emitted_text"
            ),
            return_value=outcome,
        ):
            with self.assertRaises(SouthStarError) as raised:
                writer_snapshot._advance_writer_search_snapshot_by_emitted_text(
                    snapshot,
                    prepared=prepared,
                    emitted_text="C",
                )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.UNSUPPORTED_POLICY)

    def test_advance_writer_search_snapshot_by_emitted_text_returns_advanced_outcome_snapshot(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        choice_snapshot = writer_snapshot._writer_frontier_choice_snapshot_from_snapshot(
            snapshot,
            prepared=prepared,
            include_counts=False,
        )
        choice = choice_snapshot.choices[0]
        advanced_snapshot = (
            writer_snapshot
            ._writer_search_snapshot_with_cursor_after_emitted_text(
                snapshot,
                prepared=prepared,
                cursor=choice.successor,
            )
        )
        outcome = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceOutcomeKind.ADVANCED,
            source_snapshot=snapshot,
            emitted_text=choice.emitted_text,
            choice_snapshot=choice_snapshot,
            choice=choice,
            advanced_snapshot=advanced_snapshot,
        )

        with patch(
            (
                "grimace._south_star1.writer_snapshot"
                "._writer_snapshot_advance_outcome_by_emitted_text"
            ),
            return_value=outcome,
        ):
            result = writer_snapshot._advance_writer_search_snapshot_by_emitted_text(
                snapshot,
                prepared=prepared,
                emitted_text=choice.emitted_text,
            )

        self.assertIs(result, advanced_snapshot)

    def test_advance_writer_search_snapshot_by_emitted_texts_returns_final_snapshot(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        outcome = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.ADVANCED,
            source_snapshot=snapshot,
            emitted_texts=(),
            step_outcomes=(),
            current_snapshot=snapshot,
        )

        with patch(
            (
                "grimace._south_star1.writer_snapshot"
                "._writer_snapshot_advance_sequence_outcome_by_emitted_texts"
            ),
            return_value=outcome,
        ):
            result = writer_snapshot._advance_writer_search_snapshot_by_emitted_texts(
                snapshot,
                prepared=prepared,
                emitted_texts=(),
            )

        self.assertIs(result, snapshot)

    def test_advance_writer_search_snapshot_by_emitted_texts_raises_from_failed_sequence_outcome(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        choice_snapshot = writer_snapshot._writer_frontier_choice_snapshot_from_snapshot(
            snapshot,
            prepared=prepared,
            include_counts=False,
        )
        invalid_step = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotAdvanceOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_text="bad",
            choice_snapshot=choice_snapshot,
        )
        invalid_outcome = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=(
                writer_snapshot
                ._WriterSnapshotAdvanceSequenceOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_texts=("bad",),
            step_outcomes=(invalid_step,),
            current_snapshot=snapshot,
        )
        blocked_step = writer_snapshot._WriterSnapshotAdvanceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceOutcomeKind.BLOCKED,
            source_snapshot=snapshot,
            emitted_text="blocked",
            choice_snapshot=self._test_blocked_frontier_choice_snapshot(cursor),
        )
        blocked_outcome = writer_snapshot._WriterSnapshotAdvanceSequenceOutcome(
            kind=writer_snapshot._WriterSnapshotAdvanceSequenceOutcomeKind.BLOCKED,
            source_snapshot=snapshot,
            emitted_texts=("blocked",),
            step_outcomes=(blocked_step,),
            current_snapshot=snapshot,
        )

        with patch(
            (
                "grimace._south_star1.writer_snapshot"
                "._writer_snapshot_advance_sequence_outcome_by_emitted_texts"
            ),
            return_value=invalid_outcome,
        ):
            with self.assertRaises(SouthStarError) as invalid_raised:
                writer_snapshot._advance_writer_search_snapshot_by_emitted_texts(
                    snapshot,
                    prepared=prepared,
                    emitted_texts=("bad",),
                )

        self.assertIs(
            invalid_raised.exception.kind,
            SouthStarErrorKind.INVALID_FACTS,
        )

        with patch(
            (
                "grimace._south_star1.writer_snapshot"
                "._writer_snapshot_advance_sequence_outcome_by_emitted_texts"
            ),
            return_value=blocked_outcome,
        ):
            with self.assertRaises(SouthStarError) as blocked_raised:
                writer_snapshot._advance_writer_search_snapshot_by_emitted_texts(
                    snapshot,
                    prepared=prepared,
                    emitted_texts=("blocked",),
                )

        self.assertIs(
            blocked_raised.exception.kind,
            SouthStarErrorKind.UNSUPPORTED_POLICY,
        )

    def test_resume_writer_frontier_choices_after_private_snapshot_advance_matches_choice_successor(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = writer_snapshot.capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        choices_before = writer_snapshot.resume_writer_frontier_choices_from_snapshot(
            snapshot,
            prepared=prepared,
        )
        selected = choices_before.choices[0]

        advanced = writer_snapshot._advance_writer_search_snapshot_by_emitted_text(
            snapshot,
            prepared=prepared,
            emitted_text=selected.emitted_text,
        )
        choices_after = writer_snapshot.resume_writer_frontier_choices_from_snapshot(
            advanced,
            prepared=prepared,
        )
        direct_after = writer_frontier_choices(prepared, selected.successor)

        self.assertEqual(choices_after, direct_after)

    def test_successors_from_choice_snapshot_preserve_snapshot_choice_order(self) -> None:
        parent_key = writer_state_key(_raw_initial_state(AtomId(0)))
        n_successor_key = writer_state_key(_raw_initial_state(AtomId(1)))
        c_successor_key = writer_state_key(_raw_initial_state(AtomId(2)))
        family = writer_transitions._WriterGraphPolicyActionFamily.ACYCLIC_TREE_ENTRY
        n_support = writer_frontier_module._WriterFrontierNextTokenSupport(
            state_key=parent_key,
            parent_weight=3,
            schedule_support=SimpleNamespace(
                emitted_text="N",
                graph_action_surface=object(),
                policy_family=family,
            ),  # type: ignore[arg-type]
            successor_key=n_successor_key,
        )
        c_support = writer_frontier_module._WriterFrontierNextTokenSupport(
            state_key=parent_key,
            parent_weight=2,
            schedule_support=SimpleNamespace(
                emitted_text="C",
                graph_action_surface=object(),
                policy_family=family,
            ),  # type: ignore[arg-type]
            successor_key=c_successor_key,
        )
        n_entry = writer_frontier_module._WriterFrontierNextTokenEntry(
            emitted_text="N",
            supports=(n_support,),
        )
        c_entry = writer_frontier_module._WriterFrontierNextTokenEntry(
            emitted_text="C",
            supports=(c_support,),
        )
        n_successor = WriterFrontierCursor(
            weighted_states=tuple(n_entry.weighted_successors.items())
        )
        c_successor = WriterFrontierCursor(
            weighted_states=tuple(c_entry.weighted_successors.items())
        )
        snapshot = writer_frontier_module._WriterFrontierChoiceSnapshot(
            schedule_outcome=writer_frontier_module._WriterFrontierScheduleOutcome(
                state_outcomes=(),
                terminal_by_key=Counter(),
                grouped_by_text={},
                weighted_by_text={},
            ),
            terminal=None,
            choices=(
                writer_frontier_module._WriterFrontierChoiceSnapshotEntry(
                    next_token_entry=n_entry,
                    successor=n_successor,
                ),
                writer_frontier_module._WriterFrontierChoiceSnapshotEntry(
                    next_token_entry=c_entry,
                    successor=c_successor,
                ),
            ),
        )

        successors = writer_frontier_module._successors_from_choice_snapshot(
            snapshot
        )

        self.assertEqual(
            successors,
            (
                ("N", n_successor),
                ("C", c_successor),
            ),
        )

    def test_writer_frontier_raw_successors_for_streaming_uses_uncounted_choice_snapshot(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        successor_key = cursor.weighted_states[0][0]
        support = writer_frontier_module._WriterFrontierNextTokenSupport(
            state_key=successor_key,
            parent_weight=1,
            schedule_support=SimpleNamespace(
                emitted_text="C",
                graph_action_surface=object(),
                policy_family=(
                    writer_transitions
                    ._WriterGraphPolicyActionFamily
                    .ACYCLIC_TREE_ENTRY
                ),
            ),  # type: ignore[arg-type]
            successor_key=successor_key,
        )
        entry = writer_frontier_module._WriterFrontierNextTokenEntry(
            emitted_text="C",
            supports=(support,),
        )
        choice = writer_frontier_module._WriterFrontierChoiceSnapshotEntry(
            next_token_entry=entry,
            successor=WriterFrontierCursor(weighted_states=((successor_key, 1),)),
        )
        snapshot = writer_frontier_module._WriterFrontierChoiceSnapshot(
            schedule_outcome=writer_frontier_module._WriterFrontierScheduleOutcome(
                state_outcomes=(),
                terminal_by_key=Counter(),
                grouped_by_text={"C": {successor_key}},
                weighted_by_text={"C": Counter({successor_key: 1})},
                next_token_frontier=(entry,),
            ),
            terminal=None,
            choices=(choice,),
        )

        with patch(
            (
                "grimace._south_star1.writer_frontier"
                "._checked_writer_frontier_choice_snapshot"
            ),
            return_value=snapshot,
        ) as checked_snapshot:
            successors = (
                writer_frontier_module
                ._writer_frontier_raw_successors_for_streaming(
                    prepared,
                    cursor,
                )
            )

        self.assertEqual(successors, ((choice.emitted_text, choice.successor),))
        checked_snapshot.assert_called_once_with(
            prepared,
            cursor,
            include_counts=False,
        )

    def test_writer_frontier_choices_routes_through_checked_choice_snapshot(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        successor_key = cursor.weighted_states[0][0]
        support = writer_frontier_module._WriterFrontierNextTokenSupport(
            state_key=successor_key,
            parent_weight=1,
            schedule_support=SimpleNamespace(
                emitted_text="C",
                graph_action_surface=object(),
                policy_family=(
                    writer_transitions
                    ._WriterGraphPolicyActionFamily
                    .ACYCLIC_TREE_ENTRY
                ),
            ),  # type: ignore[arg-type]
            successor_key=successor_key,
        )
        entry = writer_frontier_module._WriterFrontierNextTokenEntry(
            emitted_text="C",
            supports=(support,),
        )
        choice = writer_frontier_module._WriterFrontierChoiceSnapshotEntry(
            next_token_entry=entry,
            successor=WriterFrontierCursor(weighted_states=((successor_key, 1),)),
            support_count=1,
            completion_count=2,
        )
        snapshot = writer_frontier_module._WriterFrontierChoiceSnapshot(
            schedule_outcome=writer_frontier_module._WriterFrontierScheduleOutcome(
                state_outcomes=(),
                terminal_by_key=Counter(),
                grouped_by_text={"C": {successor_key}},
                weighted_by_text={"C": Counter({successor_key: 1})},
                next_token_frontier=(entry,),
            ),
            terminal=None,
            choices=(choice,),
        )

        with patch(
            (
                "grimace._south_star1.writer_frontier"
                "._checked_writer_frontier_choice_snapshot"
            ),
            return_value=snapshot,
        ) as checked_snapshot:
            choices = writer_frontier_choices(prepared, cursor)

        self.assertEqual(choices, snapshot.public_choices)
        checked_snapshot.assert_called_once_with(prepared, cursor)

    def test_writer_frontier_choices_uses_counted_choice_snapshot(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())

        with patch(
            (
                "grimace._south_star1.writer_frontier"
                "._checked_writer_frontier_choice_snapshot"
            ),
            wraps=writer_frontier_module._checked_writer_frontier_choice_snapshot,
        ) as checked_snapshot:
            choices = writer_frontier_choices(prepared, cursor)

        self.assertEqual(tuple(choice.emitted_text for choice in choices.choices), ("C",))
        self.assertGreater(checked_snapshot.call_count, 0)
        self.assertEqual(checked_snapshot.call_args_list[0].args, (prepared, cursor))
        self.assertNotIn("include_counts", checked_snapshot.call_args_list[0].kwargs)

    def test_writer_frontier_choices_use_next_token_entries_not_grouped_transitions(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())

        with patch(
            "grimace._south_star1.writer_frontier._group_writer_frontier_transitions",
            side_effect=AssertionError("choices used grouped transitions"),
        ), patch(
            "grimace._south_star1.writer_frontier._count_writer_frontier_support",
            return_value=1,
        ), patch(
            (
                "grimace._south_star1.writer_frontier"
                "._count_weighted_successor_completions"
            ),
            return_value=2,
        ):
            choices = writer_frontier_choices(prepared, cursor)

        self.assertEqual(tuple(choice.emitted_text for choice in choices.choices), ("C",))

    def test_count_writer_frontier_support_uses_uncounted_choice_snapshot(self) -> None:
        prepared = _prepare(cco_facts())
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())

        with patch(
            (
                "grimace._south_star1.writer_frontier"
                "._checked_writer_frontier_choice_snapshot"
            ),
            wraps=writer_frontier_module._checked_writer_frontier_choice_snapshot,
        ) as checked_snapshot:
            support_count = count_writer_frontier_support(
                prepared,
                cursor.support_state,
            )

        self.assertEqual(support_count, 4)
        self.assertGreater(checked_snapshot.call_count, 0)
        self.assertTrue(
            all(
                call.kwargs.get("include_counts") is False
                for call in checked_snapshot.call_args_list
            )
        )

    def test_count_writer_choice_snapshot_completions_counts_terminal_and_weighted_successors(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        parent_key = writer_state_key(_raw_initial_state(AtomId(0)))
        successor_a = writer_state_key(_raw_initial_state(AtomId(1)))
        successor_b = writer_state_key(_raw_initial_state(AtomId(2)))
        family = writer_transitions._WriterGraphPolicyActionFamily.ACYCLIC_TREE_ENTRY
        support_a = writer_frontier_module._WriterFrontierNextTokenSupport(
            state_key=parent_key,
            parent_weight=2,
            schedule_support=SimpleNamespace(
                emitted_text="C",
                graph_action_surface=object(),
                policy_family=family,
            ),  # type: ignore[arg-type]
            successor_key=successor_a,
        )
        support_b = writer_frontier_module._WriterFrontierNextTokenSupport(
            state_key=parent_key,
            parent_weight=5,
            schedule_support=SimpleNamespace(
                emitted_text="C",
                graph_action_surface=object(),
                policy_family=family,
            ),  # type: ignore[arg-type]
            successor_key=successor_b,
        )
        entry = writer_frontier_module._WriterFrontierNextTokenEntry(
            emitted_text="C",
            supports=(support_a, support_b),
        )
        choice = writer_frontier_module._WriterFrontierChoiceSnapshotEntry(
            next_token_entry=entry,
            successor=WriterFrontierCursor(
                weighted_states=tuple(entry.weighted_successors.items())
            ),
        )
        terminal = writer_frontier_module.WriterFrontierTerminal(
            support_count=1,
            completion_count=3,
            multiplicity=3,
            finalized_cursor=WriterFrontierCursor(weighted_states=()),
        )
        snapshot = writer_frontier_module._WriterFrontierChoiceSnapshot(
            schedule_outcome=writer_frontier_module._WriterFrontierScheduleOutcome(
                state_outcomes=(),
                terminal_by_key=Counter(),
                grouped_by_text={},
                weighted_by_text={},
            ),
            terminal=terminal,
            choices=(choice,),
        )

        def count_state(
            _prepared,
            key: WriterStateKey,
            _memo,
        ) -> int:
            if key == successor_a:
                return 7
            if key == successor_b:
                return 11
            raise AssertionError(f"unexpected successor key: {key!r}")

        with patch(
            "grimace._south_star1.writer_frontier._count_writer_state_completions",
            side_effect=count_state,
        ):
            total = (
                writer_frontier_module
                ._count_writer_choice_snapshot_completions(
                    prepared,
                    snapshot,
                    {},
                )
            )

        self.assertEqual(total, 3 + 2 * 7 + 5 * 11)

    def test_count_writer_cursor_completions_uses_uncounted_choice_snapshot(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        snapshot = writer_frontier_module._WriterFrontierChoiceSnapshot(
            schedule_outcome=writer_frontier_module._WriterFrontierScheduleOutcome(
                state_outcomes=(),
                terminal_by_key=Counter(),
                grouped_by_text={},
                weighted_by_text={},
            ),
            terminal=None,
            choices=(),
        )

        with patch(
            (
                "grimace._south_star1.writer_frontier"
                "._checked_writer_frontier_choice_snapshot"
            ),
            return_value=snapshot,
        ) as checked_snapshot, patch(
            (
                "grimace._south_star1.writer_frontier"
                "._count_writer_choice_snapshot_completions"
            ),
            return_value=123,
        ) as count_snapshot:
            count = count_writer_cursor_completions(prepared, cursor)

        self.assertEqual(count, 123)
        checked_snapshot.assert_called_once_with(
            prepared,
            cursor,
            include_counts=False,
        )
        count_snapshot.assert_called_once()
        self.assertIs(count_snapshot.call_args.args[1], snapshot)

    def test_count_writer_state_completions_uses_uncounted_choice_snapshot(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        key = writer_state_key(_raw_initial_state(AtomId(0)))
        snapshot = writer_frontier_module._WriterFrontierChoiceSnapshot(
            schedule_outcome=writer_frontier_module._WriterFrontierScheduleOutcome(
                state_outcomes=(),
                terminal_by_key=Counter(),
                grouped_by_text={},
                weighted_by_text={},
            ),
            terminal=None,
            choices=(),
        )
        memo: dict[WriterStateKey, int] = {}

        with patch(
            (
                "grimace._south_star1.writer_frontier"
                "._checked_writer_frontier_choice_snapshot"
            ),
            return_value=snapshot,
        ) as checked_snapshot, patch(
            (
                "grimace._south_star1.writer_frontier"
                "._count_writer_choice_snapshot_completions"
            ),
            return_value=17,
        ):
            count = writer_frontier_module._count_writer_state_completions(
                prepared,
                key,
                memo,
            )

        self.assertEqual(count, 17)
        checked_snapshot.assert_called_once_with(
            prepared,
            WriterFrontierCursor(weighted_states=((key, 1),)),
            include_counts=False,
        )
        self.assertEqual(memo[key], 17)

    def test_count_writer_state_completions_uses_memo_before_snapshot(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        key = writer_state_key(_raw_initial_state(AtomId(0)))

        with patch(
            (
                "grimace._south_star1.writer_frontier"
                "._checked_writer_frontier_choice_snapshot"
            ),
            side_effect=AssertionError("memoized completion used snapshot"),
        ):
            count = writer_frontier_module._count_writer_state_completions(
                prepared,
                key,
                {key: 42},
            )

        self.assertEqual(count, 42)

    def test_completion_count_uses_uncounted_choice_snapshots(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())

        with patch(
            (
                "grimace._south_star1.writer_frontier"
                "._checked_writer_frontier_choice_snapshot"
            ),
            wraps=writer_frontier_module._checked_writer_frontier_choice_snapshot,
        ) as checked_snapshot:
            count = count_writer_cursor_completions(prepared, cursor)

        self.assertEqual(count, 2)
        self.assertGreater(checked_snapshot.call_count, 0)
        self.assertTrue(
            all(
                call.kwargs.get("include_counts") is False
                for call in checked_snapshot.call_args_list
            )
        )

    def test_count_writer_frontier_support_uses_next_token_entries_not_grouped_transitions(self) -> None:
        prepared = _prepare(cco_facts())
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())

        with patch(
            "grimace._south_star1.writer_frontier._group_writer_frontier_transitions",
            side_effect=AssertionError("support count used grouped transitions"),
        ):
            support_count = count_writer_frontier_support(
                prepared,
                cursor.support_state,
            )

        self.assertEqual(support_count, 4)

    def test_iter_writer_frontier_support_uses_uncounted_choice_snapshot(self) -> None:
        prepared = _prepare(cco_facts())
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())

        with patch(
            (
                "grimace._south_star1.writer_frontier"
                "._checked_writer_frontier_choice_snapshot"
            ),
            wraps=writer_frontier_module._checked_writer_frontier_choice_snapshot,
        ) as checked_snapshot:
            strings = tuple(iter_writer_frontier_support(prepared, cursor))

        self.assertEqual(strings, ("C(C)O", "C(O)C", "CCO", "OCC"))
        self.assertGreater(checked_snapshot.call_count, 0)
        self.assertTrue(
            all(
                call.kwargs.get("include_counts") is False
                for call in checked_snapshot.call_args_list
            )
        )

    def test_iter_writer_frontier_support_uses_next_token_entries_not_grouped_transitions(self) -> None:
        prepared = _prepare(cco_facts())
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())

        with patch(
            "grimace._south_star1.writer_frontier._group_writer_frontier_transitions",
            side_effect=AssertionError("streaming used grouped transitions"),
        ):
            strings = tuple(iter_writer_frontier_support(prepared, cursor))

        self.assertEqual(strings, ("C(C)O", "C(O)C", "CCO", "OCC"))

    def test_checked_writer_frontier_schedule_outcome_raises_from_blocked_outcome(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        active_outcome = self._blocked_child_active_emitted_outcome(AtomId(0))
        blocked_top_level_outcome = (
            writer_transitions._WriterTopLevelScheduleOutcome(
                kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.BLOCKED,
                active_emitted_outcome=active_outcome,
            )
        )
        state_outcome = writer_frontier_module._WriterFrontierStateScheduleOutcome(
            state_key=cursor.weighted_states[0][0],
            parent_weight=cursor.weighted_states[0][1],
            finalized_state_key=None,
            schedule_outcome=blocked_top_level_outcome,
        )
        outcome = writer_frontier_module._WriterFrontierScheduleOutcome(
            state_outcomes=(state_outcome,),
            terminal_by_key=Counter(),
            grouped_by_text={},
            weighted_by_text={},
        )

        with patch(
            "grimace._south_star1.writer_frontier._writer_frontier_schedule_outcome",
            return_value=outcome,
        ):
            with self.assertRaises(SouthStarError) as raised:
                (
                    writer_frontier_module
                    ._checked_writer_frontier_schedule_outcome(
                        prepared,
                        cursor,
                    )
                )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.UNSUPPORTED_POLICY)

    def test_writer_frontier_counts_duplicate_token_paths_to_same_state(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            chain_facts(("C",)),
            writer_surface=SouthStarWriterSurface(),
            policy=duplicate_single_atom_policy(),
        )
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())

        choices = writer_frontier_choices(prepared, cursor)

        self.assertEqual(tuple(choice.emitted_text for choice in choices.choices), ("C",))
        choice = choices.choices[0]
        self.assertEqual(choice.immediate_multiplicity, 2)
        self.assertEqual(len(choice.successor.support_state.states), 1)
        self.assertEqual(choice.successor.weighted_states[0][1], 2)
        self.assertEqual(choice.support_count, 1)
        self.assertEqual(choice.completion_count, 2)

    def test_writer_frontier_terminal_counts_weighted_cursor(self) -> None:
        prepared = _prepare(chain_facts(("C",)))
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        after_atom = writer_frontier_choices(prepared, cursor).choices[0].successor
        terminal_key = after_atom.weighted_states[0][0]
        weighted_terminal = WriterFrontierCursor(
            weighted_states=((terminal_key, 3),)
        )

        choices = writer_frontier_choices(prepared, weighted_terminal)

        self.assertIsNotNone(choices.terminal)
        assert choices.terminal is not None
        self.assertEqual(choices.terminal.support_count, 1)
        self.assertEqual(choices.terminal.completion_count, 3)
        self.assertEqual(choices.terminal.multiplicity, 3)
        self.assertEqual(
            sum(weight for _, weight in choices.terminal.finalized_cursor.weighted_states),
            3,
        )
        self.assertEqual(choices.choices, ())

    def test_writer_support_image_keeps_witness_count_separate(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))

        support = enumerate_prepared_stereo_support(
            prepared=prepared,
            runtime_options=_writer_options(),
        )

        self.assertEqual(support.strings, ("CC",))
        self.assertEqual(support.distinct_count, 1)
        self.assertEqual(support.witness_count, 2)

    def test_writer_witness_completions_can_exceed_support_count(self) -> None:
        prepared = _prepare(chain_facts(("C", "C", "C")))
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())

        self.assertEqual(count_writer_frontier_support(prepared, cursor.support_state), 2)
        self.assertEqual(count_writer_cursor_completions(prepared, cursor), 4)

    def test_writer_support_count_does_not_call_streaming_support(self) -> None:
        prepared = _prepare(cco_facts())
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())

        with patch(
            "grimace._south_star1.writer_frontier.iter_writer_frontier_support",
            side_effect=AssertionError("count-only path streamed support strings"),
        ):
            self.assertEqual(count_writer_frontier_support(prepared, cursor.support_state), 4)

    def test_streaming_support_does_not_compute_counted_choices(self) -> None:
        prepared = _prepare(cco_facts())
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())

        with patch(
            "grimace._south_star1.writer_frontier.writer_frontier_choices",
            side_effect=AssertionError("streaming used counted choices"),
        ), patch(
            "grimace._south_star1.writer_frontier.count_writer_frontier_support",
            side_effect=AssertionError("streaming computed support count"),
        ), patch(
            "grimace._south_star1.writer_frontier.count_writer_cursor_completions",
            side_effect=AssertionError("streaming computed completion count"),
        ), patch(
            "grimace._south_star1.writer_frontier._count_writer_frontier_support",
            side_effect=AssertionError("streaming computed support count"),
        ), patch(
            (
                "grimace._south_star1.writer_frontier"
                "._count_weighted_successor_completions"
            ),
            side_effect=AssertionError("streaming computed completion count"),
        ):
            self.assertEqual(
                tuple(iter_writer_frontier_support(prepared, cursor)),
                ("C(C)O", "C(O)C", "CCO", "OCC"),
            )

    def test_unique_child_is_inline_for_rooted_chain(self) -> None:
        prepared = _prepare(chain_facts(("C", "C", "C")))

        support = enumerate_prepared_stereo_support(
            prepared=prepared,
            runtime_options=_writer_options(rooted_at_atom=0),
        )

        self.assertEqual(support.strings, ("CCC",))
        self.assertNotIn("(", support.strings[0])

    def test_true_side_branches_remain_expressible(self) -> None:
        prepared = _prepare(cco_facts())

        support = enumerate_prepared_stereo_support(
            prepared=prepared,
            runtime_options=_writer_options(rooted_at_atom=1),
        )

        self.assertEqual(support.strings, ("C(C)O", "C(O)C"))

    def test_double_bond_child_entry_is_token_granular(self) -> None:
        prepared = _prepare(two_atom_facts("C", "O", BondOrder.DOUBLE))
        cursor = initial_writer_frontier_cursor(
            prepared,
            _writer_options(rooted_at_atom=0),
        )

        first = writer_frontier_choices(prepared, cursor).choices[0]
        second = writer_frontier_choices(prepared, first.successor).choices[0]
        third = writer_frontier_choices(prepared, second.successor).choices[0]
        support = enumerate_prepared_stereo_support(
            prepared=prepared,
            runtime_options=_writer_options(rooted_at_atom=0),
        )

        self.assertEqual(first.emitted_text, "C")
        self.assertEqual(second.emitted_text, "=")
        self.assertEqual(third.emitted_text, "O")
        self.assertEqual(support.strings, ("C=O",))

    def test_triple_bond_child_entry_is_token_granular(self) -> None:
        prepared = _prepare(two_atom_facts("C", "C", BondOrder.TRIPLE))
        cursor = initial_writer_frontier_cursor(
            prepared,
            _writer_options(rooted_at_atom=0),
        )

        first = writer_frontier_choices(prepared, cursor).choices[0]
        second = writer_frontier_choices(prepared, first.successor).choices[0]
        third = writer_frontier_choices(prepared, second.successor).choices[0]

        self.assertEqual(first.emitted_text, "C")
        self.assertEqual(second.emitted_text, "#")
        self.assertEqual(third.emitted_text, "C")

    def test_writer_shaped_disconnected_components_emit_dot(self) -> None:
        prepared = _prepare(disconnected_co_facts())

        support = enumerate_prepared_stereo_support(
            prepared=prepared,
            runtime_options=_writer_options(),
        )

        self.assertEqual(support.strings, ("C.O",))

    def test_component_boundary_emits_for_graph_complete_component(self) -> None:
        prepared = _prepare(cyclopropane_plus_singleton_facts())
        state = _with_next_component_root(_cyclopropane_terminal_closed_closure_state())
        context = writer_transitions.build_writer_transition_expansion_context(
            prepared,
            state,
        )

        transitions = writer_transitions._finish_active_transitions(
            prepared,
            state,
            context,
        )

        self.assertEqual(
            tuple(transition.kind for transition in transitions),
            (writer_transitions.WriterTransitionKind.DOT,),
        )
        self.assertEqual(transitions[0].emitted_text, ".")

    def test_component_boundary_rejects_open_closure_endpoint(self) -> None:
        prepared = _prepare(cyclopropane_plus_singleton_facts())
        state = _with_next_component_root(_cyclopropane_terminal_open_closure_state())
        context = writer_transitions.build_writer_transition_expansion_context(
            prepared,
            state,
        )

        transitions = writer_transitions._finish_active_transitions(
            prepared,
            state,
            context,
        )

        self.assertEqual(transitions, ())

    def test_component_boundary_rejects_closure_candidate(self) -> None:
        prepared = _prepare(cyclopropane_plus_singleton_facts())
        state = _with_next_component_root(
            replace(
                _cyclopropane_terminal_closed_closure_state(),
                ring_state=WriterRingState(),
            )
        )
        context = writer_transitions.build_writer_transition_expansion_context(
            prepared,
            state,
        )

        transitions = writer_transitions._finish_active_transitions(
            prepared,
            state,
            context,
        )

        self.assertEqual(transitions, ())

    def test_writer_cursor_after_cc_exposes_weighted_terminal(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        first = writer_frontier_choices(prepared, cursor).choices[0]
        second = writer_frontier_choices(prepared, first.successor).choices[0]

        choices = writer_frontier_choices(prepared, second.successor)

        self.assertIsNotNone(choices.terminal)
        assert choices.terminal is not None
        self.assertEqual(choices.terminal.support_count, 1)
        self.assertEqual(choices.terminal.completion_count, 2)
        self.assertEqual(choices.terminal.multiplicity, 2)
        self.assertEqual(
            sum(weight for _, weight in choices.terminal.finalized_cursor.weighted_states),
            2,
        )
        self.assertEqual(choices.choices, ())

    def test_writer_root_restricts_initial_frontier_without_plan_route(self) -> None:
        prepared = _prepare(cco_facts())

        with _forbidden_exhaustive_routes():
            support = enumerate_prepared_stereo_support(
                prepared=prepared,
                runtime_options=_writer_options(rooted_at_atom=2),
            )

        self.assertEqual(support.strings, ("OCC",))

    def test_writer_shaped_cyclic_fails_closed_before_forbidden_routes(self) -> None:
        prepared = _prepare(cyclopropane_facts())

        with _forbidden_exhaustive_routes():
            with self.assertRaises(SouthStarError) as caught:
                enumerate_prepared_stereo_support(
                    prepared=prepared,
                    runtime_options=_writer_options(),
                )

        self.assertIs(caught.exception.kind, SouthStarErrorKind.UNSUPPORTED_POLICY)

    def test_public_initial_frontier_still_rejects_cyclic_prepared(self) -> None:
        prepared = _prepare(cyclopropane_facts())

        with self.assertRaises(SouthStarError) as caught:
            initial_writer_frontier_cursor(prepared, _writer_options(rooted_at_atom=0))

        self.assertIs(caught.exception.kind, SouthStarErrorKind.UNSUPPORTED_POLICY)

    def test_internal_transition_frontier_accepts_cyclic_prepared(self) -> None:
        prepared = _prepare(cyclopropane_facts())

        cursor = initial_writer_transition_frontier_cursor(
            prepared,
            _writer_options(rooted_at_atom=0),
        )

        self.assertEqual(len(cursor.weighted_states), 1)

    def test_internal_transition_frontier_rejects_malformed_components(self) -> None:
        prepared = _prepare(cycle_plus_isolate_component_facts())

        with self.assertRaises(SouthStarError) as caught:
            initial_writer_transition_frontier_cursor(
                prepared,
                _writer_options(rooted_at_atom=0),
            )

        self.assertIs(caught.exception.kind, SouthStarErrorKind.UNSUPPORTED_POLICY)

    def test_internal_transition_frontier_rejects_unsupported_stereo_surface(self) -> None:
        prepared = _prepare(unsupported_directional_implicit_h_facts())

        with self.assertRaises(SouthStarError) as caught:
            initial_writer_transition_frontier_cursor(
                prepared,
                _writer_options(rooted_at_atom=0),
            )

        self.assertIs(caught.exception.kind, SouthStarErrorKind.UNSUPPORTED_STEREO)

    def test_raw_legal_transitions_reject_same_unsupported_stereo_surface(self) -> None:
        prepared = _prepare(unsupported_directional_implicit_h_facts())

        with self.assertRaises(SouthStarError) as caught:
            writer_transitions.legal_writer_transitions(
                prepared,
                _raw_initial_state(AtomId(0)),
            )

        self.assertIs(caught.exception.kind, SouthStarErrorKind.UNSUPPORTED_STEREO)

    def test_writer_shaped_cycle_plus_isolate_component_fails_closed(self) -> None:
        prepared = _prepare(cycle_plus_isolate_component_facts())

        with _forbidden_exhaustive_routes():
            with self.assertRaises(SouthStarError) as caught:
                enumerate_prepared_stereo_support(
                    prepared=prepared,
                    runtime_options=_writer_options(),
                )

        self.assertIs(caught.exception.kind, SouthStarErrorKind.UNSUPPORTED_POLICY)

    def test_raw_legal_transitions_allow_cyclic_root_emission(self) -> None:
        prepared = _prepare(cyclopropane_facts())

        transitions = writer_transitions.legal_writer_transitions(
            prepared,
            _raw_initial_state(AtomId(0)),
        )

        self.assertEqual(tuple(transition.emitted_text for transition in transitions), ("C",))
        self.assertEqual(
            tuple(transition.kind for transition in transitions),
            (writer_transitions.WriterTransitionKind.ATOM,),
        )

    def test_raw_legal_transitions_reject_missing_active_frame(self) -> None:
        prepared = _prepare(chain_facts(("C",)))

        with self.assertRaises(SouthStarError) as caught:
            writer_transitions.legal_writer_transitions(
                prepared,
                replace(_raw_initial_state(AtomId(0)), active=None),
            )

        self.assertIs(caught.exception.kind, SouthStarErrorKind.INTERNAL_INVARIANT)

    def test_raw_initial_state_still_emits_atom_transition(self) -> None:
        prepared = _prepare(chain_facts(("C",)))

        transitions = writer_transitions.legal_writer_transitions(
            prepared,
            _raw_initial_state(AtomId(0)),
        )

        self.assertEqual(tuple(transition.emitted_text for transition in transitions), ("C",))

    def test_raw_closure_endpoint_transition_opens_ring_label(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        root = writer_transitions.legal_writer_transitions(
            prepared,
            _raw_initial_state(AtomId(0)),
        )[0].successor

        transitions = writer_transitions.legal_writer_transitions(prepared, root)

        self.assertTrue(transitions)
        self.assertEqual(
            {transition.kind for transition in transitions},
            {writer_transitions.WriterTransitionKind.OPEN_CLOSURE_ENDPOINT},
        )
        self.assertEqual({transition.emitted_text for transition in transitions}, {"1"})
        opened = transitions[0].successor
        self.assertEqual(len(opened.ring_state.open_endpoints), 1)
        endpoint = opened.ring_state.open_endpoints[0]
        self.assertEqual(endpoint.label, WriterClosureLabel(value=1, text="1"))
        self.assertEqual(endpoint.first_endpoint_text, "1")
        self.assertEqual(endpoint.first_endpoint_bond_text, "")
        self.assertTrue(
            any(
                factor.kind == "ring_pair" and not factor.closed
                for factor in opened.stereo_state.delayed_factors
            )
        )

    def test_raw_closure_endpoint_transition_pairs_ring_label(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        root = writer_transitions.legal_writer_transitions(
            prepared,
            _raw_initial_state(AtomId(0)),
        )[0].successor
        opened = next(
            transition.successor
            for transition in writer_transitions.legal_writer_transitions(prepared, root)
            if transition.successor.ring_state.open_endpoints[0].second_atom == AtomId(2)
        )
        after_first_child = writer_transitions.legal_writer_transitions(
            prepared,
            opened,
        )[0].successor
        at_partner = writer_transitions.legal_writer_transitions(
            prepared,
            after_first_child,
        )[0].successor

        transitions = writer_transitions.legal_writer_transitions(prepared, at_partner)

        self.assertEqual(
            tuple(transition.kind for transition in transitions),
            (writer_transitions.WriterTransitionKind.PAIR_CLOSURE_ENDPOINT,),
        )
        self.assertEqual(tuple(transition.emitted_text for transition in transitions), ("1",))
        closed = transitions[0].successor
        self.assertEqual(closed.ring_state.open_endpoints, ())
        self.assertEqual(len(closed.ring_state.closed_closures), 1)
        self.assertEqual(
            closed.ring_state.closed_closures[0].label,
            WriterClosureLabel(value=1, text="1"),
        )
        self.assertEqual(
            closed.ring_state.label_state.reusable,
            (WriterClosureLabel(value=1, text="1"),),
        )
        self.assertTrue(
            any(
                factor.kind == "ring_pair" and factor.closed
                for factor in closed.stereo_state.delayed_factors
            )
        )

    def test_internal_transition_frontier_steps_cyclic_closure_lifecycle(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        cursor = initial_writer_transition_frontier_cursor(
            prepared,
            _writer_options(rooted_at_atom=0),
        )

        initial = writer_frontier_choices(prepared, cursor)
        self.assertEqual(tuple(choice.emitted_text for choice in initial.choices), ("C",))
        after_root = initial.choices[0].successor

        root_choices = writer_frontier_choices(prepared, after_root)
        self.assertEqual(
            tuple(choice.emitted_text for choice in root_choices.choices),
            ("1",),
        )
        opened = root_choices.choices[0].successor
        self.assertTrue(
            all(
                key.ring_state.open_endpoints
                for key, _ in opened.weighted_states
            )
        )

        after_first_child = _only_choice(prepared, opened, "C").successor
        after_second_child = _only_choice(prepared, after_first_child, "C").successor
        pair_choice = _only_choice(prepared, after_second_child, "1")
        closed = pair_choice.successor

        self.assertTrue(
            all(
                not key.ring_state.open_endpoints
                and len(key.ring_state.closed_closures) == 1
                for key, _ in closed.weighted_states
            )
        )

    def test_internal_cyclic_frontier_counts_and_streams_finitely(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        cursor = initial_writer_transition_frontier_cursor(
            prepared,
            _writer_options(rooted_at_atom=0),
        )

        support_count = count_writer_frontier_support(prepared, cursor.support_state)
        completion_count = count_writer_cursor_completions(prepared, cursor)
        strings = tuple(iter_writer_frontier_support(prepared, cursor))

        self.assertEqual(support_count, len(set(strings)))
        self.assertEqual(len(strings), len(set(strings)))
        self.assertGreater(support_count, 0)
        self.assertGreaterEqual(completion_count, support_count)
        self.assertTrue(all("1" in string for string in strings))

    def test_internal_cyclic_frontier_terminal_paths_close_closures(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        cursor = initial_writer_transition_frontier_cursor(
            prepared,
            _writer_options(rooted_at_atom=0),
        )

        terminal_keys = _terminal_keys(prepared, cursor)

        self.assertTrue(terminal_keys)
        self.assertTrue(
            all(
                not key.ring_state.open_endpoints
                and key.ring_state.closed_closures
                for key in terminal_keys
            )
        )

    def test_raw_closure_label_allocator_uses_least_free_not_reusable_first(self) -> None:
        prepared = _prepare_with_policy(
            cyclopropane_facts(),
            least_free_ring_labels=True,
        )
        reusable = WriterClosureLabel(value=2, text="2")

        labels = writer_transitions._available_closure_labels_for_open(
            prepared,
            WriterRingState(
                label_state=WriterRingLabelState(reusable=(reusable,)),
            ),
        )

        self.assertEqual(labels, (WriterClosureLabel(value=1, text="1"),))

    def test_raw_closure_label_allocator_uses_reusable_when_smaller_label_is_active(self) -> None:
        prepared = _prepare_with_policy(
            cyclopropane_facts(),
            least_free_ring_labels=True,
        )

        labels = writer_transitions._available_closure_labels_for_open(
            prepared,
            WriterRingState(
                label_state=WriterRingLabelState(
                    allocated=(WriterClosureLabel(value=1, text="1"),),
                ),
            ),
        )

        self.assertEqual(labels, (WriterClosureLabel(value=2, text="2"),))

    def test_raw_closure_label_allocator_least_free_uses_label_value_not_policy_order(self) -> None:
        prepared = _prepare_with_policy(
            cyclopropane_facts(),
            least_free_ring_labels=True,
            ring_labels=(RingLabel(2), RingLabel(1)),
        )

        labels = writer_transitions._available_closure_labels_for_open(
            prepared,
            WriterRingState(),
        )

        self.assertEqual(labels, (WriterClosureLabel(value=1, text="1"),))

    def test_raw_closure_label_allocator_enumerates_labels_without_least_free(self) -> None:
        prepared = _prepare_with_policy(
            cyclopropane_facts(),
            least_free_ring_labels=False,
        )

        labels = writer_transitions._available_closure_labels_for_open(
            prepared,
            WriterRingState(),
        )

        self.assertEqual(
            labels,
            (
                WriterClosureLabel(value=1, text="1"),
                WriterClosureLabel(value=2, text="2"),
            ),
        )

    def test_raw_closure_label_allocator_nonleast_free_preserves_policy_order(self) -> None:
        prepared = _prepare_with_policy(
            cyclopropane_facts(),
            least_free_ring_labels=False,
            ring_labels=(RingLabel(2), RingLabel(1)),
        )

        labels = writer_transitions._available_closure_labels_for_open(
            prepared,
            WriterRingState(),
        )

        self.assertEqual(
            labels,
            (
                WriterClosureLabel(value=2, text="2"),
                WriterClosureLabel(value=1, text="1"),
            ),
        )

    def test_raw_closure_label_allocator_enumerates_all_free_labels_without_least_free(self) -> None:
        prepared = _prepare_with_policy(
            cyclopropane_facts(),
            least_free_ring_labels=False,
        )

        labels = writer_transitions._available_closure_labels_for_open(
            prepared,
            WriterRingState(
                label_state=WriterRingLabelState(
                    reusable=(WriterClosureLabel(value=2, text="2"),),
                ),
            ),
        )

        self.assertEqual(
            labels,
            (
                WriterClosureLabel(value=1, text="1"),
                WriterClosureLabel(value=2, text="2"),
            ),
        )

    def test_raw_closure_label_allocator_excludes_active_labels_without_least_free(self) -> None:
        prepared = _prepare_with_policy(
            cyclopropane_facts(),
            least_free_ring_labels=False,
        )

        labels = writer_transitions._available_closure_labels_for_open(
            prepared,
            WriterRingState(
                label_state=WriterRingLabelState(
                    allocated=(WriterClosureLabel(value=1, text="1"),),
                    reusable=(WriterClosureLabel(value=2, text="2"),),
                ),
            ),
        )

        self.assertEqual(labels, (WriterClosureLabel(value=2, text="2"),))

    def test_raw_closure_open_transitions_enumerate_labels_without_least_free(self) -> None:
        prepared = _prepare_with_policy(
            cyclopropane_facts(),
            least_free_ring_labels=False,
        )
        root = writer_transitions.legal_writer_transitions(
            prepared,
            _raw_initial_state(AtomId(0)),
        )[0].successor

        transitions = writer_transitions.legal_writer_transitions(prepared, root)

        self.assertEqual(
            {transition.emitted_text for transition in transitions},
            {"1", "2"},
        )
        self.assertEqual(
            {transition.successor.ring_state.open_endpoints[0].label for transition in transitions},
            {
                WriterClosureLabel(value=1, text="1"),
                WriterClosureLabel(value=2, text="2"),
            },
        )

    def test_raw_closure_label_allocator_returns_none_when_exhausted(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        labels = tuple(
            WriterClosureLabel(value=label.value, text=label.text())
            for label in prepared.policy.ring_labels
        )

        labels = writer_transitions._available_closure_labels_for_open(
            prepared,
            WriterRingState(
                label_state=WriterRingLabelState(allocated=labels),
            ),
        )

        self.assertEqual(labels, ())

    def test_raw_terminal_finalization_allows_cyclic_prepared_but_not_eos(self) -> None:
        prepared = _prepare(cyclopropane_facts())

        terminal = writer_transitions.finalize_writer_terminal_state(
            prepared,
            _raw_emitted_root_state(AtomId(0)),
        )

        self.assertIsNone(terminal)

    def test_raw_terminal_finalization_rejects_missing_active_frame(self) -> None:
        prepared = _prepare(chain_facts(("C",)))

        with self.assertRaises(SouthStarError) as caught:
            writer_transitions.finalize_writer_terminal_state(
                prepared,
                replace(_raw_emitted_root_state(AtomId(0)), active=None),
            )

        self.assertIs(caught.exception.kind, SouthStarErrorKind.INTERNAL_INVARIANT)

    def test_terminal_finalization_retains_active_final_atom(self) -> None:
        prepared = _prepare(chain_facts(("C",)))

        terminal = writer_transitions.finalize_writer_terminal_state(
            prepared,
            _raw_emitted_root_state(AtomId(0)),
        )

        self.assertIsNotNone(terminal)
        assert terminal is not None
        self.assertEqual(terminal.active.atom, AtomId(0))

    def test_terminal_finalization_rejects_open_closure_endpoint(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        state = _cyclopropane_terminal_open_closure_state()

        terminal = writer_transitions.finalize_writer_terminal_state(prepared, state)

        self.assertIsNone(terminal)
        self.assertFalse(writer_transitions.writer_state_is_eos(prepared, state))
        choices = writer_frontier_choices(
            prepared,
            WriterFrontierCursor(
                weighted_states=((writer_state_key(state), 1),),
            ),
        )
        self.assertIsNone(choices.terminal)

    def test_terminal_finalization_rejects_closure_candidate(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        state = replace(
            _cyclopropane_terminal_closed_closure_state(),
            ring_state=WriterRingState(),
        )

        terminal = writer_transitions.finalize_writer_terminal_state(prepared, state)

        self.assertIsNone(terminal)
        self.assertFalse(writer_transitions.writer_state_is_eos(prepared, state))

    def test_closed_closure_terminal_state_can_finalize(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        state = _cyclopropane_terminal_closed_closure_state()

        terminal = writer_transitions.finalize_writer_terminal_state(prepared, state)

        self.assertIsNotNone(terminal)

    def test_raw_eos_query_allows_cyclic_prepared_but_remains_false(self) -> None:
        prepared = _prepare(cyclopropane_facts())

        eos = writer_transitions.writer_state_is_eos(
            prepared,
            _raw_emitted_root_state(AtomId(0)),
        )

        self.assertFalse(eos)

    def test_raw_eos_query_rejects_missing_active_frame(self) -> None:
        prepared = _prepare(chain_facts(("C",)))

        with self.assertRaises(SouthStarError) as caught:
            writer_transitions.writer_state_is_eos(
                prepared,
                replace(_raw_emitted_root_state(AtomId(0)), active=None),
            )

        self.assertIs(caught.exception.kind, SouthStarErrorKind.INTERNAL_INVARIANT)

    def test_legal_transition_expansion_builds_one_graph_context(self) -> None:
        prepared = _prepare(cco_facts())

        with patch(
            "grimace._south_star1.writer_transitions.build_writer_graph_obligation_context",
            wraps=writer_transitions.build_writer_graph_obligation_context,
        ) as mocked:
            transitions = writer_transitions.legal_writer_transitions(
                prepared,
                _raw_initial_state(AtomId(0)),
            )

        self.assertEqual(mocked.call_count, 1)
        self.assertTrue(transitions)

    def test_terminal_finalization_builds_one_graph_context(self) -> None:
        prepared = _prepare(chain_facts(("C",)))
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        emitted = writer_frontier_choices(prepared, cursor).choices[0].successor
        state = writer_state_from_key(emitted.weighted_states[0][0])

        with patch(
            "grimace._south_star1.writer_transitions.build_writer_graph_obligation_context",
            wraps=writer_transitions.build_writer_graph_obligation_context,
        ) as mocked:
            terminal = writer_transitions.finalize_writer_terminal_state(prepared, state)

        self.assertEqual(mocked.call_count, 1)
        self.assertIsNotNone(terminal)

    def test_child_obligations_from_context_does_not_build_graph_context(self) -> None:
        prepared = _prepare(cco_facts())
        cursor = initial_writer_frontier_cursor(
            prepared,
            _writer_options(rooted_at_atom=0),
        )
        after_root = writer_frontier_choices(prepared, cursor).choices[0].successor
        state = writer_state_from_key(after_root.weighted_states[0][0])
        context = writer_transitions.build_writer_transition_expansion_context(
            prepared,
            state,
        )

        with patch(
            "grimace._south_star1.writer_transitions.build_writer_graph_obligation_context",
            side_effect=AssertionError("child obligations rebuilt graph context"),
        ):
            children = writer_transitions._child_obligations_from_context(
                context,
                state,
                AtomId(0),
            )

        self.assertEqual(len(children), 1)
        self.assertEqual(children[0].bond, BondId(0))
        self.assertEqual(children[0].child, AtomId(1))
        self.assertEqual(children[0].attachment_id, 0)
        self.assertEqual(
            children[0].attachment_action_kind,
            WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY,
        )
        self.assertFalse(children[0].pending_entry)

    def test_child_obligation_blockers_collect_closure_candidate_edges(self) -> None:
        context = SimpleNamespace(
            graph=SimpleNamespace(
                edge_partition=SimpleNamespace(
                    obligations=(
                        SimpleNamespace(
                            kind=WriterEdgeObligationKind.CLOSURE_CANDIDATE,
                            bond=BondId(7),
                        ),
                    ),
                ),
            ),
        )

        blockers = writer_transitions._child_obligation_blockers_from_context(
            context,  # type: ignore[arg-type]
        )

        self.assertEqual(len(blockers), 1)
        self.assertIs(
            blockers[0].kind,
            writer_transitions._WriterChildObligationBlockerKind.CLOSURE_CANDIDATE,
        )
        self.assertEqual(blockers[0].bond, BondId(7))

    def test_child_obligation_blockers_ignore_non_closure_candidate_edges(self) -> None:
        context = SimpleNamespace(
            graph=SimpleNamespace(
                edge_partition=SimpleNamespace(
                    obligations=(
                        SimpleNamespace(
                            kind=WriterEdgeObligationKind.TREE_ENTRY,
                            bond=BondId(1),
                        ),
                        SimpleNamespace(
                            kind=WriterEdgeObligationKind.CLOSED_CLOSURE,
                            bond=BondId(2),
                        ),
                    ),
                ),
            ),
        )

        blockers = writer_transitions._child_obligation_blockers_from_context(
            context,  # type: ignore[arg-type]
        )

        self.assertEqual(blockers, ())

    def test_child_obligation_blockers_raise_existing_unsupported_policy_error(self) -> None:
        blocker = writer_transitions._WriterChildObligationBlocker(
            kind=writer_transitions._WriterChildObligationBlockerKind.CLOSURE_CANDIDATE,
            bond=BondId(7),
        )

        with self.assertRaises(SouthStarError) as raised:
            writer_transitions._raise_for_child_obligation_blockers((blocker,))

        self.assertIs(
            raised.exception.kind,
            SouthStarErrorKind.UNSUPPORTED_POLICY,
        )
        self.assertIn(
            "WRITER_SHAPED closure-candidate edge obligations are not supported yet",
            str(raised.exception),
        )

    def test_child_obligation_blockers_for_atom_collect_multi_incidence_tree_entries(self) -> None:
        action = SimpleNamespace(
            attachment_id=7,
            kind=WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY,
        )
        attachment = SimpleNamespace(
            attachment_id=7,
            boundary=(
                SimpleNamespace(
                    bond=BondId(1),
                    written_atom=AtomId(0),
                    residual_atom=AtomId(2),
                ),
                SimpleNamespace(
                    bond=BondId(2),
                    written_atom=AtomId(0),
                    residual_atom=AtomId(3),
                ),
            ),
        )
        summary = SimpleNamespace(
            attachments=SimpleNamespace(attachments=(attachment,)),
            attachment_actions=(action,),
        )
        context = SimpleNamespace(
            graph=SimpleNamespace(
                edge_partition=SimpleNamespace(obligations=()),
                residual_summary=summary,
            ),
        )

        blockers = writer_transitions._child_obligation_blockers_for_atom(
            context,  # type: ignore[arg-type]
            AtomId(0),
        )

        self.assertEqual(len(blockers), 1)
        self.assertIs(
            blockers[0].kind,
            (
                writer_transitions._WriterChildObligationBlockerKind
                .MULTI_INCIDENCE_RESIDUAL_ATTACHMENT
            ),
        )
        self.assertEqual(blockers[0].atom, AtomId(0))
        self.assertEqual(blockers[0].attachment_id, 7)
        self.assertIs(
            blockers[0].attachment_action_kind,
            WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY,
        )

    def test_child_obligation_blockers_for_atom_ignore_multi_incidence_closure_open_actions(self) -> None:
        action = SimpleNamespace(
            attachment_id=7,
            kind=WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY,
        )
        attachment = SimpleNamespace(
            attachment_id=7,
            boundary=(
                SimpleNamespace(
                    bond=BondId(1),
                    written_atom=AtomId(0),
                    residual_atom=AtomId(2),
                ),
                SimpleNamespace(
                    bond=BondId(2),
                    written_atom=AtomId(0),
                    residual_atom=AtomId(3),
                ),
            ),
        )
        summary = SimpleNamespace(
            attachments=SimpleNamespace(attachments=(attachment,)),
            attachment_actions=(action,),
        )
        context = SimpleNamespace(
            graph=SimpleNamespace(
                edge_partition=SimpleNamespace(obligations=()),
                residual_summary=summary,
            ),
        )

        blockers = writer_transitions._child_obligation_blockers_for_atom(
            context,  # type: ignore[arg-type]
            AtomId(0),
        )

        self.assertEqual(blockers, ())

    def test_child_obligation_blockers_raise_multi_incidence_unsupported_policy_error(self) -> None:
        blocker = writer_transitions._WriterChildObligationBlocker(
            kind=(
                writer_transitions._WriterChildObligationBlockerKind
                .MULTI_INCIDENCE_RESIDUAL_ATTACHMENT
            ),
            atom=AtomId(0),
            attachment_id=7,
            attachment_action_kind=WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY,
        )

        with self.assertRaises(SouthStarError) as raised:
            writer_transitions._raise_for_child_obligation_blockers((blocker,))

        self.assertIs(
            raised.exception.kind,
            SouthStarErrorKind.UNSUPPORTED_POLICY,
        )
        self.assertIn(
            "WRITER_SHAPED multi-incidence residual attachments are not supported yet",
            str(raised.exception),
        )

    def test_active_child_scheduler_uses_atom_scoped_blockers(self) -> None:
        context = object()
        state = object()
        active_atom = AtomId(0)
        blocker = writer_transitions._WriterChildObligationBlocker(
            kind=(
                writer_transitions._WriterChildObligationBlockerKind
                .MULTI_INCIDENCE_RESIDUAL_ATTACHMENT
            ),
            atom=active_atom,
            attachment_id=7,
            attachment_action_kind=WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY,
        )

        with patch(
            "grimace._south_star1.writer_transitions._child_obligation_blockers_for_atom",
            return_value=(blocker,),
        ) as child_blockers, patch(
            "grimace._south_star1.writer_transitions._unblocked_child_obligations_from_context",
            side_effect=AssertionError("unblocked child obligations were computed"),
        ):
            with self.assertRaises(SouthStarError) as raised:
                writer_transitions._active_child_scheduled_actions_from_context(
                    context,  # type: ignore[arg-type]
                    state,  # type: ignore[arg-type]
                    active_atom,
                )

        child_blockers.assert_called_once_with(
            context,
            active_atom,
        )
        self.assertIs(
            raised.exception.kind,
            SouthStarErrorKind.UNSUPPORTED_POLICY,
        )
        self.assertIn(
            "WRITER_SHAPED multi-incidence residual attachments are not supported yet",
            str(raised.exception),
        )

    def test_active_child_schedule_surface_records_blockers_without_children(self) -> None:
        context = object()
        state = object()
        active_atom = AtomId(0)
        blocker = writer_transitions._WriterChildObligationBlocker(
            kind=(
                writer_transitions._WriterChildObligationBlockerKind
                .MULTI_INCIDENCE_RESIDUAL_ATTACHMENT
            ),
            atom=active_atom,
            attachment_id=7,
            attachment_action_kind=WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY,
        )

        with patch(
            "grimace._south_star1.writer_transitions._child_obligation_blockers_for_atom",
            return_value=(blocker,),
        ) as blockers, patch(
            "grimace._south_star1.writer_transitions._unblocked_child_obligations_from_context",
            side_effect=AssertionError("unblocked children should not be computed"),
        ) as unblocked:
            surface = writer_transitions._active_child_schedule_surface_from_context(
                context,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertEqual(surface.active_atom, active_atom)
        self.assertTrue(surface.blocked)
        self.assertEqual(surface.blockers, (blocker,))
        self.assertEqual(surface.child_obligations, ())
        self.assertEqual(surface.scheduled_actions, ())
        self.assertEqual(surface.graph_action_surfaces, ())
        blockers.assert_called_once_with(context, active_atom)
        unblocked.assert_not_called()

    def test_active_child_schedule_surface_records_child_actions_and_surfaces(self) -> None:
        context = object()
        state = object()
        active_atom = AtomId(0)
        child = writer_transitions._WriterChildObligation(
            bond=BondId(1),
            child=AtomId(2),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=9,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )

        with patch(
            "grimace._south_star1.writer_transitions._child_obligation_blockers_for_atom",
            return_value=(),
        ) as blockers, patch(
            "grimace._south_star1.writer_transitions._unblocked_child_obligations_from_context",
            return_value=(child,),
        ) as unblocked:
            surface = writer_transitions._active_child_schedule_surface_from_context(
                context,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertFalse(surface.blocked)
        self.assertEqual(surface.blockers, ())
        self.assertEqual(surface.child_obligations, (child,))
        self.assertEqual(len(surface.scheduled_actions), 1)
        graph_surface = surface.graph_action_surfaces[0]
        self.assertIs(
            graph_surface.kind,
            writer_transitions._WriterScheduledActionKind.ENTER_INLINE_CHILD,
        )
        self.assertEqual(graph_surface.active_atom, active_atom)
        self.assertEqual(graph_surface.bond, BondId(1))
        self.assertEqual(graph_surface.partner_atom, AtomId(2))
        self.assertEqual(graph_surface.boundary_atom, active_atom)
        self.assertEqual(graph_surface.attachment_id, 9)
        self.assertIs(
            graph_surface.attachment_action_kind,
            WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY,
        )
        self.assertIs(
            graph_surface.owner_kind,
            WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        blockers.assert_called_once_with(context, active_atom)
        unblocked.assert_called_once_with(context, state, active_atom)

    def test_active_child_schedule_surface_records_finish_action_when_no_children(self) -> None:
        context = object()
        state = object()
        active_atom = AtomId(5)

        with patch(
            "grimace._south_star1.writer_transitions._child_obligation_blockers_for_atom",
            return_value=(),
        ) as blockers, patch(
            "grimace._south_star1.writer_transitions._unblocked_child_obligations_from_context",
            return_value=(),
        ) as unblocked:
            surface = writer_transitions._active_child_schedule_surface_from_context(
                context,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertFalse(surface.blocked)
        self.assertEqual(surface.blockers, ())
        self.assertEqual(surface.child_obligations, ())
        self.assertEqual(len(surface.scheduled_actions), 1)
        graph_surface = surface.graph_action_surfaces[0]
        self.assertIs(
            graph_surface.kind,
            writer_transitions._WriterScheduledActionKind.FINISH_ACTIVE,
        )
        self.assertEqual(graph_surface.active_atom, active_atom)
        blockers.assert_called_once_with(context, active_atom)
        unblocked.assert_called_once_with(context, state, active_atom)

    def test_checked_child_obligations_use_atom_scoped_blockers(self) -> None:
        context = object()
        state = object()
        atom = AtomId(0)
        blocker = writer_transitions._WriterChildObligationBlocker(
            kind=(
                writer_transitions._WriterChildObligationBlockerKind
                .MULTI_INCIDENCE_RESIDUAL_ATTACHMENT
            ),
            atom=atom,
            attachment_id=7,
            attachment_action_kind=WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY,
        )

        with patch(
            "grimace._south_star1.writer_transitions._child_obligation_blockers_for_atom",
            return_value=(blocker,),
        ) as blockers, patch(
            "grimace._south_star1.writer_transitions._unblocked_child_obligations_from_context",
            side_effect=AssertionError("unblocked child obligations were computed"),
        ):
            with self.assertRaises(SouthStarError) as raised:
                writer_transitions._child_obligations_from_context(
                    context,  # type: ignore[arg-type]
                    state,  # type: ignore[arg-type]
                    atom,
                )

        blockers.assert_called_once_with(
            context,
            atom,
        )
        self.assertIs(
            raised.exception.kind,
            SouthStarErrorKind.UNSUPPORTED_POLICY,
        )
        self.assertIn(
            "WRITER_SHAPED multi-incidence residual attachments are not supported yet",
            str(raised.exception),
        )

    def test_checked_child_obligations_delegate_when_no_atom_scoped_blockers(self) -> None:
        context = object()
        state = object()
        atom = AtomId(0)
        child = writer_transitions._WriterChildObligation(
            bond=BondId(1),
            child=AtomId(2),
        )

        with patch(
            "grimace._south_star1.writer_transitions._child_obligation_blockers_for_atom",
            return_value=(),
        ) as blockers, patch(
            "grimace._south_star1.writer_transitions._unblocked_child_obligations_from_context",
            return_value=(child,),
        ) as unblocked:
            result = writer_transitions._child_obligations_from_context(
                context,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                atom,
            )

        self.assertEqual(result, (child,))
        blockers.assert_called_once_with(
            context,
            atom,
        )
        unblocked.assert_called_once_with(
            context,
            state,
            atom,
        )

    def test_checked_child_obligations_preserve_multi_incidence_policy_error(self) -> None:
        context = object()
        state = object()
        atom = AtomId(0)
        blocker = writer_transitions._WriterChildObligationBlocker(
            kind=(
                writer_transitions._WriterChildObligationBlockerKind
                .MULTI_INCIDENCE_RESIDUAL_ATTACHMENT
            ),
            atom=atom,
            attachment_id=7,
            attachment_action_kind=WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY,
        )

        with patch(
            "grimace._south_star1.writer_transitions._child_obligation_blockers_for_atom",
            return_value=(blocker,),
        ), patch(
            "grimace._south_star1.writer_transitions._unblocked_child_obligations_from_context",
            side_effect=AssertionError("unblocked builder should not run"),
        ):
            with self.assertRaises(SouthStarError) as raised:
                writer_transitions._child_obligations_from_context(
                    context,  # type: ignore[arg-type]
                    state,  # type: ignore[arg-type]
                    atom,
                )

        self.assertIs(
            raised.exception.kind,
            SouthStarErrorKind.UNSUPPORTED_POLICY,
        )
        self.assertIn(
            "WRITER_SHAPED multi-incidence residual attachments are not supported yet",
            str(raised.exception),
        )

    def test_unblocked_child_obligations_reject_multi_incidence_as_internal_invariant(self) -> None:
        action = SimpleNamespace(
            attachment_id=7,
            kind=WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY,
        )
        attachment = SimpleNamespace(
            attachment_id=7,
            boundary=(
                SimpleNamespace(
                    bond=BondId(1),
                    written_atom=AtomId(0),
                    residual_atom=AtomId(2),
                ),
                SimpleNamespace(
                    bond=BondId(2),
                    written_atom=AtomId(0),
                    residual_atom=AtomId(3),
                ),
            ),
        )
        summary = SimpleNamespace(
            attachments=SimpleNamespace(attachments=(attachment,)),
            attachment_actions=(action,),
        )
        context = SimpleNamespace(
            graph=SimpleNamespace(
                residual_summary=summary,
            ),
        )
        state = SimpleNamespace(
            obligations=SimpleNamespace(
                pending_entry=None,
            ),
        )

        with self.assertRaises(SouthStarError) as raised:
            writer_transitions._unblocked_child_obligations_from_context(
                context,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                AtomId(0),
            )

        self.assertIs(
            raised.exception.kind,
            SouthStarErrorKind.INTERNAL_INVARIANT,
        )
        self.assertIn(
            "unblocked child obligation builder received non-singleton boundary",
            str(raised.exception),
        )

    def test_unblocked_child_obligations_carry_action_incidence_metadata(self) -> None:
        action = SimpleNamespace(
            attachment_id=7,
            kind=WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY,
        )
        attachment = SimpleNamespace(
            attachment_id=7,
            boundary=(
                SimpleNamespace(
                    bond=BondId(1),
                    written_atom=AtomId(0),
                    residual_atom=AtomId(2),
                    owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
                ),
            ),
        )
        summary = SimpleNamespace(
            attachments=SimpleNamespace(attachments=(attachment,)),
            attachment_actions=(action,),
        )
        context = SimpleNamespace(
            graph=SimpleNamespace(
                residual_summary=summary,
            ),
        )
        state = SimpleNamespace(
            obligations=SimpleNamespace(
                pending_entry=None,
            ),
        )

        children = writer_transitions._unblocked_child_obligations_from_context(
            context,  # type: ignore[arg-type]
            state,  # type: ignore[arg-type]
            AtomId(0),
        )

        self.assertEqual(len(children), 1)
        self.assertEqual(children[0].bond, BondId(1))
        self.assertEqual(children[0].child, AtomId(2))
        self.assertEqual(children[0].boundary_atom, AtomId(0))
        self.assertIs(
            children[0].owner_kind,
            WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        self.assertEqual(children[0].attachment_id, 7)
        self.assertIs(
            children[0].attachment_action_kind,
            WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY,
        )
        self.assertFalse(children[0].pending_entry)

    def test_unblocked_child_obligations_carry_pending_parent_as_boundary_atom(self) -> None:
        pending = PendingWriterEntry(
            parent=AtomId(0),
            child=AtomId(2),
            bond=BondId(1),
            branch=False,
        )
        summary = SimpleNamespace(
            attachments=SimpleNamespace(attachments=()),
            attachment_actions=(),
        )
        context = SimpleNamespace(
            graph=SimpleNamespace(
                residual_summary=summary,
            ),
        )
        state = SimpleNamespace(
            obligations=SimpleNamespace(
                pending_entry=pending,
            ),
        )

        children = writer_transitions._unblocked_child_obligations_from_context(
            context,  # type: ignore[arg-type]
            state,  # type: ignore[arg-type]
            AtomId(0),
        )

        self.assertEqual(len(children), 1)
        self.assertEqual(children[0].bond, BondId(1))
        self.assertEqual(children[0].child, AtomId(2))
        self.assertEqual(children[0].boundary_atom, AtomId(0))
        self.assertIsNone(children[0].owner_kind)
        self.assertIsNone(children[0].attachment_id)
        self.assertIsNone(children[0].attachment_action_kind)
        self.assertTrue(children[0].pending_entry)

    def test_scheduled_writer_transitions_dispatches_pending_entry_actions(self) -> None:
        prepared = object()
        context = object()
        pending = PendingWriterEntry(
            parent=AtomId(0),
            child=AtomId(1),
            bond=BondId(0),
            branch=False,
        )
        state = SimpleNamespace(
            obligations=SimpleNamespace(
                pending_entry=pending,
            ),
        )
        action = object()
        emission = object()
        surviving_emission = object()
        transition = object()
        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(action,),  # type: ignore[arg-type]
            emissions=(emission,),  # type: ignore[arg-type]
            surviving_emissions=(surviving_emission,),  # type: ignore[arg-type]
        )

        with patch(
            "grimace._south_star1.writer_transitions._pending_entry_scheduled_actions",
            return_value=(action,),
        ) as pending_actions, patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            return_value=batch,
        ) as emission_batch, patch(
            "grimace._south_star1.writer_transitions._transitions_from_scheduled_action_emissions",
            return_value=(transition,),
        ) as flatten_emissions, patch(
            "grimace._south_star1.writer_transitions._root_atom_transitions",
            side_effect=AssertionError("root atom path should not run"),
        ), patch(
            "grimace._south_star1.writer_transitions._active_emitted_transitions",
            side_effect=AssertionError("active-emitted path should not run"),
        ):
            result = writer_transitions._scheduled_writer_transitions(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
            )

        self.assertEqual(result, (transition,))
        pending_actions.assert_called_once_with(state)
        emission_batch.assert_called_once_with(
            prepared,
            state,
            context,
            (action,),
        )
        flatten_emissions.assert_called_once_with(
            (surviving_emission,),
        )

    def test_scheduled_writer_transitions_dispatches_root_atom_actions(self) -> None:
        prepared = object()
        context = object()
        state = SimpleNamespace(
            obligations=SimpleNamespace(
                pending_entry=None,
            ),
            active=SimpleNamespace(
                atom=AtomId(0),
                atom_emitted=False,
            ),
        )
        action = object()
        emission = object()
        surviving_emission = object()
        transition = object()
        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(action,),  # type: ignore[arg-type]
            emissions=(emission,),  # type: ignore[arg-type]
            surviving_emissions=(surviving_emission,),  # type: ignore[arg-type]
        )

        with patch(
            "grimace._south_star1.writer_transitions._root_atom_scheduled_actions",
            return_value=(action,),
        ) as root_actions, patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            return_value=batch,
        ) as emission_batch, patch(
            "grimace._south_star1.writer_transitions._transitions_from_scheduled_action_emissions",
            return_value=(transition,),
        ) as flatten_emissions, patch(
            "grimace._south_star1.writer_transitions._active_emitted_transitions",
            side_effect=AssertionError("active-emitted path should not run"),
        ):
            result = writer_transitions._scheduled_writer_transitions(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
            )

        self.assertEqual(result, (transition,))
        root_actions.assert_called_once_with(state)
        emission_batch.assert_called_once_with(
            prepared,
            state,
            context,
            (action,),
        )
        flatten_emissions.assert_called_once_with(
            (surviving_emission,),
        )

    def test_top_level_scheduled_actions_prefer_pending_entry_over_root(self) -> None:
        pending = PendingWriterEntry(
            parent=AtomId(0),
            child=AtomId(1),
            bond=BondId(0),
            branch=False,
        )
        state = SimpleNamespace(
            obligations=SimpleNamespace(
                pending_entry=pending,
            ),
            active=SimpleNamespace(
                atom=AtomId(9),
                atom_emitted=False,
            ),
        )

        actions = writer_transitions._top_level_scheduled_actions(
            state,  # type: ignore[arg-type]
        )

        self.assertEqual(len(actions), 1)
        self.assertIs(
            actions[0].kind,
            writer_transitions._WriterScheduledActionKind.CONSUME_PENDING_ENTRY,
        )
        self.assertEqual(actions[0].pending_entry, pending)

    def test_top_level_scheduled_actions_return_root_when_no_pending(self) -> None:
        state = SimpleNamespace(
            obligations=SimpleNamespace(
                pending_entry=None,
            ),
            active=SimpleNamespace(
                atom=AtomId(3),
                atom_emitted=False,
            ),
        )

        actions = writer_transitions._top_level_scheduled_actions(
            state,  # type: ignore[arg-type]
        )

        self.assertEqual(len(actions), 1)
        self.assertIs(
            actions[0].kind,
            writer_transitions._WriterScheduledActionKind.EMIT_ROOT_ATOM,
        )
        self.assertEqual(actions[0].parent, AtomId(3))

    def test_top_level_scheduled_actions_empty_for_active_emitted_state(self) -> None:
        state = SimpleNamespace(
            obligations=SimpleNamespace(
                pending_entry=None,
            ),
            active=SimpleNamespace(
                atom=AtomId(3),
                atom_emitted=True,
            ),
        )

        self.assertEqual(
            writer_transitions._top_level_scheduled_actions(
                state,  # type: ignore[arg-type]
            ),
            (),
        )

    def test_active_emitted_schedule_decision_rejects_closure_with_child_batch(self) -> None:
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_endpoint_decision = (
            writer_transitions._WriterClosureEndpointScheduleDecision(
                pair_batch=pair_batch,
                open_batch=open_batch,
                surviving_emissions=(),
            )
        )
        closure_batch = writer_transitions._closure_endpoint_combined_batch(
            closure_endpoint_decision
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )

        with self.assertRaises(SouthStarError):
            writer_transitions._WriterActiveEmittedScheduleDecision(
                kind=(
                    writer_transitions._WriterActiveEmittedScheduleDecisionKind
                    .CLOSURE_ENDPOINT
                ),
                closure_endpoint_decision=closure_endpoint_decision,
                closure_batch=closure_batch,
                selected_batch=closure_batch,
                child_batch=child_batch,
            )

    def test_active_emitted_schedule_decision_requires_selected_child_batch(self) -> None:
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_endpoint_decision = (
            writer_transitions._WriterClosureEndpointScheduleDecision(
                pair_batch=pair_batch,
                open_batch=open_batch,
                surviving_emissions=(),
            )
        )
        closure_batch = writer_transitions._closure_endpoint_combined_batch(
            closure_endpoint_decision
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )

        with self.assertRaises(SouthStarError):
            writer_transitions._WriterActiveEmittedScheduleDecision(
                kind=(
                    writer_transitions._WriterActiveEmittedScheduleDecisionKind
                    .ACTIVE_CHILD
                ),
                closure_endpoint_decision=closure_endpoint_decision,
                closure_batch=closure_batch,
                selected_batch=closure_batch,
                child_batch=child_batch,
            )

    def test_top_level_schedule_decision_rejects_active_payload_for_top_level_actions(self) -> None:
        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        active_decision = writer_transitions._WriterActiveEmittedScheduleDecision(
            kind=(
                writer_transitions._WriterActiveEmittedScheduleDecisionKind
                .CLOSURE_ENDPOINT
            ),
            closure_endpoint_decision=writer_transitions._WriterClosureEndpointScheduleDecision(
                pair_batch=batch,
                open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                    actions=(),
                    emissions=(),
                    surviving_emissions=(),
                ),
                surviving_emissions=(),
            ),
            closure_batch=batch,
            selected_batch=batch,
        )

        with self.assertRaises(SouthStarError):
            writer_transitions._WriterTopLevelScheduleDecision(
                kind=(
                    writer_transitions._WriterTopLevelScheduleDecisionKind
                    .TOP_LEVEL_ACTIONS
                ),
                selected_batch=batch,
                top_level_batch=batch,
                active_emitted_decision=active_decision,
            )

    def test_top_level_schedule_decision_requires_selected_active_batch(self) -> None:
        active_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        wrong_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        active_decision = writer_transitions._WriterActiveEmittedScheduleDecision(
            kind=(
                writer_transitions._WriterActiveEmittedScheduleDecisionKind
                .CLOSURE_ENDPOINT
            ),
            closure_endpoint_decision=writer_transitions._WriterClosureEndpointScheduleDecision(
                pair_batch=active_batch,
                open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                    actions=(),
                    emissions=(),
                    surviving_emissions=(),
                ),
                surviving_emissions=(),
            ),
            closure_batch=active_batch,
            selected_batch=active_batch,
        )

        with self.assertRaises(SouthStarError):
            writer_transitions._WriterTopLevelScheduleDecision(
                kind=(
                    writer_transitions._WriterTopLevelScheduleDecisionKind
                    .ACTIVE_EMITTED
                ),
                selected_batch=wrong_batch,
                active_emitted_decision=active_decision,
            )

    def test_scheduler_decisions_accept_valid_payloads(self) -> None:
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_endpoint_decision = (
            writer_transitions._WriterClosureEndpointScheduleDecision(
                pair_batch=pair_batch,
                open_batch=open_batch,
                surviving_emissions=(),
            )
        )
        closure_batch = writer_transitions._closure_endpoint_combined_batch(
            closure_endpoint_decision
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=AtomId(0),
            blockers=(),
            child_obligations=(),
            scheduled_actions=child_batch.actions,
        )

        closure_decision = writer_transitions._WriterActiveEmittedScheduleDecision(
            kind=(
                writer_transitions._WriterActiveEmittedScheduleDecisionKind
                .CLOSURE_ENDPOINT
            ),
            closure_endpoint_decision=closure_endpoint_decision,
            closure_batch=closure_batch,
            selected_batch=closure_batch,
        )

        child_decision = writer_transitions._WriterActiveEmittedScheduleDecision(
            kind=(
                writer_transitions._WriterActiveEmittedScheduleDecisionKind
                .ACTIVE_CHILD
            ),
            closure_endpoint_decision=closure_endpoint_decision,
            closure_batch=closure_batch,
            child_batch=child_batch,
            child_schedule_surface=child_surface,
            selected_batch=child_batch,
        )

        writer_transitions._WriterTopLevelScheduleDecision(
            kind=(
                writer_transitions._WriterTopLevelScheduleDecisionKind
                .TOP_LEVEL_ACTIONS
            ),
            selected_batch=closure_batch,
            top_level_batch=closure_batch,
        )

        writer_transitions._WriterTopLevelScheduleDecision(
            kind=(
                writer_transitions._WriterTopLevelScheduleDecisionKind
                .ACTIVE_EMITTED
            ),
            selected_batch=closure_decision.selected_batch,
            active_emitted_decision=closure_decision,
        )

        writer_transitions._WriterTopLevelScheduleDecision(
            kind=(
                writer_transitions._WriterTopLevelScheduleDecisionKind
                .ACTIVE_EMITTED
            ),
            selected_batch=child_decision.selected_batch,
            active_emitted_decision=child_decision,
        )

    def test_active_emitted_decision_constructors_preserve_selected_batches(self) -> None:
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_endpoint_decision = (
            writer_transitions._WriterClosureEndpointScheduleDecision(
                pair_batch=pair_batch,
                open_batch=open_batch,
                surviving_emissions=(),
            )
        )
        closure_batch = writer_transitions._closure_endpoint_combined_batch(
            closure_endpoint_decision
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=AtomId(0),
            blockers=(),
            child_obligations=(),
            scheduled_actions=child_batch.actions,
        )

        closure_decision = writer_transitions._active_emitted_closure_decision(
            closure_endpoint_decision,
        )
        child_decision = writer_transitions._active_emitted_child_decision(
            closure_endpoint_decision,
            child_surface,
            child_batch,
        )

        self.assertIs(
            closure_decision.kind,
            writer_transitions._WriterActiveEmittedScheduleDecisionKind.CLOSURE_ENDPOINT,
        )
        self.assertEqual(closure_decision.closure_batch, closure_batch)
        self.assertIs(closure_decision.selected_batch, closure_decision.closure_batch)
        self.assertIsNone(closure_decision.child_batch)
        self.assertIs(
            child_decision.kind,
            writer_transitions._WriterActiveEmittedScheduleDecisionKind.ACTIVE_CHILD,
        )
        self.assertEqual(child_decision.closure_batch, closure_batch)
        self.assertIs(child_decision.child_batch, child_batch)
        self.assertIs(child_decision.child_schedule_surface, child_surface)
        self.assertIs(child_decision.selected_batch, child_batch)

    def test_active_emitted_decision_constructors_carry_closure_endpoint_decision(self) -> None:
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_endpoint_decision = (
            writer_transitions._WriterClosureEndpointScheduleDecision(
                pair_batch=pair_batch,
                open_batch=open_batch,
                surviving_emissions=(),
            )
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=AtomId(0),
            blockers=(),
            child_obligations=(),
            scheduled_actions=child_batch.actions,
        )

        closure_decision = writer_transitions._active_emitted_closure_decision(
            closure_endpoint_decision,
        )
        child_decision = writer_transitions._active_emitted_child_decision(
            closure_endpoint_decision,
            child_surface,
            child_batch,
        )

        self.assertIs(
            closure_decision.closure_endpoint_decision,
            closure_endpoint_decision,
        )
        self.assertEqual(
            closure_decision.closure_batch,
            writer_transitions._closure_endpoint_combined_batch(
                closure_endpoint_decision
            ),
        )
        self.assertIs(closure_decision.selected_batch, closure_decision.closure_batch)
        self.assertIs(
            child_decision.closure_endpoint_decision,
            closure_endpoint_decision,
        )
        self.assertEqual(
            child_decision.closure_batch,
            writer_transitions._closure_endpoint_combined_batch(
                closure_endpoint_decision
            ),
        )
        self.assertIs(
            child_decision.child_schedule_surface,
            child_surface,
        )
        self.assertIs(child_decision.selected_batch, child_batch)

    def test_active_emitted_child_decision_retains_child_schedule_surface(self) -> None:
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_endpoint_decision = (
            writer_transitions._WriterClosureEndpointScheduleDecision(
                pair_batch=pair_batch,
                open_batch=open_batch,
                surviving_emissions=(),
            )
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(1),
            child=AtomId(2),
            boundary_atom=AtomId(0),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            AtomId(0),
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=AtomId(0),
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )
        child_emission = writer_transitions._WriterScheduledActionEmission(
            action=child_action,
            transitions=(object(),),  # type: ignore[arg-type]
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(child_action,),
            emissions=(child_emission,),
            surviving_emissions=(child_emission,),
        )

        decision = writer_transitions._active_emitted_child_decision(
            closure_endpoint_decision,
            child_surface,
            child_batch,
        )

        self.assertIs(
            decision.kind,
            writer_transitions._WriterActiveEmittedScheduleDecisionKind.ACTIVE_CHILD,
        )
        self.assertIs(decision.child_schedule_surface, child_surface)
        self.assertIs(decision.child_batch, child_batch)
        self.assertIs(decision.selected_batch, child_batch)
        self.assertEqual(
            decision.considered_graph_action_surfaces,
            child_surface.graph_action_surfaces,
        )
        self.assertEqual(
            decision.selected_graph_action_surfaces,
            child_batch.surviving_graph_action_surfaces,
        )

    def test_active_emitted_closure_decision_considered_surfaces_delegate_to_closure_decision(self) -> None:
        label = WriterClosureLabel(value=1, text="1")
        endpoint = WriterOpenClosureEndpoint(
            bond=BondId(1),
            first_atom=AtomId(2),
            second_atom=AtomId(0),
            label=label,
            first_endpoint_text="1",
            first_endpoint_bond_text="",
        )
        closure = WriterClosedClosure(
            bond=BondId(1),
            first_atom=AtomId(2),
            second_atom=AtomId(0),
            label=label,
            first_endpoint_text="1",
            second_endpoint_text="1",
            first_endpoint_bond_text="",
            second_endpoint_bond_text="",
        )
        pair = writer_transitions._WriterClosurePairObligation(
            endpoint=endpoint,
            closure=closure,
        )
        pair_action = writer_transitions._pair_closure_endpoint_action(
            AtomId(0),
            pair,
        )
        surface = writer_transitions._WriterClosureEndpointScheduleSurface(
            active_atom=AtomId(0),
            pair_actions=(pair_action,),
            open_actions=(),
        )
        emission = writer_transitions._WriterScheduledActionEmission(
            action=pair_action,
            transitions=(object(),),  # type: ignore[arg-type]
        )
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(pair_action,),
            emissions=(emission,),
            surviving_emissions=(emission,),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_endpoint_decision = (
            writer_transitions._WriterClosureEndpointScheduleDecision(
                pair_batch=pair_batch,
                open_batch=open_batch,
                surviving_emissions=(emission,),
                schedule_surface=surface,
            )
        )

        decision = writer_transitions._active_emitted_closure_decision(
            closure_endpoint_decision,
        )

        self.assertEqual(
            decision.considered_graph_action_surfaces,
            closure_endpoint_decision.considered_graph_action_surfaces,
        )
        self.assertEqual(
            decision.selected_graph_action_surfaces,
            closure_endpoint_decision.selected_graph_action_surfaces,
        )

    def test_active_emitted_closure_decision_retains_graph_policy_decision(self) -> None:
        closure_endpoint_decision, _, _ = (
            self._test_closure_endpoint_decision(
                pair_survives=True,
                open_survives=False,
            )
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .CLOSURE_ENDPOINT
            ),
            active_atom=AtomId(0),
            closure_endpoint_decision=closure_endpoint_decision,
        )

        decision = writer_transitions._active_emitted_closure_decision(
            closure_endpoint_decision,
            graph_policy_decision=policy,
        )

        self.assertIs(
            decision.kind,
            writer_transitions._WriterActiveEmittedScheduleDecisionKind.CLOSURE_ENDPOINT,
        )
        self.assertIs(decision.graph_policy_decision, policy)
        self.assertIs(
            decision.closure_endpoint_decision,
            policy.closure_endpoint_decision,
        )
        self.assertIsNone(decision.child_schedule_surface)
        self.assertIsNone(decision.child_batch)

    def test_active_emitted_child_decision_retains_graph_policy_decision(self) -> None:
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_endpoint_decision = (
            writer_transitions._WriterClosureEndpointScheduleDecision(
                pair_batch=pair_batch,
                open_batch=open_batch,
                surviving_emissions=(),
            )
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(1),
            child=AtomId(2),
        )
        child_action = writer_transitions._enter_inline_child_action(
            AtomId(0),
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=AtomId(0),
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )
        child_emission = writer_transitions._WriterScheduledActionEmission(
            action=child_action,
            transitions=(object(),),  # type: ignore[arg-type]
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(child_action,),
            emissions=(child_emission,),
            surviving_emissions=(child_emission,),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD
            ),
            active_atom=AtomId(0),
            closure_endpoint_decision=closure_endpoint_decision,
            child_schedule_surface=child_surface,
        )

        decision = writer_transitions._active_emitted_child_decision(
            closure_endpoint_decision=closure_endpoint_decision,
            child_schedule_surface=child_surface,
            child_batch=child_batch,
            graph_policy_decision=policy,
        )

        self.assertIs(
            decision.kind,
            writer_transitions._WriterActiveEmittedScheduleDecisionKind.ACTIVE_CHILD,
        )
        self.assertIs(decision.graph_policy_decision, policy)
        self.assertIs(decision.child_schedule_surface, child_surface)
        self.assertIs(decision.child_batch, child_batch)
        self.assertIs(decision.selected_batch, child_batch)

    def test_active_emitted_child_decision_rejects_blocked_graph_policy_decision(self) -> None:
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=pair_batch,
            open_batch=open_batch,
            surviving_emissions=(),
        )
        blocker = writer_transitions._WriterChildObligationBlocker(
            kind=(
                writer_transitions._WriterChildObligationBlockerKind
                .MULTI_INCIDENCE_RESIDUAL_ATTACHMENT
            ),
            atom=AtomId(0),
            attachment_id=7,
            attachment_action_kind=WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY,
        )
        blocked_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=AtomId(0),
            blockers=(blocker,),
            child_obligations=(),
            scheduled_actions=(),
        )
        blocked_policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .BLOCKED_CHILD
            ),
            active_atom=AtomId(0),
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=blocked_surface,
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(1),
            child=AtomId(2),
        )
        child_action = writer_transitions._enter_inline_child_action(
            AtomId(0),
            child,
        )
        unblocked_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=AtomId(0),
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(child_action,),
            emissions=(),
            surviving_emissions=(),
        )

        with self.assertRaises(SouthStarError) as raised:
            writer_transitions._active_emitted_child_decision(
                closure_endpoint_decision=closure_decision,
                child_schedule_surface=unblocked_surface,
                child_batch=child_batch,
                graph_policy_decision=blocked_policy,
            )

        self.assertIs(
            raised.exception.kind,
            SouthStarErrorKind.INTERNAL_INVARIANT,
        )

    def test_active_emitted_child_schedule_considered_surfaces_use_retained_policy(self) -> None:
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=9,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )
        child_emission = writer_transitions._WriterScheduledActionEmission(
            action=child_action,
            transitions=(object(),),  # type: ignore[arg-type]
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(child_action,),
            emissions=(child_emission,),
            surviving_emissions=(child_emission,),
        )

        decision = writer_transitions._active_emitted_child_decision(
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
            child_batch=child_batch,
            graph_policy_decision=policy,
        )

        self.assertEqual(
            decision.considered_graph_action_surfaces,
            policy.considered_graph_action_surfaces,
        )
        self.assertEqual(
            decision.policy_chosen_graph_action_surfaces,
            policy.chosen_graph_action_surfaces,
        )
        self.assertEqual(
            decision.selected_graph_action_surfaces,
            child_batch.surviving_graph_action_surfaces,
        )

    def test_active_emitted_schedule_decision_filters_chosen_and_selected_policy_families(self) -> None:
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(
                    writer_transitions._WriterScheduledActionEmission(
                        action=open_action,
                        transitions=(),
                    ),
                ),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=9,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )
        child_emission = writer_transitions._WriterScheduledActionEmission(
            action=child_action,
            transitions=(object(),),  # type: ignore[arg-type]
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(child_action,),
            emissions=(child_emission,),
            surviving_emissions=(child_emission,),
        )

        decision = writer_transitions._active_emitted_child_decision(
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
            child_batch=child_batch,
            graph_policy_decision=policy,
        )

        self.assertEqual(
            decision.policy_chosen_graph_action_surfaces_for_policy_family(
                writer_transitions._WriterGraphPolicyActionFamily.CYCLIC_TREE_ENTRY,
            ),
            child_surface.graph_action_surfaces,
        )
        self.assertEqual(
            decision.selected_graph_action_surfaces_for_policy_family(
                writer_transitions._WriterGraphPolicyActionFamily.CYCLIC_TREE_ENTRY,
            ),
            child_batch.surviving_graph_action_surfaces,
        )

    def test_active_emitted_schedule_decision_exposes_selected_residual_policy_emission_groups(self) -> None:
        active_atom = AtomId(0)
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=pair_batch,
            open_batch=open_batch,
            surviving_emissions=(),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=9,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )
        child_emission = writer_transitions._WriterScheduledActionEmission(
            action=child_action,
            transitions=(SimpleNamespace(emitted_text="C"),),  # type: ignore[arg-type]
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(child_action,),
            emissions=(child_emission,),
            surviving_emissions=(child_emission,),
        )
        decision = writer_transitions._active_emitted_child_decision(
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
            child_batch=child_batch,
            graph_policy_decision=policy,
        )

        groups = decision.selected_residual_attachment_policy_emission_groups

        self.assertEqual(len(groups), 1)
        self.assertEqual(
            groups[0].cyclic_tree_entry_emissions,
            (child_emission,),
        )
        self.assertEqual(
            groups[0].surviving_cyclic_tree_entry_emissions,
            (child_emission,),
        )

    def test_top_level_decision_constructors_preserve_selected_batches(self) -> None:
        top_level_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        active_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        active_closure_endpoint_decision = (
            writer_transitions._WriterClosureEndpointScheduleDecision(
                pair_batch=active_batch,
                open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                    actions=(),
                    emissions=(),
                    surviving_emissions=(),
                ),
                surviving_emissions=(),
            )
        )
        active_decision = writer_transitions._active_emitted_closure_decision(
            active_closure_endpoint_decision,
        )

        top_level_decision = writer_transitions._top_level_actions_decision(
            top_level_batch,
        )
        active_top_level_decision = (
            writer_transitions._top_level_active_emitted_decision(
                active_decision,
            )
        )

        self.assertIs(
            top_level_decision.kind,
            writer_transitions._WriterTopLevelScheduleDecisionKind.TOP_LEVEL_ACTIONS,
        )
        self.assertIs(top_level_decision.selected_batch, top_level_batch)
        self.assertIs(top_level_decision.top_level_batch, top_level_batch)
        self.assertIsNone(top_level_decision.active_emitted_decision)
        self.assertIs(
            active_top_level_decision.kind,
            writer_transitions._WriterTopLevelScheduleDecisionKind.ACTIVE_EMITTED,
        )
        self.assertIs(
            active_top_level_decision.selected_batch,
            active_decision.selected_batch,
        )
        self.assertEqual(active_top_level_decision.selected_batch, active_batch)
        self.assertIs(
            active_top_level_decision.active_emitted_decision,
            active_decision,
        )
        self.assertIsNone(active_top_level_decision.top_level_batch)

    def test_top_level_actions_decision_exposes_selected_frontier_and_surfaces(self) -> None:
        action = writer_transitions._emit_root_atom_action(AtomId(0))
        transition = SimpleNamespace(emitted_text="C")
        emission = writer_transitions._WriterScheduledActionEmission(
            action=action,
            transitions=(transition,),  # type: ignore[arg-type]
        )
        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(action,),
            emissions=(emission,),
            surviving_emissions=(emission,),
        )

        decision = writer_transitions._top_level_actions_decision(batch)

        self.assertEqual(
            decision.considered_graph_action_surfaces,
            batch.graph_action_surfaces,
        )
        self.assertEqual(
            decision.selected_graph_action_surfaces,
            batch.surviving_graph_action_surfaces,
        )
        self.assertEqual(decision.selected_transitions, batch.surviving_transitions)
        self.assertEqual(
            decision.selected_next_token_frontier,
            batch.surviving_next_token_frontier,
        )

    def test_top_level_active_emitted_decision_delegates_selected_frontier(self) -> None:
        action = writer_transitions._finish_active_action(AtomId(0))
        transition = SimpleNamespace(emitted_text="")
        emission = writer_transitions._WriterScheduledActionEmission(
            action=action,
            transitions=(transition,),  # type: ignore[arg-type]
        )
        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(action,),
            emissions=(emission,),
            surviving_emissions=(emission,),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_endpoint_decision = (
            writer_transitions._WriterClosureEndpointScheduleDecision(
                pair_batch=batch,
                open_batch=open_batch,
                surviving_emissions=(emission,),
            )
        )
        active_decision = writer_transitions._active_emitted_closure_decision(
            closure_endpoint_decision,
        )

        top_decision = writer_transitions._top_level_active_emitted_decision(
            active_decision,
        )

        self.assertIs(top_decision.selected_batch, active_decision.selected_batch)
        self.assertEqual(
            top_decision.considered_graph_action_surfaces,
            active_decision.considered_graph_action_surfaces,
        )
        self.assertEqual(
            top_decision.selected_graph_action_surfaces,
            active_decision.selected_graph_action_surfaces,
        )
        self.assertEqual(
            top_decision.selected_next_token_frontier,
            active_decision.selected_next_token_frontier,
        )

    def test_top_level_schedule_outcome_validates_top_level_scheduled_payload(self) -> None:
        action = writer_transitions._emit_root_atom_action(AtomId(0))
        transition = SimpleNamespace(emitted_text="C")
        emission = writer_transitions._WriterScheduledActionEmission(
            action=action,
            transitions=(transition,),  # type: ignore[arg-type]
        )
        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(action,),
            emissions=(emission,),
            surviving_emissions=(emission,),
        )
        decision = writer_transitions._top_level_actions_decision(batch)

        outcome = writer_transitions._WriterTopLevelScheduleOutcome(
            kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.SCHEDULED,
            schedule_decision=decision,
        )

        self.assertIs(outcome.schedule_decision, decision)
        self.assertIsNone(outcome.active_emitted_outcome)
        self.assertEqual(outcome.selected_transitions, decision.selected_transitions)
        self.assertEqual(
            outcome.selected_next_token_frontier,
            decision.selected_next_token_frontier,
        )
        self.assertEqual(outcome.graph_policy_blockers, ())

        with self.assertRaises(SouthStarError) as raised:
            writer_transitions._WriterTopLevelScheduleOutcome(
                kind=(
                    writer_transitions
                    ._WriterTopLevelScheduleOutcomeKind
                    .SCHEDULED
                ),
            )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.INTERNAL_INVARIANT)

        active_policy, active_decision = self._closure_policy_for_outcome(
            AtomId(1),
        )
        active_outcome = writer_transitions._WriterActiveEmittedScheduleOutcome(
            kind=writer_transitions._WriterActiveEmittedScheduleOutcomeKind.SCHEDULED,
            graph_policy_decision=active_policy,
            schedule_decision=active_decision,
        )

        with self.assertRaises(SouthStarError) as raised:
            writer_transitions._WriterTopLevelScheduleOutcome(
                kind=(
                    writer_transitions
                    ._WriterTopLevelScheduleOutcomeKind
                    .SCHEDULED
                ),
                schedule_decision=decision,
                active_emitted_outcome=active_outcome,
            )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.INTERNAL_INVARIANT)

    def test_top_level_schedule_outcome_validates_active_emitted_scheduled_payload(self) -> None:
        policy, active_decision = self._closure_policy_for_outcome(AtomId(0))
        active_outcome = writer_transitions._WriterActiveEmittedScheduleOutcome(
            kind=writer_transitions._WriterActiveEmittedScheduleOutcomeKind.SCHEDULED,
            graph_policy_decision=policy,
            schedule_decision=active_decision,
        )
        top_decision = writer_transitions._top_level_active_emitted_decision(
            active_decision,
        )

        outcome = writer_transitions._WriterTopLevelScheduleOutcome(
            kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.SCHEDULED,
            schedule_decision=top_decision,
            active_emitted_outcome=active_outcome,
        )

        self.assertIs(outcome.schedule_decision, top_decision)
        self.assertIs(outcome.active_emitted_outcome, active_outcome)
        self.assertIs(outcome.graph_policy_decision, policy)

        other_policy, other_active_decision = self._closure_policy_for_outcome(
            AtomId(1),
        )
        other_active_outcome = writer_transitions._WriterActiveEmittedScheduleOutcome(
            kind=writer_transitions._WriterActiveEmittedScheduleOutcomeKind.SCHEDULED,
            graph_policy_decision=other_policy,
            schedule_decision=other_active_decision,
        )

        with self.assertRaises(SouthStarError) as raised:
            writer_transitions._WriterTopLevelScheduleOutcome(
                kind=(
                    writer_transitions
                    ._WriterTopLevelScheduleOutcomeKind
                    .SCHEDULED
                ),
                schedule_decision=top_decision,
                active_emitted_outcome=other_active_outcome,
            )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.INTERNAL_INVARIANT)

    def test_top_level_schedule_outcome_validates_blocked_active_emitted_payload(self) -> None:
        blocked_outcome = self._blocked_child_active_emitted_outcome(AtomId(0))

        outcome = writer_transitions._WriterTopLevelScheduleOutcome(
            kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.BLOCKED,
            active_emitted_outcome=blocked_outcome,
        )

        self.assertIsNone(outcome.schedule_decision)
        self.assertIs(outcome.active_emitted_outcome, blocked_outcome)
        self.assertEqual(
            outcome.graph_policy_blockers,
            blocked_outcome.graph_policy_blockers,
        )
        self.assertEqual(outcome.selected_transitions, ())
        self.assertEqual(outcome.selected_next_token_frontier, ())

        action = writer_transitions._emit_root_atom_action(AtomId(0))
        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(action,),
            emissions=(),
            surviving_emissions=(),
        )
        top_decision = writer_transitions._top_level_actions_decision(batch)

        with self.assertRaises(SouthStarError) as raised:
            writer_transitions._WriterTopLevelScheduleOutcome(
                kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.BLOCKED,
                schedule_decision=top_decision,
                active_emitted_outcome=blocked_outcome,
            )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.INTERNAL_INVARIANT)

        policy, active_decision = self._closure_policy_for_outcome(AtomId(1))
        scheduled_active_outcome = (
            writer_transitions._WriterActiveEmittedScheduleOutcome(
                kind=(
                    writer_transitions
                    ._WriterActiveEmittedScheduleOutcomeKind
                    .SCHEDULED
                ),
                graph_policy_decision=policy,
                schedule_decision=active_decision,
            )
        )

        with self.assertRaises(SouthStarError) as raised:
            writer_transitions._WriterTopLevelScheduleOutcome(
                kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.BLOCKED,
                active_emitted_outcome=scheduled_active_outcome,
            )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.INTERNAL_INVARIANT)

    def test_top_level_active_emitted_decision_exposes_graph_policy_decision(self) -> None:
        closure_decision, _, _ = self._test_closure_endpoint_decision(
            pair_survives=True,
            open_survives=False,
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .CLOSURE_ENDPOINT
            ),
            active_atom=AtomId(0),
            closure_endpoint_decision=closure_decision,
        )
        active_decision = writer_transitions._active_emitted_closure_decision(
            closure_decision,
            graph_policy_decision=policy,
        )

        top = writer_transitions._top_level_active_emitted_decision(
            active_decision,
        )

        self.assertIs(top.active_emitted_graph_policy_decision, policy)

    def test_top_level_active_emitted_decision_exposes_policy_chosen_surfaces(self) -> None:
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(
                    writer_transitions._WriterScheduledActionEmission(
                        action=open_action,
                        transitions=(),
                    ),
                ),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=9,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )
        child_emission = writer_transitions._WriterScheduledActionEmission(
            action=child_action,
            transitions=(object(),),  # type: ignore[arg-type]
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(child_action,),
            emissions=(child_emission,),
            surviving_emissions=(child_emission,),
        )
        active_decision = writer_transitions._active_emitted_child_decision(
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
            child_batch=child_batch,
            graph_policy_decision=policy,
        )

        top = writer_transitions._top_level_active_emitted_decision(
            active_decision,
        )

        self.assertEqual(
            top.policy_chosen_graph_action_surfaces,
            active_decision.policy_chosen_graph_action_surfaces,
        )
        self.assertEqual(
            top.considered_graph_action_surfaces,
            active_decision.considered_graph_action_surfaces,
        )
        self.assertEqual(
            top.selected_graph_action_surfaces,
            active_decision.selected_graph_action_surfaces,
        )

    def test_top_level_schedule_decision_exposes_active_child_selection(self) -> None:
        active_outcome = self._cyclic_active_child_scheduled_outcome(AtomId(0))
        active_decision = active_outcome.schedule_decision
        top = writer_transitions._top_level_active_emitted_decision(
            active_decision,
        )

        self.assertIs(
            top.considered_active_child_selection_kind,
            writer_transitions._WriterActiveChildSelectionKind.CYCLIC_TREE_ENTRY,
        )
        self.assertIs(
            top.selected_active_child_selection_kind,
            writer_transitions._WriterActiveChildSelectionKind.CYCLIC_TREE_ENTRY,
        )
        self.assertEqual(
            top.considered_cyclic_tree_entry_graph_action_surfaces,
            active_decision.considered_cyclic_tree_entry_graph_action_surfaces,
        )
        self.assertEqual(
            top.selected_cyclic_tree_entry_graph_action_surfaces,
            active_decision.selected_cyclic_tree_entry_graph_action_surfaces,
        )

        root_action = writer_transitions._emit_root_atom_action(AtomId(1))
        root_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(root_action,),
            emissions=(),
            surviving_emissions=(),
        )
        root_top = writer_transitions._top_level_actions_decision(root_batch)

        self.assertIs(
            root_top.considered_active_child_selection_kind,
            writer_transitions._WriterActiveChildSelectionKind.NONE,
        )
        self.assertIs(
            root_top.selected_active_child_selection_kind,
            writer_transitions._WriterActiveChildSelectionKind.NONE,
        )
        self.assertEqual(
            root_top.considered_cyclic_tree_entry_graph_action_surfaces,
            (),
        )
        self.assertEqual(
            root_top.selected_cyclic_tree_entry_graph_action_surfaces,
            (),
        )

    def test_top_level_schedule_outcome_exposes_active_child_selection(self) -> None:
        active_outcome = self._cyclic_active_child_scheduled_outcome(AtomId(0))
        top_decision = writer_transitions._top_level_active_emitted_decision(
            active_outcome.schedule_decision,
        )
        scheduled = writer_transitions._WriterTopLevelScheduleOutcome(
            kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.SCHEDULED,
            schedule_decision=top_decision,
            active_emitted_outcome=active_outcome,
        )

        self.assertIs(
            scheduled.considered_active_child_selection_kind,
            writer_transitions._WriterActiveChildSelectionKind.CYCLIC_TREE_ENTRY,
        )
        self.assertIs(
            scheduled.selected_active_child_selection_kind,
            writer_transitions._WriterActiveChildSelectionKind.CYCLIC_TREE_ENTRY,
        )
        self.assertEqual(
            scheduled.considered_cyclic_tree_entry_graph_action_surfaces,
            top_decision.considered_cyclic_tree_entry_graph_action_surfaces,
        )
        self.assertEqual(
            scheduled.selected_cyclic_tree_entry_graph_action_surfaces,
            top_decision.selected_cyclic_tree_entry_graph_action_surfaces,
        )

        blocked = writer_transitions._WriterTopLevelScheduleOutcome(
            kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.BLOCKED,
            active_emitted_outcome=self._blocked_child_active_emitted_outcome(
                AtomId(0)
            ),
        )

        self.assertIs(
            blocked.considered_active_child_selection_kind,
            writer_transitions._WriterActiveChildSelectionKind.NONE,
        )
        self.assertIs(
            blocked.selected_active_child_selection_kind,
            writer_transitions._WriterActiveChildSelectionKind.NONE,
        )

    def test_scheduled_writer_next_token_frontier_returns_selected_frontier(self) -> None:
        action = writer_transitions._emit_root_atom_action(AtomId(0))
        transition = SimpleNamespace(emitted_text="C")
        emission = writer_transitions._WriterScheduledActionEmission(
            action=action,
            transitions=(transition,),  # type: ignore[arg-type]
        )
        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(action,),
            emissions=(emission,),
            surviving_emissions=(emission,),
        )
        decision = writer_transitions._top_level_actions_decision(batch)
        outcome = writer_transitions._WriterTopLevelScheduleOutcome(
            kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.SCHEDULED,
            schedule_decision=decision,
        )

        with patch(
            "grimace._south_star1.writer_transitions._scheduled_writer_schedule_outcome",
            return_value=outcome,
        ) as schedule:
            frontier = writer_transitions._scheduled_writer_next_token_frontier(
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
            )

        self.assertEqual(frontier, decision.selected_next_token_frontier)
        schedule.assert_called_once()

    def test_legal_writer_schedule_outcome_builds_context_and_delegates(self) -> None:
        action = writer_transitions._emit_root_atom_action(AtomId(0))
        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(action,),
            emissions=(),
            surviving_emissions=(),
        )
        outcome = writer_transitions._WriterTopLevelScheduleOutcome(
            kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.SCHEDULED,
            schedule_decision=writer_transitions._top_level_actions_decision(batch),
        )
        context = object()

        with patch(
            "grimace._south_star1.writer_transitions.build_writer_transition_expansion_context",
            return_value=context,
        ) as build, patch(
            "grimace._south_star1.writer_transitions._scheduled_writer_schedule_outcome",
            return_value=outcome,
        ) as scheduled:
            result = writer_transitions._legal_writer_schedule_outcome(
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
            )

        self.assertIs(result, outcome)
        build.assert_called_once()
        scheduled.assert_called_once()

    def test_scheduled_writer_next_token_frontier_uses_schedule_outcome(self) -> None:
        action = writer_transitions._emit_root_atom_action(AtomId(0))
        transition = SimpleNamespace(emitted_text="C")
        emission = writer_transitions._WriterScheduledActionEmission(
            action=action,
            transitions=(transition,),  # type: ignore[arg-type]
        )
        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(action,),
            emissions=(emission,),
            surviving_emissions=(emission,),
        )
        outcome = writer_transitions._WriterTopLevelScheduleOutcome(
            kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.SCHEDULED,
            schedule_decision=writer_transitions._top_level_actions_decision(batch),
        )

        with patch(
            "grimace._south_star1.writer_transitions._scheduled_writer_schedule_outcome",
            return_value=outcome,
        ) as scheduled, patch(
            "grimace._south_star1.writer_transitions._top_level_schedule_decision",
            side_effect=AssertionError("scheduled frontier used checked decision"),
        ):
            frontier = writer_transitions._scheduled_writer_next_token_frontier(
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
            )

        self.assertEqual(frontier, outcome.selected_next_token_frontier)
        scheduled.assert_called_once()

    def test_scheduled_writer_transitions_use_schedule_outcome(self) -> None:
        action = writer_transitions._emit_root_atom_action(AtomId(0))
        transition = SimpleNamespace(emitted_text="C")
        emission = writer_transitions._WriterScheduledActionEmission(
            action=action,
            transitions=(transition,),  # type: ignore[arg-type]
        )
        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(action,),
            emissions=(emission,),
            surviving_emissions=(emission,),
        )
        outcome = writer_transitions._WriterTopLevelScheduleOutcome(
            kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.SCHEDULED,
            schedule_decision=writer_transitions._top_level_actions_decision(batch),
        )

        with patch(
            "grimace._south_star1.writer_transitions._scheduled_writer_schedule_outcome",
            return_value=outcome,
        ) as scheduled, patch(
            "grimace._south_star1.writer_transitions._top_level_schedule_decision",
            side_effect=AssertionError("scheduled transitions used checked decision"),
        ):
            transitions = writer_transitions._scheduled_writer_transitions(
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
            )

        self.assertEqual(transitions, outcome.selected_transitions)
        scheduled.assert_called_once()

    def test_scheduled_writer_helpers_raise_from_blocked_schedule_outcome(self) -> None:
        active_outcome = self._blocked_child_active_emitted_outcome(AtomId(0))
        outcome = writer_transitions._WriterTopLevelScheduleOutcome(
            kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.BLOCKED,
            active_emitted_outcome=active_outcome,
        )

        with patch(
            "grimace._south_star1.writer_transitions._scheduled_writer_schedule_outcome",
            return_value=outcome,
        ):
            with self.assertRaises(SouthStarError) as raised:
                writer_transitions._scheduled_writer_next_token_frontier(
                    object(),  # type: ignore[arg-type]
                    object(),  # type: ignore[arg-type]
                    object(),  # type: ignore[arg-type]
                )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.UNSUPPORTED_POLICY)

        with patch(
            "grimace._south_star1.writer_transitions._scheduled_writer_schedule_outcome",
            return_value=outcome,
        ):
            with self.assertRaises(SouthStarError) as raised:
                writer_transitions._scheduled_writer_transitions(
                    object(),  # type: ignore[arg-type]
                    object(),  # type: ignore[arg-type]
                    object(),  # type: ignore[arg-type]
                )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.UNSUPPORTED_POLICY)

    def test_legal_writer_next_token_frontier_builds_context_and_delegates(self) -> None:
        scheduled_frontier = (object(),)
        context = object()

        with patch(
            "grimace._south_star1.writer_transitions.build_writer_transition_expansion_context",
            return_value=context,
        ) as build, patch(
            "grimace._south_star1.writer_transitions._scheduled_writer_next_token_frontier",
            return_value=scheduled_frontier,  # type: ignore[arg-type]
        ) as scheduled:
            result = writer_transitions._legal_writer_next_token_frontier(
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
            )

        self.assertEqual(result, scheduled_frontier)
        build.assert_called_once()
        scheduled.assert_called_once()

    def test_closure_endpoint_schedule_surface_projects_pair_before_open_actions(self) -> None:
        pair_label = WriterClosureLabel(value=1, text="1")
        open_label = WriterClosureLabel(value=2, text="2")
        endpoint = WriterOpenClosureEndpoint(
            bond=BondId(1),
            first_atom=AtomId(2),
            second_atom=AtomId(0),
            label=pair_label,
            first_endpoint_text="1",
            first_endpoint_bond_text="",
        )
        closure = WriterClosedClosure(
            bond=BondId(1),
            first_atom=AtomId(2),
            second_atom=AtomId(0),
            label=pair_label,
            first_endpoint_text="1",
            second_endpoint_text="1",
            first_endpoint_bond_text="",
            second_endpoint_bond_text="",
        )
        pair = writer_transitions._WriterClosurePairObligation(
            endpoint=endpoint,
            closure=closure,
        )
        pair_action = writer_transitions._pair_closure_endpoint_action(
            AtomId(0),
            pair,
        )
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(2),
            first_atom=AtomId(0),
            second_atom=AtomId(3),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            AtomId(0),
            open_obligation,
            open_label,
        )

        surface = writer_transitions._WriterClosureEndpointScheduleSurface(
            active_atom=AtomId(0),
            pair_actions=(pair_action,),
            open_actions=(open_action,),
        )

        self.assertEqual(surface.scheduled_actions, (pair_action, open_action))
        self.assertEqual(
            surface.graph_action_surfaces,
            (
                surface.pair_graph_action_surfaces[0],
                surface.open_graph_action_surfaces[0],
            ),
        )
        pair_surface = surface.pair_graph_action_surfaces[0]
        open_surface = surface.open_graph_action_surfaces[0]
        self.assertIs(
            pair_surface.kind,
            writer_transitions._WriterScheduledActionKind.PAIR_CLOSURE_ENDPOINT,
        )
        self.assertIs(
            open_surface.kind,
            writer_transitions._WriterScheduledActionKind.OPEN_CLOSURE_ENDPOINT,
        )
        self.assertIs(pair_surface.closure_label, pair_label)
        self.assertIs(open_surface.closure_label, open_label)
        self.assertEqual(open_surface.attachment_id, 7)
        self.assertIs(open_surface.owner_kind, WriterBoundaryOwnerKind.ACTIVE_ATOM)

    def test_closure_endpoint_selection_kind_from_graph_action_surfaces(self) -> None:
        active_atom = AtomId(0)
        pair_surface = (
            writer_transitions._scheduled_graph_action_surface(
                self._test_closure_pair_action(active_atom)
            )
        )
        open_surface = (
            writer_transitions._scheduled_graph_action_surface(
                self._test_closure_open_action(active_atom)
            )
        )
        non_closure_surface = (
            writer_transitions._scheduled_graph_action_surface(
                writer_transitions._finish_active_action(active_atom)
            )
        )

        cases = (
            (
                (),
                writer_transitions._WriterClosureEndpointSelectionKind.NONE,
            ),
            (
                (pair_surface,),
                (
                    writer_transitions
                    ._WriterClosureEndpointSelectionKind
                    .CLOSURE_PAIR
                ),
            ),
            (
                (open_surface,),
                (
                    writer_transitions
                    ._WriterClosureEndpointSelectionKind
                    .CLOSURE_OPEN
                ),
            ),
            (
                (pair_surface, open_surface),
                (
                    writer_transitions
                    ._WriterClosureEndpointSelectionKind
                    .PAIR_AND_OPEN
                ),
            ),
            (
                (non_closure_surface,),
                writer_transitions._WriterClosureEndpointSelectionKind.NONE,
            ),
        )

        for surfaces, expected in cases:
            with self.subTest(surfaces=surfaces):
                self.assertIs(
                    (
                        writer_transitions
                        ._closure_endpoint_selection_kind_from_graph_action_surfaces(
                            surfaces,
                        )
                    ),
                    expected,
                )

    def test_closure_endpoint_schedule_decision_exposes_considered_and_selected_surfaces(self) -> None:
        label = WriterClosureLabel(value=1, text="1")
        endpoint = WriterOpenClosureEndpoint(
            bond=BondId(1),
            first_atom=AtomId(2),
            second_atom=AtomId(0),
            label=label,
            first_endpoint_text="1",
            first_endpoint_bond_text="",
        )
        closure = WriterClosedClosure(
            bond=BondId(1),
            first_atom=AtomId(2),
            second_atom=AtomId(0),
            label=label,
            first_endpoint_text="1",
            second_endpoint_text="1",
            first_endpoint_bond_text="",
            second_endpoint_bond_text="",
        )
        pair = writer_transitions._WriterClosurePairObligation(
            endpoint=endpoint,
            closure=closure,
        )
        pair_action = writer_transitions._pair_closure_endpoint_action(
            AtomId(0),
            pair,
        )
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(2),
            first_atom=AtomId(0),
            second_atom=AtomId(3),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            AtomId(0),
            open_obligation,
            label,
        )
        surface = writer_transitions._WriterClosureEndpointScheduleSurface(
            active_atom=AtomId(0),
            pair_actions=(pair_action,),
            open_actions=(open_action,),
        )
        pair_emission = writer_transitions._WriterScheduledActionEmission(
            action=pair_action,
            transitions=(object(),),  # type: ignore[arg-type]
        )
        open_emission = writer_transitions._WriterScheduledActionEmission(
            action=open_action,
            transitions=(),
        )
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(pair_action,),
            emissions=(pair_emission,),
            surviving_emissions=(pair_emission,),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(open_action,),
            emissions=(open_emission,),
            surviving_emissions=(),
        )
        decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=pair_batch,
            open_batch=open_batch,
            surviving_emissions=(pair_emission,),
            schedule_surface=surface,
        )

        self.assertIs(decision.schedule_surface, surface)
        self.assertEqual(
            decision.considered_graph_action_surfaces,
            surface.graph_action_surfaces,
        )
        self.assertEqual(
            decision.selected_graph_action_surfaces,
            (pair_emission.graph_action_surface,),
        )

    def test_closure_endpoint_schedule_decision_exposes_considered_and_selected_closure_families(self) -> None:
        decision, pair_emission, open_emission = (
            self._test_closure_endpoint_decision(
                pair_survives=True,
                open_survives=False,
            )
        )

        self.assertIs(
            decision.considered_closure_endpoint_selection_kind,
            (
                writer_transitions
                ._WriterClosureEndpointSelectionKind
                .PAIR_AND_OPEN
            ),
        )
        self.assertIs(
            decision.selected_closure_endpoint_selection_kind,
            (
                writer_transitions
                ._WriterClosureEndpointSelectionKind
                .CLOSURE_PAIR
            ),
        )
        self.assertEqual(
            decision.considered_closure_pair_graph_action_surfaces,
            (pair_emission.graph_action_surface,),
        )
        self.assertEqual(
            decision.considered_closure_open_graph_action_surfaces,
            (open_emission.graph_action_surface,),
        )
        self.assertEqual(
            decision.selected_closure_pair_graph_action_surfaces,
            (pair_emission.graph_action_surface,),
        )
        self.assertEqual(
            decision.selected_closure_open_graph_action_surfaces,
            (),
        )

    def test_closure_endpoint_schedule_decision_classifies_selected_closure_open(self) -> None:
        decision, _, open_emission = self._test_closure_endpoint_decision(
            pair_survives=False,
            open_survives=True,
        )

        self.assertIs(
            decision.selected_closure_endpoint_selection_kind,
            (
                writer_transitions
                ._WriterClosureEndpointSelectionKind
                .CLOSURE_OPEN
            ),
        )
        self.assertEqual(
            decision.selected_closure_open_graph_action_surfaces,
            (open_emission.graph_action_surface,),
        )
        self.assertEqual(
            decision.selected_closure_pair_graph_action_surfaces,
            (),
        )

    def test_closure_endpoint_schedule_decision_classifies_pair_and_open_survivors(self) -> None:
        decision, pair_emission, open_emission = (
            self._test_closure_endpoint_decision(
                pair_survives=True,
                open_survives=True,
            )
        )

        self.assertIs(
            decision.selected_closure_endpoint_selection_kind,
            (
                writer_transitions
                ._WriterClosureEndpointSelectionKind
                .PAIR_AND_OPEN
            ),
        )
        self.assertEqual(
            decision.selected_closure_pair_graph_action_surfaces,
            (pair_emission.graph_action_surface,),
        )
        self.assertEqual(
            decision.selected_closure_open_graph_action_surfaces,
            (open_emission.graph_action_surface,),
        )

    def test_closure_endpoint_schedule_decision_exposes_selection_booleans(self) -> None:
        pair_selected, _, _ = self._test_closure_endpoint_decision(
            pair_survives=True,
            open_survives=False,
        )

        self.assertTrue(pair_selected.considered_closure_endpoint_available)
        self.assertTrue(pair_selected.selected_closure_endpoint_survived)
        self.assertIs(
            pair_selected.considered_closure_endpoint_selection_kind,
            (
                writer_transitions
                ._WriterClosureEndpointSelectionKind
                .PAIR_AND_OPEN
            ),
        )
        self.assertIs(
            pair_selected.selected_closure_endpoint_selection_kind,
            (
                writer_transitions
                ._WriterClosureEndpointSelectionKind
                .CLOSURE_PAIR
            ),
        )

        open_action = self._test_closure_open_action(AtomId(0))
        open_emission = writer_transitions._WriterScheduledActionEmission(
            action=open_action,
            transitions=(),
        )
        open_dead = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(open_emission,),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=AtomId(0),
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )

        self.assertTrue(open_dead.considered_closure_endpoint_available)
        self.assertFalse(open_dead.selected_closure_endpoint_survived)
        self.assertIs(
            open_dead.selected_closure_endpoint_selection_kind,
            writer_transitions._WriterClosureEndpointSelectionKind.NONE,
        )

    def test_active_child_selection_kind_from_graph_action_surfaces(self) -> None:
        active_atom = AtomId(0)
        finish = writer_transitions._scheduled_graph_action_surface(
            writer_transitions._finish_active_action(active_atom)
        )
        generic_tree = writer_transitions._scheduled_graph_action_surface(
            self._test_child_action(active_atom)
        )
        acyclic_tree = writer_transitions._scheduled_graph_action_surface(
            self._test_child_action(
                active_atom,
                attachment_action_kind=(
                    WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY
                ),
            )
        )
        cyclic_tree = writer_transitions._scheduled_graph_action_surface(
            self._test_child_action(
                active_atom,
                attachment_action_kind=(
                    WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
                ),
            )
        )
        root = writer_transitions._scheduled_graph_action_surface(
            writer_transitions._emit_root_atom_action(active_atom)
        )
        closure = writer_transitions._scheduled_graph_action_surface(
            self._test_closure_open_action(active_atom)
        )

        cases = (
            ((), writer_transitions._WriterActiveChildSelectionKind.NONE),
            (
                (finish,),
                writer_transitions._WriterActiveChildSelectionKind.FINISH_ACTIVE,
            ),
            (
                (generic_tree,),
                writer_transitions._WriterActiveChildSelectionKind.TREE_ENTRY,
            ),
            (
                (acyclic_tree,),
                writer_transitions._WriterActiveChildSelectionKind.ACYCLIC_TREE_ENTRY,
            ),
            (
                (cyclic_tree,),
                writer_transitions._WriterActiveChildSelectionKind.CYCLIC_TREE_ENTRY,
            ),
            (
                (finish, cyclic_tree),
                writer_transitions._WriterActiveChildSelectionKind.MIXED,
            ),
            (
                (root, closure),
                writer_transitions._WriterActiveChildSelectionKind.NONE,
            ),
        )

        for surfaces, expected in cases:
            with self.subTest(expected=expected):
                self.assertIs(
                    (
                        writer_transitions
                        ._active_child_selection_kind_from_graph_action_surfaces(
                            surfaces,
                        )
                    ),
                    expected,
                )

    def test_active_child_schedule_surface_exposes_considered_child_family(self) -> None:
        active_atom = AtomId(0)
        cyclic_action = self._test_child_action(
            active_atom,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        cyclic_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(cyclic_action.child_obligation,),
            scheduled_actions=(cyclic_action,),
        )

        self.assertIs(
            cyclic_surface.considered_active_child_selection_kind,
            writer_transitions._WriterActiveChildSelectionKind.CYCLIC_TREE_ENTRY,
        )
        self.assertEqual(
            cyclic_surface.considered_cyclic_tree_entry_graph_action_surfaces,
            cyclic_surface.graph_action_surfaces,
        )
        self.assertEqual(
            cyclic_surface.considered_finish_active_graph_action_surfaces,
            (),
        )

        finish_action = writer_transitions._finish_active_action(active_atom)
        finish_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(),
            scheduled_actions=(finish_action,),
        )

        self.assertIs(
            finish_surface.considered_active_child_selection_kind,
            writer_transitions._WriterActiveChildSelectionKind.FINISH_ACTIVE,
        )
        self.assertEqual(
            finish_surface.considered_finish_active_graph_action_surfaces,
            finish_surface.graph_action_surfaces,
        )

    def test_closure_endpoint_schedule_decision_exposes_residual_policy_emission_groups(self) -> None:
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(2),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        surface = writer_transitions._WriterClosureEndpointScheduleSurface(
            active_atom=active_atom,
            pair_actions=(),
            open_actions=(open_action,),
        )
        closure_open_emission = writer_transitions._WriterScheduledActionEmission(
            action=open_action,
            transitions=(),
        )
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(open_action,),
            emissions=(closure_open_emission,),
            surviving_emissions=(),
        )
        decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=pair_batch,
            open_batch=open_batch,
            surviving_emissions=(),
            schedule_surface=surface,
        )

        considered = decision.considered_residual_attachment_policy_emission_groups
        selected = decision.selected_residual_attachment_policy_emission_groups

        self.assertEqual(len(considered), 1)
        self.assertEqual(
            considered[0].closure_open_emissions,
            (closure_open_emission,),
        )
        self.assertEqual(considered[0].surviving_closure_open_emissions, ())
        self.assertEqual(selected, ())

    def test_closure_endpoint_schedule_decision_separates_pair_and_open_batches(self) -> None:
        prepared = object()
        state = SimpleNamespace(
            active=SimpleNamespace(atom=AtomId(0)),
            ring_state=object(),
        )
        context = object()
        label = WriterClosureLabel(value=1, text="1")
        endpoint = WriterOpenClosureEndpoint(
            bond=BondId(1),
            first_atom=AtomId(2),
            second_atom=AtomId(0),
            label=label,
            first_endpoint_text="1",
            first_endpoint_bond_text="",
        )
        closure = WriterClosedClosure(
            bond=BondId(1),
            first_atom=AtomId(2),
            second_atom=AtomId(0),
            label=label,
            first_endpoint_text="1",
            second_endpoint_text="1",
            first_endpoint_bond_text="",
            second_endpoint_bond_text="",
        )
        pair = writer_transitions._WriterClosurePairObligation(
            endpoint=endpoint,
            closure=closure,
        )
        pair_action = writer_transitions._pair_closure_endpoint_action(
            AtomId(0),
            pair,
        )
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(2),
            first_atom=AtomId(0),
            second_atom=AtomId(3),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            AtomId(0),
            open_obligation,
            label,
        )
        pair_emission = writer_transitions._WriterScheduledActionEmission(
            action=pair_action,
            transitions=(object(),),  # type: ignore[arg-type]
        )
        open_emission = writer_transitions._WriterScheduledActionEmission(
            action=open_action,
            transitions=(object(),),  # type: ignore[arg-type]
        )
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(pair_action,),
            emissions=(pair_emission,),
            surviving_emissions=(pair_emission,),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(open_action,),
            emissions=(open_emission,),
            surviving_emissions=(open_emission,),
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_pair_scheduled_actions",
            return_value=(pair_action,),
        ) as pair_actions, patch(
            "grimace._south_star1.writer_transitions._available_closure_labels_for_open",
            return_value=(label,),
        ) as available_labels, patch(
            "grimace._south_star1.writer_transitions._closure_open_scheduled_actions",
            return_value=(open_action,),
        ) as open_actions, patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            side_effect=(pair_batch, open_batch),
        ) as emission_batch:
            decision = writer_transitions._closure_endpoint_schedule_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
            )

        self.assertIs(decision.pair_batch, pair_batch)
        self.assertIs(decision.open_batch, open_batch)
        self.assertIsNotNone(decision.schedule_surface)
        self.assertEqual(decision.schedule_surface.pair_actions, decision.pair_batch.actions)
        self.assertEqual(decision.schedule_surface.open_actions, decision.open_batch.actions)
        self.assertEqual(
            decision.considered_graph_action_surfaces,
            (
                *decision.pair_batch.graph_action_surfaces,
                *decision.open_batch.graph_action_surfaces,
            ),
        )
        self.assertEqual(
            decision.surviving_emissions,
            (pair_emission, open_emission),
        )
        pair_actions.assert_called_once_with(state, AtomId(0))
        available_labels.assert_called_once_with(prepared, state.ring_state)
        open_actions.assert_called_once_with(context, AtomId(0), (label,))
        self.assertEqual(emission_batch.call_count, 2)
        self.assertEqual(
            emission_batch.call_args_list[0].args,
            (prepared, state, context, (pair_action,)),
        )
        self.assertEqual(
            emission_batch.call_args_list[1].args,
            (prepared, state, context, (open_action,)),
        )

    def test_closure_endpoint_schedule_decision_skips_open_actions_without_labels(self) -> None:
        prepared = object()
        state = SimpleNamespace(
            active=SimpleNamespace(atom=AtomId(0)),
            ring_state=object(),
        )
        context = object()
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_pair_scheduled_actions",
            return_value=(),
        ), patch(
            "grimace._south_star1.writer_transitions._available_closure_labels_for_open",
            return_value=(),
        ), patch(
            "grimace._south_star1.writer_transitions._closure_open_scheduled_actions",
            side_effect=AssertionError("open actions should not be built without labels"),
        ), patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            side_effect=(pair_batch, open_batch),
        ) as emission_batch:
            decision = writer_transitions._closure_endpoint_schedule_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
            )

        self.assertIs(decision.pair_batch, pair_batch)
        self.assertIs(decision.open_batch, open_batch)
        self.assertEqual(decision.surviving_emissions, ())
        self.assertEqual(emission_batch.call_count, 2)
        self.assertEqual(
            emission_batch.call_args_list[1].args,
            (prepared, state, context, ()),
        )

    def test_closure_open_obligations_carry_action_incidence_metadata(self) -> None:
        action = SimpleNamespace(
            attachment_id=7,
            kind=WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY,
        )
        attachment = SimpleNamespace(
            attachment_id=7,
            boundary=(
                SimpleNamespace(
                    bond=BondId(1),
                    written_atom=AtomId(0),
                    residual_atom=AtomId(2),
                    owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
                ),
            ),
        )
        summary = SimpleNamespace(
            attachments=SimpleNamespace(attachments=(attachment,)),
            attachment_actions=(action,),
        )
        context = SimpleNamespace(
            graph=SimpleNamespace(
                residual_summary=summary,
            ),
        )

        obligations = writer_transitions._closure_open_obligations_from_context(
            context,  # type: ignore[arg-type]
            AtomId(0),
        )

        self.assertEqual(len(obligations), 1)
        self.assertEqual(obligations[0].bond, BondId(1))
        self.assertEqual(obligations[0].first_atom, AtomId(0))
        self.assertEqual(obligations[0].second_atom, AtomId(2))
        self.assertEqual(obligations[0].attachment_id, 7)
        self.assertIs(
            obligations[0].attachment_action_kind,
            WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY,
        )
        self.assertIs(
            obligations[0].owner_kind,
            WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )

    def test_closure_endpoint_combined_batch_preserves_pair_before_open(self) -> None:
        pair_action = object()
        open_action = object()
        pair_emission = object()
        open_emission = object()
        pair_survivor = object()
        open_survivor = object()
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(pair_action,),  # type: ignore[arg-type]
            emissions=(pair_emission,),  # type: ignore[arg-type]
            surviving_emissions=(pair_survivor,),  # type: ignore[arg-type]
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(open_action,),  # type: ignore[arg-type]
            emissions=(open_emission,),  # type: ignore[arg-type]
            surviving_emissions=(open_survivor,),  # type: ignore[arg-type]
        )
        decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=pair_batch,
            open_batch=open_batch,
            surviving_emissions=(pair_survivor, open_survivor),  # type: ignore[arg-type]
        )

        combined = writer_transitions._closure_endpoint_combined_batch(decision)

        self.assertEqual(combined.actions, (pair_action, open_action))
        self.assertEqual(combined.emissions, (pair_emission, open_emission))
        self.assertEqual(
            combined.surviving_emissions,
            (pair_survivor, open_survivor),
        )

    def test_top_level_schedule_decision_selects_top_level_batch(self) -> None:
        prepared = object()
        state = object()
        context = object()
        action = object()
        emission = object()
        survivor = object()

        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(action,),  # type: ignore[arg-type]
            emissions=(emission,),  # type: ignore[arg-type]
            surviving_emissions=(survivor,),  # type: ignore[arg-type]
        )

        with patch(
            "grimace._south_star1.writer_transitions._top_level_scheduled_actions",
            return_value=(action,),
        ) as top_level_actions, patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            return_value=batch,
        ) as emission_batch, patch(
            "grimace._south_star1.writer_transitions._active_emitted_schedule_outcome",
            side_effect=AssertionError("active-emitted decision should not run"),
        ):
            decision = writer_transitions._top_level_schedule_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
            )

        self.assertIs(
            decision.kind,
            writer_transitions._WriterTopLevelScheduleDecisionKind.TOP_LEVEL_ACTIONS,
        )
        self.assertIs(decision.selected_batch, batch)
        self.assertIs(decision.top_level_batch, batch)
        self.assertIsNone(decision.active_emitted_decision)
        top_level_actions.assert_called_once_with(state)
        emission_batch.assert_called_once_with(
            prepared,
            state,
            context,
            (action,),
        )

    def test_top_level_schedule_decision_keeps_zero_survivor_top_level_batch(self) -> None:
        prepared = object()
        state = object()
        context = object()
        action = object()
        emission = object()

        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(action,),  # type: ignore[arg-type]
            emissions=(emission,),  # type: ignore[arg-type]
            surviving_emissions=(),
        )

        with patch(
            "grimace._south_star1.writer_transitions._top_level_scheduled_actions",
            return_value=(action,),
        ), patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            return_value=batch,
        ), patch(
            "grimace._south_star1.writer_transitions._active_emitted_schedule_outcome",
            side_effect=AssertionError("active-emitted decision should not run"),
        ):
            decision = writer_transitions._top_level_schedule_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
            )

        self.assertIs(
            decision.kind,
            writer_transitions._WriterTopLevelScheduleDecisionKind.TOP_LEVEL_ACTIONS,
        )
        self.assertIs(decision.selected_batch, batch)
        self.assertEqual(decision.selected_batch.surviving_emissions, ())

    def test_top_level_schedule_decision_selects_active_emitted_when_no_top_level_actions(self) -> None:
        prepared = object()
        context = object()
        state = SimpleNamespace(
            active=SimpleNamespace(
                atom=AtomId(4),
            ),
        )

        selected_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_endpoint_decision = (
            writer_transitions._WriterClosureEndpointScheduleDecision(
                pair_batch=selected_batch,
                open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                    actions=(),
                    emissions=(),
                    surviving_emissions=(),
                ),
                surviving_emissions=(),
            )
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=AtomId(4),
            blockers=(),
            child_obligations=(),
            scheduled_actions=selected_batch.actions,
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD
            ),
            active_atom=AtomId(4),
            child_schedule_surface=child_surface,
            closure_endpoint_decision=closure_endpoint_decision,
        )
        active_decision = writer_transitions._active_emitted_child_decision(
            closure_endpoint_decision=closure_endpoint_decision,
            child_schedule_surface=child_surface,
            child_batch=selected_batch,
            graph_policy_decision=policy,
        )
        active_outcome = writer_transitions._WriterActiveEmittedScheduleOutcome(
            kind=writer_transitions._WriterActiveEmittedScheduleOutcomeKind.SCHEDULED,
            graph_policy_decision=policy,
            schedule_decision=active_decision,
        )

        with patch(
            "grimace._south_star1.writer_transitions._top_level_scheduled_actions",
            return_value=(),
        ) as top_level_actions, patch(
            "grimace._south_star1.writer_transitions._active_emitted_schedule_outcome",
            return_value=active_outcome,
        ) as active_emitted_outcome:
            decision = writer_transitions._top_level_schedule_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
            )

        self.assertIs(
            decision.kind,
            writer_transitions._WriterTopLevelScheduleDecisionKind.ACTIVE_EMITTED,
        )
        self.assertIs(decision.selected_batch, selected_batch)
        self.assertIsNone(decision.top_level_batch)
        self.assertIs(decision.active_emitted_decision, active_decision)
        top_level_actions.assert_called_once_with(state)
        active_emitted_outcome.assert_called_once_with(
            prepared,
            state,
            context,
            AtomId(4),
        )

    def test_top_level_schedule_outcome_keeps_top_level_action_priority(self) -> None:
        prepared = object()
        state = object()
        context = object()
        action = writer_transitions._emit_root_atom_action(AtomId(0))
        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(action,),
            emissions=(),
            surviving_emissions=(),
        )

        with patch(
            "grimace._south_star1.writer_transitions._top_level_scheduled_actions",
            return_value=(action,),
        ) as top_level_actions, patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            return_value=batch,
        ) as emission_batch, patch(
            "grimace._south_star1.writer_transitions._active_emitted_schedule_outcome",
            side_effect=AssertionError("active-emitted outcome should not run"),
        ) as active_emitted_outcome:
            outcome = writer_transitions._top_level_schedule_outcome(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
            )

        self.assertIs(
            outcome.kind,
            writer_transitions._WriterTopLevelScheduleOutcomeKind.SCHEDULED,
        )
        self.assertIs(
            outcome.schedule_decision.kind,
            writer_transitions._WriterTopLevelScheduleDecisionKind.TOP_LEVEL_ACTIONS,
        )
        top_level_actions.assert_called_once_with(state)
        emission_batch.assert_called_once_with(
            prepared,
            state,
            context,
            (action,),
        )
        active_emitted_outcome.assert_not_called()

    def test_top_level_schedule_outcome_wraps_active_emitted_scheduled_outcome(self) -> None:
        prepared = object()
        context = object()
        active_atom = AtomId(4)
        state = SimpleNamespace(
            active=SimpleNamespace(
                atom=active_atom,
            ),
        )
        policy, active_decision = self._closure_policy_for_outcome(active_atom)
        active_outcome = writer_transitions._WriterActiveEmittedScheduleOutcome(
            kind=writer_transitions._WriterActiveEmittedScheduleOutcomeKind.SCHEDULED,
            graph_policy_decision=policy,
            schedule_decision=active_decision,
        )

        with patch(
            "grimace._south_star1.writer_transitions._top_level_scheduled_actions",
            return_value=(),
        ), patch(
            "grimace._south_star1.writer_transitions._active_emitted_schedule_outcome",
            return_value=active_outcome,
        ) as active_emitted_outcome:
            outcome = writer_transitions._top_level_schedule_outcome(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
            )

        self.assertIs(
            outcome.kind,
            writer_transitions._WriterTopLevelScheduleOutcomeKind.SCHEDULED,
        )
        self.assertIs(outcome.active_emitted_outcome, active_outcome)
        self.assertIs(
            outcome.schedule_decision.kind,
            writer_transitions._WriterTopLevelScheduleDecisionKind.ACTIVE_EMITTED,
        )
        self.assertIs(
            outcome.schedule_decision.active_emitted_decision,
            active_decision,
        )
        active_emitted_outcome.assert_called_once_with(
            prepared,
            state,
            context,
            active_atom,
        )

    def test_top_level_schedule_outcome_wraps_active_emitted_blocked_outcome(self) -> None:
        prepared = object()
        context = object()
        active_atom = AtomId(4)
        state = SimpleNamespace(
            active=SimpleNamespace(
                atom=active_atom,
            ),
        )
        active_outcome = self._blocked_child_active_emitted_outcome(active_atom)

        with patch(
            "grimace._south_star1.writer_transitions._top_level_scheduled_actions",
            return_value=(),
        ), patch(
            "grimace._south_star1.writer_transitions._active_emitted_schedule_outcome",
            return_value=active_outcome,
        ):
            outcome = writer_transitions._top_level_schedule_outcome(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
            )

        self.assertIs(
            outcome.kind,
            writer_transitions._WriterTopLevelScheduleOutcomeKind.BLOCKED,
        )
        self.assertIsNone(outcome.schedule_decision)
        self.assertIs(outcome.active_emitted_outcome, active_outcome)
        self.assertEqual(
            outcome.graph_policy_blockers,
            active_outcome.graph_policy_blockers,
        )

    def test_top_level_schedule_decision_raises_from_blocked_top_level_outcome(self) -> None:
        active_atom = AtomId(4)
        outcome = self._blocked_child_active_emitted_outcome(active_atom)
        top_outcome = writer_transitions._WriterTopLevelScheduleOutcome(
            kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.BLOCKED,
            active_emitted_outcome=outcome,
        )

        with patch(
            "grimace._south_star1.writer_transitions._top_level_schedule_outcome",
            return_value=top_outcome,
        ):
            with self.assertRaises(SouthStarError) as raised:
                writer_transitions._top_level_schedule_decision(
                    object(),  # type: ignore[arg-type]
                    object(),  # type: ignore[arg-type]
                    object(),  # type: ignore[arg-type]
                )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.UNSUPPORTED_POLICY)

    def test_top_level_schedule_decision_returns_scheduled_outcome_decision(self) -> None:
        action = writer_transitions._emit_root_atom_action(AtomId(0))
        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(action,),
            emissions=(),
            surviving_emissions=(),
        )
        decision = writer_transitions._top_level_actions_decision(batch)
        outcome = writer_transitions._WriterTopLevelScheduleOutcome(
            kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.SCHEDULED,
            schedule_decision=decision,
        )

        with patch(
            "grimace._south_star1.writer_transitions._top_level_schedule_outcome",
            return_value=outcome,
        ):
            result = writer_transitions._top_level_schedule_decision(
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
            )

        self.assertIs(result, decision)

    def test_scheduled_writer_transitions_flattens_top_level_decision_survivors(self) -> None:
        prepared = object()
        state = object()
        context = object()
        action = writer_transitions._finish_active_action(AtomId(0))
        transition = SimpleNamespace(emitted_text="")
        emission = writer_transitions._WriterScheduledActionEmission(
            action=action,
            transitions=(transition,),  # type: ignore[arg-type]
        )

        selected_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(action,),
            emissions=(emission,),
            surviving_emissions=(emission,),
        )
        decision = writer_transitions._WriterTopLevelScheduleDecision(
            kind=writer_transitions._WriterTopLevelScheduleDecisionKind.TOP_LEVEL_ACTIONS,
            selected_batch=selected_batch,
            top_level_batch=selected_batch,
        )
        outcome = writer_transitions._WriterTopLevelScheduleOutcome(
            kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.SCHEDULED,
            schedule_decision=decision,
        )

        with patch(
            "grimace._south_star1.writer_transitions._scheduled_writer_schedule_outcome",
            return_value=outcome,
        ) as schedule_outcome:
            result = writer_transitions._scheduled_writer_transitions(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
            )

        self.assertEqual(result, decision.selected_transitions)
        schedule_outcome.assert_called_once_with(
            prepared,
            state,
            context,
        )

    def test_scheduled_writer_transitions_does_not_fall_through_when_top_level_actions_do_not_survive(self) -> None:
        prepared = object()
        context = object()
        state = SimpleNamespace(
            obligations=SimpleNamespace(
                pending_entry=None,
            ),
            active=SimpleNamespace(
                atom=AtomId(0),
                atom_emitted=False,
            ),
        )
        action = object()
        emission = object()
        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(action,),  # type: ignore[arg-type]
            emissions=(emission,),  # type: ignore[arg-type]
            surviving_emissions=(),
        )

        with patch(
            "grimace._south_star1.writer_transitions._top_level_scheduled_actions",
            return_value=(action,),
        ) as top_level_actions, patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            return_value=batch,
        ) as emission_batch, patch(
            "grimace._south_star1.writer_transitions._transitions_from_scheduled_action_emissions",
            return_value=(),
        ) as flatten_emissions, patch(
            "grimace._south_star1.writer_transitions._active_emitted_transitions",
            side_effect=AssertionError("active-emitted path should not run"),
        ):
            result = writer_transitions._scheduled_writer_transitions(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
            )

        self.assertEqual(result, ())
        top_level_actions.assert_called_once_with(state)
        emission_batch.assert_called_once_with(
            prepared,
            state,
            context,
            (action,),
        )
        flatten_emissions.assert_called_once_with(())

    def test_scheduled_writer_transitions_falls_through_after_empty_top_level_actions(self) -> None:
        prepared = object()
        context = object()
        state = SimpleNamespace(
            active=SimpleNamespace(
                atom=AtomId(4),
            ),
        )
        closure_endpoint_decision, survivor, _ = (
            self._test_closure_endpoint_decision(
                active_atom=AtomId(4),
                pair_survives=True,
                open_survives=False,
            )
        )
        transition = object()
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .CLOSURE_ENDPOINT
            ),
            active_atom=AtomId(4),
            closure_endpoint_decision=closure_endpoint_decision,
        )
        active_decision = writer_transitions._active_emitted_closure_decision(
            closure_endpoint_decision,
            graph_policy_decision=policy,
        )
        active_outcome = writer_transitions._WriterActiveEmittedScheduleOutcome(
            kind=writer_transitions._WriterActiveEmittedScheduleOutcomeKind.SCHEDULED,
            graph_policy_decision=policy,
            schedule_decision=active_decision,
        )

        with patch(
            "grimace._south_star1.writer_transitions._top_level_scheduled_actions",
            return_value=(),
        ) as top_level_actions, patch(
            "grimace._south_star1.writer_transitions._active_emitted_schedule_outcome",
            return_value=active_outcome,
        ) as active_emitted_outcome, patch(
            "grimace._south_star1.writer_transitions._transitions_from_scheduled_action_emissions",
            return_value=(transition,),
        ) as flatten_emissions:
            result = writer_transitions._scheduled_writer_transitions(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
            )

        self.assertEqual(result, (transition,))
        top_level_actions.assert_called_once_with(state)
        active_emitted_outcome.assert_called_once_with(
            prepared,
            state,
            context,
            AtomId(4),
        )
        flatten_emissions.assert_called_once_with((survivor,))

    def test_scheduled_action_emissions_preserve_action_identity(self) -> None:
        prepared = object()
        state = object()
        context = object()
        first_action = writer_transitions._finish_active_action(AtomId(0))
        second_action = writer_transitions._emit_root_atom_action(AtomId(1))
        transition = object()

        with patch(
            "grimace._south_star1.writer_transitions._transitions_from_scheduled_action",
            side_effect=((transition,), ()),
        ) as emit_action:
            emissions = writer_transitions._scheduled_action_emissions(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                (first_action, second_action),
            )

        self.assertEqual(len(emissions), 2)
        self.assertIs(emissions[0].action, first_action)
        self.assertEqual(emissions[0].transitions, (transition,))
        self.assertIs(emissions[1].action, second_action)
        self.assertEqual(emissions[1].transitions, ())
        self.assertEqual(emit_action.call_count, 2)
        self.assertEqual(
            emit_action.call_args_list[0].args,
            (prepared, state, context, first_action),
        )
        self.assertEqual(
            emit_action.call_args_list[1].args,
            (prepared, state, context, second_action),
        )

    def test_scheduled_action_emission_exposes_graph_action_surface(self) -> None:
        child = writer_transitions._WriterChildObligation(
            bond=BondId(1),
            child=AtomId(2),
            boundary_atom=AtomId(0),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY
            ),
        )
        action = writer_transitions._enter_inline_child_action(AtomId(0), child)
        transition = object()
        emission = writer_transitions._WriterScheduledActionEmission(
            action=action,
            transitions=(transition,),  # type: ignore[arg-type]
        )

        surface = emission.graph_action_surface

        self.assertIs(
            surface.kind,
            writer_transitions._WriterScheduledActionKind.ENTER_INLINE_CHILD,
        )
        self.assertEqual(surface.active_atom, AtomId(0))
        self.assertEqual(surface.bond, BondId(1))
        self.assertEqual(surface.partner_atom, AtomId(2))
        self.assertEqual(surface.boundary_atom, AtomId(0))
        self.assertEqual(surface.attachment_id, 7)
        self.assertTrue(emission.survived)
        self.assertEqual(emission.transitions, (transition,))

    def test_zero_transition_scheduled_action_emission_still_exposes_surface(self) -> None:
        action = writer_transitions._finish_active_action(AtomId(3))
        emission = writer_transitions._WriterScheduledActionEmission(
            action=action,
            transitions=(),
        )

        self.assertFalse(emission.survived)
        self.assertIs(
            emission.graph_action_surface.kind,
            writer_transitions._WriterScheduledActionKind.FINISH_ACTIVE,
        )
        self.assertEqual(emission.graph_action_surface.active_atom, AtomId(3))

    def test_scheduled_action_emission_batch_exposes_all_and_surviving_surfaces(self) -> None:
        first_action = writer_transitions._finish_active_action(AtomId(0))
        second_action = writer_transitions._emit_root_atom_action(AtomId(1))
        transition = object()
        first_emission = writer_transitions._WriterScheduledActionEmission(
            action=first_action,
            transitions=(transition,),  # type: ignore[arg-type]
        )
        second_emission = writer_transitions._WriterScheduledActionEmission(
            action=second_action,
            transitions=(),
        )
        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(first_action, second_action),
            emissions=(first_emission, second_emission),
            surviving_emissions=(first_emission,),
        )

        self.assertEqual(
            batch.graph_action_surfaces,
            (
                first_emission.graph_action_surface,
                second_emission.graph_action_surface,
            ),
        )
        self.assertEqual(
            batch.surviving_graph_action_surfaces,
            (first_emission.graph_action_surface,),
        )

    def test_scheduled_action_batch_exposes_surviving_next_token_frontier(self) -> None:
        first_action = writer_transitions._finish_active_action(AtomId(0))
        second_action = writer_transitions._emit_root_atom_action(AtomId(1))
        first_transition = SimpleNamespace(emitted_text="C")
        second_transition = SimpleNamespace(emitted_text="N")
        third_transition = SimpleNamespace(emitted_text="C")
        first_emission = writer_transitions._WriterScheduledActionEmission(
            action=first_action,
            transitions=(
                first_transition,  # type: ignore[arg-type]
                second_transition,  # type: ignore[arg-type]
            ),
        )
        second_emission = writer_transitions._WriterScheduledActionEmission(
            action=second_action,
            transitions=(third_transition,),  # type: ignore[arg-type]
        )
        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(first_action, second_action),
            emissions=(first_emission, second_emission),
            surviving_emissions=(first_emission, second_emission),
        )

        frontier = batch.surviving_next_token_frontier

        self.assertEqual(
            tuple(entry.emitted_text for entry in frontier),
            ("C", "N"),
        )
        self.assertEqual(len(frontier[0].supports), 2)
        self.assertEqual(len(frontier[1].supports), 1)
        self.assertIs(frontier[0].supports[0].emission, first_emission)
        self.assertIs(frontier[0].supports[0].transition, first_transition)
        self.assertIs(frontier[0].supports[1].emission, second_emission)
        self.assertIs(frontier[0].supports[1].transition, third_transition)
        self.assertIs(frontier[1].supports[0].emission, first_emission)
        self.assertIs(frontier[1].supports[0].transition, second_transition)
        self.assertEqual(
            frontier[0].supports[0].graph_action_surface,
            first_emission.graph_action_surface,
        )
        self.assertEqual(
            frontier[0].transitions,
            (first_transition, third_transition),
        )
        self.assertEqual(frontier[1].transitions, (second_transition,))

    def test_scheduled_action_batch_frontier_uses_only_surviving_emissions(self) -> None:
        first_action = writer_transitions._finish_active_action(AtomId(0))
        second_action = writer_transitions._emit_root_atom_action(AtomId(1))
        transition = SimpleNamespace(emitted_text="C")
        first_emission = writer_transitions._WriterScheduledActionEmission(
            action=first_action,
            transitions=(),
        )
        second_emission = writer_transitions._WriterScheduledActionEmission(
            action=second_action,
            transitions=(transition,),  # type: ignore[arg-type]
        )
        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(first_action, second_action),
            emissions=(first_emission, second_emission),
            surviving_emissions=(second_emission,),
        )

        frontier = batch.surviving_next_token_frontier

        self.assertEqual(len(frontier), 1)
        self.assertEqual(frontier[0].emitted_text, "C")
        self.assertEqual(len(frontier[0].supports), 1)
        self.assertIs(frontier[0].supports[0].emission, second_emission)

    def test_scheduled_graph_action_surface_policy_family_classifies_action_families(self) -> None:
        pending = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.CONSUME_PENDING_ENTRY,
            active_atom=AtomId(0),
            pending_entry=True,
        )
        root = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.EMIT_ROOT_ATOM,
            active_atom=AtomId(0),
        )
        finish = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.FINISH_ACTIVE,
            active_atom=AtomId(0),
        )
        acyclic = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.ENTER_INLINE_CHILD,
            active_atom=AtomId(0),
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY
            ),
        )
        cyclic = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.OPEN_BRANCH,
            active_atom=AtomId(0),
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        generic_tree = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.ENTER_INLINE_CHILD,
            active_atom=AtomId(0),
        )
        closure_open = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.OPEN_CLOSURE_ENDPOINT,
            active_atom=AtomId(0),
        )
        closure_pair = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.PAIR_CLOSURE_ENDPOINT,
            active_atom=AtomId(0),
        )

        self.assertIs(
            pending.policy_family,
            writer_transitions._WriterGraphPolicyActionFamily.PENDING_ENTRY,
        )
        self.assertIs(
            root.policy_family,
            writer_transitions._WriterGraphPolicyActionFamily.ROOT_ATOM,
        )
        self.assertIs(
            finish.policy_family,
            writer_transitions._WriterGraphPolicyActionFamily.FINISH_ACTIVE,
        )
        self.assertIs(
            acyclic.policy_family,
            writer_transitions._WriterGraphPolicyActionFamily.ACYCLIC_TREE_ENTRY,
        )
        self.assertIs(
            cyclic.policy_family,
            writer_transitions._WriterGraphPolicyActionFamily.CYCLIC_TREE_ENTRY,
        )
        self.assertIs(
            generic_tree.policy_family,
            writer_transitions._WriterGraphPolicyActionFamily.TREE_ENTRY,
        )
        self.assertIs(
            closure_open.policy_family,
            writer_transitions._WriterGraphPolicyActionFamily.CLOSURE_OPEN,
        )
        self.assertIs(
            closure_pair.policy_family,
            writer_transitions._WriterGraphPolicyActionFamily.CLOSURE_PAIR,
        )

    def test_graph_action_surface_residual_attachment_policy_key_uses_residual_surfaces(self) -> None:
        closure_open = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.OPEN_CLOSURE_ENDPOINT,
            active_atom=AtomId(0),
            attachment_id=7,
        )
        cyclic = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.ENTER_INLINE_CHILD,
            active_atom=AtomId(0),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        acyclic = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.OPEN_BRANCH,
            active_atom=AtomId(0),
            attachment_id=8,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY
            ),
        )
        finish = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.FINISH_ACTIVE,
            active_atom=AtomId(0),
            attachment_id=7,
        )
        closure_pair = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.PAIR_CLOSURE_ENDPOINT,
            active_atom=AtomId(0),
            attachment_id=7,
        )

        self.assertEqual(
            closure_open.residual_attachment_policy_key,
            writer_transitions._WriterResidualAttachmentPolicyKey(AtomId(0), 7),
        )
        self.assertEqual(
            cyclic.residual_attachment_policy_key,
            writer_transitions._WriterResidualAttachmentPolicyKey(AtomId(0), 7),
        )
        self.assertEqual(
            acyclic.residual_attachment_policy_key,
            writer_transitions._WriterResidualAttachmentPolicyKey(AtomId(0), 8),
        )
        self.assertIsNone(finish.residual_attachment_policy_key)
        self.assertIsNone(closure_pair.residual_attachment_policy_key)

    def test_residual_attachment_policy_groups_preserve_order_and_group_by_key(self) -> None:
        closure_open_7 = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.OPEN_CLOSURE_ENDPOINT,
            active_atom=AtomId(0),
            attachment_id=7,
        )
        finish_7 = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.FINISH_ACTIVE,
            active_atom=AtomId(0),
            attachment_id=7,
        )
        cyclic_tree_7 = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.ENTER_INLINE_CHILD,
            active_atom=AtomId(0),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        acyclic_tree_8 = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.OPEN_BRANCH,
            active_atom=AtomId(0),
            attachment_id=8,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY
            ),
        )
        closure_open_8 = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.OPEN_CLOSURE_ENDPOINT,
            active_atom=AtomId(0),
            attachment_id=8,
        )

        groups = (
            writer_transitions
            ._residual_attachment_policy_groups_from_graph_action_surfaces(
                (
                    closure_open_7,
                    finish_7,
                    cyclic_tree_7,
                    acyclic_tree_8,
                    closure_open_8,
                ),
            )
        )

        self.assertEqual(len(groups), 2)
        self.assertEqual(
            groups[0].key,
            writer_transitions._WriterResidualAttachmentPolicyKey(AtomId(0), 7),
        )
        self.assertEqual(groups[0].surfaces, (closure_open_7, cyclic_tree_7))
        self.assertEqual(
            groups[1].key,
            writer_transitions._WriterResidualAttachmentPolicyKey(AtomId(0), 8),
        )
        self.assertEqual(groups[1].surfaces, (acyclic_tree_8, closure_open_8))
        self.assertEqual(groups[0].closure_open_surfaces, (closure_open_7,))
        self.assertEqual(groups[0].cyclic_tree_entry_surfaces, (cyclic_tree_7,))
        self.assertTrue(groups[0].has_closure_open_vs_cyclic_tree_entry_choice)
        self.assertEqual(groups[1].acyclic_tree_entry_surfaces, (acyclic_tree_8,))
        self.assertFalse(groups[1].has_closure_open_vs_cyclic_tree_entry_choice)

    def test_residual_attachment_owner_scope_kind_from_owner_kinds(self) -> None:
        cases = (
            (
                (),
                writer_transitions._WriterResidualAttachmentOwnerScopeKind.NONE,
            ),
            (
                (WriterBoundaryOwnerKind.ACTIVE_ATOM,),
                (
                    writer_transitions
                    ._WriterResidualAttachmentOwnerScopeKind
                    .ACTIVE_ATOM
                ),
            ),
            (
                (WriterBoundaryOwnerKind.BRANCH_RETURN,),
                (
                    writer_transitions
                    ._WriterResidualAttachmentOwnerScopeKind
                    .BRANCH_RETURN
                ),
            ),
            (
                (WriterBoundaryOwnerKind.PENDING_PARENT,),
                (
                    writer_transitions
                    ._WriterResidualAttachmentOwnerScopeKind
                    .PENDING_PARENT
                ),
            ),
            (
                (WriterBoundaryOwnerKind.OPEN_RING_ENDPOINT,),
                (
                    writer_transitions
                    ._WriterResidualAttachmentOwnerScopeKind
                    .OPEN_RING_ENDPOINT
                ),
            ),
            (
                (WriterBoundaryOwnerKind.UNOWNED,),
                writer_transitions._WriterResidualAttachmentOwnerScopeKind.UNOWNED,
            ),
            (
                (None,),
                writer_transitions._WriterResidualAttachmentOwnerScopeKind.MISSING,
            ),
            (
                (
                    WriterBoundaryOwnerKind.ACTIVE_ATOM,
                    WriterBoundaryOwnerKind.ACTIVE_ATOM,
                ),
                (
                    writer_transitions
                    ._WriterResidualAttachmentOwnerScopeKind
                    .ACTIVE_ATOM
                ),
            ),
            (
                (
                    WriterBoundaryOwnerKind.ACTIVE_ATOM,
                    WriterBoundaryOwnerKind.BRANCH_RETURN,
                ),
                writer_transitions._WriterResidualAttachmentOwnerScopeKind.MIXED,
            ),
            (
                (WriterBoundaryOwnerKind.ACTIVE_ATOM, None),
                writer_transitions._WriterResidualAttachmentOwnerScopeKind.MIXED,
            ),
        )

        for owner_kinds, expected in cases:
            with self.subTest(owner_kinds=owner_kinds):
                self.assertIs(
                    (
                        writer_transitions
                        ._residual_attachment_owner_scope_kind_from_owner_kinds(
                            owner_kinds,
                        )
                    ),
                    expected,
                )

    def test_residual_attachment_policy_group_reports_owner_scope_for_closure_open_cyclic_choice(self) -> None:
        closure_open = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.OPEN_CLOSURE_ENDPOINT,
            active_atom=AtomId(0),
            attachment_id=7,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        cyclic_tree = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.ENTER_INLINE_CHILD,
            active_atom=AtomId(0),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        active_owned = writer_transitions._WriterResidualAttachmentPolicyGroup(
            key=writer_transitions._WriterResidualAttachmentPolicyKey(AtomId(0), 7),
            surfaces=(closure_open, cyclic_tree),
        )

        self.assertEqual(
            active_owned.closure_open_owner_kinds,
            (WriterBoundaryOwnerKind.ACTIVE_ATOM,),
        )
        self.assertEqual(
            active_owned.cyclic_tree_entry_owner_kinds,
            (WriterBoundaryOwnerKind.ACTIVE_ATOM,),
        )
        self.assertEqual(
            active_owned.closure_open_vs_cyclic_tree_entry_owner_kinds,
            (
                WriterBoundaryOwnerKind.ACTIVE_ATOM,
                WriterBoundaryOwnerKind.ACTIVE_ATOM,
            ),
        )
        self.assertIs(
            active_owned.closure_open_vs_cyclic_tree_entry_owner_scope_kind,
            (
                writer_transitions
                ._WriterResidualAttachmentOwnerScopeKind
                .ACTIVE_ATOM
            ),
        )
        self.assertTrue(
            active_owned.has_active_atom_owned_closure_open_vs_cyclic_tree_entry_choice
        )
        self.assertTrue(
            (
                active_owned
                .has_active_atom_owner_scope_closure_open_vs_cyclic_tree_entry_choice
            )
        )
        self.assertFalse(
            active_owned
            .has_unsupported_owner_scope_closure_open_vs_cyclic_tree_entry_choice
        )

        cases = (
            (
                WriterBoundaryOwnerKind.BRANCH_RETURN,
                WriterBoundaryOwnerKind.BRANCH_RETURN,
                (
                    writer_transitions
                    ._WriterResidualAttachmentOwnerScopeKind
                    .BRANCH_RETURN
                ),
                "has_branch_return_owner_scope_closure_open_vs_cyclic_tree_entry_choice",
            ),
            (
                WriterBoundaryOwnerKind.PENDING_PARENT,
                WriterBoundaryOwnerKind.PENDING_PARENT,
                (
                    writer_transitions
                    ._WriterResidualAttachmentOwnerScopeKind
                    .PENDING_PARENT
                ),
                "has_pending_parent_owner_scope_closure_open_vs_cyclic_tree_entry_choice",
            ),
            (
                WriterBoundaryOwnerKind.OPEN_RING_ENDPOINT,
                WriterBoundaryOwnerKind.OPEN_RING_ENDPOINT,
                (
                    writer_transitions
                    ._WriterResidualAttachmentOwnerScopeKind
                    .OPEN_RING_ENDPOINT
                ),
                "has_open_ring_endpoint_owner_scope_closure_open_vs_cyclic_tree_entry_choice",
            ),
            (
                WriterBoundaryOwnerKind.UNOWNED,
                WriterBoundaryOwnerKind.UNOWNED,
                writer_transitions._WriterResidualAttachmentOwnerScopeKind.UNOWNED,
                "has_unowned_owner_scope_closure_open_vs_cyclic_tree_entry_choice",
            ),
            (
                None,
                None,
                writer_transitions._WriterResidualAttachmentOwnerScopeKind.MISSING,
                "has_missing_owner_scope_closure_open_vs_cyclic_tree_entry_choice",
            ),
            (
                WriterBoundaryOwnerKind.ACTIVE_ATOM,
                WriterBoundaryOwnerKind.BRANCH_RETURN,
                writer_transitions._WriterResidualAttachmentOwnerScopeKind.MIXED,
                "has_mixed_owner_scope_closure_open_vs_cyclic_tree_entry_choice",
            ),
            (
                WriterBoundaryOwnerKind.ACTIVE_ATOM,
                None,
                writer_transitions._WriterResidualAttachmentOwnerScopeKind.MIXED,
                "has_mixed_owner_scope_closure_open_vs_cyclic_tree_entry_choice",
            ),
        )

        for closure_owner, child_owner, expected, predicate in cases:
            with self.subTest(
                closure_owner=closure_owner,
                child_owner=child_owner,
            ):
                group = writer_transitions._WriterResidualAttachmentPolicyGroup(
                    key=writer_transitions._WriterResidualAttachmentPolicyKey(
                        AtomId(0),
                        7,
                    ),
                    surfaces=(
                        replace(closure_open, owner_kind=closure_owner),
                        replace(cyclic_tree, owner_kind=child_owner),
                    ),
                )

                self.assertIs(
                    group.closure_open_vs_cyclic_tree_entry_owner_scope_kind,
                    expected,
                )
                self.assertTrue(getattr(group, predicate))
                self.assertFalse(
                    group
                    .has_active_atom_owned_closure_open_vs_cyclic_tree_entry_choice
                )
                self.assertTrue(
                    group
                    .has_unsupported_owner_scope_closure_open_vs_cyclic_tree_entry_choice
                )

        no_choice = writer_transitions._WriterResidualAttachmentPolicyGroup(
            key=writer_transitions._WriterResidualAttachmentPolicyKey(
                AtomId(0),
                7,
            ),
            surfaces=(closure_open,),
        )

        self.assertIs(
            no_choice.closure_open_vs_cyclic_tree_entry_owner_scope_kind,
            writer_transitions._WriterResidualAttachmentOwnerScopeKind.NONE,
        )
        self.assertFalse(
            no_choice
            .has_active_atom_owned_closure_open_vs_cyclic_tree_entry_choice
        )
        self.assertFalse(
            no_choice
            .has_unsupported_owner_scope_closure_open_vs_cyclic_tree_entry_choice
        )

    def test_active_emitted_graph_policy_blocker_validates_payload_shape(self) -> None:
        child_blocker = writer_transitions._WriterChildObligationBlocker(
            kind=(
                writer_transitions._WriterChildObligationBlockerKind
                .MULTI_INCIDENCE_RESIDUAL_ATTACHMENT
            ),
            atom=AtomId(0),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        surface = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.OPEN_CLOSURE_ENDPOINT,
            active_atom=AtomId(0),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
        )
        group = writer_transitions._WriterResidualAttachmentPolicyGroup(
            key=writer_transitions._WriterResidualAttachmentPolicyKey(
                AtomId(0),
                7,
            ),
            surfaces=(surface,),
        )

        writer_transitions._WriterActiveEmittedGraphPolicyBlocker(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyBlockerKind
                .CHILD_OBLIGATION
            ),
            child_blocker=child_blocker,
        )
        writer_transitions._WriterActiveEmittedGraphPolicyBlocker(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyBlockerKind
                .UNSUPPORTED_OWNER_SCOPE_RESIDUAL_ATTACHMENT_CHOICE
            ),
            residual_group=group,
        )
        writer_transitions._WriterActiveEmittedGraphPolicyBlocker(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyBlockerKind
                .MISSING_CLOSURE_OPEN_SUPPORT_EVIDENCE
            ),
            residual_group=group,
        )

        invalid_payloads = (
            {
                "kind": (
                    writer_transitions
                    ._WriterActiveEmittedGraphPolicyBlockerKind
                    .CHILD_OBLIGATION
                ),
                "residual_group": group,
            },
            {
                "kind": (
                    writer_transitions
                    ._WriterActiveEmittedGraphPolicyBlockerKind
                    .UNSUPPORTED_OWNER_SCOPE_RESIDUAL_ATTACHMENT_CHOICE
                ),
                "child_blocker": child_blocker,
            },
            {
                "kind": (
                    writer_transitions
                    ._WriterActiveEmittedGraphPolicyBlockerKind
                    .MISSING_CLOSURE_OPEN_SUPPORT_EVIDENCE
                ),
            },
        )

        for payload in invalid_payloads:
            with self.assertRaises(SouthStarError) as raised:
                writer_transitions._WriterActiveEmittedGraphPolicyBlocker(
                    **payload,
                )

            self.assertIs(
                raised.exception.kind,
                SouthStarErrorKind.INTERNAL_INVARIANT,
            )

    def test_active_emitted_graph_policy_child_blocker_has_no_residual_owner_scope(self) -> None:
        child_blocker = writer_transitions._WriterChildObligationBlocker(
            kind=(
                writer_transitions._WriterChildObligationBlockerKind
                .MULTI_INCIDENCE_RESIDUAL_ATTACHMENT
            ),
            atom=AtomId(0),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        blocker = writer_transitions._WriterActiveEmittedGraphPolicyBlocker(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyBlockerKind
                .CHILD_OBLIGATION
            ),
            child_blocker=child_blocker,
        )

        self.assertIsNone(blocker.residual_attachment_owner_scope_kind)

    def test_residual_attachment_policy_emission_group_reports_dead_closure_open_support(self) -> None:
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        closure_open = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        closure_open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            closure_open,
            label,
        )
        cyclic_child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        cyclic_action = writer_transitions._enter_inline_child_action(
            active_atom,
            cyclic_child,
        )
        closure_open_emission = writer_transitions._WriterScheduledActionEmission(
            action=closure_open_action,
            transitions=(),
        )
        cyclic_emission = writer_transitions._WriterScheduledActionEmission(
            action=cyclic_action,
            transitions=(SimpleNamespace(emitted_text="C"),),  # type: ignore[arg-type]
        )
        group = writer_transitions._WriterResidualAttachmentPolicyEmissionGroup(
            key=writer_transitions._WriterResidualAttachmentPolicyKey(
                active_atom,
                7,
            ),
            emissions=(closure_open_emission, cyclic_emission),
        )

        self.assertTrue(group.closure_open_was_considered)
        self.assertFalse(group.closure_open_support_survived)
        self.assertTrue(group.closure_open_support_dead)
        self.assertEqual(group.surviving_closure_open_emissions, ())
        self.assertEqual(
            group.surviving_cyclic_tree_entry_emissions,
            (cyclic_emission,),
        )

        surviving_closure_open_emission = (
            writer_transitions._WriterScheduledActionEmission(
                action=closure_open_action,
                transitions=(SimpleNamespace(emitted_text="1"),),  # type: ignore[arg-type]
            )
        )
        surviving_group = (
            writer_transitions._WriterResidualAttachmentPolicyEmissionGroup(
                key=writer_transitions._WriterResidualAttachmentPolicyKey(
                    active_atom,
                    7,
                ),
                emissions=(surviving_closure_open_emission, cyclic_emission),
            )
        )

        self.assertTrue(surviving_group.closure_open_was_considered)
        self.assertTrue(surviving_group.closure_open_support_survived)
        self.assertFalse(surviving_group.closure_open_support_dead)

    def test_residual_attachment_policy_emission_groups_preserve_order_and_skip_non_residual(self) -> None:
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        closure_open = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        closure_open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            closure_open,
            label,
        )
        finish_action = writer_transitions._finish_active_action(active_atom)
        cyclic_child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        cyclic_action = writer_transitions._enter_inline_child_action(
            active_atom,
            cyclic_child,
        )
        acyclic_child = writer_transitions._WriterChildObligation(
            bond=BondId(3),
            child=AtomId(5),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=8,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY
            ),
        )
        acyclic_action = writer_transitions._enter_inline_child_action(
            active_atom,
            acyclic_child,
        )
        closure_open_emission = writer_transitions._WriterScheduledActionEmission(
            action=closure_open_action,
            transitions=(),
        )
        finish_emission = writer_transitions._WriterScheduledActionEmission(
            action=finish_action,
            transitions=(SimpleNamespace(emitted_text=""),),  # type: ignore[arg-type]
        )
        cyclic_tree_emission = writer_transitions._WriterScheduledActionEmission(
            action=cyclic_action,
            transitions=(SimpleNamespace(emitted_text="C"),),  # type: ignore[arg-type]
        )
        acyclic_tree_emission = writer_transitions._WriterScheduledActionEmission(
            action=acyclic_action,
            transitions=(SimpleNamespace(emitted_text="N"),),  # type: ignore[arg-type]
        )

        groups = (
            writer_transitions
            ._residual_attachment_policy_emission_groups_from_scheduled_action_emissions(
                (
                    closure_open_emission,
                    finish_emission,
                    cyclic_tree_emission,
                    acyclic_tree_emission,
                ),
            )
        )

        self.assertEqual(len(groups), 2)
        self.assertEqual(
            groups[0].key,
            writer_transitions._WriterResidualAttachmentPolicyKey(active_atom, 7),
        )
        self.assertEqual(
            groups[0].emissions,
            (closure_open_emission, cyclic_tree_emission),
        )
        self.assertEqual(groups[0].closure_open_emissions, (closure_open_emission,))
        self.assertEqual(
            groups[0].cyclic_tree_entry_emissions,
            (cyclic_tree_emission,),
        )
        self.assertEqual(groups[0].surviving_emissions, (cyclic_tree_emission,))
        self.assertEqual(groups[0].surviving_closure_open_emissions, ())
        self.assertEqual(
            groups[0].surviving_cyclic_tree_entry_emissions,
            (cyclic_tree_emission,),
        )
        self.assertTrue(groups[0].has_closure_open_vs_cyclic_tree_entry_choice)
        self.assertEqual(
            groups[1].key,
            writer_transitions._WriterResidualAttachmentPolicyKey(active_atom, 8),
        )
        self.assertEqual(groups[1].emissions, (acyclic_tree_emission,))

    def test_scheduled_action_batch_exposes_all_and_surviving_residual_policy_emission_groups(self) -> None:
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        closure_open = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        closure_open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            closure_open,
            label,
        )
        cyclic_child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        cyclic_action = writer_transitions._enter_inline_child_action(
            active_atom,
            cyclic_child,
        )
        closure_open_zero_emission = (
            writer_transitions._WriterScheduledActionEmission(
                action=closure_open_action,
                transitions=(),
            )
        )
        cyclic_tree_surviving_emission = (
            writer_transitions._WriterScheduledActionEmission(
                action=cyclic_action,
                transitions=(SimpleNamespace(emitted_text="C"),),  # type: ignore[arg-type]
            )
        )
        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(closure_open_action, cyclic_action),
            emissions=(
                closure_open_zero_emission,
                cyclic_tree_surviving_emission,
            ),
            surviving_emissions=(cyclic_tree_surviving_emission,),
        )

        all_groups = batch.residual_attachment_policy_emission_groups
        surviving_groups = batch.surviving_residual_attachment_policy_emission_groups

        self.assertEqual(len(all_groups), 1)
        self.assertEqual(
            all_groups[0].emissions,
            (
                closure_open_zero_emission,
                cyclic_tree_surviving_emission,
            ),
        )
        self.assertEqual(len(surviving_groups), 1)
        self.assertEqual(
            surviving_groups[0].emissions,
            (cyclic_tree_surviving_emission,),
        )
        self.assertEqual(surviving_groups[0].closure_open_emissions, ())
        self.assertEqual(
            surviving_groups[0].cyclic_tree_entry_emissions,
            (cyclic_tree_surviving_emission,),
        )

    def test_next_token_frontier_entry_exposes_policy_families_per_support(self) -> None:
        active_atom = AtomId(0)
        child = writer_transitions._WriterChildObligation(
            bond=BondId(1),
            child=AtomId(2),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        label = WriterClosureLabel(value=1, text="1")
        closure_open = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(2),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=9,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        closure_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            closure_open,
            label,
        )
        child_transition = SimpleNamespace(emitted_text="1")
        closure_transition = SimpleNamespace(emitted_text="1")
        child_emission = writer_transitions._WriterScheduledActionEmission(
            action=child_action,
            transitions=(child_transition,),  # type: ignore[arg-type]
        )
        closure_emission = writer_transitions._WriterScheduledActionEmission(
            action=closure_action,
            transitions=(closure_transition,),  # type: ignore[arg-type]
        )

        frontier = (
            writer_transitions
            ._next_token_frontier_from_scheduled_action_emissions(
                (child_emission, closure_emission),
            )
        )

        self.assertEqual(len(frontier), 1)
        self.assertEqual(frontier[0].emitted_text, "1")
        self.assertEqual(
            frontier[0].policy_families,
            (
                writer_transitions._WriterGraphPolicyActionFamily.CYCLIC_TREE_ENTRY,
                writer_transitions._WriterGraphPolicyActionFamily.CLOSURE_OPEN,
            ),
        )
        self.assertIs(
            frontier[0].supports[0].policy_family,
            writer_transitions._WriterGraphPolicyActionFamily.CYCLIC_TREE_ENTRY,
        )
        self.assertIs(
            frontier[0].supports[1].policy_family,
            writer_transitions._WriterGraphPolicyActionFamily.CLOSURE_OPEN,
        )

    def test_transitions_from_scheduled_actions_flattens_emissions(self) -> None:
        prepared = object()
        state = object()
        context = object()
        first_action = writer_transitions._finish_active_action(AtomId(0))
        second_action = writer_transitions._emit_root_atom_action(AtomId(1))
        first_transition = object()
        second_transition = object()
        emissions = (
            writer_transitions._WriterScheduledActionEmission(
                action=first_action,
                transitions=(first_transition,),
            ),
            writer_transitions._WriterScheduledActionEmission(
                action=second_action,
                transitions=(second_transition,),
            ),
        )

        with patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emissions",
            return_value=emissions,
        ) as scheduled_emissions:
            result = writer_transitions._transitions_from_scheduled_actions(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                (first_action, second_action),
            )

        self.assertEqual(result, (first_transition, second_transition))
        scheduled_emissions.assert_called_once_with(
            prepared,
            state,
            context,
            (first_action, second_action),
        )

    def test_transitions_from_scheduled_action_emissions_flattens_transition_tuples(self) -> None:
        first_action = writer_transitions._finish_active_action(AtomId(0))
        second_action = writer_transitions._emit_root_atom_action(AtomId(1))
        first_transition = object()
        second_transition = object()
        third_transition = object()
        emissions = (
            writer_transitions._WriterScheduledActionEmission(
                action=first_action,
                transitions=(first_transition, second_transition),
            ),
            writer_transitions._WriterScheduledActionEmission(
                action=second_action,
                transitions=(),
            ),
            writer_transitions._WriterScheduledActionEmission(
                action=second_action,
                transitions=(third_transition,),
            ),
        )

        self.assertEqual(
            writer_transitions._transitions_from_scheduled_action_emissions(
                emissions,  # type: ignore[arg-type]
            ),
            (first_transition, second_transition, third_transition),
        )

    def test_surviving_scheduled_action_emissions_drop_zero_transition_emissions(self) -> None:
        first_action = writer_transitions._finish_active_action(AtomId(0))
        second_action = writer_transitions._emit_root_atom_action(AtomId(1))
        transition = object()
        first_emission = writer_transitions._WriterScheduledActionEmission(
            action=first_action,
            transitions=(transition,),
        )
        second_emission = writer_transitions._WriterScheduledActionEmission(
            action=second_action,
            transitions=(),
        )

        self.assertEqual(
            writer_transitions._surviving_scheduled_action_emissions(
                (first_emission, second_emission),
            ),
            (first_emission,),
        )

    def test_scheduled_action_emission_batch_preserves_actions_and_survivors(self) -> None:
        prepared = object()
        state = object()
        context = object()
        first_action = writer_transitions._finish_active_action(AtomId(0))
        second_action = writer_transitions._emit_root_atom_action(AtomId(1))
        first_emission = writer_transitions._WriterScheduledActionEmission(
            action=first_action,
            transitions=(object(),),
        )
        second_emission = writer_transitions._WriterScheduledActionEmission(
            action=second_action,
            transitions=(),
        )

        with patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emissions",
            return_value=(first_emission, second_emission),
        ) as scheduled_emissions, patch(
            "grimace._south_star1.writer_transitions._surviving_scheduled_action_emissions",
            return_value=(first_emission,),
        ) as surviving_emissions:
            batch = writer_transitions._scheduled_action_emission_batch(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                (first_action, second_action),
            )

        self.assertEqual(batch.actions, (first_action, second_action))
        self.assertEqual(batch.emissions, (first_emission, second_emission))
        self.assertEqual(batch.surviving_emissions, (first_emission,))
        scheduled_emissions.assert_called_once_with(
            prepared,
            state,
            context,
            (first_action, second_action),
        )
        surviving_emissions.assert_called_once_with(
            (first_emission, second_emission),
        )

    def test_active_emitted_graph_policy_selects_surviving_closure_without_children(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(7)
        closure_decision, _, _ = self._test_closure_endpoint_decision(
            pair_survives=False,
            open_survives=True,
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            side_effect=AssertionError("child surface should not be computed"),
        ) as child_surface:
            decision = writer_transitions._active_emitted_graph_policy_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            decision.kind,
            (
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .CLOSURE_ENDPOINT
            ),
        )
        self.assertIsNone(decision.child_schedule_surface)
        self.assertIs(
            decision.closure_endpoint_selection_kind,
            (
                writer_transitions
                ._WriterClosureEndpointSelectionKind
                .CLOSURE_OPEN
            ),
        )
        self.assertFalse(decision.blocked)
        self.assertEqual(decision.blockers, ())
        self.assertEqual(decision.graph_policy_blockers, ())
        self.assertFalse(decision.graph_policy_blocked)
        child_surface.assert_not_called()

    def test_active_emitted_graph_policy_selects_closure_from_selected_endpoint_kind(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(7)
        closure_decision, _, _ = self._test_closure_endpoint_decision(
            pair_survives=True,
            open_survives=False,
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            side_effect=AssertionError("child surface should not be computed"),
        ) as child_surface:
            decision = writer_transitions._active_emitted_graph_policy_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            decision.kind,
            (
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .CLOSURE_ENDPOINT
            ),
        )
        self.assertIs(
            decision.closure_endpoint_selection_kind,
            (
                writer_transitions
                ._WriterClosureEndpointSelectionKind
                .CLOSURE_PAIR
            ),
        )
        child_surface.assert_not_called()

    def test_active_emitted_graph_policy_selects_child_after_empty_closure_survivors(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(7)
        child = writer_transitions._WriterChildObligation(
            bond=BondId(1),
            child=AtomId(2),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY
            ),
        )
        action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(action,),
        )
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=pair_batch,
            open_batch=open_batch,
            surviving_emissions=(),
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            return_value=child_surface,
        ):
            decision = writer_transitions._active_emitted_graph_policy_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            decision.kind,
            (
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD
            ),
        )
        self.assertIs(decision.child_schedule_surface, child_surface)
        self.assertEqual(
            decision.child_scheduled_actions,
            child_surface.scheduled_actions,
        )
        self.assertEqual(decision.graph_policy_blockers, ())
        self.assertFalse(decision.graph_policy_blocked)
        self.assertEqual(
            decision.graph_action_surfaces,
            child_surface.graph_action_surfaces,
        )
        self.assertEqual(decision.blockers, ())
        self.assertFalse(decision.blocked)

    def test_active_emitted_graph_policy_reaches_child_when_closure_considered_but_not_selected(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(7)
        closure_decision, _, _ = self._test_closure_endpoint_decision(
            pair_survives=False,
            open_survives=False,
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(1),
            child=AtomId(2),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=99,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY
            ),
        )
        action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(action,),
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            return_value=child_surface,
        ):
            policy = writer_transitions._active_emitted_graph_policy_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            policy.kind,
            writer_transitions._WriterActiveEmittedGraphPolicyDecisionKind.ACTIVE_CHILD,
        )
        self.assertIsNot(
            policy.considered_closure_endpoint_selection_kind,
            writer_transitions._WriterClosureEndpointSelectionKind.NONE,
        )
        self.assertIs(
            policy.closure_endpoint_selection_kind,
            writer_transitions._WriterClosureEndpointSelectionKind.NONE,
        )
        self.assertIs(policy.child_schedule_surface, child_surface)

    def test_active_emitted_schedule_decision_exposes_selected_active_child_family(self) -> None:
        active_atom = AtomId(7)
        closure_decision, _, _ = self._test_closure_endpoint_decision(
            active_atom=active_atom,
            pair_survives=False,
            open_survives=False,
        )
        cyclic_action = self._test_child_action(
            active_atom,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(cyclic_action.child_obligation,),
            scheduled_actions=(cyclic_action,),
        )
        cyclic_emission = writer_transitions._WriterScheduledActionEmission(
            action=cyclic_action,
            transitions=(object(),),  # type: ignore[arg-type]
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(cyclic_action,),
            emissions=(cyclic_emission,),
            surviving_emissions=(cyclic_emission,),
        )
        decision = writer_transitions._active_emitted_child_decision(
            closure_decision,
            child_surface,
            child_batch,
        )

        self.assertIs(
            decision.selected_active_child_selection_kind,
            writer_transitions._WriterActiveChildSelectionKind.CYCLIC_TREE_ENTRY,
        )
        self.assertEqual(
            decision.selected_cyclic_tree_entry_graph_action_surfaces,
            decision.selected_graph_action_surfaces,
        )

        closure_selected, _, _ = self._test_closure_endpoint_decision(
            active_atom=active_atom,
            pair_survives=True,
            open_survives=False,
        )
        closure_decision = writer_transitions._active_emitted_closure_decision(
            closure_selected,
        )

        self.assertIs(
            closure_decision.selected_active_child_selection_kind,
            writer_transitions._WriterActiveChildSelectionKind.NONE,
        )
        self.assertEqual(
            closure_decision.selected_active_child_graph_action_surfaces,
            (),
        )

    def test_active_emitted_schedule_decision_selected_child_family_uses_surviving_emissions(self) -> None:
        active_atom = AtomId(7)
        closure_decision, _, _ = self._test_closure_endpoint_decision(
            active_atom=active_atom,
            pair_survives=False,
            open_survives=False,
        )
        cyclic_action = self._test_child_action(
            active_atom,
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        acyclic_action = self._test_child_action(
            active_atom,
            attachment_id=8,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY
            ),
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(
                cyclic_action.child_obligation,
                acyclic_action.child_obligation,
            ),
            scheduled_actions=(cyclic_action, acyclic_action),
        )
        cyclic_emission = writer_transitions._WriterScheduledActionEmission(
            action=cyclic_action,
            transitions=(),
        )
        acyclic_emission = writer_transitions._WriterScheduledActionEmission(
            action=acyclic_action,
            transitions=(object(),),  # type: ignore[arg-type]
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(cyclic_action, acyclic_action),
            emissions=(cyclic_emission, acyclic_emission),
            surviving_emissions=(acyclic_emission,),
        )
        decision = writer_transitions._active_emitted_child_decision(
            closure_decision,
            child_surface,
            child_batch,
        )

        self.assertIs(
            decision.selected_active_child_selection_kind,
            writer_transitions._WriterActiveChildSelectionKind.ACYCLIC_TREE_ENTRY,
        )
        self.assertEqual(
            decision.selected_cyclic_tree_entry_graph_action_surfaces,
            (),
        )

    def test_active_emitted_graph_policy_exposes_considered_active_child_family(self) -> None:
        active_atom = AtomId(7)
        closure_decision, _, _ = self._test_closure_endpoint_decision(
            active_atom=active_atom,
            pair_survives=False,
            open_survives=False,
        )
        cyclic_action = self._test_child_action(
            active_atom,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(cyclic_action.child_obligation,),
            scheduled_actions=(cyclic_action,),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD_AFTER_DEAD_CLOSURE_OPEN
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )

        self.assertIs(
            policy.considered_active_child_selection_kind,
            writer_transitions._WriterActiveChildSelectionKind.CYCLIC_TREE_ENTRY,
        )
        self.assertEqual(
            policy.considered_cyclic_tree_entry_graph_action_surfaces,
            (
                child_surface
                .considered_cyclic_tree_entry_graph_action_surfaces
            ),
        )

        closure_selected, _, _ = self._test_closure_endpoint_decision(
            active_atom=active_atom,
            pair_survives=True,
            open_survives=False,
        )
        closure_policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .CLOSURE_ENDPOINT
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_selected,
        )

        self.assertIs(
            closure_policy.considered_active_child_selection_kind,
            writer_transitions._WriterActiveChildSelectionKind.NONE,
        )
        self.assertEqual(
            closure_policy.considered_cyclic_tree_entry_graph_action_surfaces,
            (),
        )

    def test_active_emitted_graph_policy_rejects_closure_policy_without_selected_endpoint(self) -> None:
        closure_decision, _, _ = self._test_closure_endpoint_decision(
            pair_survives=False,
            open_survives=False,
        )

        with self.assertRaises(SouthStarError) as raised:
            writer_transitions._WriterActiveEmittedGraphPolicyDecision(
                kind=(
                    writer_transitions
                    ._WriterActiveEmittedGraphPolicyDecisionKind
                    .CLOSURE_ENDPOINT
                ),
                active_atom=AtomId(0),
                closure_endpoint_decision=closure_decision,
            )

        self.assertIs(
            raised.exception.kind,
            SouthStarErrorKind.INTERNAL_INVARIANT,
        )

    def test_active_emitted_graph_policy_records_blocked_child_without_raising(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(7)
        blocker = writer_transitions._WriterChildObligationBlocker(
            kind=(
                writer_transitions._WriterChildObligationBlockerKind
                .MULTI_INCIDENCE_RESIDUAL_ATTACHMENT
            ),
            atom=active_atom,
            attachment_id=7,
            attachment_action_kind=WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(blocker,),
            child_obligations=(),
            scheduled_actions=(),
        )
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=pair_batch,
            open_batch=open_batch,
            surviving_emissions=(),
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            return_value=child_surface,
        ):
            decision = writer_transitions._active_emitted_graph_policy_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            decision.kind,
            (
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .BLOCKED_CHILD
            ),
        )
        self.assertTrue(decision.blocked)
        self.assertEqual(decision.blockers, child_surface.blockers)
        self.assertEqual(decision.child_scheduled_actions, ())
        self.assertEqual(decision.graph_action_surfaces, ())

    def test_active_emitted_graph_policy_exposes_child_graph_policy_blockers(self) -> None:
        active_atom = AtomId(7)
        child_blocker = writer_transitions._WriterChildObligationBlocker(
            kind=(
                writer_transitions._WriterChildObligationBlockerKind
                .MULTI_INCIDENCE_RESIDUAL_ATTACHMENT
            ),
            atom=active_atom,
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(child_blocker,),
            child_obligations=(),
            scheduled_actions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .BLOCKED_CHILD
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )

        blockers = policy.graph_policy_blockers

        self.assertTrue(policy.graph_policy_blocked)
        self.assertEqual(policy.blocked, policy.graph_policy_blocked)
        self.assertEqual(len(blockers), 1)
        self.assertIs(
            blockers[0].kind,
            (
                writer_transitions
                ._WriterActiveEmittedGraphPolicyBlockerKind
                .CHILD_OBLIGATION
            ),
        )
        self.assertIs(blockers[0].child_blocker, child_blocker)
        self.assertIsNone(blockers[0].residual_group)

    def test_active_emitted_schedule_outcome_validates_scheduled_payload_shape(self) -> None:
        active_atom = AtomId(7)
        closure_decision, _, _ = self._test_closure_endpoint_decision(
            active_atom=active_atom,
            pair_survives=True,
            open_survives=False,
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .CLOSURE_ENDPOINT
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
        )
        schedule_decision = writer_transitions._active_emitted_closure_decision(
            closure_decision,
            graph_policy_decision=policy,
        )

        outcome = writer_transitions._WriterActiveEmittedScheduleOutcome(
            kind=writer_transitions._WriterActiveEmittedScheduleOutcomeKind.SCHEDULED,
            graph_policy_decision=policy,
            schedule_decision=schedule_decision,
        )

        self.assertIs(outcome.graph_policy_decision, policy)
        self.assertIs(outcome.schedule_decision, schedule_decision)

        with self.assertRaises(SouthStarError) as raised:
            writer_transitions._WriterActiveEmittedScheduleOutcome(
                kind=(
                    writer_transitions
                    ._WriterActiveEmittedScheduleOutcomeKind
                    .SCHEDULED
                ),
                graph_policy_decision=policy,
            )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.INTERNAL_INVARIANT)

        other_policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .CLOSURE_ENDPOINT
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
        )
        other_schedule_decision = (
            writer_transitions._active_emitted_closure_decision(
                closure_decision,
                graph_policy_decision=other_policy,
            )
        )

        with self.assertRaises(SouthStarError) as raised:
            writer_transitions._WriterActiveEmittedScheduleOutcome(
                kind=(
                    writer_transitions
                    ._WriterActiveEmittedScheduleOutcomeKind
                    .SCHEDULED
                ),
                graph_policy_decision=policy,
                schedule_decision=other_schedule_decision,
            )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.INTERNAL_INVARIANT)

    def test_active_emitted_schedule_outcome_validates_blocked_payload_shape(self) -> None:
        active_atom = AtomId(7)
        child_blocker = writer_transitions._WriterChildObligationBlocker(
            kind=(
                writer_transitions._WriterChildObligationBlockerKind
                .MULTI_INCIDENCE_RESIDUAL_ATTACHMENT
            ),
            atom=active_atom,
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(child_blocker,),
            child_obligations=(),
            scheduled_actions=(),
        )
        empty_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        blocked_closure_decision = (
            writer_transitions._WriterClosureEndpointScheduleDecision(
                pair_batch=empty_batch,
                open_batch=empty_batch,
                surviving_emissions=(),
            )
        )
        blocked_policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .BLOCKED_CHILD
            ),
            active_atom=active_atom,
            closure_endpoint_decision=blocked_closure_decision,
            child_schedule_surface=child_surface,
        )

        outcome = writer_transitions._WriterActiveEmittedScheduleOutcome(
            kind=writer_transitions._WriterActiveEmittedScheduleOutcomeKind.BLOCKED,
            graph_policy_decision=blocked_policy,
        )

        self.assertIsNone(outcome.schedule_decision)
        self.assertEqual(
            outcome.graph_policy_blockers,
            blocked_policy.graph_policy_blockers,
        )
        self.assertEqual(outcome.selected_transitions, ())
        self.assertEqual(outcome.selected_next_token_frontier, ())

        scheduled_closure_decision, _, _ = (
            self._test_closure_endpoint_decision(
                active_atom=active_atom,
                pair_survives=True,
                open_survives=False,
            )
        )
        scheduled_policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .CLOSURE_ENDPOINT
            ),
            active_atom=active_atom,
            closure_endpoint_decision=scheduled_closure_decision,
        )
        schedule_decision = writer_transitions._active_emitted_closure_decision(
            scheduled_closure_decision,
            graph_policy_decision=scheduled_policy,
        )

        with self.assertRaises(SouthStarError) as raised:
            writer_transitions._WriterActiveEmittedScheduleOutcome(
                kind=(
                    writer_transitions
                    ._WriterActiveEmittedScheduleOutcomeKind
                    .BLOCKED
                ),
                graph_policy_decision=blocked_policy,
                schedule_decision=schedule_decision,
            )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.INTERNAL_INVARIANT)

        with self.assertRaises(SouthStarError) as raised:
            writer_transitions._WriterActiveEmittedScheduleOutcome(
                kind=(
                    writer_transitions
                    ._WriterActiveEmittedScheduleOutcomeKind
                    .BLOCKED
                ),
                graph_policy_decision=scheduled_policy,
            )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.INTERNAL_INVARIANT)

    def test_active_emitted_schedule_outcome_returns_closure_schedule(self) -> None:
        active_atom = AtomId(7)
        closure_decision, _, _ = self._test_closure_endpoint_decision(
            active_atom=active_atom,
            pair_survives=True,
            open_survives=False,
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .CLOSURE_ENDPOINT
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
        )

        with patch(
            "grimace._south_star1.writer_transitions._active_emitted_graph_policy_decision",
            return_value=policy,
        ), patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            side_effect=AssertionError("closure policy should not emit child batch"),
        ) as emission_batch:
            outcome = writer_transitions._active_emitted_schedule_outcome(
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            outcome.kind,
            writer_transitions._WriterActiveEmittedScheduleOutcomeKind.SCHEDULED,
        )
        self.assertIs(outcome.graph_policy_decision, policy)
        self.assertIs(
            outcome.schedule_decision.kind,
            (
                writer_transitions
                ._WriterActiveEmittedScheduleDecisionKind
                .CLOSURE_ENDPOINT
            ),
        )
        emission_batch.assert_not_called()

    def test_active_emitted_schedule_outcome_returns_child_schedule_for_emittable_policy(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        open_emission = writer_transitions._WriterScheduledActionEmission(
            action=open_action,
            transitions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(open_emission,),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD_AFTER_DEAD_CLOSURE_OPEN
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )
        transition = SimpleNamespace(emitted_text="C")
        child_emission = writer_transitions._WriterScheduledActionEmission(
            action=child_action,
            transitions=(transition,),  # type: ignore[arg-type]
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(child_action,),
            emissions=(child_emission,),
            surviving_emissions=(child_emission,),
        )

        with patch(
            "grimace._south_star1.writer_transitions._active_emitted_graph_policy_decision",
            return_value=policy,
        ), patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            return_value=child_batch,
        ) as emission_batch:
            outcome = writer_transitions._active_emitted_schedule_outcome(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            outcome.kind,
            writer_transitions._WriterActiveEmittedScheduleOutcomeKind.SCHEDULED,
        )
        self.assertIs(
            outcome.schedule_decision.kind,
            writer_transitions._WriterActiveEmittedScheduleDecisionKind.ACTIVE_CHILD,
        )
        self.assertIs(outcome.schedule_decision.graph_policy_decision, policy)
        emission_batch.assert_called_once_with(
            prepared,
            state,
            context,
            policy.child_scheduled_actions,
        )
        self.assertEqual(
            outcome.selected_transitions,
            outcome.schedule_decision.selected_transitions,
        )

    def test_active_emitted_schedule_outcome_returns_blocked_result_without_child_emission(self) -> None:
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        open_emission = writer_transitions._WriterScheduledActionEmission(
            action=open_action,
            transitions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(open_emission,),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .UNSUPPORTED_OWNER_SCOPE_RESIDUAL_ATTACHMENT_CHOICE
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )

        with patch(
            "grimace._south_star1.writer_transitions._active_emitted_graph_policy_decision",
            return_value=policy,
        ), patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            side_effect=AssertionError("blocked outcome should not emit child batch"),
        ) as emission_batch:
            outcome = writer_transitions._active_emitted_schedule_outcome(
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            outcome.kind,
            writer_transitions._WriterActiveEmittedScheduleOutcomeKind.BLOCKED,
        )
        self.assertIsNone(outcome.schedule_decision)
        self.assertIs(outcome.graph_policy_decision, policy)
        self.assertEqual(
            outcome.graph_policy_blockers,
            policy.graph_policy_blockers,
        )
        emission_batch.assert_not_called()

    def test_active_emitted_schedule_decision_raises_from_blocked_schedule_outcome(self) -> None:
        active_atom = AtomId(7)
        child_blocker = writer_transitions._WriterChildObligationBlocker(
            kind=(
                writer_transitions._WriterChildObligationBlockerKind
                .MULTI_INCIDENCE_RESIDUAL_ATTACHMENT
            ),
            atom=active_atom,
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(child_blocker,),
            child_obligations=(),
            scheduled_actions=(),
        )
        empty_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=empty_batch,
            open_batch=empty_batch,
            surviving_emissions=(),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .BLOCKED_CHILD
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )
        outcome = writer_transitions._WriterActiveEmittedScheduleOutcome(
            kind=writer_transitions._WriterActiveEmittedScheduleOutcomeKind.BLOCKED,
            graph_policy_decision=policy,
        )

        with patch(
            "grimace._south_star1.writer_transitions._active_emitted_schedule_outcome",
            return_value=outcome,
        ):
            with self.assertRaises(SouthStarError) as raised:
                writer_transitions._active_emitted_schedule_decision(
                    object(),  # type: ignore[arg-type]
                    object(),  # type: ignore[arg-type]
                    object(),  # type: ignore[arg-type]
                    active_atom,
                )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.UNSUPPORTED_POLICY)

    def test_active_emitted_schedule_decision_returns_scheduled_outcome_decision(self) -> None:
        active_atom = AtomId(7)
        closure_decision, _, _ = self._test_closure_endpoint_decision(
            active_atom=active_atom,
            pair_survives=True,
            open_survives=False,
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .CLOSURE_ENDPOINT
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
        )
        schedule_decision = writer_transitions._active_emitted_closure_decision(
            closure_decision,
            graph_policy_decision=policy,
        )
        outcome = writer_transitions._WriterActiveEmittedScheduleOutcome(
            kind=writer_transitions._WriterActiveEmittedScheduleOutcomeKind.SCHEDULED,
            graph_policy_decision=policy,
            schedule_decision=schedule_decision,
        )

        with patch(
            "grimace._south_star1.writer_transitions._active_emitted_schedule_outcome",
            return_value=outcome,
        ):
            decision = writer_transitions._active_emitted_schedule_decision(
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(decision, schedule_decision)

    def test_active_emitted_graph_policy_closure_surfaces_distinguish_considered_and_chosen(self) -> None:
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        endpoint = WriterOpenClosureEndpoint(
            bond=BondId(1),
            first_atom=AtomId(2),
            second_atom=active_atom,
            label=label,
            first_endpoint_text="1",
            first_endpoint_bond_text="",
        )
        closure = WriterClosedClosure(
            bond=BondId(1),
            first_atom=AtomId(2),
            second_atom=active_atom,
            label=label,
            first_endpoint_text="1",
            second_endpoint_text="1",
            first_endpoint_bond_text="",
            second_endpoint_bond_text="",
        )
        pair_action = writer_transitions._pair_closure_endpoint_action(
            active_atom,
            writer_transitions._WriterClosurePairObligation(
                endpoint=endpoint,
                closure=closure,
            ),
        )
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(2),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        surface = writer_transitions._WriterClosureEndpointScheduleSurface(
            active_atom=active_atom,
            pair_actions=(pair_action,),
            open_actions=(open_action,),
        )
        pair_emission = writer_transitions._WriterScheduledActionEmission(
            action=pair_action,
            transitions=(object(),),  # type: ignore[arg-type]
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(pair_action,),
                emissions=(pair_emission,),
                surviving_emissions=(pair_emission,),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(),
                surviving_emissions=(),
            ),
            surviving_emissions=(pair_emission,),
            schedule_surface=surface,
        )

        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .CLOSURE_ENDPOINT
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
        )

        self.assertEqual(
            policy.considered_graph_action_surfaces,
            closure_decision.considered_graph_action_surfaces,
        )
        self.assertEqual(
            policy.chosen_graph_action_surfaces,
            closure_decision.selected_graph_action_surfaces,
        )
        self.assertEqual(
            policy.graph_action_surfaces,
            policy.chosen_graph_action_surfaces,
        )

    def test_active_emitted_graph_policy_exposes_closure_endpoint_selection_kind(self) -> None:
        active_atom = AtomId(0)
        closure_decision, _, open_emission = (
            self._test_closure_endpoint_decision(
                pair_survives=False,
                open_survives=True,
            )
        )
        closure_policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .CLOSURE_ENDPOINT
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
        )

        self.assertIs(
            closure_policy.closure_endpoint_selection_kind,
            (
                writer_transitions
                ._WriterClosureEndpointSelectionKind
                .CLOSURE_OPEN
            ),
        )
        self.assertIs(
            closure_policy.considered_closure_endpoint_selection_kind,
            closure_decision.considered_closure_endpoint_selection_kind,
        )
        self.assertEqual(
            closure_policy.selected_closure_open_graph_action_surfaces,
            (open_emission.graph_action_surface,),
        )
        self.assertEqual(
            closure_policy.selected_closure_pair_graph_action_surfaces,
            (),
        )

        active_child_closure_decision, _, _ = (
            self._test_closure_endpoint_decision(
                pair_survives=False,
                open_survives=False,
            )
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=9,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )
        active_child_policy = (
            writer_transitions._WriterActiveEmittedGraphPolicyDecision(
                kind=(
                    writer_transitions
                    ._WriterActiveEmittedGraphPolicyDecisionKind
                    .ACTIVE_CHILD
                ),
                active_atom=active_atom,
                closure_endpoint_decision=active_child_closure_decision,
                child_schedule_surface=child_surface,
            )
        )

        self.assertIs(
            active_child_policy.closure_endpoint_selection_kind,
            writer_transitions._WriterClosureEndpointSelectionKind.NONE,
        )
        self.assertEqual(
            (
                active_child_policy
                .considered_closure_endpoint_selection_kind
            ),
            (
                active_child_closure_decision
                .considered_closure_endpoint_selection_kind
            ),
        )
        self.assertEqual(
            active_child_policy.selected_closure_open_graph_action_surfaces,
            (),
        )
        self.assertEqual(
            active_child_policy.selected_closure_pair_graph_action_surfaces,
            (),
        )

    def test_active_emitted_graph_policy_child_considered_surfaces_include_closure_before_child(self) -> None:
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=9,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )

        self.assertEqual(
            policy.closure_considered_graph_action_surfaces,
            closure_decision.considered_graph_action_surfaces,
        )
        self.assertEqual(
            policy.child_considered_graph_action_surfaces,
            child_surface.graph_action_surfaces,
        )
        self.assertEqual(
            policy.considered_graph_action_surfaces,
            (
                *closure_decision.considered_graph_action_surfaces,
                *child_surface.graph_action_surfaces,
            ),
        )
        self.assertEqual(
            policy.chosen_graph_action_surfaces,
            child_surface.graph_action_surfaces,
        )

    def test_active_emitted_graph_policy_filters_surfaces_by_policy_family(self) -> None:
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=9,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )

        self.assertEqual(
            policy.considered_graph_action_surfaces_for_policy_family(
                writer_transitions._WriterGraphPolicyActionFamily.CLOSURE_OPEN,
            ),
            closure_decision.considered_graph_action_surfaces,
        )
        self.assertEqual(
            policy.chosen_graph_action_surfaces_for_policy_family(
                writer_transitions._WriterGraphPolicyActionFamily.CYCLIC_TREE_ENTRY,
            ),
            child_surface.graph_action_surfaces,
        )
        self.assertEqual(
            policy.chosen_graph_action_surfaces_for_policy_family(
                writer_transitions._WriterGraphPolicyActionFamily.CLOSURE_OPEN,
            ),
            (),
        )

    def test_active_emitted_graph_policy_records_unresolved_residual_attachment_choice(self) -> None:
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        open_emission = writer_transitions._WriterScheduledActionEmission(
            action=open_action,
            transitions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .UNRESOLVED_RESIDUAL_ATTACHMENT_CHOICE
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )

        groups = policy.unresolved_residual_attachment_policy_groups

        self.assertTrue(policy.blocked)
        self.assertEqual(policy.child_scheduled_actions, ())
        self.assertEqual(policy.chosen_graph_action_surfaces, ())
        self.assertEqual(
            groups,
            policy.considered_closure_open_vs_cyclic_tree_entry_groups,
        )
        self.assertEqual(len(groups), 1)
        self.assertEqual(
            groups[0].key,
            writer_transitions._WriterResidualAttachmentPolicyKey(active_atom, 11),
        )
        self.assertEqual(
            groups[0].closure_open_surfaces,
            closure_decision.considered_graph_action_surfaces,
        )
        self.assertEqual(
            groups[0].cyclic_tree_entry_surfaces,
            child_surface.graph_action_surfaces,
        )
        self.assertTrue(groups[0].has_closure_open_vs_cyclic_tree_entry_choice)
        self.assertEqual(
            policy.considered_closure_open_vs_cyclic_tree_entry_groups,
            (groups[0],),
        )

    def test_active_emitted_graph_policy_records_unsupported_owner_scope_residual_choice(self) -> None:
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        open_emission = writer_transitions._WriterScheduledActionEmission(
            action=open_action,
            transitions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(open_emission,),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .UNSUPPORTED_OWNER_SCOPE_RESIDUAL_ATTACHMENT_CHOICE
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )

        self.assertTrue(policy.blocked)
        self.assertFalse(policy.emits_child_actions)
        self.assertEqual(policy.child_scheduled_actions, ())
        self.assertTrue(
            policy.unsupported_owner_scope_residual_attachment_policy_groups
        )
        blockers = policy.graph_policy_blockers
        self.assertTrue(policy.graph_policy_blocked)
        self.assertEqual(policy.blocked, policy.graph_policy_blocked)
        self.assertEqual(len(blockers), 1)
        self.assertIs(
            blockers[0].kind,
            (
                writer_transitions
                ._WriterActiveEmittedGraphPolicyBlockerKind
                .UNSUPPORTED_OWNER_SCOPE_RESIDUAL_ATTACHMENT_CHOICE
            ),
        )
        self.assertEqual(
            blockers[0].residual_group,
            policy.unsupported_owner_scope_residual_attachment_policy_groups[0],
        )
        self.assertEqual(
            blockers[0].residual_attachment_policy_key,
            writer_transitions._WriterResidualAttachmentPolicyKey(active_atom, 11),
        )
        self.assertEqual(
            policy.unresolved_residual_attachment_policy_groups,
            (),
        )
        self.assertEqual(
            policy.resolved_residual_attachment_policy_groups,
            (),
        )

    def test_active_emitted_graph_policy_selects_unresolved_residual_choice_before_child(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        open_emission = writer_transitions._WriterScheduledActionEmission(
            action=open_action,
            transitions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            return_value=child_surface,
        ):
            decision = writer_transitions._active_emitted_graph_policy_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            decision.kind,
            (
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .UNRESOLVED_RESIDUAL_ATTACHMENT_CHOICE
            ),
        )
        self.assertIs(decision.child_schedule_surface, child_surface)
        self.assertEqual(decision.child_scheduled_actions, ())
        self.assertTrue(decision.unresolved_residual_attachment_policy_groups)

    def test_active_emitted_schedule_decision_raises_for_unresolved_residual_choice_without_child_emission(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .UNRESOLVED_RESIDUAL_ATTACHMENT_CHOICE
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )

        with patch(
            "grimace._south_star1.writer_transitions._active_emitted_graph_policy_decision",
            return_value=policy,
        ), patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            side_effect=AssertionError("unresolved policy should not emit child batch"),
        ) as emission_batch:
            with self.assertRaises(SouthStarError) as raised:
                writer_transitions._active_emitted_schedule_decision(
                    prepared,  # type: ignore[arg-type]
                    state,  # type: ignore[arg-type]
                    context,  # type: ignore[arg-type]
                    active_atom,
                )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.UNSUPPORTED_POLICY)
        emission_batch.assert_not_called()

    def test_active_emitted_schedule_decision_raises_for_unsupported_owner_scope_without_child_emission(self) -> None:
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        open_emission = writer_transitions._WriterScheduledActionEmission(
            action=open_action,
            transitions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(open_emission,),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .UNSUPPORTED_OWNER_SCOPE_RESIDUAL_ATTACHMENT_CHOICE
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )

        with patch(
            "grimace._south_star1.writer_transitions._active_emitted_graph_policy_decision",
            return_value=policy,
        ), patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            side_effect=AssertionError(
                "unsupported owner scope should not emit child batch"
            ),
        ) as emission_batch:
            with self.assertRaises(SouthStarError) as raised:
                writer_transitions._active_emitted_schedule_decision(
                    object(),  # type: ignore[arg-type]
                    object(),  # type: ignore[arg-type]
                    object(),  # type: ignore[arg-type]
                    active_atom,
                )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.UNSUPPORTED_POLICY)
        emission_batch.assert_not_called()

    def test_closure_open_vs_cyclic_tree_entry_policy_groups_require_closure_open(self) -> None:
        active_atom = AtomId(0)
        closure_decision = self._test_closure_endpoint_decision_for_actions(
            active_atom,
            pair_actions=(self._test_closure_pair_action(active_atom),),
        )
        cyclic_action = self._test_child_action(
            active_atom,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(cyclic_action.child_obligation,),
            scheduled_actions=(cyclic_action,),
        )

        self.assertEqual(
            (
                writer_transitions
                ._closure_open_vs_cyclic_tree_entry_policy_groups(
                    closure_decision,
                    child_surface,
                )
            ),
            (),
        )

    def test_closure_open_vs_cyclic_tree_entry_policy_groups_require_cyclic_tree_entry(self) -> None:
        active_atom = AtomId(0)
        closure_decision = self._test_closure_endpoint_decision_for_actions(
            active_atom,
            open_actions=(
                self._test_closure_open_action(
                    active_atom,
                    attachment_id=11,
                ),
            ),
        )
        acyclic_action = self._test_child_action(
            active_atom,
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY
            ),
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(acyclic_action.child_obligation,),
            scheduled_actions=(acyclic_action,),
        )

        self.assertEqual(
            (
                writer_transitions
                ._closure_open_vs_cyclic_tree_entry_policy_groups(
                    closure_decision,
                    child_surface,
                )
            ),
            (),
        )

    def test_closure_open_vs_cyclic_tree_entry_policy_groups_match_same_attachment(self) -> None:
        active_atom = AtomId(0)
        closure_decision = self._test_closure_endpoint_decision_for_actions(
            active_atom,
            open_actions=(
                self._test_closure_open_action(
                    active_atom,
                    attachment_id=11,
                ),
            ),
        )
        cyclic_action = self._test_child_action(
            active_atom,
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(cyclic_action.child_obligation,),
            scheduled_actions=(cyclic_action,),
        )

        groups = (
            writer_transitions
            ._closure_open_vs_cyclic_tree_entry_policy_groups(
                closure_decision,
                child_surface,
            )
        )

        self.assertEqual(len(groups), 1)
        self.assertEqual(
            groups[0].key,
            writer_transitions._WriterResidualAttachmentPolicyKey(
                active_atom,
                11,
            ),
        )
        self.assertEqual(
            groups[0].closure_open_surfaces,
            closure_decision.considered_closure_open_graph_action_surfaces,
        )
        self.assertEqual(
            groups[0].cyclic_tree_entry_surfaces,
            (
                child_surface
                .considered_cyclic_tree_entry_graph_action_surfaces
            ),
        )

    def test_closure_open_vs_cyclic_tree_entry_policy_groups_ignore_different_attachments(self) -> None:
        active_atom = AtomId(0)
        closure_decision = self._test_closure_endpoint_decision_for_actions(
            active_atom,
            open_actions=(
                self._test_closure_open_action(
                    active_atom,
                    attachment_id=11,
                ),
            ),
        )
        cyclic_action = self._test_child_action(
            active_atom,
            attachment_id=12,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(cyclic_action.child_obligation,),
            scheduled_actions=(cyclic_action,),
        )

        self.assertEqual(
            (
                writer_transitions
                ._closure_open_vs_cyclic_tree_entry_policy_groups(
                    closure_decision,
                    child_surface,
                )
            ),
            (),
        )

    def test_residual_cyclic_policy_decision_validates_payload_shape(self) -> None:
        closure_decision, child_surface = self._test_residual_cyclic_policy_inputs()
        key = writer_transitions._WriterResidualAttachmentPolicyKey(
            AtomId(0),
            11,
        )
        choice = self._test_residual_policy_group(key)
        unsupported = self._test_residual_policy_group(key)
        missing = self._test_residual_policy_group(key)
        support_dead = self._test_residual_policy_group(key)

        valid_payloads = (
            dict(
                kind=(
                    writer_transitions
                    ._WriterResidualCyclicPolicyDecisionKind
                    .NONE
                ),
            ),
            dict(
                kind=(
                    writer_transitions
                    ._WriterResidualCyclicPolicyDecisionKind
                    .UNSUPPORTED_OWNER_SCOPE
                ),
                choice_groups=(choice,),
                unsupported_owner_scope_groups=(unsupported,),
            ),
            dict(
                kind=(
                    writer_transitions
                    ._WriterResidualCyclicPolicyDecisionKind
                    .MISSING_CLOSURE_OPEN_SUPPORT_EVIDENCE
                ),
                choice_groups=(choice,),
                missing_evidence_groups=(missing,),
            ),
            dict(
                kind=(
                    writer_transitions
                    ._WriterResidualCyclicPolicyDecisionKind
                    .ACTIVE_CHILD_AFTER_DEAD_CLOSURE_OPEN
                ),
                choice_groups=(choice,),
                support_dead_groups=(support_dead,),
            ),
        )

        for payload in valid_payloads:
            with self.subTest(valid=payload["kind"]):
                writer_transitions._WriterResidualCyclicPolicyDecision(
                    closure_endpoint_decision=closure_decision,
                    child_schedule_surface=child_surface,
                    **payload,
                )

        invalid_payloads = (
            dict(
                kind=(
                    writer_transitions
                    ._WriterResidualCyclicPolicyDecisionKind
                    .NONE
                ),
                choice_groups=(choice,),
            ),
            dict(
                kind=(
                    writer_transitions
                    ._WriterResidualCyclicPolicyDecisionKind
                    .UNSUPPORTED_OWNER_SCOPE
                ),
                choice_groups=(choice,),
            ),
            dict(
                kind=(
                    writer_transitions
                    ._WriterResidualCyclicPolicyDecisionKind
                    .MISSING_CLOSURE_OPEN_SUPPORT_EVIDENCE
                ),
                choice_groups=(choice,),
                support_dead_groups=(support_dead,),
            ),
            dict(
                kind=(
                    writer_transitions
                    ._WriterResidualCyclicPolicyDecisionKind
                    .ACTIVE_CHILD_AFTER_DEAD_CLOSURE_OPEN
                ),
                choice_groups=(choice,),
                missing_evidence_groups=(missing,),
            ),
        )

        for payload in invalid_payloads:
            with self.subTest(invalid=payload["kind"]):
                with self.assertRaises(SouthStarError) as raised:
                    writer_transitions._WriterResidualCyclicPolicyDecision(
                        closure_endpoint_decision=closure_decision,
                        child_schedule_surface=child_surface,
                        **payload,
                    )

                self.assertIs(
                    raised.exception.kind,
                    SouthStarErrorKind.INTERNAL_INVARIANT,
                )

    def test_residual_cyclic_policy_decision_returns_none_without_choice_groups(self) -> None:
        active_atom = AtomId(0)
        closure_decision = self._test_closure_endpoint_decision_for_actions(
            active_atom,
            open_actions=(
                self._test_closure_open_action(
                    active_atom,
                    attachment_id=11,
                ),
            ),
        )
        acyclic_action = self._test_child_action(
            active_atom,
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY
            ),
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(acyclic_action.child_obligation,),
            scheduled_actions=(acyclic_action,),
        )

        decision = writer_transitions._residual_cyclic_policy_decision(
            closure_decision,
            child_surface,
        )

        self.assertIs(
            decision.kind,
            writer_transitions._WriterResidualCyclicPolicyDecisionKind.NONE,
        )
        self.assertFalse(decision.blocks_active_child)
        self.assertFalse(
            decision.resolves_active_child_after_dead_closure_open
        )

    def test_residual_cyclic_policy_decision_classifies_unsupported_owner_scope(self) -> None:
        closure_decision, child_surface = (
            self._test_residual_cyclic_policy_inputs(
                closure_owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
                child_owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
            )
        )

        decision = writer_transitions._residual_cyclic_policy_decision(
            closure_decision,
            child_surface,
        )

        self.assertIs(
            decision.kind,
            (
                writer_transitions
                ._WriterResidualCyclicPolicyDecisionKind
                .UNSUPPORTED_OWNER_SCOPE
            ),
        )
        self.assertTrue(decision.blocks_active_child)
        self.assertTrue(decision.unsupported_owner_scope_groups)

    def test_residual_cyclic_policy_decision_classifies_missing_closure_open_evidence(self) -> None:
        closure_decision, child_surface = (
            self._test_residual_cyclic_policy_inputs(
                include_closure_emission=False,
            )
        )

        decision = writer_transitions._residual_cyclic_policy_decision(
            closure_decision,
            child_surface,
        )

        self.assertIs(
            decision.kind,
            (
                writer_transitions
                ._WriterResidualCyclicPolicyDecisionKind
                .MISSING_CLOSURE_OPEN_SUPPORT_EVIDENCE
            ),
        )
        self.assertTrue(decision.blocks_active_child)
        self.assertTrue(decision.missing_evidence_groups)

    def test_residual_cyclic_policy_decision_classifies_dead_closure_open_resolution(self) -> None:
        closure_decision, child_surface = self._test_residual_cyclic_policy_inputs()

        decision = writer_transitions._residual_cyclic_policy_decision(
            closure_decision,
            child_surface,
        )

        self.assertIs(
            decision.kind,
            (
                writer_transitions
                ._WriterResidualCyclicPolicyDecisionKind
                .ACTIVE_CHILD_AFTER_DEAD_CLOSURE_OPEN
            ),
        )
        self.assertFalse(decision.blocks_active_child)
        self.assertTrue(
            decision.resolves_active_child_after_dead_closure_open
        )
        self.assertEqual(
            decision.resolved_residual_attachment_policy_groups,
            decision.support_dead_groups,
        )

    def test_active_emitted_graph_policy_maps_residual_cyclic_policy_decision(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(0)
        key = writer_transitions._WriterResidualAttachmentPolicyKey(
            active_atom,
            11,
        )
        group = self._test_residual_policy_group(key)
        none_closure_decision, none_child_surface = (
            self._test_residual_cyclic_policy_inputs(
                child_attachment_action_kind=(
                    WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY
                ),
            )
        )
        unsupported_closure_decision, unsupported_child_surface = (
            self._test_residual_cyclic_policy_inputs(
                closure_owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
                child_owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
            )
        )
        missing_closure_decision, missing_child_surface = (
            self._test_residual_cyclic_policy_inputs(
                include_closure_emission=False,
            )
        )
        support_dead_closure_decision, support_dead_child_surface = (
            self._test_residual_cyclic_policy_inputs()
        )
        cases = (
            (
                writer_transitions._WriterResidualCyclicPolicyDecision(
                    kind=(
                        writer_transitions
                        ._WriterResidualCyclicPolicyDecisionKind
                        .NONE
                    ),
                    closure_endpoint_decision=none_closure_decision,
                    child_schedule_surface=none_child_surface,
                ),
                none_closure_decision,
                none_child_surface,
                (
                    writer_transitions
                    ._WriterActiveEmittedGraphPolicyDecisionKind
                    .ACTIVE_CHILD
                ),
            ),
            (
                writer_transitions._WriterResidualCyclicPolicyDecision(
                    kind=(
                        writer_transitions
                        ._WriterResidualCyclicPolicyDecisionKind
                        .UNSUPPORTED_OWNER_SCOPE
                    ),
                    closure_endpoint_decision=unsupported_closure_decision,
                    child_schedule_surface=unsupported_child_surface,
                    choice_groups=(group,),
                    unsupported_owner_scope_groups=(group,),
                ),
                unsupported_closure_decision,
                unsupported_child_surface,
                (
                    writer_transitions
                    ._WriterActiveEmittedGraphPolicyDecisionKind
                    .UNSUPPORTED_OWNER_SCOPE_RESIDUAL_ATTACHMENT_CHOICE
                ),
            ),
            (
                writer_transitions._WriterResidualCyclicPolicyDecision(
                    kind=(
                        writer_transitions
                        ._WriterResidualCyclicPolicyDecisionKind
                        .MISSING_CLOSURE_OPEN_SUPPORT_EVIDENCE
                    ),
                    closure_endpoint_decision=missing_closure_decision,
                    child_schedule_surface=missing_child_surface,
                    choice_groups=(group,),
                    missing_evidence_groups=(group,),
                ),
                missing_closure_decision,
                missing_child_surface,
                (
                    writer_transitions
                    ._WriterActiveEmittedGraphPolicyDecisionKind
                    .UNRESOLVED_RESIDUAL_ATTACHMENT_CHOICE
                ),
            ),
            (
                writer_transitions._WriterResidualCyclicPolicyDecision(
                    kind=(
                        writer_transitions
                        ._WriterResidualCyclicPolicyDecisionKind
                        .ACTIVE_CHILD_AFTER_DEAD_CLOSURE_OPEN
                    ),
                    closure_endpoint_decision=support_dead_closure_decision,
                    child_schedule_surface=support_dead_child_surface,
                    choice_groups=(group,),
                    support_dead_groups=(group,),
                ),
                support_dead_closure_decision,
                support_dead_child_surface,
                (
                    writer_transitions
                    ._WriterActiveEmittedGraphPolicyDecisionKind
                    .ACTIVE_CHILD_AFTER_DEAD_CLOSURE_OPEN
                ),
            ),
        )

        for residual_decision, closure_decision, child_surface, expected_kind in cases:
            with self.subTest(kind=residual_decision.kind):
                with patch(
                    "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
                    return_value=closure_decision,
                ), patch(
                    "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
                    return_value=child_surface,
                ), patch(
                    "grimace._south_star1.writer_transitions._residual_cyclic_policy_decision",
                    return_value=residual_decision,
                ):
                    policy = writer_transitions._active_emitted_graph_policy_decision(
                        prepared,  # type: ignore[arg-type]
                        state,  # type: ignore[arg-type]
                        context,  # type: ignore[arg-type]
                        active_atom,
                    )

                self.assertIs(policy.kind, expected_kind)
                self.assertIs(
                    policy.residual_cyclic_policy_decision,
                    residual_decision,
                )
                self.assertIs(
                    policy.residual_cyclic_policy_kind,
                    residual_decision.kind,
                )

    def test_active_emitted_graph_policy_accepts_matching_retained_residual_cyclic_decision(self) -> None:
        none_closure_decision, none_child_surface = (
            self._test_residual_cyclic_policy_inputs(
                child_attachment_action_kind=(
                    WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY
                ),
            )
        )
        unsupported_closure_decision, unsupported_child_surface = (
            self._test_residual_cyclic_policy_inputs(
                closure_owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
                child_owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
            )
        )
        missing_closure_decision, missing_child_surface = (
            self._test_residual_cyclic_policy_inputs(
                include_closure_emission=False,
            )
        )
        support_dead_closure_decision, support_dead_child_surface = (
            self._test_residual_cyclic_policy_inputs()
        )

        cases = (
            (
                none_closure_decision,
                none_child_surface,
                (
                    writer_transitions
                    ._WriterActiveEmittedGraphPolicyDecisionKind
                    .ACTIVE_CHILD
                ),
            ),
            (
                unsupported_closure_decision,
                unsupported_child_surface,
                (
                    writer_transitions
                    ._WriterActiveEmittedGraphPolicyDecisionKind
                    .UNSUPPORTED_OWNER_SCOPE_RESIDUAL_ATTACHMENT_CHOICE
                ),
            ),
            (
                missing_closure_decision,
                missing_child_surface,
                (
                    writer_transitions
                    ._WriterActiveEmittedGraphPolicyDecisionKind
                    .UNRESOLVED_RESIDUAL_ATTACHMENT_CHOICE
                ),
            ),
            (
                support_dead_closure_decision,
                support_dead_child_surface,
                (
                    writer_transitions
                    ._WriterActiveEmittedGraphPolicyDecisionKind
                    .ACTIVE_CHILD_AFTER_DEAD_CLOSURE_OPEN
                ),
            ),
        )

        for closure_decision, child_surface, graph_policy_kind in cases:
            residual_decision = writer_transitions._residual_cyclic_policy_decision(
                closure_decision,
                child_surface,
            )

            with self.subTest(kind=residual_decision.kind):
                graph_policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
                    kind=graph_policy_kind,
                    active_atom=AtomId(0),
                    closure_endpoint_decision=closure_decision,
                    child_schedule_surface=child_surface,
                    residual_cyclic_policy_decision=residual_decision,
                )

                self.assertIs(
                    graph_policy.residual_cyclic_policy_decision,
                    residual_decision,
                )
                self.assertIs(
                    graph_policy.residual_cyclic_policy_kind,
                    residual_decision.kind,
                )
                self.assertEqual(
                    graph_policy.residual_cyclic_choice_groups,
                    residual_decision.choice_groups,
                )
                self.assertEqual(
                    (
                        graph_policy
                        .residual_cyclic_unsupported_owner_scope_groups
                    ),
                    residual_decision.unsupported_owner_scope_groups,
                )
                self.assertEqual(
                    graph_policy.residual_cyclic_missing_evidence_groups,
                    residual_decision.missing_evidence_groups,
                )
                self.assertEqual(
                    graph_policy.residual_cyclic_support_dead_groups,
                    residual_decision.support_dead_groups,
                )

    def test_active_emitted_graph_policy_rejects_mismatched_retained_residual_cyclic_decision(self) -> None:
        none_closure_decision, none_child_surface = (
            self._test_residual_cyclic_policy_inputs(
                child_attachment_action_kind=(
                    WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY
                ),
            )
        )
        support_dead_closure_decision, support_dead_child_surface = (
            self._test_residual_cyclic_policy_inputs()
        )
        none_decision = writer_transitions._WriterResidualCyclicPolicyDecision(
            kind=(
                writer_transitions
                ._WriterResidualCyclicPolicyDecisionKind
                .NONE
            ),
            closure_endpoint_decision=support_dead_closure_decision,
            child_schedule_surface=support_dead_child_surface,
        )
        support_dead_decision = (
            writer_transitions._residual_cyclic_policy_decision(
                support_dead_closure_decision,
                support_dead_child_surface,
            )
        )

        invalid_payloads = (
            dict(
                kind=(
                    writer_transitions
                    ._WriterActiveEmittedGraphPolicyDecisionKind
                    .ACTIVE_CHILD_AFTER_DEAD_CLOSURE_OPEN
                ),
                closure_endpoint_decision=support_dead_closure_decision,
                child_schedule_surface=support_dead_child_surface,
                residual_cyclic_policy_decision=none_decision,
            ),
            dict(
                kind=(
                    writer_transitions
                    ._WriterActiveEmittedGraphPolicyDecisionKind
                    .ACTIVE_CHILD
                ),
                closure_endpoint_decision=none_closure_decision,
                child_schedule_surface=none_child_surface,
                residual_cyclic_policy_decision=support_dead_decision,
            ),
            dict(
                kind=(
                    writer_transitions
                    ._WriterActiveEmittedGraphPolicyDecisionKind
                    .CLOSURE_ENDPOINT
                ),
                closure_endpoint_decision=(
                    self._test_closure_endpoint_decision(
                        pair_survives=True,
                        open_survives=False,
                    )[0]
                ),
                residual_cyclic_policy_decision=support_dead_decision,
            ),
        )

        for payload in invalid_payloads:
            with self.subTest(kind=payload["kind"]):
                with self.assertRaises(SouthStarError) as raised:
                    writer_transitions._WriterActiveEmittedGraphPolicyDecision(
                        active_atom=AtomId(0),
                        **payload,
                    )

                self.assertIs(
                    raised.exception.kind,
                    SouthStarErrorKind.INTERNAL_INVARIANT,
                )

    def test_active_emitted_graph_policy_uses_retained_unsupported_owner_residual_blocker_groups(self) -> None:
        closure_decision, child_surface = (
            self._test_residual_cyclic_policy_inputs(
                closure_owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
                child_owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
            )
        )
        residual_decision = writer_transitions._residual_cyclic_policy_decision(
            closure_decision,
            child_surface,
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .UNSUPPORTED_OWNER_SCOPE_RESIDUAL_ATTACHMENT_CHOICE
            ),
            active_atom=AtomId(0),
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
            residual_cyclic_policy_decision=residual_decision,
        )

        self.assertIs(
            policy.residual_cyclic_blocker_kind,
            (
                writer_transitions
                ._WriterActiveEmittedGraphPolicyBlockerKind
                .UNSUPPORTED_OWNER_SCOPE_RESIDUAL_ATTACHMENT_CHOICE
            ),
        )
        self.assertEqual(
            policy.residual_cyclic_blocker_groups,
            residual_decision.unsupported_owner_scope_groups,
        )
        self.assertEqual(len(policy.graph_policy_blockers), 1)
        self.assertIs(
            policy.graph_policy_blockers[0].kind,
            (
                writer_transitions
                ._WriterActiveEmittedGraphPolicyBlockerKind
                .UNSUPPORTED_OWNER_SCOPE_RESIDUAL_ATTACHMENT_CHOICE
            ),
        )
        self.assertIs(
            policy.graph_policy_blockers[0].residual_group,
            residual_decision.unsupported_owner_scope_groups[0],
        )

    def test_active_emitted_graph_policy_uses_retained_missing_evidence_residual_blocker_groups(self) -> None:
        closure_decision, child_surface = (
            self._test_residual_cyclic_policy_inputs(
                include_closure_emission=False,
            )
        )
        residual_decision = writer_transitions._residual_cyclic_policy_decision(
            closure_decision,
            child_surface,
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .UNRESOLVED_RESIDUAL_ATTACHMENT_CHOICE
            ),
            active_atom=AtomId(0),
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
            residual_cyclic_policy_decision=residual_decision,
        )

        self.assertIs(
            policy.residual_cyclic_blocker_kind,
            (
                writer_transitions
                ._WriterActiveEmittedGraphPolicyBlockerKind
                .MISSING_CLOSURE_OPEN_SUPPORT_EVIDENCE
            ),
        )
        self.assertEqual(
            policy.residual_cyclic_blocker_groups,
            residual_decision.missing_evidence_groups,
        )
        self.assertEqual(len(policy.graph_policy_blockers), 1)
        self.assertIs(
            policy.graph_policy_blockers[0].kind,
            (
                writer_transitions
                ._WriterActiveEmittedGraphPolicyBlockerKind
                .MISSING_CLOSURE_OPEN_SUPPORT_EVIDENCE
            ),
        )
        self.assertIs(
            policy.graph_policy_blockers[0].residual_group,
            residual_decision.missing_evidence_groups[0],
        )

    def test_active_emitted_graph_policy_residual_blockers_fall_back_without_retained_decision(self) -> None:
        unsupported_closure_decision, unsupported_child_surface = (
            self._test_residual_cyclic_policy_inputs(
                closure_owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
                child_owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
            )
        )
        unsupported_policy = (
            writer_transitions._WriterActiveEmittedGraphPolicyDecision(
                kind=(
                    writer_transitions
                    ._WriterActiveEmittedGraphPolicyDecisionKind
                    .UNSUPPORTED_OWNER_SCOPE_RESIDUAL_ATTACHMENT_CHOICE
                ),
                active_atom=AtomId(0),
                closure_endpoint_decision=unsupported_closure_decision,
                child_schedule_surface=unsupported_child_surface,
            )
        )
        missing_closure_decision, missing_child_surface = (
            self._test_residual_cyclic_policy_inputs(
                include_closure_emission=False,
            )
        )
        missing_policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .UNRESOLVED_RESIDUAL_ATTACHMENT_CHOICE
            ),
            active_atom=AtomId(0),
            closure_endpoint_decision=missing_closure_decision,
            child_schedule_surface=missing_child_surface,
        )

        cases = (
            (
                unsupported_policy,
                (
                    writer_transitions
                    ._WriterActiveEmittedGraphPolicyBlockerKind
                    .UNSUPPORTED_OWNER_SCOPE_RESIDUAL_ATTACHMENT_CHOICE
                ),
                (
                    unsupported_policy
                    .unsupported_owner_scope_residual_attachment_policy_groups
                ),
            ),
            (
                missing_policy,
                (
                    writer_transitions
                    ._WriterActiveEmittedGraphPolicyBlockerKind
                    .MISSING_CLOSURE_OPEN_SUPPORT_EVIDENCE
                ),
                missing_policy.unresolved_residual_attachment_policy_groups,
            ),
        )

        for policy, expected_kind, expected_groups in cases:
            with self.subTest(kind=policy.kind):
                self.assertIsNone(policy.residual_cyclic_policy_decision)
                self.assertIs(policy.residual_cyclic_blocker_kind, expected_kind)
                self.assertEqual(
                    policy.residual_cyclic_blocker_groups,
                    expected_groups,
                )
                self.assertEqual(
                    tuple(
                        blocker.residual_group
                        for blocker in policy.graph_policy_blockers
                    ),
                    expected_groups,
                )

    def test_active_emitted_graph_policy_non_blocking_residual_decisions_produce_no_blockers(self) -> None:
        none_closure_decision, none_child_surface = (
            self._test_residual_cyclic_policy_inputs(
                child_attachment_action_kind=(
                    WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY
                ),
            )
        )
        none_residual = writer_transitions._residual_cyclic_policy_decision(
            none_closure_decision,
            none_child_surface,
        )
        none_policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD
            ),
            active_atom=AtomId(0),
            closure_endpoint_decision=none_closure_decision,
            child_schedule_surface=none_child_surface,
            residual_cyclic_policy_decision=none_residual,
        )
        dead_closure_decision, dead_child_surface = (
            self._test_residual_cyclic_policy_inputs()
        )
        dead_residual = writer_transitions._residual_cyclic_policy_decision(
            dead_closure_decision,
            dead_child_surface,
        )
        dead_policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD_AFTER_DEAD_CLOSURE_OPEN
            ),
            active_atom=AtomId(0),
            closure_endpoint_decision=dead_closure_decision,
            child_schedule_surface=dead_child_surface,
            residual_cyclic_policy_decision=dead_residual,
        )

        for policy in (none_policy, dead_policy):
            with self.subTest(kind=policy.kind):
                self.assertIsNone(policy.residual_cyclic_blocker_kind)
                self.assertEqual(policy.residual_cyclic_blocker_groups, ())
                self.assertEqual(policy.graph_policy_blockers, ())
                self.assertFalse(policy.graph_policy_blocked)

    def test_active_emitted_graph_policy_production_blockers_use_retained_residual_decision(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(0)
        cases = (
            self._test_residual_cyclic_policy_inputs(
                closure_owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
                child_owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
            ),
            self._test_residual_cyclic_policy_inputs(
                include_closure_emission=False,
            ),
        )

        for closure_decision, child_surface in cases:
            with self.subTest(child_surface=child_surface):
                with patch(
                    "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
                    return_value=closure_decision,
                ), patch(
                    "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
                    return_value=child_surface,
                ):
                    policy = (
                        writer_transitions
                        ._active_emitted_graph_policy_decision(
                            prepared,  # type: ignore[arg-type]
                            state,  # type: ignore[arg-type]
                            context,  # type: ignore[arg-type]
                            active_atom,
                        )
                    )

                self.assertIsNotNone(policy.residual_cyclic_policy_decision)
                self.assertTrue(policy.graph_policy_blockers)
                self.assertIs(
                    policy.graph_policy_blockers[0].residual_group,
                    policy.residual_cyclic_blocker_groups[0],
                )

    def test_active_emitted_graph_policy_retained_support_dead_decision_drives_resolved_evidence(self) -> None:
        closure_decision, child_surface = self._test_residual_cyclic_policy_inputs()
        residual_decision = writer_transitions._residual_cyclic_policy_decision(
            closure_decision,
            child_surface,
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD_AFTER_DEAD_CLOSURE_OPEN
            ),
            active_atom=AtomId(0),
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
            residual_cyclic_policy_decision=residual_decision,
        )

        self.assertTrue(policy.residual_cyclic_evidence_is_retained)
        self.assertEqual(
            policy.support_dead_closure_open_vs_cyclic_tree_entry_groups,
            residual_decision.support_dead_groups,
        )
        self.assertEqual(
            policy.resolved_residual_attachment_policy_groups,
            residual_decision.support_dead_groups,
        )
        self.assertEqual(
            policy.residual_cyclic_support_dead_groups,
            residual_decision.support_dead_groups,
        )

    def test_active_emitted_graph_policy_retained_missing_evidence_decision_drives_unresolved_evidence(self) -> None:
        closure_decision, child_surface = (
            self._test_residual_cyclic_policy_inputs(
                include_closure_emission=False,
            )
        )
        residual_decision = writer_transitions._residual_cyclic_policy_decision(
            closure_decision,
            child_surface,
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .UNRESOLVED_RESIDUAL_ATTACHMENT_CHOICE
            ),
            active_atom=AtomId(0),
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
            residual_cyclic_policy_decision=residual_decision,
        )

        self.assertEqual(
            policy.unresolved_closure_open_vs_cyclic_tree_entry_groups,
            residual_decision.missing_evidence_groups,
        )
        self.assertEqual(
            policy.unresolved_residual_attachment_policy_groups,
            residual_decision.missing_evidence_groups,
        )
        self.assertEqual(
            policy.residual_cyclic_missing_evidence_groups,
            residual_decision.missing_evidence_groups,
        )

    def test_active_emitted_graph_policy_retained_unsupported_owner_decision_drives_owner_scope_evidence(self) -> None:
        closure_decision, child_surface = (
            self._test_residual_cyclic_policy_inputs(
                closure_owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
                child_owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
            )
        )
        residual_decision = writer_transitions._residual_cyclic_policy_decision(
            closure_decision,
            child_surface,
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .UNSUPPORTED_OWNER_SCOPE_RESIDUAL_ATTACHMENT_CHOICE
            ),
            active_atom=AtomId(0),
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
            residual_cyclic_policy_decision=residual_decision,
        )
        group = residual_decision.unsupported_owner_scope_groups[0]

        self.assertEqual(
            (
                policy
                .unsupported_owner_scope_closure_open_vs_cyclic_tree_entry_groups
            ),
            residual_decision.unsupported_owner_scope_groups,
        )
        self.assertEqual(
            policy.unsupported_owner_scope_residual_attachment_policy_groups,
            residual_decision.unsupported_owner_scope_groups,
        )
        self.assertEqual(
            policy.unsupported_owner_scope_kinds,
            (group.closure_open_vs_cyclic_tree_entry_owner_scope_kind,),
        )

    def test_active_emitted_graph_policy_retained_none_decision_yields_no_residual_cyclic_evidence(self) -> None:
        closure_decision, child_surface = (
            self._test_residual_cyclic_policy_inputs(
                child_attachment_action_kind=(
                    WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY
                ),
            )
        )
        residual_decision = writer_transitions._residual_cyclic_policy_decision(
            closure_decision,
            child_surface,
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD
            ),
            active_atom=AtomId(0),
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
            residual_cyclic_policy_decision=residual_decision,
        )

        self.assertTrue(policy.residual_cyclic_evidence_is_retained)
        self.assertEqual(
            policy.considered_closure_open_vs_cyclic_tree_entry_groups,
            (),
        )
        self.assertEqual(
            policy.support_dead_closure_open_vs_cyclic_tree_entry_groups,
            (),
        )
        self.assertEqual(
            policy.unresolved_closure_open_vs_cyclic_tree_entry_groups,
            (),
        )
        self.assertEqual(
            (
                policy
                .unsupported_owner_scope_closure_open_vs_cyclic_tree_entry_groups
            ),
            (),
        )
        self.assertEqual(policy.resolved_residual_attachment_policy_groups, ())

    def test_active_emitted_graph_policy_residual_evidence_falls_back_without_retained_decision(self) -> None:
        unsupported_closure_decision, unsupported_child_surface = (
            self._test_residual_cyclic_policy_inputs(
                closure_owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
                child_owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
            )
        )
        unsupported_policy = (
            writer_transitions._WriterActiveEmittedGraphPolicyDecision(
                kind=(
                    writer_transitions
                    ._WriterActiveEmittedGraphPolicyDecisionKind
                    .UNSUPPORTED_OWNER_SCOPE_RESIDUAL_ATTACHMENT_CHOICE
                ),
                active_atom=AtomId(0),
                closure_endpoint_decision=unsupported_closure_decision,
                child_schedule_surface=unsupported_child_surface,
            )
        )
        missing_closure_decision, missing_child_surface = (
            self._test_residual_cyclic_policy_inputs(
                include_closure_emission=False,
            )
        )
        missing_policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .UNRESOLVED_RESIDUAL_ATTACHMENT_CHOICE
            ),
            active_atom=AtomId(0),
            closure_endpoint_decision=missing_closure_decision,
            child_schedule_surface=missing_child_surface,
        )
        support_dead_closure_decision, support_dead_child_surface = (
            self._test_residual_cyclic_policy_inputs()
        )
        support_dead_policy = (
            writer_transitions._WriterActiveEmittedGraphPolicyDecision(
                kind=(
                    writer_transitions
                    ._WriterActiveEmittedGraphPolicyDecisionKind
                    .ACTIVE_CHILD_AFTER_DEAD_CLOSURE_OPEN
                ),
                active_atom=AtomId(0),
                closure_endpoint_decision=support_dead_closure_decision,
                child_schedule_surface=support_dead_child_surface,
            )
        )

        self.assertFalse(
            unsupported_policy.residual_cyclic_evidence_is_retained
        )
        self.assertEqual(
            (
                unsupported_policy
                .unsupported_owner_scope_residual_attachment_policy_groups
            ),
            (
                unsupported_policy
                .unsupported_owner_scope_closure_open_vs_cyclic_tree_entry_groups
            ),
        )
        self.assertTrue(
            (
                unsupported_policy
                .unsupported_owner_scope_residual_attachment_policy_groups
            )
        )
        self.assertFalse(missing_policy.residual_cyclic_evidence_is_retained)
        self.assertEqual(
            missing_policy.unresolved_residual_attachment_policy_groups,
            missing_policy.missing_closure_open_support_evidence_groups,
        )
        self.assertTrue(
            missing_policy.unresolved_residual_attachment_policy_groups
        )
        self.assertFalse(
            support_dead_policy.residual_cyclic_evidence_is_retained
        )
        self.assertEqual(
            support_dead_policy.resolved_residual_attachment_policy_groups,
            (
                support_dead_policy
                .support_dead_closure_open_vs_cyclic_tree_entry_groups
            ),
        )
        self.assertTrue(
            support_dead_policy.resolved_residual_attachment_policy_groups
        )

    def test_active_emitted_graph_policy_production_retains_residual_evidence(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(0)
        closure_decision, child_surface = self._test_residual_cyclic_policy_inputs()

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            return_value=child_surface,
        ):
            policy = writer_transitions._active_emitted_graph_policy_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIsNotNone(policy.residual_cyclic_policy_decision)
        self.assertEqual(
            policy.support_dead_closure_open_vs_cyclic_tree_entry_groups,
            policy.residual_cyclic_policy_decision.support_dead_groups,
        )
        self.assertEqual(
            policy.resolved_residual_attachment_policy_groups,
            (
                policy.residual_cyclic_policy_decision
                .resolved_residual_attachment_policy_groups
            ),
        )

    def test_active_emitted_graph_policy_retains_plain_active_child_residual_decision(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(0)
        closure_decision, child_surface = (
            self._test_residual_cyclic_policy_inputs(
                child_attachment_action_kind=(
                    WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY
                ),
            )
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            return_value=child_surface,
        ):
            policy = writer_transitions._active_emitted_graph_policy_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            policy.kind,
            writer_transitions._WriterActiveEmittedGraphPolicyDecisionKind.ACTIVE_CHILD,
        )
        self.assertIsNotNone(policy.residual_cyclic_policy_decision)
        self.assertIs(
            policy.residual_cyclic_policy_kind,
            writer_transitions._WriterResidualCyclicPolicyDecisionKind.NONE,
        )
        self.assertEqual(policy.residual_cyclic_choice_groups, ())

    def test_active_emitted_graph_policy_retains_dead_closure_residual_decision(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(0)
        closure_decision, child_surface = self._test_residual_cyclic_policy_inputs()

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            return_value=child_surface,
        ):
            policy = writer_transitions._active_emitted_graph_policy_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            policy.kind,
            (
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD_AFTER_DEAD_CLOSURE_OPEN
            ),
        )
        self.assertIsNotNone(policy.residual_cyclic_policy_decision)
        self.assertIs(
            policy.residual_cyclic_policy_kind,
            (
                writer_transitions
                ._WriterResidualCyclicPolicyDecisionKind
                .ACTIVE_CHILD_AFTER_DEAD_CLOSURE_OPEN
            ),
        )
        self.assertEqual(
            policy.residual_cyclic_support_dead_groups,
            policy.support_dead_closure_open_vs_cyclic_tree_entry_groups,
        )

    def test_active_emitted_graph_policy_does_not_compute_residual_decision_for_closure_selected_path(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(0)
        closure_decision, _, _ = self._test_closure_endpoint_decision(
            active_atom=active_atom,
            pair_survives=True,
            open_survives=False,
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            side_effect=AssertionError("child surface should not be computed"),
        ), patch(
            "grimace._south_star1.writer_transitions._residual_cyclic_policy_decision",
            side_effect=AssertionError("residual decision should not be computed"),
        ):
            policy = writer_transitions._active_emitted_graph_policy_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            policy.kind,
            (
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .CLOSURE_ENDPOINT
            ),
        )
        self.assertIsNone(policy.residual_cyclic_policy_decision)
        self.assertIs(
            policy.residual_cyclic_policy_kind,
            writer_transitions._WriterResidualCyclicPolicyDecisionKind.NONE,
        )
        self.assertEqual(policy.residual_cyclic_choice_groups, ())

    def test_active_emitted_graph_policy_does_not_compute_residual_decision_for_blocked_child_path(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(0)
        closure_decision, _, _ = self._test_closure_endpoint_decision(
            active_atom=active_atom,
            pair_survives=False,
            open_survives=False,
        )
        child_blocker = writer_transitions._WriterChildObligationBlocker(
            kind=(
                writer_transitions
                ._WriterChildObligationBlockerKind
                .MULTI_INCIDENCE_RESIDUAL_ATTACHMENT
            ),
            atom=active_atom,
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(child_blocker,),
            child_obligations=(),
            scheduled_actions=(),
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            return_value=child_surface,
        ), patch(
            "grimace._south_star1.writer_transitions._residual_cyclic_policy_decision",
            side_effect=AssertionError("residual decision should not be computed"),
        ):
            policy = writer_transitions._active_emitted_graph_policy_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            policy.kind,
            writer_transitions._WriterActiveEmittedGraphPolicyDecisionKind.BLOCKED_CHILD,
        )
        self.assertIsNone(policy.residual_cyclic_policy_decision)
        self.assertIs(
            policy.residual_cyclic_policy_kind,
            writer_transitions._WriterResidualCyclicPolicyDecisionKind.NONE,
        )
        self.assertEqual(policy.residual_cyclic_choice_groups, ())

    def test_active_emitted_graph_policy_allows_child_after_dead_closure_open_support(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        open_emission = writer_transitions._WriterScheduledActionEmission(
            action=open_action,
            transitions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(open_emission,),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            return_value=child_surface,
        ):
            decision = writer_transitions._active_emitted_graph_policy_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            decision.kind,
            (
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD_AFTER_DEAD_CLOSURE_OPEN
            ),
        )
        self.assertTrue(decision.emits_child_actions)
        self.assertIs(decision.child_schedule_surface, child_surface)
        self.assertTrue(decision.considered_cyclic_tree_entry_available)
        self.assertTrue(
            decision.support_dead_closure_open_vs_cyclic_tree_entry_groups
        )
        self.assertEqual(
            (
                decision
                .unsupported_owner_scope_closure_open_vs_cyclic_tree_entry_groups
            ),
            (),
        )
        self.assertEqual(
            decision.resolved_residual_attachment_policy_groups,
            decision.support_dead_closure_open_vs_cyclic_tree_entry_groups,
        )
        self.assertEqual(
            decision.unresolved_residual_attachment_policy_groups,
            (),
        )
        self.assertEqual(
            decision.unsupported_owner_scope_residual_attachment_policy_groups,
            (),
        )
        self.assertEqual(decision.unsupported_owner_scope_kinds, ())
        self.assertEqual(
            decision.missing_closure_open_support_evidence_groups,
            (),
        )
        self.assertEqual(
            decision.unresolved_closure_open_vs_cyclic_tree_entry_groups,
            (),
        )
        self.assertEqual(
            decision.child_scheduled_actions,
            child_surface.scheduled_actions,
        )

    def test_active_emitted_graph_policy_still_resolves_dead_closure_open_via_child_selection_evidence(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(0)
        closure_decision = self._test_closure_endpoint_decision_for_actions(
            active_atom,
            open_actions=(
                self._test_closure_open_action(
                    active_atom,
                    attachment_id=11,
                ),
            ),
        )
        cyclic_action = self._test_child_action(
            active_atom,
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(cyclic_action.child_obligation,),
            scheduled_actions=(cyclic_action,),
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            return_value=child_surface,
        ):
            policy = writer_transitions._active_emitted_graph_policy_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            policy.kind,
            (
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD_AFTER_DEAD_CLOSURE_OPEN
            ),
        )
        self.assertTrue(policy.considered_cyclic_tree_entry_available)
        self.assertTrue(
            policy.support_dead_closure_open_vs_cyclic_tree_entry_groups
        )

    def test_active_emitted_graph_policy_records_branch_return_owned_dead_closure_choice_as_unsupported_owner_scope(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        open_emission = writer_transitions._WriterScheduledActionEmission(
            action=open_action,
            transitions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(open_emission,),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            return_value=child_surface,
        ):
            decision = writer_transitions._active_emitted_graph_policy_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            decision.kind,
            (
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .UNSUPPORTED_OWNER_SCOPE_RESIDUAL_ATTACHMENT_CHOICE
            ),
        )
        self.assertEqual(decision.resolved_residual_attachment_policy_groups, ())
        self.assertTrue(
            (
                decision
                .unsupported_owner_scope_closure_open_vs_cyclic_tree_entry_groups
            )
        )
        self.assertTrue(
            decision.unsupported_owner_scope_residual_attachment_policy_groups
        )
        self.assertEqual(
            decision.missing_closure_open_support_evidence_groups,
            (),
        )
        self.assertEqual(decision.unresolved_residual_attachment_policy_groups, ())
        self.assertTrue(decision.graph_policy_blocked)
        self.assertEqual(decision.blocked, decision.graph_policy_blocked)
        self.assertEqual(decision.child_scheduled_actions, ())

    def test_active_emitted_graph_policy_records_mixed_owner_dead_closure_choice_as_unsupported_owner_scope(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        open_emission = writer_transitions._WriterScheduledActionEmission(
            action=open_action,
            transitions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(open_emission,),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            return_value=child_surface,
        ):
            decision = writer_transitions._active_emitted_graph_policy_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            decision.kind,
            (
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .UNSUPPORTED_OWNER_SCOPE_RESIDUAL_ATTACHMENT_CHOICE
            ),
        )
        self.assertTrue(
            (
                decision
                .unsupported_owner_scope_closure_open_vs_cyclic_tree_entry_groups
            )
        )
        self.assertTrue(
            decision.unsupported_owner_scope_residual_attachment_policy_groups
        )
        self.assertEqual(
            decision.unresolved_residual_attachment_policy_groups,
            (),
        )
        self.assertEqual(decision.child_scheduled_actions, ())

    def test_active_emitted_graph_policy_reports_unsupported_owner_scope_kind(self) -> None:
        cases = (
            (
                WriterBoundaryOwnerKind.BRANCH_RETURN,
                WriterBoundaryOwnerKind.BRANCH_RETURN,
                (
                    writer_transitions
                    ._WriterResidualAttachmentOwnerScopeKind
                    .BRANCH_RETURN
                ),
            ),
            (
                WriterBoundaryOwnerKind.PENDING_PARENT,
                WriterBoundaryOwnerKind.PENDING_PARENT,
                (
                    writer_transitions
                    ._WriterResidualAttachmentOwnerScopeKind
                    .PENDING_PARENT
                ),
            ),
            (
                WriterBoundaryOwnerKind.OPEN_RING_ENDPOINT,
                WriterBoundaryOwnerKind.OPEN_RING_ENDPOINT,
                (
                    writer_transitions
                    ._WriterResidualAttachmentOwnerScopeKind
                    .OPEN_RING_ENDPOINT
                ),
            ),
            (
                WriterBoundaryOwnerKind.UNOWNED,
                WriterBoundaryOwnerKind.UNOWNED,
                writer_transitions._WriterResidualAttachmentOwnerScopeKind.UNOWNED,
            ),
            (
                None,
                None,
                writer_transitions._WriterResidualAttachmentOwnerScopeKind.MISSING,
            ),
            (
                WriterBoundaryOwnerKind.ACTIVE_ATOM,
                WriterBoundaryOwnerKind.BRANCH_RETURN,
                writer_transitions._WriterResidualAttachmentOwnerScopeKind.MIXED,
            ),
            (
                WriterBoundaryOwnerKind.ACTIVE_ATOM,
                None,
                writer_transitions._WriterResidualAttachmentOwnerScopeKind.MIXED,
            ),
        )

        for closure_owner, child_owner, expected in cases:
            with self.subTest(
                closure_owner=closure_owner,
                child_owner=child_owner,
            ):
                decision = (
                    self
                    ._dead_closure_open_graph_policy_decision_for_owner_scope(
                        closure_owner_kind=closure_owner,
                        child_owner_kind=child_owner,
                    )
                )

                self.assertIs(
                    decision.kind,
                    (
                        writer_transitions
                        ._WriterActiveEmittedGraphPolicyDecisionKind
                        .UNSUPPORTED_OWNER_SCOPE_RESIDUAL_ATTACHMENT_CHOICE
                    ),
                )
                self.assertEqual(
                    decision.unsupported_owner_scope_kinds,
                    (expected,),
                )
                blockers = decision.graph_policy_blockers
                self.assertEqual(len(blockers), 1)
                self.assertIs(
                    blockers[0].residual_attachment_owner_scope_kind,
                    expected,
                )
                self.assertEqual(decision.child_scheduled_actions, ())

    def test_active_emitted_graph_policy_remains_unresolved_without_closure_open_emission_evidence(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            return_value=child_surface,
        ):
            decision = writer_transitions._active_emitted_graph_policy_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            decision.kind,
            (
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .UNRESOLVED_RESIDUAL_ATTACHMENT_CHOICE
            ),
        )
        self.assertTrue(decision.unresolved_residual_attachment_policy_groups)
        self.assertTrue(decision.missing_closure_open_support_evidence_groups)
        blockers = decision.graph_policy_blockers
        self.assertTrue(decision.graph_policy_blocked)
        self.assertEqual(decision.blocked, decision.graph_policy_blocked)
        self.assertEqual(len(blockers), 1)
        self.assertIs(
            blockers[0].kind,
            (
                writer_transitions
                ._WriterActiveEmittedGraphPolicyBlockerKind
                .MISSING_CLOSURE_OPEN_SUPPORT_EVIDENCE
            ),
        )
        self.assertEqual(
            blockers[0].residual_group,
            decision.unresolved_residual_attachment_policy_groups[0],
        )
        self.assertEqual(
            decision.unsupported_owner_scope_residual_attachment_policy_groups,
            (),
        )
        self.assertEqual(
            decision.support_dead_closure_open_vs_cyclic_tree_entry_groups,
            (),
        )
        self.assertEqual(decision.child_scheduled_actions, ())

    def test_active_emitted_schedule_decision_emits_child_after_dead_closure_open_support(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        open_emission = writer_transitions._WriterScheduledActionEmission(
            action=open_action,
            transitions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(open_emission,),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(child_action,),
            emissions=(),
            surviving_emissions=(),
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            return_value=child_surface,
        ), patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            return_value=child_batch,
        ) as emission_batch:
            decision = writer_transitions._active_emitted_schedule_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            decision.kind,
            writer_transitions._WriterActiveEmittedScheduleDecisionKind.ACTIVE_CHILD,
        )
        self.assertIs(
            decision.graph_policy_decision.kind,
            (
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD_AFTER_DEAD_CLOSURE_OPEN
            ),
        )
        self.assertTrue(decision.graph_policy_decision.emits_child_actions)
        self.assertTrue(
            (
                decision.graph_policy_decision
                .support_dead_closure_open_vs_cyclic_tree_entry_groups
            )
        )
        emission_batch.assert_called_once_with(
            prepared,
            state,
            context,
            child_surface.scheduled_actions,
        )

    def test_active_emitted_graph_policy_allows_child_when_closure_open_and_cyclic_tree_use_different_attachments(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        open_emission = writer_transitions._WriterScheduledActionEmission(
            action=open_action,
            transitions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(open_emission,),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=12,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            return_value=child_surface,
        ):
            decision = writer_transitions._active_emitted_graph_policy_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            decision.kind,
            writer_transitions._WriterActiveEmittedGraphPolicyDecisionKind.ACTIVE_CHILD,
        )
        self.assertIsNot(
            decision.kind,
            (
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD_AFTER_DEAD_CLOSURE_OPEN
            ),
        )
        self.assertTrue(decision.emits_child_actions)
        self.assertEqual(decision.resolved_residual_attachment_policy_groups, ())
        self.assertEqual(decision.graph_policy_blockers, ())
        self.assertFalse(decision.graph_policy_blocked)
        self.assertEqual(
            decision.child_scheduled_actions,
            child_surface.scheduled_actions,
        )
        self.assertEqual(
            decision.considered_closure_open_vs_cyclic_tree_entry_groups,
            (),
        )

    def test_active_emitted_graph_policy_does_not_treat_acyclic_child_as_cyclic_residual_choice(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(0)
        closure_decision = self._test_closure_endpoint_decision_for_actions(
            active_atom,
            open_actions=(
                self._test_closure_open_action(
                    active_atom,
                    attachment_id=11,
                ),
            ),
        )
        acyclic_action = self._test_child_action(
            active_atom,
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY
            ),
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(acyclic_action.child_obligation,),
            scheduled_actions=(acyclic_action,),
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            return_value=child_surface,
        ):
            policy = writer_transitions._active_emitted_graph_policy_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            policy.kind,
            writer_transitions._WriterActiveEmittedGraphPolicyDecisionKind.ACTIVE_CHILD,
        )
        self.assertIs(
            policy.considered_active_child_selection_kind,
            writer_transitions._WriterActiveChildSelectionKind.ACYCLIC_TREE_ENTRY,
        )
        self.assertFalse(policy.considered_cyclic_tree_entry_available)
        self.assertEqual(
            policy.considered_closure_open_vs_cyclic_tree_entry_groups,
            (),
        )

    def test_active_emitted_graph_policy_selects_plain_active_child_without_resolved_residual_choice(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(0)
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=12,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            return_value=child_surface,
        ):
            decision = writer_transitions._active_emitted_graph_policy_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            decision.kind,
            writer_transitions._WriterActiveEmittedGraphPolicyDecisionKind.ACTIVE_CHILD,
        )
        self.assertTrue(decision.emits_child_actions)
        self.assertEqual(decision.resolved_residual_attachment_policy_groups, ())

    def test_active_emitted_graph_policy_rejects_plain_active_child_for_dead_closure_open_choice(self) -> None:
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        open_emission = writer_transitions._WriterScheduledActionEmission(
            action=open_action,
            transitions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(open_emission,),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )

        with self.assertRaises(SouthStarError) as raised:
            writer_transitions._WriterActiveEmittedGraphPolicyDecision(
                kind=(
                    writer_transitions
                    ._WriterActiveEmittedGraphPolicyDecisionKind
                    .ACTIVE_CHILD
                ),
                active_atom=active_atom,
                closure_endpoint_decision=closure_decision,
                child_schedule_surface=child_surface,
            )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.INTERNAL_INVARIANT)

    def test_active_emitted_child_decision_rejects_unresolved_residual_policy_metadata(self) -> None:
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        blocked_child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            blocked_child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(blocked_child,),
            scheduled_actions=(child_action,),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .UNRESOLVED_RESIDUAL_ATTACHMENT_CHOICE
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(child_action,),
            emissions=(),
            surviving_emissions=(),
        )

        with self.assertRaises(SouthStarError) as raised:
            writer_transitions._active_emitted_child_decision(
                closure_endpoint_decision=closure_decision,
                child_schedule_surface=child_surface,
                child_batch=child_batch,
                graph_policy_decision=policy,
            )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.INTERNAL_INVARIANT)

    def test_active_emitted_graph_policy_chosen_residual_groups_use_chosen_surfaces(self) -> None:
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        open_emission = writer_transitions._WriterScheduledActionEmission(
            action=open_action,
            transitions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(open_emission,),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=12,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )

        chosen = policy.chosen_residual_attachment_policy_groups

        self.assertEqual(len(chosen), 1)
        self.assertEqual(chosen[0].closure_open_surfaces, ())
        self.assertEqual(
            chosen[0].cyclic_tree_entry_surfaces,
            child_surface.graph_action_surfaces,
        )

    def test_active_emitted_graph_policy_blocked_child_chooses_no_graph_surfaces(self) -> None:
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(
                    writer_transitions._WriterScheduledActionEmission(
                        action=open_action,
                        transitions=(),
                    ),
                ),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        blocker = writer_transitions._WriterChildObligationBlocker(
            kind=(
                writer_transitions._WriterChildObligationBlockerKind
                .MULTI_INCIDENCE_RESIDUAL_ATTACHMENT
            ),
            atom=active_atom,
            attachment_id=7,
            attachment_action_kind=WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(blocker,),
            child_obligations=(),
            scheduled_actions=(),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .BLOCKED_CHILD
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )

        self.assertTrue(policy.blocked)
        self.assertEqual(policy.chosen_graph_action_surfaces, ())
        self.assertEqual(policy.graph_action_surfaces, ())
        self.assertEqual(
            policy.considered_graph_action_surfaces,
            (
                *closure_decision.considered_graph_action_surfaces,
                *child_surface.graph_action_surfaces,
            ),
        )

    def test_active_emitted_schedule_decision_selects_surviving_closure_batch(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(7)
        closure_decision, pair_emission, open_emission = (
            self._test_closure_endpoint_decision(
                active_atom=active_atom,
                pair_survives=True,
                open_survives=False,
            )
        )
        pair_action = pair_emission.action
        open_action = open_emission.action

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ) as closure_schedule, patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            side_effect=AssertionError("child policy should not run"),
        ):
            decision = writer_transitions._active_emitted_schedule_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            decision.kind,
            writer_transitions._WriterActiveEmittedScheduleDecisionKind.CLOSURE_ENDPOINT,
        )
        self.assertEqual(
            decision.closure_batch.actions,
            (pair_action, open_action),
        )
        self.assertEqual(
            decision.closure_batch.emissions,
            (pair_emission, open_emission),
        )
        self.assertEqual(decision.closure_batch.surviving_emissions, (pair_emission,))
        self.assertIs(decision.selected_batch, decision.closure_batch)
        self.assertIsNone(decision.child_batch)
        self.assertIsNone(decision.child_schedule_surface)
        closure_schedule.assert_called_once_with(
            prepared,
            state,
            context,
        )

    def test_active_emitted_schedule_decision_uses_closure_endpoint_decision(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(7)
        closure_decision, pair_emission, open_emission = (
            self._test_closure_endpoint_decision(
                active_atom=active_atom,
                pair_survives=False,
                open_survives=True,
            )
        )
        pair_action = pair_emission.action
        open_action = open_emission.action

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ) as closure_schedule, patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            side_effect=AssertionError("child policy should not run"),
        ):
            decision = writer_transitions._active_emitted_schedule_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            decision.kind,
            writer_transitions._WriterActiveEmittedScheduleDecisionKind.CLOSURE_ENDPOINT,
        )
        self.assertEqual(
            decision.selected_batch.actions,
            (pair_action, open_action),
        )
        self.assertEqual(
            decision.selected_batch.emissions,
            (pair_emission, open_emission),
        )
        self.assertEqual(
            decision.selected_batch.surviving_emissions,
            (open_emission,),
        )
        closure_schedule.assert_called_once_with(
            prepared,
            state,
            context,
        )

    def test_active_emitted_schedule_decision_selects_child_batch_after_zero_closure_survivors(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(7)

        child_action = writer_transitions._finish_active_action(active_atom)
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(),
            scheduled_actions=(child_action,),
        )

        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=pair_batch,
            open_batch=open_batch,
            surviving_emissions=(),
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(child_action,),
            emissions=(object(),),  # type: ignore[arg-type]
            surviving_emissions=(object(),),  # type: ignore[arg-type]
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ) as closure_schedule, patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            return_value=child_surface,
        ) as child_surface_from_context, patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            return_value=child_batch,
        ) as emission_batch:
            decision = writer_transitions._active_emitted_schedule_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            decision.kind,
            writer_transitions._WriterActiveEmittedScheduleDecisionKind.ACTIVE_CHILD,
        )
        self.assertEqual(decision.closure_batch.actions, ())
        self.assertEqual(decision.closure_batch.emissions, ())
        self.assertEqual(decision.closure_batch.surviving_emissions, ())
        self.assertIs(decision.child_batch, child_batch)
        self.assertIs(decision.child_schedule_surface, child_surface)
        self.assertIs(decision.selected_batch, child_batch)
        closure_schedule.assert_called_once_with(
            prepared,
            state,
            context,
        )
        child_surface_from_context.assert_called_once_with(
            context,
            state,
            active_atom,
        )
        emission_batch.assert_called_once_with(
            prepared,
            state,
            context,
            (child_action,),
        )

    def test_active_emitted_schedule_decision_threads_child_surface_after_empty_closure_survivors(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(7)
        child_action = writer_transitions._finish_active_action(active_atom)
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(),
            scheduled_actions=(child_action,),
        )
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=pair_batch,
            open_batch=open_batch,
            surviving_emissions=(),
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(child_action,),
            emissions=(object(),),  # type: ignore[arg-type]
            surviving_emissions=(object(),),  # type: ignore[arg-type]
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            return_value=child_surface,
        ), patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            return_value=child_batch,
        ) as emission_batch:
            decision = writer_transitions._active_emitted_schedule_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            decision.kind,
            writer_transitions._WriterActiveEmittedScheduleDecisionKind.ACTIVE_CHILD,
        )
        self.assertIs(decision.child_schedule_surface, child_surface)
        emission_batch.assert_called_once_with(
            prepared,
            state,
            context,
            child_surface.scheduled_actions,
        )

    def test_active_emitted_schedule_decision_does_not_compute_child_surface_when_closure_survives(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(7)
        closure_decision, _, _ = self._test_closure_endpoint_decision(
            active_atom=active_atom,
            pair_survives=True,
            open_survives=False,
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            side_effect=AssertionError("child surface should not be computed"),
        ) as child_surface:
            decision = writer_transitions._active_emitted_schedule_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            decision.kind,
            writer_transitions._WriterActiveEmittedScheduleDecisionKind.CLOSURE_ENDPOINT,
        )
        self.assertIsNone(decision.child_schedule_surface)
        child_surface.assert_not_called()

    def test_active_emitted_schedule_decision_raises_from_child_surface_blockers_without_emitting_child_batch(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(7)
        blocker = writer_transitions._WriterChildObligationBlocker(
            kind=(
                writer_transitions._WriterChildObligationBlockerKind
                .MULTI_INCIDENCE_RESIDUAL_ATTACHMENT
            ),
            atom=active_atom,
            attachment_id=7,
            attachment_action_kind=WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(blocker,),
            child_obligations=(),
            scheduled_actions=(),
        )
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=pair_batch,
            open_batch=open_batch,
            surviving_emissions=(),
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            return_value=child_surface,
        ), patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            side_effect=AssertionError("child batch should not emit"),
        ) as emission_batch:
            with self.assertRaises(SouthStarError) as raised:
                writer_transitions._active_emitted_schedule_decision(
                    prepared,  # type: ignore[arg-type]
                    state,  # type: ignore[arg-type]
                    context,  # type: ignore[arg-type]
                    active_atom,
                )

        self.assertIs(
            raised.exception.kind,
            SouthStarErrorKind.UNSUPPORTED_POLICY,
        )
        emission_batch.assert_not_called()

    def test_active_emitted_schedule_decision_raises_from_blocked_graph_policy_without_child_emission(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(7)
        blocker = writer_transitions._WriterChildObligationBlocker(
            kind=(
                writer_transitions._WriterChildObligationBlockerKind
                .MULTI_INCIDENCE_RESIDUAL_ATTACHMENT
            ),
            atom=active_atom,
            attachment_id=7,
            attachment_action_kind=WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(blocker,),
            child_obligations=(),
            scheduled_actions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
        )
        policy_decision = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .BLOCKED_CHILD
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )

        with patch(
            "grimace._south_star1.writer_transitions._active_emitted_graph_policy_decision",
            return_value=policy_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            side_effect=AssertionError("child batch should not emit"),
        ) as emission_batch:
            with self.assertRaises(SouthStarError) as raised:
                writer_transitions._active_emitted_schedule_decision(
                    prepared,  # type: ignore[arg-type]
                    state,  # type: ignore[arg-type]
                    context,  # type: ignore[arg-type]
                    active_atom,
                )

        self.assertIs(
            raised.exception.kind,
            SouthStarErrorKind.UNSUPPORTED_POLICY,
        )
        emission_batch.assert_not_called()

    def test_active_emitted_schedule_decision_emits_child_actions_from_graph_policy(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(7)
        child_action = writer_transitions._finish_active_action(active_atom)
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(),
            scheduled_actions=(child_action,),  # type: ignore[arg-type]
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
        )
        policy_decision = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(child_action,),  # type: ignore[arg-type]
            emissions=(object(),),  # type: ignore[arg-type]
            surviving_emissions=(object(),),  # type: ignore[arg-type]
        )

        with patch(
            "grimace._south_star1.writer_transitions._active_emitted_graph_policy_decision",
            return_value=policy_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            return_value=child_batch,
        ) as emission_batch:
            decision = writer_transitions._active_emitted_schedule_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        emission_batch.assert_called_once_with(
            prepared,
            state,
            context,
            policy_decision.child_scheduled_actions,
        )
        self.assertIs(
            decision.kind,
            writer_transitions._WriterActiveEmittedScheduleDecisionKind.ACTIVE_CHILD,
        )
        self.assertIs(decision.child_schedule_surface, child_surface)
        self.assertIs(decision.child_batch, child_batch)

    def test_active_emitted_schedule_decision_threads_closure_graph_policy_decision(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(7)
        closure_decision, _, _ = self._test_closure_endpoint_decision(
            active_atom=active_atom,
            pair_survives=True,
            open_survives=False,
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .CLOSURE_ENDPOINT
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
        )

        with patch(
            "grimace._south_star1.writer_transitions._active_emitted_graph_policy_decision",
            return_value=policy,
        ):
            decision = writer_transitions._active_emitted_schedule_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            decision.kind,
            writer_transitions._WriterActiveEmittedScheduleDecisionKind.CLOSURE_ENDPOINT,
        )
        self.assertIs(decision.graph_policy_decision, policy)

    def test_active_emitted_schedule_decision_threads_child_graph_policy_decision(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(7)
        child_action = writer_transitions._finish_active_action(active_atom)
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(),
            scheduled_actions=(child_action,),  # type: ignore[arg-type]
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(child_action,),  # type: ignore[arg-type]
            emissions=(object(),),  # type: ignore[arg-type]
            surviving_emissions=(object(),),  # type: ignore[arg-type]
        )

        with patch(
            "grimace._south_star1.writer_transitions._active_emitted_graph_policy_decision",
            return_value=policy,
        ), patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            return_value=child_batch,
        ) as emission_batch:
            decision = writer_transitions._active_emitted_schedule_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            decision.kind,
            writer_transitions._WriterActiveEmittedScheduleDecisionKind.ACTIVE_CHILD,
        )
        self.assertIs(decision.graph_policy_decision, policy)
        emission_batch.assert_called_once_with(
            prepared,
            state,
            context,
            policy.child_scheduled_actions,
        )

    def test_active_emitted_scheduler_does_not_compute_children_when_closure_transition_survives(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(7)
        closure_decision, surviving_closure_emission, _ = (
            self._test_closure_endpoint_decision(
                active_atom=active_atom,
                pair_survives=True,
                open_survives=False,
            )
        )
        closure_transition = object()

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ) as closure_schedule, patch(
            "grimace._south_star1.writer_transitions._transitions_from_scheduled_action_emissions",
            return_value=(closure_transition,),
        ) as flatten_emissions, patch(
            "grimace._south_star1.writer_transitions._child_obligation_blockers_for_atom",
            side_effect=AssertionError("child blockers were computed too early"),
        ), patch(
            "grimace._south_star1.writer_transitions._unblocked_child_obligations_from_context",
            side_effect=AssertionError("child obligations were computed too early"),
        ):
            result = writer_transitions._active_emitted_transitions(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertEqual(result, (closure_transition,))
        closure_schedule.assert_called_once_with(
            prepared,
            state,
            context,
        )
        flatten_emissions.assert_called_once_with((surviving_closure_emission,))

    def test_active_emitted_scheduler_computes_children_after_empty_closure_transitions(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(7)
        child_obligation = object()
        child_action = writer_transitions._finish_active_action(active_atom)
        child_emission = object()
        surviving_child_emission = object()
        child_transition = object()
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=pair_batch,
            open_batch=open_batch,
            surviving_emissions=(),
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(child_action,),
            emissions=(child_emission,),  # type: ignore[arg-type]
            surviving_emissions=(surviving_child_emission,),  # type: ignore[arg-type]
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ) as closure_schedule, patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            return_value=child_batch,
        ) as emission_batch, patch(
            "grimace._south_star1.writer_transitions._transitions_from_scheduled_action_emissions",
            return_value=(child_transition,),
        ) as flatten_emissions, patch(
            "grimace._south_star1.writer_transitions._child_obligation_blockers_for_atom",
            return_value=(),
        ) as child_blockers, patch(
            "grimace._south_star1.writer_transitions._unblocked_child_obligations_from_context",
            return_value=(child_obligation,),
        ) as child_obligations, patch(
            "grimace._south_star1.writer_transitions._active_child_scheduled_actions",
            return_value=(child_action,),
        ) as child_actions:
            result = writer_transitions._active_emitted_transitions(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertEqual(result, (child_transition,))
        closure_schedule.assert_called_once_with(
            prepared,
            state,
            context,
        )
        emission_batch.assert_called_once_with(
            prepared,
            state,
            context,
            (child_action,),
        )
        flatten_emissions.assert_called_once_with(
            (surviving_child_emission,),
        )
        child_blockers.assert_called_once_with(
            context,
            active_atom,
        )
        child_obligations.assert_called_once_with(
            context,
            state,
            active_atom,
        )
        child_actions.assert_called_once_with(
            active_atom,
            (child_obligation,),
        )

    def test_active_emitted_child_fallback_returns_empty_when_no_child_emissions_survive(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(7)

        child_obligation = object()
        child_action = writer_transitions._finish_active_action(active_atom)

        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=pair_batch,
            open_batch=open_batch,
            surviving_emissions=(),
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(child_action,),
            emissions=(),
            surviving_emissions=(),
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            return_value=child_batch,
        ), patch(
            "grimace._south_star1.writer_transitions._transitions_from_scheduled_action_emissions",
            return_value=(),
        ) as flatten_emissions, patch(
            "grimace._south_star1.writer_transitions._child_obligation_blockers_for_atom",
            return_value=(),
        ), patch(
            "grimace._south_star1.writer_transitions._unblocked_child_obligations_from_context",
            return_value=(child_obligation,),
        ), patch(
            "grimace._south_star1.writer_transitions._active_child_scheduled_actions",
            return_value=(child_action,),
        ):
            result = writer_transitions._active_emitted_transitions(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertEqual(result, ())
        flatten_emissions.assert_called_once_with(())

    def test_scheduled_action_rejects_finish_payload(self) -> None:
        child = writer_transitions._WriterChildObligation(
            bond=BondId(0),
            child=AtomId(1),
        )

        with self.assertRaises(SouthStarError):
            writer_transitions._WriterScheduledAction(
                kind=writer_transitions._WriterScheduledActionKind.FINISH_ACTIVE,
                parent=AtomId(0),
                child_obligation=child,
            )

    def test_scheduled_action_requires_child_payload_for_child_action(self) -> None:
        with self.assertRaises(SouthStarError):
            writer_transitions._WriterScheduledAction(
                kind=writer_transitions._WriterScheduledActionKind.ENTER_INLINE_CHILD,
                parent=AtomId(0),
            )

    def test_scheduled_action_rejects_root_atom_payload(self) -> None:
        child = writer_transitions._WriterChildObligation(
            bond=BondId(0),
            child=AtomId(1),
        )

        with self.assertRaises(SouthStarError):
            writer_transitions._WriterScheduledAction(
                kind=writer_transitions._WriterScheduledActionKind.EMIT_ROOT_ATOM,
                parent=AtomId(0),
                child_obligation=child,
            )

    def test_scheduled_action_accepts_root_atom_action(self) -> None:
        action = writer_transitions._emit_root_atom_action(AtomId(0))

        self.assertIs(
            action.kind,
            writer_transitions._WriterScheduledActionKind.EMIT_ROOT_ATOM,
        )
        self.assertEqual(action.parent, AtomId(0))

    def test_scheduled_action_rejects_wrong_payload_family(self) -> None:
        endpoint = WriterOpenClosureEndpoint(
            bond=BondId(0),
            first_atom=AtomId(0),
            second_atom=AtomId(1),
            label=WriterClosureLabel(value=1, text="1"),
            first_endpoint_text="1",
            first_endpoint_bond_text="",
        )
        closure = WriterClosedClosure(
            bond=BondId(0),
            first_atom=AtomId(0),
            second_atom=AtomId(1),
            label=endpoint.label,
            first_endpoint_text="1",
            second_endpoint_text="1",
            first_endpoint_bond_text="",
            second_endpoint_bond_text="",
        )
        pair = writer_transitions._WriterClosurePairObligation(
            endpoint=endpoint,
            closure=closure,
        )

        with self.assertRaises(SouthStarError):
            writer_transitions._WriterScheduledAction(
                kind=writer_transitions._WriterScheduledActionKind.OPEN_BRANCH,
                parent=AtomId(0),
                closure_pair_obligation=pair,
            )

    def test_scheduled_action_requires_closure_open_label(self) -> None:
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(0),
            first_atom=AtomId(0),
            second_atom=AtomId(1),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )

        with self.assertRaises(SouthStarError):
            writer_transitions._WriterScheduledAction(
                kind=writer_transitions._WriterScheduledActionKind.OPEN_CLOSURE_ENDPOINT,
                parent=AtomId(0),
                closure_open_obligation=open_obligation,
            )

    def test_scheduled_action_accepts_valid_payloads(self) -> None:
        child = writer_transitions._WriterChildObligation(
            bond=BondId(0),
            child=AtomId(1),
        )
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=AtomId(0),
            second_atom=AtomId(2),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        label = WriterClosureLabel(value=1, text="1")
        endpoint = WriterOpenClosureEndpoint(
            bond=BondId(2),
            first_atom=AtomId(0),
            second_atom=AtomId(3),
            label=label,
            first_endpoint_text="1",
            first_endpoint_bond_text="",
        )
        closure = WriterClosedClosure(
            bond=BondId(2),
            first_atom=AtomId(0),
            second_atom=AtomId(3),
            label=label,
            first_endpoint_text="1",
            second_endpoint_text="1",
            first_endpoint_bond_text="",
            second_endpoint_bond_text="",
        )
        pair = writer_transitions._WriterClosurePairObligation(
            endpoint=endpoint,
            closure=closure,
        )

        writer_transitions._WriterScheduledAction(
            kind=writer_transitions._WriterScheduledActionKind.FINISH_ACTIVE,
            parent=AtomId(0),
        )
        writer_transitions._WriterScheduledAction(
            kind=writer_transitions._WriterScheduledActionKind.ENTER_INLINE_CHILD,
            parent=AtomId(0),
            child_obligation=child,
        )
        writer_transitions._WriterScheduledAction(
            kind=writer_transitions._WriterScheduledActionKind.OPEN_BRANCH,
            parent=AtomId(0),
            child_obligation=child,
        )
        writer_transitions._WriterScheduledAction(
            kind=writer_transitions._WriterScheduledActionKind.OPEN_CLOSURE_ENDPOINT,
            parent=AtomId(0),
            closure_open_obligation=open_obligation,
            closure_open_label=label,
        )
        writer_transitions._WriterScheduledAction(
            kind=writer_transitions._WriterScheduledActionKind.PAIR_CLOSURE_ENDPOINT,
            parent=AtomId(0),
            closure_pair_obligation=pair,
        )

    def test_scheduled_action_requires_pending_entry_payload(self) -> None:
        with self.assertRaises(SouthStarError):
            writer_transitions._WriterScheduledAction(
                kind=writer_transitions._WriterScheduledActionKind.CONSUME_PENDING_ENTRY,
                parent=AtomId(0),
            )

    def test_scheduled_action_accepts_pending_entry_payload(self) -> None:
        pending = PendingWriterEntry(
            parent=AtomId(0),
            child=AtomId(1),
            bond=BondId(0),
            branch=False,
        )

        action = writer_transitions._consume_pending_entry_action(pending)

        self.assertIs(
            action.kind,
            writer_transitions._WriterScheduledActionKind.CONSUME_PENDING_ENTRY,
        )
        self.assertEqual(action.parent, AtomId(0))
        self.assertEqual(action.pending_entry, pending)

    def test_scheduled_graph_action_surface_for_child_action_carries_incidence_metadata(self) -> None:
        child = writer_transitions._WriterChildObligation(
            bond=BondId(1),
            child=AtomId(2),
            boundary_atom=AtomId(0),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY
            ),
        )
        action = writer_transitions._enter_inline_child_action(
            AtomId(0),
            child,
        )

        surface = writer_transitions._scheduled_graph_action_surface(action)

        self.assertIs(
            surface.kind,
            writer_transitions._WriterScheduledActionKind.ENTER_INLINE_CHILD,
        )
        self.assertEqual(surface.active_atom, AtomId(0))
        self.assertEqual(surface.bond, BondId(1))
        self.assertEqual(surface.partner_atom, AtomId(2))
        self.assertEqual(surface.boundary_atom, AtomId(0))
        self.assertEqual(surface.attachment_id, 7)
        self.assertIs(
            surface.attachment_action_kind,
            WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY,
        )
        self.assertIs(surface.owner_kind, WriterBoundaryOwnerKind.ACTIVE_ATOM)
        self.assertFalse(surface.pending_entry)

    def test_scheduled_graph_action_surface_for_closure_open_carries_label_and_incidence_metadata(self) -> None:
        label = WriterClosureLabel(value=1, text="1")
        obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(3),
            first_atom=AtomId(0),
            second_atom=AtomId(4),
            attachment_id=9,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        action = writer_transitions._open_closure_endpoint_action(
            AtomId(0),
            obligation,
            label,
        )

        surface = writer_transitions._scheduled_graph_action_surface(action)

        self.assertIs(
            surface.kind,
            writer_transitions._WriterScheduledActionKind.OPEN_CLOSURE_ENDPOINT,
        )
        self.assertEqual(surface.active_atom, AtomId(0))
        self.assertEqual(surface.bond, BondId(3))
        self.assertEqual(surface.partner_atom, AtomId(4))
        self.assertEqual(surface.boundary_atom, AtomId(0))
        self.assertEqual(surface.attachment_id, 9)
        self.assertIs(
            surface.attachment_action_kind,
            WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY,
        )
        self.assertIs(surface.owner_kind, WriterBoundaryOwnerKind.ACTIVE_ATOM)
        self.assertIs(surface.closure_label, label)

    def test_scheduled_graph_action_surface_for_pending_entry_marks_pending_source(self) -> None:
        pending = PendingWriterEntry(
            parent=AtomId(0),
            child=AtomId(2),
            bond=BondId(1),
            branch=False,
        )
        action = writer_transitions._consume_pending_entry_action(pending)

        surface = writer_transitions._scheduled_graph_action_surface(action)

        self.assertIs(
            surface.kind,
            writer_transitions._WriterScheduledActionKind.CONSUME_PENDING_ENTRY,
        )
        self.assertEqual(surface.active_atom, AtomId(0))
        self.assertEqual(surface.bond, BondId(1))
        self.assertEqual(surface.partner_atom, AtomId(2))
        self.assertEqual(surface.boundary_atom, AtomId(0))
        self.assertTrue(surface.pending_entry)
        self.assertIsNone(surface.attachment_id)
        self.assertIsNone(surface.attachment_action_kind)
        self.assertIsNone(surface.owner_kind)

    def test_scheduled_graph_action_surface_for_closure_pair_carries_closure_label(self) -> None:
        label = WriterClosureLabel(value=1, text="1")
        endpoint = WriterOpenClosureEndpoint(
            bond=BondId(5),
            first_atom=AtomId(0),
            second_atom=AtomId(3),
            label=label,
            first_endpoint_text="1",
            first_endpoint_bond_text="",
        )
        closure = WriterClosedClosure(
            bond=BondId(5),
            first_atom=AtomId(0),
            second_atom=AtomId(3),
            label=label,
            first_endpoint_text="1",
            second_endpoint_text="1",
            first_endpoint_bond_text="",
            second_endpoint_bond_text="",
        )
        pair = writer_transitions._WriterClosurePairObligation(
            endpoint=endpoint,
            closure=closure,
        )
        action = writer_transitions._pair_closure_endpoint_action(
            AtomId(3),
            pair,
        )

        surface = writer_transitions._scheduled_graph_action_surface(action)

        self.assertIs(
            surface.kind,
            writer_transitions._WriterScheduledActionKind.PAIR_CLOSURE_ENDPOINT,
        )
        self.assertEqual(surface.active_atom, AtomId(3))
        self.assertEqual(surface.bond, BondId(5))
        self.assertEqual(surface.partner_atom, AtomId(0))
        self.assertEqual(surface.boundary_atom, AtomId(3))
        self.assertIs(surface.closure_label, label)

    def test_writer_shaped_acyclic_stereo_uses_writer_frontier(self) -> None:
        for facts in (tetrahedral_facts(), directional_facts()):
            with self.subTest(facts=facts):
                prepared = _prepare(facts)
                with _forbidden_exhaustive_routes():
                    support = enumerate_prepared_stereo_support(
                        prepared=prepared,
                        runtime_options=_writer_options(),
                    )

                self.assertGreater(support.distinct_count, 0)
                self.assertEqual(len(support.strings), support.distinct_count)

    def test_writer_state_key_excludes_rendered_payloads(self) -> None:
        fields = set(WriterState.__dataclass_fields__)

        self.assertNotIn("rendered", fields)
        self.assertNotIn("prefix", fields)
        self.assertNotIn("suffix", fields)
        key = next(
            iter(
                initial_writer_frontier_cursor(
                    _prepare(cco_facts()),
                    _writer_options(),
                ).support_state.states
            )
        )
        self.assertIsInstance(key, WriterStateKey)
        self.assertEqual(writer_state_key(writer_state_from_key(key)), key)

    def test_writer_state_active_frame_is_non_nullable_in_datamodel(self) -> None:
        source = inspect.getsource(writer_state_module)

        self.assertNotIn("active: WriterAtomFrame | None", source)
        self.assertNotIn('return ("none",)', source)

    def test_writer_frontier_cursor_api_deletes_unweighted_entry_points(self) -> None:
        self.assertFalse(hasattr(writer_frontier_module, "initial_writer_frontier"))
        self.assertFalse(hasattr(writer_frontier_module, "count_writer_witness_completions"))
        self.assertFalse(hasattr(writer_frontier_module, "writer_frontier_successors"))
        self.assertNotIn("initial_writer_frontier", writer_frontier_module.__all__)
        self.assertNotIn("count_writer_witness_completions", writer_frontier_module.__all__)
        self.assertNotIn("writer_frontier_successors", writer_frontier_module.__all__)

    def test_writer_frontier_cursor_uses_structural_key_ordering(self) -> None:
        cursor = initial_writer_frontier_cursor(
            _prepare(cco_facts()),
            _writer_options(),
        )
        keys = tuple(key for key, _ in reversed(cursor.weighted_states))
        reordered = WriterFrontierCursor(
            weighted_states=tuple((key, 1) for key in keys),
        )

        self.assertEqual(
            tuple(key for key, _ in reordered.weighted_states),
            tuple(sorted(keys, key=writer_state_key_sort_tuple)),
        )
        self.assertNotIn(
            "repr(",
            inspect.getsource(WriterFrontierCursor.__post_init__),
        )

    def test_initial_writer_frontier_cursor_rejects_exhaustive_options(self) -> None:
        prepared = _prepare(cco_facts())

        with self.assertRaises(SouthStarError) as caught:
            initial_writer_frontier_cursor(prepared, SouthStarRuntimeOptions())

        self.assertIs(caught.exception.kind, SouthStarErrorKind.UNSUPPORTED_POLICY)

    def test_initial_writer_frontier_cursor_invalid_root_raises_typed_error(self) -> None:
        prepared = _prepare(cco_facts())

        with self.assertRaises(SouthStarError) as caught:
            initial_writer_frontier_cursor(prepared, _writer_options(rooted_at_atom=99))

        self.assertIs(caught.exception.kind, SouthStarErrorKind.INVALID_FACTS)

    def test_missing_writer_bond_domain_fails_closed(self) -> None:
        facts = chain_facts(("C", "C"))
        prepared = prepare_south_star_mol_from_facts(
            facts,
            writer_surface=SouthStarWriterSurface(),
            policy=missing_bond_domain_policy(facts),
        )

        with self.assertRaises(SouthStarError) as caught:
            enumerate_prepared_stereo_support(
                prepared=prepared,
                runtime_options=_writer_options(),
            )

        self.assertIs(caught.exception.kind, SouthStarErrorKind.UNSUPPORTED_POLICY)

    def test_writer_modules_do_not_import_exhaustive_routes(self) -> None:
        forbidden = {
            "skeleton",
            "exhaustive_online_traversal",
            "online_stereo_witness",
            "online_search_vm",
        }

        for module_name in (
            "writer_events.py",
            "writer_graph_obligations.py",
            "writer_state.py",
            "writer_transitions.py",
            "writer_frontier.py",
            "writer_stereo.py",
            "writer_snapshot.py",
            "writer_support.py",
        ):
            with self.subTest(module=module_name):
                tree = ast.parse((SOUTH_STAR1_ROOT / module_name).read_text(encoding="utf-8"))
                imported: list[str] = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imported.extend(
                            alias.name.rsplit(".", 1)[-1]
                            for alias in node.names
                            if alias.name.rsplit(".", 1)[-1] in forbidden
                        )
                    if isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        imported.extend(
                            part for part in forbidden if module.endswith(part)
                        )
                self.assertEqual(imported, [])


def _prepare(facts: MoleculeFacts):
    return prepare_south_star_mol_from_facts(
        facts,
        writer_surface=SouthStarWriterSurface(),
    )


def _prepare_with_policy(
    facts: MoleculeFacts,
    *,
    least_free_ring_labels: bool,
    ring_labels: tuple[RingLabel, ...] = (RingLabel(1), RingLabel(2)),
):
    prepared = _prepare(facts)
    return prepare_south_star_mol_from_facts(
        facts,
        writer_surface=SouthStarWriterSurface(),
        policy=replace(
            prepared.policy,
            ring_labels=ring_labels,
            least_free_ring_labels=least_free_ring_labels,
        ),
    )


def _writer_options(*, rooted_at_atom: int = -1) -> SouthStarRuntimeOptions:
    return SouthStarRuntimeOptions(
        rooted_at_atom=rooted_at_atom,
        serialization_language=SerializationLanguageMode.WRITER_SHAPED,
    )


def _only_choice(prepared, cursor, emitted_text: str):
    matches = tuple(
        choice
        for choice in writer_frontier_choices(prepared, cursor).choices
        if choice.emitted_text == emitted_text
    )
    assert len(matches) == 1
    return matches[0]


def _terminal_keys(prepared, cursor: WriterFrontierCursor) -> tuple[WriterStateKey, ...]:
    terminals: list[WriterStateKey] = []

    def rec(current: WriterFrontierCursor) -> None:
        choices = writer_frontier_choices(prepared, current)
        if choices.terminal is not None:
            terminals.extend(
                key
                for key, _ in choices.terminal.finalized_cursor.weighted_states
            )
        for choice in choices.choices:
            rec(choice.successor)

    rec(cursor)
    return tuple(terminals)


def _raw_initial_state(root: AtomId) -> WriterState:
    return WriterState(
        component_cursor=ComponentCursor(
            component_index=0,
            component_roots=(root,),
        ),
        active=WriterAtomFrame(
            atom=root,
            parent=None,
            incoming_bond=None,
            atom_emitted=False,
        ),
        branch_stack=(),
        visited_atoms=frozenset(),
        written_bonds=frozenset(),
        obligations=ObligationState(),
        ring_state=WriterRingState(),
        stereo_state=empty_writer_stereo_state(),
        policy_state=WriterPolicyState(),
    )


def _raw_emitted_root_state(root: AtomId) -> WriterState:
    return WriterState(
        component_cursor=ComponentCursor(
            component_index=0,
            component_roots=(root,),
        ),
        active=WriterAtomFrame(
            atom=root,
            parent=None,
            incoming_bond=None,
            atom_emitted=True,
        ),
        branch_stack=(),
        visited_atoms=frozenset((root,)),
        written_bonds=frozenset(),
        obligations=ObligationState(),
        ring_state=WriterRingState(),
        stereo_state=empty_writer_stereo_state(),
        policy_state=WriterPolicyState(),
    )


def _cyclopropane_terminal_open_closure_state() -> WriterState:
    label = WriterClosureLabel(value=1, text="1")
    return WriterState(
        component_cursor=ComponentCursor(
            component_index=0,
            component_roots=(AtomId(0),),
        ),
        active=WriterAtomFrame(
            atom=AtomId(2),
            parent=AtomId(1),
            incoming_bond=BondId(1),
            atom_emitted=True,
        ),
        branch_stack=(),
        visited_atoms=frozenset((AtomId(0), AtomId(1), AtomId(2))),
        written_bonds=frozenset((BondId(0), BondId(1))),
        obligations=ObligationState(),
        ring_state=WriterRingState(
            open_endpoints=(
                WriterOpenClosureEndpoint(
                    bond=BondId(2),
                    first_atom=AtomId(0),
                    second_atom=AtomId(2),
                    label=label,
                    first_endpoint_text="1",
                    first_endpoint_bond_text="",
                ),
            ),
            label_state=WriterRingLabelState(allocated=(label,)),
        ),
        stereo_state=empty_writer_stereo_state(),
        policy_state=WriterPolicyState(),
    )


def _cyclopropane_terminal_closed_closure_state() -> WriterState:
    label = WriterClosureLabel(value=1, text="1")
    return replace(
        _cyclopropane_terminal_open_closure_state(),
        ring_state=WriterRingState(
            closed_closures=(
                WriterClosedClosure(
                    bond=BondId(2),
                    first_atom=AtomId(0),
                    second_atom=AtomId(2),
                    label=label,
                    first_endpoint_text="1",
                    second_endpoint_text="1",
                    first_endpoint_bond_text="",
                    second_endpoint_bond_text="",
                ),
            ),
            label_state=WriterRingLabelState(reusable=(label,)),
        ),
    )


def _with_next_component_root(state: WriterState) -> WriterState:
    return replace(
        state,
        component_cursor=ComponentCursor(
            component_index=0,
            component_roots=(AtomId(0), AtomId(3)),
        ),
    )


@contextlib.contextmanager
def _forbidden_exhaustive_routes():
    paths = (
        "grimace._south_star1.skeleton.enumerate_traversal_skeletons",
        "grimace._south_star1.exhaustive_online_traversal.iter_exhaustive_online_traversal_traces",
        "grimace._south_star1.exhaustive_online_traversal.iter_prepared_exhaustive_online_traversal_traces",
        "grimace._south_star1.online_stereo_witness.iter_exhaustive_online_stereo_witnesses",
        "grimace._south_star1.online_search_vm.ExhaustiveOnlineSearchVM",
        "grimace._south_star1.skeleton.TraversalSkeleton",
        "grimace._south_star1.exhaustive_online_traversal.ExhaustiveTraversalTrace",
        "grimace._south_star1.skeleton._component_spanning_trees",
        "grimace._south_star1.exhaustive_online_traversal._iter_spanning_forest_choices_lazy",
    )
    with contextlib.ExitStack() as stack:
        for path in paths:
            stack.enter_context(
                patch(
                    path,
                    side_effect=AssertionError(f"writer-shaped called {path}"),
                )
            )
        yield


def chain_facts(symbols: tuple[str, ...]) -> MoleculeFacts:
    return MoleculeFacts(
        atoms=tuple(atom(index, symbol) for index, symbol in enumerate(symbols)),
        bonds=tuple(
            single_bond(index, index, index + 1)
            for index in range(len(symbols) - 1)
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=tuple(AtomId(index) for index in range(len(symbols))),
                bonds=tuple(BondId(index) for index in range(len(symbols) - 1)),
            ),
        ),
    )


def disconnected_co_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, "C"), atom(1, "O")),
        bonds=(),
        components=(
            ComponentFacts(id=ComponentId(0), atoms=(AtomId(0),), bonds=()),
            ComponentFacts(id=ComponentId(1), atoms=(AtomId(1),), bonds=()),
        ),
    )


def cyclopropane_plus_singleton_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, "C"), atom(1, "C"), atom(2, "C"), atom(3, "O")),
        bonds=(
            single_bond(0, 0, 1),
            single_bond(1, 1, 2),
            single_bond(2, 2, 0),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1), AtomId(2)),
                bonds=(BondId(0), BondId(1), BondId(2)),
            ),
            ComponentFacts(
                id=ComponentId(1),
                atoms=(AtomId(3),),
                bonds=(),
            ),
        ),
    )


def duplicate_single_atom_policy() -> SmilesPolicy:
    return SmilesPolicy(
        ring_labels=(RingLabel(1),),
        annotation_mode=AnnotationMode.HARD,
        atom_text_domains=(
            AtomTextDomain(
                atom=AtomId(0),
                choices=(
                    AtomTextChoice(
                        name="carbon_a",
                        text_by_tetra=((TetraToken.NONE, "C"),),
                    ),
                    AtomTextChoice(
                        name="carbon_b",
                        text_by_tetra=((TetraToken.NONE, "C"),),
                    ),
                ),
            ),
        ),
        bond_text_domains=(),
    )


def missing_bond_domain_policy(facts: MoleculeFacts) -> SmilesPolicy:
    return SmilesPolicy(
        ring_labels=(RingLabel(1),),
        annotation_mode=AnnotationMode.HARD,
        atom_text_domains=tuple(
            AtomTextDomain(
                atom=item.id,
                choices=(
                    AtomTextChoice(
                        name=f"atom_{int(item.id)}",
                        text_by_tetra=((TetraToken.NONE, item.symbol),),
                    ),
                ),
            )
            for item in facts.atoms
        ),
        bond_text_domains=(),
    )


def unsupported_directional_implicit_h_facts() -> MoleculeFacts:
    facts = directional_facts()
    return replace(
        facts,
        ligand_occurrences=(
            LigandOccurrence(
                id=OccurrenceId(0),
                site=facts.ligand_occurrences[0].site,
                kind=LigandKind.IMPLICIT_H,
                atom=AtomId(0),
                bond=None,
            ),
            facts.ligand_occurrences[1],
        ),
    )


def two_atom_facts(
    left: str,
    right: str,
    order: BondOrder,
) -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, left), atom(1, right)),
        bonds=(bond(0, 0, 1, order),),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1)),
                bonds=(BondId(0),),
            ),
        ),
    )


def cycle_plus_isolate_component_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, "C"), atom(1, "C"), atom(2, "C"), atom(3, "C")),
        bonds=(
            single_bond(0, 0, 1),
            single_bond(1, 1, 2),
            single_bond(2, 2, 0),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1), AtomId(2), AtomId(3)),
                bonds=(BondId(0), BondId(1), BondId(2)),
            ),
        ),
    )


if __name__ == "__main__":
    unittest.main()
