"""Determinized frontier over writer-shaped transition states."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .ids import AtomId
from .writer_execution_evidence import WriterFiniteRelationWorkEvidence
from .writer_execution_evidence import WriterFiniteRelationWorkEnvelopeViolation
from .writer_execution_evidence import WriterGraphObligationWorkEvidence
from .writer_execution_evidence import WriterGraphObligationWorkEnvelopeViolation
from .writer_execution_evidence import WriterResidualPropagationWorkEvidence
from .writer_execution_evidence import WriterResidualWorkEnvelopeViolation
from .writer_execution_evidence import writer_finite_relation_work_envelope_violation
from .writer_execution_evidence import writer_graph_obligation_work_envelope_violation
from .writer_execution_evidence import writer_residual_work_envelope_violation
from .writer_capabilities import _WriterExecutionCapabilityKind
from .writer_capabilities import (
    _unsupported_public_writer_execution_capabilities,
)
from .writer_state import ComponentCursor
from .writer_state import ObligationState
from .writer_state import WriterAtomFrame
from .writer_state import WriterPolicyState
from .writer_state import WriterRingState
from .writer_state import WriterState
from .writer_state import WriterStateKey
from .writer_state import writer_state_from_key
from .writer_state import writer_state_key
from .writer_state import writer_state_key_sort_tuple
from .writer_stereo import initial_writer_stereo_state
from .writer_stereo import WriterStereoPolicyBlocker
from .writer_transitions import _WriterActiveEmittedGraphPolicyBlocker
from .writer_transitions import _WriterActiveEmittedGraphPolicyDecision
from .writer_transitions import _WriterActiveChildSelectionKind
from .writer_transitions import _WriterClosureEndpointSelectionKind
from .writer_transitions import _WriterGraphPolicyActionFamily
from .writer_transitions import _WriterNextTokenFrontierSupport
from .writer_transitions import _WriterResidualAttachmentOwnerScopeKind
from .writer_transitions import _WriterResidualAttachmentPolicyGroup
from .writer_transitions import _WriterResidualAttachmentPolicyKey
from .writer_transitions import _WriterTopLevelScheduleOutcome
from .writer_transitions import _raise_for_top_level_schedule_outcome_blockers
from .writer_transitions import _writer_state_expansion_outcome_from_validated_prepared
from .writer_transitions import validate_writer_transition_prepared

if TYPE_CHECKING:
    from .prepared_runtime import SouthStarPreparedMol
    from .prepared_runtime import SouthStarRuntimeOptions


@dataclass(frozen=True, slots=True)
class WriterFrontierState:
    states: frozenset[WriterStateKey]


@dataclass(frozen=True, slots=True)
class WriterFrontierCursor:
    weighted_states: tuple[tuple[WriterStateKey, int], ...]

    def __post_init__(self) -> None:
        merged: Counter[WriterStateKey] = Counter()
        for key, weight in self.weighted_states:
            if weight < 0:
                raise ValueError("writer frontier cursor weights must be nonnegative")
            if weight:
                merged[key] += weight
        object.__setattr__(
            self,
            "weighted_states",
            tuple(
                sorted(
                    merged.items(),
                    key=lambda item: writer_state_key_sort_tuple(item[0]),
                )
            ),
        )

    @property
    def support_state(self) -> WriterFrontierState:
        return WriterFrontierState(
            states=frozenset(key for key, _ in self.weighted_states)
        )


@dataclass(frozen=True, slots=True)
class WriterFrontierTerminal:
    support_count: int
    completion_count: int
    multiplicity: int
    finalized_cursor: WriterFrontierCursor


@dataclass(frozen=True, slots=True)
class WriterFrontierChoice:
    emitted_text: str
    successor: WriterFrontierCursor
    immediate_multiplicity: int
    support_count: int | None = None
    completion_count: int | None = None


@dataclass(frozen=True, slots=True)
class WriterFrontierChoices:
    terminal: WriterFrontierTerminal | None
    choices: tuple[WriterFrontierChoice, ...]


@dataclass(frozen=True, slots=True)
class _WriterFrontierSummary:
    support_count: int | None = None
    completion_count: int | None = None
    strings: tuple[str, ...] | None = None

    def require_support_count(self) -> int:
        if self.support_count is None:
            raise SouthStarError(
                SouthStarErrorKind.INTERNAL_INVARIANT,
                "writer frontier summary did not compute support count",
            )
        return self.support_count

    def require_completion_count(self) -> int:
        if self.completion_count is None:
            raise SouthStarError(
                SouthStarErrorKind.INTERNAL_INVARIANT,
                "writer frontier summary did not compute completion count",
            )
        return self.completion_count

    def require_strings(self) -> tuple[str, ...]:
        if self.strings is None:
            raise SouthStarError(
                SouthStarErrorKind.INTERNAL_INVARIANT,
                "writer frontier summary did not compute support strings",
            )
        return self.strings


@dataclass(frozen=True, slots=True)
class _GroupedWriterFrontierTransitions:
    terminal_by_key: Counter[WriterStateKey]
    grouped_by_text: dict[str, set[WriterStateKey]]
    weighted_by_text: dict[str, Counter[WriterStateKey]]


@dataclass(frozen=True, slots=True)
class _WriterFrontierStateScheduleOutcome:
    state_key: WriterStateKey
    parent_weight: int
    finalized_state_key: WriterStateKey | None
    schedule_outcome: _WriterTopLevelScheduleOutcome
    terminal_execution_capabilities: frozenset[
        _WriterExecutionCapabilityKind
    ] = frozenset()
    terminal_residual_work_evidence: tuple[
        WriterResidualPropagationWorkEvidence,
        ...
    ] = ()
    graph_obligation_work_evidence: tuple[
        WriterGraphObligationWorkEvidence,
        ...
    ] = ()

    def __post_init__(self) -> None:
        if (
            self.finalized_state_key is None
            and self.terminal_execution_capabilities
        ):
            raise ValueError(
                "terminal execution capabilities require finalized state"
            )
        if (
            self.finalized_state_key is None
            and self.terminal_residual_work_evidence
        ):
            raise ValueError(
                "terminal residual work evidence requires finalized state"
            )

    @property
    def blocked(self) -> bool:
        return bool(
            self.schedule_outcome.graph_policy_blockers
            or self.schedule_outcome.stereo_policy_blockers
        )

    @property
    def graph_policy_blockers(
        self,
    ) -> tuple[_WriterActiveEmittedGraphPolicyBlocker, ...]:
        return self.schedule_outcome.graph_policy_blockers

    @property
    def stereo_policy_blockers(
        self,
    ) -> tuple[WriterStereoPolicyBlocker, ...]:
        return self.schedule_outcome.stereo_policy_blockers

    @property
    def graph_policy_decision(
        self,
    ) -> _WriterActiveEmittedGraphPolicyDecision | None:
        return self.schedule_outcome.graph_policy_decision

    @property
    def considered_active_child_selection_kind(
        self,
    ) -> _WriterActiveChildSelectionKind:
        return self.schedule_outcome.considered_active_child_selection_kind

    @property
    def selected_active_child_selection_kind(
        self,
    ) -> _WriterActiveChildSelectionKind:
        return self.schedule_outcome.selected_active_child_selection_kind


@dataclass(frozen=True, slots=True)
class _WriterFrontierNextTokenSupport:
    state_key: WriterStateKey
    parent_weight: int
    schedule_support: _WriterNextTokenFrontierSupport
    successor_key: WriterStateKey

    @property
    def emitted_text(self) -> str:
        return self.schedule_support.emitted_text

    @property
    def graph_action_surface(self):
        return self.schedule_support.graph_action_surface

    @property
    def policy_family(self):
        return self.schedule_support.policy_family

    @property
    def execution_capabilities(
        self,
    ) -> frozenset[_WriterExecutionCapabilityKind]:
        return self.schedule_support.execution_capabilities

    @property
    def residual_work_evidence(
        self,
    ) -> tuple[WriterResidualPropagationWorkEvidence, ...]:
        return self.schedule_support.residual_work_evidence

    @property
    def finite_relation_work_evidence(
        self,
    ) -> tuple[WriterFiniteRelationWorkEvidence, ...]:
        return self.schedule_support.finite_relation_work_evidence


@dataclass(frozen=True, slots=True)
class _WriterFrontierResidualAttachmentSupportGroup:
    key: _WriterResidualAttachmentPolicyKey
    supports: tuple[_WriterFrontierNextTokenSupport, ...]

    @property
    def policy_families(
        self,
    ) -> tuple[_WriterGraphPolicyActionFamily, ...]:
        return tuple(
            support.policy_family
            for support in self.supports
        )

    def supports_for_policy_family(
        self,
        family: _WriterGraphPolicyActionFamily,
    ) -> tuple[_WriterFrontierNextTokenSupport, ...]:
        return tuple(
            support
            for support in self.supports
            if support.policy_family is family
        )

    @property
    def closure_open_supports(
        self,
    ) -> tuple[_WriterFrontierNextTokenSupport, ...]:
        return self.supports_for_policy_family(
            _WriterGraphPolicyActionFamily.CLOSURE_OPEN
        )

    @property
    def cyclic_tree_entry_supports(
        self,
    ) -> tuple[_WriterFrontierNextTokenSupport, ...]:
        return self.supports_for_policy_family(
            _WriterGraphPolicyActionFamily.CYCLIC_TREE_ENTRY
        )

    @property
    def acyclic_tree_entry_supports(
        self,
    ) -> tuple[_WriterFrontierNextTokenSupport, ...]:
        return self.supports_for_policy_family(
            _WriterGraphPolicyActionFamily.ACYCLIC_TREE_ENTRY
        )

    @property
    def tree_entry_supports(
        self,
    ) -> tuple[_WriterFrontierNextTokenSupport, ...]:
        return (
            *self.supports_for_policy_family(
                _WriterGraphPolicyActionFamily.TREE_ENTRY
            ),
            *self.acyclic_tree_entry_supports,
            *self.cyclic_tree_entry_supports,
        )


@dataclass(frozen=True, slots=True)
class _WriterFrontierResidualAttachmentEvidenceGroup:
    key: _WriterResidualAttachmentPolicyKey
    resolved_policy_groups: tuple[
        _WriterResidualAttachmentPolicyGroup,
        ...,
    ] = ()
    support_dead_closure_open_vs_cyclic_tree_entry_policy_groups: tuple[
        _WriterResidualAttachmentPolicyGroup,
        ...,
    ] = ()
    unsupported_owner_scope_policy_groups: tuple[
        _WriterResidualAttachmentPolicyGroup,
        ...,
    ] = ()
    unresolved_policy_groups: tuple[
        _WriterResidualAttachmentPolicyGroup,
        ...,
    ] = ()
    selected_support_groups: tuple[
        _WriterFrontierResidualAttachmentSupportGroup,
        ...,
    ] = ()

    @property
    def selected_supports(
        self,
    ) -> tuple[_WriterFrontierNextTokenSupport, ...]:
        return tuple(
            support
            for group in self.selected_support_groups
            for support in group.supports
        )

    @property
    def selected_policy_families(
        self,
    ) -> tuple[_WriterGraphPolicyActionFamily, ...]:
        return tuple(
            support.policy_family
            for support in self.selected_supports
        )

    def selected_supports_for_policy_family(
        self,
        family: _WriterGraphPolicyActionFamily,
    ) -> tuple[_WriterFrontierNextTokenSupport, ...]:
        return tuple(
            support
            for support in self.selected_supports
            if support.policy_family is family
        )

    @property
    def selected_closure_open_supports(
        self,
    ) -> tuple[_WriterFrontierNextTokenSupport, ...]:
        return self.selected_supports_for_policy_family(
            _WriterGraphPolicyActionFamily.CLOSURE_OPEN
        )

    @property
    def selected_cyclic_tree_entry_supports(
        self,
    ) -> tuple[_WriterFrontierNextTokenSupport, ...]:
        return self.selected_supports_for_policy_family(
            _WriterGraphPolicyActionFamily.CYCLIC_TREE_ENTRY
        )

    @property
    def selected_acyclic_tree_entry_supports(
        self,
    ) -> tuple[_WriterFrontierNextTokenSupport, ...]:
        return self.selected_supports_for_policy_family(
            _WriterGraphPolicyActionFamily.ACYCLIC_TREE_ENTRY
        )

    @property
    def has_resolved_policy_evidence(self) -> bool:
        return bool(self.resolved_policy_groups)

    @property
    def has_support_dead_closure_open_evidence(self) -> bool:
        return bool(
            self.support_dead_closure_open_vs_cyclic_tree_entry_policy_groups
        )

    @property
    def has_unsupported_owner_scope_evidence(self) -> bool:
        return bool(self.unsupported_owner_scope_policy_groups)

    @property
    def has_unresolved_policy_evidence(self) -> bool:
        return bool(self.unresolved_policy_groups)

    @property
    def has_selected_support_evidence(self) -> bool:
        return bool(self.selected_support_groups)

    @property
    def resolved_policy_owner_scope_kinds(
        self,
    ) -> tuple[_WriterResidualAttachmentOwnerScopeKind, ...]:
        return _owner_scope_kinds_from_residual_policy_groups(
            self.resolved_policy_groups
        )

    @property
    def support_dead_closure_open_vs_cyclic_tree_entry_policy_owner_scope_kinds(
        self,
    ) -> tuple[_WriterResidualAttachmentOwnerScopeKind, ...]:
        return _owner_scope_kinds_from_residual_policy_groups(
            self.support_dead_closure_open_vs_cyclic_tree_entry_policy_groups
        )

    @property
    def unsupported_owner_scope_kinds(
        self,
    ) -> tuple[_WriterResidualAttachmentOwnerScopeKind, ...]:
        return _owner_scope_kinds_from_residual_policy_groups(
            self.unsupported_owner_scope_policy_groups
        )

    @property
    def unresolved_policy_owner_scope_kinds(
        self,
    ) -> tuple[_WriterResidualAttachmentOwnerScopeKind, ...]:
        return _owner_scope_kinds_from_residual_policy_groups(
            self.unresolved_policy_groups
        )

    @property
    def policy_owner_scope_kinds(
        self,
    ) -> tuple[_WriterResidualAttachmentOwnerScopeKind, ...]:
        return (
            *self.resolved_policy_owner_scope_kinds,
            *(
                self
                .support_dead_closure_open_vs_cyclic_tree_entry_policy_owner_scope_kinds
            ),
            *self.unsupported_owner_scope_kinds,
            *self.unresolved_policy_owner_scope_kinds,
        )

    @property
    def has_active_atom_owner_scope_evidence(self) -> bool:
        return any(
            scope is _WriterResidualAttachmentOwnerScopeKind.ACTIVE_ATOM
            for scope in self.policy_owner_scope_kinds
        )

    @property
    def has_branch_return_owner_scope_evidence(self) -> bool:
        return any(
            scope is _WriterResidualAttachmentOwnerScopeKind.BRANCH_RETURN
            for scope in self.policy_owner_scope_kinds
        )

    @property
    def has_pending_parent_owner_scope_evidence(self) -> bool:
        return any(
            scope is _WriterResidualAttachmentOwnerScopeKind.PENDING_PARENT
            for scope in self.policy_owner_scope_kinds
        )

    @property
    def has_open_ring_endpoint_owner_scope_evidence(self) -> bool:
        return any(
            scope is _WriterResidualAttachmentOwnerScopeKind.OPEN_RING_ENDPOINT
            for scope in self.policy_owner_scope_kinds
        )

    @property
    def has_unowned_owner_scope_evidence(self) -> bool:
        return any(
            scope is _WriterResidualAttachmentOwnerScopeKind.UNOWNED
            for scope in self.policy_owner_scope_kinds
        )

    @property
    def has_missing_owner_scope_evidence(self) -> bool:
        return any(
            scope is _WriterResidualAttachmentOwnerScopeKind.MISSING
            for scope in self.policy_owner_scope_kinds
        )

    @property
    def has_mixed_owner_scope_evidence(self) -> bool:
        return any(
            scope is _WriterResidualAttachmentOwnerScopeKind.MIXED
            for scope in self.policy_owner_scope_kinds
        )

    @property
    def has_selected_closure_open_supports(self) -> bool:
        return bool(self.selected_closure_open_supports)

    @property
    def has_selected_cyclic_tree_entry_supports(self) -> bool:
        return bool(self.selected_cyclic_tree_entry_supports)

    @property
    def has_selected_acyclic_tree_entry_supports(self) -> bool:
        return bool(self.selected_acyclic_tree_entry_supports)

    @property
    def has_selected_tree_entry_supports(self) -> bool:
        return bool(
            self.selected_supports_for_policy_family(
                _WriterGraphPolicyActionFamily.TREE_ENTRY
            )
            or self.selected_acyclic_tree_entry_supports
            or self.selected_cyclic_tree_entry_supports
        )

    @property
    def has_dead_closure_open_resolution_evidence(self) -> bool:
        return (
            self.has_resolved_policy_evidence
            and self.has_support_dead_closure_open_evidence
        )

    @property
    def has_dead_closure_open_resolved_cyclic_tree_entry_support(
        self,
    ) -> bool:
        return (
            self.has_dead_closure_open_resolution_evidence
            and self.has_selected_cyclic_tree_entry_supports
        )

    @property
    def has_missing_closure_open_support_evidence_blocker(self) -> bool:
        return self.has_unresolved_policy_evidence

    @property
    def has_unsupported_owner_scope_blocker(self) -> bool:
        return self.has_unsupported_owner_scope_evidence


@dataclass(frozen=True, slots=True)
class _WriterFrontierChoiceResidualAttachmentEvidence:
    choice: _WriterFrontierChoiceSnapshotEntry
    residual_attachment_evidence_groups: tuple[
        _WriterFrontierResidualAttachmentEvidenceGroup,
        ...,
    ]

    @property
    def emitted_text(self) -> str:
        return self.choice.emitted_text

    @property
    def successor(self) -> WriterFrontierCursor:
        return self.choice.successor

    @property
    def supports(
        self,
    ) -> tuple[_WriterFrontierNextTokenSupport, ...]:
        return self.choice.supports

    @property
    def residual_attachment_policy_keys(
        self,
    ) -> tuple[_WriterResidualAttachmentPolicyKey, ...]:
        return tuple(
            group.key
            for group in self.residual_attachment_evidence_groups
        )

    @property
    def selected_supports(
        self,
    ) -> tuple[_WriterFrontierNextTokenSupport, ...]:
        return tuple(
            support
            for group in self.residual_attachment_evidence_groups
            for support in group.selected_supports
        )

    @property
    def selected_policy_families(
        self,
    ) -> tuple[_WriterGraphPolicyActionFamily, ...]:
        return tuple(
            support.policy_family
            for support in self.selected_supports
        )

    @property
    def has_residual_attachment_evidence(self) -> bool:
        return bool(self.residual_attachment_evidence_groups)

    @property
    def policy_owner_scope_kinds(
        self,
    ) -> tuple[_WriterResidualAttachmentOwnerScopeKind, ...]:
        return tuple(
            scope
            for group in self.residual_attachment_evidence_groups
            for scope in group.policy_owner_scope_kinds
        )

    @property
    def unsupported_owner_scope_kinds(
        self,
    ) -> tuple[_WriterResidualAttachmentOwnerScopeKind, ...]:
        return tuple(
            scope
            for group in self.residual_attachment_evidence_groups
            for scope in group.unsupported_owner_scope_kinds
        )

    @property
    def has_unsupported_owner_scope_evidence(self) -> bool:
        return bool(self.unsupported_owner_scope_kinds)

    @property
    def has_retained_dead_closure_open_resolved_cyclic_tree_entry_support(
        self,
    ) -> bool:
        return any(
            group.has_dead_closure_open_resolved_cyclic_tree_entry_support
            for group in self.residual_attachment_evidence_groups
        )

    @property
    def has_retained_residual_cyclic_blocker_evidence(self) -> bool:
        return any(
            group.has_missing_closure_open_support_evidence_blocker
            or group.has_unsupported_owner_scope_blocker
            for group in self.residual_attachment_evidence_groups
        )

    @property
    def dead_closure_open_resolved_cyclic_tree_entry_groups(
        self,
    ) -> tuple[_WriterFrontierResidualAttachmentEvidenceGroup, ...]:
        return tuple(
            group
            for group in self.residual_attachment_evidence_groups
            if (
                group
                .has_dead_closure_open_resolved_cyclic_tree_entry_support
            )
        )

    @property
    def has_dead_closure_open_resolved_cyclic_tree_entry_support(
        self,
    ) -> bool:
        return bool(
            self.dead_closure_open_resolved_cyclic_tree_entry_groups
        )

    @property
    def public_choice(self) -> WriterFrontierChoice:
        return self.choice.to_public_choice()


@dataclass(frozen=True, slots=True)
class _WriterFrontierNextTokenEntry:
    emitted_text: str
    supports: tuple[_WriterFrontierNextTokenSupport, ...]

    @property
    def successor_keys(self) -> frozenset[WriterStateKey]:
        return frozenset(
            support.successor_key
            for support in self.supports
        )

    @property
    def weighted_successors(self) -> Counter[WriterStateKey]:
        weighted: Counter[WriterStateKey] = Counter()

        for support in self.supports:
            weighted[support.successor_key] += support.parent_weight

        return weighted

    @property
    def immediate_multiplicity(self) -> int:
        return sum(self.weighted_successors.values())

    @property
    def policy_families(self):
        return tuple(
            support.policy_family
            for support in self.supports
        )

    @property
    def execution_capabilities(
        self,
    ) -> frozenset[_WriterExecutionCapabilityKind]:
        return frozenset(
            capability
            for support in self.supports
            for capability in support.execution_capabilities
        )

    @property
    def residual_work_evidence(
        self,
    ) -> tuple[WriterResidualPropagationWorkEvidence, ...]:
        return tuple(
            evidence
            for support in self.supports
            for evidence in support.residual_work_evidence
        )

    @property
    def finite_relation_work_evidence(
        self,
    ) -> tuple[WriterFiniteRelationWorkEvidence, ...]:
        return tuple(
            evidence
            for support in self.supports
            for evidence in support.finite_relation_work_evidence
        )

    @property
    def residual_attachment_support_groups(
        self,
    ) -> tuple[_WriterFrontierResidualAttachmentSupportGroup, ...]:
        return _writer_frontier_residual_attachment_support_groups_from_supports(
            self.supports
        )


@dataclass(frozen=True, slots=True)
class _WriterFrontierChoiceSnapshotEntry:
    next_token_entry: _WriterFrontierNextTokenEntry
    successor: WriterFrontierCursor
    support_count: int | None = None
    completion_count: int | None = None

    @property
    def emitted_text(self) -> str:
        return self.next_token_entry.emitted_text

    @property
    def immediate_multiplicity(self) -> int:
        return self.next_token_entry.immediate_multiplicity

    @property
    def supports(self) -> tuple[_WriterFrontierNextTokenSupport, ...]:
        return self.next_token_entry.supports

    @property
    def successor_keys(self) -> frozenset[WriterStateKey]:
        return self.next_token_entry.successor_keys

    @property
    def weighted_successors(self) -> Counter[WriterStateKey]:
        return self.next_token_entry.weighted_successors

    @property
    def policy_families(self):
        return self.next_token_entry.policy_families

    @property
    def residual_attachment_support_groups(
        self,
    ) -> tuple[_WriterFrontierResidualAttachmentSupportGroup, ...]:
        return self.next_token_entry.residual_attachment_support_groups

    @property
    def execution_capabilities(
        self,
    ) -> frozenset[_WriterExecutionCapabilityKind]:
        capabilities: set[_WriterExecutionCapabilityKind] = set()
        for support in self.supports:
            capabilities.update(support.execution_capabilities)

        return frozenset(capabilities)

    @property
    def residual_work_evidence(
        self,
    ) -> tuple[WriterResidualPropagationWorkEvidence, ...]:
        return self.next_token_entry.residual_work_evidence

    @property
    def finite_relation_work_evidence(
        self,
    ) -> tuple[WriterFiniteRelationWorkEvidence, ...]:
        return self.next_token_entry.finite_relation_work_evidence

    def to_public_choice(self) -> WriterFrontierChoice:
        return WriterFrontierChoice(
            emitted_text=self.emitted_text,
            successor=self.successor,
            immediate_multiplicity=self.immediate_multiplicity,
            support_count=self.support_count,
            completion_count=self.completion_count,
        )


@dataclass(frozen=True, slots=True)
class _WriterFrontierScheduleOutcome:
    state_outcomes: tuple[_WriterFrontierStateScheduleOutcome, ...]
    terminal_by_key: Counter[WriterStateKey]
    grouped_by_text: dict[str, set[WriterStateKey]]
    weighted_by_text: dict[str, Counter[WriterStateKey]]
    next_token_frontier: tuple[_WriterFrontierNextTokenEntry, ...] = ()

    @property
    def blocked_state_outcomes(
        self,
    ) -> tuple[_WriterFrontierStateScheduleOutcome, ...]:
        return tuple(
            state_outcome
            for state_outcome in self.state_outcomes
            if state_outcome.blocked
        )

    @property
    def graph_policy_blockers(
        self,
    ) -> tuple[_WriterActiveEmittedGraphPolicyBlocker, ...]:
        return tuple(
            blocker
            for state_outcome in self.blocked_state_outcomes
            for blocker in state_outcome.graph_policy_blockers
        )

    @property
    def stereo_policy_blockers(
        self,
    ) -> tuple[WriterStereoPolicyBlocker, ...]:
        return tuple(
            blocker
            for state_outcome in self.state_outcomes
            for blocker in state_outcome.stereo_policy_blockers
        )

    @property
    def blocked(self) -> bool:
        return bool(
            self.graph_policy_blockers
            or self.stereo_policy_blockers
        )

    @property
    def terminal_execution_capabilities(
        self,
    ) -> frozenset[_WriterExecutionCapabilityKind]:
        return frozenset(
            capability
            for outcome in self.state_outcomes
            if outcome.finalized_state_key is not None
            for capability in outcome.terminal_execution_capabilities
        )

    @property
    def terminal_residual_work_evidence(
        self,
    ) -> tuple[WriterResidualPropagationWorkEvidence, ...]:
        return tuple(
            evidence
            for outcome in self.state_outcomes
            if outcome.finalized_state_key is not None
            for evidence in outcome.terminal_residual_work_evidence
        )

    @property
    def graph_obligation_work_evidence(
        self,
    ) -> tuple[WriterGraphObligationWorkEvidence, ...]:
        return tuple(
            evidence
            for outcome in self.state_outcomes
            for evidence in outcome.graph_obligation_work_evidence
        )

    @property
    def graph_policy_decisions(
        self,
    ) -> tuple[_WriterActiveEmittedGraphPolicyDecision, ...]:
        return tuple(
            state_outcome.graph_policy_decision
            for state_outcome in self.state_outcomes
            if state_outcome.graph_policy_decision is not None
        )

    @property
    def considered_closure_endpoint_selection_kinds(
        self,
    ) -> tuple[_WriterClosureEndpointSelectionKind, ...]:
        return tuple(
            decision.considered_closure_endpoint_selection_kind
            for decision in self.graph_policy_decisions
        )

    @property
    def selected_closure_endpoint_selection_kinds(
        self,
    ) -> tuple[_WriterClosureEndpointSelectionKind, ...]:
        return tuple(
            decision.closure_endpoint_selection_kind
            for decision in self.graph_policy_decisions
            if (
                decision.closure_endpoint_selection_kind
                is not _WriterClosureEndpointSelectionKind.NONE
            )
        )

    @property
    def selected_closure_open_graph_action_surfaces(self):
        return tuple(
            surface
            for decision in self.graph_policy_decisions
            for surface in decision.selected_closure_open_graph_action_surfaces
        )

    @property
    def selected_closure_pair_graph_action_surfaces(self):
        return tuple(
            surface
            for decision in self.graph_policy_decisions
            for surface in decision.selected_closure_pair_graph_action_surfaces
        )

    @property
    def considered_active_child_selection_kinds(
        self,
    ) -> tuple[_WriterActiveChildSelectionKind, ...]:
        return tuple(
            state_outcome.considered_active_child_selection_kind
            for state_outcome in self.state_outcomes
            if (
                state_outcome.considered_active_child_selection_kind
                is not _WriterActiveChildSelectionKind.NONE
            )
        )

    @property
    def selected_active_child_selection_kinds(
        self,
    ) -> tuple[_WriterActiveChildSelectionKind, ...]:
        return tuple(
            state_outcome.selected_active_child_selection_kind
            for state_outcome in self.state_outcomes
            if (
                state_outcome.selected_active_child_selection_kind
                is not _WriterActiveChildSelectionKind.NONE
            )
        )

    @property
    def considered_cyclic_tree_entry_graph_action_surfaces(self):
        return tuple(
            surface
            for state_outcome in self.state_outcomes
            for surface in (
                state_outcome
                .schedule_outcome
                .considered_cyclic_tree_entry_graph_action_surfaces
            )
        )

    @property
    def selected_cyclic_tree_entry_graph_action_surfaces(self):
        return tuple(
            surface
            for state_outcome in self.state_outcomes
            for surface in (
                state_outcome
                .schedule_outcome
                .selected_cyclic_tree_entry_graph_action_surfaces
            )
        )

    @property
    def resolved_residual_attachment_policy_groups(self):
        return tuple(
            group
            for decision in self.graph_policy_decisions
            for group in decision.resolved_residual_attachment_policy_groups
        )

    @property
    def support_dead_closure_open_vs_cyclic_tree_entry_groups(self):
        return tuple(
            group
            for decision in self.graph_policy_decisions
            for group in (
                decision.support_dead_closure_open_vs_cyclic_tree_entry_groups
            )
        )

    @property
    def unsupported_owner_scope_residual_attachment_policy_groups(self):
        return tuple(
            group
            for decision in self.graph_policy_decisions
            for group in (
                decision
                .unsupported_owner_scope_residual_attachment_policy_groups
            )
        )

    @property
    def unresolved_residual_attachment_policy_groups(self):
        return tuple(
            group
            for decision in self.graph_policy_decisions
            for group in decision.unresolved_residual_attachment_policy_groups
        )

    @property
    def grouped_transitions(self) -> _GroupedWriterFrontierTransitions:
        if self.next_token_frontier:
            grouped_by_text = self.grouped_by_text_from_next_token_frontier
            weighted_by_text = self.weighted_by_text_from_next_token_frontier
        else:
            grouped_by_text = self.grouped_by_text
            weighted_by_text = self.weighted_by_text

        return _GroupedWriterFrontierTransitions(
            terminal_by_key=self.terminal_by_key,
            grouped_by_text=grouped_by_text,
            weighted_by_text=weighted_by_text,
        )

    @property
    def next_token_supports(
        self,
    ) -> tuple[_WriterFrontierNextTokenSupport, ...]:
        return tuple(
            support
            for entry in self.next_token_frontier
            for support in entry.supports
        )

    @property
    def residual_attachment_support_groups(
        self,
    ) -> tuple[_WriterFrontierResidualAttachmentSupportGroup, ...]:
        return _writer_frontier_residual_attachment_support_groups_from_supports(
            self.next_token_supports
        )

    @property
    def residual_attachment_evidence_groups(
        self,
    ) -> tuple[_WriterFrontierResidualAttachmentEvidenceGroup, ...]:
        return _writer_frontier_residual_attachment_evidence_groups(
            resolved_policy_groups=(
                self.resolved_residual_attachment_policy_groups
            ),
            support_dead_closure_open_vs_cyclic_tree_entry_policy_groups=(
                self.support_dead_closure_open_vs_cyclic_tree_entry_groups
            ),
            unsupported_owner_scope_policy_groups=(
                self.unsupported_owner_scope_residual_attachment_policy_groups
            ),
            unresolved_policy_groups=(
                self.unresolved_residual_attachment_policy_groups
            ),
            selected_support_groups=self.residual_attachment_support_groups,
        )

    @property
    def grouped_by_text_from_next_token_frontier(
        self,
    ) -> dict[str, set[WriterStateKey]]:
        return {
            entry.emitted_text: set(entry.successor_keys)
            for entry in self.next_token_frontier
        }

    @property
    def weighted_by_text_from_next_token_frontier(
        self,
    ) -> dict[str, Counter[WriterStateKey]]:
        return {
            entry.emitted_text: entry.weighted_successors
            for entry in self.next_token_frontier
        }


@dataclass(frozen=True, slots=True)
class _WriterFrontierChoiceSnapshot:
    schedule_outcome: _WriterFrontierScheduleOutcome
    terminal: WriterFrontierTerminal | None
    choices: tuple[_WriterFrontierChoiceSnapshotEntry, ...]

    @property
    def blocked(self) -> bool:
        return self.schedule_outcome.blocked

    @property
    def graph_policy_blockers(
        self,
    ) -> tuple[_WriterActiveEmittedGraphPolicyBlocker, ...]:
        return self.schedule_outcome.graph_policy_blockers

    @property
    def stereo_policy_blockers(
        self,
    ) -> tuple[WriterStereoPolicyBlocker, ...]:
        return self.schedule_outcome.stereo_policy_blockers

    @property
    def blocked_state_outcomes(
        self,
    ) -> tuple[_WriterFrontierStateScheduleOutcome, ...]:
        return self.schedule_outcome.blocked_state_outcomes

    @property
    def graph_policy_decisions(
        self,
    ) -> tuple[_WriterActiveEmittedGraphPolicyDecision, ...]:
        return self.schedule_outcome.graph_policy_decisions

    @property
    def execution_capabilities(
        self,
    ) -> frozenset[_WriterExecutionCapabilityKind]:
        capabilities: set[_WriterExecutionCapabilityKind] = set()

        for choice in self.choices:
            capabilities.update(choice.execution_capabilities)

        return frozenset(capabilities)

    @property
    def residual_work_evidence(
        self,
    ) -> tuple[WriterResidualPropagationWorkEvidence, ...]:
        return tuple(
            evidence
            for choice in self.choices
            for evidence in choice.residual_work_evidence
        )

    @property
    def finite_relation_work_evidence(
        self,
    ) -> tuple[WriterFiniteRelationWorkEvidence, ...]:
        return tuple(
            evidence
            for choice in self.choices
            for evidence in choice.finite_relation_work_evidence
        )

    @property
    def graph_obligation_work_evidence(
        self,
    ) -> tuple[WriterGraphObligationWorkEvidence, ...]:
        return self.schedule_outcome.graph_obligation_work_evidence

    @property
    def terminal_execution_capabilities(
        self,
    ) -> frozenset[_WriterExecutionCapabilityKind]:
        return self.schedule_outcome.terminal_execution_capabilities

    @property
    def terminal_residual_work_evidence(
        self,
    ) -> tuple[WriterResidualPropagationWorkEvidence, ...]:
        return self.schedule_outcome.terminal_residual_work_evidence

    @property
    def considered_closure_endpoint_selection_kinds(
        self,
    ) -> tuple[_WriterClosureEndpointSelectionKind, ...]:
        return (
            self.schedule_outcome
            .considered_closure_endpoint_selection_kinds
        )

    @property
    def selected_closure_endpoint_selection_kinds(
        self,
    ) -> tuple[_WriterClosureEndpointSelectionKind, ...]:
        return self.schedule_outcome.selected_closure_endpoint_selection_kinds

    @property
    def selected_closure_open_graph_action_surfaces(self):
        return self.schedule_outcome.selected_closure_open_graph_action_surfaces

    @property
    def selected_closure_pair_graph_action_surfaces(self):
        return self.schedule_outcome.selected_closure_pair_graph_action_surfaces

    @property
    def considered_active_child_selection_kinds(
        self,
    ) -> tuple[_WriterActiveChildSelectionKind, ...]:
        return self.schedule_outcome.considered_active_child_selection_kinds

    @property
    def selected_active_child_selection_kinds(
        self,
    ) -> tuple[_WriterActiveChildSelectionKind, ...]:
        return self.schedule_outcome.selected_active_child_selection_kinds

    @property
    def considered_cyclic_tree_entry_graph_action_surfaces(self):
        return (
            self.schedule_outcome
            .considered_cyclic_tree_entry_graph_action_surfaces
        )

    @property
    def selected_cyclic_tree_entry_graph_action_surfaces(self):
        return (
            self.schedule_outcome
            .selected_cyclic_tree_entry_graph_action_surfaces
        )

    @property
    def resolved_residual_attachment_policy_groups(self):
        return self.schedule_outcome.resolved_residual_attachment_policy_groups

    @property
    def support_dead_closure_open_vs_cyclic_tree_entry_groups(self):
        return (
            self.schedule_outcome
            .support_dead_closure_open_vs_cyclic_tree_entry_groups
        )

    @property
    def unsupported_owner_scope_residual_attachment_policy_groups(self):
        return (
            self.schedule_outcome
            .unsupported_owner_scope_residual_attachment_policy_groups
        )

    @property
    def unresolved_residual_attachment_policy_groups(self):
        return (
            self.schedule_outcome
            .unresolved_residual_attachment_policy_groups
        )

    @property
    def residual_attachment_support_groups(
        self,
    ) -> tuple[_WriterFrontierResidualAttachmentSupportGroup, ...]:
        return self.schedule_outcome.residual_attachment_support_groups

    @property
    def residual_attachment_evidence_groups(
        self,
    ) -> tuple[_WriterFrontierResidualAttachmentEvidenceGroup, ...]:
        return self.schedule_outcome.residual_attachment_evidence_groups

    @property
    def choice_residual_attachment_evidence(
        self,
    ) -> tuple[_WriterFrontierChoiceResidualAttachmentEvidence, ...]:
        return tuple(
            _WriterFrontierChoiceResidualAttachmentEvidence(
                choice=choice,
                residual_attachment_evidence_groups=(
                    _writer_frontier_choice_residual_attachment_evidence_groups(
                        choice=choice,
                        schedule_outcome=self.schedule_outcome,
                    )
                ),
            )
            for choice in self.choices
        )

    def choice_residual_attachment_evidence_for_emitted_text(
        self,
        emitted_text: str,
    ) -> _WriterFrontierChoiceResidualAttachmentEvidence | None:
        matches = tuple(
            evidence
            for evidence in self.choice_residual_attachment_evidence
            if evidence.emitted_text == emitted_text
        )

        if len(matches) > 1:
            raise SouthStarError(
                SouthStarErrorKind.INTERNAL_INVARIANT,
                (
                    "writer choice snapshot contains duplicate emitted-text "
                    f"residual evidence entries: {emitted_text!r}"
                ),
            )

        if not matches:
            return None

        return matches[0]

    @property
    def dead_closure_open_resolved_cyclic_tree_entry_choice_evidence(
        self,
    ) -> tuple[_WriterFrontierChoiceResidualAttachmentEvidence, ...]:
        return tuple(
            evidence
            for evidence in self.choice_residual_attachment_evidence
            if (
                evidence
                .has_dead_closure_open_resolved_cyclic_tree_entry_support
            )
        )

    @property
    def retained_dead_closure_open_resolved_cyclic_tree_entry_choice_evidence(
        self,
    ) -> tuple[_WriterFrontierChoiceResidualAttachmentEvidence, ...]:
        return tuple(
            evidence
            for evidence in self.choice_residual_attachment_evidence
            if (
                evidence
                .has_retained_dead_closure_open_resolved_cyclic_tree_entry_support
            )
        )

    @property
    def unsupported_owner_scope_choice_evidence(
        self,
    ) -> tuple[_WriterFrontierChoiceResidualAttachmentEvidence, ...]:
        return tuple(
            evidence
            for evidence in self.choice_residual_attachment_evidence
            if evidence.has_unsupported_owner_scope_evidence
        )

    @property
    def unsupported_owner_scope_kinds(
        self,
    ) -> tuple[_WriterResidualAttachmentOwnerScopeKind, ...]:
        return tuple(
            scope
            for evidence in self.choice_residual_attachment_evidence
            for scope in evidence.unsupported_owner_scope_kinds
        )

    @property
    def public_choices(self) -> WriterFrontierChoices:
        return WriterFrontierChoices(
            terminal=self.terminal,
            choices=tuple(
                choice.to_public_choice()
                for choice in self.choices
            ),
        )


def initial_writer_frontier_cursor(
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions,
) -> WriterFrontierCursor:
    return _initial_writer_transition_frontier_cursor(
        prepared,
        runtime_options,
    )


def _initial_writer_transition_frontier_cursor(
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions,
) -> WriterFrontierCursor:
    return _initial_writer_frontier_cursor(
        prepared,
        runtime_options,
        validate_prepared=validate_writer_transition_prepared,
    )


def _initial_writer_frontier_cursor(
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions,
    *,
    validate_prepared,
) -> WriterFrontierCursor:
    from .prepared_runtime import require_writer_shaped_runtime_options
    from .prepared_runtime import runtime_root_atom_for_prepared

    require_writer_shaped_runtime_options(runtime_options)
    runtime_root_atom_for_prepared(runtime_options, prepared=prepared)
    validate_prepared(prepared)
    root_domains = _root_domains_for_runtime(prepared, runtime_options)
    initial_stereo = initial_writer_stereo_state(prepared)
    weighted_states = []
    for roots in product(*(atoms for _, atoms in root_domains)):
        root_tuple = tuple(roots)
        if not root_tuple:
            continue
        weighted_states.append(
            (
                writer_state_key(
                    WriterState(
                        component_cursor=ComponentCursor(
                            component_index=0,
                            component_roots=root_tuple,
                        ),
                        active=WriterAtomFrame(
                            atom=root_tuple[0],
                            parent=None,
                            incoming_bond=None,
                            atom_emitted=False,
                        ),
                        branch_stack=(),
                        visited_atoms=frozenset(),
                        written_bonds=frozenset(),
                        obligations=ObligationState(),
                        ring_state=WriterRingState(),
                        stereo_state=initial_stereo,
                        policy_state=WriterPolicyState(),
                    )
                ),
                1,
            )
        )
    return WriterFrontierCursor(weighted_states=tuple(weighted_states))


def _cursor_from_support_state(frontier: WriterFrontierState) -> WriterFrontierCursor:
    return WriterFrontierCursor(
        weighted_states=tuple((key, 1) for key in frontier.states)
    )


def _writer_frontier_terminal_from_schedule_outcome(
    outcome: _WriterFrontierScheduleOutcome,
) -> WriterFrontierTerminal | None:
    if not outcome.terminal_by_key:
        return None

    finalized_cursor = WriterFrontierCursor(
        weighted_states=tuple(outcome.terminal_by_key.items())
    )
    terminal_weight = sum(outcome.terminal_by_key.values())

    return WriterFrontierTerminal(
        support_count=1,
        completion_count=terminal_weight,
        multiplicity=terminal_weight,
        finalized_cursor=finalized_cursor,
    )


def _writer_frontier_choice_snapshot_from_schedule_outcome(
    prepared: SouthStarPreparedMol,
    outcome: _WriterFrontierScheduleOutcome,
    *,
    include_counts: bool = True,
) -> _WriterFrontierChoiceSnapshot:
    terminal = _writer_frontier_terminal_from_schedule_outcome(outcome)

    if outcome.blocked:
        return _WriterFrontierChoiceSnapshot(
            schedule_outcome=outcome,
            terminal=terminal,
            choices=(),
        )

    summary_memo: dict[
        tuple[WriterFrontierCursor, bool, bool, bool],
        _WriterFrontierSummary,
    ] = {}
    choices: list[_WriterFrontierChoiceSnapshotEntry] = []

    for entry in sorted(
        outcome.next_token_frontier,
        key=lambda entry: entry.emitted_text,
    ):
        successor = WriterFrontierCursor(
            weighted_states=tuple(entry.weighted_successors.items())
        )

        support_count = None
        completion_count = None

        if include_counts:
            summary = _writer_frontier_summary_from_cursor(
                prepared,
                successor,
                include_support_count=True,
                include_completion_count=True,
                include_strings=False,
                memo=summary_memo,
            )
            support_count = summary.require_support_count()
            completion_count = summary.require_completion_count()

            if support_count == 0 and completion_count == 0:
                continue

        choices.append(
            _WriterFrontierChoiceSnapshotEntry(
                next_token_entry=entry,
                successor=successor,
                support_count=support_count,
                completion_count=completion_count,
            )
        )

    return _WriterFrontierChoiceSnapshot(
        schedule_outcome=outcome,
        terminal=terminal,
        choices=tuple(choices),
    )


def _writer_frontier_choice_snapshot(
    prepared: SouthStarPreparedMol,
    cursor: WriterFrontierCursor,
    *,
    include_counts: bool = True,
    stop_after_first_blocked: bool = False,
) -> _WriterFrontierChoiceSnapshot:
    outcome = _writer_frontier_schedule_outcome(
        prepared,
        cursor,
        stop_after_first_blocked=stop_after_first_blocked,
    )

    return _writer_frontier_choice_snapshot_from_schedule_outcome(
        prepared,
        outcome,
        include_counts=include_counts,
    )


def _checked_writer_frontier_choice_snapshot(
    prepared: SouthStarPreparedMol,
    cursor: WriterFrontierCursor,
    *,
    include_counts: bool = True,
) -> _WriterFrontierChoiceSnapshot:
    snapshot = _writer_frontier_choice_snapshot(
        prepared,
        cursor,
        include_counts=False,
        stop_after_first_blocked=True,
    )

    _raise_for_writer_frontier_choice_snapshot_blockers(snapshot)

    if not include_counts:
        return snapshot

    return _writer_frontier_choice_snapshot_from_schedule_outcome(
        prepared,
        snapshot.schedule_outcome,
        include_counts=True,
    )


def writer_frontier_choices(
    prepared: SouthStarPreparedMol,
    cursor: WriterFrontierCursor,
) -> WriterFrontierChoices:
    snapshot = _checked_writer_frontier_choice_snapshot(
        prepared,
        cursor,
    )

    return snapshot.public_choices


def _writer_frontier_raw_successors_for_streaming(
    prepared: SouthStarPreparedMol,
    cursor: WriterFrontierCursor,
) -> tuple[tuple[str, WriterFrontierCursor], ...]:
    snapshot = _checked_writer_frontier_choice_snapshot(
        prepared,
        cursor,
        include_counts=False,
    )

    return _successors_from_choice_snapshot(snapshot)


def _successors_from_grouped(
    grouped: _GroupedWriterFrontierTransitions,
) -> tuple[tuple[str, WriterFrontierCursor], ...]:
    return tuple(
        (
            text,
            WriterFrontierCursor(
                weighted_states=tuple(grouped.weighted_by_text[text].items())
            ),
        )
        for text in sorted(grouped.grouped_by_text)
    )


def _successors_from_next_token_frontier(
    next_token_frontier: tuple[_WriterFrontierNextTokenEntry, ...],
) -> tuple[tuple[str, WriterFrontierCursor], ...]:
    return tuple(
        (
            entry.emitted_text,
            WriterFrontierCursor(
                weighted_states=tuple(entry.weighted_successors.items()),
            ),
        )
        for entry in sorted(
            next_token_frontier,
            key=lambda entry: entry.emitted_text,
        )
    )


def _successors_from_choice_snapshot(
    snapshot: _WriterFrontierChoiceSnapshot,
) -> tuple[tuple[str, WriterFrontierCursor], ...]:
    return tuple(
        (
            choice.emitted_text,
            choice.successor,
        )
        for choice in snapshot.choices
    )


def _writer_frontier_next_token_entries_from_supports(
    supports: tuple[_WriterFrontierNextTokenSupport, ...],
) -> tuple[_WriterFrontierNextTokenEntry, ...]:
    grouped: dict[str, list[_WriterFrontierNextTokenSupport]] = {}
    order: list[str] = []

    for support in supports:
        emitted_text = support.emitted_text

        if emitted_text not in grouped:
            grouped[emitted_text] = []
            order.append(emitted_text)

        grouped[emitted_text].append(support)

    return tuple(
        _WriterFrontierNextTokenEntry(
            emitted_text=emitted_text,
            supports=tuple(grouped[emitted_text]),
        )
        for emitted_text in order
    )


def _writer_frontier_residual_attachment_support_groups_from_supports(
    supports: tuple[_WriterFrontierNextTokenSupport, ...],
) -> tuple[_WriterFrontierResidualAttachmentSupportGroup, ...]:
    grouped: dict[
        _WriterResidualAttachmentPolicyKey,
        list[_WriterFrontierNextTokenSupport],
    ] = {}
    order: list[_WriterResidualAttachmentPolicyKey] = []

    for support in supports:
        key = (
            support
            .graph_action_surface
            .residual_attachment_policy_key
        )

        if key is None:
            continue

        if key not in grouped:
            grouped[key] = []
            order.append(key)

        grouped[key].append(support)

    return tuple(
        _WriterFrontierResidualAttachmentSupportGroup(
            key=key,
            supports=tuple(grouped[key]),
        )
        for key in order
    )


def _owner_scope_kinds_from_residual_policy_groups(
    groups: tuple[_WriterResidualAttachmentPolicyGroup, ...],
) -> tuple[_WriterResidualAttachmentOwnerScopeKind, ...]:
    return tuple(
        group.closure_open_vs_cyclic_tree_entry_owner_scope_kind
        for group in groups
    )


def _writer_frontier_residual_attachment_evidence_groups(
    *,
    resolved_policy_groups: tuple[
        _WriterResidualAttachmentPolicyGroup,
        ...,
    ],
    support_dead_closure_open_vs_cyclic_tree_entry_policy_groups: tuple[
        _WriterResidualAttachmentPolicyGroup,
        ...,
    ],
    unsupported_owner_scope_policy_groups: tuple[
        _WriterResidualAttachmentPolicyGroup,
        ...,
    ],
    unresolved_policy_groups: tuple[
        _WriterResidualAttachmentPolicyGroup,
        ...,
    ],
    selected_support_groups: tuple[
        _WriterFrontierResidualAttachmentSupportGroup,
        ...,
    ],
) -> tuple[_WriterFrontierResidualAttachmentEvidenceGroup, ...]:
    order: list[_WriterResidualAttachmentPolicyKey] = []
    keys: set[_WriterResidualAttachmentPolicyKey] = set()

    def remember(key: _WriterResidualAttachmentPolicyKey) -> None:
        if key in keys:
            return

        keys.add(key)
        order.append(key)

    for group in resolved_policy_groups:
        remember(group.key)

    for group in support_dead_closure_open_vs_cyclic_tree_entry_policy_groups:
        remember(group.key)

    for group in unsupported_owner_scope_policy_groups:
        remember(group.key)

    for group in unresolved_policy_groups:
        remember(group.key)

    for group in selected_support_groups:
        remember(group.key)

    return tuple(
        _WriterFrontierResidualAttachmentEvidenceGroup(
            key=key,
            resolved_policy_groups=tuple(
                group
                for group in resolved_policy_groups
                if group.key == key
            ),
            support_dead_closure_open_vs_cyclic_tree_entry_policy_groups=tuple(
                group
                for group in (
                    support_dead_closure_open_vs_cyclic_tree_entry_policy_groups
                )
                if group.key == key
            ),
            unsupported_owner_scope_policy_groups=tuple(
                group
                for group in unsupported_owner_scope_policy_groups
                if group.key == key
            ),
            unresolved_policy_groups=tuple(
                group
                for group in unresolved_policy_groups
                if group.key == key
            ),
            selected_support_groups=tuple(
                group
                for group in selected_support_groups
                if group.key == key
            ),
        )
        for key in order
    )


def _writer_frontier_choice_residual_attachment_evidence_groups(
    *,
    choice: _WriterFrontierChoiceSnapshotEntry,
    schedule_outcome: _WriterFrontierScheduleOutcome,
) -> tuple[_WriterFrontierResidualAttachmentEvidenceGroup, ...]:
    selected_support_groups = choice.residual_attachment_support_groups

    return tuple(
        _WriterFrontierResidualAttachmentEvidenceGroup(
            key=support_group.key,
            resolved_policy_groups=tuple(
                group
                for group in (
                    schedule_outcome
                    .resolved_residual_attachment_policy_groups
                )
                if group.key == support_group.key
            ),
            support_dead_closure_open_vs_cyclic_tree_entry_policy_groups=tuple(
                group
                for group in (
                    schedule_outcome
                    .support_dead_closure_open_vs_cyclic_tree_entry_groups
                )
                if group.key == support_group.key
            ),
            unsupported_owner_scope_policy_groups=tuple(
                group
                for group in (
                    schedule_outcome
                    .unsupported_owner_scope_residual_attachment_policy_groups
                )
                if group.key == support_group.key
            ),
            unresolved_policy_groups=tuple(
                group
                for group in (
                    schedule_outcome
                    .unresolved_residual_attachment_policy_groups
                )
                if group.key == support_group.key
            ),
            selected_support_groups=(support_group,),
        )
        for support_group in selected_support_groups
    )


def _validate_writer_frontier_schedule_outcome_grouping(
    outcome: _WriterFrontierScheduleOutcome,
) -> None:
    if not outcome.next_token_frontier:
        return

    if (
        outcome.grouped_by_text
        != outcome.grouped_by_text_from_next_token_frontier
    ):
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "frontier grouped_by_text does not match next-token supports",
        )

    if (
        outcome.weighted_by_text
        != outcome.weighted_by_text_from_next_token_frontier
    ):
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "frontier weighted_by_text does not match next-token supports",
        )


def _writer_frontier_schedule_outcome(
    prepared: SouthStarPreparedMol,
    cursor: WriterFrontierCursor,
    *,
    stop_after_first_blocked: bool = False,
) -> _WriterFrontierScheduleOutcome:
    validate_writer_transition_prepared(prepared)
    terminal_by_key: Counter[WriterStateKey] = Counter()
    state_outcomes: list[_WriterFrontierStateScheduleOutcome] = []
    frontier_supports: list[_WriterFrontierNextTokenSupport] = []

    for key, parent_weight in cursor.weighted_states:
        state = writer_state_from_key(key)
        expansion = _writer_state_expansion_outcome_from_validated_prepared(
            prepared,
            state,
        )
        terminal_outcome = expansion.terminal_outcome
        finalized = terminal_outcome.state
        finalized_key = None

        if finalized is not None:
            finalized_key = writer_state_key(finalized)
            terminal_by_key[finalized_key] += parent_weight

        schedule_outcome = expansion.schedule_outcome

        state_outcome = _WriterFrontierStateScheduleOutcome(
            state_key=key,
            parent_weight=parent_weight,
            finalized_state_key=finalized_key,
            terminal_execution_capabilities=(
                terminal_outcome.execution_capabilities
            ),
            terminal_residual_work_evidence=(
                terminal_outcome.residual_work_evidence
            ),
            graph_obligation_work_evidence=(
                expansion.graph_obligation_work_evidence
            ),
            schedule_outcome=schedule_outcome,
        )
        state_outcomes.append(state_outcome)

        if state_outcome.blocked:
            if stop_after_first_blocked:
                break

            continue

        for entry in schedule_outcome.selected_next_token_frontier:
            for support in entry.supports:
                successor_key = writer_state_key(support.transition.successor)

                frontier_supports.append(
                    _WriterFrontierNextTokenSupport(
                        state_key=key,
                        parent_weight=parent_weight,
                        schedule_support=support,
                        successor_key=successor_key,
                    )
                )

    next_token_frontier = _writer_frontier_next_token_entries_from_supports(
        tuple(frontier_supports)
    )
    grouped = {
        entry.emitted_text: set(entry.successor_keys)
        for entry in next_token_frontier
    }
    weighted = {
        entry.emitted_text: entry.weighted_successors
        for entry in next_token_frontier
    }

    outcome = _WriterFrontierScheduleOutcome(
        state_outcomes=tuple(state_outcomes),
        terminal_by_key=terminal_by_key,
        grouped_by_text=grouped,
        weighted_by_text=weighted,
        next_token_frontier=next_token_frontier,
    )
    _validate_writer_frontier_schedule_outcome_grouping(outcome)

    return outcome


def _raise_for_writer_frontier_schedule_outcome_blockers(
    outcome: _WriterFrontierScheduleOutcome,
) -> None:
    for state_outcome in outcome.blocked_state_outcomes:
        _raise_for_top_level_schedule_outcome_blockers(
            state_outcome.schedule_outcome
        )


def _raise_for_writer_frontier_stereo_policy_blockers(
    outcome: _WriterFrontierScheduleOutcome,
) -> None:
    blockers = sorted(
        outcome.stereo_policy_blockers,
        key=lambda item: (
            item.kind,
            -1 if item.site is None else int(item.site),
            item.operation,
        ),
    )
    if not blockers:
        return

    first = blockers[0]
    site = "none" if first.site is None else str(int(first.site))
    raise SouthStarError(
        SouthStarErrorKind.UNSUPPORTED_POLICY,
        (
            "WRITER_SHAPED unsupported stereo operation at current "
            "frontier: "
            f"kind={first.kind}; "
            f"site={site}; "
            f"operation={first.operation!r}"
        ),
    )


def _raise_for_writer_frontier_execution_capability_blockers(
    outcome: _WriterFrontierScheduleOutcome,
) -> None:
    terminal = _unsupported_public_writer_execution_capabilities(
        outcome.terminal_execution_capabilities
    )
    if terminal:
        kind = min(terminal, key=lambda item: item.value)
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_POLICY,
            (
                "WRITER_SHAPED requires an unsupported South Star "
                f"execution capability: {kind.value}; at EOS"
            ),
        )

    for entry in sorted(
        outcome.next_token_frontier,
        key=lambda item: item.emitted_text,
    ):
        unsupported = _unsupported_public_writer_execution_capabilities(
            entry.execution_capabilities
        )
        if unsupported:
            kind = min(unsupported, key=lambda item: item.value)
            raise SouthStarError(
                SouthStarErrorKind.UNSUPPORTED_POLICY,
                (
                    "WRITER_SHAPED requires an unsupported South Star "
                    f"execution capability: {kind.value}; "
                    f"next={entry.emitted_text!r}"
                ),
            )


def _first_writer_residual_work_envelope_violation(
    evidence: tuple[WriterResidualPropagationWorkEvidence, ...],
) -> WriterResidualWorkEnvelopeViolation | None:
    for item in evidence:
        violation = writer_residual_work_envelope_violation(item)
        if violation is not None:
            return violation
    return None


def _raise_for_writer_residual_work_envelope_violation(
    violation: WriterResidualWorkEnvelopeViolation,
    *,
    location: str,
) -> None:
    evidence = violation.evidence
    raise SouthStarError(
        SouthStarErrorKind.UNSUPPORTED_POLICY,
        (
            "WRITER_SHAPED residual work exceeds the supported "
            "execution envelope: "
            f"operation={evidence.operation!r}; "
            f"metric={violation.metric}; "
            f"actual={violation.actual}; "
            f"limit={violation.limit}; "
            f"{location}"
        ),
    )


def _raise_for_writer_frontier_residual_work_envelope_blockers(
    outcome: _WriterFrontierScheduleOutcome,
) -> None:
    terminal = _first_writer_residual_work_envelope_violation(
        outcome.terminal_residual_work_evidence
    )
    if terminal is not None:
        _raise_for_writer_residual_work_envelope_violation(
            terminal,
            location="at EOS",
        )

    for entry in sorted(
        outcome.next_token_frontier,
        key=lambda item: item.emitted_text,
    ):
        violation = _first_writer_residual_work_envelope_violation(
            entry.residual_work_evidence
        )
        if violation is not None:
            _raise_for_writer_residual_work_envelope_violation(
                violation,
                location=f"next={entry.emitted_text!r}",
            )


def _first_writer_finite_relation_work_envelope_violation(
    evidence: tuple[WriterFiniteRelationWorkEvidence, ...],
) -> WriterFiniteRelationWorkEnvelopeViolation | None:
    for item in evidence:
        violation = writer_finite_relation_work_envelope_violation(item)
        if violation is not None:
            return violation
    return None


def _raise_for_writer_finite_relation_work_envelope_violation(
    violation: WriterFiniteRelationWorkEnvelopeViolation,
    *,
    location: str,
) -> None:
    evidence = violation.evidence
    bond_text = (
        "none" if evidence.bond is None else str(int(evidence.bond))
    )
    raise SouthStarError(
        SouthStarErrorKind.UNSUPPORTED_POLICY,
        (
            "WRITER_SHAPED finite relation work exceeds the supported "
            "execution envelope: "
            f"operation={evidence.operation!r}; "
            f"relation={evidence.relation_kind}; "
            f"bond={bond_text}; "
            f"metric={violation.metric}; "
            f"actual={violation.actual}; "
            f"limit={violation.limit}; "
            f"{location}"
        ),
    )


def _raise_for_writer_frontier_finite_relation_work_envelope_blockers(
    outcome: _WriterFrontierScheduleOutcome,
) -> None:
    for entry in sorted(
        outcome.next_token_frontier,
        key=lambda item: item.emitted_text,
    ):
        violation = _first_writer_finite_relation_work_envelope_violation(
            entry.finite_relation_work_evidence
        )
        if violation is not None:
            _raise_for_writer_finite_relation_work_envelope_violation(
                violation,
                location=f"next={entry.emitted_text!r}",
            )


def _first_writer_graph_obligation_work_envelope_violation(
    evidence: tuple[WriterGraphObligationWorkEvidence, ...],
) -> WriterGraphObligationWorkEnvelopeViolation | None:
    for item in evidence:
        violation = writer_graph_obligation_work_envelope_violation(item)
        if violation is not None:
            return violation
    return None


def _raise_for_writer_graph_obligation_work_envelope_violation(
    violation: WriterGraphObligationWorkEnvelopeViolation,
    *,
    location: str,
) -> None:
    evidence = violation.evidence
    raise SouthStarError(
        SouthStarErrorKind.UNSUPPORTED_POLICY,
        (
            "WRITER_SHAPED graph obligation work exceeds the supported "
            "execution envelope: "
            f"operation={evidence.operation!r}; "
            f"component={evidence.component_index}; "
            f"metric={violation.metric}; "
            f"actual={violation.actual}; "
            f"limit={violation.limit}; "
            f"{location}"
        ),
    )


def _raise_for_writer_frontier_graph_obligation_work_envelope_blockers(
    outcome: _WriterFrontierScheduleOutcome,
) -> None:
    violation = _first_writer_graph_obligation_work_envelope_violation(
        outcome.graph_obligation_work_evidence
    )
    if violation is not None:
        _raise_for_writer_graph_obligation_work_envelope_violation(
            violation,
            location="current frontier",
        )


def _raise_for_writer_frontier_choice_snapshot_blockers(
    snapshot: _WriterFrontierChoiceSnapshot,
) -> None:
    _raise_for_writer_frontier_schedule_outcome_blockers(
        snapshot.schedule_outcome
    )
    _raise_for_writer_frontier_stereo_policy_blockers(
        snapshot.schedule_outcome
    )
    _raise_for_writer_frontier_execution_capability_blockers(
        snapshot.schedule_outcome
    )
    _raise_for_writer_frontier_residual_work_envelope_blockers(
        snapshot.schedule_outcome
    )
    _raise_for_writer_frontier_finite_relation_work_envelope_blockers(
        snapshot.schedule_outcome
    )
    _raise_for_writer_frontier_graph_obligation_work_envelope_blockers(
        snapshot.schedule_outcome
    )


def _checked_writer_frontier_schedule_outcome(
    prepared: SouthStarPreparedMol,
    cursor: WriterFrontierCursor,
) -> _WriterFrontierScheduleOutcome:
    outcome = _writer_frontier_schedule_outcome(
        prepared,
        cursor,
        stop_after_first_blocked=True,
    )

    _raise_for_writer_frontier_schedule_outcome_blockers(outcome)
    _raise_for_writer_frontier_stereo_policy_blockers(outcome)
    _raise_for_writer_frontier_execution_capability_blockers(outcome)
    _raise_for_writer_frontier_residual_work_envelope_blockers(outcome)
    _raise_for_writer_frontier_finite_relation_work_envelope_blockers(
        outcome
    )
    _raise_for_writer_frontier_graph_obligation_work_envelope_blockers(
        outcome
    )

    return outcome


def _group_writer_frontier_transitions(
    prepared: SouthStarPreparedMol,
    cursor: WriterFrontierCursor,
) -> _GroupedWriterFrontierTransitions:
    outcome = _checked_writer_frontier_schedule_outcome(
        prepared,
        cursor,
    )

    return outcome.grouped_transitions


def count_writer_frontier_support(
    prepared: SouthStarPreparedMol,
    frontier: WriterFrontierState,
) -> int:
    return (
        _writer_frontier_summary(
            prepared,
            _cursor_from_support_state(frontier),
            include_support_count=True,
        )
        .require_support_count()
    )


def _writer_frontier_summary(
    prepared: SouthStarPreparedMol,
    cursor: WriterFrontierCursor,
    *,
    include_support_count: bool = False,
    include_completion_count: bool = False,
    include_strings: bool = False,
) -> _WriterFrontierSummary:
    if include_strings:
        include_support_count = True

    return _writer_frontier_summary_from_cursor(
        prepared,
        cursor,
        include_support_count=include_support_count,
        include_completion_count=include_completion_count,
        include_strings=include_strings,
        memo={},
    )


def _writer_frontier_summary_from_cursor(
    prepared: SouthStarPreparedMol,
    cursor: WriterFrontierCursor,
    *,
    include_support_count: bool,
    include_completion_count: bool,
    include_strings: bool,
    memo: dict[
        tuple[WriterFrontierCursor, bool, bool, bool],
        _WriterFrontierSummary,
    ],
) -> _WriterFrontierSummary:
    key = (
        cursor,
        include_support_count,
        include_completion_count,
        include_strings,
    )
    cached = memo.get(key)
    if cached is not None:
        return cached

    snapshot = _checked_writer_frontier_choice_snapshot(
        prepared,
        cursor,
        include_counts=False,
    )
    summary = _writer_frontier_summary_from_snapshot(
        prepared,
        snapshot,
        include_support_count=include_support_count,
        include_completion_count=include_completion_count,
        include_strings=include_strings,
        memo=memo,
    )
    memo[key] = summary
    return summary


def _writer_frontier_summary_from_snapshot(
    prepared: SouthStarPreparedMol,
    snapshot: _WriterFrontierChoiceSnapshot,
    *,
    include_support_count: bool,
    include_completion_count: bool,
    include_strings: bool,
    memo: dict[
        tuple[WriterFrontierCursor, bool, bool, bool],
        _WriterFrontierSummary,
    ],
) -> _WriterFrontierSummary:
    support_count = (
        1 if snapshot.terminal is not None else 0
    ) if include_support_count else None

    completion_count = (
        (
            snapshot.terminal.completion_count
            if snapshot.terminal is not None
            else 0
        )
    ) if include_completion_count else None

    strings: list[str] | None
    if include_strings:
        strings = [""] if snapshot.terminal is not None else []
    else:
        strings = None

    for choice in snapshot.choices:
        child = _writer_frontier_summary_from_cursor(
            prepared,
            choice.successor,
            include_support_count=include_support_count,
            include_completion_count=include_completion_count,
            include_strings=include_strings,
            memo=memo,
        )

        if include_support_count:
            assert support_count is not None
            support_count += child.require_support_count()

        if include_completion_count:
            assert completion_count is not None
            completion_count += child.require_completion_count()

        if include_strings:
            assert strings is not None
            strings.extend(
                choice.emitted_text + suffix
                for suffix in child.require_strings()
            )

    return _WriterFrontierSummary(
        support_count=support_count,
        completion_count=completion_count,
        strings=None if strings is None else tuple(strings),
    )


def count_writer_cursor_completions(
    prepared: SouthStarPreparedMol,
    cursor: WriterFrontierCursor,
) -> int:
    return (
        _writer_frontier_summary(
            prepared,
            cursor,
            include_completion_count=True,
        )
        .require_completion_count()
    )


def iter_writer_frontier_support(
    prepared: SouthStarPreparedMol,
    cursor: WriterFrontierCursor,
) -> Iterator[str]:
    yield from (
        _writer_frontier_summary(
            prepared,
            cursor,
            include_strings=True,
        )
        .require_strings()
    )


def _root_domains_for_runtime(
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions,
) -> tuple[tuple[object, tuple[AtomId, ...]], ...]:
    if runtime_options.rooted_at_atom < 0:
        return prepared.all_root_domains
    atom = AtomId(runtime_options.rooted_at_atom)
    try:
        return prepared.component_root_domains_by_explicit_root[atom]
    except KeyError as exc:
        raise SouthStarError(
            SouthStarErrorKind.INVALID_FACTS,
            f"rooted_at_atom is not present in prepared molecule: {int(atom)}",
        ) from exc


__all__ = (
    "WriterFrontierChoice",
    "WriterFrontierChoices",
    "WriterFrontierCursor",
    "WriterFrontierState",
    "WriterFrontierTerminal",
    "count_writer_cursor_completions",
    "count_writer_frontier_support",
    "initial_writer_frontier_cursor",
    "iter_writer_frontier_support",
    "writer_frontier_choices",
)
