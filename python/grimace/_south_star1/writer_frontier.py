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
from .writer_stereo import empty_writer_stereo_state
from .writer_transitions import finalize_writer_terminal_state
from .writer_transitions import _WriterActiveEmittedGraphPolicyBlocker
from .writer_transitions import _WriterActiveEmittedGraphPolicyDecision
from .writer_transitions import _WriterGraphPolicyActionFamily
from .writer_transitions import _WriterNextTokenFrontierSupport
from .writer_transitions import _WriterResidualAttachmentPolicyGroup
from .writer_transitions import _WriterResidualAttachmentPolicyKey
from .writer_transitions import _WriterTopLevelScheduleOutcome
from .writer_transitions import _legal_writer_schedule_outcome
from .writer_transitions import _raise_for_top_level_schedule_outcome_blockers
from .writer_transitions import validate_writer_supported_prepared
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

    @property
    def blocked(self) -> bool:
        return bool(self.schedule_outcome.graph_policy_blockers)

    @property
    def graph_policy_blockers(
        self,
    ) -> tuple[_WriterActiveEmittedGraphPolicyBlocker, ...]:
        return self.schedule_outcome.graph_policy_blockers

    @property
    def graph_policy_decision(
        self,
    ) -> _WriterActiveEmittedGraphPolicyDecision | None:
        return self.schedule_outcome.graph_policy_decision


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
    def blocked(self) -> bool:
        return bool(self.graph_policy_blockers)

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
    return _initial_writer_frontier_cursor(
        prepared,
        runtime_options,
        validate_prepared=validate_writer_supported_prepared,
    )


def initial_writer_transition_frontier_cursor(
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
                        stereo_state=empty_writer_stereo_state(),
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

    support_memo: dict[WriterFrontierState, int] = {}
    completion_memo: dict[WriterStateKey, int] = {}
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
            support_count = _count_writer_frontier_support(
                prepared,
                WriterFrontierState(states=entry.successor_keys),
                support_memo,
            )
            completion_count = _count_weighted_successor_completions(
                prepared,
                entry.weighted_successors,
                completion_memo,
            )

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
        include_counts=include_counts,
        stop_after_first_blocked=True,
    )

    _raise_for_writer_frontier_schedule_outcome_blockers(
        snapshot.schedule_outcome,
    )

    return snapshot


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
    terminal_by_key: Counter[WriterStateKey] = Counter()
    state_outcomes: list[_WriterFrontierStateScheduleOutcome] = []
    frontier_supports: list[_WriterFrontierNextTokenSupport] = []

    for key, parent_weight in cursor.weighted_states:
        state = writer_state_from_key(key)

        finalized = finalize_writer_terminal_state(prepared, state)
        finalized_key = None

        if finalized is not None:
            finalized_key = writer_state_key(finalized)
            terminal_by_key[finalized_key] += parent_weight

        schedule_outcome = _legal_writer_schedule_outcome(prepared, state)

        state_outcome = _WriterFrontierStateScheduleOutcome(
            state_key=key,
            parent_weight=parent_weight,
            finalized_state_key=finalized_key,
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
    return _count_writer_frontier_support(prepared, frontier, {})


def _count_writer_frontier_support(
    prepared: SouthStarPreparedMol,
    frontier: WriterFrontierState,
    memo: dict[WriterFrontierState, int],
) -> int:
    cached = memo.get(frontier)
    if cached is not None:
        return cached
    snapshot = _checked_writer_frontier_choice_snapshot(
        prepared,
        _cursor_from_support_state(frontier),
        include_counts=False,
    )
    total = 1 if snapshot.terminal is not None else 0
    for choice in snapshot.choices:
        successor = WriterFrontierState(states=choice.successor_keys)
        total += _count_writer_frontier_support(prepared, successor, memo)
    memo[frontier] = total
    return total


def count_writer_cursor_completions(
    prepared: SouthStarPreparedMol,
    cursor: WriterFrontierCursor,
) -> int:
    memo: dict[WriterStateKey, int] = {}
    snapshot = _checked_writer_frontier_choice_snapshot(
        prepared,
        cursor,
        include_counts=False,
    )

    return _count_writer_choice_snapshot_completions(
        prepared,
        snapshot,
        memo,
    )


def _count_writer_choice_snapshot_completions(
    prepared: SouthStarPreparedMol,
    snapshot: _WriterFrontierChoiceSnapshot,
    memo: dict[WriterStateKey, int],
) -> int:
    total = 0

    if snapshot.terminal is not None:
        total += snapshot.terminal.completion_count

    for choice in snapshot.choices:
        total += _count_weighted_successor_completions(
            prepared,
            choice.weighted_successors,
            memo,
        )

    return total


def _count_weighted_successor_completions(
    prepared: SouthStarPreparedMol,
    weighted_successors: Counter[WriterStateKey],
    memo: dict[WriterStateKey, int],
) -> int:
    return sum(
        multiplicity * _count_writer_state_completions(prepared, key, memo)
        for key, multiplicity in weighted_successors.items()
    )


def _count_writer_state_completions(
    prepared: SouthStarPreparedMol,
    key: WriterStateKey,
    memo: dict[WriterStateKey, int],
) -> int:
    cached = memo.get(key)
    if cached is not None:
        return cached
    snapshot = _checked_writer_frontier_choice_snapshot(
        prepared,
        WriterFrontierCursor(weighted_states=((key, 1),)),
        include_counts=False,
    )
    total = _count_writer_choice_snapshot_completions(
        prepared,
        snapshot,
        memo,
    )
    memo[key] = total
    return total


def iter_writer_frontier_support(
    prepared: SouthStarPreparedMol,
    cursor: WriterFrontierCursor,
) -> Iterator[str]:
    def rec(current: WriterFrontierCursor, prefix: str) -> Iterator[str]:
        snapshot = _checked_writer_frontier_choice_snapshot(
            prepared,
            current,
            include_counts=False,
        )

        if snapshot.terminal is not None:
            yield prefix

        for text, successor in _successors_from_choice_snapshot(snapshot):
            yield from rec(successor, prefix + text)

    yield from rec(cursor, "")


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
    "initial_writer_transition_frontier_cursor",
    "iter_writer_frontier_support",
    "writer_frontier_choices",
)
