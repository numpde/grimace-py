"""Writer-shaped frontier snapshots."""

from __future__ import annotations

from collections.abc import Iterator
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import replace
from enum import Enum

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .facts import LigandKind
from .facts import BondOrder
from .facts import SiteStatus
from .ids import AtomId
from .ids import BondId
from .ids import OccurrenceId
from .ids import SiteId
from .policy import SerializationLanguageMode
from .policy import DirectionMark
from .policy import TetraToken
from .prepared_runtime import SouthStarPreparedMol
from .prepared_runtime import SouthStarRuntimeOptions
from .prepared_runtime import _prepared_has_cyclic_writer_graph_surface
from .prepared_runtime import require_writer_shaped_runtime_options
from .prepared_runtime import runtime_root_atom_for_prepared
from .residual_constraints import ResidualStore
from .residual_constraints import ResidualStoreValueSnapshot
from .writer_capabilities import _PUBLIC_SUPPORTED_WRITER_EXECUTION_CAPABILITIES
from .writer_capabilities import _WriterExecutionCapabilityKind
from .writer_graph_obligations import WriterBoundaryOwnerKind
from .writer_graph_obligations import WriterEdgeObligationKind
from .writer_graph_obligations import WriterGraphObligationContext
from .writer_graph_obligations import WriterGraphObligationSummary
from .writer_graph_obligations import WriterClosureEndpointChoice
from .writer_graph_obligations import WriterClosureBondTextRelation
from .writer_graph_obligations import WriterResidualAttachmentActionKind
from .writer_graph_obligations import build_writer_graph_obligation_context
from .writer_graph_obligations import validate_writer_snapshot_graph_surface
from .writer_graph_obligations import writer_graph_completion_status
from .writer_graph_obligations import writer_residual_attachment_action_is_blocked
from .writer_frontier import WriterFrontierChoices
from .writer_frontier import WriterFrontierCursor
from .writer_frontier import _WriterFrontierChoiceResidualAttachmentEvidence
from .writer_frontier import _WriterFrontierChoiceSnapshot
from .writer_frontier import _WriterFrontierChoiceSnapshotEntry
from .writer_frontier import _checked_writer_frontier_choice_snapshot
from .writer_frontier import _raise_for_writer_frontier_schedule_outcome_blockers
from .writer_frontier import _residual_cyclic_policy_readiness_report
from .writer_frontier import _initial_writer_transition_frontier_cursor
from .writer_frontier import _writer_frontier_choice_snapshot
from .writer_frontier import iter_writer_frontier_support
from .writer_transitions import _WriterActiveEmittedGraphPolicyBlockerKind
from .writer_stereo import reconstruct_writer_local_order_records
from .writer_stereo import reconstruct_writer_stereo_residual_snapshot
from .writer_stereo import writer_closure_endpoint_relation
from .writer_state import ComponentCursor
from .writer_state import ObligationStateKey
from .writer_state import PendingEntryPhase
from .writer_state import PendingWriterEntry
from .writer_state import WriterAtomFrame
from .writer_state import WriterBranchFrame
from .writer_state import WriterRingStateKey
from .writer_state import WriterStateKey
from .writer_state import WriterStereoStateKey


_PUBLIC_CYCLIC_WRITER_SHAPED_ENABLED = True


@dataclass(frozen=True, slots=True)
class WriterPreparedIdentity:
    runtime: tuple[object, ...]
    atoms: tuple[tuple[object, ...], ...]
    bonds: tuple[tuple[object, ...], ...]
    components: tuple[tuple[object, ...], ...]
    ligand_occurrences: tuple[tuple[object, ...], ...]
    tetra_templates: tuple[tuple[object, ...], ...]
    directional_templates: tuple[tuple[object, ...], ...]
    policy: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class WriterDecoderBoundary:
    consumed_token_count: int = 0


@dataclass(frozen=True, slots=True)
class WriterFrontierFrame:
    cursor: WriterFrontierCursor


WriterSnapshotFrame = WriterFrontierFrame


@dataclass(frozen=True, slots=True)
class WriterSearchSnapshot:
    serialization_language: SerializationLanguageMode
    prepared_identity: WriterPreparedIdentity
    runtime_options: SouthStarRuntimeOptions
    cursor: WriterFrontierCursor
    decoder_boundary: WriterDecoderBoundary
    frame_stack: tuple[WriterSnapshotFrame, ...]


def _capture_writer_frontier_snapshot_unchecked(
    *,
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions,
    cursor: WriterFrontierCursor,
    decoder_boundary: WriterDecoderBoundary = WriterDecoderBoundary(),
) -> WriterSearchSnapshot:
    require_writer_shaped_runtime_options(runtime_options)
    snapshot = WriterSearchSnapshot(
        serialization_language=SerializationLanguageMode.WRITER_SHAPED,
        prepared_identity=_prepared_identity(prepared, runtime_options),
        runtime_options=runtime_options,
        cursor=cursor,
        decoder_boundary=decoder_boundary,
        frame_stack=(WriterFrontierFrame(cursor),),
    )
    validate_writer_search_snapshot(snapshot, prepared=prepared)
    return snapshot


def capture_writer_frontier_snapshot(
    *,
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions,
    cursor: WriterFrontierCursor,
    decoder_boundary: WriterDecoderBoundary = WriterDecoderBoundary(),
) -> WriterSearchSnapshot:
    snapshot = _capture_writer_frontier_snapshot_unchecked(
        prepared=prepared,
        runtime_options=runtime_options,
        cursor=cursor,
        decoder_boundary=decoder_boundary,
    )
    _assert_public_writer_snapshot_cyclic_admission(
        snapshot,
        prepared=prepared,
    )
    return snapshot


def _initial_public_writer_shaped_frontier_cursor_after_admission(
    *,
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions,
) -> WriterFrontierCursor:
    require_writer_shaped_runtime_options(runtime_options)
    runtime_root_atom_for_prepared(runtime_options, prepared=prepared)

    cursor = _initial_writer_transition_frontier_cursor(
        prepared,
        runtime_options,
    )

    if _prepared_has_cyclic_writer_graph_surface(prepared):
        decision = _cyclic_writer_admission_decision_from_cursor(
            prepared=prepared,
            runtime_options=runtime_options,
            cursor=cursor,
        )
        _assert_cyclic_writer_admission_decision(decision)

    return cursor


def capture_initial_writer_frontier_snapshot(
    *,
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions,
    decoder_boundary: WriterDecoderBoundary = WriterDecoderBoundary(),
) -> WriterSearchSnapshot:
    cursor = _initial_public_writer_shaped_frontier_cursor_after_admission(
        prepared=prepared,
        runtime_options=runtime_options,
    )
    return capture_writer_frontier_snapshot(
        prepared=prepared,
        runtime_options=runtime_options,
        cursor=cursor,
        decoder_boundary=decoder_boundary,
    )


def writer_frontier_cursor_from_snapshot(
    snapshot: WriterSearchSnapshot,
    *,
    prepared: SouthStarPreparedMol,
) -> WriterFrontierCursor:
    _assert_public_writer_snapshot_cyclic_admission(
        snapshot,
        prepared=prepared,
    )
    return _validated_writer_frontier_cursor_from_snapshot(
        snapshot,
        prepared=prepared,
    )


def _validated_writer_frontier_cursor_from_snapshot(
    snapshot: WriterSearchSnapshot,
    *,
    prepared: SouthStarPreparedMol,
) -> WriterFrontierCursor:
    validate_writer_search_snapshot(snapshot, prepared=prepared)
    return snapshot.cursor


def _writer_frontier_choice_snapshot_from_snapshot(
    snapshot: WriterSearchSnapshot,
    *,
    prepared: SouthStarPreparedMol,
    include_counts: bool = True,
    stop_after_first_blocked: bool = False,
) -> _WriterFrontierChoiceSnapshot:
    cursor = _validated_writer_frontier_cursor_from_snapshot(
        snapshot,
        prepared=prepared,
    )

    return _writer_frontier_choice_snapshot(
        prepared,
        cursor,
        include_counts=include_counts,
        stop_after_first_blocked=stop_after_first_blocked,
    )


def _checked_writer_frontier_choice_snapshot_from_snapshot(
    snapshot: WriterSearchSnapshot,
    *,
    prepared: SouthStarPreparedMol,
    include_counts: bool = True,
) -> _WriterFrontierChoiceSnapshot:
    cursor = _validated_writer_frontier_cursor_from_snapshot(
        snapshot,
        prepared=prepared,
    )

    return _checked_writer_frontier_choice_snapshot(
        prepared,
        cursor,
        include_counts=include_counts,
    )


class _WriterSnapshotAdvanceOutcomeKind(Enum):
    ADVANCED = "advanced"
    BLOCKED = "blocked"
    INVALID_EMITTED_TEXT = "invalid_emitted_text"


class _WriterSnapshotAdvanceSequenceOutcomeKind(Enum):
    ADVANCED = "advanced"
    BLOCKED = "blocked"
    INVALID_EMITTED_TEXT = "invalid_emitted_text"


class _WriterSnapshotReplayChoiceSnapshotOutcomeKind(Enum):
    CHOICE_SNAPSHOT = "choice_snapshot"
    REPLAY_BLOCKED = "replay_blocked"
    INVALID_EMITTED_TEXT = "invalid_emitted_text"


class _WriterSnapshotPrefixReadOutcomeKind(Enum):
    READABLE = "readable"
    REPLAY_BLOCKED = "replay_blocked"
    INVALID_EMITTED_TEXT = "invalid_emitted_text"
    FINAL_FRONTIER_BLOCKED = "final_frontier_blocked"


@dataclass(frozen=True, slots=True)
class _WriterSnapshotAdvanceOutcome:
    kind: _WriterSnapshotAdvanceOutcomeKind
    source_snapshot: WriterSearchSnapshot
    emitted_text: str
    choice_snapshot: _WriterFrontierChoiceSnapshot
    choice: _WriterFrontierChoiceSnapshotEntry | None = None
    advanced_snapshot: WriterSearchSnapshot | None = None

    def __post_init__(self) -> None:
        has_choice = self.choice is not None
        has_advanced = self.advanced_snapshot is not None

        if self.kind is _WriterSnapshotAdvanceOutcomeKind.ADVANCED:
            valid = (
                not self.choice_snapshot.blocked
                and has_choice
                and has_advanced
                and self.choice.emitted_text == self.emitted_text
            )
        elif self.kind is _WriterSnapshotAdvanceOutcomeKind.BLOCKED:
            valid = (
                self.choice_snapshot.blocked
                and not has_choice
                and not has_advanced
            )
        elif self.kind is _WriterSnapshotAdvanceOutcomeKind.INVALID_EMITTED_TEXT:
            valid = (
                not self.choice_snapshot.blocked
                and not has_choice
                and not has_advanced
            )
        else:
            valid = False

        if not valid:
            raise SouthStarError(
                SouthStarErrorKind.INTERNAL_INVARIANT,
                f"invalid writer snapshot advance outcome: {self.kind!r}",
            )

    @property
    def blocked(self) -> bool:
        return self.kind is _WriterSnapshotAdvanceOutcomeKind.BLOCKED

    @property
    def invalid_emitted_text(self) -> bool:
        return (
            self.kind
            is _WriterSnapshotAdvanceOutcomeKind.INVALID_EMITTED_TEXT
        )

    @property
    def graph_policy_blockers(self):
        return self.choice_snapshot.graph_policy_blockers

    @property
    def choice_residual_attachment_evidence(
        self,
    ) -> _WriterFrontierChoiceResidualAttachmentEvidence | None:
        if self.kind is not _WriterSnapshotAdvanceOutcomeKind.ADVANCED:
            return None

        if self.choice is None:
            raise SouthStarError(
                SouthStarErrorKind.INTERNAL_INVARIANT,
                "advanced writer snapshot outcome did not contain a choice",
            )

        evidence = (
            self.choice_snapshot
            .choice_residual_attachment_evidence_for_emitted_text(
                self.emitted_text
            )
        )

        if evidence is None:
            raise SouthStarError(
                SouthStarErrorKind.INTERNAL_INVARIANT,
                (
                    "advanced writer snapshot outcome did not contain "
                    "residual evidence for emitted text: "
                    f"{self.emitted_text!r}"
                ),
            )

        return evidence


@dataclass(frozen=True, slots=True)
class _WriterSnapshotAdvanceSequenceOutcome:
    kind: _WriterSnapshotAdvanceSequenceOutcomeKind
    source_snapshot: WriterSearchSnapshot
    emitted_texts: tuple[str, ...]
    step_outcomes: tuple[_WriterSnapshotAdvanceOutcome, ...]
    current_snapshot: WriterSearchSnapshot

    def __post_init__(self) -> None:
        if tuple(
            step.emitted_text
            for step in self.step_outcomes
        ) != self.emitted_texts[: len(self.step_outcomes)]:
            valid = False
        else:
            valid = self._payload_is_valid()

        if not valid:
            raise SouthStarError(
                SouthStarErrorKind.INTERNAL_INVARIANT,
                (
                    "invalid writer snapshot advance sequence outcome: "
                    f"{self.kind!r}"
                ),
            )

    def _payload_is_valid(self) -> bool:
        current = self.source_snapshot

        for index, step in enumerate(self.step_outcomes):
            if step.source_snapshot != current:
                return False

            is_last = index == len(self.step_outcomes) - 1

            if not is_last:
                if (
                    step.kind
                    is not _WriterSnapshotAdvanceOutcomeKind.ADVANCED
                    or step.advanced_snapshot is None
                ):
                    return False

                current = step.advanced_snapshot

        if self.kind is _WriterSnapshotAdvanceSequenceOutcomeKind.ADVANCED:
            if len(self.step_outcomes) != len(self.emitted_texts):
                return False

            for step in self.step_outcomes:
                if (
                    step.kind
                    is not _WriterSnapshotAdvanceOutcomeKind.ADVANCED
                    or step.advanced_snapshot is None
                ):
                    return False

            expected = (
                self.source_snapshot
                if not self.step_outcomes
                else self.step_outcomes[-1].advanced_snapshot
            )

            return self.current_snapshot == expected

        if self.kind is _WriterSnapshotAdvanceSequenceOutcomeKind.BLOCKED:
            return (
                bool(self.step_outcomes)
                and self.step_outcomes[-1].kind
                is _WriterSnapshotAdvanceOutcomeKind.BLOCKED
                and self.current_snapshot == self.step_outcomes[-1].source_snapshot
            )

        if (
            self.kind
            is _WriterSnapshotAdvanceSequenceOutcomeKind.INVALID_EMITTED_TEXT
        ):
            return (
                bool(self.step_outcomes)
                and self.step_outcomes[-1].kind
                is _WriterSnapshotAdvanceOutcomeKind.INVALID_EMITTED_TEXT
                and self.current_snapshot == self.step_outcomes[-1].source_snapshot
            )

        return False

    @property
    def advanced_snapshot(self) -> WriterSearchSnapshot | None:
        if self.kind is _WriterSnapshotAdvanceSequenceOutcomeKind.ADVANCED:
            return self.current_snapshot

        return None

    @property
    def failed_outcome(self) -> _WriterSnapshotAdvanceOutcome | None:
        if self.kind is _WriterSnapshotAdvanceSequenceOutcomeKind.ADVANCED:
            return None

        if not self.step_outcomes:
            return None

        return self.step_outcomes[-1]

    @property
    def consumed_emitted_texts(self) -> tuple[str, ...]:
        if self.kind is _WriterSnapshotAdvanceSequenceOutcomeKind.ADVANCED:
            return self.emitted_texts

        return self.emitted_texts[: max(0, len(self.step_outcomes) - 1)]

    @property
    def remaining_emitted_texts(self) -> tuple[str, ...]:
        return self.emitted_texts[len(self.consumed_emitted_texts) :]

    @property
    def blocked(self) -> bool:
        return self.kind is _WriterSnapshotAdvanceSequenceOutcomeKind.BLOCKED

    @property
    def invalid_emitted_text(self) -> bool:
        return (
            self.kind
            is _WriterSnapshotAdvanceSequenceOutcomeKind.INVALID_EMITTED_TEXT
        )

    @property
    def graph_policy_blockers(self):
        failed = self.failed_outcome

        if failed is None:
            return ()

        return failed.graph_policy_blockers

    @property
    def advanced_step_outcomes(
        self,
    ) -> tuple[_WriterSnapshotAdvanceOutcome, ...]:
        return tuple(
            step
            for step in self.step_outcomes
            if step.kind is _WriterSnapshotAdvanceOutcomeKind.ADVANCED
        )

    @property
    def choice_residual_attachment_evidence(
        self,
    ) -> tuple[_WriterFrontierChoiceResidualAttachmentEvidence, ...]:
        evidence: list[_WriterFrontierChoiceResidualAttachmentEvidence] = []

        for step in self.advanced_step_outcomes:
            step_evidence = step.choice_residual_attachment_evidence

            if step_evidence is None:
                raise SouthStarError(
                    SouthStarErrorKind.INTERNAL_INVARIANT,
                    "advanced replay step did not expose residual evidence",
                )

            evidence.append(step_evidence)

        return tuple(evidence)

    @property
    def residual_attachment_evidence_groups(self):
        return tuple(
            group
            for evidence in self.choice_residual_attachment_evidence
            for group in evidence.residual_attachment_evidence_groups
        )

    @property
    def selected_supports(self):
        return tuple(
            support
            for evidence in self.choice_residual_attachment_evidence
            for support in evidence.selected_supports
        )

    @property
    def selected_policy_families(self):
        return tuple(
            family
            for evidence in self.choice_residual_attachment_evidence
            for family in evidence.selected_policy_families
        )


@dataclass(frozen=True, slots=True)
class _WriterSnapshotReplayChoiceSnapshotOutcome:
    kind: _WriterSnapshotReplayChoiceSnapshotOutcomeKind
    source_snapshot: WriterSearchSnapshot
    emitted_texts: tuple[str, ...]
    sequence_outcome: _WriterSnapshotAdvanceSequenceOutcome
    choice_snapshot: _WriterFrontierChoiceSnapshot | None = None

    def __post_init__(self) -> None:
        common_valid = (
            self.sequence_outcome.source_snapshot == self.source_snapshot
            and self.sequence_outcome.emitted_texts == self.emitted_texts
        )

        if (
            self.kind
            is _WriterSnapshotReplayChoiceSnapshotOutcomeKind.CHOICE_SNAPSHOT
        ):
            valid = (
                common_valid
                and self.sequence_outcome.kind
                is _WriterSnapshotAdvanceSequenceOutcomeKind.ADVANCED
                and self.sequence_outcome.advanced_snapshot is not None
                and self.choice_snapshot is not None
            )
        elif (
            self.kind
            is _WriterSnapshotReplayChoiceSnapshotOutcomeKind.REPLAY_BLOCKED
        ):
            valid = (
                common_valid
                and self.sequence_outcome.kind
                is _WriterSnapshotAdvanceSequenceOutcomeKind.BLOCKED
                and self.choice_snapshot is None
            )
        elif (
            self.kind
            is (
                _WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .INVALID_EMITTED_TEXT
            )
        ):
            valid = (
                common_valid
                and self.sequence_outcome.kind
                is (
                    _WriterSnapshotAdvanceSequenceOutcomeKind
                    .INVALID_EMITTED_TEXT
                )
                and self.choice_snapshot is None
            )
        else:
            valid = False

        if not valid:
            raise SouthStarError(
                SouthStarErrorKind.INTERNAL_INVARIANT,
                (
                    "invalid writer snapshot replay choice snapshot outcome: "
                    f"{self.kind!r}"
                ),
            )

    @property
    def replay_succeeded(self) -> bool:
        return (
            self.kind
            is _WriterSnapshotReplayChoiceSnapshotOutcomeKind.CHOICE_SNAPSHOT
        )

    @property
    def replay_failed(self) -> bool:
        return not self.replay_succeeded

    @property
    def advanced_snapshot(self) -> WriterSearchSnapshot | None:
        return self.sequence_outcome.advanced_snapshot

    @property
    def failed_outcome(self) -> _WriterSnapshotAdvanceOutcome | None:
        return self.sequence_outcome.failed_outcome

    @property
    def consumed_emitted_texts(self) -> tuple[str, ...]:
        return self.sequence_outcome.consumed_emitted_texts

    @property
    def remaining_emitted_texts(self) -> tuple[str, ...]:
        return self.sequence_outcome.remaining_emitted_texts

    @property
    def blocked(self) -> bool:
        if (
            self.kind
            is _WriterSnapshotReplayChoiceSnapshotOutcomeKind.REPLAY_BLOCKED
        ):
            return True

        if self.choice_snapshot is None:
            return False

        return self.choice_snapshot.blocked

    @property
    def invalid_emitted_text(self) -> bool:
        return (
            self.kind
            is (
                _WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .INVALID_EMITTED_TEXT
            )
        )

    @property
    def graph_policy_blockers(self):
        if (
            self.kind
            is _WriterSnapshotReplayChoiceSnapshotOutcomeKind.REPLAY_BLOCKED
        ):
            return self.sequence_outcome.graph_policy_blockers

        if self.choice_snapshot is None:
            return ()

        return self.choice_snapshot.graph_policy_blockers

    @property
    def replayed_choice_residual_attachment_evidence(
        self,
    ) -> tuple[_WriterFrontierChoiceResidualAttachmentEvidence, ...]:
        return self.sequence_outcome.choice_residual_attachment_evidence

    @property
    def replayed_residual_attachment_evidence_groups(self):
        return self.sequence_outcome.residual_attachment_evidence_groups

    @property
    def replayed_selected_supports(self):
        return self.sequence_outcome.selected_supports

    @property
    def replayed_selected_policy_families(self):
        return self.sequence_outcome.selected_policy_families


@dataclass(frozen=True, slots=True)
class _WriterSnapshotPrefixReadOutcome:
    kind: _WriterSnapshotPrefixReadOutcomeKind
    replay_outcome: _WriterSnapshotReplayChoiceSnapshotOutcome
    support_count: int | None = None
    completion_count: int | None = None

    def __post_init__(self) -> None:
        choice_snapshot = self.replay_outcome.choice_snapshot

        if self.kind is _WriterSnapshotPrefixReadOutcomeKind.READABLE:
            valid = (
                self.replay_outcome.kind
                is _WriterSnapshotReplayChoiceSnapshotOutcomeKind.CHOICE_SNAPSHOT
                and choice_snapshot is not None
                and not choice_snapshot.blocked
            )
        elif self.kind is _WriterSnapshotPrefixReadOutcomeKind.REPLAY_BLOCKED:
            valid = (
                self.replay_outcome.kind
                is _WriterSnapshotReplayChoiceSnapshotOutcomeKind.REPLAY_BLOCKED
                and choice_snapshot is None
                and self.support_count is None
                and self.completion_count is None
            )
        elif self.kind is _WriterSnapshotPrefixReadOutcomeKind.INVALID_EMITTED_TEXT:
            valid = (
                self.replay_outcome.kind
                is (
                    _WriterSnapshotReplayChoiceSnapshotOutcomeKind
                    .INVALID_EMITTED_TEXT
                )
                and choice_snapshot is None
                and self.support_count is None
                and self.completion_count is None
            )
        elif self.kind is _WriterSnapshotPrefixReadOutcomeKind.FINAL_FRONTIER_BLOCKED:
            valid = (
                self.replay_outcome.kind
                is _WriterSnapshotReplayChoiceSnapshotOutcomeKind.CHOICE_SNAPSHOT
                and choice_snapshot is not None
                and choice_snapshot.blocked
                and self.support_count is None
                and self.completion_count is None
            )
        else:
            valid = False

        if not valid:
            raise SouthStarError(
                SouthStarErrorKind.INTERNAL_INVARIANT,
                f"invalid writer snapshot prefix read outcome: {self.kind!r}",
            )

    @property
    def source_snapshot(self) -> WriterSearchSnapshot:
        return self.replay_outcome.source_snapshot

    @property
    def emitted_texts(self) -> tuple[str, ...]:
        return self.replay_outcome.emitted_texts

    @property
    def choice_snapshot(self) -> _WriterFrontierChoiceSnapshot | None:
        return self.replay_outcome.choice_snapshot

    @property
    def public_choices(self) -> WriterFrontierChoices | None:
        if self.choice_snapshot is None:
            return None

        return self.choice_snapshot.public_choices

    @property
    def replay_succeeded(self) -> bool:
        return self.replay_outcome.replay_succeeded

    @property
    def blocked(self) -> bool:
        return self.kind in (
            _WriterSnapshotPrefixReadOutcomeKind.REPLAY_BLOCKED,
            _WriterSnapshotPrefixReadOutcomeKind.FINAL_FRONTIER_BLOCKED,
        )

    @property
    def invalid_emitted_text(self) -> bool:
        return self.kind is _WriterSnapshotPrefixReadOutcomeKind.INVALID_EMITTED_TEXT

    @property
    def graph_policy_blockers(self):
        return self.replay_outcome.graph_policy_blockers

    @property
    def graph_policy_decisions(self):
        if self.choice_snapshot is None:
            return ()

        return self.choice_snapshot.graph_policy_decisions

    @property
    def residual_cyclic_policy_decisions(self):
        if self.choice_snapshot is None:
            return ()

        return self.choice_snapshot.residual_cyclic_policy_decisions

    @property
    def residual_cyclic_policy_kinds(self):
        if self.choice_snapshot is None:
            return ()

        return self.choice_snapshot.residual_cyclic_policy_kinds

    @property
    def residual_cyclic_policy_coverage_kinds(self):
        if self.choice_snapshot is None:
            return ()

        return self.choice_snapshot.residual_cyclic_policy_coverage_kinds

    @property
    def residual_cyclic_policy_is_covered(self) -> bool:
        if self.choice_snapshot is None:
            return False

        return self.choice_snapshot.residual_cyclic_policy_is_covered

    @property
    def residual_cyclic_policy_is_blocked(self) -> bool:
        if self.choice_snapshot is None:
            return self.blocked

        return self.choice_snapshot.residual_cyclic_policy_is_blocked

    @property
    def residual_cyclic_policy_readiness_report(self):
        if self.choice_snapshot is None:
            return _residual_cyclic_policy_readiness_report(
                coverage_kinds=self.residual_cyclic_policy_coverage_kinds,
                graph_policy_blockers=self.graph_policy_blockers,
                residual_cyclic_policy_decisions=(
                    self.residual_cyclic_policy_decisions
                ),
                has_frontier=False,
            )

        return self.choice_snapshot.residual_cyclic_policy_readiness_report

    @property
    def residual_cyclic_choice_groups(self):
        if self.choice_snapshot is None:
            return ()

        return self.choice_snapshot.residual_cyclic_choice_groups

    @property
    def residual_cyclic_unsupported_owner_scope_groups(self):
        if self.choice_snapshot is None:
            return ()

        return self.choice_snapshot.residual_cyclic_unsupported_owner_scope_groups

    @property
    def residual_cyclic_missing_evidence_groups(self):
        if self.choice_snapshot is None:
            return ()

        return self.choice_snapshot.residual_cyclic_missing_evidence_groups

    @property
    def residual_cyclic_support_dead_groups(self):
        if self.choice_snapshot is None:
            return ()

        return self.choice_snapshot.residual_cyclic_support_dead_groups

    @property
    def considered_closure_endpoint_selection_kinds(self):
        if self.choice_snapshot is None:
            return ()

        return self.choice_snapshot.considered_closure_endpoint_selection_kinds

    @property
    def selected_closure_endpoint_selection_kinds(self):
        if self.choice_snapshot is None:
            return ()

        return self.choice_snapshot.selected_closure_endpoint_selection_kinds

    @property
    def selected_closure_open_graph_action_surfaces(self):
        if self.choice_snapshot is None:
            return ()

        return self.choice_snapshot.selected_closure_open_graph_action_surfaces

    @property
    def selected_closure_pair_graph_action_surfaces(self):
        if self.choice_snapshot is None:
            return ()

        return self.choice_snapshot.selected_closure_pair_graph_action_surfaces

    @property
    def considered_active_child_selection_kinds(self):
        if self.choice_snapshot is None:
            return ()

        return self.choice_snapshot.considered_active_child_selection_kinds

    @property
    def selected_active_child_selection_kinds(self):
        if self.choice_snapshot is None:
            return ()

        return self.choice_snapshot.selected_active_child_selection_kinds

    @property
    def considered_cyclic_tree_entry_graph_action_surfaces(self):
        if self.choice_snapshot is None:
            return ()

        return (
            self.choice_snapshot
            .considered_cyclic_tree_entry_graph_action_surfaces
        )

    @property
    def selected_cyclic_tree_entry_graph_action_surfaces(self):
        if self.choice_snapshot is None:
            return ()

        return (
            self.choice_snapshot
            .selected_cyclic_tree_entry_graph_action_surfaces
        )

    @property
    def resolved_residual_attachment_policy_groups(self):
        if self.choice_snapshot is None:
            return ()

        return self.choice_snapshot.resolved_residual_attachment_policy_groups

    @property
    def support_dead_closure_open_vs_cyclic_tree_entry_groups(self):
        if self.choice_snapshot is None:
            return ()

        return (
            self.choice_snapshot
            .support_dead_closure_open_vs_cyclic_tree_entry_groups
        )

    @property
    def unsupported_owner_scope_residual_attachment_policy_groups(self):
        if self.choice_snapshot is None:
            return ()

        return (
            self.choice_snapshot
            .unsupported_owner_scope_residual_attachment_policy_groups
        )

    @property
    def unresolved_residual_attachment_policy_groups(self):
        if self.choice_snapshot is None:
            return ()

        return self.choice_snapshot.unresolved_residual_attachment_policy_groups

    @property
    def residual_attachment_support_groups(self):
        if self.choice_snapshot is None:
            return ()

        return self.choice_snapshot.residual_attachment_support_groups

    @property
    def residual_attachment_evidence_groups(self):
        if self.choice_snapshot is None:
            return ()

        return self.choice_snapshot.residual_attachment_evidence_groups

    @property
    def choice_residual_attachment_evidence(self):
        if self.choice_snapshot is None:
            return ()

        return self.choice_snapshot.choice_residual_attachment_evidence

    def choice_residual_attachment_evidence_for_emitted_text(
        self,
        emitted_text: str,
    ):
        if self.choice_snapshot is None:
            return None

        return (
            self.choice_snapshot
            .choice_residual_attachment_evidence_for_emitted_text(
                emitted_text
            )
        )

    @property
    def replayed_choice_residual_attachment_evidence(self):
        return self.replay_outcome.replayed_choice_residual_attachment_evidence

    @property
    def replayed_residual_attachment_evidence_groups(self):
        return self.replay_outcome.replayed_residual_attachment_evidence_groups

    @property
    def replayed_selected_supports(self):
        return self.replay_outcome.replayed_selected_supports

    @property
    def replayed_selected_policy_families(self):
        return self.replay_outcome.replayed_selected_policy_families

    @property
    def final_choice_residual_cyclic_policy_kinds(self):
        return tuple(
            kind
            for evidence in self.choice_residual_attachment_evidence
            for kind in evidence.residual_cyclic_policy_kinds
        )

    @property
    def replayed_residual_cyclic_policy_kinds(self):
        return tuple(
            kind
            for evidence in self.replayed_choice_residual_attachment_evidence
            for kind in evidence.residual_cyclic_policy_kinds
        )

    @property
    def final_choice_dead_closure_open_resolved_cyclic_tree_entry_evidence(
        self,
    ):
        if self.choice_snapshot is None:
            return ()

        return (
            self.choice_snapshot
            .dead_closure_open_resolved_cyclic_tree_entry_choice_evidence
        )

    @property
    def replayed_dead_closure_open_resolved_cyclic_tree_entry_evidence(
        self,
    ):
        return tuple(
            evidence
            for evidence in self.replayed_choice_residual_attachment_evidence
            if (
                evidence
                .has_dead_closure_open_resolved_cyclic_tree_entry_support
            )
        )

    @property
    def final_choice_unsupported_owner_scope_evidence(self):
        if self.choice_snapshot is None:
            return ()

        return self.choice_snapshot.unsupported_owner_scope_choice_evidence

    @property
    def final_choice_unsupported_owner_scope_kinds(self):
        if self.choice_snapshot is None:
            return ()

        return self.choice_snapshot.unsupported_owner_scope_kinds

    @property
    def blocker_owner_scope_kinds(self):
        return tuple(
            blocker.residual_attachment_owner_scope_kind
            for blocker in self.graph_policy_blockers
            if blocker.residual_attachment_owner_scope_kind is not None
        )


class _WriterResidualCyclicReadinessAuditKind(Enum):
    READY = "ready"
    BLOCKED = "blocked"
    TRUNCATED = "truncated"


@dataclass(frozen=True, slots=True)
class _WriterResidualCyclicReadinessBlockedPrefix:
    emitted_texts: tuple[str, ...]
    choice_snapshot: _WriterFrontierChoiceSnapshot

    @property
    def readiness_report(self):
        return self.choice_snapshot.residual_cyclic_policy_readiness_report

    @property
    def graph_policy_blockers(self):
        return self.choice_snapshot.graph_policy_blockers

    @property
    def residual_cyclic_policy_coverage_kinds(self):
        return self.choice_snapshot.residual_cyclic_policy_coverage_kinds


@dataclass(frozen=True, slots=True)
class _WriterExecutionCapabilityUse:
    kind: _WriterExecutionCapabilityKind
    emitted_texts: tuple[str, ...]
    source_cursor: WriterFrontierCursor
    successor_cursor: WriterFrontierCursor
    next_emitted_text: str | None = None

    @property
    def terminal(self) -> bool:
        return self.next_emitted_text is None


@dataclass(frozen=True, slots=True)
class _WriterResidualCyclicReadinessAudit:
    kind: _WriterResidualCyclicReadinessAuditKind
    visited_prefixes: tuple[tuple[str, ...], ...]
    execution_capability_uses: tuple[
        _WriterExecutionCapabilityUse,
        ...,
    ] = ()
    blocked_prefixes: tuple[
        _WriterResidualCyclicReadinessBlockedPrefix,
        ...,
    ] = ()
    truncated_at_prefix: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.kind is _WriterResidualCyclicReadinessAuditKind.READY:
            valid = (
                not self.blocked_prefixes
                and self.truncated_at_prefix is None
            )
        elif self.kind is _WriterResidualCyclicReadinessAuditKind.BLOCKED:
            valid = (
                bool(self.blocked_prefixes)
                and self.truncated_at_prefix is None
            )
        elif self.kind is _WriterResidualCyclicReadinessAuditKind.TRUNCATED:
            valid = self.truncated_at_prefix is not None
        else:
            valid = False

        if not valid:
            raise SouthStarError(
                SouthStarErrorKind.INTERNAL_INVARIANT,
                f"invalid residual cyclic readiness audit: {self.kind!r}",
            )

    @property
    def ready(self) -> bool:
        return self.kind is _WriterResidualCyclicReadinessAuditKind.READY

    @property
    def blocked(self) -> bool:
        return self.kind is _WriterResidualCyclicReadinessAuditKind.BLOCKED

    @property
    def truncated(self) -> bool:
        return self.kind is _WriterResidualCyclicReadinessAuditKind.TRUNCATED

    @property
    def blocked_emitted_texts(self) -> tuple[tuple[str, ...], ...]:
        return tuple(
            blocked.emitted_texts
            for blocked in self.blocked_prefixes
        )

    @property
    def required_execution_capabilities(self) -> frozenset[
        _WriterExecutionCapabilityKind
    ]:
        return frozenset(
            use.kind
            for use in self.execution_capability_uses
        )


class _WriterResidualCyclicReadinessGateKind(Enum):
    READY = "ready"
    BLOCKED = "blocked"
    TRUNCATED = "truncated"


@dataclass(frozen=True, slots=True)
class _WriterResidualCyclicReadinessGate:
    kind: _WriterResidualCyclicReadinessGateKind
    snapshot: WriterSearchSnapshot
    audit: _WriterResidualCyclicReadinessAudit

    def __post_init__(self) -> None:
        if self.kind is _WriterResidualCyclicReadinessGateKind.READY:
            valid = self.audit.ready
        elif self.kind is _WriterResidualCyclicReadinessGateKind.BLOCKED:
            valid = self.audit.blocked
        elif self.kind is _WriterResidualCyclicReadinessGateKind.TRUNCATED:
            valid = self.audit.truncated
        else:
            valid = False

        if not valid:
            raise SouthStarError(
                SouthStarErrorKind.INTERNAL_INVARIANT,
                f"invalid residual cyclic readiness gate: {self.kind!r}",
            )

    @property
    def ready(self) -> bool:
        return self.kind is _WriterResidualCyclicReadinessGateKind.READY

    @property
    def blocked(self) -> bool:
        return self.kind is _WriterResidualCyclicReadinessGateKind.BLOCKED

    @property
    def truncated(self) -> bool:
        return self.kind is _WriterResidualCyclicReadinessGateKind.TRUNCATED

    @property
    def blocked_prefixes(
        self,
    ) -> tuple[_WriterResidualCyclicReadinessBlockedPrefix, ...]:
        return self.audit.blocked_prefixes

    @property
    def first_blocked_prefix(
        self,
    ) -> _WriterResidualCyclicReadinessBlockedPrefix | None:
        if not self.blocked_prefixes:
            return None

        return self.blocked_prefixes[0]

    @property
    def blocked_emitted_texts(self) -> tuple[tuple[str, ...], ...]:
        return self.audit.blocked_emitted_texts

    @property
    def truncated_at_prefix(self) -> tuple[str, ...] | None:
        return self.audit.truncated_at_prefix


@dataclass(frozen=True, slots=True)
class _WriterPublicExecutionCapabilityCertificate:
    required_capabilities: frozenset[_WriterExecutionCapabilityKind]
    supported_capabilities: frozenset[_WriterExecutionCapabilityKind]
    unsupported_capabilities: frozenset[_WriterExecutionCapabilityKind]
    first_unsupported_uses: tuple[_WriterExecutionCapabilityUse, ...] = ()

    def __post_init__(self) -> None:
        expected = self.required_capabilities - self.supported_capabilities
        if self.unsupported_capabilities != expected:
            raise ValueError(
                "writer execution capability certificate has wrong unsupported set"
            )
        use_kinds = frozenset(use.kind for use in self.first_unsupported_uses)
        if use_kinds != self.unsupported_capabilities:
            raise ValueError(
                "writer execution capability certificate missing unsupported uses"
            )

    @property
    def ready(self) -> bool:
        return not self.unsupported_capabilities


class _WriterCyclicAdmissionDecisionKind(Enum):
    READY_PUBLIC = "ready_public"
    READY_BUT_PUBLIC_CLOSED = "ready_but_public_closed"
    BLOCKED_PUBLIC_CYCLIC_PROFILE = "blocked_public_cyclic_profile"
    BLOCKED_PUBLIC_EXECUTION_CAPABILITY = "blocked_public_execution_capability"
    BLOCKED_RESIDUAL_CYCLIC_POLICY = "blocked_residual_cyclic_policy"
    TRUNCATED_READINESS_AUDIT = "truncated_readiness_audit"


class _WriterPublicCyclicOpeningProfileKind(Enum):
    SUPPORTED_SIMPLE_MONOCYCLE_COMPONENT = (
        "supported_simple_monocycle_component"
    )
    SUPPORTED_SIMPLE_MONOCYCLE_WITH_ACYCLIC_ATTACHMENTS = (
        "supported_simple_monocycle_with_acyclic_attachments"
    )
    SUPPORTED_TWO_BRIDGE_SEPARATED_SIMPLE_CYCLES = (
        "supported_two_bridge_separated_simple_cycles"
    )
    SUPPORTED_TWO_BRIDGE_SEPARATED_SIMPLE_CYCLES_WITH_ACYCLIC_ATTACHMENTS = (
        "supported_two_bridge_separated_simple_cycles_with_acyclic_attachments"
    )
    SUPPORTED_FUSED_RANK_TWO_DIAMOND = (
        "supported_fused_rank_two_diamond"
    )
    BLOCKED_NOT_SINGLE_COMPONENT = "blocked_not_single_component"
    BLOCKED_NOT_CONNECTED_COMPONENT = "blocked_not_connected_component"
    BLOCKED_NOT_CYCLIC_COMPONENT = "blocked_not_cyclic_component"
    BLOCKED_UNSUPPORTED_CYCLIC_RANK = "blocked_unsupported_cyclic_rank"
    BLOCKED_UNSUPPORTED_BRANCHING = "blocked_unsupported_branching"
    BLOCKED_UNSUPPORTED_CLOSURE_BOND_SURFACE = (
        "blocked_unsupported_closure_bond_surface"
    )
    BLOCKED_UNSUPPORTED_RING_LABEL_POLICY = (
        "blocked_unsupported_ring_label_policy"
    )
    BLOCKED_UNSUPPORTED_CYCLIC_STEREO_SURFACE = (
        "blocked_unsupported_cyclic_stereo_surface"
    )


class _WriterPublicCyclicRequiredCapability(Enum):
    SIMPLE_CYCLE_CORE_CLOSURE = "simple_cycle_core_closure"
    ACYCLIC_PENDANT_TREE_TRAVERSAL = "acyclic_pendant_tree_traversal"
    TREE_BOND_TEXT_EMISSION = "tree_bond_text_emission"
    RING_CORE_NON_SINGLE_CLOSURE_BOND = "ring_core_non_single_closure_bond"
    RING_CORE_VISIBLE_SINGLE_CLOSURE_BOND_TEXT = (
        "ring_core_visible_single_closure_bond_text"
    )
    RING_CORE_AROMATIC_BOND_TEXT = "ring_core_aromatic_bond_text"
    RING_CORE_TETRAHEDRAL_STEREO = "ring_core_tetrahedral_stereo"
    CYCLIC_DIRECTIONAL_STEREO = "cyclic_directional_stereo"
    CYCLIC_RING_PAIR_STEREO = "cyclic_ring_pair_stereo"
    SHARED_DIRECTIONAL_RING_CARRIER_STEREO = (
        "shared_directional_ring_carrier_stereo"
    )
    MULTI_CYCLE_TOPOLOGY = "multi_cycle_topology"
    FUSED_OR_BRIDGED_TOPOLOGY = "fused_or_bridged_topology"
    NON_FOREST_PENDANT_MATERIAL = "non_forest_pendant_material"
    MULTI_BOUNDARY_PENDANT_COMPONENT = "multi_boundary_pendant_component"


_PUBLIC_CYCLIC_SUPPORTED_CAPABILITIES = frozenset(
    {
        _WriterPublicCyclicRequiredCapability.SIMPLE_CYCLE_CORE_CLOSURE,
        _WriterPublicCyclicRequiredCapability.ACYCLIC_PENDANT_TREE_TRAVERSAL,
        _WriterPublicCyclicRequiredCapability.TREE_BOND_TEXT_EMISSION,
        _WriterPublicCyclicRequiredCapability.RING_CORE_NON_SINGLE_CLOSURE_BOND,
        _WriterPublicCyclicRequiredCapability.MULTI_CYCLE_TOPOLOGY,
        (
            _WriterPublicCyclicRequiredCapability
            .RING_CORE_VISIBLE_SINGLE_CLOSURE_BOND_TEXT
        ),
        _WriterPublicCyclicRequiredCapability.RING_CORE_AROMATIC_BOND_TEXT,
        _WriterPublicCyclicRequiredCapability.RING_CORE_TETRAHEDRAL_STEREO,
        _WriterPublicCyclicRequiredCapability.CYCLIC_DIRECTIONAL_STEREO,
        _WriterPublicCyclicRequiredCapability.CYCLIC_RING_PAIR_STEREO,
        (
            _WriterPublicCyclicRequiredCapability
            .SHARED_DIRECTIONAL_RING_CARRIER_STEREO
        ),
        _WriterPublicCyclicRequiredCapability.FUSED_OR_BRIDGED_TOPOLOGY,
    }
)


def _closure_policy_blocker_capability_for_order(
    order: BondOrder,
) -> _WriterPublicCyclicRequiredCapability:
    if order in {BondOrder.DOUBLE, BondOrder.TRIPLE}:
        return (
            _WriterPublicCyclicRequiredCapability
            .RING_CORE_NON_SINGLE_CLOSURE_BOND
        )
    if order is BondOrder.AROMATIC:
        return _WriterPublicCyclicRequiredCapability.RING_CORE_AROMATIC_BOND_TEXT
    return (
        _WriterPublicCyclicRequiredCapability
        .RING_CORE_VISIBLE_SINGLE_CLOSURE_BOND_TEXT
    )


@dataclass(frozen=True, slots=True)
class _WriterPublicCyclicOpeningProfileReport:
    kind: _WriterPublicCyclicOpeningProfileKind
    component_count: int
    cyclic_component_count: int
    cyclic_ranks: tuple[int, ...]
    ring_core_atom_count: int
    ring_core_bond_count: int
    ring_core_max_degree: int
    pendant_atom_count: int
    pendant_bond_count: int
    component_atom_count: int
    component_bond_count: int
    max_component_degree: int
    branch_atom_count: int
    unsupported_bond_count: int
    unsupported_stereo_surface_count: int
    ring_core_unsupported_bond_count: int = 0
    pendant_unsupported_bond_count: int = 0
    required_capabilities: frozenset[
        _WriterPublicCyclicRequiredCapability
    ] = frozenset()
    unsupported_capabilities: frozenset[
        _WriterPublicCyclicRequiredCapability
    ] = frozenset()
    pendant_component_count: int = 0
    pendant_component_atom_counts: tuple[int, ...] = ()
    pendant_component_boundary_counts: tuple[int, ...] = ()

    @property
    def supported(self) -> bool:
        return (
            self.kind in {
                _WriterPublicCyclicOpeningProfileKind
                .SUPPORTED_SIMPLE_MONOCYCLE_COMPONENT,
                _WriterPublicCyclicOpeningProfileKind
                .SUPPORTED_SIMPLE_MONOCYCLE_WITH_ACYCLIC_ATTACHMENTS,
                _WriterPublicCyclicOpeningProfileKind
                .SUPPORTED_TWO_BRIDGE_SEPARATED_SIMPLE_CYCLES,
                _WriterPublicCyclicOpeningProfileKind
                .SUPPORTED_TWO_BRIDGE_SEPARATED_SIMPLE_CYCLES_WITH_ACYCLIC_ATTACHMENTS,
                _WriterPublicCyclicOpeningProfileKind
                .SUPPORTED_FUSED_RANK_TWO_DIAMOND,
            }
            and not self.unsupported_capabilities
            and self.required_capabilities.issubset(
                _PUBLIC_CYCLIC_SUPPORTED_CAPABILITIES
            )
        )


@dataclass(frozen=True, slots=True)
class _WriterTwoCycleBlockEnvelope:
    cycle_atom_sets: tuple[frozenset[AtomId], frozenset[AtomId]]
    cycle_bond_sets: tuple[frozenset[BondId], frozenset[BondId]]
    connector_atom_path: tuple[AtomId, ...]
    connector_bond_path: tuple[BondId, ...]

    @property
    def connector_bonds(self) -> frozenset[BondId]:
        return frozenset(self.connector_bond_path)


@dataclass(frozen=True, slots=True)
class _WriterFusedRankTwoDiamondEnvelope:
    atoms: frozenset[AtomId]
    bonds: frozenset[BondId]
    block_id: int
    shared_bond: BondId


@dataclass(frozen=True, slots=True)
class _WriterDirectionalRingCarrierEnvelope:
    site: SiteId
    center_bond: BondId
    ring_carrier_bonds: tuple[BondId, BondId]
    noncarrier_ring_bond: BondId
    pendant_carrier_bonds: tuple[BondId, BondId]


@dataclass(frozen=True, slots=True)
class _WriterSharedDirectionalRingCarrierEnvelope:
    sites: tuple[SiteId, SiteId]
    center_bonds: tuple[BondId, BondId]
    shared_ring_carrier: BondId
    outer_ring_carriers: tuple[BondId, BondId]
    noncarrier_ring_bond: BondId
    pendant_carriers: tuple[BondId, BondId, BondId, BondId]


@dataclass(frozen=True, slots=True)
class _WriterSharedDirectionalRingCarrierShape:
    templates: tuple[DirectionalTemplate, DirectionalTemplate]
    center_bonds: tuple[BondId, BondId]
    shared_ring_carrier: BondId
    outer_ring_carriers: tuple[BondId, BondId]
    noncarrier_ring_bond: BondId
    pendant_carriers: tuple[BondId, BondId, BondId, BondId]

    @property
    def ring_carriers(self) -> tuple[BondId, BondId, BondId]:
        return (
            self.shared_ring_carrier,
            *self.outer_ring_carriers,
        )


@dataclass(frozen=True, slots=True)
class _WriterCyclicBondRoles:
    closure_candidate_bonds: frozenset[BondId]
    tree_only_bonds: frozenset[BondId]


@dataclass(frozen=True, slots=True)
class _WriterTwoCycleBondPolicyReport:
    unsupported_tree_bonds: frozenset[BondId]
    unsupported_closure_bonds: frozenset[BondId]
    visible_tree_bonds: frozenset[BondId]
    visible_closure_bonds: frozenset[BondId]

    @property
    def unsupported_bonds(self) -> frozenset[BondId]:
        return self.unsupported_tree_bonds | self.unsupported_closure_bonds

    @property
    def supported(self) -> bool:
        return not self.unsupported_bonds


@dataclass(frozen=True, slots=True)
class _WriterFusedRankTwoDiamondPolicyReport:
    unsupported_tree_bonds: frozenset[BondId]
    unsupported_closure_bonds: frozenset[BondId]

    @property
    def unsupported_bonds(self) -> frozenset[BondId]:
        return self.unsupported_tree_bonds | self.unsupported_closure_bonds

    @property
    def supported(self) -> bool:
        return not self.unsupported_bonds


@dataclass(frozen=True, slots=True)
class _WriterCyclicAdmissionDecision:
    kind: _WriterCyclicAdmissionDecisionKind
    readiness_gate: _WriterResidualCyclicReadinessGate
    public_profile: _WriterPublicCyclicOpeningProfileReport | None = None
    execution_capability_certificate: (
        _WriterPublicExecutionCapabilityCertificate | None
    ) = None

    def __post_init__(self) -> None:
        if (
            self.kind
            is _WriterCyclicAdmissionDecisionKind.READY_PUBLIC
        ):
            valid = (
                self.readiness_gate.ready
                and self.public_profile is not None
                and self.public_profile.supported
                and self.execution_capability_certificate is not None
                and self.execution_capability_certificate.ready
                and (
                    self.execution_capability_certificate.required_capabilities
                    <= self.execution_capability_certificate.supported_capabilities
                )
            )
        elif (
            self.kind
            is (
                _WriterCyclicAdmissionDecisionKind
                .READY_BUT_PUBLIC_CLOSED
            )
        ):
            valid = self.readiness_gate.ready
        elif (
            self.kind
            is (
                _WriterCyclicAdmissionDecisionKind
                .BLOCKED_RESIDUAL_CYCLIC_POLICY
            )
        ):
            valid = self.readiness_gate.blocked
        elif (
            self.kind
            is _WriterCyclicAdmissionDecisionKind.TRUNCATED_READINESS_AUDIT
        ):
            valid = self.readiness_gate.truncated
        elif (
            self.kind
            is (
                _WriterCyclicAdmissionDecisionKind
                .BLOCKED_PUBLIC_CYCLIC_PROFILE
            )
        ):
            valid = (
                (self.readiness_gate.ready or self.readiness_gate.blocked)
                and self.public_profile is not None
                and not self.public_profile.supported
            )
        elif (
            self.kind
            is (
                _WriterCyclicAdmissionDecisionKind
                .BLOCKED_PUBLIC_EXECUTION_CAPABILITY
            )
        ):
            valid = (
                self.readiness_gate.ready
                and self.public_profile is not None
                and self.public_profile.supported
                and self.execution_capability_certificate is not None
                and not self.execution_capability_certificate.ready
            )
        else:
            valid = False

        if not valid:
            raise SouthStarError(
                SouthStarErrorKind.INTERNAL_INVARIANT,
                f"invalid cyclic writer admission decision: {self.kind!r}",
            )

    @property
    def internally_ready(self) -> bool:
        return self.kind in {
            _WriterCyclicAdmissionDecisionKind.READY_PUBLIC,
            _WriterCyclicAdmissionDecisionKind.READY_BUT_PUBLIC_CLOSED,
            _WriterCyclicAdmissionDecisionKind.BLOCKED_PUBLIC_CYCLIC_PROFILE,
            _WriterCyclicAdmissionDecisionKind
            .BLOCKED_PUBLIC_EXECUTION_CAPABILITY,
        }

    @property
    def public_enabled(self) -> bool:
        return self.kind is _WriterCyclicAdmissionDecisionKind.READY_PUBLIC

    @property
    def admitted_publicly(self) -> bool:
        return self.internally_ready and self.public_enabled

    @property
    def blocked(self) -> bool:
        return (
            self.kind
            is (
                _WriterCyclicAdmissionDecisionKind
                .BLOCKED_RESIDUAL_CYCLIC_POLICY
            )
        )

    @property
    def truncated(self) -> bool:
        return (
            self.kind
            is _WriterCyclicAdmissionDecisionKind.TRUNCATED_READINESS_AUDIT
        )

    @property
    def first_blocked_prefix(
        self,
    ) -> _WriterResidualCyclicReadinessBlockedPrefix | None:
        return self.readiness_gate.first_blocked_prefix

    @property
    def blocked_emitted_texts(self) -> tuple[tuple[str, ...], ...]:
        return self.readiness_gate.blocked_emitted_texts


def _maybe_writer_frontier_choice_snapshot_entry_for_emitted_text(
    choice_snapshot: _WriterFrontierChoiceSnapshot,
    emitted_text: str,
) -> _WriterFrontierChoiceSnapshotEntry | None:
    matches = tuple(
        choice
        for choice in choice_snapshot.choices
        if choice.emitted_text == emitted_text
    )

    if len(matches) > 1:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            (
                "writer choice snapshot contains duplicate emitted-text "
                f"entries: {emitted_text!r}"
            ),
        )

    if not matches:
        return None

    return matches[0]


def _writer_frontier_choice_snapshot_entry_for_emitted_text(
    choice_snapshot: _WriterFrontierChoiceSnapshot,
    emitted_text: str,
) -> _WriterFrontierChoiceSnapshotEntry:
    choice = _maybe_writer_frontier_choice_snapshot_entry_for_emitted_text(
        choice_snapshot,
        emitted_text,
    )

    if choice is None:
        raise SouthStarError(
            SouthStarErrorKind.INVALID_FACTS,
            (
                "writer snapshot emitted text is not in the current "
                f"frontier: {emitted_text!r}"
            ),
        )

    return choice


def _writer_search_snapshot_with_cursor_after_emitted_text(
    snapshot: WriterSearchSnapshot,
    *,
    prepared: SouthStarPreparedMol,
    cursor: WriterFrontierCursor,
) -> WriterSearchSnapshot:
    next_boundary = WriterDecoderBoundary(
        consumed_token_count=(
            snapshot.decoder_boundary.consumed_token_count + 1
        )
    )

    advanced = replace(
        snapshot,
        cursor=cursor,
        decoder_boundary=next_boundary,
        frame_stack=(WriterFrontierFrame(cursor),),
    )

    validate_writer_search_snapshot(
        advanced,
        prepared=prepared,
    )

    return advanced


def _writer_snapshot_advance_outcome_by_emitted_text(
    snapshot: WriterSearchSnapshot,
    *,
    prepared: SouthStarPreparedMol,
    emitted_text: str,
) -> _WriterSnapshotAdvanceOutcome:
    choice_snapshot = _writer_frontier_choice_snapshot_from_snapshot(
        snapshot,
        prepared=prepared,
        include_counts=False,
        stop_after_first_blocked=True,
    )

    if choice_snapshot.blocked:
        return _WriterSnapshotAdvanceOutcome(
            kind=_WriterSnapshotAdvanceOutcomeKind.BLOCKED,
            source_snapshot=snapshot,
            emitted_text=emitted_text,
            choice_snapshot=choice_snapshot,
        )

    choice = _maybe_writer_frontier_choice_snapshot_entry_for_emitted_text(
        choice_snapshot,
        emitted_text,
    )

    if choice is None:
        return _WriterSnapshotAdvanceOutcome(
            kind=_WriterSnapshotAdvanceOutcomeKind.INVALID_EMITTED_TEXT,
            source_snapshot=snapshot,
            emitted_text=emitted_text,
            choice_snapshot=choice_snapshot,
        )

    advanced_snapshot = _writer_search_snapshot_with_cursor_after_emitted_text(
        snapshot,
        prepared=prepared,
        cursor=choice.successor,
    )

    return _WriterSnapshotAdvanceOutcome(
        kind=_WriterSnapshotAdvanceOutcomeKind.ADVANCED,
        source_snapshot=snapshot,
        emitted_text=emitted_text,
        choice_snapshot=choice_snapshot,
        choice=choice,
        advanced_snapshot=advanced_snapshot,
    )


def _writer_snapshot_advance_sequence_outcome_by_emitted_texts(
    snapshot: WriterSearchSnapshot,
    *,
    prepared: SouthStarPreparedMol,
    emitted_texts: tuple[str, ...],
) -> _WriterSnapshotAdvanceSequenceOutcome:
    current = snapshot
    step_outcomes: list[_WriterSnapshotAdvanceOutcome] = []

    for emitted_text in emitted_texts:
        step = _writer_snapshot_advance_outcome_by_emitted_text(
            current,
            prepared=prepared,
            emitted_text=emitted_text,
        )
        step_outcomes.append(step)

        if step.kind is _WriterSnapshotAdvanceOutcomeKind.ADVANCED:
            if step.advanced_snapshot is None:
                raise SouthStarError(
                    SouthStarErrorKind.INTERNAL_INVARIANT,
                    "advanced writer snapshot step did not contain a snapshot",
                )

            current = step.advanced_snapshot
            continue

        if step.kind is _WriterSnapshotAdvanceOutcomeKind.BLOCKED:
            return _WriterSnapshotAdvanceSequenceOutcome(
                kind=_WriterSnapshotAdvanceSequenceOutcomeKind.BLOCKED,
                source_snapshot=snapshot,
                emitted_texts=emitted_texts,
                step_outcomes=tuple(step_outcomes),
                current_snapshot=current,
            )

        if step.kind is _WriterSnapshotAdvanceOutcomeKind.INVALID_EMITTED_TEXT:
            return _WriterSnapshotAdvanceSequenceOutcome(
                kind=(
                    _WriterSnapshotAdvanceSequenceOutcomeKind
                    .INVALID_EMITTED_TEXT
                ),
                source_snapshot=snapshot,
                emitted_texts=emitted_texts,
                step_outcomes=tuple(step_outcomes),
                current_snapshot=current,
            )

        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            f"unknown writer snapshot advance step outcome: {step.kind!r}",
        )

    return _WriterSnapshotAdvanceSequenceOutcome(
        kind=_WriterSnapshotAdvanceSequenceOutcomeKind.ADVANCED,
        source_snapshot=snapshot,
        emitted_texts=emitted_texts,
        step_outcomes=tuple(step_outcomes),
        current_snapshot=current,
    )


def _writer_frontier_choice_snapshot_after_emitted_texts(
    snapshot: WriterSearchSnapshot,
    *,
    prepared: SouthStarPreparedMol,
    emitted_texts: tuple[str, ...],
    include_counts: bool = True,
    stop_after_first_blocked: bool = False,
) -> _WriterSnapshotReplayChoiceSnapshotOutcome:
    sequence_outcome = (
        _writer_snapshot_advance_sequence_outcome_by_emitted_texts(
            snapshot,
            prepared=prepared,
            emitted_texts=emitted_texts,
        )
    )

    if (
        sequence_outcome.kind
        is _WriterSnapshotAdvanceSequenceOutcomeKind.BLOCKED
    ):
        return _WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                _WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .REPLAY_BLOCKED
            ),
            source_snapshot=snapshot,
            emitted_texts=emitted_texts,
            sequence_outcome=sequence_outcome,
        )

    if (
        sequence_outcome.kind
        is (
            _WriterSnapshotAdvanceSequenceOutcomeKind
            .INVALID_EMITTED_TEXT
        )
    ):
        return _WriterSnapshotReplayChoiceSnapshotOutcome(
            kind=(
                _WriterSnapshotReplayChoiceSnapshotOutcomeKind
                .INVALID_EMITTED_TEXT
            ),
            source_snapshot=snapshot,
            emitted_texts=emitted_texts,
            sequence_outcome=sequence_outcome,
        )

    advanced_snapshot = sequence_outcome.advanced_snapshot
    if advanced_snapshot is None:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "advanced replay outcome did not contain a snapshot",
        )

    choice_snapshot = _writer_frontier_choice_snapshot_from_snapshot(
        advanced_snapshot,
        prepared=prepared,
        include_counts=include_counts,
        stop_after_first_blocked=stop_after_first_blocked,
    )

    return _WriterSnapshotReplayChoiceSnapshotOutcome(
        kind=_WriterSnapshotReplayChoiceSnapshotOutcomeKind.CHOICE_SNAPSHOT,
        source_snapshot=snapshot,
        emitted_texts=emitted_texts,
        sequence_outcome=sequence_outcome,
        choice_snapshot=choice_snapshot,
    )


def _raise_for_writer_snapshot_advance_outcome_errors(
    outcome: _WriterSnapshotAdvanceOutcome,
) -> None:
    if outcome.kind is _WriterSnapshotAdvanceOutcomeKind.BLOCKED:
        _raise_for_writer_frontier_schedule_outcome_blockers(
            outcome.choice_snapshot.schedule_outcome,
        )
        return

    if (
        outcome.kind
        is _WriterSnapshotAdvanceOutcomeKind.INVALID_EMITTED_TEXT
    ):
        raise SouthStarError(
            SouthStarErrorKind.INVALID_FACTS,
            (
                "writer snapshot emitted text is not in the current "
                f"frontier: {outcome.emitted_text!r}"
            ),
        )


def _raise_for_writer_snapshot_advance_sequence_outcome_errors(
    outcome: _WriterSnapshotAdvanceSequenceOutcome,
) -> None:
    failed = outcome.failed_outcome

    if failed is None:
        return

    _raise_for_writer_snapshot_advance_outcome_errors(failed)


def _raise_for_writer_snapshot_replay_choice_snapshot_outcome_errors(
    outcome: _WriterSnapshotReplayChoiceSnapshotOutcome,
) -> None:
    if (
        outcome.kind
        is _WriterSnapshotReplayChoiceSnapshotOutcomeKind.REPLAY_BLOCKED
    ):
        _raise_for_writer_snapshot_advance_sequence_outcome_errors(
            outcome.sequence_outcome
        )
        return

    if (
        outcome.kind
        is (
            _WriterSnapshotReplayChoiceSnapshotOutcomeKind
            .INVALID_EMITTED_TEXT
        )
    ):
        _raise_for_writer_snapshot_advance_sequence_outcome_errors(
            outcome.sequence_outcome
        )
        return

    if outcome.choice_snapshot is None:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "replay choice snapshot outcome did not contain a choice snapshot",
        )

    _raise_for_writer_frontier_schedule_outcome_blockers(
        outcome.choice_snapshot.schedule_outcome,
    )


def _advance_writer_search_snapshot_by_emitted_text(
    snapshot: WriterSearchSnapshot,
    *,
    prepared: SouthStarPreparedMol,
    emitted_text: str,
) -> WriterSearchSnapshot:
    outcome = _writer_snapshot_advance_outcome_by_emitted_text(
        snapshot,
        prepared=prepared,
        emitted_text=emitted_text,
    )

    _raise_for_writer_snapshot_advance_outcome_errors(outcome)

    if outcome.advanced_snapshot is None:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "writer snapshot advance outcome did not contain a snapshot",
        )

    return outcome.advanced_snapshot


def _advance_writer_search_snapshot_by_emitted_texts(
    snapshot: WriterSearchSnapshot,
    *,
    prepared: SouthStarPreparedMol,
    emitted_texts: tuple[str, ...],
) -> WriterSearchSnapshot:
    outcome = _writer_snapshot_advance_sequence_outcome_by_emitted_texts(
        snapshot,
        prepared=prepared,
        emitted_texts=emitted_texts,
    )

    _raise_for_writer_snapshot_advance_sequence_outcome_errors(outcome)

    if outcome.advanced_snapshot is None:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "writer snapshot advance sequence outcome did not contain a snapshot",
        )

    return outcome.advanced_snapshot


def _checked_writer_frontier_choice_snapshot_after_emitted_texts(
    snapshot: WriterSearchSnapshot,
    *,
    prepared: SouthStarPreparedMol,
    emitted_texts: tuple[str, ...],
    include_counts: bool = True,
) -> _WriterFrontierChoiceSnapshot:
    outcome = _writer_frontier_choice_snapshot_after_emitted_texts(
        snapshot,
        prepared=prepared,
        emitted_texts=emitted_texts,
        include_counts=include_counts,
        stop_after_first_blocked=True,
    )

    _raise_for_writer_snapshot_replay_choice_snapshot_outcome_errors(
        outcome
    )

    if outcome.choice_snapshot is None:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "checked replay did not contain a choice snapshot",
        )

    return outcome.choice_snapshot


def _writer_frontier_choices_after_emitted_texts(
    snapshot: WriterSearchSnapshot,
    *,
    prepared: SouthStarPreparedMol,
    emitted_texts: tuple[str, ...],
) -> WriterFrontierChoices:
    outcome = _checked_writer_snapshot_prefix_read_outcome_after_emitted_texts(
        snapshot,
        prepared=prepared,
        emitted_texts=emitted_texts,
        include_counts=True,
    )

    if outcome.public_choices is None:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "checked prefix read did not contain public choices",
        )

    return outcome.public_choices


def _count_writer_frontier_choice_snapshot_supports(
    choice_snapshot: _WriterFrontierChoiceSnapshot,
) -> int:
    total = 0

    if choice_snapshot.terminal is not None:
        total += choice_snapshot.terminal.support_count

    for choice in choice_snapshot.choices:
        if choice.support_count is None:
            raise SouthStarError(
                SouthStarErrorKind.INTERNAL_INVARIANT,
                "writer frontier choice snapshot is missing support counts",
            )

        total += choice.support_count

    return total


def _count_writer_frontier_choice_snapshot_completions(
    choice_snapshot: _WriterFrontierChoiceSnapshot,
) -> int:
    total = 0

    if choice_snapshot.terminal is not None:
        total += choice_snapshot.terminal.completion_count

    for choice in choice_snapshot.choices:
        if choice.completion_count is None:
            raise SouthStarError(
                SouthStarErrorKind.INTERNAL_INVARIANT,
                "writer frontier choice snapshot is missing completion counts",
            )

        total += choice.completion_count

    return total


def _iter_writer_frontier_support_suffixes_from_choice_snapshot(
    prepared: SouthStarPreparedMol,
    choice_snapshot: _WriterFrontierChoiceSnapshot,
) -> Iterator[str]:
    if choice_snapshot.terminal is not None:
        yield ""

    for choice in choice_snapshot.choices:
        for suffix in iter_writer_frontier_support(
            prepared,
            choice.successor,
        ):
            yield choice.emitted_text + suffix


def _writer_snapshot_prefix_read_outcome_after_emitted_texts(
    snapshot: WriterSearchSnapshot,
    *,
    prepared: SouthStarPreparedMol,
    emitted_texts: tuple[str, ...],
    include_counts: bool = True,
    stop_after_first_blocked: bool = False,
) -> _WriterSnapshotPrefixReadOutcome:
    replay_outcome = _writer_frontier_choice_snapshot_after_emitted_texts(
        snapshot,
        prepared=prepared,
        emitted_texts=emitted_texts,
        include_counts=include_counts,
        stop_after_first_blocked=stop_after_first_blocked,
    )

    if (
        replay_outcome.kind
        is _WriterSnapshotReplayChoiceSnapshotOutcomeKind.REPLAY_BLOCKED
    ):
        return _WriterSnapshotPrefixReadOutcome(
            kind=_WriterSnapshotPrefixReadOutcomeKind.REPLAY_BLOCKED,
            replay_outcome=replay_outcome,
        )

    if (
        replay_outcome.kind
        is (
            _WriterSnapshotReplayChoiceSnapshotOutcomeKind
            .INVALID_EMITTED_TEXT
        )
    ):
        return _WriterSnapshotPrefixReadOutcome(
            kind=_WriterSnapshotPrefixReadOutcomeKind.INVALID_EMITTED_TEXT,
            replay_outcome=replay_outcome,
        )

    choice_snapshot = replay_outcome.choice_snapshot
    if choice_snapshot is None:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "prefix read replay outcome did not contain a choice snapshot",
        )

    if choice_snapshot.blocked:
        return _WriterSnapshotPrefixReadOutcome(
            kind=_WriterSnapshotPrefixReadOutcomeKind.FINAL_FRONTIER_BLOCKED,
            replay_outcome=replay_outcome,
        )

    support_count = None
    completion_count = None

    if include_counts:
        support_count = _count_writer_frontier_choice_snapshot_supports(
            choice_snapshot
        )
        completion_count = _count_writer_frontier_choice_snapshot_completions(
            choice_snapshot
        )

    return _WriterSnapshotPrefixReadOutcome(
        kind=_WriterSnapshotPrefixReadOutcomeKind.READABLE,
        replay_outcome=replay_outcome,
        support_count=support_count,
        completion_count=completion_count,
    )


def _audit_residual_cyclic_readiness_from_snapshot(
    snapshot: WriterSearchSnapshot,
    *,
    prepared: SouthStarPreparedMol,
    max_depth: int | None = None,
    max_prefixes: int | None = None,
) -> _WriterResidualCyclicReadinessAudit:
    visited: list[tuple[str, ...]] = []
    blocked: list[_WriterResidualCyclicReadinessBlockedPrefix] = []
    execution_capability_uses: list[_WriterExecutionCapabilityUse] = []
    observed_execution_capability_use_signatures: set[
        tuple[
            _WriterExecutionCapabilityKind,
            tuple[str, ...],
            str | None,
            WriterFrontierCursor,
            WriterFrontierCursor,
        ]
    ] = set()
    seen_cursors: set[WriterFrontierCursor] = set()

    def rec(
        current: WriterSearchSnapshot,
        prefix: tuple[str, ...],
    ) -> tuple[bool, tuple[str, ...] | None]:
        if current.cursor in seen_cursors:
            return False, None

        seen_cursors.add(current.cursor)
        visited.append(prefix)

        if max_prefixes is not None and len(visited) > max_prefixes:
            return True, prefix

        if max_depth is not None and len(prefix) > max_depth:
            return True, prefix

        choice_snapshot = _writer_frontier_choice_snapshot_from_snapshot(
            current,
            prepared=prepared,
            include_counts=False,
            stop_after_first_blocked=True,
        )
        report = choice_snapshot.residual_cyclic_policy_readiness_report

        if report.blocked:
            blocked.append(
                _WriterResidualCyclicReadinessBlockedPrefix(
                    emitted_texts=prefix,
                    choice_snapshot=choice_snapshot,
                )
            )
            return False, None

        if choice_snapshot.terminal is not None:
            for capability in choice_snapshot.terminal_execution_capabilities:
                use = _WriterExecutionCapabilityUse(
                    kind=capability,
                    emitted_texts=prefix,
                    source_cursor=current.cursor,
                    successor_cursor=choice_snapshot.terminal.finalized_cursor,
                    next_emitted_text=None,
                )
                signature = (
                    use.kind,
                    use.emitted_texts,
                    use.next_emitted_text,
                    use.source_cursor,
                    use.successor_cursor,
                )
                if signature not in observed_execution_capability_use_signatures:
                    observed_execution_capability_use_signatures.add(signature)
                    execution_capability_uses.append(use)

        for choice in choice_snapshot.choices:
            for capability in choice.execution_capabilities:
                use = _WriterExecutionCapabilityUse(
                    kind=capability,
                    emitted_texts=prefix,
                    source_cursor=current.cursor,
                    successor_cursor=choice.successor,
                    next_emitted_text=choice.emitted_text,
                )

                signature = (
                    use.kind,
                    use.emitted_texts,
                    use.next_emitted_text,
                    use.source_cursor,
                    use.successor_cursor,
                )

                if signature in observed_execution_capability_use_signatures:
                    continue

                observed_execution_capability_use_signatures.add(signature)
                execution_capability_uses.append(use)

            successor = _writer_search_snapshot_with_cursor_after_emitted_text(
                current,
                prepared=prepared,
                cursor=choice.successor,
            )
            stopped, stopped_prefix = rec(
                successor,
                (*prefix, choice.emitted_text),
            )

            if stopped:
                return True, stopped_prefix

        return False, None

    truncated, truncated_prefix = rec(snapshot, ())

    if truncated:
        return _WriterResidualCyclicReadinessAudit(
            kind=_WriterResidualCyclicReadinessAuditKind.TRUNCATED,
            visited_prefixes=tuple(visited),
            blocked_prefixes=tuple(blocked),
            truncated_at_prefix=truncated_prefix,
            execution_capability_uses=tuple(execution_capability_uses),
        )

    if blocked:
        return _WriterResidualCyclicReadinessAudit(
            kind=_WriterResidualCyclicReadinessAuditKind.BLOCKED,
            visited_prefixes=tuple(visited),
            blocked_prefixes=tuple(blocked),
            execution_capability_uses=tuple(execution_capability_uses),
        )

    return _WriterResidualCyclicReadinessAudit(
        kind=_WriterResidualCyclicReadinessAuditKind.READY,
        visited_prefixes=tuple(visited),
        execution_capability_uses=tuple(execution_capability_uses),
    )


def _assert_residual_cyclic_readiness_from_snapshot(
    snapshot: WriterSearchSnapshot,
    *,
    prepared: SouthStarPreparedMol,
    max_depth: int | None = None,
    max_prefixes: int | None = None,
) -> _WriterResidualCyclicReadinessAudit:
    audit = _audit_residual_cyclic_readiness_from_snapshot(
        snapshot,
        prepared=prepared,
        max_depth=max_depth,
        max_prefixes=max_prefixes,
    )

    if not audit.ready:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_POLICY,
            f"residual cyclic readiness audit failed: {audit.kind!r}",
        )

    return audit


def _residual_cyclic_readiness_gate_from_snapshot(
    snapshot: WriterSearchSnapshot,
    *,
    prepared: SouthStarPreparedMol,
    max_depth: int | None = None,
    max_prefixes: int | None = None,
) -> _WriterResidualCyclicReadinessGate:
    audit = _audit_residual_cyclic_readiness_from_snapshot(
        snapshot,
        prepared=prepared,
        max_depth=max_depth,
        max_prefixes=max_prefixes,
    )

    if audit.ready:
        kind = _WriterResidualCyclicReadinessGateKind.READY
    elif audit.blocked:
        kind = _WriterResidualCyclicReadinessGateKind.BLOCKED
    elif audit.truncated:
        kind = _WriterResidualCyclicReadinessGateKind.TRUNCATED
    else:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            f"unknown residual cyclic readiness audit state: {audit.kind!r}",
        )

    return _WriterResidualCyclicReadinessGate(
        kind=kind,
        snapshot=snapshot,
        audit=audit,
    )


def _residual_cyclic_readiness_gate_from_cursor(
    *,
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions,
    cursor: WriterFrontierCursor,
    max_depth: int | None = None,
    max_prefixes: int | None = None,
) -> _WriterResidualCyclicReadinessGate:
    snapshot = _capture_writer_frontier_snapshot_unchecked(
        prepared=prepared,
        runtime_options=runtime_options,
        cursor=cursor,
    )

    return _residual_cyclic_readiness_gate_from_snapshot(
        snapshot,
        prepared=prepared,
        max_depth=max_depth,
        max_prefixes=max_prefixes,
    )


def _assert_residual_cyclic_readiness_gate(
    gate: _WriterResidualCyclicReadinessGate,
) -> _WriterResidualCyclicReadinessGate:
    if gate.ready:
        return gate

    if gate.blocked:
        first = gate.first_blocked_prefix
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_POLICY,
            (
                "residual cyclic readiness gate blocked"
                if first is None
                else (
                    "residual cyclic readiness gate blocked at prefix "
                    f"{first.emitted_texts!r}"
                )
            ),
        )

    if gate.truncated:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_POLICY,
            (
                "residual cyclic readiness gate truncated at prefix "
                f"{gate.truncated_at_prefix!r}"
            ),
        )

    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"unknown residual cyclic readiness gate state: {gate.kind!r}",
    )


def _writer_public_cyclic_opening_profile_report(
    *,
    prepared: SouthStarPreparedMol,
) -> _WriterPublicCyclicOpeningProfileReport:
    surfaces = tuple(prepared.writer_graph_metadata.component_surfaces)

    def blocked_profile(
        *,
        kind: _WriterPublicCyclicOpeningProfileKind,
        **extra: object,
    ) -> _WriterPublicCyclicOpeningProfileReport:
        defaults = _WriterPublicCyclicOpeningProfileReport(
            kind=kind,
            component_count=len(surfaces),
            cyclic_component_count=len(tuple(
                surface
                for surface in surfaces
                if (
                    surface.connected
                    and not surface.tree
                    and (
                        surface.cyclic_rank > 0
                        or surface.cyclic_block_ids
                    )
                )
            )),
            cyclic_ranks=tuple(
                int(surface.cyclic_rank)
                for surface in surfaces
                if (
                    surface.connected
                    and not surface.tree
                    and (
                        surface.cyclic_rank > 0
                        or surface.cyclic_block_ids
                    )
                )
            ),
            ring_core_atom_count=0,
            ring_core_bond_count=0,
            ring_core_max_degree=0,
            pendant_atom_count=0,
            pendant_bond_count=0,
            component_atom_count=0,
            component_bond_count=0,
            max_component_degree=0,
            branch_atom_count=0,
            unsupported_bond_count=0,
            unsupported_stereo_surface_count=0,
            pendant_component_count=0,
            pendant_component_atom_counts=(),
            pendant_component_boundary_counts=(),
        )

        return replace(
            defaults,
            **extra,
        )

    if len(surfaces) != 1:
        return blocked_profile(
            kind=_WriterPublicCyclicOpeningProfileKind.BLOCKED_NOT_SINGLE_COMPONENT,
        )

    surface = surfaces[0]
    component_atom_ids = tuple(surface.atoms)
    component_bond_ids = tuple(surface.bonds)
    component_atom_count = len(component_atom_ids)
    component_bond_count = len(component_bond_ids)
    component_bond_index = prepared.graph_index.bond_by_id

    component_adjacency: dict[AtomId, set[AtomId]] = {
        atom: set() for atom in component_atom_ids
    }

    for bond_id in component_bond_ids:
        bond = component_bond_index.get(bond_id)
        if bond is None:
            continue

        left = bond.a
        right = bond.b
        component_adjacency.setdefault(left, set()).add(right)
        component_adjacency.setdefault(right, set()).add(left)

    adjacency: dict[AtomId, set[AtomId]] = {
        atom: set(neighbors)
        for atom, neighbors in component_adjacency.items()
    }

    component_atom_degrees = tuple(
        len(component_adjacency[atom]) for atom in component_atom_ids
    )
    max_component_degree = max(component_atom_degrees, default=0)
    component_core_atoms = set(component_atom_ids)
    prune = [
        atom
        for atom in component_atom_ids
        if len(adjacency[atom]) <= 1
    ]

    while prune:
        atom = prune.pop()
        if atom not in component_core_atoms:
            continue

        component_core_atoms.remove(atom)
        neighbors = tuple(adjacency.get(atom, ()))
        for neighbor in neighbors:
            if neighbor not in component_core_atoms:
                continue

            adjacency[neighbor].discard(atom)
            if neighbor in component_core_atoms and len(adjacency[neighbor]) <= 1:
                prune.append(neighbor)

    unsupported_stereo_surface_count = sum(
        1
        for template in prepared.directional_templates
        if template.center_bond in surface.bonds
    )

    ring_core_atom_count = len(component_core_atoms)
    ring_core_atom_set = set(component_core_atoms)
    ring_core_bond_ids = tuple(
        bond_id
        for bond_id in component_bond_ids
        if (
            (bond := component_bond_index.get(bond_id))
            is not None
            and bond.a in ring_core_atom_set
            and bond.b in ring_core_atom_set
        )
    )
    ring_core_bond_count = len(ring_core_bond_ids)
    ring_core_bond_id_set = frozenset(ring_core_bond_ids)
    pendant_bond_ids = tuple(
        bond_id
        for bond_id in component_bond_ids
        if bond_id not in ring_core_bond_id_set
    )
    pendant_atom_set = set(component_atom_ids) - ring_core_atom_set
    ring_core_max_degree = max(
        (
            len(adjacency[atom])
            for atom in component_core_atoms
            if atom in adjacency
        ),
        default=0,
    )
    two_cycle_envelope = _writer_two_bridge_separated_simple_cycles(
        prepared,
        surface,
        ring_core_atoms=frozenset(component_core_atoms),
        ring_core_bonds=frozenset(ring_core_bond_ids),
    )
    fused_diamond_envelope = _writer_fused_rank_two_diamond(
        prepared,
        surface,
        ring_core_atoms=frozenset(component_core_atoms),
        ring_core_bonds=frozenset(ring_core_bond_ids),
    )
    fused_diamond_policy_report = (
        _writer_fused_rank_two_diamond_policy_report(
            prepared,
            fused_diamond_envelope,
        )
        if fused_diamond_envelope is not None
        else None
    )
    bond_roles = _writer_public_cyclic_bond_roles(
        ring_core_bond_ids=ring_core_bond_ids,
        two_cycle_envelope=two_cycle_envelope,
    )
    two_cycle_bond_policy_report = (
        _writer_two_cycle_bond_policy_report(
            prepared,
            bond_roles,
            two_cycle_envelope,
        )
        if two_cycle_envelope is not None
        else None
    )
    closure_bond_ids = tuple(
        sorted(bond_roles.closure_candidate_bonds)
    )
    ring_core_aromatic_bond_ids = tuple(
        bond_id
        for bond_id in closure_bond_ids
        if (
            (bond := component_bond_index.get(bond_id))
            is not None
            and bond.order is BondOrder.AROMATIC
        )
    )
    ring_core_non_single_bond_ids = tuple(
        bond_id
        for bond_id in closure_bond_ids
        if (
            (bond := component_bond_index.get(bond_id))
            is not None
            and bond.order not in {BondOrder.SINGLE, BondOrder.AROMATIC}
        )
    )
    shared_directional_ring_carrier_shape = (
        _writer_public_shared_directional_ring_carrier_shape(
            prepared=prepared,
            ring_core_atom_ids=frozenset(ring_core_atom_set),
            ring_core_bond_ids=frozenset(ring_core_bond_ids),
            pendant_atom_ids=frozenset(pendant_atom_set),
            pendant_bond_ids=frozenset(pendant_bond_ids),
            ring_core_non_single_bond_ids=ring_core_non_single_bond_ids,
        )
    )
    shared_directional_ring_carrier_envelope = (
        _writer_public_shared_directional_semantics(
            prepared,
            shared_directional_ring_carrier_shape,
        )
        if (
            shared_directional_ring_carrier_shape is not None
            and _writer_public_shared_directional_raw_policy_is_supported(
                prepared,
                shared_directional_ring_carrier_shape,
            )
        )
        else None
    )
    ring_core_single_closure_relations = tuple(
        relation
        for bond_id in closure_bond_ids
        if (
            shared_directional_ring_carrier_shape is None
            and fused_diamond_envelope is None
            and
            (bond := component_bond_index.get(bond_id))
            is not None
            and bond.order is BondOrder.SINGLE
            and (
                relation := _writer_public_single_closure_relation(
                    prepared,
                    bond_id,
                )
            )
            is not None
        )
    )
    ring_core_single_bond_ids = tuple(
        bond_id
        for bond_id in closure_bond_ids
        if (
            (bond := component_bond_index.get(bond_id))
            is not None
            and bond.order is BondOrder.SINGLE
        )
    )
    ring_core_unsupported_single_closure_bond_count = (
        0
        if (
            shared_directional_ring_carrier_shape is not None
            or fused_diamond_envelope is not None
        )
        else len(ring_core_single_bond_ids)
        - len(ring_core_single_closure_relations)
    )
    ring_core_has_visible_single_closure_bond_text = any(
        "-" in relation.texts
        for relation in ring_core_single_closure_relations
    )
    ring_core_aromatic_is_supported = (
        fused_diamond_envelope is None
        and bool(ring_core_aromatic_bond_ids)
        and len(ring_core_aromatic_bond_ids) == len(ring_core_bond_ids)
        and not prepared.tetra_templates
        and not prepared.directional_templates
        and all(
            prepared.graph_index.atom_by_id[atom_id].is_aromatic
            for atom_id in ring_core_atom_set
        )
        and all(
            _writer_public_aromatic_ring_bond_is_supported(
                prepared,
                bond_id,
            )
            for bond_id in ring_core_aromatic_bond_ids
        )
    )
    ring_core_tetra_template_count = sum(
        1
        for template in prepared.tetra_templates
        if template.center in ring_core_atom_set
    )
    ring_core_tetra_is_supported = (
        ring_core_tetra_template_count > 0
        and (
            _writer_public_two_cycle_ring_tetra_is_supported(
                prepared=prepared,
                envelope=two_cycle_envelope,
                ring_core_bond_ids=frozenset(ring_core_bond_ids),
                component_bond_ids=component_bond_ids,
                bond_policy_report=two_cycle_bond_policy_report,
            )
            if (
                two_cycle_envelope is not None
                and two_cycle_bond_policy_report is not None
            )
            else _writer_public_ring_core_tetrahedral_stereo_is_supported(
                prepared=prepared,
                ring_core_atom_set=frozenset(ring_core_atom_set),
                ring_core_bond_id_set=frozenset(ring_core_bond_ids),
                ring_core_non_single_bond_ids=ring_core_non_single_bond_ids,
            )
        )
    )
    if shared_directional_ring_carrier_envelope is not None:
        expected_non_single_closure_bonds = frozenset(
            ring_core_non_single_bond_ids,
        )
        supported_non_single_closure_bonds = frozenset(
            shared_directional_ring_carrier_envelope.center_bonds,
        )
        ring_core_has_supported_non_single_closure_bond = (
            bool(expected_non_single_closure_bonds)
            and expected_non_single_closure_bonds
            == supported_non_single_closure_bonds
        )
    elif shared_directional_ring_carrier_shape is not None:
        ring_core_has_supported_non_single_closure_bond = False
    elif fused_diamond_envelope is not None:
        ring_core_has_supported_non_single_closure_bond = False
    elif two_cycle_bond_policy_report is not None:
        expected_non_single_closure_bonds = frozenset(
            ring_core_non_single_bond_ids,
        )
        supported_non_single_closure_bonds = (
            two_cycle_bond_policy_report.visible_closure_bonds
            - two_cycle_bond_policy_report.unsupported_closure_bonds
        )
        ring_core_has_supported_non_single_closure_bond = (
            bool(expected_non_single_closure_bonds)
            and expected_non_single_closure_bonds
            == supported_non_single_closure_bonds
        )
    else:
        ring_core_has_supported_non_single_closure_bond = (
            len(ring_core_non_single_bond_ids) == 1
            and (
                bond := component_bond_index.get(
                    ring_core_non_single_bond_ids[0],
                )
            )
            is not None
            and bond.order in {BondOrder.DOUBLE, BondOrder.TRIPLE}
            and _writer_public_non_single_closure_bond_is_supported(
                prepared,
                ring_core_non_single_bond_ids[0],
            )
        )
    ring_core_unsupported_bond_ids: set[BondId] = set()
    if (
        ring_core_non_single_bond_ids
        and not ring_core_has_supported_non_single_closure_bond
    ):
        ring_core_unsupported_bond_ids.update(ring_core_non_single_bond_ids)
    if ring_core_unsupported_single_closure_bond_count:
        ring_core_unsupported_bond_ids.update(
            bond_id
            for bond_id in ring_core_single_bond_ids
            if (
                _writer_public_single_closure_relation(
                    prepared,
                    bond_id,
                )
                is None
            )
        )
    if ring_core_aromatic_bond_ids and not ring_core_aromatic_is_supported:
        ring_core_unsupported_bond_ids.update(ring_core_aromatic_bond_ids)
    if two_cycle_bond_policy_report is not None:
        ring_core_unsupported_bond_ids.update(
            two_cycle_bond_policy_report.unsupported_bonds,
        )
    if fused_diamond_policy_report is not None:
        ring_core_unsupported_bond_ids.update(
            fused_diamond_policy_report.unsupported_bonds,
        )
    ring_core_unsupported_bond_count = len(
        ring_core_unsupported_bond_ids,
    )
    pendant_unsupported_bond_count = 0
    for bond_id in pendant_bond_ids:
        if bond_id not in component_bond_index:
            pendant_unsupported_bond_count += 1
            continue

        try:
            bond_text_domain = prepared.policy.bond_text_domain_unchecked(
                bond_id,
                slot_kind="tree",
            )
        except KeyError:
            # Missing tree-domain registration is treated as unsupported.
            pendant_unsupported_bond_count += 1
        else:
            if not bond_text_domain:
                pendant_unsupported_bond_count += 1

    unsupported_bond_count = (
        ring_core_unsupported_bond_count
        + pendant_unsupported_bond_count
    )

    ring_core_atom_count_is_nontrivial = ring_core_atom_count >= 3
    ring_core_is_simple_cycle = (
        ring_core_atom_count_is_nontrivial
        and ring_core_bond_count == ring_core_atom_count
        and ring_core_max_degree == 2
    )

    branch_atom_count = sum(
        1 for degree in component_atom_degrees if degree > 2
    )
    pendant_atom_count = component_atom_count - ring_core_atom_count
    pendant_bond_count = component_bond_count - ring_core_bond_count

    pendant_component_atom_counts: tuple[int, ...] = ()
    pendant_component_boundary_counts: tuple[int, ...] = ()
    directional_ring_carrier_envelope = (
        _writer_public_directional_ring_carrier_envelope(
            prepared=prepared,
            ring_core_atom_ids=frozenset(ring_core_atom_set),
            ring_core_bond_ids=frozenset(ring_core_bond_ids),
            pendant_atom_ids=frozenset(pendant_atom_set),
            pendant_bond_ids=frozenset(pendant_bond_ids),
            ring_core_non_single_bond_ids=ring_core_non_single_bond_ids,
            non_single_closure_supported=(
                ring_core_has_supported_non_single_closure_bond
            ),
        )
    )
    if (
        directional_ring_carrier_envelope is not None
        or shared_directional_ring_carrier_envelope is not None
    ):
        unsupported_stereo_surface_count = 0
    if pendant_atom_set:
        pendant_components: list[tuple[frozenset[AtomId], int]] = []
        seen_pendant_atoms: set[AtomId] = set()

        for atom in sorted(pendant_atom_set):
            if atom in seen_pendant_atoms:
                continue

            component_atoms: set[AtomId] = set()
            frontier = [atom]
            seen_pendant_atoms.add(atom)

            while frontier:
                current = frontier.pop()
                component_atoms.add(current)
                for next_atom in component_adjacency[current]:
                    if (
                        next_atom not in pendant_atom_set
                        or next_atom in seen_pendant_atoms
                    ):
                        continue

                    seen_pendant_atoms.add(next_atom)
                    frontier.append(next_atom)

            boundary_count = 0
            for component_atom in component_atoms:
                for adjacent_atom in component_adjacency[component_atom]:
                    if adjacent_atom not in component_atoms:
                        boundary_count += 1

            pendant_components.append(
                (frozenset(component_atoms), boundary_count),
            )

        pendant_components.sort(
            key=lambda entry: (
                len(entry[0]),
                tuple(sorted(entry[0])),
            )
        )
        pendant_component_atom_counts = tuple(
            len(entry[0]) for entry in pendant_components
        )
        pendant_component_boundary_counts = tuple(
            entry[1] for entry in pendant_components
        )

    pendant_component_count = len(pendant_component_atom_counts)
    pendant_multi_boundary = any(
        boundary_count != 1
        for boundary_count in pendant_component_boundary_counts
    )
    pendant_non_tree = False
    if pendant_atom_count > 0:
        for component_atoms, _boundary_count in (
            pendant_components
            if pendant_atom_set
            else ()
        ):
            internal_bond_count = 0
            for bond_id in component_bond_ids:
                bond = component_bond_index.get(bond_id)
                if bond is None:
                    continue
                if bond.a in component_atoms and bond.b in component_atoms:
                    internal_bond_count += 1

            if internal_bond_count != len(component_atoms) - 1:
                pendant_non_tree = True
                break

    pendant_forest_supported = (
        pendant_unsupported_bond_count == 0
        and not pendant_multi_boundary
        and not pendant_non_tree
    )

    cyclic_surfaces = tuple(
        surface
        for surface in (surface,)
        if (
            surface.connected
            and not surface.tree
            and (
                surface.cyclic_rank > 0
                or surface.cyclic_block_ids
            )
        )
    )

    cyclic_ranks = tuple(int(surface.cyclic_rank) for surface in cyclic_surfaces)
    required_capabilities: set[
        _WriterPublicCyclicRequiredCapability
    ] = set()
    unsupported_capabilities: set[
        _WriterPublicCyclicRequiredCapability
    ] = set()
    two_cycle_topology_supported = (
        cyclic_ranks == (2,)
        and two_cycle_envelope is not None
    )
    fused_diamond_topology_supported = (
        cyclic_ranks == (2,)
        and fused_diamond_envelope is not None
    )
    two_cycle_label_policy_supported = (
        prepared.policy.least_free_ring_labels
        and len(prepared.policy.ring_labels) >= 2
    )
    fused_diamond_label_policy_supported = (
        prepared.policy.least_free_ring_labels
        and len(prepared.policy.ring_labels) >= 2
    )
    two_cycle_backbone_supported = (
        two_cycle_topology_supported
        and two_cycle_label_policy_supported
        and two_cycle_bond_policy_report is not None
        and two_cycle_bond_policy_report.supported
        and (not prepared.tetra_templates or ring_core_tetra_is_supported)
        and not prepared.directional_templates
    )
    fused_diamond_supported = (
        fused_diamond_topology_supported
        and fused_diamond_label_policy_supported
        and fused_diamond_policy_report is not None
        and fused_diamond_policy_report.supported
        and not prepared.tetra_templates
        and not prepared.directional_templates
    )

    if not surface.connected:
        return blocked_profile(
            kind=_WriterPublicCyclicOpeningProfileKind.BLOCKED_NOT_CONNECTED_COMPONENT,
            component_atom_count=component_atom_count,
            component_bond_count=component_bond_count,
            max_component_degree=max_component_degree,
            branch_atom_count=branch_atom_count,
            ring_core_bond_count=ring_core_bond_count,
            ring_core_atom_count=ring_core_atom_count,
            ring_core_max_degree=ring_core_max_degree,
            pendant_atom_count=pendant_atom_count,
            pendant_bond_count=pendant_bond_count,
            pendant_component_count=pendant_component_count,
            pendant_component_atom_counts=pendant_component_atom_counts,
            pendant_component_boundary_counts=pendant_component_boundary_counts,
            cyclic_ranks=cyclic_ranks,
            cyclic_component_count=len(cyclic_surfaces),
            unsupported_bond_count=unsupported_bond_count,
            ring_core_unsupported_bond_count=ring_core_unsupported_bond_count,
            pendant_unsupported_bond_count=pendant_unsupported_bond_count,
            unsupported_stereo_surface_count=unsupported_stereo_surface_count,
            required_capabilities=frozenset(required_capabilities),
            unsupported_capabilities=frozenset(unsupported_capabilities),
        )

    if not cyclic_surfaces:
        return blocked_profile(
            kind=_WriterPublicCyclicOpeningProfileKind.BLOCKED_NOT_CYCLIC_COMPONENT,
            component_atom_count=component_atom_count,
            component_bond_count=component_bond_count,
            max_component_degree=max_component_degree,
            branch_atom_count=branch_atom_count,
            ring_core_bond_count=ring_core_bond_count,
            ring_core_atom_count=ring_core_atom_count,
            ring_core_max_degree=ring_core_max_degree,
            pendant_atom_count=pendant_atom_count,
            pendant_bond_count=pendant_bond_count,
            pendant_component_count=pendant_component_count,
            pendant_component_atom_counts=pendant_component_atom_counts,
            pendant_component_boundary_counts=pendant_component_boundary_counts,
            cyclic_ranks=cyclic_ranks,
            cyclic_component_count=len(cyclic_surfaces),
            unsupported_bond_count=unsupported_bond_count,
            ring_core_unsupported_bond_count=ring_core_unsupported_bond_count,
            pendant_unsupported_bond_count=pendant_unsupported_bond_count,
            unsupported_stereo_surface_count=unsupported_stereo_surface_count,
            required_capabilities=frozenset(required_capabilities),
            unsupported_capabilities=frozenset(unsupported_capabilities),
        )

    required_capabilities.add(
        _WriterPublicCyclicRequiredCapability.SIMPLE_CYCLE_CORE_CLOSURE,
    )

    if ring_core_non_single_bond_ids:
        required_capabilities.add(
            _WriterPublicCyclicRequiredCapability.
            RING_CORE_NON_SINGLE_CLOSURE_BOND,
        )
    if ring_core_has_visible_single_closure_bond_text:
        required_capabilities.add(
            (
                _WriterPublicCyclicRequiredCapability
                .RING_CORE_VISIBLE_SINGLE_CLOSURE_BOND_TEXT
            ),
        )
    if ring_core_aromatic_bond_ids:
        required_capabilities.add(
            _WriterPublicCyclicRequiredCapability.RING_CORE_AROMATIC_BOND_TEXT,
        )
    if ring_core_tetra_template_count:
        required_capabilities.add(
            _WriterPublicCyclicRequiredCapability.RING_CORE_TETRAHEDRAL_STEREO,
        )
    if directional_ring_carrier_envelope is not None:
        required_capabilities.update(
            (
                _WriterPublicCyclicRequiredCapability.CYCLIC_DIRECTIONAL_STEREO,
                _WriterPublicCyclicRequiredCapability.CYCLIC_RING_PAIR_STEREO,
            )
        )
    if shared_directional_ring_carrier_envelope is not None:
        required_capabilities.update(
            (
                _WriterPublicCyclicRequiredCapability.CYCLIC_DIRECTIONAL_STEREO,
                _WriterPublicCyclicRequiredCapability.CYCLIC_RING_PAIR_STEREO,
                (
                    _WriterPublicCyclicRequiredCapability
                    .SHARED_DIRECTIONAL_RING_CARRIER_STEREO
                ),
            )
        )
    if (
        cyclic_ranks == (2,)
        and (two_cycle_envelope is not None or fused_diamond_envelope is not None)
    ):
        required_capabilities.add(
            _WriterPublicCyclicRequiredCapability.MULTI_CYCLE_TOPOLOGY,
        )
    if fused_diamond_envelope is not None:
        required_capabilities.add(
            _WriterPublicCyclicRequiredCapability.FUSED_OR_BRIDGED_TOPOLOGY,
        )
    if (
        two_cycle_bond_policy_report is not None
        and two_cycle_bond_policy_report.visible_tree_bonds
    ):
        required_capabilities.add(
            _WriterPublicCyclicRequiredCapability.TREE_BOND_TEXT_EMISSION,
        )

    if (
        cyclic_ranks != (1,)
        and not two_cycle_topology_supported
        and not fused_diamond_topology_supported
    ):
        unsupported_capabilities.add(
            _WriterPublicCyclicRequiredCapability.MULTI_CYCLE_TOPOLOGY,
        )
        return _WriterPublicCyclicOpeningProfileReport(
            kind=_WriterPublicCyclicOpeningProfileKind.BLOCKED_UNSUPPORTED_CYCLIC_RANK,
            component_count=1,
            cyclic_component_count=len(cyclic_surfaces),
            cyclic_ranks=cyclic_ranks,
            ring_core_atom_count=ring_core_atom_count,
            ring_core_bond_count=ring_core_bond_count,
            ring_core_max_degree=ring_core_max_degree,
            pendant_atom_count=pendant_atom_count,
            pendant_bond_count=pendant_bond_count,
            pendant_component_count=pendant_component_count,
            pendant_component_atom_counts=pendant_component_atom_counts,
            pendant_component_boundary_counts=pendant_component_boundary_counts,
            component_atom_count=component_atom_count,
            component_bond_count=component_bond_count,
            max_component_degree=max_component_degree,
            branch_atom_count=branch_atom_count,
            unsupported_bond_count=unsupported_bond_count,
            unsupported_stereo_surface_count=unsupported_stereo_surface_count,
            ring_core_unsupported_bond_count=ring_core_unsupported_bond_count,
            pendant_unsupported_bond_count=pendant_unsupported_bond_count,
            required_capabilities=frozenset(required_capabilities),
            unsupported_capabilities=frozenset(unsupported_capabilities),
        )

    if (
        two_cycle_topology_supported
        and not two_cycle_label_policy_supported
    ) or (
        fused_diamond_topology_supported
        and not fused_diamond_label_policy_supported
    ):
        kind = (
            _WriterPublicCyclicOpeningProfileKind
            .BLOCKED_UNSUPPORTED_RING_LABEL_POLICY
        )
    elif ring_core_unsupported_bond_count:
        fused_unsupported_single_closure_bonds = frozenset(
            bond_id
            for bond_id in (
                (
                    fused_diamond_policy_report
                    .unsupported_closure_bonds
                )
                if fused_diamond_policy_report is not None
                else frozenset()
            )
            if (
                prepared.graph_index.bond_by_id[bond_id].order
                is BondOrder.SINGLE
            )
        )
        fused_unsupported_non_single_bonds = frozenset(
            bond_id
            for bond_id in (
                fused_diamond_policy_report.unsupported_bonds
                if fused_diamond_policy_report is not None
                else frozenset()
            )
            if prepared.graph_index.bond_by_id[bond_id].order
            in {BondOrder.DOUBLE, BondOrder.TRIPLE}
        )
        fused_unsupported_aromatic_bonds = frozenset(
            bond_id
            for bond_id in (
                fused_diamond_policy_report.unsupported_bonds
                if fused_diamond_policy_report is not None
                else frozenset()
            )
            if (
                prepared.graph_index.bond_by_id[bond_id].order
                is BondOrder.AROMATIC
            )
        )
        if (
            fused_diamond_policy_report is not None
            and fused_diamond_policy_report.unsupported_tree_bonds
        ):
            unsupported_capabilities.add(
                _WriterPublicCyclicRequiredCapability.TREE_BOND_TEXT_EMISSION,
            )
        if fused_unsupported_single_closure_bonds:
            unsupported_capabilities.add(
                (
                    _WriterPublicCyclicRequiredCapability
                    .RING_CORE_VISIBLE_SINGLE_CLOSURE_BOND_TEXT
                ),
            )
        if fused_unsupported_non_single_bonds:
            unsupported_capabilities.add(
                (
                    _WriterPublicCyclicRequiredCapability
                    .RING_CORE_NON_SINGLE_CLOSURE_BOND
                ),
            )
        if fused_unsupported_aromatic_bonds:
            unsupported_capabilities.add(
                (
                    _WriterPublicCyclicRequiredCapability
                    .RING_CORE_AROMATIC_BOND_TEXT
                ),
            )
        two_cycle_unsupported_single_closure_bonds = frozenset(
            bond_id
            for bond_id in (
                (
                    two_cycle_bond_policy_report
                    .unsupported_closure_bonds
                )
                if two_cycle_bond_policy_report is not None
                else frozenset()
            )
            if (
                prepared.graph_index.bond_by_id[bond_id].order
                is BondOrder.SINGLE
            )
        )
        if (
            two_cycle_bond_policy_report is not None
            and two_cycle_bond_policy_report.unsupported_tree_bonds
        ):
            unsupported_capabilities.add(
                _WriterPublicCyclicRequiredCapability.TREE_BOND_TEXT_EMISSION,
            )
        if (
            ring_core_unsupported_single_closure_bond_count
            or two_cycle_unsupported_single_closure_bonds
        ):
            unsupported_capabilities.add(
                (
                    _WriterPublicCyclicRequiredCapability
                    .RING_CORE_VISIBLE_SINGLE_CLOSURE_BOND_TEXT
                ),
            )
        if ring_core_aromatic_bond_ids and not ring_core_aromatic_is_supported:
            unsupported_capabilities.add(
                _WriterPublicCyclicRequiredCapability.RING_CORE_AROMATIC_BOND_TEXT,
            )
        if (
            ring_core_non_single_bond_ids
            and not ring_core_has_supported_non_single_closure_bond
        ):
            unsupported_capabilities.add(
                (
                    _WriterPublicCyclicRequiredCapability
                    .RING_CORE_NON_SINGLE_CLOSURE_BOND
                ),
            )
        kind = (
            _WriterPublicCyclicOpeningProfileKind.BLOCKED_UNSUPPORTED_CLOSURE_BOND_SURFACE
        )
    elif pendant_unsupported_bond_count:
        unsupported_capabilities.add(
            _WriterPublicCyclicRequiredCapability.TREE_BOND_TEXT_EMISSION,
        )
        kind = (
            _WriterPublicCyclicOpeningProfileKind.BLOCKED_UNSUPPORTED_CLOSURE_BOND_SURFACE
        )
    elif unsupported_stereo_surface_count:
        unsupported_capabilities.add(
            _WriterPublicCyclicRequiredCapability.CYCLIC_DIRECTIONAL_STEREO,
        )
        if len(prepared.directional_templates) > 1:
            unsupported_capabilities.add(
                (
                    _WriterPublicCyclicRequiredCapability
                    .SHARED_DIRECTIONAL_RING_CARRIER_STEREO
                ),
            )
        kind = (
            _WriterPublicCyclicOpeningProfileKind.BLOCKED_UNSUPPORTED_CYCLIC_STEREO_SURFACE
        )
    elif (
        ring_core_tetra_template_count
        and not ring_core_tetra_is_supported
    ):
        unsupported_capabilities.add(
            _WriterPublicCyclicRequiredCapability.RING_CORE_TETRAHEDRAL_STEREO,
        )
        kind = (
            _WriterPublicCyclicOpeningProfileKind.BLOCKED_UNSUPPORTED_CYCLIC_STEREO_SURFACE
        )
    else:
        if pendant_multi_boundary:
            unsupported_capabilities.add(
                _WriterPublicCyclicRequiredCapability.MULTI_BOUNDARY_PENDANT_COMPONENT,
            )
        if pendant_non_tree:
            unsupported_capabilities.add(
                _WriterPublicCyclicRequiredCapability.NON_FOREST_PENDANT_MATERIAL,
            )

        if two_cycle_backbone_supported and pendant_atom_count == 0:
            kind = (
                _WriterPublicCyclicOpeningProfileKind
                .SUPPORTED_TWO_BRIDGE_SEPARATED_SIMPLE_CYCLES
            )
        elif (
            two_cycle_backbone_supported
            and pendant_atom_count > 0
            and pendant_forest_supported
        ):
            kind = (
                _WriterPublicCyclicOpeningProfileKind
                .SUPPORTED_TWO_BRIDGE_SEPARATED_SIMPLE_CYCLES_WITH_ACYCLIC_ATTACHMENTS
            )
            required_capabilities.update(
                (
                    _WriterPublicCyclicRequiredCapability
                    .ACYCLIC_PENDANT_TREE_TRAVERSAL,
                    _WriterPublicCyclicRequiredCapability
                    .TREE_BOND_TEXT_EMISSION,
                )
            )
        elif fused_diamond_supported:
            kind = (
                _WriterPublicCyclicOpeningProfileKind
                .SUPPORTED_FUSED_RANK_TWO_DIAMOND
            )
        elif ring_core_is_simple_cycle and pendant_atom_count == 0:
            kind = (
                _WriterPublicCyclicOpeningProfileKind
                .SUPPORTED_SIMPLE_MONOCYCLE_COMPONENT
            )
        elif (
            ring_core_is_simple_cycle
            and pendant_atom_count > 0
            and pendant_forest_supported
        ):
            kind = (
                _WriterPublicCyclicOpeningProfileKind
                .SUPPORTED_SIMPLE_MONOCYCLE_WITH_ACYCLIC_ATTACHMENTS
            )
            required_capabilities.update(
                (
                    _WriterPublicCyclicRequiredCapability
                    .ACYCLIC_PENDANT_TREE_TRAVERSAL,
                    _WriterPublicCyclicRequiredCapability
                    .TREE_BOND_TEXT_EMISSION,
                )
            )
        else:
            kind = (
                _WriterPublicCyclicOpeningProfileKind
                .BLOCKED_UNSUPPORTED_BRANCHING
            )
            if not ring_core_is_simple_cycle and not two_cycle_topology_supported:
                unsupported_capabilities.add(
                    _WriterPublicCyclicRequiredCapability.MULTI_CYCLE_TOPOLOGY,
                )

    return _WriterPublicCyclicOpeningProfileReport(
        kind=kind,
        component_count=len(surfaces),
        cyclic_component_count=len(cyclic_surfaces),
        cyclic_ranks=cyclic_ranks,
        ring_core_atom_count=ring_core_atom_count,
        ring_core_bond_count=ring_core_bond_count,
        ring_core_max_degree=ring_core_max_degree,
        pendant_atom_count=pendant_atom_count,
        pendant_bond_count=pendant_bond_count,
        pendant_component_count=pendant_component_count,
        pendant_component_atom_counts=pendant_component_atom_counts,
        pendant_component_boundary_counts=pendant_component_boundary_counts,
        component_atom_count=component_atom_count,
        component_bond_count=component_bond_count,
        max_component_degree=max_component_degree,
        branch_atom_count=branch_atom_count,
        unsupported_bond_count=unsupported_bond_count,
        ring_core_unsupported_bond_count=ring_core_unsupported_bond_count,
        pendant_unsupported_bond_count=pendant_unsupported_bond_count,
        unsupported_stereo_surface_count=unsupported_stereo_surface_count,
        required_capabilities=frozenset(required_capabilities),
        unsupported_capabilities=frozenset(unsupported_capabilities),
    )


def _writer_two_bridge_separated_simple_cycles(
    prepared: SouthStarPreparedMol,
    surface,
    *,
    ring_core_atoms: frozenset[AtomId],
    ring_core_bonds: frozenset[BondId],
) -> _WriterTwoCycleBlockEnvelope | None:
    if not surface.connected or surface.cyclic_rank != 2:
        return None

    block_cut = prepared.writer_graph_metadata.block_cut
    block_by_bond = dict(block_cut.biconnected_block_by_bond)
    block_ids = tuple(sorted(surface.cyclic_block_ids))
    if len(block_ids) != 2:
        return None

    blocks: list[tuple[frozenset[AtomId], frozenset[BondId]]] = []
    for block_id in block_ids:
        bonds = frozenset(
            bond_id
            for bond_id in ring_core_bonds
            if block_by_bond.get(bond_id) == block_id
        )
        atoms: set[AtomId] = set()
        degrees: dict[AtomId, int] = {}

        for bond_id in bonds:
            bond = prepared.graph_index.bond_by_id[bond_id]
            atoms.update((bond.a, bond.b))
            degrees[bond.a] = degrees.get(bond.a, 0) + 1
            degrees[bond.b] = degrees.get(bond.b, 0) + 1

        if (
            len(atoms) < 3
            or len(bonds) != len(atoms)
            or any(degrees[atom] != 2 for atom in atoms)
        ):
            return None

        blocks.append((frozenset(atoms), bonds))

    (left_atoms, left_bonds), (right_atoms, right_bonds) = blocks
    if not left_atoms.isdisjoint(right_atoms):
        return None

    connector_bonds = ring_core_bonds - left_bonds - right_bonds
    if not connector_bonds:
        return None
    if not connector_bonds.issubset(block_cut.bridge_bonds):
        return None

    adjacency: dict[AtomId, list[tuple[BondId, AtomId]]] = {}
    for bond_id in connector_bonds:
        bond = prepared.graph_index.bond_by_id[bond_id]
        adjacency.setdefault(bond.a, []).append((bond_id, bond.b))
        adjacency.setdefault(bond.b, []).append((bond_id, bond.a))

    connector_atoms = frozenset(adjacency)
    left_attachments = connector_atoms & left_atoms
    right_attachments = connector_atoms & right_atoms
    if len(left_attachments) != 1 or len(right_attachments) != 1:
        return None

    left_attachment = next(iter(left_attachments))
    right_attachment = next(iter(right_attachments))
    endpoints = frozenset(
        atom
        for atom, edges in adjacency.items()
        if len(edges) == 1
    )
    if endpoints != frozenset((left_attachment, right_attachment)):
        return None
    if any(
        len(edges) != 2
        for atom, edges in adjacency.items()
        if atom not in endpoints
    ):
        return None
    if len(connector_bonds) != len(connector_atoms) - 1:
        return None

    path_atoms, path_bonds = _ordered_writer_connector_path(
        adjacency,
        start=left_attachment,
        end=right_attachment,
    )
    if frozenset(path_bonds) != connector_bonds:
        return None

    internal_atoms = connector_atoms - left_atoms - right_atoms
    if ring_core_atoms != left_atoms | right_atoms | internal_atoms:
        return None

    return _WriterTwoCycleBlockEnvelope(
        cycle_atom_sets=(left_atoms, right_atoms),
        cycle_bond_sets=(left_bonds, right_bonds),
        connector_atom_path=path_atoms,
        connector_bond_path=path_bonds,
    )


def _writer_fused_rank_two_diamond(
    prepared: SouthStarPreparedMol,
    surface,
    *,
    ring_core_atoms: frozenset[AtomId],
    ring_core_bonds: frozenset[BondId],
) -> _WriterFusedRankTwoDiamondEnvelope | None:
    if not surface.connected or surface.cyclic_rank != 2:
        return None
    if ring_core_atoms != surface.atoms or ring_core_bonds != surface.bonds:
        return None
    if len(ring_core_atoms) != 4 or len(ring_core_bonds) != 5:
        return None

    block_ids = tuple(sorted(surface.cyclic_block_ids))
    if len(block_ids) != 1:
        return None

    block_id = block_ids[0]
    block_cut = prepared.writer_graph_metadata.block_cut
    if ring_core_bonds & block_cut.bridge_bonds:
        return None

    block_by_bond = dict(block_cut.biconnected_block_by_bond)
    if any(block_by_bond.get(bond) != block_id for bond in ring_core_bonds):
        return None

    degrees: dict[AtomId, int] = {atom: 0 for atom in ring_core_atoms}
    shared_bond: BondId | None = None
    for bond_id in ring_core_bonds:
        bond = prepared.graph_index.bond_by_id[bond_id]
        if bond.a not in ring_core_atoms or bond.b not in ring_core_atoms:
            return None
        degrees[bond.a] += 1
        degrees[bond.b] += 1

    if tuple(sorted(degrees.values())) != (2, 2, 3, 3):
        return None

    degree_three_atoms = frozenset(
        atom for atom, degree in degrees.items() if degree == 3
    )
    shared_bonds = tuple(
        bond_id
        for bond_id in ring_core_bonds
        if (
            (bond := prepared.graph_index.bond_by_id[bond_id])
            is not None
            and frozenset((bond.a, bond.b)) == degree_three_atoms
        )
    )
    if len(shared_bonds) != 1:
        return None

    shared_bond = shared_bonds[0]
    return _WriterFusedRankTwoDiamondEnvelope(
        atoms=ring_core_atoms,
        bonds=ring_core_bonds,
        block_id=block_id,
        shared_bond=shared_bond,
    )


def _writer_fused_rank_two_diamond_policy_report(
    prepared: SouthStarPreparedMol,
    envelope: _WriterFusedRankTwoDiamondEnvelope,
) -> _WriterFusedRankTwoDiamondPolicyReport:
    unsupported_tree: set[BondId] = set()

    for bond_id in sorted(envelope.bonds):
        bond = prepared.graph_index.bond_by_id[bond_id]
        if (
            bond.order is not BondOrder.SINGLE
            or not _writer_elided_single_tree_slot_is_supported(
                prepared,
                bond_id,
            )
        ):
            unsupported_tree.add(bond_id)

    if unsupported_tree:
        return _WriterFusedRankTwoDiamondPolicyReport(
            unsupported_tree_bonds=frozenset(unsupported_tree),
            unsupported_closure_bonds=frozenset(),
        )

    for bond_id in sorted(envelope.bonds):
        choices = prepared.policy.bond_text_domain_unchecked(
            bond_id,
            slot_kind="tree",
        )
        if not prepared.semantics.bond_decode_ok(
            prepared.facts,
            bond_id,
            choices[0],
            DirectionMark.ABSENT,
        ):
            unsupported_tree.add(bond_id)

    return _WriterFusedRankTwoDiamondPolicyReport(
        unsupported_tree_bonds=frozenset(unsupported_tree),
        unsupported_closure_bonds=frozenset(),
    )


def _ordered_writer_connector_path(
    adjacency: dict[AtomId, list[tuple[BondId, AtomId]]],
    *,
    start: AtomId,
    end: AtomId,
) -> tuple[tuple[AtomId, ...], tuple[BondId, ...]]:
    atoms: list[AtomId] = [start]
    bonds: list[BondId] = []
    previous: AtomId | None = None
    current = start

    while current != end:
        candidates = tuple(
            (bond_id, atom)
            for bond_id, atom in adjacency.get(current, ())
            if atom != previous
        )
        if len(candidates) != 1:
            return ((), ())

        bond_id, next_atom = candidates[0]
        bonds.append(bond_id)
        atoms.append(next_atom)
        previous = current
        current = next_atom

    return (tuple(atoms), tuple(bonds))


def _writer_public_cyclic_bond_roles(
    *,
    ring_core_bond_ids: tuple[BondId, ...],
    two_cycle_envelope: _WriterTwoCycleBlockEnvelope | None,
) -> _WriterCyclicBondRoles:
    if two_cycle_envelope is None:
        return _WriterCyclicBondRoles(
            closure_candidate_bonds=frozenset(ring_core_bond_ids),
            tree_only_bonds=frozenset(),
        )

    closure_bonds = (
        two_cycle_envelope.cycle_bond_sets[0]
        | two_cycle_envelope.cycle_bond_sets[1]
    )
    tree_only = two_cycle_envelope.connector_bonds
    if closure_bonds & tree_only:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "two-cycle closure and connector bond roles overlap",
        )

    return _WriterCyclicBondRoles(
        closure_candidate_bonds=closure_bonds,
        tree_only_bonds=tree_only,
    )


def _writer_two_cycle_bond_policy_report(
    prepared: SouthStarPreparedMol,
    roles: _WriterCyclicBondRoles,
    envelope: _WriterTwoCycleBlockEnvelope,
) -> _WriterTwoCycleBondPolicyReport:
    unsupported_tree: set[BondId] = set()
    unsupported_closure: set[BondId] = set()
    non_single_backbone: set[BondId] = set()
    visible_tree: set[BondId] = set()
    visible_closure: set[BondId] = set()

    tree_bonds = roles.closure_candidate_bonds | roles.tree_only_bonds
    non_single_backbone = frozenset(
        bond_id
        for bond_id in tree_bonds
        if (
            prepared.graph_index.bond_by_id[bond_id].order
            is not BondOrder.SINGLE
        )
    )
    shape_supported = _writer_two_cycle_non_single_shape_is_supported(
        non_single_backbone=non_single_backbone,
        roles=roles,
        envelope=envelope,
    )

    for bond_id in sorted(tree_bonds):
        bond = prepared.graph_index.bond_by_id[bond_id]
        if bond.order is BondOrder.SINGLE:
            if not _writer_elided_single_tree_slot_is_supported(
                prepared,
                bond_id,
            ):
                unsupported_tree.add(bond_id)
            continue

        if not shape_supported:
            unsupported_tree.add(bond_id)
            continue

        if (
            _writer_exact_non_single_tree_marker(
                prepared,
                bond_id,
            )
            is not None
        ):
            visible_tree.add(bond_id)
        else:
            unsupported_tree.add(bond_id)

    for bond_id in sorted(roles.closure_candidate_bonds):
        bond = prepared.graph_index.bond_by_id[bond_id]
        if bond.order is BondOrder.SINGLE:
            relation = _writer_public_single_closure_relation(
                prepared,
                bond_id,
            )
            if (
                relation is None
                or relation.texts != ("",)
                or relation.compatible_pairs != (("", ""),)
            ):
                unsupported_closure.add(bond_id)
            continue

        if not shape_supported:
            unsupported_closure.add(bond_id)
            continue

        relation = _writer_public_non_single_closure_relation(
            prepared,
            bond_id,
        )
        if relation is None:
            unsupported_closure.add(bond_id)
        else:
            visible_closure.add(bond_id)

    return _WriterTwoCycleBondPolicyReport(
        unsupported_tree_bonds=frozenset(unsupported_tree),
        unsupported_closure_bonds=frozenset(unsupported_closure),
        visible_tree_bonds=frozenset(visible_tree),
        visible_closure_bonds=frozenset(visible_closure),
    )


def _writer_two_cycle_non_single_shape_is_supported(
    *,
    non_single_backbone: frozenset[BondId],
    roles: _WriterCyclicBondRoles,
    envelope: _WriterTwoCycleBlockEnvelope,
) -> bool:
    connector_non_single = non_single_backbone & roles.tree_only_bonds
    cycle_non_single = non_single_backbone & roles.closure_candidate_bonds

    if len(connector_non_single) > 1:
        return False

    if any(
        len(cycle_non_single & block_bonds) > 1
        for block_bonds in envelope.cycle_bond_sets
    ):
        return False

    return len(non_single_backbone) <= 2


def _writer_elided_single_tree_slot_is_supported(
    prepared: SouthStarPreparedMol,
    bond_id: BondId,
) -> bool:
    try:
        choices = prepared.policy.bond_text_domain_unchecked(
            bond_id,
            slot_kind="tree",
        )
    except KeyError:
        return False

    return len(choices) == 1 and choices[0].base_text == ""


def _writer_raw_double_tree_marker_slot_is_supported(
    prepared: SouthStarPreparedMol,
    bond_id: BondId,
) -> bool:
    bond = prepared.graph_index.bond_by_id.get(bond_id)
    if bond is None or bond.order is not BondOrder.DOUBLE:
        return False
    try:
        choices = prepared.policy.bond_text_domain_unchecked(
            bond_id,
            slot_kind="tree",
        )
    except KeyError:
        return False
    return (
        len(choices) == 1
        and choices[0].base_text == "="
        and not choices[0].permits_direction
    )


def _writer_raw_double_joint_ring_endpoint_slot_is_supported(
    prepared: SouthStarPreparedMol,
    bond_id: BondId,
) -> bool:
    bond = prepared.graph_index.bond_by_id.get(bond_id)
    if bond is None or bond.order is not BondOrder.DOUBLE:
        return False
    try:
        choices = prepared.policy.bond_text_domain_unchecked(
            bond_id,
            slot_kind="ring_endpoint",
        )
    except KeyError:
        return False
    texts = tuple(choice.base_text for choice in choices)
    return (
        len(choices) == 2
        and not any(choice.permits_direction for choice in choices)
        and len(frozenset(texts)) == 2
        and frozenset(texts) == frozenset(("", "="))
    )


def _writer_raw_elided_single_ring_endpoint_slot_is_supported(
    prepared: SouthStarPreparedMol,
    bond_id: BondId,
) -> bool:
    bond = prepared.graph_index.bond_by_id.get(bond_id)
    if bond is None or bond.order is not BondOrder.SINGLE:
        return False
    try:
        choices = prepared.policy.bond_text_domain_unchecked(
            bond_id,
            slot_kind="ring_endpoint",
        )
    except KeyError:
        return False
    return len(choices) == 1 and choices[0].base_text == ""


def _writer_exact_non_single_tree_marker(
    prepared: SouthStarPreparedMol,
    bond_id: BondId,
) -> str | None:
    bond = prepared.graph_index.bond_by_id[bond_id]
    marker = {
        BondOrder.DOUBLE: "=",
        BondOrder.TRIPLE: "#",
    }.get(bond.order)
    if marker is None:
        return None

    try:
        choices = prepared.policy.bond_text_domain_unchecked(
            bond_id,
            slot_kind="tree",
        )
    except KeyError:
        return None

    if len(choices) != 1:
        return None

    choice = choices[0]
    if choice.base_text != marker or choice.permits_direction:
        return None

    if not prepared.semantics.bond_decode_ok(
        prepared.facts,
        bond_id,
        choice,
        DirectionMark.ABSENT,
    ):
        return None

    return marker


def _writer_public_non_single_closure_bond_is_supported(
    prepared: SouthStarPreparedMol,
    bond_id: BondId,
) -> bool:
    return (
        _writer_public_non_single_closure_relation(
            prepared,
            bond_id,
        )
        is not None
    )


def _writer_public_closure_bond_text_relation_from_choices(
    prepared: SouthStarPreparedMol,
    bond_id: BondId,
    choices,
    *,
    max_choice_count: int,
) -> WriterClosureBondTextRelation | None:
    if len(choices) > max_choice_count:
        return None

    rows = tuple(
        (
            first.base_text,
            tuple(
                second.base_text
                for second in choices
                if prepared.semantics.ring_pair_decode_ok(
                    prepared.facts,
                    bond_id,
                    first,
                    DirectionMark.ABSENT,
                    second,
                    DirectionMark.ABSENT,
                )
            ),
        )
        for first in choices
    )
    return WriterClosureBondTextRelation(rows=rows)


def _writer_public_non_single_closure_relation(
    prepared: SouthStarPreparedMol,
    bond_id: BondId,
):
    order = prepared.graph_index.bond_by_id[bond_id].order
    marker = {
        BondOrder.DOUBLE: "=",
        BondOrder.TRIPLE: "#",
    }.get(order)
    if marker is None:
        return None

    try:
        raw_choices = prepared.policy.bond_text_domain_unchecked(
            bond_id,
            slot_kind="ring_endpoint",
        )
    except KeyError:
        return None

    if len(raw_choices) != 2:
        return None
    if any(choice.permits_direction for choice in raw_choices):
        return None

    raw_texts = tuple(choice.base_text for choice in raw_choices)
    if (
        len(set(raw_texts)) != 2
        or frozenset(raw_texts) != frozenset(("", marker))
    ):
        return None

    relation = _writer_public_closure_bond_text_relation_from_choices(
        prepared,
        bond_id,
        raw_choices,
        max_choice_count=2,
    )
    if relation is None:
        return None

    expected_pairs = tuple(
        (first_text, second_text)
        for first_text in raw_texts
        for second_text in raw_texts
        if first_text != second_text
    )
    if (
        relation.texts != raw_texts
        or relation.compatible_pairs != expected_pairs
    ):
        return None

    return relation


def _writer_public_single_closure_relation(
    prepared: SouthStarPreparedMol,
    bond_id: BondId,
):
    if prepared.graph_index.bond_by_id[bond_id].order is not BondOrder.SINGLE:
        return None

    try:
        raw_choices = prepared.policy.bond_text_domain_unchecked(
            bond_id,
            slot_kind="ring_endpoint",
        )
    except KeyError:
        return None

    if not 1 <= len(raw_choices) <= 2:
        return None

    raw_texts = tuple(choice.base_text for choice in raw_choices)
    if any(text not in {"", "-"} for text in raw_texts):
        return None

    relation = _writer_public_closure_bond_text_relation_from_choices(
        prepared,
        bond_id,
        raw_choices,
        max_choice_count=2,
    )
    if relation is None:
        return None

    if relation.texts != raw_texts:
        return None

    expected_pairs = tuple(
        (first_text, second_text)
        for first_text in relation.texts
        for second_text in relation.texts
    )
    if relation.compatible_pairs != expected_pairs:
        return None

    return relation


def _bounded_aromatic_texts(choices) -> tuple[str, ...] | None:
    if not 1 <= len(choices) <= 2:
        return None
    if any(choice.permits_direction for choice in choices):
        return None
    texts = tuple(choice.base_text for choice in choices)
    if len(set(texts)) != len(texts):
        return None
    if any(text not in {"", ":"} for text in texts):
        return None
    return texts


def _writer_public_aromatic_ring_bond_is_supported(
    prepared: SouthStarPreparedMol,
    bond_id: BondId,
) -> bool:
    bond = prepared.graph_index.bond_by_id[bond_id]
    if bond.order is not BondOrder.AROMATIC:
        return False

    try:
        tree_choices = prepared.policy.bond_text_domain_unchecked(
            bond_id,
            slot_kind="tree",
        )
        ring_choices = prepared.policy.bond_text_domain_unchecked(
            bond_id,
            slot_kind="ring_endpoint",
        )
    except KeyError:
        return False

    tree_texts = _bounded_aromatic_texts(tree_choices)
    ring_texts = _bounded_aromatic_texts(ring_choices)
    if tree_texts is None or ring_texts is None:
        return False

    if any(
        not prepared.semantics.bond_decode_ok(
            prepared.facts,
            bond_id,
            choice,
            DirectionMark.ABSENT,
        )
        for choice in tree_choices
    ):
        return False

    relation = _writer_public_closure_bond_text_relation_from_choices(
        prepared,
        bond_id,
        ring_choices,
        max_choice_count=2,
    )
    if relation is None:
        return False

    expected_pairs = tuple(
        (first_text, second_text)
        for first_text in ring_texts
        for second_text in ring_texts
    )
    return (
        relation.texts == ring_texts
        and relation.compatible_pairs == expected_pairs
    )


def _writer_public_directional_ring_carrier_envelope(
    *,
    prepared: SouthStarPreparedMol,
    ring_core_atom_ids: frozenset[AtomId],
    ring_core_bond_ids: frozenset[BondId],
    pendant_atom_ids: frozenset[AtomId],
    pendant_bond_ids: frozenset[BondId],
    ring_core_non_single_bond_ids: tuple[BondId, ...],
    non_single_closure_supported: bool,
) -> _WriterDirectionalRingCarrierEnvelope | None:
    if len(prepared.directional_templates) != 1:
        return None
    if prepared.tetra_templates:
        return None

    template = prepared.directional_templates[0]
    if template.status is not SiteStatus.SPECIFIED:
        return None
    if template.center_bond not in ring_core_bond_ids:
        return None

    center = prepared.graph_index.bond_by_id.get(template.center_bond)
    if center is None or center.order is not BondOrder.DOUBLE:
        return None
    if {center.a, center.b} != {template.left_endpoint, template.right_endpoint}:
        return None
    if center.a not in ring_core_atom_ids or center.b not in ring_core_atom_ids:
        return None
    if (
        ring_core_non_single_bond_ids != (template.center_bond,)
        or not non_single_closure_supported
    ):
        return None

    left = _writer_public_directional_side_carriers(
        prepared=prepared,
        endpoint=template.left_endpoint,
        ligand_ids=template.left_ligands,
        ring_core_bond_ids=ring_core_bond_ids,
        pendant_atom_ids=pendant_atom_ids,
    )
    right = _writer_public_directional_side_carriers(
        prepared=prepared,
        endpoint=template.right_endpoint,
        ligand_ids=template.right_ligands,
        ring_core_bond_ids=ring_core_bond_ids,
        pendant_atom_ids=pendant_atom_ids,
    )
    if left is None or right is None:
        return None

    left_ring, left_pendant = left
    right_ring, right_pendant = right
    left_ring_atom = _other_bond_atom(
        prepared,
        left_ring,
        template.left_endpoint,
    )
    right_ring_atom = _other_bond_atom(
        prepared,
        right_ring,
        template.right_endpoint,
    )
    left_pendant_atom = _other_bond_atom(
        prepared,
        left_pendant,
        template.left_endpoint,
    )
    right_pendant_atom = _other_bond_atom(
        prepared,
        right_pendant,
        template.right_endpoint,
    )
    resolved_atoms = (
        left_ring_atom,
        right_ring_atom,
        left_pendant_atom,
        right_pendant_atom,
    )
    if any(atom is None for atom in resolved_atoms):
        return None
    assert left_ring_atom is not None
    assert right_ring_atom is not None
    assert left_pendant_atom is not None
    assert right_pendant_atom is not None

    expected_ring_atoms = frozenset((
        template.left_endpoint,
        template.right_endpoint,
        left_ring_atom,
        right_ring_atom,
    ))
    expected_pendant_atoms = frozenset((
        left_pendant_atom,
        right_pendant_atom,
    ))
    if (
        len(expected_ring_atoms) != 4
        or ring_core_atom_ids != expected_ring_atoms
        or pendant_atom_ids != expected_pendant_atoms
    ):
        return None

    known_ring_bonds = frozenset((
        template.center_bond,
        left_ring,
        right_ring,
    ))
    remaining_ring_bonds = ring_core_bond_ids - known_ring_bonds
    if len(remaining_ring_bonds) != 1:
        return None

    noncarrier_ring_bond = next(iter(remaining_ring_bonds))
    noncarrier = prepared.graph_index.bond_by_id.get(noncarrier_ring_bond)
    if (
        noncarrier is None
        or noncarrier.order is not BondOrder.SINGLE
        or frozenset((noncarrier.a, noncarrier.b))
        != frozenset((left_ring_atom, right_ring_atom))
    ):
        return None
    if frozenset(pendant_bond_ids) != frozenset((left_pendant, right_pendant)):
        return None

    all_carriers = (left_ring, right_ring, left_pendant, right_pendant)
    if len(frozenset(all_carriers)) != 4:
        return None

    for bond_id in all_carriers:
        bond = prepared.graph_index.bond_by_id.get(bond_id)
        if bond is None or bond.order is not BondOrder.SINGLE:
            return None
        if not _writer_public_directional_single_tree_slot_is_supported(
            prepared,
            bond_id,
        ):
            return None
    if _writer_exact_non_single_tree_marker(
        prepared,
        template.center_bond,
    ) != "=":
        return None

    noncarrier_relation = _writer_public_single_closure_relation(
        prepared,
        noncarrier_ring_bond,
    )
    if (
        not _writer_elided_single_tree_slot_is_supported(
            prepared,
            noncarrier_ring_bond,
        )
        or noncarrier_relation is None
        or noncarrier_relation.texts != ("",)
        or noncarrier_relation.compatible_pairs != (("", ""),)
    ):
        return None

    for bond_id, first_atom, second_atom in (
        (
            left_ring,
            template.left_endpoint,
            _other_bond_atom(
                prepared,
                left_ring,
                template.left_endpoint,
            ),
        ),
        (
            right_ring,
            template.right_endpoint,
            _other_bond_atom(
                prepared,
                right_ring,
                template.right_endpoint,
            ),
        ),
    ):
        if second_atom is None:
            return None
        if (
            _writer_public_directional_ring_endpoint_relation(
                prepared,
                bond_id,
                first_atom=first_atom,
                second_atom=second_atom,
            )
            is None
        ):
            return None
        if (
            _writer_public_directional_ring_endpoint_relation(
                prepared,
                bond_id,
                first_atom=second_atom,
                second_atom=first_atom,
            )
            is None
        ):
            return None

    return _WriterDirectionalRingCarrierEnvelope(
        site=template.site,
        center_bond=template.center_bond,
        ring_carrier_bonds=(left_ring, right_ring),
        noncarrier_ring_bond=noncarrier_ring_bond,
        pendant_carrier_bonds=(left_pendant, right_pendant),
    )


def _writer_public_shared_directional_ring_carrier_envelope(
    *,
    prepared: SouthStarPreparedMol,
    ring_core_atom_ids: frozenset[AtomId],
    ring_core_bond_ids: frozenset[BondId],
    pendant_atom_ids: frozenset[AtomId],
    pendant_bond_ids: frozenset[BondId],
    ring_core_non_single_bond_ids: tuple[BondId, ...],
) -> _WriterSharedDirectionalRingCarrierEnvelope | None:
    shape = _writer_public_shared_directional_ring_carrier_shape(
        prepared=prepared,
        ring_core_atom_ids=ring_core_atom_ids,
        ring_core_bond_ids=ring_core_bond_ids,
        pendant_atom_ids=pendant_atom_ids,
        pendant_bond_ids=pendant_bond_ids,
        ring_core_non_single_bond_ids=ring_core_non_single_bond_ids,
    )
    if shape is None:
        return None
    if not _writer_public_shared_directional_raw_policy_is_supported(
        prepared,
        shape,
    ):
        return None
    return _writer_public_shared_directional_semantics(prepared, shape)


def _writer_public_shared_directional_ring_carrier_shape(
    *,
    prepared: SouthStarPreparedMol,
    ring_core_atom_ids: frozenset[AtomId],
    ring_core_bond_ids: frozenset[BondId],
    pendant_atom_ids: frozenset[AtomId],
    pendant_bond_ids: frozenset[BondId],
    ring_core_non_single_bond_ids: tuple[BondId, ...],
) -> _WriterSharedDirectionalRingCarrierShape | None:
    if len(prepared.directional_templates) != 2:
        return None
    if prepared.tetra_templates:
        return None

    templates = tuple(
        sorted(prepared.directional_templates, key=lambda item: int(item.site))
    )
    if any(template.status is not SiteStatus.SPECIFIED for template in templates):
        return None

    center_bonds = tuple(template.center_bond for template in templates)
    if len(frozenset(center_bonds)) != 2:
        return None
    if frozenset(ring_core_non_single_bond_ids) != frozenset(center_bonds):
        return None

    center_endpoint_sets: list[frozenset[AtomId]] = []
    side_data: list[tuple[
        DirectionalTemplate,
        tuple[BondId, BondId],
        tuple[BondId, BondId],
    ]] = []
    for template in templates:
        center = prepared.graph_index.bond_by_id.get(template.center_bond)
        if center is None or center.order is not BondOrder.DOUBLE:
            return None
        endpoint_set = frozenset((template.left_endpoint, template.right_endpoint))
        if frozenset((center.a, center.b)) != endpoint_set:
            return None
        if not endpoint_set <= ring_core_atom_ids:
            return None

        left = _writer_public_directional_side_carriers(
            prepared=prepared,
            endpoint=template.left_endpoint,
            ligand_ids=template.left_ligands,
            ring_core_bond_ids=ring_core_bond_ids,
            pendant_atom_ids=pendant_atom_ids,
        )
        right = _writer_public_directional_side_carriers(
            prepared=prepared,
            endpoint=template.right_endpoint,
            ligand_ids=template.right_ligands,
            ring_core_bond_ids=ring_core_bond_ids,
            pendant_atom_ids=pendant_atom_ids,
        )
        if left is None or right is None:
            return None
        side_data.append((template, (left[0], right[0]), (left[1], right[1])))
        center_endpoint_sets.append(endpoint_set)

    if center_endpoint_sets[0] & center_endpoint_sets[1]:
        return None

    ring_carrier_sets = tuple(frozenset(item[1]) for item in side_data)
    shared = ring_carrier_sets[0] & ring_carrier_sets[1]
    if len(shared) != 1:
        return None
    shared_ring_carrier = next(iter(shared))
    outer_ring_carriers = tuple(
        sorted(
            (ring_carrier_sets[0] | ring_carrier_sets[1]) - shared,
            key=int,
        )
    )
    if len(outer_ring_carriers) != 2:
        return None

    pendant_carriers = tuple(
        bond_id
        for _template, _ring_carriers, pendants in side_data
        for bond_id in pendants
    )
    if (
        len(frozenset(pendant_carriers)) != 4
        or frozenset(pendant_bond_ids) != frozenset(pendant_carriers)
    ):
        return None

    ring_carriers = frozenset((shared_ring_carrier, *outer_ring_carriers))
    known_ring_bonds = frozenset(center_bonds) | ring_carriers
    remaining_ring_bonds = ring_core_bond_ids - known_ring_bonds
    if len(remaining_ring_bonds) != 1:
        return None
    noncarrier_ring_bond = next(iter(remaining_ring_bonds))

    if any(
        (bond := prepared.graph_index.bond_by_id.get(bond_id)) is None
        or bond.order is not BondOrder.SINGLE
        for bond_id in (
            shared_ring_carrier,
            *outer_ring_carriers,
            noncarrier_ring_bond,
            *pendant_carriers,
        )
    ):
        return None

    ring_atoms: set[AtomId] = set()
    for bond_id in ring_core_bond_ids:
        bond = prepared.graph_index.bond_by_id.get(bond_id)
        if bond is None:
            return None
        ring_atoms.update((bond.a, bond.b))
    if frozenset(ring_atoms) != ring_core_atom_ids:
        return None

    pendant_atoms = tuple(
        _other_bond_atom(
            prepared,
            bond_id,
            endpoint,
        )
        for template, _ring_carriers, pendants in side_data
        for bond_id, endpoint in zip(
            pendants,
            (template.left_endpoint, template.right_endpoint),
        )
    )
    if any(atom is None for atom in pendant_atoms):
        return None
    if frozenset(pendant_atoms) != pendant_atom_ids:
        return None

    return _WriterSharedDirectionalRingCarrierShape(
        templates=templates,
        center_bonds=center_bonds,
        shared_ring_carrier=shared_ring_carrier,
        outer_ring_carriers=outer_ring_carriers,
        noncarrier_ring_bond=noncarrier_ring_bond,
        pendant_carriers=pendant_carriers,
    )


def _writer_public_shared_directional_raw_policy_is_supported(
    prepared: SouthStarPreparedMol,
    shape: _WriterSharedDirectionalRingCarrierShape,
) -> bool:
    for center_bond in shape.center_bonds:
        if not _writer_raw_double_tree_marker_slot_is_supported(
            prepared,
            center_bond,
        ):
            return False
        if not _writer_raw_double_joint_ring_endpoint_slot_is_supported(
            prepared,
            center_bond,
        ):
            return False

    for bond_id in shape.ring_carriers:
        if not _writer_public_directional_single_tree_slot_is_supported(
            prepared,
            bond_id,
        ):
            return False
        if not _writer_raw_directional_ring_endpoint_slot_is_supported(
            prepared,
            bond_id,
        ):
            return False

    if (
        not _writer_elided_single_tree_slot_is_supported(
            prepared,
            shape.noncarrier_ring_bond,
        )
        or not _writer_raw_elided_single_ring_endpoint_slot_is_supported(
            prepared,
            shape.noncarrier_ring_bond,
        )
    ):
        return False

    return all(
        _writer_public_directional_single_tree_slot_is_supported(
            prepared,
            bond_id,
        )
        for bond_id in shape.pendant_carriers
    )


def _writer_public_shared_directional_semantics(
    prepared: SouthStarPreparedMol,
    shape: _WriterSharedDirectionalRingCarrierShape,
) -> _WriterSharedDirectionalRingCarrierEnvelope | None:
    center_relations = tuple(
        _writer_public_non_single_closure_relation(prepared, center_bond)
        for center_bond in shape.center_bonds
    )
    if any(relation is None for relation in center_relations):
        return None

    noncarrier_relation = _writer_public_single_closure_relation(
        prepared,
        shape.noncarrier_ring_bond,
    )
    if (
        noncarrier_relation is None
        or noncarrier_relation.texts != ("",)
        or noncarrier_relation.compatible_pairs != (("", ""),)
    ):
        return None

    for bond_id in shape.ring_carriers:
        bond = prepared.graph_index.bond_by_id[bond_id]
        for first_atom, second_atom in (
            (bond.a, bond.b),
            (bond.b, bond.a),
        ):
            if (
                _writer_public_directional_ring_endpoint_relation(
                    prepared,
                    bond_id,
                    first_atom=first_atom,
                    second_atom=second_atom,
                )
                is None
            ):
                return None

    return _WriterSharedDirectionalRingCarrierEnvelope(
        sites=tuple(template.site for template in shape.templates),
        center_bonds=shape.center_bonds,
        shared_ring_carrier=shape.shared_ring_carrier,
        outer_ring_carriers=shape.outer_ring_carriers,
        noncarrier_ring_bond=shape.noncarrier_ring_bond,
        pendant_carriers=shape.pendant_carriers,
    )


def _writer_public_directional_side_carriers(
    *,
    prepared: SouthStarPreparedMol,
    endpoint: AtomId,
    ligand_ids: tuple[OccurrenceId, ...],
    ring_core_bond_ids: frozenset[BondId],
    pendant_atom_ids: frozenset[AtomId],
) -> tuple[BondId, BondId] | None:
    occurrence_by_id = {
        occurrence.id: occurrence
        for occurrence in prepared.facts.ligand_occurrences
    }
    if len(ligand_ids) != 2:
        return None

    ring: list[BondId] = []
    pendant: list[BondId] = []
    for occurrence_id in ligand_ids:
        occurrence = occurrence_by_id.get(occurrence_id)
        if occurrence is None or occurrence.kind is not LigandKind.NEIGHBOR_ATOM:
            return None
        if occurrence.atom is None or occurrence.bond is None:
            return None
        bond = prepared.graph_index.bond_by_id.get(occurrence.bond)
        if bond is None or endpoint not in {bond.a, bond.b}:
            return None
        if occurrence.atom != _other_bond_atom(prepared, occurrence.bond, endpoint):
            return None
        if occurrence.bond in ring_core_bond_ids:
            ring.append(occurrence.bond)
        elif occurrence.atom in pendant_atom_ids:
            pendant.append(occurrence.bond)
        else:
            return None

    if len(ring) != 1 or len(pendant) != 1:
        return None
    return (ring[0], pendant[0])


def _other_bond_atom(
    prepared: SouthStarPreparedMol,
    bond_id: BondId,
    atom_id: AtomId,
) -> AtomId | None:
    bond = prepared.graph_index.bond_by_id.get(bond_id)
    if bond is None:
        return None
    if bond.a == atom_id:
        return bond.b
    if bond.b == atom_id:
        return bond.a
    return None


def _writer_public_directional_single_tree_slot_is_supported(
    prepared: SouthStarPreparedMol,
    bond_id: BondId,
) -> bool:
    try:
        choices = prepared.policy.bond_text_domain_unchecked(
            bond_id,
            slot_kind="tree",
        )
    except KeyError:
        return False
    return (
        len(choices) == 1
        and choices[0].base_text == ""
        and choices[0].permits_direction
    )


def _writer_raw_directional_ring_endpoint_slot_is_supported(
    prepared: SouthStarPreparedMol,
    bond_id: BondId,
) -> bool:
    bond = prepared.graph_index.bond_by_id.get(bond_id)
    if bond is None or bond.order is not BondOrder.SINGLE:
        return False
    try:
        choices = prepared.policy.bond_text_domain_unchecked(
            bond_id,
            slot_kind="ring_endpoint",
        )
    except KeyError:
        return False
    return (
        len(choices) == 1
        and choices[0].base_text == ""
        and choices[0].permits_direction
    )


def _writer_public_directional_ring_endpoint_relation(
    prepared: SouthStarPreparedMol,
    bond_id: BondId,
    *,
    first_atom: AtomId,
    second_atom: AtomId,
):
    try:
        raw = prepared.policy.bond_text_domain_unchecked(
            bond_id,
            slot_kind="ring_endpoint",
        )
    except KeyError:
        return None

    if (
        len(raw) != 1
        or raw[0].base_text != ""
        or not raw[0].permits_direction
    ):
        return None

    try:
        relation = writer_closure_endpoint_relation(
            prepared,
            bond=bond_id,
            first_atom=first_atom,
            second_atom=second_atom,
        )
    except SouthStarError:
        return None

    absent = WriterClosureEndpointChoice("", DirectionMark.ABSENT)
    fwd = WriterClosureEndpointChoice("", DirectionMark.FWD)
    rev = WriterClosureEndpointChoice("", DirectionMark.REV)
    expected_rows = (
        (absent, (absent, fwd, rev)),
        (fwd, (absent, rev)),
        (rev, (absent, fwd)),
    )
    return relation if relation.rows == expected_rows else None


def _writer_public_ring_core_tetrahedral_stereo_is_supported(
    *,
    prepared: SouthStarPreparedMol,
    ring_core_atom_set: frozenset[AtomId],
    ring_core_bond_id_set: frozenset[BondId],
    ring_core_non_single_bond_ids: tuple[BondId, ...],
) -> bool:
    if prepared.directional_templates:
        return False
    if len(ring_core_non_single_bond_ids) > 1:
        return False
    if len(prepared.tetra_templates) != 1:
        return False

    template = prepared.tetra_templates[0]
    if template.center not in ring_core_atom_set:
        return False

    return _writer_public_ring_tetra_template_is_supported(
        prepared=prepared,
        template=template,
        cycle_bond_ids=ring_core_bond_id_set,
        forbidden_neighbor_bond_ids=frozenset(ring_core_non_single_bond_ids),
    )


def _writer_public_two_cycle_ring_tetra_is_supported(
    *,
    prepared: SouthStarPreparedMol,
    envelope: _WriterTwoCycleBlockEnvelope,
    ring_core_bond_ids: frozenset[BondId],
    component_bond_ids: tuple[BondId, ...],
    bond_policy_report: _WriterTwoCycleBondPolicyReport,
) -> bool:
    templates = prepared.tetra_templates
    if not 1 <= len(templates) <= 2:
        return False
    if prepared.directional_templates:
        return False
    if (
        bond_policy_report.visible_tree_bonds
        or bond_policy_report.visible_closure_bonds
    ):
        return False

    if any(
        prepared.graph_index.bond_by_id[bond_id].order is not BondOrder.SINGLE
        or not _writer_elided_single_tree_slot_is_supported(prepared, bond_id)
        for bond_id in component_bond_ids
    ):
        return False

    used_blocks: set[int] = set()
    for template in templates:
        matching_blocks = tuple(
            index
            for index, atoms in enumerate(envelope.cycle_atom_sets)
            if template.center in atoms
        )
        if len(matching_blocks) != 1:
            return False

        block = matching_blocks[0]
        if block in used_blocks:
            return False

        if not _writer_public_ring_tetra_template_is_supported(
            prepared=prepared,
            template=template,
            cycle_bond_ids=envelope.cycle_bond_sets[block],
            forbidden_neighbor_bond_ids=ring_core_bond_ids
            - envelope.cycle_bond_sets[block],
        ):
            return False

        used_blocks.add(block)

    return True


def _writer_public_ring_tetra_template_is_supported(
    *,
    prepared: SouthStarPreparedMol,
    template,
    cycle_bond_ids: frozenset[BondId],
    forbidden_neighbor_bond_ids: frozenset[BondId],
) -> bool:
    if template.status is not SiteStatus.SPECIFIED:
        return False

    occurrence_by_id = {
        occurrence.id: occurrence
        for occurrence in prepared.facts.ligand_occurrences
    }
    try:
        occurrences = tuple(
            occurrence_by_id[occurrence_id]
            for occurrence_id in template.ligand_occurrences
        )
    except KeyError:
        return False

    neighbor_occurrences = tuple(
        occurrence
        for occurrence in occurrences
        if occurrence.kind is LigandKind.NEIGHBOR_ATOM
    )
    implicit_h_occurrences = tuple(
        occurrence
        for occurrence in occurrences
        if occurrence.kind is LigandKind.IMPLICIT_H
    )
    if len(neighbor_occurrences) != 3 or len(implicit_h_occurrences) != 1:
        return False
    if any(occurrence.atom is None for occurrence in neighbor_occurrences):
        return False
    if any(occurrence.bond is None for occurrence in neighbor_occurrences):
        return False

    ring_neighbor_bonds = frozenset(
        occurrence.bond
        for occurrence in neighbor_occurrences
        if occurrence.bond in cycle_bond_ids
    )
    if len(ring_neighbor_bonds) != 2:
        return False

    neighbor_bonds = frozenset(
        occurrence.bond for occurrence in neighbor_occurrences
    )
    if neighbor_bonds & forbidden_neighbor_bond_ids:
        return False

    return len(neighbor_bonds - cycle_bond_ids) == 1


def _cyclic_writer_admission_decision_from_readiness_gate(
    gate: _WriterResidualCyclicReadinessGate,
    *,
    prepared: SouthStarPreparedMol,
) -> _WriterCyclicAdmissionDecision:
    if gate.ready:
        profile = _writer_public_cyclic_opening_profile_report(
            prepared=prepared,
        )
        if _PUBLIC_CYCLIC_WRITER_SHAPED_ENABLED and profile.supported:
            certificate = _writer_public_execution_capability_certificate(
                gate.audit,
            )
            if not certificate.ready:
                return _WriterCyclicAdmissionDecision(
                    kind=(
                        _WriterCyclicAdmissionDecisionKind
                        .BLOCKED_PUBLIC_EXECUTION_CAPABILITY
                    ),
                    readiness_gate=gate,
                    public_profile=profile,
                    execution_capability_certificate=certificate,
                )
            return _WriterCyclicAdmissionDecision(
                kind=_WriterCyclicAdmissionDecisionKind.READY_PUBLIC,
                readiness_gate=gate,
                public_profile=profile,
                execution_capability_certificate=certificate,
            )
        if _PUBLIC_CYCLIC_WRITER_SHAPED_ENABLED:
            return _WriterCyclicAdmissionDecision(
                kind=(
                    _WriterCyclicAdmissionDecisionKind
                    .BLOCKED_PUBLIC_CYCLIC_PROFILE
                ),
                readiness_gate=gate,
                public_profile=profile,
            )
        return _WriterCyclicAdmissionDecision(
            kind=(
                _WriterCyclicAdmissionDecisionKind
                .READY_BUT_PUBLIC_CLOSED
            ),
            readiness_gate=gate,
            public_profile=profile,
        )

    if gate.blocked:
        profile = _writer_public_profile_from_live_graph_policy_blocker(
            prepared,
            gate,
        )
        if profile is not None:
            return _WriterCyclicAdmissionDecision(
                kind=(
                    _WriterCyclicAdmissionDecisionKind
                    .BLOCKED_PUBLIC_CYCLIC_PROFILE
                ),
                readiness_gate=gate,
                public_profile=profile,
            )

        return _WriterCyclicAdmissionDecision(
            kind=(
                _WriterCyclicAdmissionDecisionKind
                .BLOCKED_RESIDUAL_CYCLIC_POLICY
            ),
            readiness_gate=gate,
        )

    if gate.truncated:
        return _WriterCyclicAdmissionDecision(
            kind=(
                _WriterCyclicAdmissionDecisionKind.TRUNCATED_READINESS_AUDIT
            ),
            readiness_gate=gate,
        )

    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"unknown residual cyclic readiness gate state: {gate.kind!r}",
    )


def _writer_public_profile_from_live_graph_policy_blocker(
    prepared: SouthStarPreparedMol,
    gate: _WriterResidualCyclicReadinessGate,
) -> _WriterPublicCyclicOpeningProfileReport | None:
    first = gate.first_blocked_prefix
    if first is None:
        return None

    blockers = first.graph_policy_blockers
    if not blockers:
        return None

    blocker = blockers[0]
    if (
        blocker.kind
        is not (
            _WriterActiveEmittedGraphPolicyBlockerKind
            .EMPTY_CLOSURE_BOND_TEXT_RELATION
        )
    ):
        return None

    if blocker.bond is None:
        return None

    profile = _writer_public_cyclic_opening_profile_report(
        prepared=prepared,
    )
    bond = prepared.graph_index.bond_by_id[blocker.bond]
    unsupported = _closure_policy_blocker_capability_for_order(
        bond.order,
    )
    return replace(
        profile,
        kind=(
            _WriterPublicCyclicOpeningProfileKind
            .BLOCKED_UNSUPPORTED_CLOSURE_BOND_SURFACE
        ),
        ring_core_unsupported_bond_count=max(
            profile.ring_core_unsupported_bond_count,
            1,
        ),
        unsupported_capabilities=frozenset({
            *profile.unsupported_capabilities,
            unsupported,
        }),
    )


def _writer_public_execution_capability_certificate(
    audit: _WriterResidualCyclicReadinessAudit,
) -> _WriterPublicExecutionCapabilityCertificate:
    required = audit.required_execution_capabilities
    supported = _PUBLIC_SUPPORTED_WRITER_EXECUTION_CAPABILITIES
    unsupported = required - supported

    first_by_kind: dict[
        _WriterExecutionCapabilityKind,
        _WriterExecutionCapabilityUse,
    ] = {}
    for use in audit.execution_capability_uses:
        if use.kind not in unsupported:
            continue
        first_by_kind.setdefault(use.kind, use)

    return _WriterPublicExecutionCapabilityCertificate(
        required_capabilities=required,
        supported_capabilities=supported,
        unsupported_capabilities=unsupported,
        first_unsupported_uses=tuple(
            first_by_kind[kind]
            for kind in sorted(first_by_kind, key=lambda item: item.value)
        ),
    )


def _cyclic_writer_admission_ready_gate_from_snapshot(
    snapshot: WriterSearchSnapshot,
) -> _WriterResidualCyclicReadinessGate:
    return _WriterResidualCyclicReadinessGate(
        kind=_WriterResidualCyclicReadinessGateKind.READY,
        snapshot=snapshot,
        audit=_WriterResidualCyclicReadinessAudit(
            kind=_WriterResidualCyclicReadinessAuditKind.READY,
            visited_prefixes=((),),
        ),
    )


def _cyclic_writer_admission_decision_from_snapshot(
    snapshot: WriterSearchSnapshot,
    *,
    prepared: SouthStarPreparedMol,
    max_depth: int | None = None,
    max_prefixes: int | None = None,
) -> _WriterCyclicAdmissionDecision:
    if _PUBLIC_CYCLIC_WRITER_SHAPED_ENABLED:
        profile = _writer_public_cyclic_opening_profile_report(
            prepared=prepared,
        )
        if not profile.supported:
            return _WriterCyclicAdmissionDecision(
                kind=(_WriterCyclicAdmissionDecisionKind
                      .BLOCKED_PUBLIC_CYCLIC_PROFILE),
                readiness_gate=_cyclic_writer_admission_ready_gate_from_snapshot(
                    snapshot,
                ),
                public_profile=profile,
            )

    gate = _residual_cyclic_readiness_gate_from_snapshot(
        snapshot,
        prepared=prepared,
        max_depth=max_depth,
        max_prefixes=max_prefixes,
    )

    return _cyclic_writer_admission_decision_from_readiness_gate(
        gate,
        prepared=prepared,
    )


def _cyclic_writer_admission_decision_from_cursor(
    *,
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions,
    cursor: WriterFrontierCursor,
    max_depth: int | None = None,
    max_prefixes: int | None = None,
) -> _WriterCyclicAdmissionDecision:
    if _PUBLIC_CYCLIC_WRITER_SHAPED_ENABLED:
        profile = _writer_public_cyclic_opening_profile_report(
            prepared=prepared,
        )
        if not profile.supported:
            snapshot = _capture_writer_frontier_snapshot_unchecked(
                prepared=prepared,
                runtime_options=runtime_options,
                cursor=cursor,
            )

            return _WriterCyclicAdmissionDecision(
                kind=(_WriterCyclicAdmissionDecisionKind
                      .BLOCKED_PUBLIC_CYCLIC_PROFILE),
                readiness_gate=_cyclic_writer_admission_ready_gate_from_snapshot(
                    snapshot,
                ),
                public_profile=profile,
            )

    gate = _residual_cyclic_readiness_gate_from_cursor(
        prepared=prepared,
        runtime_options=runtime_options,
        cursor=cursor,
        max_depth=max_depth,
        max_prefixes=max_prefixes,
    )

    return _cyclic_writer_admission_decision_from_readiness_gate(
        gate,
        prepared=prepared,
    )


def _assert_cyclic_writer_admission_decision(
    decision: _WriterCyclicAdmissionDecision,
) -> _WriterCyclicAdmissionDecision:
    if decision.admitted_publicly:
        return decision

    if decision.kind is (
        _WriterCyclicAdmissionDecisionKind.BLOCKED_PUBLIC_CYCLIC_PROFILE
    ):
        unsupported = ""
        if decision.public_profile is not None:
            unsupported = ", ".join(
                sorted(
                    capability.value
                    for capability in (
                        decision.public_profile.unsupported_capabilities
                    )
                )
            )
            if unsupported:
                unsupported = f" unsupported_capabilities=[{unsupported}]"

        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_POLICY,
            (
                "cyclic WRITER_SHAPED blocked by public opening profile: "
                f"{decision.public_profile.kind.value!r}{unsupported}"
            ),
        )

    if decision.kind is (
        _WriterCyclicAdmissionDecisionKind.BLOCKED_PUBLIC_EXECUTION_CAPABILITY
    ):
        certificate = decision.execution_capability_certificate
        if certificate is None:
            raise SouthStarError(
                SouthStarErrorKind.INTERNAL_INVARIANT,
                "missing writer execution capability certificate",
            )
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_POLICY,
            _writer_execution_capability_block_message(certificate),
        )

    if decision.internally_ready and not decision.public_enabled:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_POLICY,
            (
                "cyclic WRITER_SHAPED is internally ready but public "
                "support is closed"
            ),
        )

    if decision.blocked:
        first = decision.first_blocked_prefix
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_POLICY,
            (
                "cyclic WRITER_SHAPED blocked by residual cyclic policy"
                if first is None
                else (
                    "cyclic WRITER_SHAPED blocked by residual cyclic policy "
                    f"at prefix {first.emitted_texts!r}"
                )
            ),
        )

    if decision.truncated:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_POLICY,
            (
                "cyclic WRITER_SHAPED readiness audit truncated at prefix "
                f"{decision.readiness_gate.truncated_at_prefix!r}"
            ),
        )

    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"unknown cyclic writer admission decision: {decision.kind!r}",
    )


def _writer_execution_capability_block_message(
    certificate: _WriterPublicExecutionCapabilityCertificate,
) -> str:
    use = certificate.first_unsupported_uses[0]
    if use.terminal:
        location = f"prefix={use.emitted_texts!r}; at EOS"
    else:
        location = (
            f"prefix={use.emitted_texts!r}; "
            f"next={use.next_emitted_text!r}"
        )
    return (
        "WRITER_SHAPED requires an unsupported South Star "
        "execution capability: "
        f"{use.kind.value}; "
        f"{location}"
    )


def _checked_writer_snapshot_prefix_read_outcome_after_emitted_texts(
    snapshot: WriterSearchSnapshot,
    *,
    prepared: SouthStarPreparedMol,
    emitted_texts: tuple[str, ...],
    include_counts: bool = True,
) -> _WriterSnapshotPrefixReadOutcome:
    outcome = _writer_snapshot_prefix_read_outcome_after_emitted_texts(
        snapshot,
        prepared=prepared,
        emitted_texts=emitted_texts,
        include_counts=include_counts,
        stop_after_first_blocked=True,
    )

    _raise_for_writer_snapshot_replay_choice_snapshot_outcome_errors(
        outcome.replay_outcome
    )

    if outcome.kind is not _WriterSnapshotPrefixReadOutcomeKind.READABLE:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            (
                "checked prefix read did not produce a readable outcome: "
                f"{outcome.kind!r}"
            ),
        )

    return outcome


def _count_writer_frontier_support_after_emitted_texts(
    snapshot: WriterSearchSnapshot,
    *,
    prepared: SouthStarPreparedMol,
    emitted_texts: tuple[str, ...],
) -> int:
    outcome = _checked_writer_snapshot_prefix_read_outcome_after_emitted_texts(
        snapshot,
        prepared=prepared,
        emitted_texts=emitted_texts,
        include_counts=True,
    )

    if outcome.support_count is None:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "checked prefix read did not contain a support count",
        )

    return outcome.support_count


def _count_writer_completions_after_emitted_texts(
    snapshot: WriterSearchSnapshot,
    *,
    prepared: SouthStarPreparedMol,
    emitted_texts: tuple[str, ...],
) -> int:
    outcome = _checked_writer_snapshot_prefix_read_outcome_after_emitted_texts(
        snapshot,
        prepared=prepared,
        emitted_texts=emitted_texts,
        include_counts=True,
    )

    if outcome.completion_count is None:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "checked prefix read did not contain a completion count",
        )

    return outcome.completion_count


def _iter_writer_frontier_support_suffixes_after_emitted_texts(
    snapshot: WriterSearchSnapshot,
    *,
    prepared: SouthStarPreparedMol,
    emitted_texts: tuple[str, ...],
) -> Iterator[str]:
    outcome = _checked_writer_snapshot_prefix_read_outcome_after_emitted_texts(
        snapshot,
        prepared=prepared,
        emitted_texts=emitted_texts,
        include_counts=False,
    )

    if outcome.choice_snapshot is None:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "checked prefix read did not contain a choice snapshot",
        )

    yield from _iter_writer_frontier_support_suffixes_from_choice_snapshot(
        prepared,
        outcome.choice_snapshot,
    )


def _assert_public_writer_snapshot_cyclic_admission(
    snapshot: WriterSearchSnapshot,
    *,
    prepared: SouthStarPreparedMol,
) -> None:
    if not _prepared_has_cyclic_writer_graph_surface(prepared):
        return
    decision = _cyclic_writer_admission_decision_from_snapshot(
        snapshot,
        prepared=prepared,
    )
    _assert_cyclic_writer_admission_decision(decision)


def resume_writer_frontier_choices_from_snapshot(
    snapshot: WriterSearchSnapshot,
    *,
    prepared: SouthStarPreparedMol,
) -> WriterFrontierChoices:
    _assert_public_writer_snapshot_cyclic_admission(
        snapshot,
        prepared=prepared,
    )
    return _writer_frontier_choices_after_emitted_texts(
        snapshot,
        prepared=prepared,
        emitted_texts=(),
    )


def advance_writer_frontier_snapshot(
    snapshot: WriterSearchSnapshot,
    *,
    prepared: SouthStarPreparedMol,
    emitted_text: str,
) -> WriterSearchSnapshot:
    _assert_public_writer_snapshot_cyclic_admission(
        snapshot,
        prepared=prepared,
    )

    outcome = (
        _checked_writer_snapshot_prefix_read_outcome_after_emitted_texts(
            snapshot,
            prepared=prepared,
            emitted_texts=(emitted_text,),
            include_counts=False,
        )
    )

    advanced_snapshot = outcome.replay_outcome.advanced_snapshot
    if advanced_snapshot is None:
        raise AssertionError(
            "checked writer snapshot advance did not produce snapshot"
        )

    _assert_public_writer_snapshot_cyclic_admission(
        advanced_snapshot,
        prepared=prepared,
    )
    return advanced_snapshot


def validate_writer_search_snapshot(
    snapshot: WriterSearchSnapshot,
    *,
    prepared: SouthStarPreparedMol,
) -> None:
    if snapshot.serialization_language is not SerializationLanguageMode.WRITER_SHAPED:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_POLICY,
            "writer snapshot requires serialization_language=WRITER_SHAPED",
        )
    require_writer_shaped_runtime_options(snapshot.runtime_options)
    if snapshot.prepared_identity != _prepared_identity(
        prepared,
        snapshot.runtime_options,
    ):
        raise SouthStarError(
            SouthStarErrorKind.INVALID_FACTS,
            "writer snapshot prepared identity does not match prepared molecule",
        )
    _validate_cursor_active_frames(snapshot.cursor)
    if snapshot.cursor != WriterFrontierCursor(snapshot.cursor.weighted_states):
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "writer snapshot cursor is not canonical",
        )
    _validate_frames(snapshot.frame_stack, snapshot.cursor)
    stereo_residual_cache: dict[
        tuple[WriterStereoStateKey, WriterRingStateKey],
        ResidualStoreValueSnapshot,
    ] = {}
    validate_writer_cursor_against_prepared(
        prepared,
        snapshot.cursor,
        runtime_options=snapshot.runtime_options,
        stereo_residual_cache=stereo_residual_cache,
    )


def validate_writer_cursor_against_prepared(
    prepared: SouthStarPreparedMol,
    cursor: WriterFrontierCursor,
    *,
    runtime_options: SouthStarRuntimeOptions | None = None,
    stereo_residual_cache: dict[
        tuple[WriterStereoStateKey, WriterRingStateKey],
        ResidualStoreValueSnapshot,
    ] | None = None,
) -> None:
    _validate_cursor_active_frames(cursor)
    atom_ids = frozenset(prepared.atom_ids)
    bond_ids = frozenset(bond.id for bond in prepared.facts.bonds)
    allowed_roots = _allowed_component_roots(prepared, runtime_options)
    atom_component = _atom_component_index(prepared)
    bond_component = _bond_component_index(prepared)
    for key, weight in cursor.weighted_states:
        if weight <= 0:
            _invalid_snapshot("writer cursor contains nonpositive weight")
        _validate_component_cursor(key.component_cursor, allowed_roots)
        context = build_writer_graph_obligation_context(prepared, key)
        validate_writer_snapshot_graph_surface(prepared, key, context)
        _validate_edge_partition_supported_for_snapshot(context)
        _validate_residual_attachments_supported_for_snapshot(context)
        _validate_atom_frame(key.active, atom_ids, bond_ids, prepared)
        for frame in key.branch_stack:
            _validate_branch_frame(frame, atom_ids, bond_ids, prepared)
        _validate_known_atoms("visited_atoms", key.visited_atoms, atom_ids)
        _validate_known_bonds("written_bonds", key.written_bonds, bond_ids)
        _validate_active_coherence(key)
        _validate_component_membership(prepared, key, atom_component, bond_component)
        _validate_current_component_tree_fragment(prepared, key)
        _validate_writer_frame_tree_path(prepared, key)
        _validate_written_bond_coherence(prepared, key)
        _validate_obligations(
            key.obligations,
            key,
            atom_ids,
            bond_ids,
            prepared,
            context,
        )
        _validate_live_frontier_ownership(prepared, key, context)
        _validate_terminal_graph_completion(prepared, key, context)
        _validate_stereo_occurrences_bound_to_graph_state(prepared, key, context)
        _validate_ring_state(prepared, key, context)
        _validate_policy_state(key, atom_ids, bond_ids)
        _validate_stereo_state(
            prepared,
            key.stereo_state,
            ring_state=key.ring_state,
            stereo_residual_cache=stereo_residual_cache,
        )


def _validate_cursor_active_frames(cursor: WriterFrontierCursor) -> None:
    for key, _ in cursor.weighted_states:
        if key.active is None:
            _invalid_snapshot("writer snapshot state missing active frame")


def _validate_frames(
    frame_stack: tuple[object, ...],
    cursor: WriterFrontierCursor,
) -> None:
    if len(frame_stack) != 1:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "writer snapshot currently requires exactly one frontier frame",
        )
    frame = frame_stack[0]
    if not isinstance(frame, WriterFrontierFrame):
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "writer snapshot top frame must be a frontier frame",
        )
    if frame.cursor != cursor:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "writer snapshot frontier frame cursor must match snapshot cursor",
        )


def _round_trip_residual_snapshot(snapshot: ResidualStoreValueSnapshot) -> None:
    try:
        round_tripped = ResidualStore.from_value_snapshot(snapshot).value_snapshot()
    except ValueError as exc:
        _invalid_snapshot(f"writer residual snapshot is invalid: {exc}")
    if round_tripped != snapshot:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "writer residual snapshot does not round-trip",
        )


def _validate_edge_partition_supported_for_snapshot(
    context: WriterGraphObligationContext,
) -> None:
    if any(
        obligation.kind is WriterEdgeObligationKind.CLOSURE_CANDIDATE
        for obligation in context.edge_partition.obligations
    ):
        _invalid_snapshot("writer snapshot has unsupported cyclic edge obligation")


def _validate_residual_attachments_supported_for_snapshot(
    context: WriterGraphObligationContext,
) -> None:
    if context.residual_summary.has_unsupported_attachment:
        _invalid_snapshot("writer snapshot has unsupported residual attachment")


def _allowed_component_roots(
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions | None,
) -> tuple[frozenset[AtomId], ...]:
    if runtime_options is None or runtime_options.rooted_at_atom < 0:
        domains = prepared.all_root_domains
    else:
        try:
            domains = prepared.component_root_domains_by_explicit_root[
                AtomId(runtime_options.rooted_at_atom)
            ]
        except KeyError as exc:
            raise SouthStarError(
                SouthStarErrorKind.INVALID_FACTS,
                "writer snapshot runtime root is not in prepared molecule",
            ) from exc
    return tuple(frozenset(atoms) for _, atoms in domains)


def _validate_component_cursor(
    cursor: ComponentCursor,
    allowed_roots: tuple[frozenset[AtomId], ...],
) -> None:
    if len(cursor.component_roots) != len(allowed_roots):
        _invalid_snapshot("writer component root count does not match prepared domains")
    if cursor.component_index < 0 or cursor.component_index >= len(cursor.component_roots):
        _invalid_snapshot("writer component index is outside component roots")
    for index, root in enumerate(cursor.component_roots):
        if root not in allowed_roots[index]:
            _invalid_snapshot("writer component root is outside runtime root domain")


def _validate_atom_frame(
    frame: WriterAtomFrame,
    atom_ids: frozenset[AtomId],
    bond_ids: frozenset[BondId],
    prepared: SouthStarPreparedMol,
) -> None:
    if frame.atom not in atom_ids:
        _invalid_snapshot("writer atom frame references unknown atom")
    if frame.parent is None or frame.incoming_bond is None:
        if frame.parent is not None or frame.incoming_bond is not None:
            _invalid_snapshot("writer atom frame has partial incoming edge")
        return
    if frame.parent not in atom_ids or frame.incoming_bond not in bond_ids:
        _invalid_snapshot("writer atom frame references unknown incoming edge")
    _require_graph_bond(prepared, frame.parent, frame.atom, frame.incoming_bond)


def _validate_branch_frame(
    frame: WriterBranchFrame,
    atom_ids: frozenset[AtomId],
    bond_ids: frozenset[BondId],
    prepared: SouthStarPreparedMol,
) -> None:
    _validate_atom_frame(frame.return_atom, atom_ids, bond_ids, prepared)
    if not frame.return_atom.atom_emitted:
        _invalid_snapshot("writer branch return frame must be emitted")


def _validate_known_atoms(
    label: str,
    atoms: frozenset[AtomId],
    atom_ids: frozenset[AtomId],
) -> None:
    if not atoms.issubset(atom_ids):
        _invalid_snapshot(f"writer {label} references unknown atom")


def _validate_known_bonds(
    label: str,
    bonds: frozenset[BondId],
    bond_ids: frozenset[BondId],
) -> None:
    if not bonds.issubset(bond_ids):
        _invalid_snapshot(f"writer {label} references unknown bond")


def _validate_active_coherence(key: WriterStateKey) -> None:
    active = key.active
    if active.parent is None:
        if active.atom != key.component_cursor.component_roots[
            key.component_cursor.component_index
        ]:
            _invalid_snapshot("writer root active frame does not match component root")
        if active.incoming_bond is not None:
            _invalid_snapshot("writer root active frame has incoming bond")
    elif active.incoming_bond is None:
        _invalid_snapshot("writer non-root active frame lacks incoming bond")
    if active.atom_emitted:
        if active.atom not in key.visited_atoms:
            _invalid_snapshot("writer emitted active atom is not visited")
    elif active.atom in key.visited_atoms:
        _invalid_snapshot("writer un-emitted active atom is already visited")
    if active.parent is None:
        if active.incoming_bond is not None:
            _invalid_snapshot("writer root active frame has incoming bond")
    elif active.parent not in key.visited_atoms:
        _invalid_snapshot("writer active parent is not visited")
    if active.incoming_bond is not None and active.atom_emitted:
        if active.incoming_bond not in key.written_bonds:
            _invalid_snapshot("writer emitted child lacks written incoming bond")
    for frame in key.branch_stack:
        if frame.return_atom.atom not in key.visited_atoms:
            _invalid_snapshot("writer branch return atom is not visited")


def _validate_component_membership(
    prepared: SouthStarPreparedMol,
    key: WriterStateKey,
    atom_component: dict[AtomId, int],
    bond_component: dict[BondId, int],
) -> None:
    current = key.component_cursor.component_index
    allowed_components = set(range(current + 1))
    active = key.active
    if atom_component[active.atom] != current:
        _invalid_snapshot("writer active atom is outside current component")
    for atom in key.visited_atoms:
        if atom_component[atom] not in allowed_components:
            _invalid_snapshot("writer visited atom is outside completed/current components")
    for bond in key.written_bonds:
        if bond_component[bond] not in allowed_components:
            _invalid_snapshot("writer written bond is outside completed/current components")
    pending = key.obligations.pending_entry
    if pending is not None:
        if (
            atom_component[pending.parent] != current
            or atom_component[pending.child] != current
            or bond_component[pending.bond] != current
        ):
            _invalid_snapshot("writer pending entry is outside current component")
    for frame in key.branch_stack:
        if atom_component[frame.return_atom.atom] != current:
            _invalid_snapshot("writer branch return atom is outside current component")
    for index, component in enumerate(prepared.facts.components):
        if index >= current:
            break
        if not frozenset(component.atoms).issubset(key.visited_atoms):
            _invalid_snapshot("writer completed component has unvisited atoms")
        if not frozenset(component.bonds).issubset(key.written_bonds):
            _invalid_snapshot("writer completed component has unwritten bonds")


def _validate_current_component_tree_fragment(
    prepared: SouthStarPreparedMol,
    key: WriterStateKey,
) -> None:
    current = key.component_cursor.component_index
    component = prepared.facts.components[current]
    component_atoms = frozenset(component.atoms)
    component_bonds = frozenset(component.bonds)
    root = key.component_cursor.component_roots[current]
    visited = frozenset(atom for atom in key.visited_atoms if atom in component_atoms)
    written = frozenset(bond for bond in key.written_bonds if bond in component_bonds)
    if not visited:
        if written:
            _invalid_snapshot("writer current component has written bonds before root")
        return
    if root not in visited:
        _invalid_snapshot("writer current component visited atoms do not include root")
    if len(written) != len(visited) - 1:
        _invalid_snapshot("writer current component written graph is not a tree fragment")
    reachable = _reachable_written_atoms(prepared, root, written)
    if reachable != visited:
        _invalid_snapshot("writer current component visited atoms are not root-reachable")
    active = key.active
    if active.atom_emitted and active.atom not in reachable:
        _invalid_snapshot("writer active atom is not in reachable written graph")
    for frame in key.branch_stack:
        if frame.return_atom.atom not in reachable:
            _invalid_snapshot("writer branch return atom is not in reachable written graph")
    pending = key.obligations.pending_entry
    if pending is not None:
        if pending.parent not in reachable:
            _invalid_snapshot("writer pending parent is not in reachable written graph")
        if pending.phase is PendingEntryPhase.NEEDS_ATOM_AFTER_BOND:
            if pending.bond in written or pending.child in visited:
                _invalid_snapshot("writer pending post-bond edge is already materialized")


def _reachable_written_atoms(
    prepared: SouthStarPreparedMol,
    root: AtomId,
    written_bonds: frozenset[BondId],
) -> frozenset[AtomId]:
    adjacency: dict[AtomId, set[AtomId]] = {}
    for bond in written_bonds:
        fact = prepared.graph_index.bond_by_id[bond]
        adjacency.setdefault(fact.a, set()).add(fact.b)
        adjacency.setdefault(fact.b, set()).add(fact.a)
    seen = {root}
    stack = [root]
    while stack:
        atom = stack.pop()
        for neighbor in adjacency.get(atom, ()):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            stack.append(neighbor)
    return frozenset(seen)


def _validate_writer_frame_tree_path(
    prepared: SouthStarPreparedMol,
    key: WriterStateKey,
) -> None:
    parent_links = _written_tree_parent_links(prepared, key)
    root = key.component_cursor.component_roots[key.component_cursor.component_index]
    _validate_atom_frame_tree_edge(key.active, root, parent_links)
    for frame in key.branch_stack:
        _validate_atom_frame_tree_edge(frame.return_atom, root, parent_links)
    if not key.active.atom_emitted:
        if key.branch_stack:
            _invalid_snapshot("writer branch stack requires emitted active atom")
        return
    active_path = _root_to_atom_path(root, key.active.atom, parent_links)
    ancestor_positions = {atom: index for index, atom in enumerate(active_path[:-1])}
    previous_position = -1
    for frame in key.branch_stack:
        position = ancestor_positions.get(frame.return_atom.atom)
        if position is None:
            _invalid_snapshot("writer branch return atom is not an active ancestor")
        if position <= previous_position:
            _invalid_snapshot("writer branch stack does not follow root-to-active path")
        previous_position = position


def _validate_atom_frame_tree_edge(
    frame: WriterAtomFrame,
    root: AtomId,
    parent_links: dict[AtomId, tuple[AtomId, BondId]],
) -> None:
    if not frame.atom_emitted:
        return
    if frame.atom == root:
        if frame.parent is not None or frame.incoming_bond is not None:
            _invalid_snapshot("writer root frame disagrees with written-tree root")
        return
    expected = parent_links.get(frame.atom)
    if expected is None:
        _invalid_snapshot("writer atom frame is missing from written-tree parent links")
    if (frame.parent, frame.incoming_bond) != expected:
        _invalid_snapshot("writer atom frame disagrees with written-tree orientation")


def _root_to_atom_path(
    root: AtomId,
    atom: AtomId,
    parent_links: dict[AtomId, tuple[AtomId, BondId]],
) -> tuple[AtomId, ...]:
    reversed_path = [atom]
    current = atom
    seen = {atom}
    while current != root:
        parent = parent_links.get(current)
        if parent is None:
            _invalid_snapshot("writer active atom is not connected to written-tree root")
        current = parent[0]
        if current in seen:
            _invalid_snapshot("writer written-tree parent links contain a cycle")
        seen.add(current)
        reversed_path.append(current)
    return tuple(reversed(reversed_path))


def _validate_written_bond_coherence(
    prepared: SouthStarPreparedMol,
    key: WriterStateKey,
) -> None:
    for bond in key.written_bonds:
        fact = prepared.graph_index.bond_by_id[bond]
        left, right = fact.a, fact.b
        if left not in key.visited_atoms or right not in key.visited_atoms:
            _invalid_snapshot("writer written bond has unvisited endpoint")


def _validate_obligations(
    obligations: ObligationStateKey,
    key: WriterStateKey,
    atom_ids: frozenset[AtomId],
    bond_ids: frozenset[BondId],
    prepared: SouthStarPreparedMol,
    context: WriterGraphObligationContext,
) -> None:
    pending = obligations.pending_entry
    if pending is None:
        return
    _validate_pending_entry(pending, atom_ids, bond_ids, prepared)
    _validate_pending_entry_role(context, pending)
    if key.active.atom != pending.parent:
        _invalid_snapshot("writer pending entry parent is not active")
    if not key.active.atom_emitted:
        _invalid_snapshot("writer pending entry parent is not emitted")
    if pending.parent not in key.visited_atoms:
        _invalid_snapshot("writer pending parent is not visited")
    if pending.child in key.visited_atoms or pending.bond in key.written_bonds:
        _invalid_snapshot("writer pending entry is already written")
    has_bond_record = _has_bond_occurrence_record(
        key.stereo_state,
        pending.bond,
        pending.parent,
        pending.child,
    )
    if pending.phase is PendingEntryPhase.NEEDS_ATOM_AFTER_BOND:
        if not has_bond_record:
            _invalid_snapshot("writer pending post-bond entry lacks bond occurrence")
    elif pending.phase is PendingEntryPhase.NEEDS_BOND_OR_ATOM:
        if has_bond_record:
            _invalid_snapshot("writer pending pre-bond entry already has bond occurrence")
    else:
        _invalid_snapshot("writer pending entry has unknown phase")


def _validate_pending_entry(
    pending: PendingWriterEntry,
    atom_ids: frozenset[AtomId],
    bond_ids: frozenset[BondId],
    prepared: SouthStarPreparedMol,
) -> None:
    if pending.parent not in atom_ids or pending.child not in atom_ids:
        _invalid_snapshot("writer pending entry references unknown atom")
    if pending.bond not in bond_ids:
        _invalid_snapshot("writer pending entry references unknown bond")
    _require_graph_bond(prepared, pending.parent, pending.child, pending.bond)


def _validate_pending_entry_role(
    context: WriterGraphObligationContext,
    pending: PendingWriterEntry,
) -> None:
    summary = context.residual_summary
    children = tuple(
        sorted(
            (*_boundary_children_for_atom(summary, pending.parent), (pending.bond, pending.child)),
            key=lambda item: (int(item[0]), int(item[1])),
        )
    )
    if (pending.bond, pending.child) not in children:
        _invalid_snapshot("writer pending entry is not a live child obligation")
    if pending.branch:
        if len(children) <= 1:
            _invalid_snapshot("writer pending branch entry has no sibling obligations")
    elif children != ((pending.bond, pending.child),):
        _invalid_snapshot("writer pending inline entry is not the final child")


def _validate_live_frontier_ownership(
    prepared: SouthStarPreparedMol,
    key: WriterStateKey,
    context: WriterGraphObligationContext,
) -> None:
    summary = context.residual_summary
    current = key.component_cursor.component_index
    component = prepared.facts.components[current]
    component_atoms = frozenset(component.atoms)
    visited = frozenset(atom for atom in key.visited_atoms if atom in component_atoms)
    unvisited = component_atoms - visited
    if not visited:
        return
    boundary_edges = [
        incidence
        for attachment in summary.attachments.attachments
        for incidence in attachment.boundary
    ]
    blocked_actions = tuple(
        action
        for action in summary.attachment_actions
        if writer_residual_attachment_action_is_blocked(action)
    )
    if any(
        action.kind is WriterResidualAttachmentActionKind.BLOCKED_ORPHAN
        for action in blocked_actions
    ):
        _invalid_snapshot("writer residual attachment has no boundary incidence")
    branch_return_atoms = tuple(frame.return_atom.atom for frame in key.branch_stack)
    branch_owned_atoms = {
        incidence.written_atom
        for incidence in boundary_edges
        if incidence.owner_kind is WriterBoundaryOwnerKind.BRANCH_RETURN
    }
    if any(
        action.kind is WriterResidualAttachmentActionKind.BLOCKED_UNOWNED
        for action in blocked_actions
    ):
        _invalid_snapshot("writer live frontier does not own unvisited obligation")
    action_by_id = {
        action.attachment_id: action for action in summary.attachment_actions
    }
    for attachment in summary.attachments.attachments:
        action = action_by_id[attachment.attachment_id]
        if action.kind is WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY:
            continue
        if any(
            incidence.owner_kind is WriterBoundaryOwnerKind.UNOWNED
            for incidence in attachment.boundary
        ):
            _invalid_snapshot("writer live frontier does not own unvisited obligation")
    pending_owned_attachment = any(
        _attachment_is_owned_by_pending_entry(key, attachment.atoms)
        for attachment in summary.attachments.attachments
    )
    if unvisited and not boundary_edges and not pending_owned_attachment:
        _invalid_snapshot("writer current component has unvisited atoms without frontier")
    if key.branch_stack and not unvisited:
        _invalid_snapshot("writer branch stack has no unresolved return obligation")
    if any(atom not in branch_owned_atoms for atom in branch_return_atoms):
        _invalid_snapshot("writer branch return frame owns no unresolved obligation")
    if (
        not unvisited
        and key.obligations.pending_entry is None
        and not key.branch_stack
        and not _active_is_terminal_leaf(prepared, key)
    ):
        _invalid_snapshot("writer completed component active frame is not terminal")


def _validate_terminal_graph_completion(
    prepared: SouthStarPreparedMol,
    key: WriterStateKey,
    context: WriterGraphObligationContext,
) -> None:
    if not _state_is_terminal_shape(prepared, key, context):
        return
    completion = writer_graph_completion_status(prepared, key, context)
    if not completion.complete:
        _invalid_snapshot("writer terminal state has unresolved graph obligations")


def _boundary_children_for_atom(
    summary: WriterGraphObligationSummary,
    atom: AtomId,
) -> tuple[tuple[BondId, AtomId], ...]:
    attachments_by_id = {
        attachment.attachment_id: attachment
        for attachment in summary.attachments.attachments
    }
    children = []
    for action in summary.attachment_actions:
        if action.kind not in (
            WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY,
            WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY,
        ):
            continue
        attachment = attachments_by_id[action.attachment_id]
        boundary = tuple(
            incidence
            for incidence in attachment.boundary
            if incidence.written_atom == atom
        )
        if not boundary:
            continue
        if len(boundary) != 1:
            _invalid_snapshot("writer residual attachment has multiple incidences")
        incidence = boundary[0]
        children.append((incidence.bond, incidence.residual_atom))
    return tuple(sorted(children, key=lambda item: (int(item[0]), int(item[1]))))


def _attachment_is_owned_by_pending_entry(
    key: WriterStateKey,
    atoms: frozenset[AtomId],
) -> bool:
    pending = key.obligations.pending_entry
    return pending is not None and pending.child in atoms


def _active_is_terminal_leaf(
    prepared: SouthStarPreparedMol,
    key: WriterStateKey,
) -> bool:
    active = key.active
    if not active.atom_emitted:
        return False
    current = key.component_cursor.component_index
    component = prepared.facts.components[current]
    if len(component.atoms) == 1:
        return active.atom == key.component_cursor.component_roots[current]
    parent_links = _written_tree_parent_links(prepared, key)
    children = {
        child
        for child, (parent, _) in parent_links.items()
        if parent == active.atom
    }
    return not children


def _validate_stereo_occurrences_bound_to_graph_state(
    prepared: SouthStarPreparedMol,
    key: WriterStateKey,
    context: WriterGraphObligationContext,
) -> None:
    atom_occurrence_atoms = frozenset(
        record.atom for record in key.stereo_state.atom_occurrences
    )
    if atom_occurrence_atoms != key.visited_atoms:
        _invalid_snapshot("writer atom occurrences do not cover visited atoms")
    pending_bond = _pending_post_bond_edge(key)
    expected_bonds = set(key.written_bonds)
    if pending_bond is not None:
        expected_bonds.add(pending_bond.bond)
    directional_carrier_bonds = frozenset(
        _directional_sites_by_carrier_bond(prepared)
    )
    closed_directional_closure_bonds = frozenset(
        closure.bond
        for closure in key.ring_state.closed_closures
        if closure.bond in directional_carrier_bonds
    )
    expected_bonds.update(closed_directional_closure_bonds)
    bond_occurrence_bonds = frozenset(
        record.bond for record in key.stereo_state.bond_occurrences
    )
    if bond_occurrence_bonds != frozenset(expected_bonds):
        _invalid_snapshot("writer bond occurrences do not cover emitted bonds")
    parent_links = _written_tree_parent_links(prepared, key)
    parent_by_child = {
        child: parent
        for child, (parent, _bond) in parent_links.items()
    }
    _validate_atom_occurrence_traversal_order(prepared, key, parent_links)
    for record in key.stereo_state.atom_occurrences:
        if record.atom not in key.visited_atoms:
            _invalid_snapshot("writer atom occurrence is not backed by visited atom")
    for record in key.stereo_state.local_orders:
        if record.atom not in key.visited_atoms:
            _invalid_snapshot("writer local-order record is not backed by visited atom")
    for record in key.stereo_state.bond_occurrences:
        if record.bond in key.written_bonds:
            if record.parent not in key.visited_atoms or record.child not in key.visited_atoms:
                _invalid_snapshot("writer bond occurrence has unvisited written endpoint")
            expected = parent_links.get(record.child)
            if expected != (record.parent, record.bond):
                _invalid_snapshot("writer bond occurrence has wrong writer orientation")
            continue
        if (
            pending_bond is not None
            and pending_bond.bond == record.bond
            and pending_bond.parent == record.parent
            and pending_bond.child == record.child
        ):
            if record.parent not in key.visited_atoms:
                _invalid_snapshot("writer pending bond occurrence has unvisited parent")
            if record.child in key.visited_atoms or record.bond in key.written_bonds:
                _invalid_snapshot("writer pending bond occurrence is already materialized")
            continue
        if record.bond in closed_directional_closure_bonds:
            closure = next(
                item
                for item in key.ring_state.closed_closures
                if item.bond == record.bond
            )
            if (
                record.parent not in key.visited_atoms
                or record.child not in key.visited_atoms
            ):
                _invalid_snapshot("writer closure bond occurrence has unvisited endpoint")
            if frozenset((record.parent, record.child)) != frozenset(
                (closure.first_atom, closure.second_atom)
            ):
                _invalid_snapshot("writer closure bond occurrence has wrong endpoints")
            continue
        _invalid_snapshot("writer bond occurrence is not backed by emitted graph state")

    actual_by_atom = {
        record.atom: record
        for record in key.stereo_state.local_orders
    }
    open_frame_atoms = {
        frame.return_atom.atom
        for frame in key.branch_stack
    }
    if key.active.atom_emitted:
        open_frame_atoms.add(key.active.atom)

    closed_atoms = set(key.visited_atoms) - open_frame_atoms
    active_record = actual_by_atom.get(key.active.atom)
    active_is_closed = active_record is not None and active_record.closed
    if active_is_closed:
        if not _state_is_terminal_shape(prepared, key, context):
            _invalid_snapshot(
                "writer active local order is closed before terminal shape"
            )
        closed_atoms.add(key.active.atom)

    for atom in {
        frame.return_atom.atom
        for frame in key.branch_stack
    }:
        record = actual_by_atom.get(atom)
        if record is not None and record.closed:
            _invalid_snapshot(
                "writer branch-return local order is prematurely closed"
            )

    expected_records = reconstruct_writer_local_order_records(
        prepared,
        atom_occurrences=key.stereo_state.atom_occurrences,
        parent_by_child=parent_by_child,
        closed_atoms=frozenset(closed_atoms),
        ring_incidences_by_atom=_ring_incidences_by_atom(key),
    )
    expected_by_atom = {
        record.atom: record
        for record in expected_records
    }
    if actual_by_atom != expected_by_atom:
        _invalid_snapshot(
            "writer local-order history does not match emitted tree history"
        )


def _ring_incidences_by_atom(
    key: WriterStateKey,
) -> dict[AtomId, tuple[tuple[BondId, AtomId], ...]]:
    incidences: dict[AtomId, list[tuple[BondId, AtomId]]] = {}
    for endpoint in key.ring_state.open_endpoints:
        incidences.setdefault(endpoint.first_atom, []).append(
            (endpoint.bond, endpoint.second_atom),
        )
    for closure in key.ring_state.closed_closures:
        incidences.setdefault(closure.first_atom, []).append(
            (closure.bond, closure.second_atom),
        )
        incidences.setdefault(closure.second_atom, []).append(
            (closure.bond, closure.first_atom),
        )
    return {
        atom: tuple(entries)
        for atom, entries in incidences.items()
    }


def _validate_atom_occurrence_traversal_order(
    prepared: SouthStarPreparedMol,
    key: WriterStateKey,
    parent_links: Mapping[AtomId, tuple[AtomId, BondId]],
) -> None:
    occurrence_atoms = tuple(
        record.atom
        for record in key.stereo_state.atom_occurrences
    )
    position = {
        atom: index
        for index, atom in enumerate(occurrence_atoms)
    }

    for child, (parent, _bond) in parent_links.items():
        if position[parent] >= position[child]:
            _invalid_snapshot(
                "writer atom occurrence precedes its tree parent"
            )

    children_by_parent: dict[AtomId, list[AtomId]] = {}
    for child, (parent, _bond) in parent_links.items():
        children_by_parent.setdefault(parent, []).append(child)
    for children in children_by_parent.values():
        children.sort(key=position.__getitem__)

    component_by_atom = _atom_component_index(prepared)
    component_sequence = tuple(
        component_by_atom[atom]
        for atom in occurrence_atoms
    )
    if any(
        left > right
        for left, right in zip(component_sequence, component_sequence[1:])
    ):
        _invalid_snapshot(
            "writer atom occurrence component order is not depth-first"
        )

    for index in range(key.component_cursor.component_index + 1):
        root = key.component_cursor.component_roots[index]
        component_occurrences = tuple(
            atom
            for atom in occurrence_atoms
            if component_by_atom.get(atom) == index
        )
        if not component_occurrences:
            continue
        if component_occurrences[0] != root:
            _invalid_snapshot(
                "writer atom occurrence component does not start at root"
            )

        expected: list[AtomId] = []

        def visit(atom: AtomId) -> None:
            expected.append(atom)
            for child in children_by_parent.get(atom, ()):
                visit(child)

        visit(root)
        expected_seen = tuple(
            atom
            for atom in expected
            if atom in position and component_by_atom.get(atom) == index
        )
        if component_occurrences != expected_seen:
            _invalid_snapshot(
                "writer atom occurrence order is not depth-first"
            )

    current_index = key.component_cursor.component_index
    current_component_occurrences = tuple(
        atom
        for atom in occurrence_atoms
        if component_by_atom.get(atom) == current_index
    )
    if key.active.atom_emitted:
        if not current_component_occurrences:
            _invalid_snapshot(
                "writer emitted active frame lacks atom occurrence"
            )
        if not _is_tree_ancestor_or_self(
            key.active.atom,
            current_component_occurrences[-1],
            parent_links,
        ):
            _invalid_snapshot(
                "writer active frame is inconsistent with atom occurrence order"
            )
    elif current_component_occurrences:
        _invalid_snapshot(
            "writer unemitted active frame has atom occurrence history"
        )


def _is_tree_ancestor_or_self(
    ancestor: AtomId,
    atom: AtomId,
    parent_links: Mapping[AtomId, tuple[AtomId, BondId]],
) -> bool:
    current = atom
    while True:
        if current == ancestor:
            return True
        link = parent_links.get(current)
        if link is None:
            return False
        current = link[0]


def _pending_post_bond_edge(key: WriterStateKey) -> PendingWriterEntry | None:
    pending = key.obligations.pending_entry
    if pending is None or pending.phase is not PendingEntryPhase.NEEDS_ATOM_AFTER_BOND:
        return None
    return pending


def _written_tree_parent_links(
    prepared: SouthStarPreparedMol,
    key: WriterStateKey,
) -> dict[AtomId, tuple[AtomId, BondId]]:
    parent_by_child: dict[AtomId, tuple[AtomId, BondId]] = {}
    for index in range(key.component_cursor.component_index + 1):
        component = prepared.facts.components[index]
        component_bonds = frozenset(component.bonds)
        written = frozenset(bond for bond in key.written_bonds if bond in component_bonds)
        root = key.component_cursor.component_roots[index]
        adjacency: dict[AtomId, list[tuple[AtomId, BondId]]] = {}
        for bond in written:
            fact = prepared.graph_index.bond_by_id[bond]
            adjacency.setdefault(fact.a, []).append((fact.b, bond))
            adjacency.setdefault(fact.b, []).append((fact.a, bond))
        seen = {root}
        stack = [root]
        while stack:
            parent = stack.pop()
            for child, bond in adjacency.get(parent, ()):
                if child in seen:
                    continue
                seen.add(child)
                parent_by_child[child] = (parent, bond)
                stack.append(child)
    return parent_by_child


def _validate_ring_state(
    prepared: SouthStarPreparedMol,
    key: WriterStateKey,
    context: WriterGraphObligationContext,
) -> None:
    partition_by_bond = {
        obligation.bond: obligation.kind
        for obligation in context.edge_partition.obligations
    }
    open_labels = tuple(endpoint.label for endpoint in key.ring_state.open_endpoints)
    if len(set(open_labels)) != len(open_labels):
        _invalid_snapshot("writer open closure labels contain duplicates")
    for endpoint in key.ring_state.open_endpoints:
        if partition_by_bond.get(endpoint.bond) is not WriterEdgeObligationKind.OPEN_CLOSURE_ENDPOINT:
            _invalid_snapshot("writer open closure endpoint lacks edge obligation")
        fact = prepared.graph_index.bond_by_id.get(endpoint.bond)
        if fact is None:
            _invalid_snapshot("writer open closure endpoint references unknown bond")
        if {endpoint.first_atom, endpoint.second_atom} != {fact.a, fact.b}:
            _invalid_snapshot("writer open closure endpoint has wrong atoms")
        if endpoint.first_atom not in key.visited_atoms:
            _invalid_snapshot("writer open closure first atom is not visited")
    for closure in key.ring_state.closed_closures:
        if partition_by_bond.get(closure.bond) is not WriterEdgeObligationKind.CLOSED_CLOSURE:
            _invalid_snapshot("writer closed closure lacks edge obligation")
        fact = prepared.graph_index.bond_by_id.get(closure.bond)
        if fact is None:
            _invalid_snapshot("writer closed closure references unknown bond")
        if {closure.first_atom, closure.second_atom} != {fact.a, fact.b}:
            _invalid_snapshot("writer closed closure has wrong atoms")
        if closure.first_atom not in key.visited_atoms or closure.second_atom not in key.visited_atoms:
            _invalid_snapshot("writer closed closure endpoint is not visited")
    if key.ring_state.open_endpoints and _state_is_terminal_shape(prepared, key, context):
        _invalid_snapshot("writer terminal snapshot has open closure endpoints")


def _state_is_terminal_shape(
    prepared: SouthStarPreparedMol,
    key: WriterStateKey,
    context: WriterGraphObligationContext,
) -> bool:
    if key.obligations.pending_entry is not None or key.branch_stack:
        return False
    if key.ring_state.open_endpoints:
        return False
    if _active_owns_live_attachment_action(key, context):
        return False
    if key.component_cursor.component_index + 1 < len(key.component_cursor.component_roots):
        return False
    return _active_is_terminal_leaf(prepared, key)


def _active_owns_live_attachment_action(
    key: WriterStateKey,
    context: WriterGraphObligationContext,
) -> bool:
    live_kinds = (
        WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY,
        WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY,
        WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY,
    )
    return any(
        action.kind in live_kinds and key.active.atom in action.owner_atoms
        for action in context.residual_summary.attachment_actions
    )


def _validate_policy_state(
    key: WriterStateKey,
    atom_ids: frozenset[AtomId],
    bond_ids: frozenset[BondId],
) -> None:
    if any(atom not in atom_ids for atom, _ in key.policy_state.atom_text):
        _invalid_snapshot("writer policy atom text references unknown atom")
    if any(bond not in bond_ids for bond, _ in key.policy_state.bond_text):
        _invalid_snapshot("writer policy bond text references unknown bond")


def _validate_stereo_state(
    prepared: SouthStarPreparedMol,
    stereo_state: WriterStereoStateKey,
    *,
    ring_state: WriterRingStateKey | None = None,
    stereo_residual_cache: dict[
        tuple[WriterStereoStateKey, WriterRingStateKey],
        ResidualStoreValueSnapshot,
    ] | None = None,
) -> None:
    _round_trip_residual_snapshot(stereo_state.residual_snapshot)
    _validate_unique_stereo_records(stereo_state)
    occurrence_by_id = {item.id: item for item in prepared.facts.ligand_occurrences}
    atom_ids = frozenset(prepared.atom_ids)
    bond_ids = frozenset(bond.id for bond in prepared.facts.bonds)
    tetra_by_center = {template.center: template for template in prepared.tetra_templates}
    directional_sites_by_bond = _directional_sites_by_carrier_bond(prepared)
    _validate_atom_occurrence_records(
        stereo_state,
        atom_ids,
        tetra_by_center,
    )
    _validate_bond_occurrence_records(
        stereo_state,
        atom_ids,
        bond_ids,
        prepared,
        directional_sites_by_bond,
    )
    _validate_local_order_records(
        prepared,
        stereo_state,
        occurrence_by_id,
        atom_ids,
        tetra_by_center,
    )
    try:
        cache_key = None
        if ring_state is not None:
            cache_key = (stereo_state, ring_state)
        if (
            stereo_residual_cache is not None
            and cache_key is not None
            and cache_key in stereo_residual_cache
        ):
            expected_residual = stereo_residual_cache[cache_key]
        else:
            expected_residual = reconstruct_writer_stereo_residual_snapshot(
                prepared,
                stereo_state,
                ring_state=ring_state,
            )
            if stereo_residual_cache is not None and cache_key is not None:
                stereo_residual_cache[cache_key] = expected_residual
    except (ValueError, SouthStarError) as exc:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "writer stereo history does not define a valid residual state",
        ) from exc

    if stereo_state.residual_snapshot != expected_residual:
        _invalid_snapshot(
            "writer residual snapshot does not match stereo event history"
        )


def _validate_unique_stereo_records(stereo_state: WriterStereoStateKey) -> None:
    _reject_duplicate_items(
        (record.atom for record in stereo_state.atom_occurrences),
        "writer atom occurrence records contain duplicates",
    )
    _reject_duplicate_items(
        ((record.bond, record.parent, record.child) for record in stereo_state.bond_occurrences),
        "writer bond occurrence records contain duplicate orientations",
    )
    _reject_duplicate_items(
        (record.bond for record in stereo_state.bond_occurrences),
        "writer bond occurrence records contain duplicate bonds",
    )
    _reject_duplicate_items(
        (record.atom for record in stereo_state.local_orders),
        "writer local-order records contain duplicate atoms",
    )
    _reject_duplicate_items(
        stereo_state.residual_snapshot.factors,
        "writer residual factor snapshots contain duplicates",
    )
    _reject_duplicate_items(
        (var for var, _ in stereo_state.residual_snapshot.domains),
        "writer residual domains contain duplicate variables",
    )
    _reject_duplicate_items(
        (var for var, _ in stereo_state.residual_snapshot.assignments),
        "writer residual assignments contain duplicate variables",
    )


def _validate_atom_occurrence_records(
    stereo_state: WriterStereoStateKey,
    atom_ids: frozenset[AtomId],
    tetra_by_center,
) -> None:
    for record in stereo_state.atom_occurrences:
        if record.atom not in atom_ids:
            _invalid_snapshot("writer atom occurrence references unknown atom")
        template = tetra_by_center.get(record.atom)
        if template is None:
            if record.token is not TetraToken.NONE:
                _invalid_snapshot("writer atom occurrence has unexpected tetra token")
            continue
        if template.status is SiteStatus.UNSPECIFIED and record.token is not TetraToken.NONE:
            _invalid_snapshot("writer unspecified tetra occurrence has token")
        if template.status is SiteStatus.SPECIFIED and record.token not in {
            TetraToken.AT,
            TetraToken.ATAT,
        }:
            _invalid_snapshot("writer specified tetra occurrence lacks token")


def _validate_bond_occurrence_records(
    stereo_state: WriterStereoStateKey,
    atom_ids: frozenset[AtomId],
    bond_ids: frozenset[BondId],
    prepared: SouthStarPreparedMol,
    directional_sites_by_bond: dict[BondId, tuple[SiteId, ...]],
) -> None:
    for record in stereo_state.bond_occurrences:
        if record.bond not in bond_ids or record.parent not in atom_ids or record.child not in atom_ids:
            _invalid_snapshot("writer bond occurrence references unknown graph item")
        _require_graph_bond(prepared, record.parent, record.child, record.bond)
        eligible_sites = directional_sites_by_bond.get(record.bond, ())
        if not eligible_sites and record.mark is not DirectionMark.ABSENT:
            _invalid_snapshot("writer bond occurrence has unexpected direction mark")


def _validate_local_order_records(
    prepared: SouthStarPreparedMol,
    stereo_state: WriterStereoStateKey,
    occurrence_by_id,
    atom_ids: frozenset[AtomId],
    tetra_by_center,
) -> None:
    for record in stereo_state.local_orders:
        if record.atom not in atom_ids:
            _invalid_snapshot("writer local-order record references unknown atom")
        if len(set(record.order)) != len(record.order):
            _invalid_snapshot("writer local-order record repeats ligand occurrence")
        template = tetra_by_center.get(record.atom)
        allowed = _allowed_local_order_occurrences(prepared, record.atom, template)
        for occurrence_id in record.order:
            occurrence = occurrence_by_id.get(occurrence_id)
            if occurrence is None:
                _invalid_snapshot("writer local-order record references unknown ligand occurrence")
            if occurrence_id not in allowed:
                _invalid_snapshot("writer local-order occurrence belongs to another site")
            if occurrence.kind is LigandKind.IMPLICIT_H:
                if occurrence.atom != record.atom:
                    _invalid_snapshot("writer local-order implicit-H occurrence is on another atom")
            elif occurrence.kind is LigandKind.NEIGHBOR_ATOM:
                if occurrence.atom not in atom_ids or occurrence.bond is None:
                    _invalid_snapshot("writer local-order neighbor occurrence references unknown atom")
                _require_graph_bond(prepared, record.atom, occurrence.atom, occurrence.bond)
            else:
                _invalid_snapshot("writer local-order pseudo occurrence is unsupported")
        if record.closed and template is not None:
            if set(record.order) != set(template.ligand_occurrences):
                _invalid_snapshot("writer closed tetra local order is incomplete")


def _directional_reference_pair(template) -> tuple[OccurrenceId, OccurrenceId]:
    if template.reference_pair is not None:
        return template.reference_pair
    return (min(template.left_ligands, key=int), min(template.right_ligands, key=int))


def _neighbor_ligands_by_bond(
    occurrence_by_id,
    ligand_ids: tuple[OccurrenceId, ...],
) -> dict[BondId, OccurrenceId]:
    out = {}
    for ligand_id in ligand_ids:
        occurrence = occurrence_by_id[ligand_id]
        if occurrence.kind is not LigandKind.NEIGHBOR_ATOM:
            continue
        if occurrence.bond is None:
            _invalid_snapshot("writer directional neighbor occurrence lacks bond")
        out[occurrence.bond] = ligand_id
    return out


def _directional_template_substituent_bonds(
    prepared: SouthStarPreparedMol,
    template,
) -> frozenset[BondId]:
    occurrence_by_id = {item.id: item for item in prepared.facts.ligand_occurrences}
    bonds: set[BondId] = set()
    for occurrence_id in template.left_ligands + template.right_ligands:
        occurrence = occurrence_by_id[occurrence_id]
        if occurrence.kind is not LigandKind.NEIGHBOR_ATOM:
            continue
        if occurrence.bond is None:
            _invalid_snapshot("writer directional neighbor occurrence lacks bond")
        bonds.add(occurrence.bond)
    return frozenset(bonds)


def _directional_sites_by_carrier_bond(
    prepared: SouthStarPreparedMol,
) -> dict[BondId, tuple[SiteId, ...]]:
    by_bond: dict[BondId, list[SiteId]] = {}
    for template in prepared.directional_templates:
        for bond in _directional_template_substituent_bonds(prepared, template):
            by_bond.setdefault(bond, []).append(template.site)
    return {
        bond: tuple(sorted(sites, key=int))
        for bond, sites in by_bond.items()
    }


def _allowed_local_order_occurrences(
    prepared: SouthStarPreparedMol,
    atom: AtomId,
    template,
) -> frozenset[OccurrenceId]:
    if template is not None:
        return frozenset(template.ligand_occurrences)
    return frozenset(
        occurrence.id
        for occurrence in prepared.facts.ligand_occurrences
        if occurrence.kind is LigandKind.IMPLICIT_H and occurrence.atom == atom
    )


def _local_order_record(
    stereo_state: WriterStereoStateKey,
    atom: AtomId,
):
    for record in stereo_state.local_orders:
        if record.atom == atom:
            return record
    return None


def _has_bond_occurrence_record(
    stereo_state: WriterStereoStateKey,
    bond: BondId,
    parent: AtomId,
    child: AtomId,
) -> bool:
    return any(
        record.bond == bond
        and record.parent == parent
        and record.child == child
        for record in stereo_state.bond_occurrences
    )


def _reject_duplicate_items(items, message: str) -> None:
    seen = set()
    for item in items:
        if item in seen:
            _invalid_snapshot(message)
        seen.add(item)


def _require_graph_bond(
    prepared: SouthStarPreparedMol,
    left: AtomId,
    right: AtomId,
    bond: BondId,
) -> None:
    actual = prepared.graph_index.bond_between.get((min(left, right), max(left, right)))
    if actual != bond:
        _invalid_snapshot("writer state contains graph-invalid atom/bond triple")


def _atom_component_index(prepared: SouthStarPreparedMol) -> dict[AtomId, int]:
    out: dict[AtomId, int] = {}
    for index, component in enumerate(prepared.facts.components):
        for atom in component.atoms:
            out[atom] = index
    return out


def _bond_component_index(prepared: SouthStarPreparedMol) -> dict[BondId, int]:
    out: dict[BondId, int] = {}
    for index, component in enumerate(prepared.facts.components):
        for bond in component.bonds:
            out[bond] = index
    return out


def _invalid_snapshot(message: str) -> None:
    raise SouthStarError(SouthStarErrorKind.INTERNAL_INVARIANT, message)


def _prepared_identity(
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions,
) -> WriterPreparedIdentity:
    return WriterPreparedIdentity(
        runtime=(
            runtime_options.serialization_language.value,
            runtime_options.rooted_at_atom,
            runtime_options.canonical,
            runtime_options.do_random,
        ),
        atoms=tuple(
            (
                int(atom.id),
                atom.atomic_num,
                atom.symbol,
                atom.isotope,
                atom.formal_charge,
                atom.is_aromatic,
                atom.explicit_h_count,
                atom.implicit_h_count,
                atom.no_implicit,
            )
            for atom in prepared.facts.atoms
        ),
        bonds=tuple(
            (
                int(bond.id),
                int(bond.a),
                int(bond.b),
                bond.order.value,
                bond.is_aromatic,
                bond.is_conjugated,
            )
            for bond in prepared.facts.bonds
        ),
        components=tuple(
            (
                int(component.id),
                tuple(int(atom) for atom in component.atoms),
                tuple(int(bond) for bond in component.bonds),
            )
            for component in prepared.facts.components
        ),
        ligand_occurrences=tuple(
            (
                int(occurrence.id),
                int(occurrence.site),
                occurrence.kind.value,
                None if occurrence.atom is None else int(occurrence.atom),
                None if occurrence.bond is None else int(occurrence.bond),
                occurrence.ordinal,
            )
            for occurrence in prepared.facts.ligand_occurrences
        ),
        tetra_templates=tuple(
            (
                int(template.site),
                int(template.center),
                template.status.value,
                template.target.value,
                tuple(int(item) for item in template.reference_order),
                tuple(int(item) for item in template.ligand_occurrences),
            )
            for template in prepared.tetra_templates
        ),
        directional_templates=tuple(
            (
                int(template.site),
                int(template.center_bond),
                int(template.left_endpoint),
                int(template.right_endpoint),
                template.status.value,
                template.target.value,
                tuple(int(item) for item in template.left_ligands),
                tuple(int(item) for item in template.right_ligands),
                None
                if template.reference_pair is None
                else tuple(int(item) for item in template.reference_pair),
            )
            for template in prepared.directional_templates
        ),
        policy=(
            tuple(int(label.value) for label in prepared.policy.ring_labels),
            prepared.policy.annotation_mode.value,
            prepared.policy.least_free_ring_labels,
            tuple(
                (
                    int(domain.atom),
                    tuple(
                        (
                            choice.name,
                            tuple((token.value, text) for token, text in choice.text_by_tetra),
                        )
                        for choice in domain.choices
                    ),
                )
                for domain in prepared.policy.atom_text_domains
            ),
            tuple(
                (
                    int(domain.bond),
                    domain.slot_kind,
                    tuple(
                        (choice.name, choice.base_text, choice.permits_direction)
                        for choice in domain.choices
                    ),
                )
                for domain in prepared.policy.bond_text_domains
            ),
        ),
    )


__all__ = (
    "WriterDecoderBoundary",
    "WriterFrontierFrame",
    "WriterPreparedIdentity",
    "WriterSearchSnapshot",
    "WriterSnapshotFrame",
    "capture_writer_frontier_snapshot",
    "capture_initial_writer_frontier_snapshot",
    "advance_writer_frontier_snapshot",
    "resume_writer_frontier_choices_from_snapshot",
    "validate_writer_cursor_against_prepared",
    "validate_writer_search_snapshot",
    "writer_frontier_cursor_from_snapshot",
)
