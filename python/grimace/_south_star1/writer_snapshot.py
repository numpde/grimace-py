"""Writer-shaped frontier snapshots."""

from __future__ import annotations

from collections.abc import Iterator
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
from .residual_constraints import VarId
from .residual_constraints import directional_site_carrier_var
from .residual_constraints import tetra_parity_var
from .residual_constraints import tetra_token_var
from .writer_graph_obligations import WriterBoundaryOwnerKind
from .writer_graph_obligations import WriterEdgeObligationKind
from .writer_graph_obligations import WriterGraphObligationContext
from .writer_graph_obligations import WriterGraphObligationSummary
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
from .writer_frontier import initial_writer_frontier_cursor
from .writer_frontier import iter_writer_frontier_support
from .writer_stereo import _writer_stereo_relation_definitions
from .writer_state import ComponentCursor
from .writer_state import ObligationStateKey
from .writer_state import PendingEntryPhase
from .writer_state import PendingWriterEntry
from .writer_state import WriterAtomFrame
from .writer_state import WriterBranchFrame
from .writer_state import WriterStateKey
from .writer_state import WriterStereoStateKey
from .writer_transitions import _WriterExecutionCapabilityKind


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

    if _prepared_has_cyclic_writer_graph_surface(prepared):
        cursor = _initial_writer_transition_frontier_cursor(
            prepared,
            runtime_options,
        )
        decision = _cyclic_writer_admission_decision_from_cursor(
            prepared=prepared,
            runtime_options=runtime_options,
            cursor=cursor,
        )
        _assert_cyclic_writer_admission_decision(decision)
        return cursor

    return initial_writer_frontier_cursor(prepared, runtime_options)


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
    next_emitted_text: str
    source_cursor: WriterFrontierCursor
    successor_cursor: WriterFrontierCursor


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


class _WriterCyclicAdmissionDecisionKind(Enum):
    READY_PUBLIC = "ready_public"
    READY_BUT_PUBLIC_CLOSED = "ready_but_public_closed"
    BLOCKED_PUBLIC_CYCLIC_PROFILE = "blocked_public_cyclic_profile"
    BLOCKED_RESIDUAL_CYCLIC_POLICY = "blocked_residual_cyclic_policy"
    TRUNCATED_READINESS_AUDIT = "truncated_readiness_audit"


class _WriterPublicCyclicOpeningProfileKind(Enum):
    SUPPORTED_SIMPLE_MONOCYCLE_COMPONENT = (
        "supported_simple_monocycle_component"
    )
    SUPPORTED_SIMPLE_MONOCYCLE_WITH_ACYCLIC_ATTACHMENTS = (
        "supported_simple_monocycle_with_acyclic_attachments"
    )
    BLOCKED_NOT_SINGLE_COMPONENT = "blocked_not_single_component"
    BLOCKED_NOT_CONNECTED_COMPONENT = "blocked_not_connected_component"
    BLOCKED_NOT_CYCLIC_COMPONENT = "blocked_not_cyclic_component"
    BLOCKED_UNSUPPORTED_CYCLIC_RANK = "blocked_unsupported_cyclic_rank"
    BLOCKED_UNSUPPORTED_BRANCHING = "blocked_unsupported_branching"
    BLOCKED_UNSUPPORTED_CLOSURE_BOND_SURFACE = (
        "blocked_unsupported_closure_bond_surface"
    )
    BLOCKED_UNSUPPORTED_CYCLIC_STEREO_SURFACE = (
        "blocked_unsupported_cyclic_stereo_surface"
    )


class _WriterPublicCyclicRequiredCapability(Enum):
    SIMPLE_CYCLE_CORE_CLOSURE = "simple_cycle_core_closure"
    ACYCLIC_PENDANT_TREE_TRAVERSAL = "acyclic_pendant_tree_traversal"
    TREE_BOND_TEXT_EMISSION = "tree_bond_text_emission"
    RING_CORE_NON_SINGLE_CLOSURE_BOND = "ring_core_non_single_closure_bond"
    CYCLIC_DIRECTIONAL_STEREO = "cyclic_directional_stereo"
    CYCLIC_RING_PAIR_STEREO = "cyclic_ring_pair_stereo"
    MULTI_CYCLE_TOPOLOGY = "multi_cycle_topology"
    FUSED_OR_BRIDGED_TOPOLOGY = "fused_or_bridged_topology"
    NON_FOREST_PENDANT_MATERIAL = "non_forest_pendant_material"
    MULTI_BOUNDARY_PENDANT_COMPONENT = "multi_boundary_pendant_component"


_PUBLIC_CYCLIC_SUPPORTED_CAPABILITIES = frozenset(
    {
        _WriterPublicCyclicRequiredCapability.SIMPLE_CYCLE_CORE_CLOSURE,
        _WriterPublicCyclicRequiredCapability.ACYCLIC_PENDANT_TREE_TRAVERSAL,
        _WriterPublicCyclicRequiredCapability.TREE_BOND_TEXT_EMISSION,
    }
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
            }
            and not self.unsupported_capabilities
            and self.required_capabilities.issubset(
                _PUBLIC_CYCLIC_SUPPORTED_CAPABILITIES
            )
        )


@dataclass(frozen=True, slots=True)
class _WriterCyclicAdmissionDecision:
    kind: _WriterCyclicAdmissionDecisionKind
    readiness_gate: _WriterResidualCyclicReadinessGate
    public_profile: _WriterPublicCyclicOpeningProfileReport | None = None

    def __post_init__(self) -> None:
        if (
            self.kind
            is _WriterCyclicAdmissionDecisionKind.READY_PUBLIC
        ):
            valid = self.readiness_gate.ready and bool(
                self.public_profile is not None and self.public_profile.supported
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
                self.readiness_gate.ready
                and self.public_profile is not None
                and not self.public_profile.supported
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
            str,
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

        for choice in choice_snapshot.choices:
            for capability in choice.execution_capabilities:
                use = _WriterExecutionCapabilityUse(
                    kind=capability,
                    emitted_texts=prefix,
                    next_emitted_text=choice.emitted_text,
                    source_cursor=current.cursor,
                    successor_cursor=choice.successor,
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
    ring_core_max_degree = max(
        (
            len(adjacency[atom])
            for atom in component_core_atoms
            if atom in adjacency
        ),
        default=0,
    )
    ring_core_unsupported_bond_count = sum(
        1
        for bond_id in ring_core_bond_ids
        if (
            (bond := component_bond_index.get(bond_id))
            is not None
            and bond.order is not BondOrder.SINGLE
        )
    )
    ring_core_bond_id_set = frozenset(ring_core_bond_ids)
    pendant_bond_ids = tuple(
        bond_id for bond_id in component_bond_ids if bond_id not in ring_core_bond_id_set
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
    pendant_atom_set = set(component_atom_ids) - ring_core_atom_set
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

    if cyclic_ranks != (1,):
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

    if ring_core_unsupported_bond_count:
        unsupported_capabilities.add(
            _WriterPublicCyclicRequiredCapability.RING_CORE_NON_SINGLE_CLOSURE_BOND,
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
        kind = (
            _WriterPublicCyclicOpeningProfileKind.BLOCKED_UNSUPPORTED_CYCLIC_STEREO_SURFACE
        )
    else:
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

            if pendant_multi_boundary:
                unsupported_capabilities.add(
                    _WriterPublicCyclicRequiredCapability.MULTI_BOUNDARY_PENDANT_COMPONENT,
                )
            if pendant_non_tree:
                unsupported_capabilities.add(
                    _WriterPublicCyclicRequiredCapability.NON_FOREST_PENDANT_MATERIAL,
                )

        if ring_core_is_simple_cycle and pendant_atom_count == 0:
            kind = (
                _WriterPublicCyclicOpeningProfileKind
                .SUPPORTED_SIMPLE_MONOCYCLE_COMPONENT
            )
        elif (
            ring_core_is_simple_cycle
            and pendant_atom_count > 0
            and not pendant_multi_boundary
            and not pendant_non_tree
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
            if not ring_core_is_simple_cycle:
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
            return _WriterCyclicAdmissionDecision(
                kind=_WriterCyclicAdmissionDecisionKind.READY_PUBLIC,
                readiness_gate=gate,
                public_profile=profile,
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
    validate_writer_cursor_against_prepared(
        prepared,
        snapshot.cursor,
        runtime_options=snapshot.runtime_options,
    )


def validate_writer_cursor_against_prepared(
    prepared: SouthStarPreparedMol,
    cursor: WriterFrontierCursor,
    *,
    runtime_options: SouthStarRuntimeOptions | None = None,
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
        _validate_stereo_occurrences_bound_to_graph_state(prepared, key)
        _validate_ring_state(prepared, key, context)
        _validate_policy_state(key, atom_ids, bond_ids)
        _validate_stereo_state(prepared, key.stereo_state)


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
    bond_occurrence_bonds = frozenset(
        record.bond for record in key.stereo_state.bond_occurrences
    )
    if bond_occurrence_bonds != frozenset(expected_bonds):
        _invalid_snapshot("writer bond occurrences do not cover emitted bonds")
    parent_by_child = _written_tree_parent_links(prepared, key)
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
            expected = parent_by_child.get(record.child)
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
        _invalid_snapshot("writer bond occurrence is not backed by emitted graph state")


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
    _validate_live_residual_stereo_coverage(
        prepared,
        stereo_state,
        tetra_by_center,
        directional_sites_by_bond,
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


def _validate_live_residual_stereo_coverage(
    prepared: SouthStarPreparedMol,
    stereo_state: WriterStereoStateKey,
    tetra_by_center,
    directional_sites_by_bond: dict[BondId, tuple[SiteId, ...]],
) -> None:
    expected_factors = _expected_live_residual_stereo_factor_snapshots(
        prepared,
        stereo_state,
    )
    actual_factors = {
        snapshot.key: snapshot
        for snapshot in stereo_state.residual_snapshot.factors
    }
    if actual_factors != expected_factors:
        _invalid_snapshot("writer live residual factors do not match obligations")

    assignment_vars = _residual_assignment_vars(stereo_state)
    domain_vars = _residual_domain_vars(stereo_state)
    factor_vars = frozenset(
        var
        for snapshot in stereo_state.residual_snapshot.factors
        for var in snapshot.scope
    )
    if domain_vars != factor_vars:
        _invalid_snapshot("writer residual domains must be exactly covered by live factors")
    if not assignment_vars.issubset(domain_vars):
        _invalid_snapshot("writer residual assignment lacks live residual domain")
    for var in domain_vars:
        if var.kind == "tetra_token":
            site = SiteId(int(var.key[0]))
            template = next((item for item in prepared.tetra_templates if item.site == site), None)
            if template is None:
                _invalid_snapshot("writer live tetra token variable references unknown site")
            if _local_order_record(stereo_state, template.center) is not None and _local_order_record(stereo_state, template.center).closed:
                _invalid_snapshot("writer closed tetra site still has live token variable")
            if var in assignment_vars and not any(
                record.atom == template.center
                for record in stereo_state.atom_occurrences
            ):
                _invalid_snapshot("writer tetra token assignment lacks atom occurrence")
        elif var.kind == "tetra_local_parity":
            site = SiteId(int(var.key[0]))
            template = next((item for item in prepared.tetra_templates if item.site == site), None)
            if template is None:
                _invalid_snapshot("writer live tetra parity variable references unknown site")
            record = _local_order_record(stereo_state, template.center)
            if record is not None and record.closed:
                _invalid_snapshot("writer closed tetra site still has live parity variable")
        elif var.kind == "directional_site_carrier":
            site = SiteId(int(var.key[0]))
            bond = BondId(int(var.key[1]))
            if site not in directional_sites_by_bond.get(bond, ()):
                _invalid_snapshot("writer directional carrier variable has wrong site/bond")
            if var in assignment_vars and not any(
                record.bond == bond
                for record in stereo_state.bond_occurrences
            ):
                _invalid_snapshot("writer directional assignment lacks bond occurrence")
        else:
            _invalid_snapshot("writer residual variable has unknown live stereo kind")


def _expected_live_residual_stereo_factor_snapshots(
    prepared: SouthStarPreparedMol,
    stereo_state: WriterStereoStateKey,
) -> dict[object, object]:
    _domains, factors = _writer_stereo_relation_definitions(prepared)
    emitted_bonds = frozenset(record.bond for record in stereo_state.bond_occurrences)
    tetra_by_site = {template.site: template for template in prepared.tetra_templates}

    expected: dict[object, object] = {}
    for factor in factors:
        key = factor.key
        if key.kind == "tetra_site":
            site = SiteId(int(key.key[0]))
            template = tetra_by_site.get(site)
            if template is None:
                _invalid_snapshot("writer tetra factor references unknown site")
            record = _local_order_record(stereo_state, template.center)
            if record is not None and record.closed:
                continue
        elif key.kind == "directional_bond_emission":
            bond = BondId(int(key.key[0]))
            if bond in emitted_bonds:
                continue
        elif key.kind == "directional_site":
            carrier_bonds = frozenset(
                BondId(int(var.key[1]))
                for var in factor.scope
            )
            if carrier_bonds.issubset(emitted_bonds):
                continue
        else:
            _invalid_snapshot("writer residual factor has unknown live stereo kind")

        expected[key] = factor.value_snapshot()

    return expected


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


def _var_sort_tuple(var: VarId) -> tuple[object, ...]:
    return (var.kind, tuple(_value_sort_tuple(item) for item in var.key))


def _value_sort_tuple(value: object) -> tuple[object, ...]:
    if isinstance(value, (int, str)):
        return (type(value).__name__, value)
    if isinstance(value, (TetraToken, DirectionMark)):
        return (value.__class__.__name__, value.value)
    if isinstance(value, tuple):
        return ("tuple", tuple(_value_sort_tuple(item) for item in value))
    return (value.__class__.__name__, str(value))


def _residual_domain_vars(stereo_state: WriterStereoStateKey) -> frozenset[VarId]:
    return frozenset(var for var, _ in stereo_state.residual_snapshot.domains)


def _residual_assignment_vars(stereo_state: WriterStereoStateKey) -> frozenset[VarId]:
    return frozenset(var for var, _ in stereo_state.residual_snapshot.assignments)


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
