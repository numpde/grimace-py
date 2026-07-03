"""Public facade for the writer-shaped live runtime."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from .prepared_runtime import SouthStarPreparedMol
from .prepared_runtime import SouthStarRuntimeOptions
from .writer_frontier import WriterFrontierChoice
from .writer_frontier import WriterFrontierChoices
from .writer_frontier import WriterFrontierTerminal
from .writer_frontier import _checked_writer_frontier_branch_supports
from .writer_frontier import (
    _checked_writer_frontier_branch_completion_count_certificate,
)
from .writer_frontier import _writer_frontier_diagnostics
from .writer_snapshot import WriterDecoderBoundary
from .writer_snapshot import WriterSearchSnapshot
from .writer_snapshot import _count_writer_frontier_support_after_emitted_texts
from .writer_snapshot import _iter_writer_snapshot_certified_support_strings
from .writer_snapshot import _writer_search_snapshot_after_checked_branch_support
from .writer_snapshot import _writer_search_snapshot_after_checked_choice
from .writer_snapshot import _writer_search_snapshot_after_certified_emitted_text
from .writer_snapshot import advance_writer_frontier_snapshot
from .writer_snapshot import capture_initial_writer_frontier_snapshot
from .writer_snapshot import validate_writer_search_snapshot


@dataclass(frozen=True, slots=True)
class WriterRuntimeState:
    snapshot: WriterSearchSnapshot


@dataclass(frozen=True, slots=True)
class WriterRuntimeChoiceTransition:
    choice: WriterFrontierChoice
    next_state: WriterRuntimeState
    snapshot_step_certificate: object | None = None


@dataclass(frozen=True, slots=True)
class WriterRuntimeChoiceTransitions:
    choices: WriterFrontierChoices
    transitions: tuple[WriterRuntimeChoiceTransition, ...]

    @property
    def terminal(self) -> WriterFrontierTerminal | None:
        return self.choices.terminal

    @property
    def support_count(self) -> int:
        total = sum(
            transition.choice.support_count or 0
            for transition in self.transitions
        )
        if self.terminal is not None:
            total += self.terminal.support_count
        return total

    @property
    def completion_count(self) -> int:
        total = sum(
            transition.choice.completion_count or 0
            for transition in self.transitions
        )
        if self.terminal is not None:
            total += self.terminal.completion_count
        return total

    @property
    def has_eos(self) -> bool:
        return self.terminal is not None


@dataclass(frozen=True, slots=True)
class WriterRuntimeCertifiedSupportString:
    string: str
    certificate: object


@dataclass(frozen=True, slots=True)
class WriterRuntimeBranchTransition:
    emitted_text: str
    source_state: object
    successor_state: object
    parent_weight: int
    branch_ordinal: int
    transition_kind: object
    events: tuple[object, ...]
    evidence: object
    execution_capabilities: frozenset[object]
    residual_work_evidence: tuple[object, ...]
    finite_relation_work_evidence: tuple[object, ...]
    graph_obligation_work_evidence: tuple[object, ...]
    graph_action_surface: object | None
    policy_family: object | None
    closure_candidate_resolution_evidence: tuple[object, ...]
    closure_candidate_lifecycle_evidence: tuple[object, ...]
    closure_candidate_branch_certificates: tuple[object, ...]
    residual_attachment_lifecycle_evidence: tuple[object, ...]
    residual_attachment_branch_certificates: tuple[object, ...]
    stereo_lifecycle_evidence: tuple[object, ...]
    stereo_branch_certificates: tuple[object, ...]
    residual_attachment_policy_evidence: tuple[object, ...]
    checked_branch_certificate: object | None
    next_state: WriterRuntimeState


@dataclass(frozen=True, slots=True)
class WriterRuntimeTerminalSupport:
    source_state: object
    finalized_state: object
    parent_weight: int
    terminal_execution_capabilities: frozenset[object]
    terminal_residual_work_evidence: tuple[object, ...]
    terminal_stereo_lifecycle_evidence: tuple[object, ...]
    graph_obligation_work_evidence: tuple[object, ...]
    terminal_certificates: tuple[object, ...]
    checked_terminal_certificate: object | None


@dataclass(frozen=True, slots=True)
class WriterRuntimeBranchTransitions:
    choices: WriterFrontierChoices
    transitions: tuple[WriterRuntimeBranchTransition, ...]
    terminal_supports: tuple[WriterRuntimeTerminalSupport, ...] = ()
    text_choice_projection_certificates: tuple[object, ...] = ()
    terminal_projection_certificate: object | None = None

    @property
    def branch_transitions(self) -> tuple[WriterRuntimeBranchTransition, ...]:
        return self.transitions

    @property
    def terminal(self) -> WriterFrontierTerminal | None:
        return self.choices.terminal


@dataclass(frozen=True, slots=True)
class WriterRuntimeDiagnostics:
    blocked: bool
    graph_policy_blockers: tuple[object, ...]
    stereo_policy_blockers: tuple[object, ...]
    execution_capabilities: frozenset[object]
    terminal_execution_capabilities: frozenset[object]
    unsupported_execution_capabilities: frozenset[object]
    unsupported_terminal_execution_capabilities: frozenset[object]
    residual_work_evidence: tuple[object, ...]
    terminal_residual_work_evidence: tuple[object, ...]
    finite_relation_work_evidence: tuple[object, ...]
    graph_obligation_work_evidence: tuple[object, ...]
    residual_work_envelope_violations: tuple[object, ...]
    terminal_residual_work_envelope_violations: tuple[object, ...]
    finite_relation_work_envelope_violations: tuple[object, ...]
    graph_obligation_work_envelope_violations: tuple[object, ...]
    choice_texts: tuple[str, ...]
    has_eos: bool

    @property
    def all_execution_capabilities(self) -> frozenset[object]:
        return self.execution_capabilities | self.terminal_execution_capabilities

    @property
    def all_unsupported_execution_capabilities(self) -> frozenset[object]:
        return (
            self.unsupported_execution_capabilities
            | self.unsupported_terminal_execution_capabilities
        )

    @property
    def has_policy_blockers(self) -> bool:
        return bool(self.graph_policy_blockers or self.stereo_policy_blockers)

    @property
    def has_unsupported_execution_capabilities(self) -> bool:
        return bool(self.all_unsupported_execution_capabilities)

    @property
    def work_envelope_violations(self) -> tuple[object, ...]:
        return (
            *self.residual_work_envelope_violations,
            *self.terminal_residual_work_envelope_violations,
            *self.finite_relation_work_envelope_violations,
            *self.graph_obligation_work_envelope_violations,
        )

    @property
    def has_work_envelope_violations(self) -> bool:
        return bool(self.work_envelope_violations)


def initial_writer_runtime_state(
    *,
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions,
    decoder_boundary: WriterDecoderBoundary = WriterDecoderBoundary(),
) -> WriterRuntimeState:
    return WriterRuntimeState(
        capture_initial_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=runtime_options,
            decoder_boundary=decoder_boundary,
        )
    )


def writer_runtime_state_from_snapshot(
    snapshot: WriterSearchSnapshot,
    *,
    prepared: SouthStarPreparedMol,
) -> WriterRuntimeState:
    validate_writer_search_snapshot(snapshot, prepared=prepared)
    return WriterRuntimeState(snapshot)


def writer_runtime_diagnostics(
    *,
    prepared: SouthStarPreparedMol,
    state: WriterRuntimeState,
) -> WriterRuntimeDiagnostics:
    validate_writer_search_snapshot(state.snapshot, prepared=prepared)
    frontier = _writer_frontier_diagnostics(
        prepared=prepared,
        cursor=state.snapshot.cursor,
    )
    return WriterRuntimeDiagnostics(
        blocked=frontier.blocked,
        graph_policy_blockers=frontier.graph_policy_blockers,
        stereo_policy_blockers=frontier.stereo_policy_blockers,
        execution_capabilities=frontier.execution_capabilities,
        terminal_execution_capabilities=frontier.terminal_execution_capabilities,
        unsupported_execution_capabilities=frontier.unsupported_execution_capabilities,
        unsupported_terminal_execution_capabilities=(
            frontier.unsupported_terminal_execution_capabilities
        ),
        residual_work_evidence=frontier.residual_work_evidence,
        terminal_residual_work_evidence=frontier.terminal_residual_work_evidence,
        finite_relation_work_evidence=frontier.finite_relation_work_evidence,
        graph_obligation_work_evidence=frontier.graph_obligation_work_evidence,
        residual_work_envelope_violations=frontier.residual_work_envelope_violations,
        terminal_residual_work_envelope_violations=(
            frontier.terminal_residual_work_envelope_violations
        ),
        finite_relation_work_envelope_violations=(
            frontier.finite_relation_work_envelope_violations
        ),
        graph_obligation_work_envelope_violations=(
            frontier.graph_obligation_work_envelope_violations
        ),
        choice_texts=frontier.choice_texts,
        has_eos=frontier.has_eos,
    )


def writer_runtime_choices(
    *,
    prepared: SouthStarPreparedMol,
    state: WriterRuntimeState,
) -> WriterFrontierChoices:
    return writer_runtime_branch_transitions(
        prepared=prepared,
        state=state,
        include_counts=True,
    ).choices


def writer_runtime_choice_transitions(
    *,
    prepared: SouthStarPreparedMol,
    state: WriterRuntimeState,
) -> WriterRuntimeChoiceTransitions:
    branch_batch = writer_runtime_branch_transitions(
        prepared=prepared,
        state=state,
        include_counts=True,
    )
    return WriterRuntimeChoiceTransitions(
        choices=branch_batch.choices,
        transitions=tuple(
            WriterRuntimeChoiceTransition(
                choice=choice,
                next_state=next_state,
                snapshot_step_certificate=step_certificate,
            )
            for choice in branch_batch.choices.choices
            for next_state, step_certificate in (
                _writer_runtime_state_after_certified_choice_text(
                    prepared=prepared,
                    state=state,
                    choice=choice,
                ),
            )
        ),
    )


def writer_runtime_branch_transitions(
    *,
    prepared: SouthStarPreparedMol,
    state: WriterRuntimeState,
    include_counts: bool = True,
) -> WriterRuntimeBranchTransitions:
    validate_writer_search_snapshot(state.snapshot, prepared=prepared)
    branch_batch = _checked_writer_frontier_branch_supports(
        prepared,
        state.snapshot.cursor,
        include_counts=include_counts,
    )
    branch_transitions: list[WriterRuntimeBranchTransition] = []
    for support in branch_batch.supports:
        successor_state = _writer_runtime_state_after_checked_branch_support(
            prepared=prepared,
            state=state,
            support=support,
        )
        branch_transitions.append(
            WriterRuntimeBranchTransition(
                emitted_text=support.emitted_text,
                source_state=support.source_state,
                successor_state=support.successor_state,
                parent_weight=support.parent_weight,
                branch_ordinal=support.branch_ordinal,
                transition_kind=support.transition_kind,
                events=support.events,
                evidence=support.evidence,
                execution_capabilities=support.execution_capabilities,
                residual_work_evidence=support.residual_work_evidence,
                finite_relation_work_evidence=(
                    support.finite_relation_work_evidence
                ),
                graph_obligation_work_evidence=(
                    support.graph_obligation_work_evidence
                ),
                graph_action_surface=support.graph_action_surface,
                policy_family=support.policy_family,
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
                checked_branch_certificate=(
                    support.checked_branch_certificate
                ),
                next_state=successor_state,
            )
        )
    return WriterRuntimeBranchTransitions(
        choices=branch_batch.choices,
        transitions=tuple(branch_transitions),
        terminal_supports=tuple(
            WriterRuntimeTerminalSupport(
                source_state=support.source_state,
                finalized_state=support.finalized_state,
                parent_weight=support.parent_weight,
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
                checked_terminal_certificate=(
                    support.checked_terminal_certificate
                ),
            )
            for support in branch_batch.terminal_supports
        ),
        text_choice_projection_certificates=(
            branch_batch.text_choice_projection_certificates
        ),
        terminal_projection_certificate=(
            branch_batch.terminal_projection_certificate
        ),
    )


def writer_runtime_terminal(
    *,
    prepared: SouthStarPreparedMol,
    state: WriterRuntimeState,
) -> WriterFrontierTerminal | None:
    return writer_runtime_choices(prepared=prepared, state=state).terminal


def writer_runtime_has_eos(
    *,
    prepared: SouthStarPreparedMol,
    state: WriterRuntimeState,
) -> bool:
    return writer_runtime_terminal(prepared=prepared, state=state) is not None


def advance_writer_runtime_state(
    *,
    prepared: SouthStarPreparedMol,
    state: WriterRuntimeState,
    emitted_text: str,
) -> WriterRuntimeState:
    return WriterRuntimeState(
        advance_writer_frontier_snapshot(
            state.snapshot,
            prepared=prepared,
            emitted_text=emitted_text,
        )
    )


def _writer_runtime_state_after_checked_choice(
    *,
    prepared: SouthStarPreparedMol,
    state: WriterRuntimeState,
    choice: WriterFrontierChoice,
) -> WriterRuntimeState:
    return WriterRuntimeState(
        _writer_search_snapshot_after_checked_choice(
            state.snapshot,
            prepared=prepared,
            choice=choice,
        )
    )


def _writer_runtime_state_after_certified_choice_text(
    *,
    prepared: SouthStarPreparedMol,
    state: WriterRuntimeState,
    choice: WriterFrontierChoice,
) -> tuple[WriterRuntimeState, object]:
    snapshot, certificate = _writer_search_snapshot_after_certified_emitted_text(
        state.snapshot,
        prepared=prepared,
        emitted_text=choice.emitted_text,
    )
    return WriterRuntimeState(snapshot), certificate


def _writer_runtime_state_after_checked_branch_support(
    *,
    prepared: SouthStarPreparedMol,
    state: WriterRuntimeState,
    support,
) -> WriterRuntimeState:
    return WriterRuntimeState(
        _writer_search_snapshot_after_checked_branch_support(
            state.snapshot,
            prepared=prepared,
            support=support,
        )
    )


def count_writer_runtime_support(
    *,
    prepared: SouthStarPreparedMol,
    state: WriterRuntimeState,
) -> int:
    return _count_writer_frontier_support_after_emitted_texts(
        state.snapshot,
        prepared=prepared,
        emitted_texts=(),
    )


def count_writer_runtime_completions(
    *,
    prepared: SouthStarPreparedMol,
    state: WriterRuntimeState,
) -> int:
    return count_writer_runtime_branch_completions(
        prepared=prepared,
        state=state,
    )


def count_writer_runtime_branch_completions(
    *,
    prepared: SouthStarPreparedMol,
    state: WriterRuntimeState,
) -> int:
    return writer_runtime_branch_completion_count_certificate(
        prepared=prepared,
        state=state,
    ).completion_count


def writer_runtime_branch_completion_count_certificate(
    *,
    prepared: SouthStarPreparedMol,
    state: WriterRuntimeState,
):
    validate_writer_search_snapshot(state.snapshot, prepared=prepared)
    return _checked_writer_frontier_branch_completion_count_certificate(
        prepared,
        state.snapshot.cursor,
    )


def iter_writer_runtime_support(
    *,
    prepared: SouthStarPreparedMol,
    state: WriterRuntimeState,
) -> Iterator[str]:
    for item in iter_writer_runtime_certified_support(
        prepared=prepared,
        state=state,
    ):
        yield item.string


def iter_writer_runtime_certified_support(
    *,
    prepared: SouthStarPreparedMol,
    state: WriterRuntimeState,
) -> Iterator[WriterRuntimeCertifiedSupportString]:
    validate_writer_search_snapshot(state.snapshot, prepared=prepared)
    for item in _iter_writer_snapshot_certified_support_strings(
        state.snapshot,
        prepared=prepared,
    ):
        yield WriterRuntimeCertifiedSupportString(
            string=item.string,
            certificate=item.certificate,
        )


def writer_runtime_support_image_certificate(
    *,
    prepared: SouthStarPreparedMol,
    state: WriterRuntimeState,
    witness_count: int | None = None,
):
    certified = tuple(
        iter_writer_runtime_certified_support(
            prepared=prepared,
            state=state,
        )
    )
    count_certificate = writer_runtime_branch_completion_count_certificate(
        prepared=prepared,
        state=state,
    )
    if witness_count is None:
        witness_count = count_certificate.completion_count

    from .writer_support_certificates import writer_support_image_certificate

    return writer_support_image_certificate(
        source_snapshot=state.snapshot,
        string_certificates=tuple(item.certificate for item in certified),
        witness_count=witness_count,
        witness_count_certificate=count_certificate,
    )


__all__ = (
    "WriterRuntimeBranchTransition",
    "WriterRuntimeBranchTransitions",
    "WriterRuntimeChoiceTransition",
    "WriterRuntimeChoiceTransitions",
    "WriterRuntimeCertifiedSupportString",
    "WriterRuntimeDiagnostics",
    "WriterRuntimeState",
    "advance_writer_runtime_state",
    "count_writer_runtime_branch_completions",
    "count_writer_runtime_completions",
    "count_writer_runtime_support",
    "initial_writer_runtime_state",
    "iter_writer_runtime_support",
    "iter_writer_runtime_certified_support",
    "writer_runtime_branch_transitions",
    "writer_runtime_choice_transitions",
    "writer_runtime_choices",
    "writer_runtime_diagnostics",
    "writer_runtime_branch_completion_count_certificate",
    "writer_runtime_has_eos",
    "writer_runtime_state_from_snapshot",
    "writer_runtime_terminal",
    "writer_runtime_support_image_certificate",
)
