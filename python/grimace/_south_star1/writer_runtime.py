"""Public facade for the writer-shaped live runtime."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterator
from dataclasses import dataclass

from .prepared_runtime import SouthStarPreparedMol
from .prepared_runtime import SouthStarRuntimeOptions
from .writer_capabilities import _unsupported_public_writer_execution_capabilities
from .writer_execution_evidence import writer_finite_relation_work_envelope_violation
from .writer_execution_evidence import writer_graph_obligation_work_envelope_violation
from .writer_execution_evidence import writer_residual_work_envelope_violation
from .writer_frontier import WriterFrontierChoice
from .writer_frontier import WriterFrontierChoices
from .writer_frontier import WriterFrontierCursor
from .writer_frontier import WriterFrontierTerminal
from .writer_frontier import _checked_writer_frontier_branch_supports
from .writer_frontier import _count_checked_writer_frontier_branch_completions
from .writer_snapshot import WriterDecoderBoundary
from .writer_snapshot import WriterSearchSnapshot
from .writer_snapshot import _count_writer_frontier_support_after_emitted_texts
from .writer_snapshot import _iter_writer_frontier_support_suffixes_after_emitted_texts
from .writer_snapshot import _writer_frontier_choice_snapshot_from_snapshot
from .writer_snapshot import _writer_search_snapshot_with_cursor_after_emitted_text
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
    next_state: WriterRuntimeState


@dataclass(frozen=True, slots=True)
class WriterRuntimeBranchTransitions:
    choices: WriterFrontierChoices
    transitions: tuple[WriterRuntimeBranchTransition, ...]

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
    choice_snapshot = _writer_frontier_choice_snapshot_from_snapshot(
        state.snapshot,
        prepared=prepared,
        include_counts=False,
        stop_after_first_blocked=False,
    )
    return WriterRuntimeDiagnostics(
        blocked=choice_snapshot.blocked,
        graph_policy_blockers=choice_snapshot.graph_policy_blockers,
        stereo_policy_blockers=choice_snapshot.stereo_policy_blockers,
        execution_capabilities=choice_snapshot.execution_capabilities,
        terminal_execution_capabilities=choice_snapshot.terminal_execution_capabilities,
        unsupported_execution_capabilities=(
            _unsupported_public_writer_execution_capabilities(
                choice_snapshot.execution_capabilities,
            )
        ),
        unsupported_terminal_execution_capabilities=(
            _unsupported_public_writer_execution_capabilities(
                choice_snapshot.terminal_execution_capabilities,
            )
        ),
        residual_work_evidence=choice_snapshot.residual_work_evidence,
        terminal_residual_work_evidence=choice_snapshot.terminal_residual_work_evidence,
        finite_relation_work_evidence=choice_snapshot.finite_relation_work_evidence,
        graph_obligation_work_evidence=choice_snapshot.graph_obligation_work_evidence,
        residual_work_envelope_violations=_writer_work_envelope_violations(
            choice_snapshot.residual_work_evidence,
            writer_residual_work_envelope_violation,
        ),
        terminal_residual_work_envelope_violations=(
            _writer_work_envelope_violations(
                choice_snapshot.terminal_residual_work_evidence,
                writer_residual_work_envelope_violation,
            )
        ),
        finite_relation_work_envelope_violations=(
            _writer_work_envelope_violations(
                choice_snapshot.finite_relation_work_evidence,
                writer_finite_relation_work_envelope_violation,
            )
        ),
        graph_obligation_work_envelope_violations=(
            _writer_work_envelope_violations(
                choice_snapshot.graph_obligation_work_evidence,
                writer_graph_obligation_work_envelope_violation,
            )
        ),
        choice_texts=tuple(choice.emitted_text for choice in choice_snapshot.choices),
        has_eos=choice_snapshot.terminal is not None,
    )


def _writer_work_envelope_violations(
    evidence: tuple[object, ...],
    violation_check: Callable[[object], object | None],
) -> tuple[object, ...]:
    return tuple(
        violation
        for item in evidence
        if (violation := violation_check(item)) is not None
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
                next_state=_writer_runtime_state_after_checked_choice(
                    prepared=prepared,
                    state=state,
                    choice=choice,
                ),
            )
            for choice in branch_batch.choices.choices
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
        successor_state = _writer_runtime_state_for_successor_key(
            prepared=prepared,
            state=state,
            successor_state=support.successor_state,
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
                next_state=successor_state,
            )
        )
    return WriterRuntimeBranchTransitions(
        choices=branch_batch.choices,
        transitions=tuple(branch_transitions),
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
        _writer_search_snapshot_with_cursor_after_emitted_text(
            state.snapshot,
            prepared=prepared,
            cursor=choice.successor,
        )
    )


def _writer_runtime_state_for_successor_key(
    *,
    prepared: SouthStarPreparedMol,
    state: WriterRuntimeState,
    successor_state: object,
) -> WriterRuntimeState:
    return WriterRuntimeState(
        _writer_search_snapshot_with_cursor_after_emitted_text(
            state.snapshot,
            prepared=prepared,
            cursor=WriterFrontierCursor(weighted_states=((successor_state, 1),)),
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
    validate_writer_search_snapshot(state.snapshot, prepared=prepared)
    return _count_checked_writer_frontier_branch_completions(
        prepared,
        state.snapshot.cursor,
    )


def iter_writer_runtime_support(
    *,
    prepared: SouthStarPreparedMol,
    state: WriterRuntimeState,
) -> Iterator[str]:
    return _iter_writer_frontier_support_suffixes_after_emitted_texts(
        state.snapshot,
        prepared=prepared,
        emitted_texts=(),
    )


__all__ = (
    "WriterRuntimeBranchTransition",
    "WriterRuntimeBranchTransitions",
    "WriterRuntimeChoiceTransition",
    "WriterRuntimeChoiceTransitions",
    "WriterRuntimeDiagnostics",
    "WriterRuntimeState",
    "advance_writer_runtime_state",
    "count_writer_runtime_branch_completions",
    "count_writer_runtime_completions",
    "count_writer_runtime_support",
    "initial_writer_runtime_state",
    "iter_writer_runtime_support",
    "writer_runtime_branch_transitions",
    "writer_runtime_choice_transitions",
    "writer_runtime_choices",
    "writer_runtime_diagnostics",
    "writer_runtime_has_eos",
    "writer_runtime_state_from_snapshot",
    "writer_runtime_terminal",
)
