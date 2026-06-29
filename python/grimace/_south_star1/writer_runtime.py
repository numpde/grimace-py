"""Small public facade for the writer-shaped live runtime.

This module intentionally owns no support logic.  It names the runtime boundary
that public callers should use, while delegating every decision to the existing
checked snapshot/frontier operations.  Keeping this layer thin prevents a second
support authority from growing beside the live writer transition engine.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterator
from dataclasses import dataclass
from dataclasses import replace

from .errors import SouthStarError
from .errors import SouthStarErrorKind
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
from .writer_snapshot import WriterDecoderBoundary
from .writer_snapshot import WriterFrontierFrame
from .writer_snapshot import WriterSearchSnapshot
from .writer_snapshot import _checked_writer_frontier_choice_snapshot_from_snapshot
from .writer_snapshot import _count_writer_completions_after_emitted_texts
from .writer_snapshot import _count_writer_frontier_support_after_emitted_texts
from .writer_snapshot import _iter_writer_frontier_support_suffixes_after_emitted_texts
from .writer_snapshot import _writer_frontier_choice_snapshot_from_snapshot
from .writer_snapshot import _writer_search_snapshot_with_cursor_after_emitted_text
from .writer_snapshot import advance_writer_frontier_snapshot
from .writer_snapshot import capture_initial_writer_frontier_snapshot
from .writer_snapshot import resume_writer_frontier_choices_from_snapshot
from .writer_snapshot import validate_writer_search_snapshot


@dataclass(frozen=True, slots=True)
class WriterRuntimeState:
    """Opaque writer-runtime state for public traversal.

    The payload is a writer snapshot rather than a separate state encoding.  A
    snapshot already carries the saved writer cursor plus structural identity;
    checked runtime operations below are responsible for enforcing support by
    running the live frontier.
    """

    snapshot: WriterSearchSnapshot


@dataclass(frozen=True, slots=True)
class WriterRuntimeChoiceTransition:
    """A checked writer choice paired with its already-packaged successor state."""

    choice: WriterFrontierChoice
    next_state: WriterRuntimeState


@dataclass(frozen=True, slots=True)
class WriterRuntimeChoiceTransitions:
    """Checked choices plus successors from one live frontier evaluation.

    Adapters should use this instead of accepting an arbitrary public choice and
    trying to advance it.  The runtime computes the checked frontier once, then
    packages each live successor state from that same evidence.  Aggregate
    counts below are derived from that checked result so adapters do not
    recompute frontier summaries themselves.
    """

    choices: WriterFrontierChoices
    transitions: tuple[WriterRuntimeChoiceTransition, ...]

    @property
    def terminal(self) -> WriterFrontierTerminal | None:
        return self.choices.terminal

    @property
    def support_count(self) -> int:
        support_count = sum(
            transition.choice.support_count or 0
            for transition in self.transitions
        )
        if self.terminal is not None:
            support_count += self.terminal.support_count
        return support_count

    @property
    def completion_count(self) -> int:
        completion_count = sum(
            transition.choice.completion_count or 0
            for transition in self.transitions
        )
        if self.terminal is not None:
            completion_count += self.terminal.completion_count
        return completion_count

    @property
    def has_eos(self) -> bool:
        return self.terminal is not None


@dataclass(frozen=True, slots=True)
class _WriterRuntimeBranchTransition:
    """One branch-preserving checked transition support.

    ``emitted_text`` is payload.  It is not the identity of this transition:
    same-text supports intentionally remain separate until a text projection
    groups them for the current public choice surface.
    """

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


@dataclass(frozen=True, slots=True)
class _WriterRuntimeBranchTransitionBatch:
    """Checked text projection plus the branch-preserving supports below it."""

    choices: WriterFrontierChoices
    branch_transitions: tuple[_WriterRuntimeBranchTransition, ...]


@dataclass(frozen=True, slots=True)
class WriterRuntimeDiagnostics:
    """Raw live-frontier evidence for debugging and audit surfaces.

    Diagnostics are observational: they expose blockers, capabilities, work
    evidence, and envelope violations produced by running the current writer
    frontier, but they do not decide support.  Checked runtime operations remain
    the only enforcement boundary.
    """

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
    """Resume a structurally valid snapshot without classifying support.

    Snapshot validation checks that the saved writer state is coherent for the
    prepared molecule.  Calls such as ``writer_runtime_choices`` and
    ``advance_writer_runtime_state`` perform the checked frontier operation that
    can reject unsupported live execution.
    """

    validate_writer_search_snapshot(snapshot, prepared=prepared)
    return WriterRuntimeState(snapshot)


def writer_runtime_diagnostics(
    *,
    prepared: SouthStarPreparedMol,
    state: WriterRuntimeState,
) -> WriterRuntimeDiagnostics:
    # Use the raw snapshot/frontier read here on purpose.  Checked operations
    # raise at the enforcement boundary; diagnostics must instead preserve the
    # live evidence so callers can see why that boundary would reject.
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
    return resume_writer_frontier_choices_from_snapshot(
        state.snapshot,
        prepared=prepared,
    )


def writer_runtime_choice_transitions(
    *,
    prepared: SouthStarPreparedMol,
    state: WriterRuntimeState,
) -> WriterRuntimeChoiceTransitions:
    branch_batch = _writer_runtime_branch_transition_batch(
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


def _writer_runtime_branch_transition_batch(
    *,
    prepared: SouthStarPreparedMol,
    state: WriterRuntimeState,
    include_counts: bool,
) -> _WriterRuntimeBranchTransitionBatch:
    choice_snapshot = _checked_writer_frontier_choice_snapshot_from_snapshot(
        state.snapshot,
        prepared=prepared,
        include_counts=include_counts,
    )
    branch_transitions: list[_WriterRuntimeBranchTransition] = []
    branch_ordinal = 0

    for choice in choice_snapshot.choices:
        for support in choice.supports:
            transition = support.schedule_support.transition
            branch_transitions.append(
                _WriterRuntimeBranchTransition(
                    emitted_text=choice.emitted_text,
                    source_state=support.state_key,
                    successor_state=support.successor_key,
                    parent_weight=support.parent_weight,
                    branch_ordinal=branch_ordinal,
                    transition_kind=transition.kind,
                    events=transition.events,
                    evidence=transition.evidence,
                    execution_capabilities=frozenset(support.execution_capabilities),
                    residual_work_evidence=tuple(support.residual_work_evidence),
                    finite_relation_work_evidence=tuple(support.finite_relation_work_evidence),
                )
            )
            branch_ordinal += 1

    return _WriterRuntimeBranchTransitionBatch(
        choices=choice_snapshot.public_choices,
        branch_transitions=tuple(branch_transitions),
    )


def writer_runtime_terminal(
    *,
    prepared: SouthStarPreparedMol,
    state: WriterRuntimeState,
) -> WriterFrontierTerminal | None:
    return writer_runtime_choices(
        prepared=prepared,
        state=state,
    ).terminal


def writer_runtime_has_eos(
    *,
    prepared: SouthStarPreparedMol,
    state: WriterRuntimeState,
) -> bool:
    return writer_runtime_terminal(
        prepared=prepared,
        state=state,
    ) is not None


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
    """Package a successor from the same checked frontier evaluation.

    This helper stays private because it trusts choice provenance.  The public
    runtime surface exposes ``writer_runtime_choice_transitions`` instead, which
    computes choices and successor states together from one live frontier.
    """

    return WriterRuntimeState(
        _writer_search_snapshot_with_cursor_after_emitted_text(
            state.snapshot,
            prepared=prepared,
            cursor=choice.successor,
        )
    )


def _writer_runtime_state_after_branch_transition(
    *,
    prepared: SouthStarPreparedMol,
    state: WriterRuntimeState,
    branch_transition: _WriterRuntimeBranchTransition,
) -> WriterRuntimeState:
    return _writer_runtime_state_with_cursor(
        prepared=prepared,
        state=state,
        cursor=WriterFrontierCursor(
            weighted_states=((branch_transition.successor_state, 1),)
        ),
    )


def _writer_runtime_state_with_cursor(
    *,
    prepared: SouthStarPreparedMol,
    state: WriterRuntimeState,
    cursor: WriterFrontierCursor,
) -> WriterRuntimeState:
    snapshot = replace(
        state.snapshot,
        cursor=cursor,
        frame_stack=(WriterFrontierFrame(cursor),),
    )
    validate_writer_search_snapshot(snapshot, prepared=prepared)
    return WriterRuntimeState(snapshot)


def count_writer_runtime_support(
    *,
    prepared: SouthStarPreparedMol,
    state: WriterRuntimeState,
) -> int:
    # Route counts through the checked snapshot-prefix operation, not through
    # the support-image adapter.  This keeps the runtime facade below adapters
    # while preserving the same live frontier authority.
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
    return _count_writer_completions_after_emitted_texts(
        state.snapshot,
        prepared=prepared,
        emitted_texts=(),
    )


def count_writer_runtime_branch_completions(
    *,
    prepared: SouthStarPreparedMol,
    state: WriterRuntimeState,
) -> int:
    """Count branch-preserving completions without materializing strings.

    This is intentionally separate from support-string counting.  It follows the
    checked live transition supports and memoizes by canonical successor state.
    """

    return _count_writer_runtime_branch_completions(
        prepared=prepared,
        state=state,
        memo={},
        active=frozenset(),
    )


def _count_writer_runtime_branch_completions(
    *,
    prepared: SouthStarPreparedMol,
    state: WriterRuntimeState,
    memo: dict[object, int],
    active: frozenset[object],
) -> int:
    cursor = state.snapshot.cursor
    total = 0

    for state_key, weight in cursor.weighted_states:
        total += weight * _count_writer_runtime_branch_completions_for_state_key(
            prepared=prepared,
            state=state,
            state_key=state_key,
            memo=memo,
            active=active,
        )

    return total


def _count_writer_runtime_branch_completions_for_state_key(
    *,
    prepared: SouthStarPreparedMol,
    state: WriterRuntimeState,
    state_key: object,
    memo: dict[object, int],
    active: frozenset[object],
) -> int:
    if state_key in memo:
        return memo[state_key]
    if state_key in active:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "writer branch-completion count encountered a recursive state",
        )

    single_state = _writer_runtime_state_with_cursor(
        prepared=prepared,
        state=state,
        cursor=WriterFrontierCursor(weighted_states=((state_key, 1),)),
    )
    branch_batch = _writer_runtime_branch_transition_batch(
        prepared=prepared,
        state=single_state,
        include_counts=False,
    )
    total = 0
    if branch_batch.choices.terminal is not None:
        total += branch_batch.choices.terminal.completion_count

    next_active = active | frozenset((state_key,))
    for branch_transition in branch_batch.branch_transitions:
        successor_state = _writer_runtime_state_after_branch_transition(
            prepared=prepared,
            state=single_state,
            branch_transition=branch_transition,
        )
        total += _count_writer_runtime_branch_completions(
            prepared=prepared,
            state=successor_state,
            memo=memo,
            active=next_active,
        )

    memo[state_key] = total
    return total


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
    "writer_runtime_choice_transitions",
    "writer_runtime_choices",
    "writer_runtime_diagnostics",
    "writer_runtime_has_eos",
    "writer_runtime_state_from_snapshot",
    "writer_runtime_terminal",
)
