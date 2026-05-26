"""Residual snapshot-backed online decoder continuations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

from .facts import MoleculeFacts
from .graph_index import GraphIndex
from .ids import AtomId
from .online_decisions import OnlineDecisionFrontier
from .online_decisions import OnlineDecisionPath
from .online_search_vm import RenderContinuationPayloadShape
from .online_search_vm import OnlineSearchSnapshot
from .online_search_vm import OnlineSearchVM
from .online_search_vm import residual_snapshot_frame_audit
from .online_search_vm import render_continuation_payload_shape
from .online_search_vm import validate_residual_frame_stack
from .policy import SmilesPolicy
from .residual_constraints import ResidualStoreValueSnapshot
from .semantics import ParserSemantics
from .stereo_templates import StereoTemplateBundle

if TYPE_CHECKING:
    from .prepared_runtime import SouthStarPreparedMol


@dataclass(frozen=True, slots=True)
class OnlineResidualContinuation:
    prefix: str
    snapshot: OnlineSearchSnapshot
    frontier_path: OnlineDecisionPath
    token_text: str
    completion_count: int = 1


@dataclass(frozen=True, slots=True)
class OnlineResidualContinuationFrontier:
    prefix: str
    continuations: tuple[OnlineResidualContinuation, ...]


@dataclass(frozen=True, slots=True)
class OnlineResidualDecoderState:
    prefix: str
    frontier: OnlineResidualContinuationFrontier | None = None


@dataclass(frozen=True, slots=True)
class OnlineResidualDecoderChoice:
    text: str
    next_state: OnlineResidualDecoderState
    multiplicity: int = 1
    completion_count: int = 0


@dataclass(slots=True)
class OnlineResidualDecoderStats:
    root_dfs_runs: int = 0
    resumed_snapshots: int = 0
    sink_rejections: int = 0
    completions_seen: int = 0
    eos_completions_seen: int = 0
    eos_frontier_paths: int = 0
    candidate_state_size: "OnlineResidualStateSizeStats" = field(
        default_factory=lambda: OnlineResidualStateSizeStats()
    )
    retained_state_size: "OnlineResidualStateSizeStats" = field(
        default_factory=lambda: OnlineResidualStateSizeStats()
    )


@dataclass(frozen=True, slots=True)
class ResidualStoreSnapshotShape:
    var_count: int = 0
    assignment_count: int = 0
    factor_count: int = 0


@dataclass(frozen=True, slots=True)
class OnlineSearchSnapshotShape:
    frame_stack_depth: int = 0
    residual_var_count: int = 0
    residual_assignment_count: int = 0
    residual_factor_count: int = 0
    decision_path_length: int = 0
    output_snapshot_length: int = 0
    ring_endpoint_count: int = 0
    ring_open_interval_count: int = 0
    render_payload: RenderContinuationPayloadShape = field(
        default_factory=RenderContinuationPayloadShape
    )


@dataclass(frozen=True, slots=True)
class OnlineResidualStateSizeStats:
    continuation_count: int = 0
    unique_continuation_count: int = 0
    merged_continuation_count: int = 0
    max_merge_count_per_key: int = 0
    max_merge_count_per_token: int = 0
    max_frame_stack_depth: int = 0
    total_frame_stack_depth: int = 0
    max_residual_var_count: int = 0
    total_residual_var_count: int = 0
    max_residual_assignment_count: int = 0
    total_residual_assignment_count: int = 0
    max_residual_factor_count: int = 0
    total_residual_factor_count: int = 0
    max_decision_path_length: int = 0
    total_decision_path_length: int = 0
    max_output_snapshot_length: int = 0
    total_output_snapshot_length: int = 0
    max_ring_endpoint_count: int = 0
    total_ring_endpoint_count: int = 0
    max_ring_open_interval_count: int = 0
    total_ring_open_interval_count: int = 0
    resumable_frame_count: int = 0
    context_frame_count: int = 0
    unknown_frame_count: int = 0
    render_resume_continuation_count: int = 0
    max_render_piece_count: int = 0
    total_render_piece_count: int = 0
    max_remaining_render_piece_count: int = 0
    total_remaining_render_piece_count: int = 0
    max_render_payload_chars: int = 0
    total_render_payload_chars: int = 0
    render_cursor_count: int = 0
    max_render_program_event_count: int = 0
    total_render_program_event_count: int = 0
    max_remaining_render_event_count: int = 0
    total_remaining_render_event_count: int = 0
    max_render_program_choice_count: int = 0
    total_render_program_choice_count: int = 0
    scheduler_frame_count: int = 0
    prefix_enumeration_frame_count: int = 0
    max_prefix_domain_count: int = 0
    total_prefix_domain_count: int = 0
    max_prefix_assignment_count: int = 0
    total_prefix_assignment_count: int = 0
    direction_enumeration_frame_count: int = 0
    max_direction_carrier_count: int = 0
    total_direction_carrier_count: int = 0
    max_direction_assignment_count: int = 0
    total_direction_assignment_count: int = 0
    support_maximal_frame_count: int = 0
    max_support_maximal_candidate_count: int = 0
    total_support_maximal_candidate_count: int = 0
    max_support_maximal_selected_count: int = 0
    total_support_maximal_selected_count: int = 0
    max_support_maximal_remaining_count: int = 0
    total_support_maximal_remaining_count: int = 0


@dataclass(frozen=True, slots=True)
class OnlineResidualRawChoiceResult:
    choices: tuple[OnlineResidualDecoderChoice, ...]
    eos_completion_count: int
    eos_frontier: OnlineDecisionFrontier
    stats: OnlineResidualDecoderStats


@dataclass(frozen=True, slots=True)
class ResidualFrontierSinkCheckpoint:
    emitted: tuple[str, ...]
    pending_token_text: str | None
    pending_snapshot: OnlineSearchSnapshot | None
    pending_frontier_path: OnlineDecisionPath | None


@dataclass(slots=True)
class ResidualFrontierSink:
    required_prefix: str
    snapshot_provider: Callable[[], OnlineSearchSnapshot] | None = None
    decision_path_provider: Callable[[], OnlineDecisionPath] | None = None
    emitted: list[str] = field(default_factory=list)
    completed_by_token: dict[str, list[OnlineResidualContinuation]] = field(default_factory=dict)
    eos_by_frontier: list[OnlineResidualContinuation] = field(default_factory=list)
    pending_token_text: str | None = None
    pending_snapshot: OnlineSearchSnapshot | None = None
    pending_frontier_path: OnlineDecisionPath | None = None
    sink_rejections: int = 0
    completions_seen: int = 0

    def checkpoint(self) -> ResidualFrontierSinkCheckpoint:
        return ResidualFrontierSinkCheckpoint(
            emitted=tuple(self.emitted),
            pending_token_text=self.pending_token_text,
            pending_snapshot=self.pending_snapshot,
            pending_frontier_path=self.pending_frontier_path,
        )

    def rollback(self, checkpoint: object) -> None:
        if not isinstance(checkpoint, ResidualFrontierSinkCheckpoint):
            raise ValueError(f"invalid residual frontier checkpoint: {checkpoint!r}")
        self.emitted = list(checkpoint.emitted)
        self.pending_token_text = checkpoint.pending_token_text
        self.pending_snapshot = checkpoint.pending_snapshot
        self.pending_frontier_path = checkpoint.pending_frontier_path

    def append(self, text: str, *, token_text: str | None = None) -> bool:
        if not text:
            return True
        current = self.value()
        candidate = current + text
        prefix = self.required_prefix
        if len(current) < len(prefix):
            compare_len = min(len(candidate), len(prefix))
            if candidate[:compare_len] != prefix[:compare_len]:
                self.sink_rejections += 1
                return False
        elif len(current) == len(prefix) and token_text is not None:
            self.pending_token_text = token_text

        self.emitted.append(text)
        if len(current) == len(prefix) and token_text is not None:
            self.pending_snapshot = self._snapshot()
            self.pending_frontier_path = self._decision_path()
        return True

    def complete(self) -> bool:
        rendered = self.value()
        if not rendered.startswith(self.required_prefix):
            return False
        self.completions_seen += 1
        if rendered == self.required_prefix:
            self.eos_by_frontier.append(
                OnlineResidualContinuation(
                    prefix=self.required_prefix,
                    snapshot=self._snapshot(),
                    frontier_path=self._decision_path(),
                    token_text="",
                )
            )
        if (
            self.pending_token_text is not None
            and self.pending_snapshot is not None
            and self.pending_frontier_path is not None
        ):
            continuation = OnlineResidualContinuation(
                prefix=self.required_prefix + self.pending_token_text,
                snapshot=self.pending_snapshot,
                frontier_path=self.pending_frontier_path,
                token_text=self.pending_token_text,
            )
            self.completed_by_token.setdefault(self.pending_token_text, []).append(continuation)
        return True

    def value(self) -> str:
        return "".join(self.emitted)

    def _snapshot(self) -> OnlineSearchSnapshot:
        if self.snapshot_provider is None:
            raise ValueError("residual frontier sink lacks snapshot provider")
        return self.snapshot_provider()

    def _decision_path(self) -> OnlineDecisionPath:
        if self.decision_path_provider is None:
            raise ValueError("residual frontier sink lacks decision path provider")
        return self.decision_path_provider()


def online_branch_preserving_residual_choice_result(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    state: OnlineResidualDecoderState,
    templates: StereoTemplateBundle | None = None,
    rooted_at_atom: AtomId | None = None,
    graph_index: GraphIndex | None = None,
    component_root_domains: tuple[tuple[AtomId, ...], ...] | None = None,
    prepared: SouthStarPreparedMol | None = None,
) -> OnlineResidualRawChoiceResult:
    if state.frontier is None:
        return _root_residual_choice_result(
            facts=facts,
            policy=policy,
            semantics=semantics,
            state=state,
            templates=templates,
            rooted_at_atom=rooted_at_atom,
            graph_index=graph_index,
            component_root_domains=component_root_domains,
            prepared=prepared,
        )
    return _resume_residual_choice_result(
        facts=facts,
        policy=policy,
        semantics=semantics,
        state=state,
        templates=templates,
        rooted_at_atom=rooted_at_atom,
        graph_index=graph_index,
        component_root_domains=component_root_domains,
        prepared=prepared,
    )


def online_determinized_residual_choice_result(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    state: OnlineResidualDecoderState,
    templates: StereoTemplateBundle | None = None,
    rooted_at_atom: AtomId | None = None,
    graph_index: GraphIndex | None = None,
    component_root_domains: tuple[tuple[AtomId, ...], ...] | None = None,
    prepared: SouthStarPreparedMol | None = None,
) -> OnlineResidualRawChoiceResult:
    branch_result = online_branch_preserving_residual_choice_result(
        facts=facts,
        policy=policy,
        semantics=semantics,
        state=state,
        templates=templates,
        rooted_at_atom=rooted_at_atom,
        graph_index=graph_index,
        component_root_domains=component_root_domains,
        prepared=prepared,
    )
    grouped: dict[str, dict[tuple[object, ...], OnlineResidualContinuation]] = defaultdict(dict)
    for choice in branch_result.choices:
        if choice.next_state.frontier is None:
            raise ValueError("residual branch choice lacks frontier")
        for continuation in choice.next_state.frontier.continuations:
            key = residual_continuation_key(continuation)
            existing = grouped[choice.text].get(key)
            if existing is None:
                grouped[choice.text][key] = continuation
                continue
            grouped[choice.text][key] = _merge_continuation_counts(existing, continuation)
    retained = tuple(
        continuation
        for continuations in grouped.values()
        for continuation in continuations.values()
    )
    branch_result.stats.retained_state_size = _state_size_from_continuations(
        retained,
        require_resumable=True,
    )
    return OnlineResidualRawChoiceResult(
        choices=tuple(
            OnlineResidualDecoderChoice(
                text=text,
                next_state=OnlineResidualDecoderState(
                    prefix=state.prefix + text,
                    frontier=OnlineResidualContinuationFrontier(
                        prefix=state.prefix + text,
                        continuations=tuple(continuations.values()),
                    ),
                ),
                multiplicity=len(continuations),
                completion_count=sum(
                    continuation.completion_count
                    for continuation in continuations.values()
                ),
            )
            for text, continuations in sorted(grouped.items())
        ),
        eos_completion_count=branch_result.eos_completion_count,
        eos_frontier=branch_result.eos_frontier,
        stats=branch_result.stats,
    )


def _root_residual_choice_result(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    state: OnlineResidualDecoderState,
    templates: StereoTemplateBundle | None,
    rooted_at_atom: AtomId | None,
    graph_index: GraphIndex | None,
    component_root_domains: tuple[tuple[AtomId, ...], ...] | None,
    prepared: SouthStarPreparedMol | None,
) -> OnlineResidualRawChoiceResult:
    sink = ResidualFrontierSink(required_prefix=state.prefix)
    vm = OnlineSearchVM(
        facts=facts,
        policy=policy,
        semantics=semantics,
        templates=templates,
        rooted_at_atom=rooted_at_atom,
        graph_index=graph_index,
        component_root_domains=component_root_domains,
        online_traversal_graph=(
            None if prepared is None else prepared.online_traversal_graph
        ),
        assume_prepared=prepared is not None,
        sink_factory=lambda: sink,
    )
    sink.snapshot_provider = vm.checkpoint
    sink.decision_path_provider = vm.state.decisions.path
    while vm.run_until_witness_or_exhausted() is not None:
        pass
    stats = OnlineResidualDecoderStats(
        root_dfs_runs=1,
        sink_rejections=sink.sink_rejections,
        completions_seen=sink.completions_seen,
        eos_completions_seen=len(sink.eos_by_frontier),
        eos_frontier_paths=len(frozenset(item.frontier_path for item in sink.eos_by_frontier)),
    )
    return _result_from_sink(state.prefix, sink, stats)


def _resume_residual_choice_result(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    state: OnlineResidualDecoderState,
    templates: StereoTemplateBundle | None,
    rooted_at_atom: AtomId | None,
    graph_index: GraphIndex | None,
    component_root_domains: tuple[tuple[AtomId, ...], ...] | None,
    prepared: SouthStarPreparedMol | None,
) -> OnlineResidualRawChoiceResult:
    if state.frontier is None:
        raise ValueError("cannot resume residual decoder without a frontier")
    if state.frontier.prefix != state.prefix:
        raise ValueError("residual frontier prefix does not match decoder state")
    merged_sink = ResidualFrontierSink(required_prefix=state.prefix)
    stats = OnlineResidualDecoderStats(
        resumed_snapshots=len(state.frontier.continuations),
    )
    for continuation in state.frontier.continuations:
        if continuation.prefix != state.prefix:
            raise ValueError("residual continuation prefix does not match decoder state")
        sink = ResidualFrontierSink(required_prefix=state.prefix)
        vm = OnlineSearchVM.from_snapshot(
            facts=facts,
            policy=policy,
            semantics=semantics,
            templates=templates,
            rooted_at_atom=rooted_at_atom,
            graph_index=graph_index,
            component_root_domains=component_root_domains,
            online_traversal_graph=(
                None if prepared is None else prepared.online_traversal_graph
            ),
            snapshot=continuation.snapshot,
            sink=sink,
            assume_prepared=prepared is not None,
        )
        sink.snapshot_provider = vm.checkpoint
        sink.decision_path_provider = vm.state.decisions.path
        while vm.run_until_witness_or_exhausted() is not None:
            pass
        stats.sink_rejections += sink.sink_rejections
        stats.completions_seen += sink.completions_seen
        stats.eos_completions_seen += len(sink.eos_by_frontier)
        _merge_sink(merged_sink, sink)
    stats.eos_frontier_paths = len(
        frozenset(item.frontier_path for item in merged_sink.eos_by_frontier)
    )
    return _result_from_sink(state.prefix, merged_sink, stats)


def _result_from_sink(
    prefix: str,
    sink: ResidualFrontierSink,
    stats: OnlineResidualDecoderStats,
) -> OnlineResidualRawChoiceResult:
    by_token = {
        text: merge_residual_continuations_by_key(continuations)
        for text, continuations in sink.completed_by_token.items()
    }
    choices = [
        OnlineResidualDecoderChoice(
            text=text,
            next_state=OnlineResidualDecoderState(
                prefix=prefix + text,
                frontier=OnlineResidualContinuationFrontier(
                    prefix=prefix + text,
                    continuations=(continuation,),
                ),
            ),
            completion_count=continuation.completion_count,
        )
        for text, continuations in by_token.items()
        for continuation in continuations
    ]
    eos_frontier = OnlineDecisionFrontier(
        frozenset(item.frontier_path for item in sink.eos_by_frontier)
    )
    retained = tuple(
        continuation
        for continuations in by_token.values()
        for continuation in continuations
    )
    stats.candidate_state_size = _state_size_from_continuations(_sink_continuations(sink))
    stats.retained_state_size = _state_size_from_continuations(
        retained,
        require_resumable=True,
    )
    return OnlineResidualRawChoiceResult(
        choices=tuple(sorted(choices, key=lambda choice: (choice.text, repr(choice.next_state.frontier)))),
        eos_completion_count=sum(
            continuation.completion_count
            for continuation in merge_residual_continuations_by_key(sink.eos_by_frontier)
        ),
        eos_frontier=eos_frontier,
        stats=stats,
    )


def _merge_sink(target: ResidualFrontierSink, source: ResidualFrontierSink) -> None:
    for text, continuations in source.completed_by_token.items():
        target.completed_by_token.setdefault(text, []).extend(continuations)
    target.eos_by_frontier.extend(source.eos_by_frontier)


def residual_continuation_key(
    continuation: OnlineResidualContinuation,
) -> tuple[object, ...]:
    snapshot = continuation.snapshot
    return (
        continuation.prefix,
        continuation.token_text,
        continuation.frontier_path,
        snapshot.traversal_state,
        snapshot.residual_snapshot,
        snapshot.ring_state,
        snapshot.output_snapshot,
        snapshot.decision_snapshot,
        snapshot.frame_stack,
    )


def residual_store_snapshot_shape(snapshot: object) -> ResidualStoreSnapshotShape:
    if not isinstance(snapshot, ResidualStoreValueSnapshot):
        return ResidualStoreSnapshotShape()
    return ResidualStoreSnapshotShape(
        var_count=len(snapshot.domains),
        assignment_count=len(snapshot.assignments),
        factor_count=len(snapshot.factors),
    )


def online_search_snapshot_shape(snapshot: OnlineSearchSnapshot) -> OnlineSearchSnapshotShape:
    residual_shape = residual_store_snapshot_shape(snapshot.residual_snapshot)
    ring_endpoint_count, ring_open_interval_count = _ring_snapshot_counts(snapshot.ring_state)
    return OnlineSearchSnapshotShape(
        frame_stack_depth=len(snapshot.frame_stack),
        residual_var_count=residual_shape.var_count,
        residual_assignment_count=residual_shape.assignment_count,
        residual_factor_count=residual_shape.factor_count,
        decision_path_length=_decision_snapshot_length(snapshot.decision_snapshot),
        output_snapshot_length=_output_snapshot_length(snapshot.output_snapshot),
        ring_endpoint_count=ring_endpoint_count,
        ring_open_interval_count=ring_open_interval_count,
        render_payload=render_continuation_payload_shape(snapshot.frame_stack),
    )


def residual_frontier_shape(
    frontier: OnlineResidualContinuationFrontier,
) -> OnlineResidualStateSizeStats:
    return _state_size_from_continuations(frontier.continuations)


def merge_residual_continuations_by_key(
    continuations: list[OnlineResidualContinuation],
) -> tuple[OnlineResidualContinuation, ...]:
    by_key: dict[tuple[object, ...], OnlineResidualContinuation] = {}
    for continuation in continuations:
        key = residual_continuation_key(continuation)
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = continuation
            continue
        by_key[key] = _merge_continuation_counts(existing, continuation)
    return tuple(by_key[key] for key in sorted(by_key, key=repr))


def _merge_continuation_counts(
    left: OnlineResidualContinuation,
    right: OnlineResidualContinuation,
) -> OnlineResidualContinuation:
    if residual_continuation_key(left) != residual_continuation_key(right):
        raise ValueError("cannot merge different residual continuations")
    return OnlineResidualContinuation(
        prefix=left.prefix,
        snapshot=left.snapshot,
        frontier_path=left.frontier_path,
        token_text=left.token_text,
        completion_count=left.completion_count + right.completion_count,
    )


def _sink_continuations(sink: ResidualFrontierSink) -> tuple[OnlineResidualContinuation, ...]:
    out: list[OnlineResidualContinuation] = []
    for continuations in sink.completed_by_token.values():
        out.extend(continuations)
    out.extend(sink.eos_by_frontier)
    return tuple(out)


def _state_size_from_continuations(
    continuations: tuple[OnlineResidualContinuation, ...],
    *,
    require_resumable: bool = False,
) -> OnlineResidualStateSizeStats:
    if not continuations:
        return OnlineResidualStateSizeStats()
    if require_resumable:
        for continuation in continuations:
            validate_residual_frame_stack(continuation.snapshot.frame_stack)
    unique = merge_residual_continuations_by_key(list(continuations))
    shapes = tuple(online_search_snapshot_shape(continuation.snapshot) for continuation in continuations)
    audits = tuple(
        residual_snapshot_frame_audit(continuation.snapshot)
        for continuation in continuations
    )
    key_counts: dict[tuple[object, ...], int] = defaultdict(int)
    token_counts: dict[str, int] = defaultdict(int)
    for continuation in continuations:
        key_counts[residual_continuation_key(continuation)] += 1
        token_counts[continuation.token_text] += 1
    return OnlineResidualStateSizeStats(
        continuation_count=len(continuations),
        unique_continuation_count=len(unique),
        merged_continuation_count=len(continuations) - len(unique),
        max_merge_count_per_key=max(key_counts.values()),
        max_merge_count_per_token=max(token_counts.values()),
        max_frame_stack_depth=max(shape.frame_stack_depth for shape in shapes),
        total_frame_stack_depth=sum(shape.frame_stack_depth for shape in shapes),
        max_residual_var_count=max(shape.residual_var_count for shape in shapes),
        total_residual_var_count=sum(shape.residual_var_count for shape in shapes),
        max_residual_assignment_count=max(shape.residual_assignment_count for shape in shapes),
        total_residual_assignment_count=sum(shape.residual_assignment_count for shape in shapes),
        max_residual_factor_count=max(shape.residual_factor_count for shape in shapes),
        total_residual_factor_count=sum(shape.residual_factor_count for shape in shapes),
        max_decision_path_length=max(shape.decision_path_length for shape in shapes),
        total_decision_path_length=sum(shape.decision_path_length for shape in shapes),
        max_output_snapshot_length=max(shape.output_snapshot_length for shape in shapes),
        total_output_snapshot_length=sum(shape.output_snapshot_length for shape in shapes),
        max_ring_endpoint_count=max(shape.ring_endpoint_count for shape in shapes),
        total_ring_endpoint_count=sum(shape.ring_endpoint_count for shape in shapes),
        max_ring_open_interval_count=max(shape.ring_open_interval_count for shape in shapes),
        total_ring_open_interval_count=sum(shape.ring_open_interval_count for shape in shapes),
        resumable_frame_count=sum(
            audit.resumable_frame_count for audit in audits
        ),
        context_frame_count=sum(audit.context_frame_count for audit in audits),
        unknown_frame_count=sum(audit.unknown_frame_count for audit in audits),
        render_resume_continuation_count=sum(
            shape.render_payload.render_resume_continuation_count
            for shape in shapes
        ),
        max_render_piece_count=max(
            shape.render_payload.max_render_piece_count
            for shape in shapes
        ),
        total_render_piece_count=sum(
            shape.render_payload.total_render_piece_count
            for shape in shapes
        ),
        max_remaining_render_piece_count=max(
            shape.render_payload.max_remaining_render_piece_count
            for shape in shapes
        ),
        total_remaining_render_piece_count=sum(
            shape.render_payload.total_remaining_render_piece_count
            for shape in shapes
        ),
        max_render_payload_chars=max(
            shape.render_payload.max_render_payload_chars
            for shape in shapes
        ),
        total_render_payload_chars=sum(
            shape.render_payload.total_render_payload_chars
            for shape in shapes
        ),
        render_cursor_count=sum(
            shape.render_payload.render_cursor_count
            for shape in shapes
        ),
        max_render_program_event_count=max(
            shape.render_payload.max_render_program_event_count
            for shape in shapes
        ),
        total_render_program_event_count=sum(
            shape.render_payload.total_render_program_event_count
            for shape in shapes
        ),
        max_remaining_render_event_count=max(
            shape.render_payload.max_remaining_render_event_count
            for shape in shapes
        ),
        total_remaining_render_event_count=sum(
            shape.render_payload.total_remaining_render_event_count
            for shape in shapes
        ),
        max_render_program_choice_count=max(
            shape.render_payload.max_render_program_choice_count
            for shape in shapes
        ),
        total_render_program_choice_count=sum(
            shape.render_payload.total_render_program_choice_count
            for shape in shapes
        ),
        scheduler_frame_count=sum(
            shape.render_payload.scheduler_frame_count
            for shape in shapes
        ),
        prefix_enumeration_frame_count=sum(
            shape.render_payload.prefix_enumeration_frame_count
            for shape in shapes
        ),
        max_prefix_domain_count=max(
            shape.render_payload.max_prefix_domain_count
            for shape in shapes
        ),
        total_prefix_domain_count=sum(
            shape.render_payload.total_prefix_domain_count
            for shape in shapes
        ),
        max_prefix_assignment_count=max(
            shape.render_payload.max_prefix_assignment_count
            for shape in shapes
        ),
        total_prefix_assignment_count=sum(
            shape.render_payload.total_prefix_assignment_count
            for shape in shapes
        ),
        direction_enumeration_frame_count=sum(
            shape.render_payload.direction_enumeration_frame_count
            for shape in shapes
        ),
        max_direction_carrier_count=max(
            shape.render_payload.max_direction_carrier_count
            for shape in shapes
        ),
        total_direction_carrier_count=sum(
            shape.render_payload.total_direction_carrier_count
            for shape in shapes
        ),
        max_direction_assignment_count=max(
            shape.render_payload.max_direction_assignment_count
            for shape in shapes
        ),
        total_direction_assignment_count=sum(
            shape.render_payload.total_direction_assignment_count
            for shape in shapes
        ),
        support_maximal_frame_count=sum(
            shape.render_payload.support_maximal_frame_count
            for shape in shapes
        ),
        max_support_maximal_candidate_count=max(
            shape.render_payload.max_support_maximal_candidate_count
            for shape in shapes
        ),
        total_support_maximal_candidate_count=sum(
            shape.render_payload.total_support_maximal_candidate_count
            for shape in shapes
        ),
        max_support_maximal_selected_count=max(
            shape.render_payload.max_support_maximal_selected_count
            for shape in shapes
        ),
        total_support_maximal_selected_count=sum(
            shape.render_payload.total_support_maximal_selected_count
            for shape in shapes
        ),
        max_support_maximal_remaining_count=max(
            shape.render_payload.max_support_maximal_remaining_count
            for shape in shapes
        ),
        total_support_maximal_remaining_count=sum(
            shape.render_payload.total_support_maximal_remaining_count
            for shape in shapes
        ),
    )


def _decision_snapshot_length(snapshot: object) -> int:
    if isinstance(snapshot, OnlineDecisionPath):
        return len(snapshot.items)
    if isinstance(snapshot, int):
        return snapshot
    return 0


def _output_snapshot_length(snapshot: object) -> int:
    if isinstance(snapshot, ResidualFrontierSinkCheckpoint):
        if snapshot.pending_snapshot is not None:
            raise ValueError(
                "residual state-size audit does not recursively count nested pending snapshots"
            )
        return sum(len(item) for item in snapshot.emitted)
    if isinstance(snapshot, tuple):
        return sum(len(item) for item in snapshot if isinstance(item, str))
    if isinstance(snapshot, int):
        return snapshot
    return 0


def _ring_snapshot_counts(snapshot: object) -> tuple[int, int]:
    if not isinstance(snapshot, tuple) or len(snapshot) != 4:
        return (0, 0)
    _, label_by_endpoint, open_intervals, _ = snapshot
    endpoint_count = len(label_by_endpoint) if isinstance(label_by_endpoint, tuple) else 0
    open_count = len(open_intervals) if isinstance(open_intervals, tuple) else 0
    return (endpoint_count, open_count)


__all__ = (
    "OnlineResidualContinuation",
    "OnlineResidualContinuationFrontier",
    "OnlineResidualDecoderChoice",
    "OnlineResidualDecoderState",
    "OnlineResidualDecoderStats",
    "OnlineResidualRawChoiceResult",
    "OnlineResidualStateSizeStats",
    "OnlineSearchSnapshotShape",
    "ResidualStoreSnapshotShape",
    "ResidualFrontierSinkCheckpoint",
    "ResidualFrontierSink",
    "online_branch_preserving_residual_choice_result",
    "online_determinized_residual_choice_result",
    "online_search_snapshot_shape",
    "merge_residual_continuations_by_key",
    "residual_frontier_shape",
    "residual_continuation_key",
    "residual_store_snapshot_shape",
)
