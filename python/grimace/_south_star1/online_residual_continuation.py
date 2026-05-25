"""Residual snapshot-backed online decoder continuations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field

from .facts import MoleculeFacts
from .online_decisions import OnlineDecisionFrontier
from .online_decisions import OnlineDecisionPath
from .online_search_vm import OnlineSearchSnapshot
from .online_search_vm import OnlineSearchVM
from .policy import SmilesPolicy
from .semantics import ParserSemantics


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
    completed_by_token_lengths: tuple[tuple[str, int], ...]
    eos_by_frontier_length: int


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
            completed_by_token_lengths=tuple(
                sorted(
                    (token, len(continuations))
                    for token, continuations in self.completed_by_token.items()
                )
            ),
            eos_by_frontier_length=len(self.eos_by_frontier),
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
) -> OnlineResidualRawChoiceResult:
    if state.frontier is None:
        return _root_residual_choice_result(
            facts=facts,
            policy=policy,
            semantics=semantics,
            state=state,
        )
    return _resume_residual_choice_result(
        facts=facts,
        policy=policy,
        semantics=semantics,
        state=state,
    )


def online_determinized_residual_choice_result(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    state: OnlineResidualDecoderState,
) -> OnlineResidualRawChoiceResult:
    branch_result = online_branch_preserving_residual_choice_result(
        facts=facts,
        policy=policy,
        semantics=semantics,
        state=state,
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
) -> OnlineResidualRawChoiceResult:
    sink = ResidualFrontierSink(required_prefix=state.prefix)
    vm = OnlineSearchVM(
        facts=facts,
        policy=policy,
        semantics=semantics,
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
            snapshot=continuation.snapshot,
            sink=sink,
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


__all__ = (
    "OnlineResidualContinuation",
    "OnlineResidualContinuationFrontier",
    "OnlineResidualDecoderChoice",
    "OnlineResidualDecoderState",
    "OnlineResidualDecoderStats",
    "OnlineResidualRawChoiceResult",
    "ResidualFrontierSinkCheckpoint",
    "ResidualFrontierSink",
    "online_branch_preserving_residual_choice_result",
    "online_determinized_residual_choice_result",
    "merge_residual_continuations_by_key",
    "residual_continuation_key",
)
