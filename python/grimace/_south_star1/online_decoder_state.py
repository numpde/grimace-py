"""Stateful online decoder choices for South Star witnesses."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field

from .facts import MoleculeFacts
from .online_decisions import DecisionPathFilter
from .online_decisions import FrontierCompactionMode
from .online_decisions import OnlineDecision
from .online_decisions import OnlineDecisionFrontier
from .online_decisions import OnlineDecisionPath
from .online_decisions import OnlineDecisionRecorder
from .online_decisions import compact_frontier_path
from .online_stereo_witness import iter_online_stereo_witnesses_with_sink
from .policy import SmilesPolicy
from .semantics import ParserSemantics
from .stereo_templates import StereoTemplateBundle


@dataclass(frozen=True, slots=True)
class OnlineDecoderState:
    prefix: str
    allowed_frontier: OnlineDecisionFrontier | None = None

    @property
    def allowed_paths(self) -> frozenset[OnlineDecisionPath] | None:
        if self.allowed_frontier is None:
            return None
        return self.allowed_frontier.paths


@dataclass(frozen=True, slots=True)
class OnlineDecoderChoice:
    text: str
    next_state: OnlineDecoderState
    multiplicity: int = 1
    completion_count: int = 0


@dataclass(slots=True)
class OnlineStateDecoderStats:
    dfs_runs: int = 0
    decision_prefix_rejections: int = 0
    sink_rejections: int = 0
    completions_seen: int = 0
    eos_completions_seen: int = 0
    eos_frontier_paths: int = 0


@dataclass(frozen=True, slots=True)
class OnlineRawChoiceResult:
    choices: tuple[OnlineDecoderChoice, ...]
    eos_completion_count: int
    eos_frontier: OnlineDecisionFrontier
    stats: OnlineStateDecoderStats


@dataclass(slots=True)
class BranchFrontierSink:
    required_prefix: str
    decision_path_provider: Callable[[], OnlineDecisionPath]
    compaction_mode: FrontierCompactionMode = FrontierCompactionMode.TRAVERSAL_ONLY
    emitted: list[str] = field(default_factory=list)
    completed_frontier: dict[str, dict[OnlineDecisionPath, int]] = field(default_factory=dict)
    eos_frontier: dict[OnlineDecisionPath, int] = field(default_factory=dict)
    pending_token_text: str | None = None
    pending_frontier_path: OnlineDecisionPath | None = None
    sink_rejections: int = 0
    completions_seen: int = 0

    def checkpoint(self) -> tuple[int, str | None, OnlineDecisionPath | None]:
        return (len(self.emitted), self.pending_token_text, self.pending_frontier_path)

    def rollback(self, checkpoint: object) -> None:
        if not isinstance(checkpoint, tuple) or len(checkpoint) != 3:
            raise ValueError(f"invalid branch frontier checkpoint: {checkpoint!r}")
        emitted_len, pending_token_text, pending_frontier_path = checkpoint
        if not isinstance(emitted_len, int) or emitted_len < 0 or emitted_len > len(self.emitted):
            raise ValueError(f"invalid branch frontier emitted checkpoint: {checkpoint!r}")
        del self.emitted[emitted_len:]
        self.pending_token_text = pending_token_text  # type: ignore[assignment]
        self.pending_frontier_path = pending_frontier_path  # type: ignore[assignment]

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
            self.pending_frontier_path = compact_frontier_path(
                self.decision_path_provider(),
                mode=self.compaction_mode,
            )
        self.emitted.append(text)
        return True

    def complete(self) -> bool:
        if not self.value().startswith(self.required_prefix):
            return False
        self.completions_seen += 1
        if self.value() == self.required_prefix:
            frontier_path = compact_frontier_path(
                self.decision_path_provider(),
                mode=self.compaction_mode,
            )
            self.eos_frontier[frontier_path] = self.eos_frontier.get(frontier_path, 0) + 1
        if self.pending_token_text is not None and self.pending_frontier_path is not None:
            by_path = self.completed_frontier.setdefault(self.pending_token_text, {})
            by_path[self.pending_frontier_path] = by_path.get(self.pending_frontier_path, 0) + 1
        return True

    def value(self) -> str:
        return "".join(self.emitted)


def online_branch_preserving_choices(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    state: OnlineDecoderState,
    compaction_mode: FrontierCompactionMode = FrontierCompactionMode.TRAVERSAL_ONLY,
    templates: StereoTemplateBundle | None = None,
) -> tuple[OnlineDecoderChoice, ...]:
    choices, _ = online_branch_preserving_choices_with_stats(
        facts=facts,
        policy=policy,
        semantics=semantics,
        state=state,
        compaction_mode=compaction_mode,
        templates=templates,
    )
    return choices


def online_branch_preserving_choices_with_stats(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    state: OnlineDecoderState,
    compaction_mode: FrontierCompactionMode = FrontierCompactionMode.TRAVERSAL_ONLY,
    templates: StereoTemplateBundle | None = None,
) -> tuple[tuple[OnlineDecoderChoice, ...], OnlineStateDecoderStats]:
    result = online_branch_preserving_choice_result(
        facts=facts,
        policy=policy,
        semantics=semantics,
        state=state,
        compaction_mode=compaction_mode,
        templates=templates,
    )
    return result.choices, result.stats


def online_branch_preserving_choice_result(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    state: OnlineDecoderState,
    compaction_mode: FrontierCompactionMode = FrontierCompactionMode.TRAVERSAL_ONLY,
    templates: StereoTemplateBundle | None = None,
) -> OnlineRawChoiceResult:
    recorder = OnlineDecisionRecorder()
    path_filter = (
        None
        if state.allowed_frontier is None
        else DecisionPathFilter(state.allowed_frontier)
    )
    sink = BranchFrontierSink(
        required_prefix=state.prefix,
        decision_path_provider=recorder.path,
        compaction_mode=compaction_mode,
    )
    stats = OnlineStateDecoderStats(dfs_runs=1)
    for _ in iter_online_stereo_witnesses_with_sink(
        facts=facts,
        policy=policy,
        semantics=semantics,
        sink_factory=lambda: sink,
        decision_recorder=recorder,
        decision_filter=path_filter,
        templates=templates,
    ):
        pass
    if path_filter is not None:
        stats.decision_prefix_rejections = path_filter.rejection_count
    stats.sink_rejections = sink.sink_rejections
    stats.completions_seen = sink.completions_seen
    stats.eos_completions_seen = sum(sink.eos_frontier.values())
    stats.eos_frontier_paths = len(sink.eos_frontier)

    choices = [
        OnlineDecoderChoice(
            text=text,
            next_state=OnlineDecoderState(
                prefix=state.prefix + text,
                allowed_frontier=OnlineDecisionFrontier(frozenset({path})),
            ),
            completion_count=completion_count,
        )
        for text, paths in sink.completed_frontier.items()
        for path, completion_count in paths.items()
    ]
    return OnlineRawChoiceResult(
        choices=tuple(sorted(choices, key=lambda choice: (choice.text, repr(choice.next_state.allowed_paths)))),
        eos_completion_count=sum(sink.eos_frontier.values()),
        eos_frontier=OnlineDecisionFrontier(frozenset(sink.eos_frontier)),
        stats=stats,
    )


def online_determinized_choices(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    state: OnlineDecoderState,
    compaction_mode: FrontierCompactionMode = FrontierCompactionMode.TRAVERSAL_ONLY,
    templates: StereoTemplateBundle | None = None,
) -> tuple[OnlineDecoderChoice, ...]:
    choices, _ = online_determinized_choices_with_stats(
        facts=facts,
        policy=policy,
        semantics=semantics,
        state=state,
        compaction_mode=compaction_mode,
        templates=templates,
    )
    return choices


def online_determinized_choices_with_stats(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    state: OnlineDecoderState,
    compaction_mode: FrontierCompactionMode = FrontierCompactionMode.TRAVERSAL_ONLY,
    templates: StereoTemplateBundle | None = None,
) -> tuple[tuple[OnlineDecoderChoice, ...], OnlineStateDecoderStats]:
    result = online_determinized_choice_result(
        facts=facts,
        policy=policy,
        semantics=semantics,
        state=state,
        compaction_mode=compaction_mode,
        templates=templates,
    )
    return result.choices, result.stats


def online_determinized_choice_result(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    state: OnlineDecoderState,
    compaction_mode: FrontierCompactionMode = FrontierCompactionMode.TRAVERSAL_ONLY,
    templates: StereoTemplateBundle | None = None,
) -> OnlineRawChoiceResult:
    grouped: dict[str, dict[OnlineDecisionPath, int]] = defaultdict(dict)
    branch_result = online_branch_preserving_choice_result(
        facts=facts,
        policy=policy,
        semantics=semantics,
        state=state,
        compaction_mode=compaction_mode,
        templates=templates,
    )
    for choice in branch_result.choices:
        if choice.next_state.allowed_frontier is None:
            raise ValueError("branch-preserving choice lacks allowed frontier")
        for path in choice.next_state.allowed_frontier.paths:
            grouped[choice.text][path] = (
                grouped[choice.text].get(path, 0) + choice.completion_count
            )

    return OnlineRawChoiceResult(
        choices=tuple(
            OnlineDecoderChoice(
                text=text,
                next_state=OnlineDecoderState(
                    prefix=state.prefix + text,
                    allowed_frontier=OnlineDecisionFrontier(frozenset(paths)),
                ),
                multiplicity=len(paths),
                completion_count=sum(paths.values()),
            )
            for text, paths in sorted(grouped.items())
        ),
        eos_completion_count=branch_result.eos_completion_count,
        eos_frontier=branch_result.eos_frontier,
        stats=branch_result.stats,
    )


__all__ = (
    "BranchFrontierSink",
    "FrontierCompactionMode",
    "OnlineDecision",
    "OnlineDecisionFrontier",
    "OnlineDecisionPath",
    "OnlineDecoderChoice",
    "OnlineDecoderState",
    "OnlineRawChoiceResult",
    "OnlineStateDecoderStats",
    "online_branch_preserving_choice_result",
    "online_branch_preserving_choices",
    "online_branch_preserving_choices_with_stats",
    "online_determinized_choice_result",
    "online_determinized_choices",
    "online_determinized_choices_with_stats",
)
