"""Completion-backed continuation states for South Star online decoding.

The current continuation mode stores proven rendered completions and advances
through their token sequences. It avoids root replay after the first frontier
collection, but it is not a suspended traversal/residual DFS frame.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from enum import Enum
from typing import TYPE_CHECKING

from .facts import MoleculeFacts
from .graph_index import GraphIndex
from .ids import AtomId
from .online_decisions import FrontierCompactionMode
from .online_decisions import OnlineDecisionFrontier
from .online_decisions import OnlineDecisionPath
from .online_decisions import OnlineDecisionRecorder
from .online_decisions import compact_frontier_path
from .online_stereo_witness import iter_online_stereo_witnesses_with_sink
from .policy import SmilesPolicy
from .semantics import ParserSemantics
from .stereo_templates import StereoTemplateBundle

if TYPE_CHECKING:
    from .prepared_runtime import SouthStarPreparedMol


class OnlineDecoderExecutionMode(Enum):
    PREFIX_REPLAY = "prefix_replay"
    CACHED_COMPLETIONS = "cached_completions"
    RESIDUAL_CONTINUATIONS = "residual_continuations"
    # Deprecated compatibility alias for the first completion-backed mode.
    RESUMABLE_CONTINUATIONS = "cached_completions"

    @classmethod
    def _missing_(cls, value: object) -> "OnlineDecoderExecutionMode | None":
        if value == "resumable_continuations":
            return cls.CACHED_COMPLETIONS
        return None


@dataclass(frozen=True, slots=True)
class OnlineContinuation:
    """A cached-completion continuation, not a suspended residual DFS frame."""

    prefix: str
    traversal_cursor: object
    traversal_state: object
    residual_snapshot: object
    ring_state: object
    output_position: int
    decision_frontier: OnlineDecisionPath
    rendered: str
    tokens: tuple[str, ...]
    token_index: int
    completion_count: int = 1


@dataclass(frozen=True, slots=True)
class OnlineContinuationFrontier:
    """Viable cached completions sharing the same rendered prefix."""

    prefix: str
    continuations: tuple[OnlineContinuation, ...]


@dataclass(frozen=True, slots=True)
class OnlineContinuationDecoderState:
    prefix: str
    frontier: OnlineContinuationFrontier | None = None


@dataclass(frozen=True, slots=True)
class OnlineContinuationDecoderChoice:
    text: str
    next_state: OnlineContinuationDecoderState
    multiplicity: int = 1
    completion_count: int = 0


@dataclass(slots=True)
class OnlineContinuationStats:
    root_dfs_runs: int = 0
    resumed_continuations: int = 0
    sink_rejections: int = 0
    completions_seen: int = 0
    eos_completions_seen: int = 0
    eos_frontier_paths: int = 0


@dataclass(frozen=True, slots=True)
class OnlineContinuationRawChoiceResult:
    choices: tuple[OnlineContinuationDecoderChoice, ...]
    eos_completion_count: int
    eos_frontier: OnlineDecisionFrontier
    stats: OnlineContinuationStats


@dataclass(slots=True)
class ContinuationFrontierSink:
    required_prefix: str
    decision_path_provider: Callable[[], OnlineDecisionPath]
    compaction_mode: FrontierCompactionMode = FrontierCompactionMode.TRAVERSAL_ONLY
    emitted: list[str] = field(default_factory=list)
    emitted_tokens: list[str] = field(default_factory=list)
    completed_token_frontiers: dict[str, list[OnlineContinuation]] = field(default_factory=dict)
    eos_continuations: list[OnlineContinuation] = field(default_factory=list)
    pending_token_text: str | None = None
    pending_token_index: int | None = None
    pending_output_position: int | None = None
    pending_frontier_path: OnlineDecisionPath | None = None
    sink_rejections: int = 0
    completions_seen: int = 0

    def checkpoint(
        self,
    ) -> tuple[int, int, str | None, int | None, int | None, OnlineDecisionPath | None]:
        return (
            len(self.emitted),
            len(self.emitted_tokens),
            self.pending_token_text,
            self.pending_token_index,
            self.pending_output_position,
            self.pending_frontier_path,
        )

    def rollback(self, checkpoint: object) -> None:
        if not isinstance(checkpoint, tuple) or len(checkpoint) != 6:
            raise ValueError(f"invalid continuation frontier checkpoint: {checkpoint!r}")
        (
            emitted_len,
            token_len,
            pending_token_text,
            pending_token_index,
            pending_output_position,
            pending_frontier_path,
        ) = checkpoint
        if not isinstance(emitted_len, int) or emitted_len < 0 or emitted_len > len(self.emitted):
            raise ValueError(f"invalid continuation emitted checkpoint: {checkpoint!r}")
        if not isinstance(token_len, int) or token_len < 0 or token_len > len(self.emitted_tokens):
            raise ValueError(f"invalid continuation token checkpoint: {checkpoint!r}")
        del self.emitted[emitted_len:]
        del self.emitted_tokens[token_len:]
        self.pending_token_text = pending_token_text  # type: ignore[assignment]
        self.pending_token_index = pending_token_index  # type: ignore[assignment]
        self.pending_output_position = pending_output_position  # type: ignore[assignment]
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
            self.pending_token_index = len(self.emitted_tokens) + 1
            self.pending_output_position = len(candidate)
            self.pending_frontier_path = compact_frontier_path(
                self.decision_path_provider(),
                mode=self.compaction_mode,
            )
        self.emitted.append(text)
        if token_text is not None:
            self.emitted_tokens.append(token_text)
        return True

    def complete(self) -> bool:
        rendered = self.value()
        if not rendered.startswith(self.required_prefix):
            return False
        self.completions_seen += 1
        if rendered == self.required_prefix:
            self.eos_continuations.append(
                self._continuation(
                    prefix=self.required_prefix,
                    output_position=len(self.required_prefix),
                    token_index=len(self.emitted_tokens),
                    decision_frontier=compact_frontier_path(
                        self.decision_path_provider(),
                        mode=self.compaction_mode,
                    ),
                    rendered=rendered,
                )
            )
        if (
            self.pending_token_text is not None
            and self.pending_token_index is not None
            and self.pending_output_position is not None
            and self.pending_frontier_path is not None
        ):
            continuation = self._continuation(
                prefix=self.required_prefix + self.pending_token_text,
                output_position=self.pending_output_position,
                token_index=self.pending_token_index,
                decision_frontier=self.pending_frontier_path,
                rendered=rendered,
            )
            self.completed_token_frontiers.setdefault(
                self.pending_token_text,
                [],
            ).append(continuation)
        return True

    def value(self) -> str:
        return "".join(self.emitted)

    def _continuation(
        self,
        *,
        prefix: str,
        output_position: int,
        token_index: int,
        decision_frontier: OnlineDecisionPath,
        rendered: str,
    ) -> OnlineContinuation:
        return OnlineContinuation(
            prefix=prefix,
            traversal_cursor=(),
            traversal_state=decision_frontier.items,
            residual_snapshot=(),
            ring_state=(),
            output_position=output_position,
            decision_frontier=decision_frontier,
            rendered=rendered,
            tokens=tuple(self.emitted_tokens),
            token_index=token_index,
        )


def online_branch_preserving_continuation_choice_result(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    state: OnlineContinuationDecoderState,
    compaction_mode: FrontierCompactionMode = FrontierCompactionMode.TRAVERSAL_ONLY,
    templates: StereoTemplateBundle | None = None,
    rooted_at_atom: AtomId | None = None,
    graph_index: GraphIndex | None = None,
    component_root_domains: tuple[tuple[AtomId, ...], ...] | None = None,
    prepared: SouthStarPreparedMol | None = None,
) -> OnlineContinuationRawChoiceResult:
    if state.frontier is None:
        return _root_continuation_choice_result(
            facts=facts,
            policy=policy,
            semantics=semantics,
            state=state,
            compaction_mode=compaction_mode,
            templates=templates,
            rooted_at_atom=rooted_at_atom,
            graph_index=graph_index,
            component_root_domains=component_root_domains,
            prepared=prepared,
        )
    return _resume_branch_preserving_choice_result(state)


def online_determinized_continuation_choice_result(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    state: OnlineContinuationDecoderState,
    compaction_mode: FrontierCompactionMode = FrontierCompactionMode.TRAVERSAL_ONLY,
    templates: StereoTemplateBundle | None = None,
    rooted_at_atom: AtomId | None = None,
    graph_index: GraphIndex | None = None,
    component_root_domains: tuple[tuple[AtomId, ...], ...] | None = None,
    prepared: SouthStarPreparedMol | None = None,
) -> OnlineContinuationRawChoiceResult:
    branch_result = online_branch_preserving_continuation_choice_result(
        facts=facts,
        policy=policy,
        semantics=semantics,
        state=state,
        compaction_mode=compaction_mode,
        templates=templates,
        rooted_at_atom=rooted_at_atom,
        graph_index=graph_index,
        component_root_domains=component_root_domains,
        prepared=prepared,
    )
    grouped: dict[str, list[OnlineContinuation]] = defaultdict(list)
    completion_counts: dict[str, int] = defaultdict(int)
    for choice in branch_result.choices:
        if choice.next_state.frontier is None:
            raise ValueError("continuation branch choice lacks frontier")
        grouped[choice.text].extend(choice.next_state.frontier.continuations)
        completion_counts[choice.text] += choice.completion_count

    return OnlineContinuationRawChoiceResult(
        choices=tuple(
            OnlineContinuationDecoderChoice(
                text=text,
                next_state=OnlineContinuationDecoderState(
                    prefix=state.prefix + text,
                    frontier=OnlineContinuationFrontier(
                        prefix=state.prefix + text,
                        continuations=tuple(continuations),
                    ),
                ),
                multiplicity=len(continuations),
                completion_count=completion_counts[text],
            )
            for text, continuations in sorted(grouped.items())
        ),
        eos_completion_count=branch_result.eos_completion_count,
        eos_frontier=branch_result.eos_frontier,
        stats=branch_result.stats,
    )


def _root_continuation_choice_result(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    state: OnlineContinuationDecoderState,
    compaction_mode: FrontierCompactionMode,
    templates: StereoTemplateBundle | None,
    rooted_at_atom: AtomId | None,
    graph_index: GraphIndex | None,
    component_root_domains: tuple[tuple[AtomId, ...], ...] | None,
    prepared: SouthStarPreparedMol | None,
) -> OnlineContinuationRawChoiceResult:
    recorder = OnlineDecisionRecorder()
    sink = ContinuationFrontierSink(
        required_prefix=state.prefix,
        decision_path_provider=recorder.path,
        compaction_mode=compaction_mode,
    )
    stats = OnlineContinuationStats(root_dfs_runs=1)
    for _ in iter_online_stereo_witnesses_with_sink(
        facts=facts,
        policy=policy,
        semantics=semantics,
        sink_factory=lambda: sink,
        decision_recorder=recorder,
        templates=templates,
        rooted_at_atom=rooted_at_atom,
        graph_index=graph_index,
        component_root_domains=component_root_domains,
        prepared=prepared,
    ):
        pass
    stats.sink_rejections = sink.sink_rejections
    stats.completions_seen = sink.completions_seen
    stats.eos_completions_seen = len(sink.eos_continuations)
    stats.eos_frontier_paths = len(
        frozenset(item.decision_frontier for item in sink.eos_continuations)
    )
    choices = [
        OnlineContinuationDecoderChoice(
            text=text,
            next_state=OnlineContinuationDecoderState(
                prefix=state.prefix + text,
                frontier=OnlineContinuationFrontier(
                    prefix=state.prefix + text,
                    continuations=(continuation,),
                ),
            ),
            completion_count=continuation.completion_count,
        )
        for text, continuations in sink.completed_token_frontiers.items()
        for continuation in continuations
    ]
    eos_frontier = OnlineDecisionFrontier(
        frozenset(item.decision_frontier for item in sink.eos_continuations)
    )
    return OnlineContinuationRawChoiceResult(
        choices=tuple(
            sorted(
                choices,
                key=lambda choice: (
                    choice.text,
                    choice.next_state.frontier.continuations[0].rendered
                    if choice.next_state.frontier is not None
                    else "",
                    repr(choice.next_state.frontier),
                ),
            )
        ),
        eos_completion_count=len(sink.eos_continuations),
        eos_frontier=eos_frontier,
        stats=stats,
    )


def _resume_branch_preserving_choice_result(
    state: OnlineContinuationDecoderState,
) -> OnlineContinuationRawChoiceResult:
    if state.frontier is None:
        raise ValueError("cannot resume continuation decoder without a frontier")
    if state.frontier.prefix != state.prefix:
        raise ValueError("continuation frontier prefix does not match decoder state")
    stats = OnlineContinuationStats(
        resumed_continuations=len(state.frontier.continuations),
        completions_seen=sum(
            continuation.completion_count
            for continuation in state.frontier.continuations
        ),
    )
    choices: list[OnlineContinuationDecoderChoice] = []
    eos_continuations: list[OnlineContinuation] = []
    for continuation in state.frontier.continuations:
        _validate_continuation_for_state(continuation, state.prefix)
        if continuation.token_index < len(continuation.tokens):
            token = continuation.tokens[continuation.token_index]
            next_prefix = state.prefix + token
            next_continuation = replace(
                continuation,
                prefix=next_prefix,
                output_position=continuation.output_position + len(token),
                token_index=continuation.token_index + 1,
            )
            choices.append(
                OnlineContinuationDecoderChoice(
                    text=token,
                    next_state=OnlineContinuationDecoderState(
                        prefix=next_prefix,
                        frontier=OnlineContinuationFrontier(
                            prefix=next_prefix,
                            continuations=(next_continuation,),
                        ),
                    ),
                    completion_count=continuation.completion_count,
                )
            )
            continue
        if continuation.rendered == state.prefix:
            eos_continuations.append(continuation)
            continue
        raise ValueError("continuation has no token but is not at rendered completion")

    stats.eos_completions_seen = sum(
        continuation.completion_count
        for continuation in eos_continuations
    )
    stats.eos_frontier_paths = len(
        frozenset(item.decision_frontier for item in eos_continuations)
    )
    return OnlineContinuationRawChoiceResult(
        choices=tuple(sorted(choices, key=lambda choice: (choice.text, repr(choice.next_state.frontier)))),
        eos_completion_count=sum(
            continuation.completion_count
            for continuation in eos_continuations
        ),
        eos_frontier=OnlineDecisionFrontier(
            frozenset(item.decision_frontier for item in eos_continuations)
        ),
        stats=stats,
    )


def _validate_continuation_for_state(
    continuation: OnlineContinuation,
    prefix: str,
) -> None:
    if continuation.prefix != prefix:
        raise ValueError("continuation prefix does not match decoder state")
    if not continuation.rendered.startswith(prefix):
        raise ValueError("continuation rendered string does not start with state prefix")
    if continuation.output_position != len(prefix):
        raise ValueError("continuation output position does not match state prefix")
    if continuation.token_index < 0 or continuation.token_index > len(continuation.tokens):
        raise ValueError("continuation token index is out of range")


__all__ = (
    "ContinuationFrontierSink",
    "OnlineContinuation",
    "OnlineContinuationDecoderChoice",
    "OnlineContinuationDecoderState",
    "OnlineContinuationFrontier",
    "OnlineContinuationRawChoiceResult",
    "OnlineContinuationStats",
    "OnlineDecoderExecutionMode",
    "online_branch_preserving_continuation_choice_result",
    "online_determinized_continuation_choice_result",
)
