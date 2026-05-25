"""Stateful online decoder choices for South Star witnesses."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field

from .facts import MoleculeFacts
from .online_decisions import DecisionPathFilter
from .online_decisions import OnlineDecision
from .online_decisions import OnlineDecisionFrontier
from .online_decisions import OnlineDecisionPath
from .online_decisions import OnlineDecisionRecorder
from .online_decisions import compact_frontier_path
from .online_stereo_witness import iter_online_stereo_witnesses_with_sink
from .policy import SmilesPolicy
from .semantics import ParserSemantics


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
class BranchFrontierSink:
    required_prefix: str
    decision_path_provider: Callable[[], OnlineDecisionPath]
    emitted: list[str] = field(default_factory=list)
    completed_frontier: dict[str, dict[OnlineDecisionPath, int]] = field(default_factory=dict)
    pending_token_text: str | None = None
    pending_frontier_path: OnlineDecisionPath | None = None

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
                return False
        elif len(current) == len(prefix) and token_text is not None:
            self.pending_token_text = token_text
            self.pending_frontier_path = compact_frontier_path(
                self.decision_path_provider()
            )
        self.emitted.append(text)
        return True

    def complete(self) -> bool:
        if not self.value().startswith(self.required_prefix):
            return False
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
) -> tuple[OnlineDecoderChoice, ...]:
    recorder = OnlineDecisionRecorder()
    path_filter = (
        None
        if state.allowed_frontier is None
        else DecisionPathFilter(state.allowed_frontier)
    )
    sink = BranchFrontierSink(
        required_prefix=state.prefix,
        decision_path_provider=recorder.path,
    )
    for _ in iter_online_stereo_witnesses_with_sink(
        facts=facts,
        policy=policy,
        semantics=semantics,
        sink_factory=lambda: sink,
        decision_recorder=recorder,
        decision_filter=path_filter,
    ):
        pass

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
    return tuple(sorted(choices, key=lambda choice: (choice.text, repr(choice.next_state.allowed_paths))))


def online_determinized_choices(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    state: OnlineDecoderState,
) -> tuple[OnlineDecoderChoice, ...]:
    grouped: dict[str, dict[OnlineDecisionPath, int]] = defaultdict(dict)
    for choice in online_branch_preserving_choices(
        facts=facts,
        policy=policy,
        semantics=semantics,
        state=state,
    ):
        if choice.next_state.allowed_frontier is None:
            raise ValueError("branch-preserving choice lacks allowed frontier")
        for path in choice.next_state.allowed_frontier.paths:
            grouped[choice.text][path] = (
                grouped[choice.text].get(path, 0) + choice.completion_count
            )

    return tuple(
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
    )


__all__ = (
    "BranchFrontierSink",
    "OnlineDecision",
    "OnlineDecisionFrontier",
    "OnlineDecisionPath",
    "OnlineDecoderChoice",
    "OnlineDecoderState",
    "online_branch_preserving_choices",
    "online_determinized_choices",
)
