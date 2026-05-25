"""Stateful online decoder choices for South Star witnesses."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field

from .facts import MoleculeFacts
from .online_decisions import DecisionPathFilter
from .online_decisions import OnlineDecision
from .online_decisions import OnlineDecisionPath
from .online_decisions import OnlineDecisionRecorder
from .online_stereo_witness import iter_online_stereo_witnesses_with_sink
from .policy import SmilesPolicy
from .semantics import ParserSemantics


@dataclass(frozen=True, slots=True)
class OnlineDecoderState:
    prefix: str
    allowed_paths: frozenset[OnlineDecisionPath] | None = None


@dataclass(frozen=True, slots=True)
class OnlineDecoderChoice:
    text: str
    next_state: OnlineDecoderState
    multiplicity: int = 1


@dataclass(slots=True)
class BranchFrontierSink:
    required_prefix: str
    decision_path_provider: Callable[[], OnlineDecisionPath]
    emitted: list[str] = field(default_factory=list)
    completed_frontier: dict[str, list[OnlineDecisionPath]] = field(default_factory=dict)
    pending_token_text: str | None = None

    def checkpoint(self) -> tuple[int, str | None]:
        return (len(self.emitted), self.pending_token_text)

    def rollback(self, checkpoint: object) -> None:
        if not isinstance(checkpoint, tuple) or len(checkpoint) != 2:
            raise ValueError(f"invalid branch frontier checkpoint: {checkpoint!r}")
        emitted_len, pending_token_text = checkpoint
        if not isinstance(emitted_len, int) or emitted_len < 0 or emitted_len > len(self.emitted):
            raise ValueError(f"invalid branch frontier emitted checkpoint: {checkpoint!r}")
        del self.emitted[emitted_len:]
        self.pending_token_text = pending_token_text  # type: ignore[assignment]

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
        self.emitted.append(text)
        return True

    def complete(self) -> bool:
        if not self.value().startswith(self.required_prefix):
            return False
        if self.pending_token_text is not None:
            self.completed_frontier.setdefault(self.pending_token_text, []).append(
                self.decision_path_provider()
            )
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
        if state.allowed_paths is None
        else DecisionPathFilter(state.allowed_paths)
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
                allowed_paths=frozenset({path}),
            ),
        )
        for text, paths in sink.completed_frontier.items()
        for path in paths
    ]
    return tuple(sorted(choices, key=lambda choice: (choice.text, repr(choice.next_state.allowed_paths))))


def online_determinized_choices(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    state: OnlineDecoderState,
) -> tuple[OnlineDecoderChoice, ...]:
    grouped: dict[str, list[OnlineDecisionPath]] = defaultdict(list)
    for choice in online_branch_preserving_choices(
        facts=facts,
        policy=policy,
        semantics=semantics,
        state=state,
    ):
        if choice.next_state.allowed_paths is None:
            raise ValueError("branch-preserving choice lacks allowed paths")
        grouped[choice.text].extend(choice.next_state.allowed_paths)

    return tuple(
        OnlineDecoderChoice(
            text=text,
            next_state=OnlineDecoderState(
                prefix=state.prefix + text,
                allowed_paths=frozenset(paths),
            ),
            multiplicity=len(paths),
        )
        for text, paths in sorted(grouped.items())
    )


__all__ = (
    "BranchFrontierSink",
    "OnlineDecision",
    "OnlineDecisionPath",
    "OnlineDecoderChoice",
    "OnlineDecoderState",
    "online_branch_preserving_choices",
    "online_determinized_choices",
)
