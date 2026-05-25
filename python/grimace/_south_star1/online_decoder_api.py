"""Main-branch-shaped facade for South Star online decoder states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .facts import MoleculeFacts
from .online_decoder import online_decode_tokens
from .online_decisions import FrontierCompactionMode
from .online_decoder_state import OnlineDecoderState
from .online_decoder_state import OnlineStateDecoderStats
from .online_decoder_state import online_branch_preserving_choices_with_stats
from .online_decoder_state import online_determinized_choices_with_stats
from .online_stereo_witness import iter_online_stereo_witness_strings
from .policy import SmilesPolicy
from .semantics import ParserSemantics


EOS = "<EOS>"


@dataclass(frozen=True, slots=True)
class SouthStarOnlineChoice:
    text: str
    next_state: "SouthStarOnlineDecoderState | None"
    is_eos: bool = False
    multiplicity: int = 1
    completion_count: int = 0


@dataclass(frozen=True, slots=True)
class SouthStarOnlineChoiceResult:
    choices: tuple[SouthStarOnlineChoice, ...]
    stats: OnlineStateDecoderStats


@dataclass(frozen=True, slots=True)
class SouthStarOnlineDecoderState:
    prefix: str
    raw_state: OnlineDecoderState
    decoder: "SouthStarOnlineDecoder"

    def choices(self) -> tuple[SouthStarOnlineChoice, ...]:
        return self.decoder.choices(self)

    def choices_with_stats(self) -> SouthStarOnlineChoiceResult:
        return self.decoder.choices_with_stats(self)


@dataclass(frozen=True, slots=True)
class SouthStarOnlineDecoder:
    facts: MoleculeFacts
    policy: SmilesPolicy
    semantics: ParserSemantics
    branch_mode: Literal["branch_preserving", "determinized"] = "determinized"
    compaction_mode: FrontierCompactionMode = FrontierCompactionMode.TRAVERSAL_ONLY
    include_eos: bool = False

    def initial_state(self) -> SouthStarOnlineDecoderState:
        raw = OnlineDecoderState(prefix="")
        return SouthStarOnlineDecoderState(prefix="", raw_state=raw, decoder=self)

    def choices(
        self,
        state: SouthStarOnlineDecoderState,
    ) -> tuple[SouthStarOnlineChoice, ...]:
        return self.choices_with_stats(state).choices

    def choices_with_stats(
        self,
        state: SouthStarOnlineDecoderState,
    ) -> SouthStarOnlineChoiceResult:
        _validate_state_belongs_to_decoder(state, self)
        raw_choices, stats = self._raw_choices_with_stats(state.raw_state)
        out = [
            SouthStarOnlineChoice(
                text=choice.text,
                next_state=SouthStarOnlineDecoderState(
                    prefix=choice.next_state.prefix,
                    raw_state=choice.next_state,
                    decoder=self,
                ),
                multiplicity=choice.multiplicity,
                completion_count=choice.completion_count,
            )
            for choice in raw_choices
        ]
        if self.include_eos and _prefix_is_complete_witness(
            facts=self.facts,
            policy=self.policy,
            semantics=self.semantics,
            prefix=state.prefix,
        ):
            out.append(
                SouthStarOnlineChoice(
                    text=EOS,
                    next_state=None,
                    is_eos=True,
                    multiplicity=1,
                    completion_count=1,
                )
            )
        return SouthStarOnlineChoiceResult(choices=tuple(out), stats=stats)

    def _raw_choices_with_stats(
        self,
        state: OnlineDecoderState,
    ):
        if self.branch_mode == "branch_preserving":
            return online_branch_preserving_choices_with_stats(
                facts=self.facts,
                policy=self.policy,
                semantics=self.semantics,
                state=state,
                compaction_mode=self.compaction_mode,
            )
        if self.branch_mode == "determinized":
            return online_determinized_choices_with_stats(
                facts=self.facts,
                policy=self.policy,
                semantics=self.semantics,
                state=state,
                compaction_mode=self.compaction_mode,
            )
        raise ValueError(f"unknown online decoder branch mode: {self.branch_mode!r}")


def make_branch_preserving_online_decoder(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    compaction_mode: FrontierCompactionMode = FrontierCompactionMode.TRAVERSAL_ONLY,
    include_eos: bool = False,
) -> SouthStarOnlineDecoder:
    return SouthStarOnlineDecoder(
        facts=facts,
        policy=policy,
        semantics=semantics,
        branch_mode="branch_preserving",
        compaction_mode=compaction_mode,
        include_eos=include_eos,
    )


def make_determinized_online_decoder(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    compaction_mode: FrontierCompactionMode = FrontierCompactionMode.TRAVERSAL_ONLY,
    include_eos: bool = False,
) -> SouthStarOnlineDecoder:
    return SouthStarOnlineDecoder(
        facts=facts,
        policy=policy,
        semantics=semantics,
        branch_mode="determinized",
        compaction_mode=compaction_mode,
        include_eos=include_eos,
    )


def online_decode_token_texts_for_policy(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    include_eos: bool = False,
) -> tuple[str, ...]:
    policy.validate_for_facts(facts)
    token_texts = {token.text for token in online_decode_tokens(policy)}
    if include_eos:
        token_texts.add(EOS)
    return tuple(sorted(token_texts))


def _validate_state_belongs_to_decoder(
    state: SouthStarOnlineDecoderState,
    decoder: SouthStarOnlineDecoder,
) -> None:
    if state.decoder is not decoder:
        raise ValueError("online decoder state belongs to a different decoder")
    if state.prefix != state.raw_state.prefix:
        raise ValueError("online decoder state prefix does not match raw state")


def _prefix_is_complete_witness(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    prefix: str,
) -> bool:
    return any(
        witness == prefix
        for witness in iter_online_stereo_witness_strings(
            facts=facts,
            policy=policy,
            semantics=semantics,
        )
    )


__all__ = (
    "EOS",
    "SouthStarOnlineChoice",
    "SouthStarOnlineChoiceResult",
    "SouthStarOnlineDecoder",
    "SouthStarOnlineDecoderState",
    "make_branch_preserving_online_decoder",
    "make_determinized_online_decoder",
    "online_decode_token_texts_for_policy",
)
