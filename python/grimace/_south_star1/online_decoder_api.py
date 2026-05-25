"""Main-branch-shaped facade for South Star online decoder states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .facts import MoleculeFacts
from .graph_index import GraphIndex
from .ids import AtomId
from .online_decoder import online_decode_tokens
from .online_continuation import OnlineContinuationDecoderState
from .online_continuation import OnlineContinuationRawChoiceResult
from .online_continuation import OnlineContinuationStats
from .online_continuation import OnlineDecoderExecutionMode
from .online_continuation import online_branch_preserving_continuation_choice_result
from .online_continuation import online_determinized_continuation_choice_result
from .online_decisions import FrontierCompactionMode
from .online_decoder_state import OnlineDecoderState
from .online_decoder_state import OnlineRawChoiceResult
from .online_decoder_state import OnlineStateDecoderStats
from .online_decoder_state import online_branch_preserving_choice_result
from .online_decoder_state import online_determinized_choice_result
from .online_residual_continuation import OnlineResidualDecoderState
from .online_residual_continuation import OnlineResidualDecoderStats
from .online_residual_continuation import OnlineResidualRawChoiceResult
from .online_residual_continuation import online_branch_preserving_residual_choice_result
from .online_residual_continuation import online_determinized_residual_choice_result
from .policy import SmilesPolicy
from .prepared_runtime import SouthStarPreparedMol
from .prepared_runtime import SouthStarRuntimeOptions
from .prepared_runtime import component_root_domains_for_prepared
from .prepared_runtime import runtime_root_atom
from .prepared_runtime import runtime_root_atom_for_prepared
from .prepared_runtime import validate_south_star_runtime_options
from .semantics import ParserSemantics
from .stereo_templates import StereoTemplateBundle


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
    stats: OnlineStateDecoderStats | OnlineContinuationStats | OnlineResidualDecoderStats


@dataclass(frozen=True, slots=True)
class SouthStarOnlineDecoderState:
    prefix: str
    raw_state: OnlineDecoderState | OnlineContinuationDecoderState | OnlineResidualDecoderState
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
    prepared: SouthStarPreparedMol | None = None
    templates: StereoTemplateBundle | None = None
    rooted_at_atom: AtomId | None = None
    graph_index: GraphIndex | None = None
    component_root_domains: tuple[tuple[AtomId, ...], ...] | None = None
    runtime_options: SouthStarRuntimeOptions = SouthStarRuntimeOptions()
    branch_mode: Literal["branch_preserving", "determinized"] = "determinized"
    compaction_mode: FrontierCompactionMode = FrontierCompactionMode.TRAVERSAL_ONLY
    include_eos: bool = False
    execution_mode: OnlineDecoderExecutionMode = OnlineDecoderExecutionMode.PREFIX_REPLAY

    def __post_init__(self) -> None:
        expected_root = (
            runtime_root_atom_for_prepared(self.runtime_options, prepared=self.prepared)
            if self.prepared is not None
            else runtime_root_atom(self.runtime_options, facts=self.facts)
        )
        if self.rooted_at_atom != expected_root:
            raise ValueError("online decoder rooted_at_atom does not match runtime_options")

    def initial_state(self) -> SouthStarOnlineDecoderState:
        if self.execution_mode is OnlineDecoderExecutionMode.PREFIX_REPLAY:
            raw: OnlineDecoderState | OnlineContinuationDecoderState = OnlineDecoderState(prefix="")
        elif self.execution_mode is OnlineDecoderExecutionMode.CACHED_COMPLETIONS:
            raw = OnlineContinuationDecoderState(prefix="")
        elif self.execution_mode is OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS:
            raw = OnlineResidualDecoderState(prefix="")
        else:
            raise ValueError(f"unknown online decoder execution mode: {self.execution_mode!r}")
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
        raw_result = self._raw_choice_result(state.raw_state)
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
            for choice in raw_result.choices
        ]
        if self.include_eos and raw_result.eos_completion_count:
            out.append(
                SouthStarOnlineChoice(
                    text=EOS,
                    next_state=None,
                    is_eos=True,
                    multiplicity=len(raw_result.eos_frontier.paths),
                    completion_count=raw_result.eos_completion_count,
                )
            )
        return SouthStarOnlineChoiceResult(choices=tuple(out), stats=raw_result.stats)

    def _raw_choice_result(
        self,
        state: OnlineDecoderState | OnlineContinuationDecoderState | OnlineResidualDecoderState,
    ) -> OnlineRawChoiceResult | OnlineContinuationRawChoiceResult | OnlineResidualRawChoiceResult:
        if self.execution_mode is OnlineDecoderExecutionMode.CACHED_COMPLETIONS:
            if not isinstance(state, OnlineContinuationDecoderState):
                raise ValueError("cached-completion decoder received prefix-replay state")
            if self.branch_mode == "branch_preserving":
                return online_branch_preserving_continuation_choice_result(
                    facts=self.facts,
                    policy=self.policy,
                    semantics=self.semantics,
                    state=state,
                    compaction_mode=self.compaction_mode,
                    templates=self.templates,
                    rooted_at_atom=self.rooted_at_atom,
                    graph_index=self.graph_index,
                    component_root_domains=self.component_root_domains,
                    prepared=self.prepared,
                )
            if self.branch_mode == "determinized":
                return online_determinized_continuation_choice_result(
                    facts=self.facts,
                    policy=self.policy,
                    semantics=self.semantics,
                    state=state,
                    compaction_mode=self.compaction_mode,
                    templates=self.templates,
                    rooted_at_atom=self.rooted_at_atom,
                    graph_index=self.graph_index,
                    component_root_domains=self.component_root_domains,
                    prepared=self.prepared,
                )
            raise ValueError(f"unknown online decoder branch mode: {self.branch_mode!r}")
        if self.execution_mode is OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS:
            if not isinstance(state, OnlineResidualDecoderState):
                raise ValueError("residual-continuation decoder received non-residual state")
            if self.branch_mode == "branch_preserving":
                return online_branch_preserving_residual_choice_result(
                    facts=self.facts,
                    policy=self.policy,
                    semantics=self.semantics,
                    state=state,
                    templates=self.templates,
                    rooted_at_atom=self.rooted_at_atom,
                    graph_index=self.graph_index,
                    component_root_domains=self.component_root_domains,
                    prepared=self.prepared,
                )
            if self.branch_mode == "determinized":
                return online_determinized_residual_choice_result(
                    facts=self.facts,
                    policy=self.policy,
                    semantics=self.semantics,
                    state=state,
                    templates=self.templates,
                    rooted_at_atom=self.rooted_at_atom,
                    graph_index=self.graph_index,
                    component_root_domains=self.component_root_domains,
                    prepared=self.prepared,
                )
            raise ValueError(f"unknown online decoder branch mode: {self.branch_mode!r}")

        if not isinstance(state, OnlineDecoderState):
            raise ValueError("prefix-replay decoder received continuation state")
        if self.branch_mode == "branch_preserving":
            return online_branch_preserving_choice_result(
                facts=self.facts,
                policy=self.policy,
                semantics=self.semantics,
                state=state,
                compaction_mode=self.compaction_mode,
                templates=self.templates,
                rooted_at_atom=self.rooted_at_atom,
                graph_index=self.graph_index,
                component_root_domains=self.component_root_domains,
                prepared=self.prepared,
            )
        if self.branch_mode == "determinized":
            return online_determinized_choice_result(
                facts=self.facts,
                policy=self.policy,
                semantics=self.semantics,
                state=state,
                compaction_mode=self.compaction_mode,
                templates=self.templates,
                rooted_at_atom=self.rooted_at_atom,
                graph_index=self.graph_index,
                component_root_domains=self.component_root_domains,
                prepared=self.prepared,
            )
        raise ValueError(f"unknown online decoder branch mode: {self.branch_mode!r}")


def make_branch_preserving_online_decoder(
    *,
    prepared: SouthStarPreparedMol | None = None,
    facts: MoleculeFacts | None = None,
    policy: SmilesPolicy | None = None,
    semantics: ParserSemantics | None = None,
    compaction_mode: FrontierCompactionMode = FrontierCompactionMode.TRAVERSAL_ONLY,
    include_eos: bool = False,
    execution_mode: OnlineDecoderExecutionMode = OnlineDecoderExecutionMode.PREFIX_REPLAY,
    runtime_options: SouthStarRuntimeOptions = SouthStarRuntimeOptions(),
) -> SouthStarOnlineDecoder:
    facts, policy, semantics, templates, graph_index = _resolve_decoder_inputs(
        prepared=prepared,
        facts=facts,
        policy=policy,
        semantics=semantics,
    )
    rooted_at_atom = _runtime_root_atom_for_decoder(runtime_options, prepared, facts)
    root_domains = _prepared_root_domains(
        prepared=prepared,
        rooted_at_atom=rooted_at_atom,
    )
    return SouthStarOnlineDecoder(
        prepared=prepared,
        facts=facts,
        policy=policy,
        semantics=semantics,
        templates=templates,
        rooted_at_atom=rooted_at_atom,
        graph_index=graph_index,
        component_root_domains=root_domains,
        runtime_options=runtime_options,
        branch_mode="branch_preserving",
        compaction_mode=compaction_mode,
        include_eos=include_eos,
        execution_mode=execution_mode,
    )


def make_determinized_online_decoder(
    *,
    prepared: SouthStarPreparedMol | None = None,
    facts: MoleculeFacts | None = None,
    policy: SmilesPolicy | None = None,
    semantics: ParserSemantics | None = None,
    compaction_mode: FrontierCompactionMode = FrontierCompactionMode.TRAVERSAL_ONLY,
    include_eos: bool = False,
    execution_mode: OnlineDecoderExecutionMode = OnlineDecoderExecutionMode.PREFIX_REPLAY,
    runtime_options: SouthStarRuntimeOptions = SouthStarRuntimeOptions(),
) -> SouthStarOnlineDecoder:
    facts, policy, semantics, templates, graph_index = _resolve_decoder_inputs(
        prepared=prepared,
        facts=facts,
        policy=policy,
        semantics=semantics,
    )
    rooted_at_atom = _runtime_root_atom_for_decoder(runtime_options, prepared, facts)
    root_domains = _prepared_root_domains(
        prepared=prepared,
        rooted_at_atom=rooted_at_atom,
    )
    return SouthStarOnlineDecoder(
        prepared=prepared,
        facts=facts,
        policy=policy,
        semantics=semantics,
        templates=templates,
        rooted_at_atom=rooted_at_atom,
        graph_index=graph_index,
        component_root_domains=root_domains,
        runtime_options=runtime_options,
        branch_mode="determinized",
        compaction_mode=compaction_mode,
        include_eos=include_eos,
        execution_mode=execution_mode,
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


def _resolve_decoder_inputs(
    *,
    prepared: SouthStarPreparedMol | None,
    facts: MoleculeFacts | None,
    policy: SmilesPolicy | None,
    semantics: ParserSemantics | None,
) -> tuple[
    MoleculeFacts,
    SmilesPolicy,
    ParserSemantics,
    StereoTemplateBundle | None,
    GraphIndex | None,
]:
    if prepared is not None:
        if facts is not None or policy is not None or semantics is not None:
            raise ValueError("prepared decoder input cannot be mixed with raw inputs")
        return (
            prepared.facts,
            prepared.policy,
            prepared.semantics,
            prepared.stereo_template_bundle(),
            prepared.graph_index,
        )
    if facts is None or policy is None or semantics is None:
        raise ValueError("decoder construction requires prepared or facts/policy/semantics")
    return facts, policy, semantics, None, None


def _prepared_root_domains(
    *,
    prepared: SouthStarPreparedMol | None,
    rooted_at_atom: AtomId | None,
) -> tuple[tuple[AtomId, ...], ...] | None:
    if prepared is None:
        return None
    return tuple(
        atoms
        for _, atoms in component_root_domains_for_prepared(
            prepared=prepared,
            rooted_at_atom=rooted_at_atom,
        )
    )


def _runtime_root_atom_for_decoder(
    runtime_options: SouthStarRuntimeOptions,
    prepared: SouthStarPreparedMol | None,
    facts: MoleculeFacts,
) -> AtomId | None:
    if prepared is not None:
        return runtime_root_atom_for_prepared(runtime_options, prepared=prepared)
    validate_south_star_runtime_options(runtime_options, facts=facts)
    return runtime_root_atom(runtime_options, facts=facts)


def _validate_state_belongs_to_decoder(
    state: SouthStarOnlineDecoderState,
    decoder: SouthStarOnlineDecoder,
) -> None:
    if state.decoder is not decoder:
        raise ValueError("online decoder state belongs to a different decoder")
    if state.prefix != state.raw_state.prefix:
        raise ValueError("online decoder state prefix does not match raw state")


__all__ = (
    "EOS",
    "OnlineDecoderExecutionMode",
    "SouthStarOnlineChoice",
    "SouthStarOnlineChoiceResult",
    "SouthStarOnlineDecoder",
    "SouthStarOnlineDecoderState",
    "make_branch_preserving_online_decoder",
    "make_determinized_online_decoder",
    "online_decode_token_texts_for_policy",
)
