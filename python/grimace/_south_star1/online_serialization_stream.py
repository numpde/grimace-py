"""Online serialization stream built on the South Star decoder facade."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from .facts import MoleculeFacts
from .online_continuation import OnlineDecoderExecutionMode
from .online_decoder_api import make_branch_preserving_online_decoder
from .online_decoder_api import make_determinized_online_decoder
from .policy import SmilesPolicy
from .prepared_runtime import SouthStarPreparedMol
from .prepared_runtime import SouthStarRuntimeOptions
from .prepared_runtime import SouthStarWriterSurface
from .prepared_runtime import prepare_south_star_mol_from_facts
from .prepared_runtime import validate_south_star_runtime_options
from .semantics import ParserSemantics


@dataclass(frozen=True, slots=True)
class OnlineSerialization:
    text: str
    completion_count: int
    multiplicity: int


@dataclass(frozen=True, slots=True)
class OnlineSerializationStreamStats:
    frontier_queries: int
    emitted_support_count: int
    witness_completion_count: int
    max_choice_count: int
    max_retained_continuation_count: int | None = None


@dataclass(frozen=True, slots=True)
class OnlineSerializationSupportResult:
    strings: tuple[str, ...]
    support_count: int
    witness_completion_count: int
    stats: OnlineSerializationStreamStats


def iter_online_serializations(
    *,
    prepared: SouthStarPreparedMol | None = None,
    facts: MoleculeFacts | None = None,
    policy: SmilesPolicy | None = None,
    semantics: ParserSemantics | None = None,
    writer_surface: SouthStarWriterSurface = SouthStarWriterSurface(),
    runtime_options: SouthStarRuntimeOptions = SouthStarRuntimeOptions(),
    execution_mode: OnlineDecoderExecutionMode = OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
    branch_mode: Literal["determinized", "branch_preserving"] = "determinized",
) -> Iterator[OnlineSerialization]:
    """Yield complete online serialization strings from decoder EOS choices."""

    validate_south_star_runtime_options(runtime_options)
    prepared = _resolve_prepared(
        prepared=prepared,
        facts=facts,
        policy=policy,
        semantics=semantics,
        writer_surface=writer_surface,
    )
    decoder = _make_decoder(
        prepared=prepared,
        runtime_options=runtime_options,
        execution_mode=execution_mode,
        branch_mode=branch_mode,
    )
    stack = [decoder.initial_state()]
    emitted: set[str] = set()
    while stack:
        state = stack.pop()
        result = state.choices_with_stats()
        for choice in reversed(result.choices):
            if choice.is_eos:
                if branch_mode == "determinized":
                    if state.prefix in emitted:
                        raise ValueError(
                            "online serialization stream emitted duplicate "
                            f"support string: {state.prefix!r}"
                        )
                    emitted.add(state.prefix)
                yield OnlineSerialization(
                    text=state.prefix,
                    completion_count=choice.completion_count,
                    multiplicity=choice.multiplicity,
                )
                continue
            if choice.next_state is None:
                raise ValueError("non-EOS online serialization choice lacks next_state")
            stack.append(choice.next_state)


def collect_online_serializations(
    *,
    prepared: SouthStarPreparedMol | None = None,
    facts: MoleculeFacts | None = None,
    policy: SmilesPolicy | None = None,
    semantics: ParserSemantics | None = None,
    writer_surface: SouthStarWriterSurface = SouthStarWriterSurface(),
    runtime_options: SouthStarRuntimeOptions = SouthStarRuntimeOptions(),
    execution_mode: OnlineDecoderExecutionMode = OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
) -> OnlineSerializationSupportResult:
    """Materialize the determinized online serialization support and counts."""

    validate_south_star_runtime_options(runtime_options)
    prepared = _resolve_prepared(
        prepared=prepared,
        facts=facts,
        policy=policy,
        semantics=semantics,
        writer_surface=writer_surface,
    )
    decoder = make_determinized_online_decoder(
        prepared=prepared,
        include_eos=True,
        runtime_options=runtime_options,
        execution_mode=execution_mode,
    )
    stack = [decoder.initial_state()]
    strings: list[str] = []
    emitted: set[str] = set()
    frontier_queries = 0
    witness_completion_count = 0
    max_choice_count = 0
    max_retained_continuation_count: int | None = None
    while stack:
        state = stack.pop()
        result = state.choices_with_stats()
        frontier_queries += 1
        max_choice_count = max(max_choice_count, len(result.choices))
        retained_count = _retained_continuation_count(result.stats)
        if retained_count is not None:
            if max_retained_continuation_count is None:
                max_retained_continuation_count = retained_count
            else:
                max_retained_continuation_count = max(
                    max_retained_continuation_count,
                    retained_count,
                )
        for choice in reversed(result.choices):
            if choice.is_eos:
                if state.prefix in emitted:
                    raise ValueError(
                        "online serialization collection emitted duplicate "
                        f"support string: {state.prefix!r}"
                    )
                emitted.add(state.prefix)
                strings.append(state.prefix)
                witness_completion_count += choice.completion_count
                continue
            if choice.next_state is None:
                raise ValueError("non-EOS online serialization choice lacks next_state")
            stack.append(choice.next_state)
    stats = OnlineSerializationStreamStats(
        frontier_queries=frontier_queries,
        emitted_support_count=len(strings),
        witness_completion_count=witness_completion_count,
        max_choice_count=max_choice_count,
        max_retained_continuation_count=max_retained_continuation_count,
    )
    return OnlineSerializationSupportResult(
        strings=tuple(strings),
        support_count=len(strings),
        witness_completion_count=witness_completion_count,
        stats=stats,
    )


def _make_decoder(
    *,
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions,
    execution_mode: OnlineDecoderExecutionMode,
    branch_mode: Literal["determinized", "branch_preserving"],
):
    if branch_mode == "determinized":
        return make_determinized_online_decoder(
            prepared=prepared,
            include_eos=True,
            runtime_options=runtime_options,
            execution_mode=execution_mode,
        )
    if branch_mode == "branch_preserving":
        return make_branch_preserving_online_decoder(
            prepared=prepared,
            include_eos=True,
            runtime_options=runtime_options,
            execution_mode=execution_mode,
        )
    raise ValueError(f"unknown online serialization branch mode: {branch_mode!r}")


def _resolve_prepared(
    *,
    prepared: SouthStarPreparedMol | None,
    facts: MoleculeFacts | None,
    policy: SmilesPolicy | None,
    semantics: ParserSemantics | None,
    writer_surface: SouthStarWriterSurface,
) -> SouthStarPreparedMol:
    if prepared is not None:
        if facts is not None or policy is not None or semantics is not None:
            raise ValueError("prepared serialization input cannot be mixed with raw inputs")
        return prepared
    if facts is None:
        raise ValueError("online serialization requires prepared or facts")
    return prepare_south_star_mol_from_facts(
        facts,
        writer_surface=writer_surface,
        policy=policy,
        semantics=semantics,
    )


def _retained_continuation_count(stats: object) -> int | None:
    retained = getattr(stats, "retained_state_size", None)
    if retained is None:
        return None
    return int(retained.continuation_count)


__all__ = (
    "OnlineSerialization",
    "OnlineSerializationStreamStats",
    "OnlineSerializationSupportResult",
    "collect_online_serializations",
    "iter_online_serializations",
)
