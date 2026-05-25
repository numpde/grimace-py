"""Prepared prefix-query workload checks for South Star online decoders."""

from __future__ import annotations

from dataclasses import dataclass

from .online_continuation import OnlineDecoderExecutionMode
from .online_decoder_api import SouthStarOnlineDecoder
from .online_decoder_api import SouthStarOnlineDecoderState
from .online_decoder_api import make_determinized_online_decoder
from .prepared_bench_matrix import PreparedRuntimeProbe
from .prepared_bench_matrix import PreparedRuntimeProbeResult
from .prepared_runtime import SouthStarPreparedMol
from .prepared_runtime import SouthStarRuntimeOptions


@dataclass(frozen=True, slots=True)
class PreparedPrefixQueryObservation:
    prefix: str
    execution_mode: OnlineDecoderExecutionMode
    next_token_texts: tuple[str, ...]
    next_token_completion_counts: tuple[tuple[str, int], ...]
    has_eos: bool
    eos_completion_count: int
    frontier_queries: int
    root_dfs_runs: int | None
    resumed_snapshots: int | None
    retained_continuation_count: int | None
    retained_render_payload_chars: int | None


@dataclass(frozen=True, slots=True)
class PreparedPrefixWorkloadRow:
    fixture_name: str
    rooted_at_atom: int
    prefix: str
    prefix_replay: PreparedPrefixQueryObservation
    cached_completions: PreparedPrefixQueryObservation
    residual_continuations: PreparedPrefixQueryObservation


@dataclass(frozen=True, slots=True)
class PreparedPrefixWorkloadResult:
    rows: tuple[PreparedPrefixWorkloadRow, ...]
    total_prefix_replay_root_dfs_runs: int
    total_residual_root_dfs_runs: int
    total_residual_resumed_snapshots: int
    max_residual_retained_render_payload_chars: int
    probe: PreparedRuntimeProbeResult


def collect_token_boundary_prefixes(
    *,
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions = SouthStarRuntimeOptions(),
    limit: int | None = None,
) -> tuple[str, ...]:
    """Collect reachable decoder-token-boundary prefixes in deterministic order."""

    decoder = make_determinized_online_decoder(
        prepared=prepared,
        include_eos=True,
        runtime_options=runtime_options,
        execution_mode=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
    )
    stack = [decoder.initial_state()]
    seen: set[str] = set()
    prefixes: list[str] = []
    while stack:
        state = stack.pop()
        if state.prefix in seen:
            continue
        seen.add(state.prefix)
        prefixes.append(state.prefix)
        if limit is not None and len(prefixes) >= limit:
            break
        result = state.choices_with_stats()
        for choice in reversed(result.choices):
            if choice.is_eos:
                continue
            if choice.next_state is None:
                raise ValueError("non-EOS prefix workload choice lacks next_state")
            stack.append(choice.next_state)
    return tuple(prefixes)


def collect_prepared_prefix_workload(
    *,
    fixture_name: str,
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions = SouthStarRuntimeOptions(),
    prefix_limit: int | None = 16,
) -> PreparedPrefixWorkloadResult:
    with PreparedRuntimeProbe() as probe:
        prefixes = collect_token_boundary_prefixes(
            prepared=prepared,
            runtime_options=runtime_options,
            limit=prefix_limit,
        )
        rows = tuple(
            _collect_row(
                fixture_name=fixture_name,
                prepared=prepared,
                runtime_options=runtime_options,
                prefix=prefix,
            )
            for prefix in prefixes
        )
    return _result_from_rows(rows, probe.result())


def advance_decoder_to_prefix(
    decoder: SouthStarOnlineDecoder,
    prefix: str,
) -> SouthStarOnlineDecoderState:
    """Advance through legal determinized token choices to an exact prefix."""

    stack = [decoder.initial_state()]
    seen: set[str] = set()
    while stack:
        state = stack.pop()
        if state.prefix == prefix:
            return state
        if state.prefix in seen:
            continue
        seen.add(state.prefix)
        if len(state.prefix) > len(prefix):
            continue
        if not prefix.startswith(state.prefix):
            continue
        result = state.choices_with_stats()
        for choice in reversed(result.choices):
            if choice.is_eos:
                continue
            if choice.next_state is None:
                raise ValueError("non-EOS prefix advance choice lacks next_state")
            next_prefix = choice.next_state.prefix
            if prefix.startswith(next_prefix) or next_prefix.startswith(prefix):
                stack.append(choice.next_state)
    raise ValueError(f"prefix is not reachable at a decoder token boundary: {prefix!r}")


def validate_prepared_prefix_workload_result(
    result: PreparedPrefixWorkloadResult,
) -> None:
    for row in result.rows:
        _validate_row(row)
    if not result.rows:
        raise ValueError("prepared prefix workload produced no rows")
    if result.total_residual_root_dfs_runs >= result.total_prefix_replay_root_dfs_runs:
        raise ValueError("residual prefix workload did not reduce root DFS runs")
    if result.total_residual_resumed_snapshots <= 0:
        raise ValueError("residual prefix workload did not resume snapshots")
    if result.max_residual_retained_render_payload_chars != 0:
        raise ValueError("residual prefix workload retained rendered-suffix payload")


def _collect_row(
    *,
    fixture_name: str,
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions,
    prefix: str,
) -> PreparedPrefixWorkloadRow:
    prefix_replay = _observe_prefix_query(
        prepared=prepared,
        runtime_options=runtime_options,
        prefix=prefix,
        execution_mode=OnlineDecoderExecutionMode.PREFIX_REPLAY,
    )
    cached = _observe_prefix_query(
        prepared=prepared,
        runtime_options=runtime_options,
        prefix=prefix,
        execution_mode=OnlineDecoderExecutionMode.CACHED_COMPLETIONS,
    )
    residual = _observe_prefix_query(
        prepared=prepared,
        runtime_options=runtime_options,
        prefix=prefix,
        execution_mode=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
    )
    return PreparedPrefixWorkloadRow(
        fixture_name=fixture_name,
        rooted_at_atom=runtime_options.rooted_at_atom,
        prefix=prefix,
        prefix_replay=prefix_replay,
        cached_completions=cached,
        residual_continuations=residual,
    )


def _observe_prefix_query(
    *,
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions,
    prefix: str,
    execution_mode: OnlineDecoderExecutionMode,
) -> PreparedPrefixQueryObservation:
    decoder = make_determinized_online_decoder(
        prepared=prepared,
        include_eos=True,
        runtime_options=runtime_options,
        execution_mode=execution_mode,
    )
    state = advance_decoder_to_prefix(decoder, prefix)
    result = state.choices_with_stats()
    eos_completion_count = sum(
        choice.completion_count for choice in result.choices if choice.is_eos
    )
    retained = getattr(result.stats, "retained_state_size", None)
    return PreparedPrefixQueryObservation(
        prefix=prefix,
        execution_mode=execution_mode,
        next_token_texts=tuple(
            choice.text for choice in result.choices if not choice.is_eos
        ),
        next_token_completion_counts=tuple(
            (choice.text, choice.completion_count)
            for choice in result.choices
            if not choice.is_eos
        ),
        has_eos=eos_completion_count > 0,
        eos_completion_count=eos_completion_count,
        frontier_queries=1,
        root_dfs_runs=_root_dfs_runs(result.stats),
        resumed_snapshots=_resumed_snapshots(result.stats),
        retained_continuation_count=(
            None if retained is None else int(retained.continuation_count)
        ),
        retained_render_payload_chars=(
            None if retained is None else int(retained.total_render_payload_chars)
        ),
    )


def _result_from_rows(
    rows: tuple[PreparedPrefixWorkloadRow, ...],
    probe: PreparedRuntimeProbeResult,
) -> PreparedPrefixWorkloadResult:
    residual_payloads = tuple(
        row.residual_continuations.retained_render_payload_chars or 0
        for row in rows
    )
    return PreparedPrefixWorkloadResult(
        rows=rows,
        total_prefix_replay_root_dfs_runs=sum(
            row.prefix_replay.root_dfs_runs or 0 for row in rows
        ),
        total_residual_root_dfs_runs=sum(
            row.residual_continuations.root_dfs_runs or 0 for row in rows
        ),
        total_residual_resumed_snapshots=sum(
            row.residual_continuations.resumed_snapshots or 0 for row in rows
        ),
        max_residual_retained_render_payload_chars=(
            max(residual_payloads) if residual_payloads else 0
        ),
        probe=probe,
    )


def _validate_row(row: PreparedPrefixWorkloadRow) -> None:
    observations = (
        row.prefix_replay,
        row.cached_completions,
        row.residual_continuations,
    )
    next_tokens = {item.next_token_texts for item in observations}
    if len(next_tokens) != 1:
        raise ValueError(f"prefix workload next-token disagreement at {row.prefix!r}")
    next_completion_counts = {
        item.next_token_completion_counts for item in observations
    }
    if len(next_completion_counts) != 1:
        raise ValueError(
            f"prefix workload next-token completion count disagreement at {row.prefix!r}"
        )
    eos_availability = {item.has_eos for item in observations}
    if len(eos_availability) != 1:
        raise ValueError(f"prefix workload EOS availability disagreement at {row.prefix!r}")
    eos_counts = {item.eos_completion_count for item in observations}
    if len(eos_counts) != 1:
        raise ValueError(f"prefix workload EOS count disagreement at {row.prefix!r}")


def _root_dfs_runs(stats: object) -> int | None:
    if hasattr(stats, "root_dfs_runs"):
        return int(getattr(stats, "root_dfs_runs"))
    if hasattr(stats, "dfs_runs"):
        return int(getattr(stats, "dfs_runs"))
    return None


def _resumed_snapshots(stats: object) -> int | None:
    if hasattr(stats, "resumed_snapshots"):
        return int(getattr(stats, "resumed_snapshots"))
    return None


__all__ = (
    "PreparedPrefixQueryObservation",
    "PreparedPrefixWorkloadResult",
    "PreparedPrefixWorkloadRow",
    "advance_decoder_to_prefix",
    "collect_prepared_prefix_workload",
    "collect_token_boundary_prefixes",
    "validate_prepared_prefix_workload_result",
)
