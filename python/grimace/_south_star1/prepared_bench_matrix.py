"""Prepared South Star conformance and structural-efficiency matrix helpers."""

from __future__ import annotations

from dataclasses import dataclass

from .online_continuation import OnlineDecoderExecutionMode
from .online_decoder_api import make_determinized_online_decoder
from .prepared_runtime import SouthStarPreparedMol
from .prepared_runtime import SouthStarRuntimeOptions
from .prepared_runtime import enumerate_prepared_stereo_support


@dataclass(frozen=True, slots=True)
class PreparedEnumerationMatrixRow:
    fixture_name: str
    rooted_at_atom: int
    execution_mode: OnlineDecoderExecutionMode
    offline_support_count: int
    online_support_count: int
    offline_witness_count: int
    online_witness_completion_count: int
    frontier_queries: int
    max_choice_count: int
    max_pending_stream_states: int
    max_retained_continuations: int | None
    root_dfs_runs: int | None
    resumed_snapshots: int | None
    retained_render_payload_chars: int | None
    retained_render_cursor_count: int | None
    graph_rebuild_count_after_prepare: int
    root_domain_recompute_count_after_prepare: int
    stereo_template_rebuild_count_after_prepare: int
    facts_validate_count_after_prepare: int | None
    policy_validate_count_after_prepare: int | None


@dataclass(frozen=True, slots=True)
class PreparedEnumerationMatrixEntry:
    row: PreparedEnumerationMatrixRow
    offline_strings: frozenset[str]
    online_strings: frozenset[str]


@dataclass(frozen=True, slots=True)
class PreparedPrefixWorkloadStats:
    execution_mode: OnlineDecoderExecutionMode
    frontier_queries: int
    root_dfs_runs: int
    resumed_snapshots: int
    max_choice_count: int
    max_pending_states: int


def collect_prepared_enumeration_matrix_entry(
    *,
    fixture_name: str,
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions = SouthStarRuntimeOptions(),
    execution_mode: OnlineDecoderExecutionMode = OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
    graph_rebuild_count_after_prepare: int = 0,
    root_domain_recompute_count_after_prepare: int = 0,
    stereo_template_rebuild_count_after_prepare: int = 0,
    facts_validate_count_after_prepare: int | None = None,
    policy_validate_count_after_prepare: int | None = None,
) -> PreparedEnumerationMatrixEntry:
    offline = enumerate_prepared_stereo_support(
        prepared=prepared,
        runtime_options=runtime_options,
    )
    online = _walk_prepared_decoder(
        prepared=prepared,
        runtime_options=runtime_options,
        execution_mode=execution_mode,
    )
    row = PreparedEnumerationMatrixRow(
        fixture_name=fixture_name,
        rooted_at_atom=runtime_options.rooted_at_atom,
        execution_mode=execution_mode,
        offline_support_count=offline.distinct_count,
        online_support_count=len(online.strings),
        offline_witness_count=offline.witness_count,
        online_witness_completion_count=online.witness_completion_count,
        frontier_queries=online.frontier_queries,
        max_choice_count=online.max_choice_count,
        max_pending_stream_states=online.max_pending_stream_states,
        max_retained_continuations=online.max_retained_continuations,
        root_dfs_runs=online.root_dfs_runs,
        resumed_snapshots=online.resumed_snapshots,
        retained_render_payload_chars=online.retained_render_payload_chars,
        retained_render_cursor_count=online.retained_render_cursor_count,
        graph_rebuild_count_after_prepare=graph_rebuild_count_after_prepare,
        root_domain_recompute_count_after_prepare=root_domain_recompute_count_after_prepare,
        stereo_template_rebuild_count_after_prepare=stereo_template_rebuild_count_after_prepare,
        facts_validate_count_after_prepare=facts_validate_count_after_prepare,
        policy_validate_count_after_prepare=policy_validate_count_after_prepare,
    )
    return PreparedEnumerationMatrixEntry(
        row=row,
        offline_strings=frozenset(offline.strings),
        online_strings=frozenset(online.strings),
    )


def collect_prepared_prefix_workload_stats(
    *,
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions = SouthStarRuntimeOptions(),
    execution_mode: OnlineDecoderExecutionMode,
) -> PreparedPrefixWorkloadStats:
    online = _walk_prepared_decoder(
        prepared=prepared,
        runtime_options=runtime_options,
        execution_mode=execution_mode,
    )
    return PreparedPrefixWorkloadStats(
        execution_mode=execution_mode,
        frontier_queries=online.frontier_queries,
        root_dfs_runs=online.root_dfs_runs or 0,
        resumed_snapshots=online.resumed_snapshots or 0,
        max_choice_count=online.max_choice_count,
        max_pending_states=online.max_pending_stream_states,
    )


@dataclass(frozen=True, slots=True)
class _OnlineWalkResult:
    strings: tuple[str, ...]
    witness_completion_count: int
    frontier_queries: int
    max_choice_count: int
    max_pending_stream_states: int
    max_retained_continuations: int | None
    root_dfs_runs: int | None
    resumed_snapshots: int | None
    retained_render_payload_chars: int | None
    retained_render_cursor_count: int | None


def _walk_prepared_decoder(
    *,
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions,
    execution_mode: OnlineDecoderExecutionMode,
) -> _OnlineWalkResult:
    decoder = make_determinized_online_decoder(
        prepared=prepared,
        include_eos=True,
        runtime_options=runtime_options,
        execution_mode=execution_mode,
    )
    stack = [decoder.initial_state()]
    emitted: set[str] = set()
    strings: list[str] = []
    witness_completion_count = 0
    frontier_queries = 0
    max_choice_count = 0
    max_pending_stream_states = len(stack)
    max_retained_continuations: int | None = None
    root_dfs_runs = 0
    resumed_snapshots = 0
    retained_render_payload_chars: int | None = None
    retained_render_cursor_count: int | None = None

    while stack:
        max_pending_stream_states = max(max_pending_stream_states, len(stack))
        state = stack.pop()
        result = state.choices_with_stats()
        frontier_queries += 1
        max_choice_count = max(max_choice_count, len(result.choices))
        root_dfs_runs += _root_dfs_runs(result.stats)
        resumed_snapshots += int(getattr(result.stats, "resumed_snapshots", 0))
        retained = getattr(result.stats, "retained_state_size", None)
        if retained is not None:
            max_retained_continuations = _optional_max(
                max_retained_continuations,
                int(retained.continuation_count),
            )
            retained_render_payload_chars = _optional_max(
                retained_render_payload_chars,
                int(retained.total_render_payload_chars),
            )
            retained_render_cursor_count = _optional_max(
                retained_render_cursor_count,
                int(retained.render_cursor_count),
            )
        for choice in reversed(result.choices):
            if choice.is_eos:
                if state.prefix in emitted:
                    raise ValueError(
                        "prepared matrix walk emitted duplicate support string: "
                        f"{state.prefix!r}"
                    )
                emitted.add(state.prefix)
                strings.append(state.prefix)
                witness_completion_count += choice.completion_count
                continue
            if choice.next_state is None:
                raise ValueError("non-EOS prepared matrix choice lacks next_state")
            stack.append(choice.next_state)
        max_pending_stream_states = max(max_pending_stream_states, len(stack))

    return _OnlineWalkResult(
        strings=tuple(strings),
        witness_completion_count=witness_completion_count,
        frontier_queries=frontier_queries,
        max_choice_count=max_choice_count,
        max_pending_stream_states=max_pending_stream_states,
        max_retained_continuations=max_retained_continuations,
        root_dfs_runs=root_dfs_runs,
        resumed_snapshots=resumed_snapshots,
        retained_render_payload_chars=retained_render_payload_chars,
        retained_render_cursor_count=retained_render_cursor_count,
    )


def _root_dfs_runs(stats: object) -> int:
    if hasattr(stats, "root_dfs_runs"):
        return int(getattr(stats, "root_dfs_runs"))
    if hasattr(stats, "dfs_runs"):
        return int(getattr(stats, "dfs_runs"))
    return 0


def _optional_max(left: int | None, right: int) -> int:
    if left is None:
        return right
    return max(left, right)


__all__ = (
    "PreparedEnumerationMatrixEntry",
    "PreparedEnumerationMatrixRow",
    "PreparedPrefixWorkloadStats",
    "collect_prepared_enumeration_matrix_entry",
    "collect_prepared_prefix_workload_stats",
)
