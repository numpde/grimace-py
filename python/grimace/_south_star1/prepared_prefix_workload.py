"""Prepared prefix-query workload checks for South Star online decoders."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .online_continuation import OnlineDecoderExecutionMode
from .online_decoder_api import SouthStarOnlineDecoder
from .online_decoder_api import SouthStarOnlineDecoderState
from .online_decoder_api import make_determinized_online_decoder
from .prepared_bench_matrix import PreparedRuntimeProbe
from .prepared_bench_matrix import PreparedRuntimeProbeResult
from .prepared_runtime import SouthStarPreparedMol
from .prepared_runtime import SouthStarRuntimeOptions


_PREFIX_WORKLOAD_MODES = (
    OnlineDecoderExecutionMode.PREFIX_REPLAY,
    OnlineDecoderExecutionMode.CACHED_COMPLETIONS,
    OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
)


@dataclass(frozen=True, slots=True)
class PreparedPrefixQueryObservation:
    prefix: str
    execution_mode: OnlineDecoderExecutionMode
    next_token_texts: tuple[str, ...]
    next_token_text_set: frozenset[str]
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


@dataclass(frozen=True, slots=True)
class PreparedDecoderWalkStep:
    prefix: str
    selected_token: str | None
    next_token_set_by_mode: tuple[tuple[str, frozenset[str]], ...]
    eos_count_by_mode: tuple[tuple[str, int], ...]
    root_dfs_runs_by_mode: tuple[tuple[str, int | None], ...]
    resumed_snapshots_by_mode: tuple[tuple[str, int | None], ...]
    retained_render_payload_chars_by_mode: tuple[tuple[str, int | None], ...]


@dataclass(frozen=True, slots=True)
class PreparedDecoderWalkResult:
    fixture_name: str
    rooted_at_atom: int
    steps: tuple[PreparedDecoderWalkStep, ...]
    total_prefix_replay_root_dfs_runs: int
    total_cached_root_dfs_runs: int
    total_residual_root_dfs_runs: int
    total_residual_resumed_snapshots: int
    max_residual_retained_render_payload_chars: int
    probe: PreparedRuntimeProbeResult


@dataclass(frozen=True, slots=True)
class _PreparedStateQuery:
    observation: PreparedPrefixQueryObservation
    choices: tuple[object, ...]


def collect_token_boundary_prefixes(
    *,
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions = SouthStarRuntimeOptions(),
    limit: int | None = None,
) -> tuple[str, ...]:
    """Collect mode-independent decoder-token-boundary prefixes."""

    return collect_mode_union_token_boundary_prefixes(
        prepared=prepared,
        runtime_options=runtime_options,
        limit_per_mode=limit,
    )


def collect_mode_union_token_boundary_prefixes(
    *,
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions = SouthStarRuntimeOptions(),
    limit_per_mode: int | None = 16,
) -> tuple[str, ...]:
    """Collect the deterministic union of per-mode token-boundary prefixes."""

    seen: set[str] = set()
    prefixes: list[str] = []
    for mode in _PREFIX_WORKLOAD_MODES:
        for prefix in _collect_token_boundary_prefixes_for_mode(
            prepared=prepared,
            runtime_options=runtime_options,
            execution_mode=mode,
            limit=limit_per_mode,
        ):
            if prefix in seen:
                continue
            seen.add(prefix)
            prefixes.append(prefix)
    _require_prefixes_reachable_by_all_modes(
        prepared=prepared,
        runtime_options=runtime_options,
        prefixes=tuple(prefixes),
    )
    return tuple(prefixes)


def _collect_token_boundary_prefixes_for_mode(
    *,
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions,
    execution_mode: OnlineDecoderExecutionMode,
    limit: int | None,
) -> tuple[str, ...]:
    decoder = make_determinized_online_decoder(
        prepared=prepared,
        include_eos=True,
        runtime_options=runtime_options,
        execution_mode=execution_mode,
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
        prefixes = collect_mode_union_token_boundary_prefixes(
            prepared=prepared,
            runtime_options=runtime_options,
            limit_per_mode=prefix_limit,
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


def collect_prepared_decoder_walk(
    *,
    fixture_name: str,
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions = SouthStarRuntimeOptions(),
    max_steps: int = 32,
) -> PreparedDecoderWalkResult:
    with PreparedRuntimeProbe() as probe:
        states = {
            mode: make_determinized_online_decoder(
                prepared=prepared,
                include_eos=True,
                runtime_options=runtime_options,
                execution_mode=mode,
            ).initial_state()
            for mode in _PREFIX_WORKLOAD_MODES
        }
        steps: list[PreparedDecoderWalkStep] = []
        for _ in range(max_steps):
            queries = {
                mode: _query_state(mode=mode, state=states[mode])
                for mode in _PREFIX_WORKLOAD_MODES
            }
            observations = {
                mode: query.observation for mode, query in queries.items()
            }
            _validate_observations(
                states[OnlineDecoderExecutionMode.PREFIX_REPLAY].prefix,
                observations,
            )
            selected_token = choose_walk_token(
                observations[OnlineDecoderExecutionMode.PREFIX_REPLAY]
            )
            steps.append(_walk_step_from_observations(observations, selected_token))
            if selected_token is None:
                return _decoder_walk_result(
                    fixture_name=fixture_name,
                    rooted_at_atom=runtime_options.rooted_at_atom,
                    steps=tuple(steps),
                    probe=probe.result(),
                )
            states = {
                mode: _advance_state_by_token(
                    observation=observations[mode],
                    choices=queries[mode].choices,
                    token=selected_token,
                )
                for mode in _PREFIX_WORKLOAD_MODES
            }
        raise ValueError("prepared decoder walk exceeded max_steps")


def choose_walk_token(observation: PreparedPrefixQueryObservation) -> str | None:
    for token in observation.next_token_texts:
        return token
    return None


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


def _require_prefixes_reachable_by_all_modes(
    *,
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions,
    prefixes: tuple[str, ...],
) -> None:
    for prefix in prefixes:
        for mode in _PREFIX_WORKLOAD_MODES:
            decoder = make_determinized_online_decoder(
                prepared=prepared,
                include_eos=True,
                runtime_options=runtime_options,
                execution_mode=mode,
            )
            try:
                advance_decoder_to_prefix(decoder, prefix)
            except ValueError as exc:
                raise ValueError(
                    f"prefix {prefix!r} is not reachable by execution mode {mode.value}"
                ) from exc


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


def validate_prepared_decoder_walk_result(result: PreparedDecoderWalkResult) -> None:
    if not result.steps:
        raise ValueError("prepared decoder walk produced no steps")
    for step in result.steps:
        _validate_walk_step(step)
    if result.total_residual_root_dfs_runs >= result.total_prefix_replay_root_dfs_runs:
        raise ValueError("residual decoder walk did not reduce root DFS runs")
    if result.total_residual_resumed_snapshots <= 0:
        raise ValueError("residual decoder walk did not resume snapshots")
    if result.max_residual_retained_render_payload_chars != 0:
        raise ValueError("residual decoder walk retained rendered-suffix payload")


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
    return _observe_state_query(mode=execution_mode, state=state)


def _observe_state_query(
    *,
    mode: OnlineDecoderExecutionMode,
    state: SouthStarOnlineDecoderState,
) -> PreparedPrefixQueryObservation:
    return _query_state(mode=mode, state=state).observation


def _query_state(
    *,
    mode: OnlineDecoderExecutionMode,
    state: SouthStarOnlineDecoderState,
) -> _PreparedStateQuery:
    result = state.choices_with_stats()
    eos_choices = tuple(choice for choice in result.choices if choice.is_eos)
    has_eos = bool(eos_choices)
    eos_completion_count = sum(choice.completion_count for choice in eos_choices)
    if has_eos and eos_completion_count <= 0:
        raise ValueError("EOS choice must have positive completion_count")
    next_token_texts = tuple(
        choice.text for choice in result.choices if not choice.is_eos
    )
    next_token_completion_counts = _next_token_completion_counts(result.choices)
    retained = getattr(result.stats, "retained_state_size", None)
    return _PreparedStateQuery(
        observation=PreparedPrefixQueryObservation(
            prefix=state.prefix,
            execution_mode=mode,
            next_token_texts=next_token_texts,
            next_token_text_set=frozenset(next_token_texts),
            next_token_completion_counts=next_token_completion_counts,
            has_eos=has_eos,
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
        ),
        choices=tuple(result.choices),
    )


def _advance_state_by_token(
    *,
    observation: PreparedPrefixQueryObservation,
    choices: tuple[object, ...],
    token: str,
) -> SouthStarOnlineDecoderState:
    matches = tuple(
        choice for choice in choices if not choice.is_eos and choice.text == token
    )
    if len(matches) != 1:
        raise ValueError(
            f"decoder walk token {token!r} is not uniquely legal at {observation.prefix!r}"
        )
    next_state = matches[0].next_state
    if next_state is None:
        raise ValueError("non-EOS decoder walk choice lacks next_state")
    if next_state.prefix != observation.prefix + token:
        raise ValueError("decoder walk choice advanced to unexpected prefix")
    return next_state


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


def _decoder_walk_result(
    *,
    fixture_name: str,
    rooted_at_atom: int,
    steps: tuple[PreparedDecoderWalkStep, ...],
    probe: PreparedRuntimeProbeResult,
) -> PreparedDecoderWalkResult:
    return PreparedDecoderWalkResult(
        fixture_name=fixture_name,
        rooted_at_atom=rooted_at_atom,
        steps=steps,
        total_prefix_replay_root_dfs_runs=sum(
            _step_int(step.root_dfs_runs_by_mode, OnlineDecoderExecutionMode.PREFIX_REPLAY)
            for step in steps
        ),
        total_cached_root_dfs_runs=sum(
            _step_int(
                step.root_dfs_runs_by_mode,
                OnlineDecoderExecutionMode.CACHED_COMPLETIONS,
            )
            for step in steps
        ),
        total_residual_root_dfs_runs=sum(
            _step_int(
                step.root_dfs_runs_by_mode,
                OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
            )
            for step in steps
        ),
        total_residual_resumed_snapshots=sum(
            _step_int(
                step.resumed_snapshots_by_mode,
                OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
            )
            for step in steps
        ),
        max_residual_retained_render_payload_chars=max(
            (
                _step_int(
                    step.retained_render_payload_chars_by_mode,
                    OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
                )
                for step in steps
            ),
            default=0,
        ),
        probe=probe,
    )


def _walk_step_from_observations(
    observations: dict[OnlineDecoderExecutionMode, PreparedPrefixQueryObservation],
    selected_token: str | None,
) -> PreparedDecoderWalkStep:
    prefix = observations[OnlineDecoderExecutionMode.PREFIX_REPLAY].prefix
    return PreparedDecoderWalkStep(
        prefix=prefix,
        selected_token=selected_token,
        next_token_set_by_mode=tuple(
            (mode.value, observations[mode].next_token_text_set)
            for mode in _PREFIX_WORKLOAD_MODES
        ),
        eos_count_by_mode=tuple(
            (mode.value, observations[mode].eos_completion_count)
            for mode in _PREFIX_WORKLOAD_MODES
        ),
        root_dfs_runs_by_mode=tuple(
            (mode.value, observations[mode].root_dfs_runs)
            for mode in _PREFIX_WORKLOAD_MODES
        ),
        resumed_snapshots_by_mode=tuple(
            (mode.value, observations[mode].resumed_snapshots)
            for mode in _PREFIX_WORKLOAD_MODES
        ),
        retained_render_payload_chars_by_mode=tuple(
            (mode.value, observations[mode].retained_render_payload_chars)
            for mode in _PREFIX_WORKLOAD_MODES
        ),
    )


def _validate_observations(
    prefix: str,
    observations: dict[OnlineDecoderExecutionMode, PreparedPrefixQueryObservation],
) -> None:
    row = PreparedPrefixWorkloadRow(
        fixture_name="decoder-walk",
        rooted_at_atom=-1,
        prefix=prefix,
        prefix_replay=observations[OnlineDecoderExecutionMode.PREFIX_REPLAY],
        cached_completions=observations[OnlineDecoderExecutionMode.CACHED_COMPLETIONS],
        residual_continuations=observations[
            OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS
        ],
    )
    _validate_row(row)


def _validate_row(row: PreparedPrefixWorkloadRow) -> None:
    observations = (
        row.prefix_replay,
        row.cached_completions,
        row.residual_continuations,
    )
    prefixes = {item.prefix for item in observations}
    if prefixes != {row.prefix}:
        raise ValueError(f"prefix workload compared mismatched prefixes at {row.prefix!r}")
    next_tokens = {item.next_token_text_set for item in observations}
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
    for item in observations:
        if item.has_eos and item.eos_completion_count <= 0:
            raise ValueError("EOS choice must have positive completion_count")


def _validate_walk_step(step: PreparedDecoderWalkStep) -> None:
    next_token_sets = {item[1] for item in step.next_token_set_by_mode}
    if len(next_token_sets) != 1:
        raise ValueError(f"decoder walk next-token disagreement at {step.prefix!r}")
    eos_counts = {item[1] for item in step.eos_count_by_mode}
    if len(eos_counts) != 1:
        raise ValueError(f"decoder walk EOS count disagreement at {step.prefix!r}")


def _step_int(
    values: tuple[tuple[str, int | None], ...],
    mode: OnlineDecoderExecutionMode,
) -> int:
    by_mode = dict(values)
    value = by_mode.get(mode.value)
    return 0 if value is None else int(value)


def _next_token_completion_counts(
    choices: Iterable[object],
) -> tuple[tuple[str, int], ...]:
    counts: dict[str, int] = {}
    for choice in choices:
        if choice.is_eos:
            continue
        counts[choice.text] = counts.get(choice.text, 0) + int(choice.completion_count)
    return tuple((text, counts[text]) for text in sorted(counts))


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
    "PreparedDecoderWalkResult",
    "PreparedDecoderWalkStep",
    "PreparedPrefixQueryObservation",
    "PreparedPrefixWorkloadResult",
    "PreparedPrefixWorkloadRow",
    "advance_decoder_to_prefix",
    "choose_walk_token",
    "collect_prepared_decoder_walk",
    "collect_mode_union_token_boundary_prefixes",
    "collect_prepared_prefix_workload",
    "collect_token_boundary_prefixes",
    "validate_prepared_decoder_walk_result",
    "validate_prepared_prefix_workload_result",
)
