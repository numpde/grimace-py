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

PREFIX_SCHEDULER_EVIDENCE_FIXTURES = frozenset(
    {
        "tetrahedral",
        "directional",
        "ring",
        "prefix-scheduler",
    }
)

DIRECTION_SCHEDULER_EVIDENCE_FIXTURES = frozenset(
    {
        "directional",
    }
)

SUPPORT_MAXIMAL_SCHEDULER_EVIDENCE_FIXTURES = frozenset(
    {
        "support-maximal",
    }
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
    retained_scheduler_frame_count: int | None
    retained_prefix_enumeration_frame_count: int | None
    max_retained_prefix_domain_count: int | None
    total_retained_prefix_domain_count: int | None
    max_retained_prefix_assignment_count: int | None
    total_retained_prefix_assignment_count: int | None
    retained_direction_enumeration_frame_count: int | None
    max_retained_direction_carrier_count: int | None
    total_retained_direction_carrier_count: int | None
    max_retained_direction_assignment_count: int | None
    total_retained_direction_assignment_count: int | None
    retained_support_maximal_frame_count: int | None
    max_retained_support_maximal_candidate_count: int | None
    total_retained_support_maximal_candidate_count: int | None
    max_retained_support_maximal_selected_count: int | None
    total_retained_support_maximal_selected_count: int | None
    max_retained_support_maximal_remaining_count: int | None
    total_retained_support_maximal_remaining_count: int | None


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
    total_residual_prefix_enumeration_frame_count: int
    max_residual_prefix_domain_count: int
    total_residual_prefix_domain_count: int
    max_residual_prefix_assignment_count: int
    total_residual_prefix_assignment_count: int
    total_residual_direction_enumeration_frame_count: int
    max_residual_direction_carrier_count: int
    total_residual_direction_carrier_count: int
    max_residual_direction_assignment_count: int
    total_residual_direction_assignment_count: int
    total_residual_support_maximal_frame_count: int
    max_residual_support_maximal_candidate_count: int
    total_residual_support_maximal_candidate_count: int
    max_residual_support_maximal_selected_count: int
    total_residual_support_maximal_selected_count: int
    max_residual_support_maximal_remaining_count: int
    total_residual_support_maximal_remaining_count: int
    probe: PreparedRuntimeProbeResult


@dataclass(frozen=True, slots=True)
class PreparedDecoderWalkStep:
    prefix: str
    selected_token: str | None
    next_token_set_by_mode: tuple[tuple[str, frozenset[str]], ...]
    next_token_completion_counts_by_mode: tuple[
        tuple[str, tuple[tuple[str, int], ...]], ...
    ]
    eos_count_by_mode: tuple[tuple[str, int], ...]
    root_dfs_runs_by_mode: tuple[tuple[str, int | None], ...]
    resumed_snapshots_by_mode: tuple[tuple[str, int | None], ...]
    retained_continuation_count_by_mode: tuple[tuple[str, int | None], ...]
    retained_render_payload_chars_by_mode: tuple[tuple[str, int | None], ...]
    retained_prefix_enumeration_frame_count_by_mode: tuple[
        tuple[str, int | None], ...
    ]
    retained_max_prefix_domain_count_by_mode: tuple[tuple[str, int | None], ...]
    retained_total_prefix_domain_count_by_mode: tuple[tuple[str, int | None], ...]
    retained_max_prefix_assignment_count_by_mode: tuple[tuple[str, int | None], ...]
    retained_total_prefix_assignment_count_by_mode: tuple[tuple[str, int | None], ...]
    retained_direction_enumeration_frame_count_by_mode: tuple[
        tuple[str, int | None], ...
    ]
    retained_max_direction_carrier_count_by_mode: tuple[tuple[str, int | None], ...]
    retained_total_direction_carrier_count_by_mode: tuple[tuple[str, int | None], ...]
    retained_max_direction_assignment_count_by_mode: tuple[
        tuple[str, int | None], ...
    ]
    retained_total_direction_assignment_count_by_mode: tuple[
        tuple[str, int | None], ...
    ]
    retained_support_maximal_frame_count_by_mode: tuple[
        tuple[str, int | None], ...
    ]
    retained_max_support_maximal_candidate_count_by_mode: tuple[
        tuple[str, int | None], ...
    ]
    retained_total_support_maximal_candidate_count_by_mode: tuple[
        tuple[str, int | None], ...
    ]
    retained_max_support_maximal_selected_count_by_mode: tuple[
        tuple[str, int | None], ...
    ]
    retained_total_support_maximal_selected_count_by_mode: tuple[
        tuple[str, int | None], ...
    ]
    retained_max_support_maximal_remaining_count_by_mode: tuple[
        tuple[str, int | None], ...
    ]
    retained_total_support_maximal_remaining_count_by_mode: tuple[
        tuple[str, int | None], ...
    ]


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
    total_residual_prefix_enumeration_frame_count: int
    max_residual_prefix_domain_count: int
    total_residual_prefix_domain_count: int
    max_residual_prefix_assignment_count: int
    total_residual_prefix_assignment_count: int
    total_residual_direction_enumeration_frame_count: int
    max_residual_direction_carrier_count: int
    total_residual_direction_carrier_count: int
    max_residual_direction_assignment_count: int
    total_residual_direction_assignment_count: int
    total_residual_support_maximal_frame_count: int
    max_residual_support_maximal_candidate_count: int
    total_residual_support_maximal_candidate_count: int
    max_residual_support_maximal_selected_count: int
    total_residual_support_maximal_selected_count: int
    max_residual_support_maximal_remaining_count: int
    total_residual_support_maximal_remaining_count: int
    probe: PreparedRuntimeProbeResult


@dataclass(frozen=True, slots=True)
class PreparedDecoderBranchWalkResult:
    fixture_name: str
    rooted_at_atom: int
    walks: tuple[PreparedDecoderWalkResult, ...]
    total_prefix_replay_root_dfs_runs: int
    total_residual_root_dfs_runs: int
    total_residual_resumed_snapshots: int
    max_residual_retained_render_payload_chars: int
    total_residual_prefix_enumeration_frame_count: int
    max_residual_prefix_domain_count: int
    total_residual_prefix_domain_count: int
    max_residual_prefix_assignment_count: int
    total_residual_prefix_assignment_count: int
    total_residual_direction_enumeration_frame_count: int
    max_residual_direction_carrier_count: int
    total_residual_direction_carrier_count: int
    max_residual_direction_assignment_count: int
    total_residual_direction_assignment_count: int
    total_residual_support_maximal_frame_count: int
    max_residual_support_maximal_candidate_count: int
    total_residual_support_maximal_candidate_count: int
    max_residual_support_maximal_selected_count: int
    total_residual_support_maximal_selected_count: int
    max_residual_support_maximal_remaining_count: int
    total_residual_support_maximal_remaining_count: int
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
        steps = _collect_decoder_walk_steps(
            prepared=prepared,
            runtime_options=runtime_options,
            max_steps=max_steps,
            token_script=(),
        )
    return _decoder_walk_result(
        fixture_name=fixture_name,
        rooted_at_atom=runtime_options.rooted_at_atom,
        steps=steps,
        probe=probe.result(),
    )


def collect_prepared_branch_decoder_walks(
    *,
    fixture_name: str,
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions = SouthStarRuntimeOptions(),
    max_walks: int = 8,
    max_steps_per_walk: int = 32,
) -> PreparedDecoderBranchWalkResult:
    if max_walks <= 0:
        raise ValueError("branch decoder workload requires max_walks > 0")
    queue: list[tuple[str, ...]] = [()]
    seen_scripts: set[tuple[str, ...]] = {()}
    collected_steps: list[tuple[PreparedDecoderWalkStep, ...]] = []
    with PreparedRuntimeProbe() as probe:
        while queue and len(collected_steps) < max_walks:
            script = queue.pop(0)
            collected_steps.append(
                _collect_decoder_walk_steps(
                    prepared=prepared,
                    runtime_options=runtime_options,
                    max_steps=max_steps_per_walk,
                    token_script=script,
                    branch_queue=queue,
                    seen_scripts=seen_scripts,
                    max_scripts=max_walks,
                )
            )
    probe_result = probe.result()
    walks = tuple(
        _decoder_walk_result(
            fixture_name=fixture_name,
            rooted_at_atom=runtime_options.rooted_at_atom,
            steps=steps,
            probe=probe_result,
        )
        for steps in collected_steps
    )
    return _branch_walk_result(
        fixture_name=fixture_name,
        rooted_at_atom=runtime_options.rooted_at_atom,
        walks=walks,
        probe=probe_result,
    )


def choose_walk_token(observation: PreparedPrefixQueryObservation) -> str | None:
    for token in observation.next_token_texts:
        return token
    return None


def _collect_decoder_walk_steps(
    *,
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions,
    max_steps: int,
    token_script: tuple[str, ...],
    branch_queue: list[tuple[str, ...]] | None = None,
    seen_scripts: set[tuple[str, ...]] | None = None,
    max_scripts: int | None = None,
) -> tuple[PreparedDecoderWalkStep, ...]:
    states = {
        mode: make_determinized_online_decoder(
            prepared=prepared,
            include_eos=True,
            runtime_options=runtime_options,
            execution_mode=mode,
        ).initial_state()
        for mode in _PREFIX_WORKLOAD_MODES
    }
    selected_path: tuple[str, ...] = ()
    steps: list[PreparedDecoderWalkStep] = []
    for step_index in range(max_steps):
        queries = {
            mode: _query_state(mode=mode, state=states[mode])
            for mode in _PREFIX_WORKLOAD_MODES
        }
        observations = {mode: query.observation for mode, query in queries.items()}
        _validate_observations(
            states[OnlineDecoderExecutionMode.PREFIX_REPLAY].prefix,
            observations,
        )
        prefix_replay = observations[OnlineDecoderExecutionMode.PREFIX_REPLAY]
        selected_token = _scripted_walk_token(
            prefix_replay,
            token_script=token_script,
            step_index=step_index,
        )
        if branch_queue is not None and seen_scripts is not None:
            _enqueue_branch_scripts(
                selected_path=selected_path,
                selected_token=selected_token,
                legal_tokens=prefix_replay.next_token_texts,
                queue=branch_queue,
                seen_scripts=seen_scripts,
                max_scripts=max_scripts,
            )
        steps.append(_walk_step_from_observations(observations, selected_token))
        if selected_token is None:
            return tuple(steps)
        states = {
            mode: _advance_state_by_token(
                observation=observations[mode],
                choices=queries[mode].choices,
                token=selected_token,
            )
            for mode in _PREFIX_WORKLOAD_MODES
        }
        selected_path = (*selected_path, selected_token)
    raise ValueError("prepared decoder walk exceeded max_steps")


def _scripted_walk_token(
    observation: PreparedPrefixQueryObservation,
    *,
    token_script: tuple[str, ...],
    step_index: int,
) -> str | None:
    if step_index < len(token_script):
        token = token_script[step_index]
        if token not in observation.next_token_text_set:
            raise ValueError(
                f"branch decoder walk script token {token!r} is not legal at "
                f"{observation.prefix!r}"
            )
        return token
    return choose_walk_token(observation)


def _enqueue_branch_scripts(
    *,
    selected_path: tuple[str, ...],
    selected_token: str | None,
    legal_tokens: tuple[str, ...],
    queue: list[tuple[str, ...]],
    seen_scripts: set[tuple[str, ...]],
    max_scripts: int | None,
) -> None:
    if not legal_tokens:
        return
    for token in (legal_tokens[0], legal_tokens[-1]):
        if token == selected_token:
            continue
        if max_scripts is not None and len(seen_scripts) >= max_scripts:
            return
        script = (*selected_path, token)
        if script in seen_scripts:
            continue
        seen_scripts.add(script)
        queue.append(script)


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


def validate_prepared_branch_decoder_walk_result(
    result: PreparedDecoderBranchWalkResult,
) -> None:
    if not result.walks:
        raise ValueError("prepared branch decoder workload produced no walks")
    for walk in result.walks:
        validate_prepared_decoder_walk_result(walk)
    if result.total_residual_root_dfs_runs >= result.total_prefix_replay_root_dfs_runs:
        raise ValueError("residual branch decoder walk did not reduce root DFS runs")
    if result.total_residual_resumed_snapshots <= 0:
        raise ValueError("residual branch decoder walk did not resume snapshots")
    if result.max_residual_retained_render_payload_chars != 0:
        raise ValueError("residual branch decoder walk retained rendered-suffix payload")


def require_prefix_scheduler_frame_evidence(
    result: (
        PreparedPrefixWorkloadResult
        | PreparedDecoderWalkResult
        | PreparedDecoderBranchWalkResult
    ),
    *,
    fixture_name: str,
) -> None:
    if fixture_name not in PREFIX_SCHEDULER_EVIDENCE_FIXTURES:
        return
    _require_positive_int(
        result.total_residual_prefix_enumeration_frame_count,
        field_name="prefix_enumeration_frame_count",
        fixture_name=fixture_name,
    )
    _require_positive_int(
        result.max_residual_prefix_domain_count,
        field_name="max_prefix_domain_count",
        fixture_name=fixture_name,
    )
    _require_positive_int(
        result.total_residual_prefix_domain_count,
        field_name="total_prefix_domain_count",
        fixture_name=fixture_name,
    )
    _require_positive_int(
        result.max_residual_prefix_assignment_count,
        field_name="max_prefix_assignment_count",
        fixture_name=fixture_name,
    )
    _require_positive_int(
        result.total_residual_prefix_assignment_count,
        field_name="total_prefix_assignment_count",
        fixture_name=fixture_name,
    )


def require_direction_scheduler_frame_evidence(
    result: (
        PreparedPrefixWorkloadResult
        | PreparedDecoderWalkResult
        | PreparedDecoderBranchWalkResult
    ),
    *,
    fixture_name: str,
) -> None:
    if fixture_name not in DIRECTION_SCHEDULER_EVIDENCE_FIXTURES:
        return
    _require_positive_int(
        result.total_residual_direction_enumeration_frame_count,
        field_name="direction_enumeration_frame_count",
        fixture_name=fixture_name,
    )
    _require_positive_int(
        result.max_residual_direction_carrier_count,
        field_name="max_direction_carrier_count",
        fixture_name=fixture_name,
    )
    _require_positive_int(
        result.total_residual_direction_carrier_count,
        field_name="total_direction_carrier_count",
        fixture_name=fixture_name,
    )
    _require_positive_int(
        result.max_residual_direction_assignment_count,
        field_name="max_direction_assignment_count",
        fixture_name=fixture_name,
    )
    _require_positive_int(
        result.total_residual_direction_assignment_count,
        field_name="total_direction_assignment_count",
        fixture_name=fixture_name,
    )


def require_support_maximal_scheduler_frame_evidence(
    result: (
        PreparedPrefixWorkloadResult
        | PreparedDecoderWalkResult
        | PreparedDecoderBranchWalkResult
    ),
    *,
    fixture_name: str,
) -> None:
    if fixture_name not in SUPPORT_MAXIMAL_SCHEDULER_EVIDENCE_FIXTURES:
        return
    _require_positive_int(
        result.total_residual_support_maximal_frame_count,
        field_name="support_maximal_frame_count",
        fixture_name=fixture_name,
    )
    _require_positive_int(
        result.max_residual_support_maximal_candidate_count,
        field_name="max_support_maximal_candidate_count",
        fixture_name=fixture_name,
    )
    _require_positive_int(
        result.total_residual_support_maximal_candidate_count,
        field_name="total_support_maximal_candidate_count",
        fixture_name=fixture_name,
    )
    _require_positive_int(
        result.max_residual_support_maximal_selected_count,
        field_name="max_support_maximal_selected_count",
        fixture_name=fixture_name,
    )
    _require_positive_int(
        result.total_residual_support_maximal_selected_count,
        field_name="total_support_maximal_selected_count",
        fixture_name=fixture_name,
    )
    _require_positive_int(
        result.max_residual_support_maximal_remaining_count,
        field_name="max_support_maximal_remaining_count",
        fixture_name=fixture_name,
    )
    _require_positive_int(
        result.total_residual_support_maximal_remaining_count,
        field_name="total_support_maximal_remaining_count",
        fixture_name=fixture_name,
    )


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
            retained_scheduler_frame_count=(
                None if retained is None else int(retained.scheduler_frame_count)
            ),
            retained_prefix_enumeration_frame_count=(
                None if retained is None else int(retained.prefix_enumeration_frame_count)
            ),
            max_retained_prefix_domain_count=(
                None if retained is None else int(retained.max_prefix_domain_count)
            ),
            total_retained_prefix_domain_count=(
                None if retained is None else int(retained.total_prefix_domain_count)
            ),
            max_retained_prefix_assignment_count=(
                None if retained is None else int(retained.max_prefix_assignment_count)
            ),
            total_retained_prefix_assignment_count=(
                None if retained is None else int(retained.total_prefix_assignment_count)
            ),
            retained_direction_enumeration_frame_count=(
                None if retained is None else int(retained.direction_enumeration_frame_count)
            ),
            max_retained_direction_carrier_count=(
                None if retained is None else int(retained.max_direction_carrier_count)
            ),
            total_retained_direction_carrier_count=(
                None if retained is None else int(retained.total_direction_carrier_count)
            ),
            max_retained_direction_assignment_count=(
                None if retained is None else int(retained.max_direction_assignment_count)
            ),
            total_retained_direction_assignment_count=(
                None if retained is None else int(retained.total_direction_assignment_count)
            ),
            retained_support_maximal_frame_count=(
                None if retained is None else int(retained.support_maximal_frame_count)
            ),
            max_retained_support_maximal_candidate_count=(
                None
                if retained is None
                else int(retained.max_support_maximal_candidate_count)
            ),
            total_retained_support_maximal_candidate_count=(
                None
                if retained is None
                else int(retained.total_support_maximal_candidate_count)
            ),
            max_retained_support_maximal_selected_count=(
                None
                if retained is None
                else int(retained.max_support_maximal_selected_count)
            ),
            total_retained_support_maximal_selected_count=(
                None
                if retained is None
                else int(retained.total_support_maximal_selected_count)
            ),
            max_retained_support_maximal_remaining_count=(
                None
                if retained is None
                else int(retained.max_support_maximal_remaining_count)
            ),
            total_retained_support_maximal_remaining_count=(
                None
                if retained is None
                else int(retained.total_support_maximal_remaining_count)
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
    return PreparedPrefixWorkloadResult(
        rows=rows,
        total_prefix_replay_root_dfs_runs=sum(
            _required_observation_int(
                row.prefix_replay,
                field_name="root_dfs_runs",
            )
            for row in rows
        ),
        total_residual_root_dfs_runs=sum(
            _required_residual_observation_int(
                row.residual_continuations,
                field_name="root_dfs_runs",
            )
            for row in rows
        ),
        total_residual_resumed_snapshots=sum(
            _required_residual_observation_int(
                row.residual_continuations,
                field_name="resumed_snapshots",
            )
            for row in rows
        ),
        max_residual_retained_render_payload_chars=(
            max(
                _required_residual_observation_int(
                    row.residual_continuations,
                    field_name="retained_render_payload_chars",
                )
                for row in rows
            )
            if rows
            else 0
        ),
        total_residual_prefix_enumeration_frame_count=sum(
            _required_residual_observation_int(
                row.residual_continuations,
                field_name="retained_prefix_enumeration_frame_count",
            )
            for row in rows
        ),
        max_residual_prefix_domain_count=max(
            (
                _required_residual_observation_int(
                    row.residual_continuations,
                    field_name="max_retained_prefix_domain_count",
                )
                for row in rows
            ),
            default=0,
        ),
        total_residual_prefix_domain_count=sum(
            _required_residual_observation_int(
                row.residual_continuations,
                field_name="total_retained_prefix_domain_count",
            )
            for row in rows
        ),
        max_residual_prefix_assignment_count=max(
            (
                _required_residual_observation_int(
                    row.residual_continuations,
                    field_name="max_retained_prefix_assignment_count",
                )
                for row in rows
            ),
            default=0,
        ),
        total_residual_prefix_assignment_count=sum(
            _required_residual_observation_int(
                row.residual_continuations,
                field_name="total_retained_prefix_assignment_count",
            )
            for row in rows
        ),
        total_residual_direction_enumeration_frame_count=sum(
            _required_residual_observation_int(
                row.residual_continuations,
                field_name="retained_direction_enumeration_frame_count",
            )
            for row in rows
        ),
        max_residual_direction_carrier_count=max(
            (
                _required_residual_observation_int(
                    row.residual_continuations,
                    field_name="max_retained_direction_carrier_count",
                )
                for row in rows
            ),
            default=0,
        ),
        total_residual_direction_carrier_count=sum(
            _required_residual_observation_int(
                row.residual_continuations,
                field_name="total_retained_direction_carrier_count",
            )
            for row in rows
        ),
        max_residual_direction_assignment_count=max(
            (
                _required_residual_observation_int(
                    row.residual_continuations,
                    field_name="max_retained_direction_assignment_count",
                )
                for row in rows
            ),
            default=0,
        ),
        total_residual_direction_assignment_count=sum(
            _required_residual_observation_int(
                row.residual_continuations,
                field_name="total_retained_direction_assignment_count",
            )
            for row in rows
        ),
        total_residual_support_maximal_frame_count=sum(
            _required_residual_observation_int(
                row.residual_continuations,
                field_name="retained_support_maximal_frame_count",
            )
            for row in rows
        ),
        max_residual_support_maximal_candidate_count=max(
            (
                _required_residual_observation_int(
                    row.residual_continuations,
                    field_name="max_retained_support_maximal_candidate_count",
                )
                for row in rows
            ),
            default=0,
        ),
        total_residual_support_maximal_candidate_count=sum(
            _required_residual_observation_int(
                row.residual_continuations,
                field_name="total_retained_support_maximal_candidate_count",
            )
            for row in rows
        ),
        max_residual_support_maximal_selected_count=max(
            (
                _required_residual_observation_int(
                    row.residual_continuations,
                    field_name="max_retained_support_maximal_selected_count",
                )
                for row in rows
            ),
            default=0,
        ),
        total_residual_support_maximal_selected_count=sum(
            _required_residual_observation_int(
                row.residual_continuations,
                field_name="total_retained_support_maximal_selected_count",
            )
            for row in rows
        ),
        max_residual_support_maximal_remaining_count=max(
            (
                _required_residual_observation_int(
                    row.residual_continuations,
                    field_name="max_retained_support_maximal_remaining_count",
                )
                for row in rows
            ),
            default=0,
        ),
        total_residual_support_maximal_remaining_count=sum(
            _required_residual_observation_int(
                row.residual_continuations,
                field_name="total_retained_support_maximal_remaining_count",
            )
            for row in rows
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
            _required_mode_int(
                step.root_dfs_runs_by_mode,
                mode_name=OnlineDecoderExecutionMode.PREFIX_REPLAY.value,
                field_name="root_dfs_runs",
            )
            for step in steps
        ),
        total_cached_root_dfs_runs=sum(
            _optional_mode_int(
                step.root_dfs_runs_by_mode,
                OnlineDecoderExecutionMode.CACHED_COMPLETIONS,
            )
            for step in steps
        ),
        total_residual_root_dfs_runs=sum(
            _required_mode_int(
                step.root_dfs_runs_by_mode,
                mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
                field_name="root_dfs_runs",
            )
            for step in steps
        ),
        total_residual_resumed_snapshots=sum(
            _required_mode_int(
                step.resumed_snapshots_by_mode,
                mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
                field_name="resumed_snapshots",
            )
            for step in steps
        ),
        max_residual_retained_render_payload_chars=max(
            (
                _required_mode_int(
                    step.retained_render_payload_chars_by_mode,
                    mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
                    field_name="retained_render_payload_chars",
                )
                for step in steps
            ),
            default=0,
        ),
        total_residual_prefix_enumeration_frame_count=sum(
            _required_mode_int(
                step.retained_prefix_enumeration_frame_count_by_mode,
                mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
                field_name="prefix_enumeration_frame_count",
            )
            for step in steps
        ),
        max_residual_prefix_domain_count=max(
            (
                _required_mode_int(
                    step.retained_max_prefix_domain_count_by_mode,
                    mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
                    field_name="max_prefix_domain_count",
                )
                for step in steps
            ),
            default=0,
        ),
        total_residual_prefix_domain_count=sum(
            _required_mode_int(
                step.retained_total_prefix_domain_count_by_mode,
                mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
                field_name="total_prefix_domain_count",
            )
            for step in steps
        ),
        max_residual_prefix_assignment_count=max(
            (
                _required_mode_int(
                    step.retained_max_prefix_assignment_count_by_mode,
                    mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
                    field_name="max_prefix_assignment_count",
                )
                for step in steps
            ),
            default=0,
        ),
        total_residual_prefix_assignment_count=sum(
            _required_mode_int(
                step.retained_total_prefix_assignment_count_by_mode,
                mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
                field_name="total_prefix_assignment_count",
            )
            for step in steps
        ),
        total_residual_direction_enumeration_frame_count=sum(
            _required_mode_int(
                step.retained_direction_enumeration_frame_count_by_mode,
                mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
                field_name="direction_enumeration_frame_count",
            )
            for step in steps
        ),
        max_residual_direction_carrier_count=max(
            (
                _required_mode_int(
                    step.retained_max_direction_carrier_count_by_mode,
                    mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
                    field_name="max_direction_carrier_count",
                )
                for step in steps
            ),
            default=0,
        ),
        total_residual_direction_carrier_count=sum(
            _required_mode_int(
                step.retained_total_direction_carrier_count_by_mode,
                mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
                field_name="total_direction_carrier_count",
            )
            for step in steps
        ),
        max_residual_direction_assignment_count=max(
            (
                _required_mode_int(
                    step.retained_max_direction_assignment_count_by_mode,
                    mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
                    field_name="max_direction_assignment_count",
                )
                for step in steps
            ),
            default=0,
        ),
        total_residual_direction_assignment_count=sum(
            _required_mode_int(
                step.retained_total_direction_assignment_count_by_mode,
                mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
                field_name="total_direction_assignment_count",
            )
            for step in steps
        ),
        total_residual_support_maximal_frame_count=sum(
            _required_mode_int(
                step.retained_support_maximal_frame_count_by_mode,
                mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
                field_name="support_maximal_frame_count",
            )
            for step in steps
        ),
        max_residual_support_maximal_candidate_count=max(
            (
                _required_mode_int(
                    step.retained_max_support_maximal_candidate_count_by_mode,
                    mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
                    field_name="max_support_maximal_candidate_count",
                )
                for step in steps
            ),
            default=0,
        ),
        total_residual_support_maximal_candidate_count=sum(
            _required_mode_int(
                step.retained_total_support_maximal_candidate_count_by_mode,
                mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
                field_name="total_support_maximal_candidate_count",
            )
            for step in steps
        ),
        max_residual_support_maximal_selected_count=max(
            (
                _required_mode_int(
                    step.retained_max_support_maximal_selected_count_by_mode,
                    mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
                    field_name="max_support_maximal_selected_count",
                )
                for step in steps
            ),
            default=0,
        ),
        total_residual_support_maximal_selected_count=sum(
            _required_mode_int(
                step.retained_total_support_maximal_selected_count_by_mode,
                mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
                field_name="total_support_maximal_selected_count",
            )
            for step in steps
        ),
        max_residual_support_maximal_remaining_count=max(
            (
                _required_mode_int(
                    step.retained_max_support_maximal_remaining_count_by_mode,
                    mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
                    field_name="max_support_maximal_remaining_count",
                )
                for step in steps
            ),
            default=0,
        ),
        total_residual_support_maximal_remaining_count=sum(
            _required_mode_int(
                step.retained_total_support_maximal_remaining_count_by_mode,
                mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
                field_name="total_support_maximal_remaining_count",
            )
            for step in steps
        ),
        probe=probe,
    )


def _branch_walk_result(
    *,
    fixture_name: str,
    rooted_at_atom: int,
    walks: tuple[PreparedDecoderWalkResult, ...],
    probe: PreparedRuntimeProbeResult,
) -> PreparedDecoderBranchWalkResult:
    return PreparedDecoderBranchWalkResult(
        fixture_name=fixture_name,
        rooted_at_atom=rooted_at_atom,
        walks=walks,
        total_prefix_replay_root_dfs_runs=sum(
            walk.total_prefix_replay_root_dfs_runs for walk in walks
        ),
        total_residual_root_dfs_runs=sum(
            walk.total_residual_root_dfs_runs for walk in walks
        ),
        total_residual_resumed_snapshots=sum(
            walk.total_residual_resumed_snapshots for walk in walks
        ),
        max_residual_retained_render_payload_chars=max(
            (walk.max_residual_retained_render_payload_chars for walk in walks),
            default=0,
        ),
        total_residual_prefix_enumeration_frame_count=sum(
            walk.total_residual_prefix_enumeration_frame_count for walk in walks
        ),
        max_residual_prefix_domain_count=max(
            (walk.max_residual_prefix_domain_count for walk in walks),
            default=0,
        ),
        total_residual_prefix_domain_count=sum(
            walk.total_residual_prefix_domain_count for walk in walks
        ),
        max_residual_prefix_assignment_count=max(
            (walk.max_residual_prefix_assignment_count for walk in walks),
            default=0,
        ),
        total_residual_prefix_assignment_count=sum(
            walk.total_residual_prefix_assignment_count for walk in walks
        ),
        total_residual_direction_enumeration_frame_count=sum(
            walk.total_residual_direction_enumeration_frame_count for walk in walks
        ),
        max_residual_direction_carrier_count=max(
            (walk.max_residual_direction_carrier_count for walk in walks),
            default=0,
        ),
        total_residual_direction_carrier_count=sum(
            walk.total_residual_direction_carrier_count for walk in walks
        ),
        max_residual_direction_assignment_count=max(
            (walk.max_residual_direction_assignment_count for walk in walks),
            default=0,
        ),
        total_residual_direction_assignment_count=sum(
            walk.total_residual_direction_assignment_count for walk in walks
        ),
        total_residual_support_maximal_frame_count=sum(
            walk.total_residual_support_maximal_frame_count for walk in walks
        ),
        max_residual_support_maximal_candidate_count=max(
            (walk.max_residual_support_maximal_candidate_count for walk in walks),
            default=0,
        ),
        total_residual_support_maximal_candidate_count=sum(
            walk.total_residual_support_maximal_candidate_count for walk in walks
        ),
        max_residual_support_maximal_selected_count=max(
            (walk.max_residual_support_maximal_selected_count for walk in walks),
            default=0,
        ),
        total_residual_support_maximal_selected_count=sum(
            walk.total_residual_support_maximal_selected_count for walk in walks
        ),
        max_residual_support_maximal_remaining_count=max(
            (walk.max_residual_support_maximal_remaining_count for walk in walks),
            default=0,
        ),
        total_residual_support_maximal_remaining_count=sum(
            walk.total_residual_support_maximal_remaining_count for walk in walks
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
        next_token_completion_counts_by_mode=tuple(
            (mode.value, observations[mode].next_token_completion_counts)
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
        retained_continuation_count_by_mode=tuple(
            (mode.value, observations[mode].retained_continuation_count)
            for mode in _PREFIX_WORKLOAD_MODES
        ),
        retained_render_payload_chars_by_mode=tuple(
            (mode.value, observations[mode].retained_render_payload_chars)
            for mode in _PREFIX_WORKLOAD_MODES
        ),
        retained_prefix_enumeration_frame_count_by_mode=tuple(
            (mode.value, observations[mode].retained_prefix_enumeration_frame_count)
            for mode in _PREFIX_WORKLOAD_MODES
        ),
        retained_max_prefix_domain_count_by_mode=tuple(
            (mode.value, observations[mode].max_retained_prefix_domain_count)
            for mode in _PREFIX_WORKLOAD_MODES
        ),
        retained_total_prefix_domain_count_by_mode=tuple(
            (mode.value, observations[mode].total_retained_prefix_domain_count)
            for mode in _PREFIX_WORKLOAD_MODES
        ),
        retained_max_prefix_assignment_count_by_mode=tuple(
            (mode.value, observations[mode].max_retained_prefix_assignment_count)
            for mode in _PREFIX_WORKLOAD_MODES
        ),
        retained_total_prefix_assignment_count_by_mode=tuple(
            (mode.value, observations[mode].total_retained_prefix_assignment_count)
            for mode in _PREFIX_WORKLOAD_MODES
        ),
        retained_direction_enumeration_frame_count_by_mode=tuple(
            (mode.value, observations[mode].retained_direction_enumeration_frame_count)
            for mode in _PREFIX_WORKLOAD_MODES
        ),
        retained_max_direction_carrier_count_by_mode=tuple(
            (mode.value, observations[mode].max_retained_direction_carrier_count)
            for mode in _PREFIX_WORKLOAD_MODES
        ),
        retained_total_direction_carrier_count_by_mode=tuple(
            (mode.value, observations[mode].total_retained_direction_carrier_count)
            for mode in _PREFIX_WORKLOAD_MODES
        ),
        retained_max_direction_assignment_count_by_mode=tuple(
            (mode.value, observations[mode].max_retained_direction_assignment_count)
            for mode in _PREFIX_WORKLOAD_MODES
        ),
        retained_total_direction_assignment_count_by_mode=tuple(
            (mode.value, observations[mode].total_retained_direction_assignment_count)
            for mode in _PREFIX_WORKLOAD_MODES
        ),
        retained_support_maximal_frame_count_by_mode=tuple(
            (mode.value, observations[mode].retained_support_maximal_frame_count)
            for mode in _PREFIX_WORKLOAD_MODES
        ),
        retained_max_support_maximal_candidate_count_by_mode=tuple(
            (
                mode.value,
                observations[mode].max_retained_support_maximal_candidate_count,
            )
            for mode in _PREFIX_WORKLOAD_MODES
        ),
        retained_total_support_maximal_candidate_count_by_mode=tuple(
            (
                mode.value,
                observations[mode].total_retained_support_maximal_candidate_count,
            )
            for mode in _PREFIX_WORKLOAD_MODES
        ),
        retained_max_support_maximal_selected_count_by_mode=tuple(
            (
                mode.value,
                observations[mode].max_retained_support_maximal_selected_count,
            )
            for mode in _PREFIX_WORKLOAD_MODES
        ),
        retained_total_support_maximal_selected_count_by_mode=tuple(
            (
                mode.value,
                observations[mode].total_retained_support_maximal_selected_count,
            )
            for mode in _PREFIX_WORKLOAD_MODES
        ),
        retained_max_support_maximal_remaining_count_by_mode=tuple(
            (
                mode.value,
                observations[mode].max_retained_support_maximal_remaining_count,
            )
            for mode in _PREFIX_WORKLOAD_MODES
        ),
        retained_total_support_maximal_remaining_count_by_mode=tuple(
            (
                mode.value,
                observations[mode].total_retained_support_maximal_remaining_count,
            )
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
    _required_observation_int(row.prefix_replay, field_name="root_dfs_runs")
    _require_residual_observation_stats(row.residual_continuations)
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
    _required_mode_int(
        step.root_dfs_runs_by_mode,
        mode_name=OnlineDecoderExecutionMode.PREFIX_REPLAY.value,
        field_name="root_dfs_runs",
    )
    _required_mode_int(
        step.root_dfs_runs_by_mode,
        mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
        field_name="root_dfs_runs",
    )
    _required_mode_int(
        step.resumed_snapshots_by_mode,
        mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
        field_name="resumed_snapshots",
    )
    _required_mode_int(
        step.retained_continuation_count_by_mode,
        mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
        field_name="retained_continuation_count",
    )
    _required_mode_int(
        step.retained_render_payload_chars_by_mode,
        mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
        field_name="retained_render_payload_chars",
    )
    _required_mode_int(
        step.retained_prefix_enumeration_frame_count_by_mode,
        mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
        field_name="prefix_enumeration_frame_count",
    )
    _required_mode_int(
        step.retained_max_prefix_domain_count_by_mode,
        mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
        field_name="max_prefix_domain_count",
    )
    _required_mode_int(
        step.retained_total_prefix_domain_count_by_mode,
        mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
        field_name="total_prefix_domain_count",
    )
    _required_mode_int(
        step.retained_max_prefix_assignment_count_by_mode,
        mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
        field_name="max_prefix_assignment_count",
    )
    _required_mode_int(
        step.retained_total_prefix_assignment_count_by_mode,
        mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
        field_name="total_prefix_assignment_count",
    )
    _required_mode_int(
        step.retained_direction_enumeration_frame_count_by_mode,
        mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
        field_name="direction_enumeration_frame_count",
    )
    _required_mode_int(
        step.retained_max_direction_carrier_count_by_mode,
        mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
        field_name="max_direction_carrier_count",
    )
    _required_mode_int(
        step.retained_total_direction_carrier_count_by_mode,
        mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
        field_name="total_direction_carrier_count",
    )
    _required_mode_int(
        step.retained_max_direction_assignment_count_by_mode,
        mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
        field_name="max_direction_assignment_count",
    )
    _required_mode_int(
        step.retained_total_direction_assignment_count_by_mode,
        mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
        field_name="total_direction_assignment_count",
    )
    _required_mode_int(
        step.retained_support_maximal_frame_count_by_mode,
        mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
        field_name="support_maximal_frame_count",
    )
    _required_mode_int(
        step.retained_max_support_maximal_candidate_count_by_mode,
        mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
        field_name="max_support_maximal_candidate_count",
    )
    _required_mode_int(
        step.retained_total_support_maximal_candidate_count_by_mode,
        mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
        field_name="total_support_maximal_candidate_count",
    )
    _required_mode_int(
        step.retained_max_support_maximal_selected_count_by_mode,
        mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
        field_name="max_support_maximal_selected_count",
    )
    _required_mode_int(
        step.retained_total_support_maximal_selected_count_by_mode,
        mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
        field_name="total_support_maximal_selected_count",
    )
    _required_mode_int(
        step.retained_max_support_maximal_remaining_count_by_mode,
        mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
        field_name="max_support_maximal_remaining_count",
    )
    _required_mode_int(
        step.retained_total_support_maximal_remaining_count_by_mode,
        mode_name=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
        field_name="total_support_maximal_remaining_count",
    )
    next_token_sets = {item[1] for item in step.next_token_set_by_mode}
    if len(next_token_sets) != 1:
        raise ValueError(f"decoder walk next-token disagreement at {step.prefix!r}")
    next_completion_counts = {
        item[1] for item in step.next_token_completion_counts_by_mode
    }
    if len(next_completion_counts) != 1:
        raise ValueError(
            f"decoder walk next-token completion count disagreement at {step.prefix!r}"
        )
    eos_counts = {item[1] for item in step.eos_count_by_mode}
    if len(eos_counts) != 1:
        raise ValueError(f"decoder walk EOS count disagreement at {step.prefix!r}")
    if step.selected_token is not None:
        for _, token_set in step.next_token_set_by_mode:
            if step.selected_token not in token_set:
                raise ValueError(
                    f"decoder walk selected illegal token at {step.prefix!r}: "
                    f"{step.selected_token!r}"
                )


def _required_mode_int(
    values: tuple[tuple[str, int | None], ...],
    *,
    mode_name: str,
    field_name: str,
) -> int:
    by_mode = dict(values)
    if mode_name not in by_mode:
        raise ValueError(f"{field_name} is missing for mode {mode_name}")
    value = by_mode[mode_name]
    if value is None:
        raise ValueError(f"{field_name} is missing for mode {mode_name}")
    return int(value)


def _optional_mode_int(
    values: tuple[tuple[str, int | None], ...],
    mode: OnlineDecoderExecutionMode,
) -> int:
    by_mode = dict(values)
    value = by_mode.get(mode.value)
    return 0 if value is None else int(value)


def _require_residual_observation_stats(
    observation: PreparedPrefixQueryObservation,
) -> None:
    if observation.execution_mode is not OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS:
        raise ValueError("residual stat validation received non-residual observation")
    _required_residual_observation_int(observation, field_name="root_dfs_runs")
    _required_residual_observation_int(observation, field_name="resumed_snapshots")
    _required_residual_observation_int(
        observation,
        field_name="retained_continuation_count",
    )
    _required_residual_observation_int(
        observation,
        field_name="retained_render_payload_chars",
    )
    _required_residual_observation_int(
        observation,
        field_name="retained_scheduler_frame_count",
    )
    _required_residual_observation_int(
        observation,
        field_name="retained_prefix_enumeration_frame_count",
    )
    _required_residual_observation_int(
        observation,
        field_name="max_retained_prefix_domain_count",
    )
    _required_residual_observation_int(
        observation,
        field_name="total_retained_prefix_domain_count",
    )
    _required_residual_observation_int(
        observation,
        field_name="max_retained_prefix_assignment_count",
    )
    _required_residual_observation_int(
        observation,
        field_name="total_retained_prefix_assignment_count",
    )
    _required_residual_observation_int(
        observation,
        field_name="retained_direction_enumeration_frame_count",
    )
    _required_residual_observation_int(
        observation,
        field_name="max_retained_direction_carrier_count",
    )
    _required_residual_observation_int(
        observation,
        field_name="total_retained_direction_carrier_count",
    )
    _required_residual_observation_int(
        observation,
        field_name="max_retained_direction_assignment_count",
    )
    _required_residual_observation_int(
        observation,
        field_name="total_retained_direction_assignment_count",
    )
    _required_residual_observation_int(
        observation,
        field_name="retained_support_maximal_frame_count",
    )
    _required_residual_observation_int(
        observation,
        field_name="max_retained_support_maximal_candidate_count",
    )
    _required_residual_observation_int(
        observation,
        field_name="total_retained_support_maximal_candidate_count",
    )
    _required_residual_observation_int(
        observation,
        field_name="max_retained_support_maximal_selected_count",
    )
    _required_residual_observation_int(
        observation,
        field_name="total_retained_support_maximal_selected_count",
    )
    _required_residual_observation_int(
        observation,
        field_name="max_retained_support_maximal_remaining_count",
    )
    _required_residual_observation_int(
        observation,
        field_name="total_retained_support_maximal_remaining_count",
    )


def _require_positive_int(
    value: int | None,
    *,
    field_name: str,
    fixture_name: str,
) -> None:
    if value is None:
        raise ValueError(
            f"{field_name} is missing for prefix-scheduler fixture {fixture_name!r}"
        )
    if value <= 0:
        raise ValueError(
            f"{field_name} is zero for prefix-scheduler fixture {fixture_name!r}"
        )


def _required_residual_observation_int(
    observation: PreparedPrefixQueryObservation,
    *,
    field_name: str,
) -> int:
    if observation.execution_mode is not OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS:
        raise ValueError(f"{field_name} requires a residual observation")
    return _required_observation_int(observation, field_name=field_name)


def _required_observation_int(
    observation: PreparedPrefixQueryObservation,
    *,
    field_name: str,
) -> int:
    value = getattr(observation, field_name)
    if value is None:
        raise ValueError(
            f"{field_name} is missing for mode {observation.execution_mode.value}"
        )
    return int(value)


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
    "PreparedDecoderBranchWalkResult",
    "PreparedDecoderWalkResult",
    "PreparedDecoderWalkStep",
    "PreparedPrefixQueryObservation",
    "PreparedPrefixWorkloadResult",
    "PreparedPrefixWorkloadRow",
    "DIRECTION_SCHEDULER_EVIDENCE_FIXTURES",
    "PREFIX_SCHEDULER_EVIDENCE_FIXTURES",
    "SUPPORT_MAXIMAL_SCHEDULER_EVIDENCE_FIXTURES",
    "advance_decoder_to_prefix",
    "choose_walk_token",
    "collect_prepared_branch_decoder_walks",
    "collect_prepared_decoder_walk",
    "collect_mode_union_token_boundary_prefixes",
    "collect_prepared_prefix_workload",
    "collect_token_boundary_prefixes",
    "require_direction_scheduler_frame_evidence",
    "require_prefix_scheduler_frame_evidence",
    "require_support_maximal_scheduler_frame_evidence",
    "validate_prepared_branch_decoder_walk_result",
    "validate_prepared_decoder_walk_result",
    "validate_prepared_prefix_workload_result",
)
