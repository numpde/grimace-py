"""Prepared South Star conformance and structural-efficiency matrix helpers."""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Callable

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
    probe: "PreparedRuntimeProbeResult"


@dataclass(frozen=True, slots=True)
class PreparedRuntimeProbeResult:
    graph_index_rebuild_count: int
    online_traversal_graph_from_facts_count: int
    online_traversal_graph_from_index_count: int
    prepare_from_facts_count: int
    prepare_from_rdkit_count: int
    root_domain_recompute_count: int
    root_domain_from_metadata_count: int
    stereo_template_rebuild_count: int
    facts_validate_count: int
    policy_validate_count: int
    online_traversal_graph_view_rebuild_count: int
    online_vm_graph_view_rebuild_count: int


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
) -> PreparedEnumerationMatrixEntry:
    with PreparedRuntimeProbe() as probe:
        offline = enumerate_prepared_stereo_support(
            prepared=prepared,
            runtime_options=runtime_options,
        )
        online = _walk_prepared_decoder(
            prepared=prepared,
            runtime_options=runtime_options,
            execution_mode=execution_mode,
        )
    probe_result = probe.result()
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
        retained_scheduler_frame_count=online.retained_scheduler_frame_count,
        retained_prefix_enumeration_frame_count=(
            online.retained_prefix_enumeration_frame_count
        ),
        max_retained_prefix_domain_count=online.max_retained_prefix_domain_count,
        total_retained_prefix_domain_count=online.total_retained_prefix_domain_count,
        max_retained_prefix_assignment_count=(
            online.max_retained_prefix_assignment_count
        ),
        total_retained_prefix_assignment_count=(
            online.total_retained_prefix_assignment_count
        ),
        retained_direction_enumeration_frame_count=(
            online.retained_direction_enumeration_frame_count
        ),
        max_retained_direction_carrier_count=online.max_retained_direction_carrier_count,
        total_retained_direction_carrier_count=(
            online.total_retained_direction_carrier_count
        ),
        max_retained_direction_assignment_count=(
            online.max_retained_direction_assignment_count
        ),
        total_retained_direction_assignment_count=(
            online.total_retained_direction_assignment_count
        ),
        probe=probe_result,
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


class PreparedRuntimeProbe:
    def __init__(self) -> None:
        self._counts = {
            "graph_index_rebuild_count": 0,
            "online_traversal_graph_from_facts_count": 0,
            "online_traversal_graph_from_index_count": 0,
            "prepare_from_facts_count": 0,
            "prepare_from_rdkit_count": 0,
            "root_domain_recompute_count": 0,
            "root_domain_from_metadata_count": 0,
            "stereo_template_rebuild_count": 0,
            "facts_validate_count": 0,
            "policy_validate_count": 0,
            "online_traversal_graph_view_rebuild_count": 0,
            "online_vm_graph_view_rebuild_count": 0,
        }
        self._patches: list[tuple[object, str, object]] = []

    def __enter__(self) -> "PreparedRuntimeProbe":
        self._install_patches()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        for owner, name, original in reversed(self._patches):
            setattr(owner, name, original)
        self._patches.clear()

    def result(self) -> PreparedRuntimeProbeResult:
        return PreparedRuntimeProbeResult(**self._counts)

    def _install_patches(self) -> None:
        from . import graph_index as graph_index_module
        from . import online_search_vm as online_search_vm_module
        from . import online_serialization_stream as online_serialization_stream_module
        from . import online_stereo_witness as online_stereo_witness_module
        from . import online_traversal as online_traversal_module
        from . import online_traversal_graph as online_traversal_graph_module
        from . import prepared_runtime as prepared_runtime_module
        from . import root_domains as root_domains_module
        from . import skeleton as skeleton_module
        from . import stereo_templates as stereo_templates_module
        from . import support_enumeration as support_enumeration_module
        from .facts import MoleculeFacts
        from .policy import SmilesPolicy

        self._patch_callable(
            graph_index_module,
            "build_graph_index",
            "graph_index_rebuild_count",
        )
        self._patch_callable(
            prepared_runtime_module,
            "build_graph_index",
            "graph_index_rebuild_count",
        )
        self._patch_callable(
            support_enumeration_module,
            "build_graph_index",
            "graph_index_rebuild_count",
        )
        self._patch_callable(
            online_traversal_graph_module,
            "build_online_traversal_graph_from_facts",
            "online_traversal_graph_from_facts_count",
        )
        self._patch_callable(
            online_traversal_module,
            "build_online_traversal_graph_from_facts",
            "online_traversal_graph_from_facts_count",
        )
        self._patch_callable(
            online_search_vm_module,
            "build_online_traversal_graph_from_facts",
            "online_traversal_graph_from_facts_count",
        )
        self._patch_callable(
            online_traversal_graph_module,
            "build_online_traversal_graph_from_index",
            "online_traversal_graph_from_index_count",
        )
        self._patch_callable(
            online_traversal_module,
            "build_online_traversal_graph_from_index",
            "online_traversal_graph_from_index_count",
        )
        self._patch_callable(
            online_search_vm_module,
            "build_online_traversal_graph_from_index",
            "online_traversal_graph_from_index_count",
        )
        self._patch_callable(
            prepared_runtime_module,
            "build_online_traversal_graph_from_index",
            "online_traversal_graph_from_index_count",
        )
        self._patch_callable(
            prepared_runtime_module,
            "prepare_south_star_mol_from_facts",
            "prepare_from_facts_count",
        )
        self._patch_callable(
            online_serialization_stream_module,
            "prepare_south_star_mol_from_facts",
            "prepare_from_facts_count",
        )
        self._patch_callable(
            prepared_runtime_module,
            "prepare_south_star_mol_from_rdkit",
            "prepare_from_rdkit_count",
        )
        self._patch_callable(
            root_domains_module,
            "component_root_domains_for_facts",
            "root_domain_recompute_count",
        )
        self._patch_callable(
            prepared_runtime_module,
            "component_root_domains_for_facts",
            "root_domain_recompute_count",
        )
        self._patch_callable(
            skeleton_module,
            "component_root_domains_for_facts",
            "root_domain_recompute_count",
        )
        self._patch_callable(
            online_traversal_module,
            "component_root_domains_for_facts",
            "root_domain_recompute_count",
        )
        self._patch_callable(
            online_search_vm_module,
            "component_root_domains_for_facts",
            "root_domain_recompute_count",
        )
        self._patch_callable(
            root_domains_module,
            "component_root_domains_from_metadata",
            "root_domain_from_metadata_count",
        )
        self._patch_callable(
            prepared_runtime_module,
            "component_root_domains_from_metadata",
            "root_domain_from_metadata_count",
        )
        self._patch_callable(
            stereo_templates_module,
            "build_stereo_templates",
            "stereo_template_rebuild_count",
        )
        self._patch_callable(
            prepared_runtime_module,
            "build_stereo_templates",
            "stereo_template_rebuild_count",
        )
        self._patch_callable(
            online_search_vm_module,
            "build_stereo_templates",
            "stereo_template_rebuild_count",
        )
        self._patch_callable(
            online_stereo_witness_module,
            "build_stereo_templates",
            "stereo_template_rebuild_count",
        )
        self._patch_callable(
            online_traversal_module,
            "_graph_from_facts",
            "online_traversal_graph_view_rebuild_count",
        )
        self._patch_callable(
            online_search_vm_module,
            "_graph_from_facts",
            "online_vm_graph_view_rebuild_count",
        )
        self._patch_callable(MoleculeFacts, "validate", "facts_validate_count")
        self._patch_callable(
            SmilesPolicy,
            "validate_for_facts",
            "policy_validate_count",
        )

    def _patch_callable(
        self,
        owner: ModuleType | type,
        name: str,
        counter_name: str,
    ) -> None:
        original = getattr(owner, name)
        self._patches.append((owner, name, original))
        setattr(owner, name, self._counting_wrapper(counter_name, original))

    def _counting_wrapper(
        self,
        counter_name: str,
        original: Callable[..., object],
    ) -> Callable[..., object]:
        def wrapped(*args: object, **kwargs: object) -> object:
            self._counts[counter_name] += 1
            return original(*args, **kwargs)

        return wrapped


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
    retained_scheduler_frame_count: int | None = None
    retained_prefix_enumeration_frame_count: int | None = None
    max_retained_prefix_domain_count: int | None = None
    total_retained_prefix_domain_count: int | None = None
    max_retained_prefix_assignment_count: int | None = None
    total_retained_prefix_assignment_count: int | None = None
    retained_direction_enumeration_frame_count: int | None = None
    max_retained_direction_carrier_count: int | None = None
    total_retained_direction_carrier_count: int | None = None
    max_retained_direction_assignment_count: int | None = None
    total_retained_direction_assignment_count: int | None = None

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
            retained_scheduler_frame_count = _optional_max(
                retained_scheduler_frame_count,
                int(retained.scheduler_frame_count),
            )
            retained_prefix_enumeration_frame_count = _optional_max(
                retained_prefix_enumeration_frame_count,
                int(retained.prefix_enumeration_frame_count),
            )
            max_retained_prefix_domain_count = _optional_max(
                max_retained_prefix_domain_count,
                int(retained.max_prefix_domain_count),
            )
            total_retained_prefix_domain_count = _optional_max(
                total_retained_prefix_domain_count,
                int(retained.total_prefix_domain_count),
            )
            max_retained_prefix_assignment_count = _optional_max(
                max_retained_prefix_assignment_count,
                int(retained.max_prefix_assignment_count),
            )
            total_retained_prefix_assignment_count = _optional_max(
                total_retained_prefix_assignment_count,
                int(retained.total_prefix_assignment_count),
            )
            retained_direction_enumeration_frame_count = _optional_max(
                retained_direction_enumeration_frame_count,
                int(retained.direction_enumeration_frame_count),
            )
            max_retained_direction_carrier_count = _optional_max(
                max_retained_direction_carrier_count,
                int(retained.max_direction_carrier_count),
            )
            total_retained_direction_carrier_count = _optional_max(
                total_retained_direction_carrier_count,
                int(retained.total_direction_carrier_count),
            )
            max_retained_direction_assignment_count = _optional_max(
                max_retained_direction_assignment_count,
                int(retained.max_direction_assignment_count),
            )
            total_retained_direction_assignment_count = _optional_max(
                total_retained_direction_assignment_count,
                int(retained.total_direction_assignment_count),
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
        retained_scheduler_frame_count=retained_scheduler_frame_count,
        retained_prefix_enumeration_frame_count=(
            retained_prefix_enumeration_frame_count
        ),
        max_retained_prefix_domain_count=max_retained_prefix_domain_count,
        total_retained_prefix_domain_count=total_retained_prefix_domain_count,
        max_retained_prefix_assignment_count=max_retained_prefix_assignment_count,
        total_retained_prefix_assignment_count=total_retained_prefix_assignment_count,
        retained_direction_enumeration_frame_count=(
            retained_direction_enumeration_frame_count
        ),
        max_retained_direction_carrier_count=max_retained_direction_carrier_count,
        total_retained_direction_carrier_count=total_retained_direction_carrier_count,
        max_retained_direction_assignment_count=(
            max_retained_direction_assignment_count
        ),
        total_retained_direction_assignment_count=(
            total_retained_direction_assignment_count
        ),
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
    "PreparedRuntimeProbe",
    "PreparedRuntimeProbeResult",
    "collect_prepared_enumeration_matrix_entry",
    "collect_prepared_prefix_workload_stats",
)
