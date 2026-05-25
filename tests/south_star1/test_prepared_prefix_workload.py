"""Prepared prefix-query workload tests for South Star online decoders."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from dataclasses import replace
from unittest.mock import patch

import grimace._south_star1.online_serialization_stream as serialization_stream_module
import grimace._south_star1.prepared_prefix_workload as workload_module
import grimace._south_star1.prepared_runtime as prepared_runtime_module
import grimace._south_star1.support_enumeration as support_enumeration_module
from grimace._south_star1.facts import ComponentFacts
from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import ComponentId
from grimace._south_star1.online_continuation import OnlineDecoderExecutionMode
from grimace._south_star1.online_decoder_api import make_determinized_online_decoder
from grimace._south_star1.prepared_prefix_workload import advance_decoder_to_prefix
from grimace._south_star1.prepared_prefix_workload import collect_prepared_branch_decoder_walks
from grimace._south_star1.prepared_prefix_workload import collect_prepared_decoder_walk
from grimace._south_star1.prepared_prefix_workload import collect_mode_union_token_boundary_prefixes
from grimace._south_star1.prepared_prefix_workload import collect_prepared_prefix_workload
from grimace._south_star1.prepared_prefix_workload import collect_token_boundary_prefixes
from grimace._south_star1.prepared_prefix_workload import validate_prepared_branch_decoder_walk_result
from grimace._south_star1.prepared_prefix_workload import validate_prepared_decoder_walk_result
from grimace._south_star1.prepared_prefix_workload import validate_prepared_prefix_workload_result
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from tests.south_star1.helpers import atom
from tests.south_star1.helpers import cyclopropane_facts
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import single_bond
from tests.south_star1.helpers import tetrahedral_facts


_WORKLOAD_RESULTS_CACHE = None
_DECODER_WALK_RESULTS_CACHE = None
_BRANCH_WALK_RESULTS_CACHE = None


class PreparedPrefixWorkloadTest(unittest.TestCase):
    def test_branch_walk_visits_multiple_legal_token_branches(self) -> None:
        first_tokens = set()
        for result in _branch_walk_results():
            first_tokens.update(
                walk.steps[0].selected_token
                for walk in result.walks
                if walk.steps and walk.steps[0].selected_token is not None
            )

        self.assertGreaterEqual(len(first_tokens), 2)

    def test_branch_walk_next_token_sets_agree_at_every_state(self) -> None:
        for result in _branch_walk_results():
            with self.subTest(fixture=result.fixture_name):
                for walk in result.walks:
                    for step in walk.steps:
                        token_sets = {item[1] for item in step.next_token_set_by_mode}
                        self.assertEqual(len(token_sets), 1)

    def test_branch_walk_per_token_completion_counts_agree_at_every_state(
        self,
    ) -> None:
        for result in _branch_walk_results():
            with self.subTest(fixture=result.fixture_name):
                for walk in result.walks:
                    for step in walk.steps:
                        count_maps = {
                            item[1]
                            for item in step.next_token_completion_counts_by_mode
                        }
                        self.assertEqual(len(count_maps), 1)

    def test_branch_walk_eos_counts_agree_at_every_state(self) -> None:
        for result in _branch_walk_results():
            with self.subTest(fixture=result.fixture_name):
                for walk in result.walks:
                    for step in walk.steps:
                        eos_counts = {item[1] for item in step.eos_count_by_mode}
                        self.assertEqual(len(eos_counts), 1)

    def test_branch_walk_residual_root_dfs_runs_less_than_prefix_replay(self) -> None:
        for result in _branch_walk_results():
            with self.subTest(fixture=result.fixture_name):
                self.assertLess(
                    result.total_residual_root_dfs_runs,
                    result.total_prefix_replay_root_dfs_runs,
                )

    def test_branch_walk_residual_resumed_snapshots_positive(self) -> None:
        for result in _branch_walk_results():
            with self.subTest(fixture=result.fixture_name):
                self.assertGreater(result.total_residual_resumed_snapshots, 0)

    def test_branch_walk_residual_render_payload_zero(self) -> None:
        for result in _branch_walk_results():
            with self.subTest(fixture=result.fixture_name):
                self.assertEqual(result.max_residual_retained_render_payload_chars, 0)

    def test_branch_walk_probe_reports_zero_hot_path_rebuilds(self) -> None:
        for result in _branch_walk_results():
            with self.subTest(fixture=result.fixture_name):
                probe = result.probe
                self.assertEqual(probe.graph_index_rebuild_count, 0)
                self.assertEqual(probe.online_traversal_graph_from_facts_count, 0)
                self.assertEqual(probe.online_traversal_graph_from_index_count, 0)
                self.assertEqual(probe.prepare_from_facts_count, 0)
                self.assertEqual(probe.prepare_from_rdkit_count, 0)
                self.assertEqual(probe.root_domain_recompute_count, 0)
                self.assertEqual(probe.root_domain_from_metadata_count, 0)
                self.assertEqual(probe.stereo_template_rebuild_count, 0)
                self.assertEqual(probe.facts_validate_count, 0)
                self.assertEqual(probe.policy_validate_count, 0)
                self.assertEqual(probe.online_traversal_graph_view_rebuild_count, 0)
                self.assertEqual(probe.online_vm_graph_view_rebuild_count, 0)

    def test_branch_walk_does_not_call_serialization_collectors(self) -> None:
        prepared = _prepare(tetrahedral_facts())

        with patch.object(
            serialization_stream_module,
            "collect_online_serializations",
            side_effect=AssertionError("branch walk called serialization collector"),
        ), patch.object(
            serialization_stream_module,
            "iter_online_serializations",
            side_effect=AssertionError("branch walk called serialization iterator"),
        ), patch.object(
            serialization_stream_module,
            "count_online_serializations",
            side_effect=AssertionError("branch walk called serialization counter"),
        ):
            result = collect_prepared_branch_decoder_walks(
                fixture_name="tetrahedral",
                prepared=prepared,
                max_walks=2,
            )

        validate_prepared_branch_decoder_walk_result(result)

    def test_branch_walk_positive_control_detects_per_token_completion_count_disagreement(
        self,
    ) -> None:
        result = _branch_walk_results()[0]
        walk = result.walks[0]
        step = walk.steps[0]
        tampered = replace(
            result,
            walks=(
                replace(
                    walk,
                    steps=(
                        replace(
                            step,
                            next_token_completion_counts_by_mode=(
                                step.next_token_completion_counts_by_mode[0],
                                step.next_token_completion_counts_by_mode[1],
                                (
                                    OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
                                    (("not-a-token", 99),),
                                ),
                            ),
                        ),
                        *walk.steps[1:],
                    ),
                ),
                *result.walks[1:],
            ),
        )

        with self.assertRaisesRegex(ValueError, "completion count disagreement"):
            validate_prepared_branch_decoder_walk_result(tampered)

    def test_branch_walk_positive_control_detects_sibling_branch_mode_disagreement(
        self,
    ) -> None:
        result = _branch_walk_results()[0]
        walk = result.walks[-1]
        step = walk.steps[0]
        tampered = replace(
            result,
            walks=(
                *result.walks[:-1],
                replace(
                    walk,
                    steps=(
                        replace(
                            step,
                            next_token_set_by_mode=(
                                step.next_token_set_by_mode[0],
                                step.next_token_set_by_mode[1],
                                (
                                    OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
                                    frozenset({"not-a-token"}),
                                ),
                            ),
                        ),
                        *walk.steps[1:],
                    ),
                ),
            ),
        )

        with self.assertRaisesRegex(ValueError, "next-token disagreement"):
            validate_prepared_branch_decoder_walk_result(tampered)

    def test_decoder_walk_next_token_sets_agree_at_every_step(self) -> None:
        for result in _decoder_walk_results():
            with self.subTest(fixture=result.fixture_name):
                for step in result.steps:
                    token_sets = {item[1] for item in step.next_token_set_by_mode}
                    self.assertEqual(len(token_sets), 1)

    def test_decoder_walk_eos_counts_agree_at_every_step(self) -> None:
        for result in _decoder_walk_results():
            with self.subTest(fixture=result.fixture_name):
                for step in result.steps:
                    eos_counts = {item[1] for item in step.eos_count_by_mode}
                    self.assertEqual(len(eos_counts), 1)

    def test_decoder_walk_accumulates_intermediate_choice_stats(self) -> None:
        for result in _decoder_walk_results():
            with self.subTest(fixture=result.fixture_name):
                self.assertGreater(len(result.steps), 1)
                self.assertEqual(
                    result.total_prefix_replay_root_dfs_runs,
                    sum(
                        _step_value(
                            step.root_dfs_runs_by_mode,
                            OnlineDecoderExecutionMode.PREFIX_REPLAY,
                        )
                        for step in result.steps
                    ),
                )
                self.assertGreater(result.total_prefix_replay_root_dfs_runs, 1)

    def test_decoder_walk_residual_root_dfs_runs_less_than_prefix_replay(self) -> None:
        for result in _decoder_walk_results():
            with self.subTest(fixture=result.fixture_name):
                self.assertLess(
                    result.total_residual_root_dfs_runs,
                    result.total_prefix_replay_root_dfs_runs,
                )

    def test_decoder_walk_residual_resumed_snapshots_positive(self) -> None:
        for result in _decoder_walk_results():
            with self.subTest(fixture=result.fixture_name):
                self.assertGreater(result.total_residual_resumed_snapshots, 0)

    def test_decoder_walk_residual_render_payload_zero(self) -> None:
        for result in _decoder_walk_results():
            with self.subTest(fixture=result.fixture_name):
                self.assertEqual(result.max_residual_retained_render_payload_chars, 0)

    def test_decoder_walk_probe_reports_zero_hot_path_rebuilds(self) -> None:
        for result in _decoder_walk_results():
            with self.subTest(fixture=result.fixture_name):
                probe = result.probe
                self.assertEqual(probe.graph_index_rebuild_count, 0)
                self.assertEqual(probe.online_traversal_graph_from_facts_count, 0)
                self.assertEqual(probe.online_traversal_graph_from_index_count, 0)
                self.assertEqual(probe.prepare_from_facts_count, 0)
                self.assertEqual(probe.prepare_from_rdkit_count, 0)
                self.assertEqual(probe.root_domain_recompute_count, 0)
                self.assertEqual(probe.root_domain_from_metadata_count, 0)
                self.assertEqual(probe.stereo_template_rebuild_count, 0)
                self.assertEqual(probe.facts_validate_count, 0)
                self.assertEqual(probe.policy_validate_count, 0)
                self.assertEqual(probe.online_traversal_graph_view_rebuild_count, 0)
                self.assertEqual(probe.online_vm_graph_view_rebuild_count, 0)

    def test_decoder_walk_does_not_call_serialization_collectors(self) -> None:
        prepared = _prepare(tetrahedral_facts())

        with patch.object(
            serialization_stream_module,
            "collect_online_serializations",
            side_effect=AssertionError("decoder walk called serialization collector"),
        ), patch.object(
            serialization_stream_module,
            "iter_online_serializations",
            side_effect=AssertionError("decoder walk called serialization iterator"),
        ), patch.object(
            serialization_stream_module,
            "count_online_serializations",
            side_effect=AssertionError("decoder walk called serialization counter"),
        ):
            result = collect_prepared_decoder_walk(
                fixture_name="tetrahedral",
                prepared=prepared,
            )

        validate_prepared_decoder_walk_result(result)

    def test_decoder_walk_positive_control_detects_intermediate_mode_disagreement(
        self,
    ) -> None:
        result = _decoder_walk_results()[0]
        step = result.steps[0]
        tampered = replace(
            result,
            steps=(
                replace(
                    step,
                    next_token_set_by_mode=(
                        step.next_token_set_by_mode[0],
                        step.next_token_set_by_mode[1],
                        (
                            OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS.value,
                            frozenset({"not-a-token"}),
                        ),
                    ),
                ),
                *result.steps[1:],
            ),
        )

        with self.assertRaisesRegex(ValueError, "next-token disagreement"):
            validate_prepared_decoder_walk_result(tampered)

    def test_prefix_workload_collects_prefixes_from_all_execution_modes(self) -> None:
        with patch.object(
            workload_module,
            "make_determinized_online_decoder",
            side_effect=_fake_decoder_factory(),
        ):
            prefixes = collect_mode_union_token_boundary_prefixes(
                prepared=object(),
                limit_per_mode=2,
            )

        self.assertEqual(prefixes, ("", "A", "B", "C"))

    def test_prefix_workload_requires_every_mode_to_reach_every_prefix(self) -> None:
        with patch.object(
            workload_module,
            "make_determinized_online_decoder",
            side_effect=_fake_decoder_factory(
                missing={
                    OnlineDecoderExecutionMode.CACHED_COMPLETIONS: frozenset({"B"}),
                },
            ),
        ):
            with self.assertRaisesRegex(ValueError, "not reachable"):
                collect_mode_union_token_boundary_prefixes(
                    prepared=object(),
                    limit_per_mode=3,
                )

    def test_prefix_workload_fails_if_residual_omits_prefix_seen_by_prefix_replay(
        self,
    ) -> None:
        with patch.object(
            workload_module,
            "make_determinized_online_decoder",
            side_effect=_fake_decoder_factory(
                missing={
                    OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS: frozenset({"B"}),
                },
            ),
        ):
            with self.assertRaisesRegex(ValueError, "residual_continuations"):
                collect_mode_union_token_boundary_prefixes(
                    prepared=object(),
                    limit_per_mode=3,
                )

    def test_prefix_workload_fails_if_cached_completions_omits_prefix_seen_by_prefix_replay(
        self,
    ) -> None:
        with patch.object(
            workload_module,
            "make_determinized_online_decoder",
            side_effect=_fake_decoder_factory(
                missing={
                    OnlineDecoderExecutionMode.CACHED_COMPLETIONS: frozenset({"B"}),
                },
            ),
        ):
            with self.assertRaisesRegex(ValueError, "cached_completions"):
                collect_mode_union_token_boundary_prefixes(
                    prepared=object(),
                    limit_per_mode=3,
                )

    def test_prefix_workload_eos_availability_is_observed_directly(self) -> None:
        with patch.object(
            workload_module,
            "make_determinized_online_decoder",
            side_effect=_fake_decoder_factory(eos_completion_count=2),
        ):
            result = collect_prepared_prefix_workload(
                fixture_name="fake",
                prepared=object(),
                prefix_limit=1,
            )

        self.assertTrue(result.rows[0].prefix_replay.has_eos)
        self.assertEqual(result.rows[0].prefix_replay.eos_completion_count, 2)

    def test_prefix_workload_rejects_zero_completion_count_eos(self) -> None:
        with patch.object(
            workload_module,
            "make_determinized_online_decoder",
            side_effect=_fake_decoder_factory(eos_completion_count=0),
        ):
            with self.assertRaisesRegex(ValueError, "positive completion_count"):
                collect_prepared_prefix_workload(
                    fixture_name="fake",
                    prepared=object(),
                    prefix_limit=1,
                )

    def test_prefix_workload_next_token_sets_agree_across_execution_modes(self) -> None:
        for result in _workload_results():
            with self.subTest(fixture=result.rows[0].fixture_name):
                for row in result.rows:
                    self.assertEqual(
                        row.prefix_replay.next_token_text_set,
                        row.cached_completions.next_token_text_set,
                    )
                    self.assertEqual(
                        row.prefix_replay.next_token_text_set,
                        row.residual_continuations.next_token_text_set,
                    )
                    self.assertEqual(
                        row.prefix_replay.next_token_completion_counts,
                        row.cached_completions.next_token_completion_counts,
                    )
                    self.assertEqual(
                        row.prefix_replay.next_token_completion_counts,
                        row.residual_continuations.next_token_completion_counts,
                    )

    def test_prefix_workload_eos_availability_agrees_across_execution_modes(self) -> None:
        for result in _workload_results():
            with self.subTest(fixture=result.rows[0].fixture_name):
                for row in result.rows:
                    self.assertEqual(
                        row.prefix_replay.has_eos,
                        row.cached_completions.has_eos,
                    )
                    self.assertEqual(
                        row.prefix_replay.has_eos,
                        row.residual_continuations.has_eos,
                    )

    def test_prefix_workload_eos_completion_count_agrees_across_execution_modes(
        self,
    ) -> None:
        for result in _workload_results():
            with self.subTest(fixture=result.rows[0].fixture_name):
                for row in result.rows:
                    self.assertEqual(
                        row.prefix_replay.eos_completion_count,
                        row.cached_completions.eos_completion_count,
                    )
                    self.assertEqual(
                        row.prefix_replay.eos_completion_count,
                        row.residual_continuations.eos_completion_count,
                    )

    def test_prefix_workload_still_reports_residual_resumed_snapshots(self) -> None:
        for result in _workload_results():
            with self.subTest(fixture=result.rows[0].fixture_name):
                self.assertGreater(result.total_residual_resumed_snapshots, 0)

    def test_prefix_workload_still_reports_residual_root_dfs_reduction(
        self,
    ) -> None:
        for result in _workload_results():
            with self.subTest(fixture=result.rows[0].fixture_name):
                self.assertLess(
                    result.total_residual_root_dfs_runs,
                    result.total_prefix_replay_root_dfs_runs,
                )

    def test_prefix_workload_still_reports_zero_retained_render_payload(self) -> None:
        for result in _workload_results():
            with self.subTest(fixture=result.rows[0].fixture_name):
                self.assertEqual(result.max_residual_retained_render_payload_chars, 0)
                for row in result.rows:
                    self.assertIn(
                        row.residual_continuations.retained_render_payload_chars,
                        (0, None),
                    )

    def test_prefix_workload_uses_token_boundary_prefixes_only(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        prefixes = collect_token_boundary_prefixes(prepared=prepared, limit=16)
        decoder = make_determinized_online_decoder(
            prepared=prepared,
            include_eos=True,
            execution_mode=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
        )

        for prefix in prefixes:
            self.assertEqual(advance_decoder_to_prefix(decoder, prefix).prefix, prefix)
        self.assertIn("", prefixes)
        self._assert_rejects_mid_token_prefix(prepared, prefixes)

    def test_prefix_workload_does_not_call_offline_support_for_legality(self) -> None:
        prepared = _prepare(tetrahedral_facts())

        with patch.object(
            prepared_runtime_module,
            "enumerate_prepared_stereo_support",
            side_effect=AssertionError("prefix workload called prepared support"),
        ), patch.object(
            support_enumeration_module,
            "enumerate_stereo_support",
            side_effect=AssertionError("prefix workload called offline support"),
        ):
            result = collect_prepared_prefix_workload(
                fixture_name="tetrahedral",
                prepared=prepared,
            )

        validate_prepared_prefix_workload_result(result)

    def test_prefix_workload_does_not_call_serialization_collectors(self) -> None:
        prepared = _prepare(tetrahedral_facts())

        with patch.object(
            serialization_stream_module,
            "collect_online_serializations",
            side_effect=AssertionError("prefix workload called serialization collector"),
        ), patch.object(
            serialization_stream_module,
            "iter_online_serializations",
            side_effect=AssertionError("prefix workload called serialization iterator"),
        ), patch.object(
            serialization_stream_module,
            "count_online_serializations",
            side_effect=AssertionError("prefix workload called serialization counter"),
        ):
            result = collect_prepared_prefix_workload(
                fixture_name="tetrahedral",
                prepared=prepared,
            )

        validate_prepared_prefix_workload_result(result)

    def test_prefix_workload_probe_still_reports_zero_hot_path_rebuilds(self) -> None:
        for result in _workload_results():
            with self.subTest(fixture=result.rows[0].fixture_name):
                probe = result.probe
                self.assertEqual(probe.graph_index_rebuild_count, 0)
                self.assertEqual(probe.online_traversal_graph_from_facts_count, 0)
                self.assertEqual(probe.online_traversal_graph_from_index_count, 0)
                self.assertEqual(probe.prepare_from_facts_count, 0)
                self.assertEqual(probe.prepare_from_rdkit_count, 0)
                self.assertEqual(probe.root_domain_recompute_count, 0)
                self.assertEqual(probe.root_domain_from_metadata_count, 0)
                self.assertEqual(probe.stereo_template_rebuild_count, 0)
                self.assertEqual(probe.facts_validate_count, 0)
                self.assertEqual(probe.policy_validate_count, 0)
                self.assertEqual(probe.online_traversal_graph_view_rebuild_count, 0)
                self.assertEqual(probe.online_vm_graph_view_rebuild_count, 0)

    def test_prefix_workload_detects_forced_next_token_disagreement(self) -> None:
        result = _workload_results()[0]
        row = result.rows[0]
        tampered = replace(
            result,
            rows=(
                replace(
                    row,
                    residual_continuations=replace(
                        row.residual_continuations,
                        next_token_texts=("not-a-token",),
                        next_token_text_set=frozenset({"not-a-token"}),
                    ),
                ),
                *result.rows[1:],
            ),
        )

        with self.assertRaisesRegex(ValueError, "next-token disagreement"):
            validate_prepared_prefix_workload_result(tampered)

    def _assert_rejects_mid_token_prefix(
        self,
        prepared,
        prefixes: tuple[str, ...],
    ) -> None:
        for prefix in prefixes:
            decoder = make_determinized_online_decoder(
                prepared=prepared,
                include_eos=True,
                execution_mode=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
            )
            state = advance_decoder_to_prefix(decoder, prefix)
            result = state.choices_with_stats()
            for choice in result.choices:
                if choice.is_eos or len(choice.text) < 2:
                    continue
                mid_token = prefix + choice.text[:1]
                if mid_token in prefixes:
                    continue
                with self.assertRaisesRegex(ValueError, "token boundary"):
                    advance_decoder_to_prefix(decoder, mid_token)
                return
        self.skipTest("fixture did not expose a multi-character decoder token")


@dataclass(frozen=True, slots=True)
class _Fixture:
    name: str
    facts: MoleculeFacts
    runtime_options: SouthStarRuntimeOptions = SouthStarRuntimeOptions()


def _workload_results():
    global _WORKLOAD_RESULTS_CACHE
    if _WORKLOAD_RESULTS_CACHE is not None:
        return _WORKLOAD_RESULTS_CACHE
    _WORKLOAD_RESULTS_CACHE = tuple(
        collect_prepared_prefix_workload(
            fixture_name=fixture.name,
            prepared=_prepare(fixture.facts),
            runtime_options=fixture.runtime_options,
            prefix_limit=12,
        )
        for fixture in _fixtures()
    )
    return _WORKLOAD_RESULTS_CACHE


def _decoder_walk_results():
    global _DECODER_WALK_RESULTS_CACHE
    if _DECODER_WALK_RESULTS_CACHE is not None:
        return _DECODER_WALK_RESULTS_CACHE
    _DECODER_WALK_RESULTS_CACHE = tuple(
        collect_prepared_decoder_walk(
            fixture_name=fixture.name,
            prepared=_prepare(fixture.facts),
            runtime_options=fixture.runtime_options,
            max_steps=16,
        )
        for fixture in _fixtures()
    )
    return _DECODER_WALK_RESULTS_CACHE


def _branch_walk_results():
    global _BRANCH_WALK_RESULTS_CACHE
    if _BRANCH_WALK_RESULTS_CACHE is not None:
        return _BRANCH_WALK_RESULTS_CACHE
    _BRANCH_WALK_RESULTS_CACHE = tuple(
        collect_prepared_branch_decoder_walks(
            fixture_name=fixture.name,
            prepared=_prepare(fixture.facts),
            runtime_options=fixture.runtime_options,
            max_walks=4,
            max_steps_per_walk=32,
        )
        for fixture in _branch_fixtures()
    )
    return _BRANCH_WALK_RESULTS_CACHE


def _step_value(values, mode: OnlineDecoderExecutionMode) -> int:
    value = dict(values)[mode.value]
    return 0 if value is None else int(value)


def _branch_fixtures() -> tuple[_Fixture, ...]:
    return (
        _Fixture("tetrahedral", tetrahedral_facts()),
        _Fixture("directional", directional_facts()),
        _Fixture("ring", cyclopropane_facts()),
        _Fixture(
            "disconnected-stereo",
            _disconnected_tetra_and_bond_facts(),
            SouthStarRuntimeOptions(rooted_at_atom=5),
        ),
    )


def _fixtures() -> tuple[_Fixture, ...]:
    return (
        _Fixture("tetrahedral", tetrahedral_facts()),
        _Fixture("directional", directional_facts()),
        _Fixture("ring", cyclopropane_facts()),
        _Fixture("support-maximal", directional_facts()),
        _Fixture("duplicate-render", cyclopropane_facts()),
        _Fixture(
            "connected-multi-root",
            tetrahedral_facts(),
            SouthStarRuntimeOptions(rooted_at_atom=0),
        ),
        _Fixture(
            "disconnected-stereo",
            _disconnected_tetra_and_bond_facts(),
            SouthStarRuntimeOptions(rooted_at_atom=5),
        ),
        _Fixture("sparse-atom-id", _sparse_two_atom_facts()),
    )


def _prepare(facts):
    return prepare_south_star_mol_from_facts(
        facts,
        writer_surface=SouthStarWriterSurface(),
    )


@dataclass(frozen=True, slots=True)
class _FakeChoice:
    text: str
    next_state: object | None
    is_eos: bool = False
    completion_count: int = 1
    multiplicity: int = 1


@dataclass(frozen=True, slots=True)
class _FakeChoiceResult:
    choices: tuple[_FakeChoice, ...]
    stats: object


@dataclass(frozen=True, slots=True)
class _FakeStats:
    root_dfs_runs: int = 0
    resumed_snapshots: int = 0


@dataclass(frozen=True, slots=True)
class _FakeState:
    prefix: str
    execution_mode: OnlineDecoderExecutionMode
    missing: frozenset[str]
    eos_completion_count: int

    def choices_with_stats(self) -> _FakeChoiceResult:
        choices: list[_FakeChoice] = []
        if self.prefix == "":
            for text in _mode_order(self.execution_mode):
                if text in self.missing:
                    continue
                choices.append(
                    _FakeChoice(
                        text=text,
                        next_state=_FakeState(
                            prefix=text,
                            execution_mode=self.execution_mode,
                            missing=self.missing,
                            eos_completion_count=self.eos_completion_count,
                        ),
                    )
                )
        choices.append(
            _FakeChoice(
                text="<EOS>",
                next_state=None,
                is_eos=True,
                completion_count=self.eos_completion_count,
            )
        )
        is_residual_resume = (
            self.execution_mode is OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS
            and bool(self.prefix)
        )
        return _FakeChoiceResult(
            choices=tuple(choices),
            stats=_FakeStats(
                root_dfs_runs=0 if is_residual_resume else 1,
                resumed_snapshots=1 if is_residual_resume else 0,
            ),
        )


@dataclass(frozen=True, slots=True)
class _FakeDecoder:
    execution_mode: OnlineDecoderExecutionMode
    missing: frozenset[str]
    eos_completion_count: int

    def initial_state(self) -> _FakeState:
        return _FakeState(
            prefix="",
            execution_mode=self.execution_mode,
            missing=self.missing,
            eos_completion_count=self.eos_completion_count,
        )


def _fake_decoder_factory(
    *,
    missing: dict[OnlineDecoderExecutionMode, frozenset[str]] | None = None,
    eos_completion_count: int = 1,
):
    missing_by_mode = missing or {}

    def make_decoder(**kwargs):
        mode = kwargs["execution_mode"]
        return _FakeDecoder(
            execution_mode=mode,
            missing=missing_by_mode.get(mode, frozenset()),
            eos_completion_count=eos_completion_count,
        )

    return make_decoder


def _mode_order(mode: OnlineDecoderExecutionMode) -> tuple[str, ...]:
    if mode is OnlineDecoderExecutionMode.PREFIX_REPLAY:
        return ("A", "B", "C")
    if mode is OnlineDecoderExecutionMode.CACHED_COMPLETIONS:
        return ("B", "A", "C")
    if mode is OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS:
        return ("C", "A", "B")
    raise ValueError(f"unexpected fake decoder mode: {mode!r}")


def _disconnected_tetra_and_bond_facts() -> MoleculeFacts:
    tetra = tetrahedral_facts()
    return MoleculeFacts(
        atoms=(
            atom(0, "C"),
            atom(1, "F"),
            atom(2, "Cl"),
            atom(3, "Br"),
            atom(5, "C"),
            atom(6, "O"),
        ),
        bonds=(
            single_bond(0, 0, 1),
            single_bond(1, 0, 2),
            single_bond(2, 0, 3),
            single_bond(4, 5, 6),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1), AtomId(2), AtomId(3)),
                bonds=(BondId(0), BondId(1), BondId(2)),
            ),
            ComponentFacts(
                id=ComponentId(1),
                atoms=(AtomId(5), AtomId(6)),
                bonds=(BondId(4),),
            ),
        ),
        stereo=tetra.stereo,
        ligand_occurrences=tetra.ligand_occurrences,
    )


def _sparse_two_atom_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(10, "C"), atom(20, "O")),
        bonds=(single_bond(30, 10, 20),),
        components=(
            ComponentFacts(
                id=ComponentId(7),
                atoms=(AtomId(10), AtomId(20)),
                bonds=(BondId(30),),
            ),
        ),
    )


if __name__ == "__main__":
    unittest.main()
