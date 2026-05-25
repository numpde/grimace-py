"""Prepared prefix-query workload tests for South Star online decoders."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from dataclasses import replace
from unittest.mock import patch

import grimace._south_star1.online_serialization_stream as serialization_stream_module
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
from grimace._south_star1.prepared_prefix_workload import collect_prepared_prefix_workload
from grimace._south_star1.prepared_prefix_workload import collect_token_boundary_prefixes
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


class PreparedPrefixWorkloadTest(unittest.TestCase):
    def test_prefix_workload_next_tokens_agree_across_execution_modes(self) -> None:
        for result in _workload_results():
            with self.subTest(fixture=result.rows[0].fixture_name):
                for row in result.rows:
                    self.assertEqual(
                        row.prefix_replay.next_token_texts,
                        row.cached_completions.next_token_texts,
                    )
                    self.assertEqual(
                        row.prefix_replay.next_token_texts,
                        row.residual_continuations.next_token_texts,
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

    def test_prefix_workload_residual_uses_resumed_snapshots(self) -> None:
        for result in _workload_results():
            with self.subTest(fixture=result.rows[0].fixture_name):
                self.assertGreater(result.total_residual_resumed_snapshots, 0)

    def test_prefix_workload_residual_has_fewer_root_dfs_runs_than_prefix_replay(
        self,
    ) -> None:
        for result in _workload_results():
            with self.subTest(fixture=result.rows[0].fixture_name):
                self.assertLess(
                    result.total_residual_root_dfs_runs,
                    result.total_prefix_replay_root_dfs_runs,
                )

    def test_prefix_workload_residual_retained_render_payload_chars_zero(self) -> None:
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

    def test_prefix_workload_prepared_hot_path_has_zero_probe_rebuilds(self) -> None:
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
