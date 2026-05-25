"""Prepared South Star conformance and structural-efficiency matrix tests."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import patch

import grimace._south_star1.graph_index as graph_index_module
from grimace._south_star1.facts import ComponentFacts
from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import ComponentId
from grimace._south_star1.online_continuation import OnlineDecoderExecutionMode
from grimace._south_star1.prepared_bench_matrix import PreparedEnumerationMatrixEntry
from grimace._south_star1.prepared_bench_matrix import PreparedRuntimeProbe
from grimace._south_star1.prepared_bench_matrix import collect_prepared_enumeration_matrix_entry
from grimace._south_star1.prepared_bench_matrix import collect_prepared_prefix_workload_stats
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from tests.south_star1.helpers import atom
from tests.south_star1.helpers import cyclopropane_facts
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import single_bond
from tests.south_star1.helpers import tetrahedral_facts


class PreparedEfficiencyMatrixTest(unittest.TestCase):
    def test_prepared_matrix_matches_offline_support_and_counts(self) -> None:
        for fixture in _matrix_fixtures():
            prepared = prepare_south_star_mol_from_facts(
                fixture.facts,
                writer_surface=SouthStarWriterSurface(),
            )
            for mode in (
                OnlineDecoderExecutionMode.PREFIX_REPLAY,
                OnlineDecoderExecutionMode.CACHED_COMPLETIONS,
                OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
            ):
                with self.subTest(fixture=fixture.name, mode=mode.value):
                    entry = _guarded_matrix_entry(
                        fixture_name=fixture.name,
                        prepared=prepared,
                        runtime_options=fixture.runtime_options,
                        execution_mode=mode,
                    )
                    _assert_entry_conforms(self, entry)

    def test_prepared_matrix_probe_reports_zero_graph_index_rebuilds_after_prepare(
        self,
    ) -> None:
        prepared = prepare_south_star_mol_from_facts(
            tetrahedral_facts(),
            writer_surface=SouthStarWriterSurface(),
        )

        entry = _guarded_matrix_entry(
            fixture_name="tetrahedral",
            prepared=prepared,
            execution_mode=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
        )

        self.assertEqual(entry.row.probe.graph_index_rebuild_count, 0)

    def test_prepared_matrix_probe_reports_zero_online_traversal_graph_rebuilds_after_prepare(
        self,
    ) -> None:
        prepared = prepare_south_star_mol_from_facts(
            tetrahedral_facts(),
            writer_surface=SouthStarWriterSurface(),
        )

        entry = _guarded_matrix_entry(
            fixture_name="tetrahedral",
            prepared=prepared,
            execution_mode=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
        )

        self.assertEqual(entry.row.probe.online_traversal_graph_from_facts_count, 0)
        self.assertEqual(entry.row.probe.online_traversal_graph_from_index_count, 0)
        self.assertEqual(entry.row.probe.online_traversal_graph_view_rebuild_count, 0)
        self.assertEqual(entry.row.probe.online_vm_graph_view_rebuild_count, 0)
        self.assertEqual(entry.row.probe.prepare_from_facts_count, 0)
        self.assertEqual(entry.row.probe.prepare_from_rdkit_count, 0)

    def test_prepared_matrix_probe_reports_zero_root_domain_recomputes_after_prepare(
        self,
    ) -> None:
        prepared = prepare_south_star_mol_from_facts(
            tetrahedral_facts(),
            writer_surface=SouthStarWriterSurface(),
        )

        entry = _guarded_matrix_entry(
            fixture_name="tetrahedral",
            prepared=prepared,
            execution_mode=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
        )

        self.assertEqual(entry.row.probe.root_domain_recompute_count, 0)
        self.assertEqual(entry.row.probe.root_domain_from_metadata_count, 0)

    def test_prepared_matrix_probe_reports_zero_stereo_template_rebuilds_after_prepare(
        self,
    ) -> None:
        prepared = prepare_south_star_mol_from_facts(
            tetrahedral_facts(),
            writer_surface=SouthStarWriterSurface(),
        )

        entry = _guarded_matrix_entry(
            fixture_name="tetrahedral",
            prepared=prepared,
            execution_mode=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
        )

        self.assertEqual(entry.row.probe.stereo_template_rebuild_count, 0)

    def test_prepared_matrix_probe_reports_zero_facts_validation_after_prepare(
        self,
    ) -> None:
        prepared = prepare_south_star_mol_from_facts(
            tetrahedral_facts(),
            writer_surface=SouthStarWriterSurface(),
        )

        entry = _guarded_matrix_entry(
            fixture_name="tetrahedral",
            prepared=prepared,
            execution_mode=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
        )

        self.assertEqual(entry.row.probe.facts_validate_count, 0)

    def test_prepared_matrix_probe_reports_zero_policy_validation_after_prepare(
        self,
    ) -> None:
        prepared = prepare_south_star_mol_from_facts(
            tetrahedral_facts(),
            writer_surface=SouthStarWriterSurface(),
        )

        entry = _guarded_matrix_entry(
            fixture_name="tetrahedral",
            prepared=prepared,
            execution_mode=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
        )

        self.assertEqual(entry.row.probe.policy_validate_count, 0)

    def test_prepared_matrix_probe_is_owned_by_matrix_not_caller_supplied(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            tetrahedral_facts(),
            writer_surface=SouthStarWriterSurface(),
        )

        with self.assertRaises(TypeError):
            collect_prepared_enumeration_matrix_entry(
                fixture_name="tetrahedral",
                prepared=prepared,
                graph_rebuild_count_after_prepare=0,
            )

    def test_prepared_matrix_probe_detects_forced_graph_rebuild(self) -> None:
        with PreparedRuntimeProbe() as probe:
            graph_index_module.build_graph_index(tetrahedral_facts())

        result = probe.result()

        self.assertEqual(result.graph_index_rebuild_count, 1)
        self.assertGreaterEqual(result.facts_validate_count, 1)

    def test_residual_matrix_rows_have_no_rendered_suffix_payload(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            cyclopropane_facts(),
            writer_surface=SouthStarWriterSurface(),
        )

        entry = _guarded_matrix_entry(
            fixture_name="ring",
            prepared=prepared,
            execution_mode=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
        )

        self.assertEqual(entry.row.retained_render_payload_chars, 0)
        self.assertIsNotNone(entry.row.retained_render_cursor_count)
        self.assertEqual(
            entry.row.retained_render_cursor_count,
            entry.row.max_retained_continuations,
        )

    def test_residual_prefix_workload_uses_fewer_root_dfs_runs_than_prefix_replay(
        self,
    ) -> None:
        prepared = prepare_south_star_mol_from_facts(
            tetrahedral_facts(),
            writer_surface=SouthStarWriterSurface(),
        )

        prefix = collect_prepared_prefix_workload_stats(
            prepared=prepared,
            execution_mode=OnlineDecoderExecutionMode.PREFIX_REPLAY,
        )
        residual = collect_prepared_prefix_workload_stats(
            prepared=prepared,
            execution_mode=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
        )

        self.assertEqual(prefix.root_dfs_runs, prefix.frontier_queries)
        self.assertLess(residual.root_dfs_runs, prefix.root_dfs_runs)
        self.assertGreater(residual.resumed_snapshots, 0)


@dataclass(frozen=True, slots=True)
class _Fixture:
    name: str
    facts: MoleculeFacts
    runtime_options: SouthStarRuntimeOptions = SouthStarRuntimeOptions()


def _matrix_fixtures() -> tuple[_Fixture, ...]:
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
        _Fixture("disconnected-multi-component", _disconnected_two_bond_components_facts()),
        _Fixture(
            "disconnected-stereo",
            _disconnected_tetra_and_bond_facts(),
            SouthStarRuntimeOptions(rooted_at_atom=5),
        ),
        _Fixture("sparse-atom-id", _sparse_two_atom_facts()),
    )


def _guarded_matrix_entry(
    *,
    fixture_name: str,
    prepared,
    execution_mode: OnlineDecoderExecutionMode,
    runtime_options: SouthStarRuntimeOptions = SouthStarRuntimeOptions(),
) -> PreparedEnumerationMatrixEntry:
    with patch(
        "grimace._south_star1.prepared_runtime.build_graph_index",
        side_effect=AssertionError("prepared matrix rebuilt graph index"),
    ), patch(
        "grimace._south_star1.root_domains.component_root_domains_for_facts",
        side_effect=AssertionError("prepared matrix recomputed root domains"),
    ), patch(
        "grimace._south_star1.skeleton.component_root_domains_for_facts",
        side_effect=AssertionError("prepared matrix recomputed root domains"),
    ), patch(
        "grimace._south_star1.online_traversal.component_root_domains_for_facts",
        side_effect=AssertionError("prepared matrix recomputed root domains"),
    ), patch(
        "grimace._south_star1.online_search_vm.component_root_domains_for_facts",
        side_effect=AssertionError("prepared matrix recomputed root domains"),
    ), patch(
        "grimace._south_star1.online_search_vm.build_stereo_templates",
        side_effect=AssertionError("prepared matrix rebuilt stereo templates"),
    ), patch(
        "grimace._south_star1.online_stereo_witness.build_stereo_templates",
        side_effect=AssertionError("prepared matrix rebuilt stereo templates"),
    ):
        return collect_prepared_enumeration_matrix_entry(
            fixture_name=fixture_name,
            prepared=prepared,
            runtime_options=runtime_options,
            execution_mode=execution_mode,
        )


def _assert_entry_conforms(
    test: unittest.TestCase,
    entry: PreparedEnumerationMatrixEntry,
) -> None:
    row = entry.row
    test.assertEqual(entry.online_strings, entry.offline_strings)
    test.assertEqual(row.online_support_count, row.offline_support_count)
    test.assertEqual(row.online_witness_completion_count, row.offline_witness_count)
    test.assertEqual(row.probe.graph_index_rebuild_count, 0)
    test.assertEqual(row.probe.online_traversal_graph_from_facts_count, 0)
    test.assertEqual(row.probe.online_traversal_graph_from_index_count, 0)
    test.assertEqual(row.probe.prepare_from_facts_count, 0)
    test.assertEqual(row.probe.prepare_from_rdkit_count, 0)
    test.assertEqual(row.probe.root_domain_recompute_count, 0)
    test.assertEqual(row.probe.root_domain_from_metadata_count, 0)
    test.assertEqual(row.probe.stereo_template_rebuild_count, 0)
    test.assertEqual(row.probe.facts_validate_count, 0)
    test.assertEqual(row.probe.policy_validate_count, 0)
    test.assertEqual(row.probe.online_traversal_graph_view_rebuild_count, 0)
    test.assertEqual(row.probe.online_vm_graph_view_rebuild_count, 0)
    test.assertGreater(row.frontier_queries, 0)
    test.assertGreater(row.max_choice_count, 0)
    test.assertGreater(row.max_pending_stream_states, 0)
    if row.execution_mode is OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS:
        test.assertEqual(row.retained_render_payload_chars, 0)
        test.assertIsNotNone(row.max_retained_continuations)
        test.assertIsNotNone(row.retained_render_cursor_count)


def _disconnected_two_bond_components_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, "C"), atom(1, "O"), atom(2, "F"), atom(3, "Cl")),
        bonds=(single_bond(0, 0, 1), single_bond(1, 2, 3)),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1)),
                bonds=(BondId(0),),
            ),
            ComponentFacts(
                id=ComponentId(1),
                atoms=(AtomId(2), AtomId(3)),
                bonds=(BondId(1),),
            ),
        ),
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


def _disconnected_tetra_and_bond_facts() -> MoleculeFacts:
    base = tetrahedral_facts()
    return MoleculeFacts(
        atoms=base.atoms + (atom(4, "F"), atom(5, "Cl")),
        bonds=base.bonds + (single_bond(3, 4, 5),),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1), AtomId(2), AtomId(3)),
                bonds=(BondId(0), BondId(1), BondId(2)),
            ),
            ComponentFacts(
                id=ComponentId(1),
                atoms=(AtomId(4), AtomId(5)),
                bonds=(BondId(3),),
            ),
        ),
        stereo=base.stereo,
        ligand_occurrences=base.ligand_occurrences,
    )


if __name__ == "__main__":
    unittest.main()
