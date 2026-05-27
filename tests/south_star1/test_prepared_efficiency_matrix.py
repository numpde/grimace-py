"""Prepared South Star conformance and structural-efficiency matrix tests."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from dataclasses import replace
from unittest.mock import patch

import grimace._south_star1.graph_index as graph_index_module
from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.facts import ComponentFacts
from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import ComponentId
from grimace._south_star1.online_continuation import OnlineDecoderExecutionMode
from grimace._south_star1.online_decoder_api import make_determinized_online_decoder
from grimace._south_star1.online_search_vm import residual_snapshot_frame_audit
from grimace._south_star1.online_search_vm import validate_residual_frame_stack
from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.prepared_bench_matrix import PreparedEnumerationMatrixEntry
from grimace._south_star1.prepared_bench_matrix import PreparedRuntimeProbe
from grimace._south_star1.prepared_bench_matrix import collect_prepared_enumeration_matrix_entry
from grimace._south_star1.prepared_bench_matrix import collect_prepared_prefix_workload_stats
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_smiles
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

    def test_prepared_matrix_rejects_writer_shaped_runtime_before_old_vm(
        self,
    ) -> None:
        facts = ordinary_molecule_facts_from_smiles("CCO")
        prepared = prepare_south_star_mol_from_facts(
            facts,
            writer_surface=SouthStarWriterSurface(),
        )

        with self.assertRaises(SouthStarError):
            _guarded_matrix_entry(
                fixture_name="writer-shaped-chain",
                prepared=prepared,
                runtime_options=SouthStarRuntimeOptions(
                    serialization_language=SerializationLanguageMode.WRITER_SHAPED,
                ),
                execution_mode=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
            )

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

    def test_prepared_matrix_all_residual_snapshots_pass_frame_stack_audit(
        self,
    ) -> None:
        for fixture in _matrix_fixtures():
            prepared = prepare_south_star_mol_from_facts(
                fixture.facts,
                writer_surface=SouthStarWriterSurface(),
            )

            with self.subTest(fixture=fixture.name):
                _assert_prepared_residual_snapshots_pass_frame_stack_audit(
                    self,
                    prepared=prepared,
                    runtime_options=fixture.runtime_options,
                )

    def test_prepared_matrix_observes_prefix_scheduler_frames_on_prefix_branching_fixture(
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

        _assert_matrix_prefix_scheduler_evidence(self, entry)

    def test_prepared_matrix_observes_direction_scheduler_frames_on_directional_fixture(
        self,
    ) -> None:
        prepared = prepare_south_star_mol_from_facts(
            directional_facts(),
            writer_surface=SouthStarWriterSurface(),
        )

        entry = _guarded_matrix_entry(
            fixture_name="directional",
            prepared=prepared,
            execution_mode=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
        )

        _assert_matrix_direction_scheduler_evidence(self, entry)

    def test_prepared_matrix_observes_support_maximal_scheduler_frames(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            directional_facts(),
            writer_surface=SouthStarWriterSurface(),
        )

        entry = _guarded_matrix_entry(
            fixture_name="support-maximal",
            prepared=prepared,
            execution_mode=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
        )

        _assert_matrix_support_maximal_scheduler_evidence(self, entry)

    def test_prepared_matrix_prefix_scheduler_evidence_fails_if_count_zero(
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
        tampered = replace(
            entry,
            row=replace(
                entry.row,
                retained_scheduler_frame_count=0,
                retained_prefix_enumeration_frame_count=0,
            ),
        )

        with self.assertRaises(AssertionError):
            _assert_matrix_prefix_scheduler_evidence(self, tampered)

    def test_support_maximal_scheduler_evidence_fails_if_frame_count_zero(
        self,
    ) -> None:
        prepared = prepare_south_star_mol_from_facts(
            directional_facts(),
            writer_surface=SouthStarWriterSurface(),
        )
        entry = _guarded_matrix_entry(
            fixture_name="support-maximal",
            prepared=prepared,
            execution_mode=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
        )
        tampered = replace(
            entry,
            row=replace(
                entry.row,
                retained_scheduler_frame_count=0,
                retained_support_maximal_frame_count=0,
            ),
        )

        with self.assertRaises(AssertionError):
            _assert_matrix_support_maximal_scheduler_evidence(self, tampered)

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
        "grimace._south_star1.exhaustive_online_traversal.component_root_domains_for_facts",
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
        test.assertIsNotNone(row.retained_scheduler_frame_count)
        test.assertIsNotNone(row.retained_prefix_enumeration_frame_count)
        test.assertIsNotNone(row.max_retained_prefix_domain_count)
        test.assertIsNotNone(row.total_retained_prefix_domain_count)
        test.assertIsNotNone(row.max_retained_prefix_assignment_count)
        test.assertIsNotNone(row.total_retained_prefix_assignment_count)
        test.assertIsNotNone(row.retained_direction_enumeration_frame_count)
        test.assertIsNotNone(row.max_retained_direction_carrier_count)
        test.assertIsNotNone(row.total_retained_direction_carrier_count)
        test.assertIsNotNone(row.max_retained_direction_assignment_count)
        test.assertIsNotNone(row.total_retained_direction_assignment_count)
        test.assertIsNotNone(row.retained_support_maximal_frame_count)
        test.assertIsNotNone(row.max_retained_support_maximal_candidate_count)
        test.assertIsNotNone(row.total_retained_support_maximal_candidate_count)
        test.assertIsNotNone(row.max_retained_support_maximal_selected_count)
        test.assertIsNotNone(row.total_retained_support_maximal_selected_count)
        test.assertIsNotNone(row.max_retained_support_maximal_remaining_count)
        test.assertIsNotNone(row.total_retained_support_maximal_remaining_count)


def _assert_matrix_prefix_scheduler_evidence(
    test: unittest.TestCase,
    entry: PreparedEnumerationMatrixEntry,
) -> None:
    row = entry.row
    test.assertEqual(row.execution_mode, OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS)
    test.assertGreater(row.retained_scheduler_frame_count or 0, 0)
    test.assertGreater(row.retained_prefix_enumeration_frame_count or 0, 0)
    test.assertGreater(row.max_retained_prefix_domain_count or 0, 0)
    test.assertGreater(row.total_retained_prefix_domain_count or 0, 0)
    test.assertGreater(row.max_retained_prefix_assignment_count or 0, 0)
    test.assertGreater(row.total_retained_prefix_assignment_count or 0, 0)


def _assert_matrix_direction_scheduler_evidence(
    test: unittest.TestCase,
    entry: PreparedEnumerationMatrixEntry,
) -> None:
    row = entry.row
    test.assertEqual(row.execution_mode, OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS)
    test.assertGreater(row.retained_direction_enumeration_frame_count or 0, 0)
    test.assertGreater(row.max_retained_direction_carrier_count or 0, 0)
    test.assertGreater(row.total_retained_direction_carrier_count or 0, 0)
    test.assertGreater(row.max_retained_direction_assignment_count or 0, 0)
    test.assertGreater(row.total_retained_direction_assignment_count or 0, 0)


def _assert_matrix_support_maximal_scheduler_evidence(
    test: unittest.TestCase,
    entry: PreparedEnumerationMatrixEntry,
) -> None:
    row = entry.row
    test.assertEqual(row.execution_mode, OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS)
    test.assertGreater(row.retained_support_maximal_frame_count or 0, 0)
    test.assertGreater(row.max_retained_support_maximal_candidate_count or 0, 0)
    test.assertGreater(row.total_retained_support_maximal_candidate_count or 0, 0)
    test.assertGreater(row.max_retained_support_maximal_selected_count or 0, 0)
    test.assertGreater(row.total_retained_support_maximal_selected_count or 0, 0)
    test.assertGreater(row.max_retained_support_maximal_remaining_count or 0, 0)
    test.assertGreater(row.total_retained_support_maximal_remaining_count or 0, 0)


def _assert_prepared_residual_snapshots_pass_frame_stack_audit(
    test: unittest.TestCase,
    *,
    prepared,
    runtime_options: SouthStarRuntimeOptions,
) -> None:
    decoder = make_determinized_online_decoder(
        prepared=prepared,
        include_eos=True,
        runtime_options=runtime_options,
        execution_mode=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
    )
    stack = [decoder.initial_state()]
    seen: set[str] = set()
    audited = 0
    while stack:
        state = stack.pop()
        if state.prefix in seen:
            continue
        seen.add(state.prefix)
        result = state.choices_with_stats()
        for choice in result.choices:
            if choice.is_eos or choice.next_state is None:
                continue
            frontier = choice.next_state.raw_state.frontier
            test.assertIsNotNone(frontier)
            assert frontier is not None
            for continuation in frontier.continuations:
                audit = residual_snapshot_frame_audit(continuation.snapshot)
                test.assertGreater(audit.resumable_frame_count, 0)
                test.assertEqual(audit.unknown_frame_count, 0)
                validate_residual_frame_stack(continuation.snapshot.frame_stack)
                audited += 1
            stack.append(choice.next_state)
    test.assertGreater(audited, 0)


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
