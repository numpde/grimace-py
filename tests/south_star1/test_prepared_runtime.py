"""Tests for the South Star prepared runtime boundary."""

from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.facts import ComponentFacts
from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import ComponentId
from grimace._south_star1.online_continuation import OnlineDecoderExecutionMode
from grimace._south_star1.online_decoder_api import make_determinized_online_decoder
from grimace._south_star1.online_decoder_api import SouthStarOnlineDecoder
from grimace._south_star1.online_decoder_api import online_decode_token_texts_for_policy
from grimace._south_star1.online_serialization_stream import collect_online_serializations
from grimace._south_star1.online_serialization_stream import iter_online_serializations
from grimace._south_star1.online_traversal import iter_online_traversal_traces
from grimace._south_star1.online_traversal import iter_prepared_online_traversal_traces
from grimace._south_star1.online_traversal_graph import OnlineTraversalGraph
from grimace._south_star1.online_traversal_graph import build_online_traversal_graph_from_index
from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.ordinary_semantics import OrdinarySmilesSemantics
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import component_root_domains_for_prepared
from grimace._south_star1.prepared_runtime import enumerate_prepared_stereo_support
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.prepared_runtime import runtime_root_atom_for_prepared
from grimace._south_star1.root_domains import component_root_domains_for_facts
from grimace._south_star1.skeleton import enumerate_traversal_skeletons
from grimace._south_star1.graph_index import build_graph_index
from grimace._south_star1.support_enumeration import enumerate_stereo_support
from tests.south_star1.helpers import atom
from tests.south_star1.helpers import cyclopropane_facts
from tests.south_star1.helpers import single_bond
from tests.south_star1.helpers import tetrahedral_facts


class PreparedRuntimeTest(unittest.TestCase):
    def test_prepared_mol_reuses_facts_policy_semantics(self) -> None:
        facts = tetrahedral_facts()
        policy = ordinary_policy_for_facts(facts)
        semantics = OrdinarySmilesSemantics()

        prepared = prepare_south_star_mol_from_facts(
            facts,
            writer_surface=SouthStarWriterSurface(),
            policy=policy,
            semantics=semantics,
        )

        self.assertIs(prepared.facts, facts)
        self.assertIs(prepared.policy, policy)
        self.assertIs(prepared.semantics, semantics)

    def test_prepared_stores_graph_index(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            tetrahedral_facts(),
            writer_surface=SouthStarWriterSurface(),
        )

        self.assertEqual(set(prepared.graph_index.atom_by_id), set(prepared.atom_ids))
        self.assertEqual(
            set(prepared.graph_index.bond_by_id),
            {bond.id for bond in prepared.facts.bonds},
        )

    def test_prepared_writer_surface_flags_are_baked(self) -> None:
        surface = SouthStarWriterSurface(ignore_atom_map_numbers=True)

        prepared = prepare_south_star_mol_from_facts(
            cyclopropane_facts(),
            writer_surface=surface,
        )

        self.assertEqual(prepared.writer_surface, surface)
        self.assertTrue(prepared.writer_surface.ignore_atom_map_numbers)

    def test_conflicting_writer_surface_flags_raise(self) -> None:
        with self.assertRaises(SouthStarError):
            prepare_south_star_mol_from_facts(
                tetrahedral_facts(),
                writer_surface=SouthStarWriterSurface(isomeric_smiles=False),
            )

    def test_prepared_root_minus_one_matches_all_root_union(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            tetrahedral_facts(),
            writer_surface=SouthStarWriterSurface(),
        )

        default_result = collect_online_serializations(prepared=prepared)
        explicit_all_result = collect_online_serializations(
            prepared=prepared,
            runtime_options=SouthStarRuntimeOptions(rooted_at_atom=-1),
        )

        self.assertEqual(set(explicit_all_result.strings), set(default_result.strings))
        self.assertEqual(
            explicit_all_result.witness_completion_count,
            default_result.witness_completion_count,
        )

    def test_prepared_negative_root_below_minus_one_matches_all_roots(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            tetrahedral_facts(),
            writer_surface=SouthStarWriterSurface(),
        )

        default_result = collect_online_serializations(prepared=prepared)
        negative_result = collect_online_serializations(
            prepared=prepared,
            runtime_options=SouthStarRuntimeOptions(rooted_at_atom=-7),
        )

        self.assertEqual(set(negative_result.strings), set(default_result.strings))
        self.assertEqual(
            negative_result.witness_completion_count,
            default_result.witness_completion_count,
        )

    def test_prepared_explicit_root_is_subset_of_all_root_support(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            tetrahedral_facts(),
            writer_surface=SouthStarWriterSurface(),
        )

        all_roots = collect_online_serializations(prepared=prepared)
        root_zero = collect_online_serializations(
            prepared=prepared,
            runtime_options=SouthStarRuntimeOptions(rooted_at_atom=0),
        )

        self.assertLess(root_zero.support_count, all_roots.support_count)
        self.assertLessEqual(set(root_zero.strings), set(all_roots.strings))

    def test_different_roots_can_have_different_support_counts(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            tetrahedral_facts(),
            writer_surface=SouthStarWriterSurface(),
        )

        root_zero = collect_online_serializations(
            prepared=prepared,
            runtime_options=SouthStarRuntimeOptions(rooted_at_atom=0),
        )
        root_one = collect_online_serializations(
            prepared=prepared,
            runtime_options=SouthStarRuntimeOptions(rooted_at_atom=1),
        )

        self.assertNotEqual(root_zero.support_count, root_one.support_count)

    def test_online_rooted_support_matches_offline_rooted_support(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            tetrahedral_facts(),
            writer_surface=SouthStarWriterSurface(),
        )
        options = SouthStarRuntimeOptions(rooted_at_atom=0)

        online = collect_online_serializations(
            prepared=prepared,
            runtime_options=options,
        )
        offline = enumerate_prepared_stereo_support(
            prepared=prepared,
            runtime_options=options,
        )

        self.assertEqual(set(online.strings), set(offline.strings))
        self.assertEqual(online.support_count, offline.distinct_count)
        self.assertEqual(online.witness_completion_count, offline.witness_count)

    def test_rooted_support_equivalent_across_execution_modes(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            tetrahedral_facts(),
            writer_surface=SouthStarWriterSurface(),
        )
        options = SouthStarRuntimeOptions(rooted_at_atom=0)
        offline = enumerate_prepared_stereo_support(
            prepared=prepared,
            runtime_options=options,
        )

        for mode in (
            OnlineDecoderExecutionMode.PREFIX_REPLAY,
            OnlineDecoderExecutionMode.CACHED_COMPLETIONS,
            OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
        ):
            with self.subTest(mode=mode.value):
                online = collect_online_serializations(
                    prepared=prepared,
                    runtime_options=options,
                    execution_mode=mode,
                )
                self.assertEqual(set(online.strings), set(offline.strings))
                self.assertEqual(online.support_count, offline.distinct_count)
                self.assertEqual(
                    online.witness_completion_count,
                    offline.witness_count,
                )

    def test_rooted_decoder_frontier_equivalent_across_execution_modes(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            tetrahedral_facts(),
            writer_surface=SouthStarWriterSurface(),
        )
        options = SouthStarRuntimeOptions(rooted_at_atom=0)
        frontiers: dict[OnlineDecoderExecutionMode, tuple[str, ...]] = {}
        for mode in (
            OnlineDecoderExecutionMode.PREFIX_REPLAY,
            OnlineDecoderExecutionMode.CACHED_COMPLETIONS,
            OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
        ):
            decoder = make_determinized_online_decoder(
                prepared=prepared,
                runtime_options=options,
                include_eos=True,
                execution_mode=mode,
            )
            frontiers[mode] = tuple(choice.text for choice in decoder.initial_state().choices())

        self.assertEqual(
            set(frontiers[OnlineDecoderExecutionMode.PREFIX_REPLAY]),
            set(frontiers[OnlineDecoderExecutionMode.CACHED_COMPLETIONS]),
        )
        self.assertEqual(
            set(frontiers[OnlineDecoderExecutionMode.PREFIX_REPLAY]),
            set(frontiers[OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS]),
        )

    def test_rooted_eos_completion_counts_equivalent_across_execution_modes(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            tetrahedral_facts(),
            writer_surface=SouthStarWriterSurface(),
        )
        options = SouthStarRuntimeOptions(rooted_at_atom=0)
        counts = tuple(
            collect_online_serializations(
                prepared=prepared,
                runtime_options=options,
                execution_mode=mode,
            ).witness_completion_count
            for mode in (
                OnlineDecoderExecutionMode.PREFIX_REPLAY,
                OnlineDecoderExecutionMode.CACHED_COMPLETIONS,
                OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
            )
        )

        self.assertEqual(len(set(counts)), 1)

    def test_disconnected_explicit_root_preserves_fragment_order(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            disconnected_two_bond_components_facts(),
            writer_surface=SouthStarWriterSurface(),
        )

        result = collect_online_serializations(
            prepared=prepared,
            runtime_options=SouthStarRuntimeOptions(rooted_at_atom=3),
        )

        self.assertTrue(result.strings)
        for rendered in result.strings:
            first, second = rendered.split(".")
            self.assertTrue(first.startswith(("C", "O")), rendered)
            self.assertTrue(second.startswith("Cl"), rendered)
            self.assertNotIn("F", first)

    def test_disconnected_explicit_root_restricts_only_target_component(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            disconnected_two_bond_components_facts(),
            writer_surface=SouthStarWriterSurface(),
        )

        result = collect_online_serializations(
            prepared=prepared,
            runtime_options=SouthStarRuntimeOptions(rooted_at_atom=3),
        )
        first_fragments = {rendered.split(".")[0] for rendered in result.strings}
        second_fragments = {rendered.split(".")[1] for rendered in result.strings}

        self.assertEqual({fragment[0] for fragment in first_fragments}, {"C", "O"})
        self.assertEqual({fragment[:2] for fragment in second_fragments}, {"Cl"})

    def test_disconnected_explicit_root_leaves_other_component_root_domains(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            disconnected_two_bond_components_facts(),
            writer_surface=SouthStarWriterSurface(),
        )

        root_three = collect_online_serializations(
            prepared=prepared,
            runtime_options=SouthStarRuntimeOptions(rooted_at_atom=3),
        )

        self.assertEqual(root_three.support_count, 8)
        self.assertEqual({text.split(".")[0][0] for text in root_three.strings}, {"C", "O"})

    def test_disconnected_all_roots_equals_union_of_explicit_roots(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            disconnected_two_bond_components_facts(),
            writer_surface=SouthStarWriterSurface(),
        )

        all_roots = collect_online_serializations(prepared=prepared)
        union: set[str] = set()
        for atom_id in prepared.atom_ids:
            rooted = collect_online_serializations(
                prepared=prepared,
                runtime_options=SouthStarRuntimeOptions(rooted_at_atom=int(atom_id)),
            )
            union.update(rooted.strings)

        self.assertEqual(set(all_roots.strings), union)

    def test_disconnected_rooted_support_equivalent_across_modes_and_offline(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            disconnected_two_bond_components_facts(),
            writer_surface=SouthStarWriterSurface(),
        )
        options = SouthStarRuntimeOptions(rooted_at_atom=3)
        offline = enumerate_prepared_stereo_support(
            prepared=prepared,
            runtime_options=options,
        )

        for mode in (
            OnlineDecoderExecutionMode.PREFIX_REPLAY,
            OnlineDecoderExecutionMode.CACHED_COMPLETIONS,
            OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
        ):
            with self.subTest(mode=mode.value):
                online = collect_online_serializations(
                    prepared=prepared,
                    runtime_options=options,
                    execution_mode=mode,
                )
                self.assertEqual(set(online.strings), set(offline.strings))
                self.assertEqual(online.support_count, offline.distinct_count)
                self.assertEqual(
                    online.witness_completion_count,
                    offline.witness_count,
                )

    def test_rooted_runtime_option_does_not_change_prepared_identity(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            tetrahedral_facts(),
            writer_surface=SouthStarWriterSurface(),
        )

        collect_online_serializations(
            prepared=prepared,
            runtime_options=SouthStarRuntimeOptions(rooted_at_atom=0),
        )

        self.assertEqual(prepared.atom_count, len(prepared.facts.atoms))
        self.assertEqual(prepared.component_count, len(prepared.facts.components))
        self.assertEqual(prepared.atom_ids, tuple(atom.id for atom in prepared.facts.atoms))
        self.assertEqual(
            prepared.component_atom_ids,
            tuple(component.atoms for component in prepared.facts.components),
        )

    def test_prepared_atom_ids_do_not_assume_dense_ids(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            sparse_two_atom_facts(),
            writer_surface=SouthStarWriterSurface(),
        )

        self.assertEqual(prepared.atom_ids, (AtomId(10), AtomId(20)))
        self.assertNotEqual(prepared.atom_ids, tuple(AtomId(i) for i in range(prepared.atom_count)))

    def test_sparse_atom_ids_explicit_root_works(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            sparse_two_atom_facts(),
            writer_surface=SouthStarWriterSurface(),
        )

        rooted = collect_online_serializations(
            prepared=prepared,
            runtime_options=SouthStarRuntimeOptions(rooted_at_atom=20),
        )

        self.assertTrue(rooted.strings)
        self.assertTrue(all(text.startswith("O") for text in rooted.strings))

    def test_sparse_atom_ids_all_roots_uses_actual_atom_ids(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            sparse_two_atom_facts(),
            writer_surface=SouthStarWriterSurface(),
        )

        all_roots = collect_online_serializations(prepared=prepared)
        union: set[str] = set()
        for atom_id in prepared.atom_ids:
            rooted = collect_online_serializations(
                prepared=prepared,
                runtime_options=SouthStarRuntimeOptions(rooted_at_atom=int(atom_id)),
            )
            union.update(rooted.strings)

        self.assertEqual(set(all_roots.strings), union)

    def test_connected_all_root_witness_count_equals_sum_over_explicit_roots(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            tetrahedral_facts(),
            writer_surface=SouthStarWriterSurface(),
        )

        all_roots = collect_online_serializations(prepared=prepared)
        witness_count = 0
        support_union: set[str] = set()
        for atom_id in prepared.atom_ids:
            rooted = collect_online_serializations(
                prepared=prepared,
                runtime_options=SouthStarRuntimeOptions(rooted_at_atom=int(atom_id)),
            )
            witness_count += rooted.witness_completion_count
            support_union.update(rooted.strings)

        self.assertEqual(all_roots.witness_completion_count, witness_count)
        self.assertEqual(set(all_roots.strings), support_union)

    def test_disconnected_all_root_witness_count_equals_sum_over_roots_in_each_component(
        self,
    ) -> None:
        prepared = prepare_south_star_mol_from_facts(
            disconnected_two_bond_components_facts(),
            writer_surface=SouthStarWriterSurface(),
        )
        all_roots = collect_online_serializations(prepared=prepared)

        for component_atoms in prepared.component_atom_ids:
            witness_count = 0
            support_union: set[str] = set()
            for atom_id in component_atoms:
                rooted = collect_online_serializations(
                    prepared=prepared,
                    runtime_options=SouthStarRuntimeOptions(rooted_at_atom=int(atom_id)),
                )
                witness_count += rooted.witness_completion_count
                support_union.update(rooted.strings)
            with self.subTest(component=tuple(int(atom) for atom in component_atoms)):
                self.assertEqual(all_roots.witness_completion_count, witness_count)
                self.assertEqual(set(all_roots.strings), support_union)

    def test_shared_component_root_domain_helper_used_by_online_and_offline(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            disconnected_two_bond_components_facts(),
            writer_surface=SouthStarWriterSurface(),
        )
        expected = tuple(
            atoms
            for _, atoms in component_root_domains_for_facts(
                prepared.facts,
                AtomId(3),
            )
        )
        online_roots = {
            trace.roots
            for trace in iter_online_traversal_traces(
                facts=prepared.facts,
                policy=prepared.policy,
                rooted_at_atom=AtomId(3),
            )
        }
        offline_roots = {
            skeleton.roots
            for skeleton in enumerate_traversal_skeletons(
                facts=prepared.facts,
                index=build_graph_index(prepared.facts),
                policy=prepared.policy,
                rooted_at_atom=AtomId(3),
            )
        }

        self.assertEqual(expected, ((AtomId(0), AtomId(1)), (AtomId(3),)))
        self.assertEqual(online_roots, {(AtomId(0), AtomId(3)), (AtomId(1), AtomId(3))})
        self.assertEqual(offline_roots, online_roots)

    def test_prepared_root_domain_cache_matches_facts_helper(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            disconnected_two_bond_components_facts(),
            writer_surface=SouthStarWriterSurface(),
        )

        for root in (None, AtomId(0), AtomId(3)):
            with self.subTest(root=root):
                self.assertEqual(
                    component_root_domains_for_prepared(
                        prepared=prepared,
                        rooted_at_atom=root,
                    ),
                    component_root_domains_for_facts(prepared.facts, root),
                )

    def test_prepared_stores_online_traversal_graph(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            tetrahedral_facts(),
            writer_surface=SouthStarWriterSurface(),
        )

        self.assertIsInstance(prepared.online_traversal_graph, OnlineTraversalGraph)
        self.assertEqual(prepared.online_traversal_graph.atoms, prepared.atom_ids)
        self.assertEqual(
            prepared.online_traversal_graph.components,
            tuple(
                (component.atoms, component.bonds)
                for component in prepared.facts.components
            ),
        )

    def test_prepared_traversal_graph_matches_raw_graph_view(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            disconnected_two_bond_components_facts(),
            writer_surface=SouthStarWriterSurface(),
        )
        expected = build_online_traversal_graph_from_index(
            prepared.facts,
            prepared.graph_index,
        )

        self.assertEqual(prepared.online_traversal_graph, expected)

    def test_prepared_online_traversal_does_not_call_graph_from_facts(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            tetrahedral_facts(),
            writer_surface=SouthStarWriterSurface(),
        )
        root_domains = tuple(atoms for _, atoms in prepared.all_root_domains)

        with patch(
            "grimace._south_star1.online_traversal._graph_from_facts",
            side_effect=AssertionError("prepared traversal rebuilt graph view"),
        ):
            traces = tuple(
                iter_prepared_online_traversal_traces(
                    prepared=prepared,
                    rooted_at_atom=None,
                    component_root_domains=root_domains,
                )
            )

        self.assertTrue(traces)

    def test_prepared_explicit_root_domains_match_all_root_metadata(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            disconnected_two_bond_components_facts(),
            writer_surface=SouthStarWriterSurface(),
        )

        for atom_id, component_id in prepared.atom_component:
            domains = component_root_domains_for_prepared(
                prepared=prepared,
                rooted_at_atom=atom_id,
            )
            for current_component_id, atoms in domains:
                if current_component_id == component_id:
                    self.assertEqual(atoms, (atom_id,))
                else:
                    self.assertEqual(
                        atoms,
                        dict(prepared.all_root_domains)[current_component_id],
                    )

    def test_disconnected_stereo_rooted_support_matches_offline_and_all_modes(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            disconnected_tetra_and_bond_facts(),
            writer_surface=SouthStarWriterSurface(),
        )
        options = SouthStarRuntimeOptions(rooted_at_atom=5)
        offline = enumerate_prepared_stereo_support(
            prepared=prepared,
            runtime_options=options,
        )

        for mode in (
            OnlineDecoderExecutionMode.PREFIX_REPLAY,
            OnlineDecoderExecutionMode.CACHED_COMPLETIONS,
            OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
        ):
            with self.subTest(mode=mode.value):
                online = collect_online_serializations(
                    prepared=prepared,
                    runtime_options=options,
                    execution_mode=mode,
                )
                self.assertEqual(set(online.strings), set(offline.strings))
                self.assertEqual(online.support_count, offline.distinct_count)
                self.assertEqual(
                    online.witness_completion_count,
                    offline.witness_count,
                )

    def test_invalid_root_rejected(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            tetrahedral_facts(),
            writer_surface=SouthStarWriterSurface(),
        )

        with self.assertRaises(SouthStarError):
            collect_online_serializations(
                prepared=prepared,
                runtime_options=SouthStarRuntimeOptions(rooted_at_atom=99),
            )

    def test_unsupported_canonical_runtime_option_rejected(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            tetrahedral_facts(),
            writer_surface=SouthStarWriterSurface(),
        )

        with self.assertRaises(SouthStarError):
            collect_online_serializations(
                prepared=prepared,
                runtime_options=SouthStarRuntimeOptions(canonical=True),
            )

    def test_decoder_normalizes_root_once_at_construction(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            tetrahedral_facts(),
            writer_surface=SouthStarWriterSurface(),
        )
        decoder = make_determinized_online_decoder(
            prepared=prepared,
            runtime_options=SouthStarRuntimeOptions(rooted_at_atom=0),
            execution_mode=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
            include_eos=True,
        )

        with patch(
            "grimace._south_star1.online_decoder_api.runtime_root_atom",
            side_effect=AssertionError("root normalized during choice query"),
        ):
            choices = decoder.initial_state().choices()

        self.assertTrue(choices)

    def test_decoder_state_from_root_zero_rejected_by_root_one_decoder(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            tetrahedral_facts(),
            writer_surface=SouthStarWriterSurface(),
        )
        root_zero = make_determinized_online_decoder(
            prepared=prepared,
            runtime_options=SouthStarRuntimeOptions(rooted_at_atom=0),
        )
        root_one = make_determinized_online_decoder(
            prepared=prepared,
            runtime_options=SouthStarRuntimeOptions(rooted_at_atom=1),
        )

        with self.assertRaises(ValueError):
            root_one.choices(root_zero.initial_state())

    def test_decoder_rejects_inconsistent_manual_root_construction(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            tetrahedral_facts(),
            writer_surface=SouthStarWriterSurface(),
        )

        with self.assertRaises(ValueError):
            SouthStarOnlineDecoder(
                facts=prepared.facts,
                policy=prepared.policy,
                semantics=prepared.semantics,
                rooted_at_atom=None,
                runtime_options=SouthStarRuntimeOptions(rooted_at_atom=0),
            )

    def test_prepared_online_stream_matches_facts_level_stream(self) -> None:
        facts = tetrahedral_facts()
        prepared = prepare_south_star_mol_from_facts(
            facts,
            writer_surface=SouthStarWriterSurface(),
        )

        prepared_result = collect_online_serializations(prepared=prepared)
        facts_result = collect_online_serializations(
            facts=facts,
            policy=prepared.policy,
            semantics=prepared.semantics,
        )

        self.assertEqual(set(prepared_result.strings), set(facts_result.strings))
        self.assertEqual(
            prepared_result.witness_completion_count,
            facts_result.witness_completion_count,
        )

    def test_prepared_offline_support_matches_facts_level_support(self) -> None:
        facts = cyclopropane_facts()
        prepared = prepare_south_star_mol_from_facts(
            facts,
            writer_surface=SouthStarWriterSurface(),
        )

        prepared_support = enumerate_stereo_support(
            facts=prepared.facts,
            policy=prepared.policy,
            semantics=prepared.semantics,
        )
        facts_support = enumerate_stereo_support(
            facts=facts,
            policy=ordinary_policy_for_facts(facts),
            semantics=OrdinarySmilesSemantics(),
        )

        self.assertEqual(set(prepared_support.strings), set(facts_support.strings))
        self.assertEqual(prepared_support.distinct_count, facts_support.distinct_count)
        self.assertEqual(prepared_support.witness_count, facts_support.witness_count)

    def test_prepared_templates_are_not_recomputed_per_decoder_query(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            tetrahedral_facts(),
            writer_surface=SouthStarWriterSurface(),
        )

        with patch(
            "grimace._south_star1.online_search_vm.build_stereo_templates",
            side_effect=AssertionError("query recomputed stereo templates"),
        ), patch(
            "grimace._south_star1.online_stereo_witness.build_stereo_templates",
            side_effect=AssertionError("query recomputed stereo templates"),
        ):
            result = collect_online_serializations(prepared=prepared)

        self.assertEqual(result.support_count, len(result.strings))

    def test_prepared_offline_support_does_not_rebuild_graph_index(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            cyclopropane_facts(),
            writer_surface=SouthStarWriterSurface(),
        )

        with patch(
            "grimace._south_star1.prepared_runtime.build_graph_index",
            side_effect=AssertionError("prepared offline support rebuilt graph index"),
        ):
            result = enumerate_prepared_stereo_support(prepared=prepared)

        self.assertGreater(result.distinct_count, 0)

    def test_prepared_online_decoder_does_not_rebuild_graph_index_per_choice_query(
        self,
    ) -> None:
        prepared = prepare_south_star_mol_from_facts(
            tetrahedral_facts(),
            writer_surface=SouthStarWriterSurface(),
        )
        decoder = make_determinized_online_decoder(
            prepared=prepared,
            include_eos=True,
            execution_mode=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
        )

        self.assertIs(decoder.graph_index, prepared.graph_index)
        with patch(
            "grimace._south_star1.prepared_runtime.build_graph_index",
            side_effect=AssertionError("prepared online query rebuilt graph index"),
        ), patch(
            "grimace._south_star1.graph_index.build_graph_index",
            side_effect=AssertionError("prepared online query rebuilt graph index"),
        ):
            choices = decoder.initial_state().choices()

        self.assertTrue(choices)

    def test_prepared_decoder_choices_does_not_rebuild_online_traversal_graph(
        self,
    ) -> None:
        prepared = prepare_south_star_mol_from_facts(
            tetrahedral_facts(),
            writer_surface=SouthStarWriterSurface(),
        )

        for mode in (
            OnlineDecoderExecutionMode.PREFIX_REPLAY,
            OnlineDecoderExecutionMode.CACHED_COMPLETIONS,
            OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
        ):
            decoder = make_determinized_online_decoder(
                prepared=prepared,
                include_eos=True,
                execution_mode=mode,
            )
            with self.subTest(mode=mode.value), patch(
                "grimace._south_star1.online_traversal._graph_from_facts",
                side_effect=AssertionError("prepared decoder rebuilt graph view"),
            ), patch(
                "grimace._south_star1.online_search_vm._graph_from_facts",
                side_effect=AssertionError("prepared decoder rebuilt graph view"),
            ):
                choices = decoder.initial_state().choices()
            self.assertTrue(choices)

    def test_prepared_serialization_stream_does_not_rebuild_online_traversal_graph(
        self,
    ) -> None:
        prepared = prepare_south_star_mol_from_facts(
            tetrahedral_facts(),
            writer_surface=SouthStarWriterSurface(),
        )

        with patch(
            "grimace._south_star1.online_traversal._graph_from_facts",
            side_effect=AssertionError("prepared stream rebuilt graph view"),
        ), patch(
            "grimace._south_star1.online_search_vm._graph_from_facts",
            side_effect=AssertionError("prepared stream rebuilt graph view"),
        ):
            emitted = tuple(iter_online_serializations(prepared=prepared))

        self.assertTrue(emitted)

    def test_raw_online_traversal_still_builds_traversal_graph(self) -> None:
        facts = tetrahedral_facts()
        policy = ordinary_policy_for_facts(facts)

        with patch(
            "grimace._south_star1.online_traversal._graph_from_facts",
            side_effect=AssertionError("raw traversal built graph view"),
        ):
            with self.assertRaisesRegex(AssertionError, "built graph view"):
                next(iter_online_traversal_traces(facts=facts, policy=policy))

    def test_prepared_online_decoder_does_not_validate_facts_for_root_domains_per_choice_query(
        self,
    ) -> None:
        prepared = prepare_south_star_mol_from_facts(
            tetrahedral_facts(),
            writer_surface=SouthStarWriterSurface(),
        )

        for mode in (
            OnlineDecoderExecutionMode.PREFIX_REPLAY,
            OnlineDecoderExecutionMode.CACHED_COMPLETIONS,
            OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
        ):
            decoder = make_determinized_online_decoder(
                prepared=prepared,
                include_eos=True,
                execution_mode=mode,
                runtime_options=SouthStarRuntimeOptions(rooted_at_atom=0),
            )
            with self.subTest(mode=mode.value), patch(
                "grimace._south_star1.online_traversal.component_root_domains_for_facts",
                side_effect=AssertionError("prepared online query recomputed root domains"),
            ), patch(
                "grimace._south_star1.online_search_vm.component_root_domains_for_facts",
                side_effect=AssertionError("prepared online query recomputed root domains"),
            ):
                choices = decoder.initial_state().choices()
            self.assertTrue(choices)

    def test_prepared_decoder_choices_does_not_validate_facts(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            tetrahedral_facts(),
            writer_surface=SouthStarWriterSurface(),
        )
        decoder = make_determinized_online_decoder(
            prepared=prepared,
            include_eos=True,
            execution_mode=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
        )

        with patch.object(
            type(prepared.facts),
            "validate",
            side_effect=AssertionError("prepared decoder choices validated facts"),
        ):
            choices = decoder.initial_state().choices()

        self.assertTrue(choices)

    def test_prepared_decoder_choices_does_not_validate_policy(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            tetrahedral_facts(),
            writer_surface=SouthStarWriterSurface(),
        )
        decoder = make_determinized_online_decoder(
            prepared=prepared,
            include_eos=True,
            execution_mode=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
        )

        with patch.object(
            type(prepared.policy),
            "validate_for_facts",
            side_effect=AssertionError("prepared decoder choices validated policy"),
        ):
            choices = decoder.initial_state().choices()

        self.assertTrue(choices)

    def test_prepared_online_stream_does_not_validate_facts(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            tetrahedral_facts(),
            writer_surface=SouthStarWriterSurface(),
        )

        with patch.object(
            type(prepared.facts),
            "validate",
            side_effect=AssertionError("prepared online stream validated facts"),
        ):
            emitted = tuple(iter_online_serializations(prepared=prepared))

        self.assertTrue(emitted)

    def test_prepared_online_stream_does_not_validate_policy(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            tetrahedral_facts(),
            writer_surface=SouthStarWriterSurface(),
        )

        with patch.object(
            type(prepared.policy),
            "validate_for_facts",
            side_effect=AssertionError("prepared online stream validated policy"),
        ):
            emitted = tuple(iter_online_serializations(prepared=prepared))

        self.assertTrue(emitted)

    def test_prepared_runtime_root_normalization_does_not_scan_facts_atoms(self) -> None:
        class PoisonAtoms(tuple):
            def __iter__(self):
                raise AssertionError("prepared root normalization scanned facts atoms")

        prepared = prepare_south_star_mol_from_facts(
            tetrahedral_facts(),
            writer_surface=SouthStarWriterSurface(),
        )
        poisoned_facts = replace(prepared.facts, atoms=PoisonAtoms(prepared.facts.atoms))
        poisoned_prepared = replace(prepared, facts=poisoned_facts)

        root = runtime_root_atom_for_prepared(
            SouthStarRuntimeOptions(rooted_at_atom=0),
            prepared=poisoned_prepared,
        )

        self.assertEqual(root, AtomId(0))

    def test_raw_online_traversal_still_validates_facts(self) -> None:
        facts = tetrahedral_facts()
        policy = ordinary_policy_for_facts(facts)

        with patch.object(
            type(facts),
            "validate",
            side_effect=AssertionError("raw online traversal validated facts"),
        ):
            with self.assertRaisesRegex(AssertionError, "validated facts"):
                next(iter_online_traversal_traces(facts=facts, policy=policy))

    def test_raw_online_traversal_still_validates_policy(self) -> None:
        facts = tetrahedral_facts()
        policy = ordinary_policy_for_facts(facts)

        with patch.object(
            type(policy),
            "validate_for_facts",
            side_effect=AssertionError("raw online traversal validated policy"),
        ):
            with self.assertRaisesRegex(AssertionError, "validated policy"):
                next(iter_online_traversal_traces(facts=facts, policy=policy))

    def test_prepared_token_inventory_superset_contains_exact_online_inventory(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            tetrahedral_facts(),
            writer_surface=SouthStarWriterSurface(),
        )
        exact_inventory = online_decode_token_texts_for_policy(
            facts=prepared.facts,
            policy=prepared.policy,
        )

        self.assertGreater(len(prepared.token_inventory_superset), 0)
        self.assertLessEqual(set(exact_inventory), set(prepared.token_inventory_superset))


def disconnected_two_bond_components_facts() -> MoleculeFacts:
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


def sparse_two_atom_facts() -> MoleculeFacts:
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


def disconnected_tetra_and_bond_facts() -> MoleculeFacts:
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
