from __future__ import annotations

import unittest

from grimace._south_star.constraint_vocabulary import SouthStarRendererInput
from grimace._south_star.component_support_state import (
    SouthStarComponentSupportState,
)
from grimace._south_star.enum_s import mol_to_smiles_enum_s_graph_native_for_case
from grimace._south_star.enum_s import mol_to_smiles_enum_s_graph_native
from grimace._south_star.enum_s import (
    mol_to_smiles_enum_s_tree_traversals_for_case,
)
from grimace._south_star.enum_s import render_south_star_traversal
from grimace._south_star.enum_s import render_south_star_tree_traversal
from grimace._south_star.reference_model import SouthStarMarkerSlot
from grimace._south_star.reference_model import SouthStarMarkerSlotAssignment
from grimace._south_star.reference_model import SouthStarRingClosure
from grimace._south_star.reference_model import SouthStarTraversalEvent
from tests.helpers.south_star_enum_s import mol_to_smiles_enum_s_prototype_for_case
from tests.helpers.south_star_exact_support import (
    load_south_star_expanded_support_cases,
)
from tests.helpers.south_star_semantics import SouthStarAnnotationPolicyExpectation
from tests.helpers.south_star_semantics import SouthStarSemanticCase
from tests.helpers.south_star_semantics import load_south_star_semantic_cases
from tests.helpers.south_star_semantic_oracle import graph_signature
from tests.helpers.south_star_semantic_oracle import semantic_signature


def _expanded_support_case(case_id: str):
    return next(
        case
        for case in load_south_star_expanded_support_cases()
        if case.case_id == case_id
    )


class SouthStarEnumSPrototypeTests(unittest.TestCase):
    def test_prototype_returns_fixture_positive_semantic_outputs(self) -> None:
        for case in load_south_star_semantic_cases():
            result = mol_to_smiles_enum_s_prototype_for_case(case)

            with self.subTest(case_id=case.case_id):
                self.assertEqual(case.case_id, result.case_id)
                self.assertEqual(case.positive_semantic_smiles, result.outputs)
                self.assertEqual(
                    "south_star_semantic_fixture_witnesses",
                    result.generation_basis,
                )

    def test_prototype_excludes_negative_semantic_witnesses(self) -> None:
        for case in load_south_star_semantic_cases():
            result = mol_to_smiles_enum_s_prototype_for_case(case)
            negative_outputs = {
                negative.smiles for negative in case.negative_semantic_smiles
            }

            with self.subTest(case_id=case.case_id):
                self.assertFalse(negative_outputs.intersection(result.outputs))

    def test_prototype_exposes_support_state_complexity_snapshot(self) -> None:
        for case in load_south_star_semantic_cases():
            result = mol_to_smiles_enum_s_prototype_for_case(case)
            expected = SouthStarComponentSupportState.from_case(
                case
            ).complexity_snapshot()

            with self.subTest(case_id=case.case_id):
                self.assertEqual(expected, result.complexity_snapshot)

    def test_graph_native_tree_traversal_includes_fixture_witnesses(self) -> None:
        for case in load_south_star_semantic_cases():
            result = mol_to_smiles_enum_s_graph_native_for_case(case)

            with self.subTest(case_id=case.case_id):
                self.assertEqual(case.case_id, result.case_id)
                self.assertTrue(
                    set(case.positive_semantic_smiles).issubset(result.outputs),
                    set(case.positive_semantic_smiles) - set(result.outputs),
                )
                self.assertEqual(
                    "south_star_graph_native_equation_solved_tree_traversal",
                    result.generation_basis,
                )

    def test_graph_native_result_names_default_policies(self) -> None:
        case = next(
            case
            for case in load_south_star_semantic_cases()
            if case.case_id == "isolated_alkene_z"
        )
        result = mol_to_smiles_enum_s_graph_native_for_case(case)

        self.assertEqual("maximal_eligible_carrier", result.annotation_policy)
        self.assertEqual("all_fragment_orders", result.fragment_order_policy)
        self.assertEqual(
            "first_occurrence_deduplication",
            result.output_order_policy,
        )

    def test_graph_native_renders_bracket_atom_modifiers_from_atom_text_fields(
        self,
    ) -> None:
        cases = (
            ("[2H][H]", ("[2H][H]", "[H][2H]")),
            ("[H+]", ("[H+]",)),
            ("[CH3:1]C", ("[CH3:1]C", "C[CH3:1]")),
            ("[NH4+]", ("[NH4+]",)),
        )

        for smiles, expected_outputs in cases:
            result = mol_to_smiles_enum_s_graph_native(smiles)

            with self.subTest(smiles=smiles):
                self.assertEqual(expected_outputs, result.outputs)
                for output in result.outputs:
                    self.assertEqual(graph_signature(smiles), graph_signature(output))
                    self.assertEqual(
                        semantic_signature(smiles),
                        semantic_signature(output),
                    )

    def test_graph_native_renders_triple_bond_through_bond_text_policy(self) -> None:
        result = mol_to_smiles_enum_s_graph_native("C#N")

        self.assertEqual(("C#N", "N#C"), result.outputs)
        for output in result.outputs:
            self.assertEqual(graph_signature("C#N"), graph_signature(output))
            self.assertEqual(semantic_signature("C#N"), semantic_signature(output))

    def test_graph_native_result_pins_first_domain_generation_diagnostics(self) -> None:
        case = next(
            case
            for case in load_south_star_semantic_cases()
            if case.case_id == "isolated_alkene_z"
        )
        result = mol_to_smiles_enum_s_graph_native_for_case(case)

        self.assertIsNotNone(result.generation_diagnostics)
        diagnostics = result.generation_diagnostics
        self.assertEqual(1, diagnostics.fragment_count)
        self.assertEqual((12,), diagnostics.fragment_output_counts)
        self.assertEqual(12, diagnostics.traversal_skeleton_count)
        self.assertEqual(24, diagnostics.marker_slot_count)
        self.assertEqual(2, diagnostics.local_assignment_count)
        self.assertEqual(12, diagnostics.solved_assignment_count)
        self.assertEqual(12, diagnostics.estimated_product_size)

    def test_graph_native_result_pins_ring_generation_diagnostics(self) -> None:
        case = _expanded_support_case("ring_stereo_monocycle_cyclooctene")
        result = mol_to_smiles_enum_s_graph_native(
            case.source_smiles,
            case_id=case.case_id,
        )

        self.assertIsNotNone(result.generation_diagnostics)
        diagnostics = result.generation_diagnostics
        self.assertEqual(1, diagnostics.fragment_count)
        self.assertEqual((112,), diagnostics.fragment_output_counts)
        self.assertEqual(224, diagnostics.traversal_skeleton_count)
        self.assertEqual(448, diagnostics.marker_slot_count)
        self.assertEqual(2, diagnostics.local_assignment_count)
        self.assertEqual(224, diagnostics.solved_assignment_count)
        self.assertEqual(224, diagnostics.estimated_product_size)

    def test_graph_native_result_pins_polycyclic_generation_diagnostics(self) -> None:
        case = _expanded_support_case(
            "nonstereo_polycyclic_skeleton_bicyclo_2_2_1_heptane"
        )
        result = mol_to_smiles_enum_s_graph_native(
            case.source_smiles,
            case_id=case.case_id,
        )

        self.assertIsNotNone(result.generation_diagnostics)
        diagnostics = result.generation_diagnostics
        self.assertEqual(1, diagnostics.fragment_count)
        self.assertEqual((192,), diagnostics.fragment_output_counts)
        self.assertEqual(456, diagnostics.traversal_skeleton_count)
        self.assertEqual(0, diagnostics.marker_slot_count)
        self.assertEqual(1, diagnostics.local_assignment_count)
        self.assertEqual(456, diagnostics.solved_assignment_count)
        self.assertEqual(456, diagnostics.estimated_product_size)
        self.assertEqual(21, diagnostics.spanning_tree_count)
        self.assertEqual(2, diagnostics.closure_edge_count)
        self.assertEqual(2, diagnostics.closure_label_count)
        self.assertEqual(
            diagnostics.traversal_skeleton_count,
            len(diagnostics.closure_edge_set_records),
        )
        self.assertEqual(
            diagnostics.spanning_tree_count,
            len(
                {
                    frozenset(record.closure_edges)
                    for record in diagnostics.closure_edge_set_records
                }
            ),
        )
        for record in diagnostics.closure_edge_set_records:
            with self.subTest(record=record):
                self.assertEqual(2, len(record.closure_edges))
                self.assertEqual(2, len(record.closure_ids))
                self.assertEqual({"1", "2"}, set(record.closure_labels))

    def test_graph_native_result_pins_disconnected_generation_diagnostics(self) -> None:
        case = _expanded_support_case("disconnected_stereo_fragment_and_atom")
        result = mol_to_smiles_enum_s_graph_native(
            case.source_smiles,
            case_id=case.case_id,
        )

        self.assertIsNotNone(result.generation_diagnostics)
        diagnostics = result.generation_diagnostics
        self.assertEqual(2, diagnostics.fragment_count)
        self.assertEqual((12, 1), diagnostics.fragment_output_counts)
        self.assertEqual(
            ("fragment:0", "fragment:1"),
            tuple(
                record.fragment_id
                for record in diagnostics.fragment_generation_records
            ),
        )
        self.assertEqual(
            ((0, 1, 2, 3), (4,)),
            tuple(
                record.source_atom_indices
                for record in diagnostics.fragment_generation_records
            ),
        )
        self.assertEqual(
            ("F/C=C\\Cl", "O"),
            tuple(
                record.source_fragment_smiles
                for record in diagnostics.fragment_generation_records
            ),
        )
        self.assertEqual(
            diagnostics.fragment_output_counts,
            tuple(
                record.output_count
                for record in diagnostics.fragment_generation_records
            ),
        )
        self.assertEqual(13, diagnostics.traversal_skeleton_count)
        self.assertEqual(24, diagnostics.marker_slot_count)
        self.assertEqual(2, diagnostics.local_assignment_count)
        self.assertEqual(13, diagnostics.solved_assignment_count)
        self.assertEqual(24, diagnostics.estimated_product_size)

    def test_graph_native_tree_traversal_excludes_negative_semantic_witnesses(
        self,
    ) -> None:
        for case in load_south_star_semantic_cases():
            result = mol_to_smiles_enum_s_graph_native_for_case(case)
            negative_outputs = {
                negative.smiles for negative in case.negative_semantic_smiles
            }

            with self.subTest(case_id=case.case_id):
                self.assertFalse(negative_outputs.intersection(result.outputs))

    def test_graph_native_tree_traversal_expands_beyond_seed_root(self) -> None:
        case = next(
            case
            for case in load_south_star_semantic_cases()
            if case.case_id == "isolated_alkene_z"
        )
        result = mol_to_smiles_enum_s_graph_native_for_case(case)

        self.assertIn("Cl\\C=C/F", result.outputs)
        self.assertIn("Cl/C=C\\F", result.outputs)

    def test_graph_native_outputs_are_rendered_from_traversal_events(self) -> None:
        case = next(
            case
            for case in load_south_star_semantic_cases()
            if case.case_id == "branched_substituted_alkene"
        )
        result = mol_to_smiles_enum_s_graph_native_for_case(case)
        traversals = mol_to_smiles_enum_s_tree_traversals_for_case(case)

        self.assertEqual(
            result.outputs,
            tuple(
                dict.fromkeys(
                    render_south_star_tree_traversal(traversal)
                    for traversal in traversals
                )
            ),
        )

    def test_graph_native_traversal_events_expose_tree_context(self) -> None:
        case = next(
            case
            for case in load_south_star_semantic_cases()
            if case.case_id == "branched_substituted_alkene"
        )
        traversals = mol_to_smiles_enum_s_tree_traversals_for_case(case)
        events = tuple(event for traversal in traversals for event in traversal.events)
        event_kinds = {event.kind for event in events}

        self.assertTrue({"atom", "bond", "branch_open", "branch_close"} <= event_kinds)
        for event in events:
            if event.kind == "atom":
                self.assertIsNotNone(event.atom_idx)
            elif event.kind == "bond":
                self.assertIsNotNone(event.edge)
                self.assertIsNotNone(event.begin_atom_idx)
                self.assertIsNotNone(event.end_atom_idx)

    def test_directional_markers_are_rendered_from_marker_slots(self) -> None:
        case = next(
            case
            for case in load_south_star_semantic_cases()
            if case.case_id == "branched_substituted_alkene"
        )
        traversals = mol_to_smiles_enum_s_tree_traversals_for_case(case)
        marker_events = tuple(
            (traversal, event)
            for traversal in traversals
            for event in traversal.events
            if event.marker_slot is not None
        )

        self.assertGreater(len(marker_events), 0)
        for traversal, event in marker_events:
            slot = event.marker_slot
            marker_by_slot = {
                assignment.slot_id: assignment.marker
                for assignment in traversal.marker_assignments
            }
            with self.subTest(slot_id=slot.slot_id):
                self.assertEqual("", event.text)
                self.assertIn(marker_by_slot[slot.slot_id], {"/", "\\"})
                self.assertEqual(event.edge, slot.edge)
                self.assertEqual(
                    event.begin_atom_idx,
                    slot.begin_atom_idx,
                )
                self.assertEqual(event.end_atom_idx, slot.end_atom_idx)
                self.assertIn(slot.syntax_position, {"branch", "main"})
                self.assertNotEqual((), slot.adjacent_contexts)

    def test_non_directional_bond_events_do_not_have_marker_slots(self) -> None:
        case = next(
            case
            for case in load_south_star_semantic_cases()
            if case.case_id == "branched_substituted_alkene"
        )
        traversals = mol_to_smiles_enum_s_tree_traversals_for_case(case)

        for event in (
            event
            for traversal in traversals
            for event in traversal.events
            if event.kind == "bond" and event.marker_slot is None
        ):
            with self.subTest(text=event.text, edge=event.edge):
                self.assertIsNone(event.marker_slot)

    def test_renderer_consumes_supplied_marker_assignments(self) -> None:
        slot_id = "main:root->0->1"
        slot = SouthStarMarkerSlot(
            slot_id=slot_id,
            edge=(0, 1),
            begin_atom_idx=0,
            end_atom_idx=1,
            begin_parent_idx=None,
            syntax_position="main",
            adjacent_contexts=(),
        )
        events = (
            SouthStarTraversalEvent(kind="atom", text="C", atom_idx=0),
            SouthStarTraversalEvent(
                kind="bond",
                text="",
                edge=(0, 1),
                begin_atom_idx=0,
                end_atom_idx=1,
                begin_parent_idx=None,
                marker_slot=slot,
            ),
            SouthStarTraversalEvent(kind="atom", text="C", atom_idx=1),
        )

        self.assertEqual(
            "C/C",
            render_south_star_traversal(
                events,
                marker_assignments=(
                    SouthStarMarkerSlotAssignment(
                        slot_id=slot_id,
                        marker="/",
                    ),
                ),
            ),
        )
        self.assertEqual(
            "C\\C",
            render_south_star_traversal(
                events,
                marker_assignments=(
                    SouthStarMarkerSlotAssignment(
                        slot_id=slot_id,
                        marker="\\",
                    ),
                ),
            ),
        )

    def test_renderer_requires_exact_marker_assignment_coverage(self) -> None:
        slot = SouthStarMarkerSlot(
            slot_id="main:root->0->1",
            edge=(0, 1),
            begin_atom_idx=0,
            end_atom_idx=1,
            begin_parent_idx=None,
            syntax_position="main",
            adjacent_contexts=(),
        )
        events = (
            SouthStarTraversalEvent(
                kind="bond",
                text="",
                edge=(0, 1),
                begin_atom_idx=0,
                end_atom_idx=1,
                begin_parent_idx=None,
                marker_slot=slot,
            ),
        )

        with self.assertRaisesRegex(ValueError, "exactly cover"):
            render_south_star_traversal(events, marker_assignments=())
        with self.assertRaisesRegex(ValueError, "duplicate marker assignment"):
            render_south_star_traversal(
                events,
                marker_assignments=(
                    SouthStarMarkerSlotAssignment(
                        slot_id="main:root->0->1",
                        marker="/",
                    ),
                    SouthStarMarkerSlotAssignment(
                        slot_id="main:root->0->1",
                        marker="/",
                    ),
                ),
            )

    def test_renderer_requires_exact_tetrahedral_renderer_inputs(self) -> None:
        renderer_input = SouthStarRendererInput(
            family_id="tetrahedral_traversal_token",
            syntax_slot_id="tetrahedral_token:1:atom:0,atom:2,atom:3,implicit_hydrogen",
            token_family="tetrahedral_stereo_token",
            value="@",
        )
        events = (
            SouthStarTraversalEvent(
                kind="atom",
                text="[C@H]",
                atom_idx=1,
                renderer_input=renderer_input,
            ),
        )

        with self.assertRaisesRegex(ValueError, "renderer inputs"):
            render_south_star_traversal(events, marker_assignments=())
        with self.assertRaisesRegex(ValueError, "duplicate renderer input"):
            render_south_star_traversal(
                events,
                marker_assignments=(),
                renderer_inputs=(renderer_input, renderer_input),
            )
        with self.assertRaisesRegex(ValueError, "does not match traversal"):
            render_south_star_traversal(
                events,
                marker_assignments=(),
                renderer_inputs=(
                    SouthStarRendererInput(
                        family_id=renderer_input.family_id,
                        syntax_slot_id=renderer_input.syntax_slot_id,
                        token_family=renderer_input.token_family,
                        value="@@",
                    ),
                ),
            )

        self.assertEqual(
            "[C@H]",
            render_south_star_traversal(
                events,
                marker_assignments=(),
                renderer_inputs=(renderer_input,),
            ),
        )

    def test_graph_native_tetrahedral_events_carry_renderer_inputs(self) -> None:
        case = _expanded_support_case("implicit_h_tetrahedral_center")
        traversal = next(
            traversal
            for traversal in mol_to_smiles_enum_s_tree_traversals_for_case(case)
            if any(event.renderer_input is not None for event in traversal.events)
        )
        renderer_inputs = tuple(
            event.renderer_input
            for event in traversal.events
            if event.renderer_input is not None
        )

        self.assertEqual(1, len(renderer_inputs))
        self.assertIn(renderer_inputs[0].value, {"@", "@@"})
        with self.assertRaisesRegex(ValueError, "renderer inputs"):
            render_south_star_traversal(
                traversal.events,
                marker_assignments=traversal.marker_assignments,
            )
        self.assertEqual(
            render_south_star_tree_traversal(traversal),
            render_south_star_traversal(
                traversal.events,
                marker_assignments=traversal.marker_assignments,
                renderer_inputs=renderer_inputs,
            ),
        )

    def test_renderer_renders_ring_closure_labels_from_event_data(self) -> None:
        events = (
            SouthStarTraversalEvent(kind="atom", text="C", atom_idx=0),
            SouthStarTraversalEvent(
                kind="ring_open",
                text="",
                edge=(0, 1),
                begin_atom_idx=0,
                end_atom_idx=1,
                begin_parent_idx=None,
                ring_closure=SouthStarRingClosure(
                    closure_id="0-1",
                    label="1",
                    role="open",
                ),
            ),
            SouthStarTraversalEvent(kind="atom", text="C", atom_idx=1),
            SouthStarTraversalEvent(
                kind="ring_close",
                text="",
                edge=(0, 1),
                begin_atom_idx=1,
                end_atom_idx=0,
                begin_parent_idx=0,
                ring_closure=SouthStarRingClosure(
                    closure_id="0-1",
                    label="1",
                    role="close",
                ),
            ),
        )

        self.assertEqual(
            "C1C1",
            render_south_star_traversal(events, marker_assignments=()),
        )

    def test_renderer_renders_ring_marker_slots_from_event_data(self) -> None:
        slot_id = "ring_open:root->0->1"
        slot = SouthStarMarkerSlot(
            slot_id=slot_id,
            edge=(0, 1),
            begin_atom_idx=0,
            end_atom_idx=1,
            begin_parent_idx=None,
            syntax_position="ring_open",
            adjacent_contexts=(),
        )
        events = (
            SouthStarTraversalEvent(kind="atom", text="C", atom_idx=0),
            SouthStarTraversalEvent(
                kind="ring_open",
                text="",
                edge=(0, 1),
                begin_atom_idx=0,
                end_atom_idx=1,
                begin_parent_idx=None,
                marker_slot=slot,
                ring_closure=SouthStarRingClosure(
                    closure_id="0-1",
                    label="1",
                    role="open",
                ),
            ),
            SouthStarTraversalEvent(kind="atom", text="C", atom_idx=1),
            SouthStarTraversalEvent(
                kind="ring_close",
                text="",
                edge=(0, 1),
                begin_atom_idx=1,
                end_atom_idx=0,
                begin_parent_idx=0,
                ring_closure=SouthStarRingClosure(
                    closure_id="0-1",
                    label="1",
                    role="close",
                ),
            ),
        )

        self.assertEqual(
            "C/1C1",
            render_south_star_traversal(
                events,
                marker_assignments=(
                    SouthStarMarkerSlotAssignment(slot_id=slot_id, marker="/"),
                ),
            ),
        )

    def test_graph_native_traversal_enumerates_simple_saturated_ring(self) -> None:
        case = _expanded_support_case("simple_saturated_monocycle_cyclohexane")
        result = mol_to_smiles_enum_s_graph_native(
            case.source_smiles,
            case_id=case.case_id,
        )

        self.assertEqual(case.case_id, result.case_id)
        self.assertEqual(case.expected_support, result.outputs)
        for output in result.outputs:
            with self.subTest(output=output):
                self.assertEqual(
                    graph_signature(case.source_smiles),
                    graph_signature(output),
                )

    def test_graph_native_traversal_enumerates_unsaturated_nonstereo_ring(
        self,
    ) -> None:
        case = _expanded_support_case("unsaturated_nonstereo_monocycle_cyclohexene")
        result = mol_to_smiles_enum_s_graph_native(
            case.source_smiles,
            case_id=case.case_id,
        )

        self.assertEqual(case.case_id, result.case_id)
        self.assertEqual(case.expected_support, result.outputs)
        for output in result.outputs:
            with self.subTest(output=output):
                self.assertEqual(
                    graph_signature(case.source_smiles),
                    graph_signature(output),
                )

    def test_graph_native_traversal_enumerates_ring_stereo_monocycle(self) -> None:
        case = _expanded_support_case("ring_stereo_monocycle_cyclooctene")
        result = mol_to_smiles_enum_s_graph_native(
            case.source_smiles,
            case_id=case.case_id,
        )

        self.assertEqual(case.case_id, result.case_id)
        self.assertEqual(case.expected_support, result.outputs)
        for output in result.outputs:
            with self.subTest(output=output):
                self.assertEqual(
                    semantic_signature(case.source_smiles),
                    semantic_signature(output),
                )

    def test_graph_native_traversal_enumerates_nonstereo_polycycle(self) -> None:
        case = _expanded_support_case(
            "nonstereo_polycyclic_skeleton_bicyclo_2_2_1_heptane"
        )
        result = mol_to_smiles_enum_s_graph_native(
            case.source_smiles,
            case_id=case.case_id,
        )

        self.assertEqual(case.case_id, result.case_id)
        self.assertEqual(case.expected_support, result.outputs)
        for output in result.outputs:
            with self.subTest(output=output):
                self.assertEqual(
                    graph_signature(case.source_smiles),
                    graph_signature(output),
                )

    def test_graph_native_polycyclic_skeleton_examples_parse_back(self) -> None:
        cases = (
            "C1CC2CCCC2C1",
            "C1CCC2(CC1)CCCC2",
        )

        for source_smiles in cases:
            result = mol_to_smiles_enum_s_graph_native(source_smiles)
            source_graph = graph_signature(source_smiles)

            with self.subTest(source_smiles=source_smiles):
                self.assertGreater(len(result.outputs), 0)
                for output in result.outputs:
                    self.assertEqual(source_graph, graph_signature(output))

    def test_simple_ring_traversals_expose_real_ring_closure_events(self) -> None:
        case = SouthStarSemanticCase(
            case_id="cyclohexane",
            semantic_feature="simple saturated monocycle",
            source_smiles="C1CCCCC1",
            eligible_carrier_edges=(),
            maximal_eligible_carrier=SouthStarAnnotationPolicyExpectation(
                required_marker_edge_count=0,
            ),
            rdkit_writer_membership_status="not_checked",
            rdkit_writer_membership_notes="Synthetic simple-ring traversal test case.",
            positive_semantic_smiles=(),
            negative_semantic_smiles=(),
        )

        traversals = mol_to_smiles_enum_s_tree_traversals_for_case(case)
        ring_events = tuple(
            event
            for traversal in traversals
            for event in traversal.events
            if event.ring_closure is not None
        )

        self.assertGreater(len(ring_events), 0)
        self.assertEqual(
            {"ring_open", "ring_close"},
            {event.kind for event in ring_events},
        )
        self.assertTrue(all(event.ring_closure.label == "1" for event in ring_events))
        self.assertTrue(
            all(event.ring_closure.closure_id for event in ring_events)
        )

    def test_unsaturated_ring_closure_events_carry_bond_text(self) -> None:
        case = _expanded_support_case("unsaturated_nonstereo_monocycle_cyclohexene")

        traversals = mol_to_smiles_enum_s_tree_traversals_for_case(case)
        ring_open_texts = {
            event.text
            for traversal in traversals
            for event in traversal.events
            if event.kind == "ring_open"
        }

        self.assertIn("=", ring_open_texts)

    def test_ring_closure_carriers_emit_event_local_marker_slots(self) -> None:
        case = _expanded_support_case("ring_stereo_monocycle_cyclooctene")

        traversals = mol_to_smiles_enum_s_tree_traversals_for_case(case)
        ring_marker_slots = tuple(
            event.marker_slot
            for traversal in traversals
            for event in traversal.events
            if event.kind == "ring_open" and event.marker_slot is not None
        )

        self.assertGreater(len(ring_marker_slots), 0)
        self.assertTrue(
            all(slot.syntax_position == "ring_open" for slot in ring_marker_slots)
        )

    def test_stereo_double_bond_ring_closures_are_event_modeled(self) -> None:
        case = _expanded_support_case("ring_stereo_monocycle_cyclooctene")

        traversals = mol_to_smiles_enum_s_tree_traversals_for_case(case)
        central_closure_traversals = tuple(
            traversal
            for traversal in traversals
            if any(
                event.kind == "ring_open"
                and event.edge == (1, 2)
                and event.text == "="
                for event in traversal.events
            )
        )

        self.assertGreater(len(central_closure_traversals), 0)
        self.assertTrue(
            all(
                event.marker_slot is None
                for traversal in central_closure_traversals
                for event in traversal.events
                if event.ring_closure is not None and event.edge == (1, 2)
            )
        )
        self.assertTrue(
            all(
                any(
                    event.marker_slot is not None
                    and event.marker_slot.syntax_position in {"branch", "main"}
                    for event in traversal.events
                )
                for traversal in central_closure_traversals
            )
        )

    def test_graph_native_composes_markerless_disconnected_fragments(self) -> None:
        case = _expanded_support_case("markerless_disconnected_ring_and_atom")
        result = mol_to_smiles_enum_s_graph_native(
            case.source_smiles,
            case_id=case.case_id,
        )

        self.assertEqual(case.case_id, result.case_id)
        self.assertEqual(case.expected_support, result.outputs)
        for output in result.outputs:
            with self.subTest(output=output):
                self.assertEqual(
                    graph_signature(case.source_smiles),
                    graph_signature(output),
                )

    def test_graph_native_composes_disconnected_stereo_fragments(self) -> None:
        case = _expanded_support_case("disconnected_stereo_fragment_and_atom")
        result = mol_to_smiles_enum_s_graph_native(
            case.source_smiles,
            case_id=case.case_id,
        )

        self.assertEqual(case.case_id, result.case_id)
        self.assertEqual(case.expected_support, result.outputs)
        for output in result.outputs:
            with self.subTest(output=output):
                self.assertEqual(
                    graph_signature(case.source_smiles),
                    graph_signature(output),
                )
                self.assertEqual(
                    semantic_signature(case.source_smiles),
                    semantic_signature(output),
                )

    def test_graph_native_preserves_implicit_h_tetrahedral_stereo(self) -> None:
        case = _expanded_support_case("implicit_h_tetrahedral_center")
        result = mol_to_smiles_enum_s_graph_native(
            case.source_smiles,
            case_id=case.case_id,
        )

        self.assertEqual(case.case_id, result.case_id)
        self.assertEqual(case.expected_support, result.outputs)
        for output in result.outputs:
            with self.subTest(output=output):
                self.assertEqual(
                    graph_signature(case.source_smiles),
                    graph_signature(output),
                )
                self.assertEqual(
                    semantic_signature(case.source_smiles),
                    semantic_signature(output),
                )

    def test_graph_native_preserves_quaternary_tetrahedral_stereo(self) -> None:
        case = _expanded_support_case("quaternary_tetrahedral_center")
        result = mol_to_smiles_enum_s_graph_native(
            case.source_smiles,
            case_id=case.case_id,
        )

        self.assertEqual(case.case_id, result.case_id)
        self.assertEqual(case.expected_support, result.outputs)
        for output in result.outputs:
            with self.subTest(output=output):
                self.assertEqual(
                    graph_signature(case.source_smiles),
                    graph_signature(output),
                )
                self.assertEqual(
                    semantic_signature(case.source_smiles),
                    semantic_signature(output),
                )

    def test_graph_native_tree_traversal_rejects_unsupported_before_output(
        self,
    ) -> None:
        case = SouthStarSemanticCase(
            case_id="unsupported_ring",
            semantic_feature="unsupported modified aromatic atom text boundary",
            source_smiles="c1cc[nH]c1",
            eligible_carrier_edges=(),
            maximal_eligible_carrier=SouthStarAnnotationPolicyExpectation(
                required_marker_edge_count=0,
            ),
            rdkit_writer_membership_status="not_checked",
            rdkit_writer_membership_notes="Synthetic unsupported-boundary test case.",
            positive_semantic_smiles=(),
            negative_semantic_smiles=(),
        )

        with self.assertRaisesRegex(NotImplementedError, "aromatic_ring_surface"):
            mol_to_smiles_enum_s_graph_native_for_case(case)
