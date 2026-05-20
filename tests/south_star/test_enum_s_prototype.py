from __future__ import annotations

import unittest

from tests.helpers.south_star_component_support_state import (
    SouthStarComponentSupportState,
)
from tests.helpers.south_star_enum_s import mol_to_smiles_enum_s_graph_native_for_case
from tests.helpers.south_star_enum_s import mol_to_smiles_enum_s_prototype_for_case
from tests.helpers.south_star_enum_s import (
    mol_to_smiles_enum_s_tree_traversals_for_case,
)
from tests.helpers.south_star_semantics import SouthStarAnnotationPolicyExpectation
from tests.helpers.south_star_semantics import SouthStarSemanticCase
from tests.helpers.south_star_semantics import load_south_star_semantic_cases


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
                    "south_star_graph_native_tree_traversal",
                    result.generation_basis,
                )

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
            tuple(dict.fromkeys(traversal.render() for traversal in traversals)),
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
            event
            for traversal in traversals
            for event in traversal.events
            if event.text in {"/", "\\"}
        )

        self.assertGreater(len(marker_events), 0)
        for event in marker_events:
            self.assertIsNotNone(event.marker_slot)
            slot = event.marker_slot
            with self.subTest(slot_id=slot.slot_id):
                self.assertEqual(event.text, slot.selected_marker)
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
            if event.kind == "bond" and event.text not in {"/", "\\"}
        ):
            with self.subTest(text=event.text, edge=event.edge):
                self.assertIsNone(event.marker_slot)

    def test_graph_native_tree_traversal_rejects_unsupported_before_output(
        self,
    ) -> None:
        case = SouthStarSemanticCase(
            case_id="unsupported_ring",
            source_smiles="C1/C=C\\CCCCC1",
            eligible_carrier_edges=(),
            maximal_eligible_carrier=SouthStarAnnotationPolicyExpectation(
                required_marker_edge_count=0,
            ),
            rdkit_writer_membership_status="not_checked",
            positive_semantic_smiles=(),
            negative_semantic_smiles=(),
        )

        with self.assertRaisesRegex(NotImplementedError, "ring_molecule"):
            mol_to_smiles_enum_s_graph_native_for_case(case)
