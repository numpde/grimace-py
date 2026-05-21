from __future__ import annotations

from functools import lru_cache
import unittest

from grimace._south_star.annotation_policy import (
    MaximalEligibleCarrierAnnotationPolicy,
)
from grimace._south_star.component_support_state import (
    SouthStarComponentSupportState,
)
from grimace._south_star.enum_s import _ring_system_traversals
from grimace._south_star.enum_s import _supported_polycyclic_closure_edge_sets
from grimace._south_star.enum_s import mol_to_smiles_enum_s_graph_native
from grimace._south_star.enum_s import mol_to_smiles_enum_s_tree_traversals_for_case
from grimace._south_star.marker_equations import (
    marker_slot_parity_equations_for_case,
)
from grimace._south_star.marker_equations import (
    marker_slot_parity_equations_for_traversal,
)
from grimace._south_star.molecule_facts import SouthStarMoleculeFacts
from grimace._south_star.parity_solver import solve_marker_slot_parity_equations
from grimace._south_star.reference_model import SouthStarTraversal
from grimace._south_star.support_gates import south_star_support_gate_report
from tests.helpers.south_star_semantic_oracle import parse_smiles
from tests.helpers.south_star_semantic_oracle import south_star_conformance_report
from tests.helpers.south_star_exact_support import (
    load_south_star_expanded_support_cases,
)
from tests.helpers.south_star_semantics import load_south_star_semantic_cases


POLYCYCLIC_STEREO_DIAGNOSTIC_SMILES = "C1CCC/C=C\\C2CCCC2C1"


class SouthStarMarkerSlotEquationTests(unittest.TestCase):
    def test_each_marker_slot_has_one_parity_equation(self) -> None:
        for case in load_south_star_semantic_cases():
            state = SouthStarComponentSupportState.from_mol(
                parse_smiles(case.source_smiles)
            )
            traversals = mol_to_smiles_enum_s_tree_traversals_for_case(case)

            with self.subTest(case_id=case.case_id):
                for traversal in traversals:
                    equations = marker_slot_parity_equations_for_traversal(
                        state,
                        traversal,
                    )
                    slot_ids = tuple(
                        event.marker_slot.slot_id
                        for event in traversal.events
                        if event.marker_slot is not None
                    )
                    self.assertEqual(
                        slot_ids,
                        tuple(equation.slot_id for equation in equations),
                    )
                    for equation in equations:
                        self.assertTrue(equation.equation_id)
                        self.assertIn(
                            equation.syntax_position,
                            {"branch", "main", "ring_open"},
                        )
                        self.assertIn(equation.graph_marker, {"/", "\\"})
                        self.assertIn(equation.emitted_marker, {"/", "\\"})
                        self.assertEqual(
                            equation.traversal_orientation_flip,
                            equation.graph_marker != equation.emitted_marker,
                        )
                        self.assertNotEqual((), equation.component_ids)
                        self.assertNotEqual((), equation.feature_terms)

    def test_equations_record_feature_phase_without_rendering_strings(self) -> None:
        case = _case("branched_substituted_alkene")
        equation_groups = marker_slot_parity_equations_for_case(case)

        feature_terms = tuple(
            term
            for equations in equation_groups
            for equation in equations
            for term in equation.feature_terms
        )

        self.assertGreater(len(feature_terms), 0)
        for term in feature_terms:
            with self.subTest(feature_id=term.feature_id, side=term.carrier_side):
                self.assertIn(term.carrier_side, {"left", "right"})
                self.assertIn(term.source_marker, {"/", "\\"})
                self.assertTrue(term.required_stereo_phase.startswith("STEREO"))

    def test_shared_carrier_incidence_is_explicit(self) -> None:
        case = _case("linear_diene_same_phase")
        equation_groups = marker_slot_parity_equations_for_case(case)
        shared_equations = tuple(
            equation
            for equations in equation_groups
            for equation in equations
            if any(
                term.shared_carrier_incidence_count > 1
                for term in equation.feature_terms
            )
        )

        self.assertGreater(len(shared_equations), 0)
        for equation in shared_equations:
            with self.subTest(equation_id=equation.equation_id):
                self.assertGreater(len(equation.feature_terms), 1)

    def test_ring_open_marker_slots_have_parity_equations(self) -> None:
        case = _expanded_case("ring_stereo_monocycle_cyclooctene")
        state = SouthStarComponentSupportState.from_mol(parse_smiles(case.source_smiles))
        traversals = mol_to_smiles_enum_s_tree_traversals_for_case(case)

        ring_open_equations = []
        for traversal in traversals:
            equations = marker_slot_parity_equations_for_traversal(state, traversal)
            ring_open_equations.extend(
                equation
                for equation in equations
                if equation.syntax_position == "ring_open"
            )

        self.assertGreater(len(ring_open_equations), 0)
        for equation in ring_open_equations:
            with self.subTest(equation_id=equation.equation_id):
                self.assertTrue(equation.slot_id.startswith("ring_open:"))
                self.assertIn(equation.graph_marker, {"/", "\\"})
                self.assertIn(equation.emitted_marker, {"/", "\\"})

    def test_ring_open_marker_slots_solve_through_shared_parity_solver(self) -> None:
        case = _expanded_case("ring_stereo_monocycle_cyclooctene")
        state = SouthStarComponentSupportState.from_mol(parse_smiles(case.source_smiles))
        traversals = mol_to_smiles_enum_s_tree_traversals_for_case(case)
        ring_open_traversals = tuple(
            traversal
            for traversal in traversals
            if any(
                event.marker_slot is not None
                and event.marker_slot.syntax_position == "ring_open"
                for event in traversal.events
            )
        )

        self.assertGreater(len(ring_open_traversals), 0)
        for traversal in ring_open_traversals:
            equations = marker_slot_parity_equations_for_traversal(state, traversal)
            solver_result = solve_marker_slot_parity_equations(equations)
            expected_assignment = tuple(
                sorted(
                    (assignment.slot_id, assignment.marker)
                    for assignment in traversal.marker_assignments
                )
            )

            with self.subTest(root=traversal.root_atom_idx):
                self.assertTrue(
                    any(
                        equation.syntax_position == "ring_open"
                        for equation in equations
                    )
                )
                self.assertEqual(1, len(solver_result.assignments))
                self.assertEqual(
                    expected_assignment,
                    solver_result.assignments[0].marker_by_slot,
                )

    def test_stereo_double_bond_ring_closure_uses_carrier_equations(
        self,
    ) -> None:
        case = _expanded_case("ring_stereo_monocycle_cyclooctene")
        state = SouthStarComponentSupportState.from_mol(parse_smiles(case.source_smiles))
        traversals = mol_to_smiles_enum_s_tree_traversals_for_case(case)

        central_closure_equation_groups = []
        for traversal in traversals:
            if not any(
                event.kind == "ring_open"
                and event.edge == (1, 2)
                and event.text == "="
                for event in traversal.events
            ):
                continue
            central_closure_equation_groups.append(
                marker_slot_parity_equations_for_traversal(state, traversal)
            )

        self.assertGreater(len(central_closure_equation_groups), 0)
        for equations in central_closure_equation_groups:
            with self.subTest(equations=equations):
                self.assertEqual(2, len(equations))
                self.assertTrue(
                    all(
                        equation.syntax_position in {"main", "branch"}
                        for equation in equations
                    )
                )
                self.assertEqual(
                    {(1, 2)},
                    {
                        term.central_bond
                        for equation in equations
                        for term in equation.feature_terms
                    },
                )

    def test_polycyclic_ring_open_carriers_have_component_equations(self) -> None:
        state, traversals = _polycyclic_stereo_diagnostic_state_and_traversals()
        ring_open_examples = tuple(
            (traversal, equation)
            for traversal in traversals
            for equation in marker_slot_parity_equations_for_traversal(
                state,
                traversal,
            )
            if equation.syntax_position == "ring_open"
        )

        self.assertGreater(len(ring_open_examples), 0)
        for traversal, equation in ring_open_examples[:20]:
            ring_open_events_by_slot = {
                event.marker_slot.slot_id: event
                for event in traversal.events
                if event.kind == "ring_open" and event.marker_slot is not None
            }

            with self.subTest(equation_id=equation.equation_id):
                self.assertIn(equation.slot_id, ring_open_events_by_slot)
                event = ring_open_events_by_slot[equation.slot_id]
                self.assertIsNotNone(event.ring_closure)
                self.assertTrue(equation.slot_id.startswith("ring_open:"))
                self.assertEqual((3, 4), equation.edge)
                self.assertEqual(("component:0",), equation.component_ids)
                self.assertEqual(1, len(equation.feature_terms))
                term = equation.feature_terms[0]
                self.assertEqual("bond:4", term.feature_id)
                self.assertEqual((4, 5), term.central_bond)
                self.assertEqual("left", term.carrier_side)
                self.assertEqual("/", term.source_marker)

    def test_polycyclic_ring_open_carriers_solve_with_same_solver(self) -> None:
        state, traversals = _polycyclic_stereo_diagnostic_state_and_traversals()
        ring_open_traversals = tuple(
            traversal
            for traversal in traversals
            if any(
                event.kind == "ring_open" and event.marker_slot is not None
                for event in traversal.events
            )
        )

        self.assertGreater(len(ring_open_traversals), 0)
        for traversal in ring_open_traversals[:20]:
            equations = marker_slot_parity_equations_for_traversal(state, traversal)
            solver_result = solve_marker_slot_parity_equations(equations)
            expected_assignment = tuple(
                sorted(
                    (assignment.slot_id, assignment.marker)
                    for assignment in traversal.marker_assignments
                )
            )

            with self.subTest(rendered=traversal.render()):
                self.assertTrue(
                    any(
                        equation.syntax_position == "ring_open"
                        for equation in equations
                    )
                )
                self.assertEqual(1, len(solver_result.assignments))
                self.assertEqual(
                    expected_assignment,
                    solver_result.assignments[0].marker_by_slot,
                )

    def test_polycyclic_stereo_uses_public_support_path(self) -> None:
        report = south_star_support_gate_report(
            parse_smiles(POLYCYCLIC_STEREO_DIAGNOSTIC_SMILES)
        )
        result = mol_to_smiles_enum_s_graph_native(POLYCYCLIC_STEREO_DIAGNOSTIC_SMILES)

        self.assertTrue(report.supported, report.unsupported_features)
        self.assertGreater(len(result.outputs), 0)
        self.assertIsNotNone(result.generation_diagnostics)

    def test_polycyclic_stereo_public_outputs_parse_to_source_semantics(self) -> None:
        source_smiles = "C1CCC/C=C\\C2C1C2"
        result = mol_to_smiles_enum_s_graph_native(source_smiles)

        self.assertGreater(len(result.outputs), 0)
        for output in result.outputs:
            with self.subTest(output=output):
                report = south_star_conformance_report(
                    source_smiles=source_smiles,
                    candidate_smiles=output,
                )
                self.assertTrue(report.accepted, report.rejection_reasons)


def _case(case_id: str):
    return next(
        case for case in load_south_star_semantic_cases() if case.case_id == case_id
    )


def _expanded_case(case_id: str):
    return next(
        case
        for case in load_south_star_expanded_support_cases()
        if case.case_id == case_id
    )


@lru_cache(maxsize=1)
def _polycyclic_stereo_diagnostic_state_and_traversals():
    mol = parse_smiles(POLYCYCLIC_STEREO_DIAGNOSTIC_SMILES)
    facts = SouthStarMoleculeFacts.from_mol(mol)
    if len(facts.components) != 1:
        raise AssertionError("polycyclic stereo diagnostic expects one component")
    state = SouthStarComponentSupportState.from_molecule_facts(
        facts,
        annotation_policy=MaximalEligibleCarrierAnnotationPolicy(),
    )
    assignment = state.component_marker_assignments()[0][0]
    traversals = _ring_system_traversals(
        mol,
        molecule_facts=facts,
        state=state,
        closure_edge_sets=_supported_polycyclic_closure_edge_sets(mol),
        marker_by_edge=dict(assignment.marker_by_edge),
        component_marker_assignments=(assignment,),
    )
    return state, _assert_traversal_tuple(traversals)


def _assert_traversal_tuple(
    traversals: tuple[SouthStarTraversal, ...],
) -> tuple[SouthStarTraversal, ...]:
    if not traversals:
        raise AssertionError("polycyclic stereo diagnostic produced no traversals")
    return traversals
