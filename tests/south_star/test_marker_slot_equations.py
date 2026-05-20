from __future__ import annotations

import unittest

from grimace._south_star.component_support_state import (
    SouthStarComponentSupportState,
)
from grimace._south_star.enum_s import mol_to_smiles_enum_s_tree_traversals_for_case
from grimace._south_star.marker_equations import (
    marker_slot_parity_equations_for_case,
)
from grimace._south_star.marker_equations import (
    marker_slot_parity_equations_for_traversal,
)
from tests.helpers.south_star_semantic_oracle import parse_smiles
from tests.helpers.south_star_semantics import load_south_star_semantic_cases


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
                        self.assertIn(equation.syntax_position, {"branch", "main"})
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


def _case(case_id: str):
    return next(
        case for case in load_south_star_semantic_cases() if case.case_id == case_id
    )
