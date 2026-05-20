from __future__ import annotations

import unittest

from tests.helpers.south_star_component_support_state import (
    SouthStarComponentSupportState,
)
from tests.helpers.south_star_enum_s import (
    mol_to_smiles_enum_s_tree_traversals_for_case,
)
from tests.helpers.south_star_marker_equations import (
    marker_slot_parity_equations_for_traversal,
)
from tests.helpers.south_star_marker_equations import expected_marker_from_equation
from tests.helpers.south_star_semantic_oracle import parse_smiles
from tests.helpers.south_star_semantics import load_south_star_semantic_cases
from tests.helpers.south_star_parity_solver import solve_marker_slot_parity_equations
from tests.helpers.south_star_z3_oracle import (
    z3_marker_assignments_for_equations,
)


class SouthStarZ3EquationOracleTests(unittest.TestCase):
    def test_z3_assignment_sets_match_current_equation_assignments(self) -> None:
        for case_id in (
            "isolated_alkene_z",
            "branched_substituted_alkene",
            "independent_two_alkenes",
            "linear_diene_same_phase",
        ):
            case = _case(case_id)
            state = SouthStarComponentSupportState.from_mol(
                parse_smiles(case.source_smiles)
            )
            traversals = mol_to_smiles_enum_s_tree_traversals_for_case(case)

            for traversal in traversals:
                equations = marker_slot_parity_equations_for_traversal(
                    state,
                    traversal,
                )
                z3_assignments = z3_marker_assignments_for_equations(equations)
                expected_assignment = tuple(
                    sorted(
                        (
                            assignment.slot_id,
                            assignment.marker,
                        )
                        for assignment in traversal.marker_assignments
                    )
                )

                with self.subTest(case_id=case_id, root=traversal.root_atom_idx):
                    self.assertEqual(1, len(z3_assignments))
                    self.assertEqual(
                        expected_assignment,
                        tuple(sorted(z3_assignments[0].marker_by_slot)),
                    )

    def test_z3_oracle_consumes_equations_before_string_rendering(self) -> None:
        case = _case("branched_substituted_alkene")
        state = SouthStarComponentSupportState.from_mol(parse_smiles(case.source_smiles))
        traversal = mol_to_smiles_enum_s_tree_traversals_for_case(case)[0]
        equations = marker_slot_parity_equations_for_traversal(state, traversal)

        self.assertGreater(len(equations), 0)
        for equation in equations:
            with self.subTest(equation_id=equation.equation_id):
                self.assertEqual(
                    equation.emitted_marker,
                    expected_marker_from_equation(equation),
                )

    def test_custom_solver_assignment_sets_match_z3(self) -> None:
        case = _case("linear_diene_same_phase")
        state = SouthStarComponentSupportState.from_mol(
            parse_smiles(case.source_smiles)
        )
        traversals = mol_to_smiles_enum_s_tree_traversals_for_case(case)

        for traversal in traversals:
            equations = marker_slot_parity_equations_for_traversal(state, traversal)
            z3_assignments = z3_marker_assignments_for_equations(equations)
            solver_result = solve_marker_slot_parity_equations(equations)

            with self.subTest(root=traversal.root_atom_idx):
                self.assertEqual(
                    tuple(
                        sorted(
                            tuple(sorted(assignment.marker_by_slot))
                            for assignment in z3_assignments
                        )
                    ),
                    tuple(
                        sorted(
                            assignment.marker_by_slot
                            for assignment in solver_result.assignments
                        )
                    ),
                )


def _case(case_id: str):
    return next(
        case for case in load_south_star_semantic_cases() if case.case_id == case_id
    )
