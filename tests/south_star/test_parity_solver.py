from __future__ import annotations

import unittest

from grimace._south_star.component_support_state import (
    SouthStarComponentSupportState,
)
from grimace._south_star.enum_s import (
    mol_to_smiles_enum_s_tree_traversals_for_case,
)
from grimace._south_star.marker_equations import (
    marker_slot_parity_equations_for_traversal,
)
from grimace._south_star.parity_solver import (
    solve_marker_slot_parity_equations,
)
from tests.helpers.south_star_semantic_oracle import parse_smiles
from tests.helpers.south_star_semantics import load_south_star_semantic_cases


class SouthStarParitySolverTests(unittest.TestCase):
    def test_solver_assignments_match_current_equation_assignments(self) -> None:
        for case in load_south_star_semantic_cases():
            state = SouthStarComponentSupportState.from_mol(
                parse_smiles(case.source_smiles)
            )
            traversals = mol_to_smiles_enum_s_tree_traversals_for_case(case)

            for traversal in traversals:
                equations = marker_slot_parity_equations_for_traversal(
                    state,
                    traversal,
                )
                result = solve_marker_slot_parity_equations(equations)
                expected_assignment = tuple(
                    sorted(
                        (
                            assignment.slot_id,
                            assignment.marker,
                        )
                        for assignment in traversal.marker_assignments
                    )
                )

                with self.subTest(case_id=case.case_id, root=traversal.root_atom_idx):
                    self.assertEqual(1, len(result.assignments))
                    self.assertEqual(
                        expected_assignment,
                        result.assignments[0].marker_by_slot,
                    )

    def test_solver_diagnostic_exposes_component_factorization(self) -> None:
        case = _case("independent_two_alkenes")
        state = SouthStarComponentSupportState.from_mol(
            parse_smiles(case.source_smiles)
        )
        traversal = mol_to_smiles_enum_s_tree_traversals_for_case(case)[0]
        equations = marker_slot_parity_equations_for_traversal(state, traversal)
        diagnostic = solve_marker_slot_parity_equations(equations).diagnostic

        self.assertEqual(2, diagnostic.component_count)
        self.assertEqual(
            ("component:0", "component:1"),
            diagnostic.affected_component_ids,
        )
        self.assertEqual(len(equations), diagnostic.equation_count)
        self.assertEqual(1, diagnostic.local_assignment_count)
        self.assertEqual(1, diagnostic.estimated_product_size)

    def test_solver_diagnostic_exposes_shared_carrier_coupling(self) -> None:
        case = _case("linear_diene_same_phase")
        state = SouthStarComponentSupportState.from_mol(
            parse_smiles(case.source_smiles)
        )
        traversal = mol_to_smiles_enum_s_tree_traversals_for_case(case)[0]
        equations = marker_slot_parity_equations_for_traversal(state, traversal)
        diagnostic = solve_marker_slot_parity_equations(equations).diagnostic

        self.assertEqual(("shared_carrier_edge",), diagnostic.coupling_causes)


def _case(case_id: str):
    return next(
        case for case in load_south_star_semantic_cases() if case.case_id == case_id
    )
