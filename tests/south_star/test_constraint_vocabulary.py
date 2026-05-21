from __future__ import annotations

import unittest

from grimace._south_star.constraint_vocabulary import SouthStarConstraintAssignment
from grimace._south_star.constraint_vocabulary import SouthStarConstraintFamily
from grimace._south_star.constraint_vocabulary import SouthStarRendererInput
from grimace._south_star.marker_equations import (
    DIRECTIONAL_MARKER_CONSTRAINT_FAMILY,
)
from grimace._south_star.marker_equations import (
    constraint_equation_for_marker_equation,
)
from grimace._south_star.marker_equations import (
    constraint_syntax_slot_for_marker_equation,
)
from grimace._south_star.marker_equations import (
    marker_slot_parity_equations_for_case,
)
from grimace._south_star.tetrahedral import (
    TETRAHEDRAL_TRAVERSAL_CONSTRAINT_FAMILY,
)
from grimace._south_star.tetrahedral import (
    constraint_obligation_for_ring_tetrahedral_interaction,
)
from grimace._south_star.tetrahedral import (
    extract_ring_tetrahedral_interaction_obligations,
)
from tests.helpers.south_star_semantic_oracle import parse_smiles
from tests.helpers.south_star_semantics import load_south_star_semantic_cases


class SouthStarConstraintVocabularyTests(unittest.TestCase):
    def test_constraint_family_requires_nonempty_identity(self) -> None:
        with self.assertRaisesRegex(ValueError, "constraint family id"):
            SouthStarConstraintFamily(family_id="", description="bad")

    def test_directional_marker_equation_projects_to_shared_vocabulary(self) -> None:
        equation = next(
            equation
            for equations in marker_slot_parity_equations_for_case(
                _semantic_case("branched_substituted_alkene")
            )
            for equation in equations
        )

        syntax_slot = constraint_syntax_slot_for_marker_equation(equation)
        structural_equation = constraint_equation_for_marker_equation(equation)

        self.assertEqual(
            DIRECTIONAL_MARKER_CONSTRAINT_FAMILY.family_id,
            syntax_slot.family_id,
        )
        self.assertEqual(equation.slot_id, syntax_slot.slot_id)
        self.assertEqual("directional_marker", syntax_slot.slot_kind)
        self.assertEqual(equation.syntax_position, syntax_slot.syntax_position)
        self.assertEqual(equation.edge, syntax_slot.edge)
        self.assertEqual(
            DIRECTIONAL_MARKER_CONSTRAINT_FAMILY.family_id,
            structural_equation.family_id,
        )
        self.assertEqual(equation.equation_id, structural_equation.equation_id)
        self.assertEqual((equation.slot_id,), structural_equation.syntax_slot_ids)
        self.assertEqual(
            tuple(
                dict.fromkeys(term.feature_id for term in equation.feature_terms)
            ),
            structural_equation.obligation_ids,
        )

    def test_ring_tetrahedral_obligation_projects_to_shared_vocabulary(self) -> None:
        obligation = extract_ring_tetrahedral_interaction_obligations(
            parse_smiles("F[C@H]1CCCC(C)C1")
        )[0]

        structural_obligation = constraint_obligation_for_ring_tetrahedral_interaction(
            obligation
        )

        self.assertEqual(
            TETRAHEDRAL_TRAVERSAL_CONSTRAINT_FAMILY.family_id,
            structural_obligation.family_id,
        )
        self.assertEqual("ring_tetrahedral:1", structural_obligation.obligation_id)
        self.assertEqual("atom:1", structural_obligation.subject_id)
        self.assertEqual(
            obligation.required_fact_and_event_fields,
            structural_obligation.required_fact_ids,
        )
        self.assertEqual((), structural_obligation.syntax_slot_ids)

    def test_assignment_and_renderer_input_are_structural_records_only(self) -> None:
        assignment = SouthStarConstraintAssignment(
            family_id="example_family",
            assignment_id="assignment:0",
            syntax_slot_id="slot:0",
            value="/",
        )
        renderer_input = SouthStarRendererInput(
            family_id=assignment.family_id,
            syntax_slot_id=assignment.syntax_slot_id,
            token_family="directional_marker",
            value=assignment.value,
        )

        self.assertEqual("example_family", renderer_input.family_id)
        self.assertEqual("slot:0", renderer_input.syntax_slot_id)
        self.assertEqual("/", renderer_input.value)


def _semantic_case(case_id: str):
    return next(
        case for case in load_south_star_semantic_cases() if case.case_id == case_id
    )


if __name__ == "__main__":
    unittest.main()
