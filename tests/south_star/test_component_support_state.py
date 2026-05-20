from __future__ import annotations

import unittest

from tests.helpers.south_star_annotation_policy import DIRECTIONAL_MARKERS
from tests.helpers.south_star_component_support_state import (
    SouthStarComponentSupportState,
)
from tests.helpers.south_star_semantic_oracle import parse_smiles
from tests.helpers.south_star_semantics import load_south_star_semantic_cases


class SouthStarComponentSupportStateTests(unittest.TestCase):
    def test_fixture_carrier_markers_affect_extracted_components(self) -> None:
        for case in load_south_star_semantic_cases():
            state = SouthStarComponentSupportState.from_case(case)

            for edge in case.eligible_carrier_edges:
                support = state.explain_directional_marker(edge=edge, marker="/")

                with self.subTest(case_id=case.case_id, edge=edge):
                    self.assertTrue(support.token_allowed)
                    self.assertNotEqual((), support.affected_components)

    def test_shared_carrier_query_affects_one_coupled_component(self) -> None:
        state = SouthStarComponentSupportState.from_mol(
            parse_smiles("C/C=C/C=C/C"),
        )
        support = state.explain_directional_marker(edge=(2, 3), marker="/")

        self.assertTrue(support.token_allowed)
        self.assertEqual(1, len(support.affected_components))
        self.assertEqual("component:0", support.affected_components[0].component_id)

    def test_independent_component_query_does_not_touch_unaffected_component(self) -> None:
        state = SouthStarComponentSupportState.from_mol(
            parse_smiles("F/C=C\\CC/C=C\\Cl"),
        )
        support = state.explain_directional_marker(edge=(0, 1), marker="/")

        self.assertTrue(support.token_allowed)
        self.assertEqual(2, len(state.components))
        self.assertEqual(1, len(support.affected_components))
        self.assertEqual("component:0", support.affected_components[0].component_id)

    def test_noncarrier_edge_rejects_without_affecting_components(self) -> None:
        state = SouthStarComponentSupportState.from_mol(
            parse_smiles("F/C=C\\CC/C=C\\Cl"),
        )
        support = state.explain_directional_marker(edge=(3, 4), marker="/")

        self.assertFalse(support.token_allowed)
        self.assertEqual("edge_affects_no_semantic_component", support.reason)
        self.assertEqual((), support.affected_components)

    def test_invalid_marker_fails_fast(self) -> None:
        state = SouthStarComponentSupportState.from_mol(parse_smiles("F/C=C\\Cl"))

        with self.assertRaisesRegex(ValueError, "directional marker"):
            state.explain_directional_marker(edge=(0, 1), marker="?")

    def test_allowed_directional_markers_are_component_local(self) -> None:
        state = SouthStarComponentSupportState.from_mol(parse_smiles("F/C=C\\Cl"))

        self.assertEqual(
            DIRECTIONAL_MARKERS,
            state.allowed_directional_markers(edge=(0, 1)),
        )
