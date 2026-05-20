from __future__ import annotations

import unittest

from tests.helpers.south_star_annotation_policy import (
    DIRECTIONAL_MARKERS,
    AnnotationPolicyDecision,
    EmittedEdgeBasis,
    SemanticCarrierOpportunity,
    SurvivingSemanticAssignment,
    normalized_edge,
)
from tests.helpers.south_star_component_support_state import (
    SouthStarComponentSupportState,
)
from tests.helpers.south_star_semantic_oracle import parse_smiles
from tests.helpers.south_star_semantics import load_south_star_semantic_cases


class ComponentSlashOnlyPolicy:
    def decision(
        self,
        *,
        carrier_opportunities: tuple[SemanticCarrierOpportunity, ...],
        emitted_edge: EmittedEdgeBasis,
        surviving_assignments: tuple[SurvivingSemanticAssignment, ...],
    ) -> AnnotationPolicyDecision:
        del carrier_opportunities, surviving_assignments
        return AnnotationPolicyDecision(
            edge=normalized_edge(emitted_edge.edge),
            marker_required=True,
            allowed_markers=("/",),
        )


class ComponentNoMarkerPolicy:
    def decision(
        self,
        *,
        carrier_opportunities: tuple[SemanticCarrierOpportunity, ...],
        emitted_edge: EmittedEdgeBasis,
        surviving_assignments: tuple[SurvivingSemanticAssignment, ...],
    ) -> AnnotationPolicyDecision:
        del carrier_opportunities, surviving_assignments
        return AnnotationPolicyDecision(
            edge=normalized_edge(emitted_edge.edge),
            marker_required=False,
            allowed_markers=(),
        )


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

    def test_complexity_snapshot_counts_independent_components(self) -> None:
        state = SouthStarComponentSupportState.from_mol(
            parse_smiles("F/C=C\\CC/C=C\\Cl"),
        )

        snapshot = state.complexity_snapshot()

        self.assertEqual(2, snapshot.component_count)
        self.assertEqual(4, snapshot.estimated_product_size)
        self.assertEqual(
            [2, 2],
            [
                estimate.estimated_local_assignment_count
                for estimate in snapshot.local_assignment_estimates
            ],
        )

    def test_complexity_snapshot_exposes_coupled_component_size(self) -> None:
        state = SouthStarComponentSupportState.from_mol(
            parse_smiles("C/C=C/C=C/C"),
        )

        snapshot = state.complexity_snapshot()

        self.assertEqual(1, snapshot.component_count)
        self.assertEqual(4, snapshot.estimated_product_size)
        self.assertEqual(
            1,
            snapshot.local_assignment_estimates[0].coupling_cause_count,
        )
        self.assertEqual(
            2,
            snapshot.local_assignment_estimates[0].source_feature_count,
        )

    def test_injected_policy_controls_component_marker_options(self) -> None:
        state = SouthStarComponentSupportState.from_mol(
            parse_smiles("F/C=C\\Cl"),
            annotation_policy=ComponentSlashOnlyPolicy(),
        )

        self.assertEqual(("/",), state.allowed_directional_markers(edge=(0, 1)))
        rejected = state.explain_directional_marker(edge=(0, 1), marker="\\")
        self.assertFalse(rejected.token_allowed)
        self.assertEqual("marker_rejected_by_annotation_policy", rejected.reason)

    def test_injected_policy_controls_component_marker_requirement(self) -> None:
        state = SouthStarComponentSupportState.from_mol(
            parse_smiles("F/C=C\\Cl"),
            annotation_policy=ComponentNoMarkerPolicy(),
        )

        support = state.explain_directional_marker(edge=(0, 1), marker="/")
        self.assertFalse(support.token_allowed)
        self.assertEqual("marker_not_required_by_annotation_policy", support.reason)
