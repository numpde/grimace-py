from __future__ import annotations

import unittest

from grimace._south_star.enum_s import mol_to_smiles_enum_s_tree_traversals_for_case
from tests.helpers.south_star_semantics import SouthStarAnnotationPolicyExpectation
from tests.helpers.south_star_semantics import SouthStarSemanticCase


class SouthStarConnectedGraphTraversalSpineTests(unittest.TestCase):
    def test_connected_acyclic_traversals_carry_empty_closure_plan(self) -> None:
        traversal = mol_to_smiles_enum_s_tree_traversals_for_case(
            SouthStarSemanticCase(
                case_id="propane",
                semantic_feature="connected acyclic",
                source_smiles="CCC",
                eligible_carrier_edges=(),
                maximal_eligible_carrier=SouthStarAnnotationPolicyExpectation(
                    required_marker_edge_count=0,
                ),
                rdkit_writer_membership_status="not_checked",
                rdkit_writer_membership_notes=(
                    "Synthetic connected-graph traversal spine test case."
                ),
                positive_semantic_smiles=(),
                negative_semantic_smiles=(),
            )
        )[0]

        plan = traversal.connected_graph_plan
        self.assertIsNotNone(plan)
        assert plan is not None

        self.assertEqual(traversal.root_atom_idx, plan.root_atom_idx)
        self.assertEqual(3, len(plan.atom_order))
        self.assertEqual(2, len(plan.tree_edges))
        self.assertEqual((), plan.closure_edges)
        self.assertEqual((), plan.closure_endpoints)

    def test_simple_monocycle_fits_connected_graph_traversal_plan(self) -> None:
        traversal = mol_to_smiles_enum_s_tree_traversals_for_case(
            SouthStarSemanticCase(
                case_id="cyclohexane",
                semantic_feature="simple saturated monocycle",
                source_smiles="C1CCCCC1",
                eligible_carrier_edges=(),
                maximal_eligible_carrier=SouthStarAnnotationPolicyExpectation(
                    required_marker_edge_count=0,
                ),
                rdkit_writer_membership_status="not_checked",
                rdkit_writer_membership_notes=(
                    "Synthetic connected-graph traversal spine test case."
                ),
                positive_semantic_smiles=(),
                negative_semantic_smiles=(),
            )
        )[0]

        plan = traversal.connected_graph_plan
        self.assertIsNotNone(plan)
        assert plan is not None

        self.assertEqual(traversal.root_atom_idx, plan.root_atom_idx)
        self.assertEqual(6, len(plan.atom_order))
        self.assertEqual(5, len(plan.tree_edges))
        self.assertEqual(1, len(plan.closure_edges))
        self.assertEqual(2, len(plan.closure_endpoints))
        self.assertEqual(
            frozenset(edge.edge for edge in plan.tree_edges)
            | frozenset(edge.edge for edge in plan.closure_edges),
            {
                (0, 1),
                (0, 5),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
            },
        )

    def test_connected_graph_plan_keeps_closure_endpoints_event_local(self) -> None:
        traversal = mol_to_smiles_enum_s_tree_traversals_for_case(
            SouthStarSemanticCase(
                case_id="cyclohexene",
                semantic_feature="unsaturated nonstereo monocycle",
                source_smiles="C1=CCCCC1",
                eligible_carrier_edges=(),
                maximal_eligible_carrier=SouthStarAnnotationPolicyExpectation(
                    required_marker_edge_count=0,
                ),
                rdkit_writer_membership_status="not_checked",
                rdkit_writer_membership_notes=(
                    "Synthetic connected-graph traversal spine test case."
                ),
                positive_semantic_smiles=(),
                negative_semantic_smiles=(),
            )
        )[0]

        plan = traversal.connected_graph_plan
        self.assertIsNotNone(plan)
        assert plan is not None

        self.assertEqual(
            ("open", "close"),
            tuple(endpoint.role for endpoint in plan.closure_endpoints),
        )
        self.assertEqual(
            ("ring_open", "ring_close"),
            tuple(endpoint.syntax_position for endpoint in plan.closure_endpoints),
        )
        self.assertEqual(
            ("1",),
            tuple(edge.label for edge in plan.closure_edges),
        )
        self.assertEqual(
            ("1", "1"),
            tuple(endpoint.label for endpoint in plan.closure_endpoints),
        )


if __name__ == "__main__":
    unittest.main()
