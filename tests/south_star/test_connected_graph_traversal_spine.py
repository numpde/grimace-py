from __future__ import annotations

import unittest

from grimace._south_star.annotation_policy import normalized_edge
from grimace._south_star.enum_s import mol_to_smiles_enum_s_tree_traversals_for_case
from grimace._south_star.reference_model import (
    SouthStarConnectedGraphTraversalPlan,
    SouthStarTraversalClosureEdge,
    SouthStarTraversalClosureEndpoint,
    SouthStarTraversalEvent,
    SouthStarTraversalTreeEdge,
)
from tests.helpers.south_star_semantics import SouthStarAnnotationPolicyExpectation
from tests.helpers.south_star_semantics import SouthStarSemanticCase


class SouthStarConnectedGraphTraversalSpineTests(unittest.TestCase):
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

        plan = _connected_graph_plan_from_events(
            root_atom_idx=traversal.root_atom_idx,
            events=traversal.events,
        )

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

        plan = _connected_graph_plan_from_events(
            root_atom_idx=traversal.root_atom_idx,
            events=traversal.events,
        )

        self.assertEqual(
            ("open", "close"),
            tuple(endpoint.role for endpoint in plan.closure_endpoints),
        )
        self.assertEqual(
            ("ring_open", "ring_close"),
            tuple(endpoint.syntax_position for endpoint in plan.closure_endpoints),
        )
        self.assertEqual(
            tuple(edge.label for edge in plan.closure_edges),
            tuple(endpoint.label for endpoint in plan.closure_endpoints[:1]),
        )


def _connected_graph_plan_from_events(
    *,
    root_atom_idx: int,
    events: tuple[SouthStarTraversalEvent, ...],
) -> SouthStarConnectedGraphTraversalPlan:
    atom_order = tuple(
        event.atom_idx
        for event in events
        if event.kind == "atom" and event.atom_idx is not None
    )
    tree_edges = tuple(
        SouthStarTraversalTreeEdge(
            edge=event.edge,
            begin_atom_idx=event.begin_atom_idx,
            end_atom_idx=event.end_atom_idx,
            begin_parent_idx=event.begin_parent_idx,
            syntax_position=event.syntax_position,
        )
        for event in events
        if event.kind == "bond"
        and event.edge is not None
        and event.begin_atom_idx is not None
        and event.end_atom_idx is not None
    )
    closure_edges_by_id: dict[str, SouthStarTraversalClosureEdge] = {}
    closure_endpoints = []
    for event in events:
        if event.ring_closure is None:
            continue
        if event.edge is None or event.begin_atom_idx is None:
            raise AssertionError("closure event must carry edge and begin atom")
        partner_atom_idx = event.end_atom_idx
        if partner_atom_idx is None:
            raise AssertionError("closure event must carry partner atom")
        closure = event.ring_closure
        closure_edges_by_id[closure.closure_id] = SouthStarTraversalClosureEdge(
            edge=normalized_edge(event.edge),
            closure_id=closure.closure_id,
            label=closure.label,
        )
        closure_endpoints.append(
            SouthStarTraversalClosureEndpoint(
                closure_id=closure.closure_id,
                edge=normalized_edge(event.edge),
                atom_idx=event.begin_atom_idx,
                partner_atom_idx=partner_atom_idx,
                begin_parent_idx=event.begin_parent_idx,
                role=closure.role,
                label=closure.label,
                syntax_position=event.kind,
            )
        )
    return SouthStarConnectedGraphTraversalPlan(
        root_atom_idx=root_atom_idx,
        atom_order=atom_order,
        tree_edges=tree_edges,
        closure_edges=tuple(closure_edges_by_id.values()),
        closure_endpoints=tuple(closure_endpoints),
    )


if __name__ == "__main__":
    unittest.main()
