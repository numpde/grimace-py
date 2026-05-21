from __future__ import annotations

from grimace._south_star.annotation_policy import normalized_edge
from grimace._south_star.reference_model import (
    SouthStarConnectedGraphTraversalPlan,
)
from grimace._south_star.reference_model import SouthStarTraversalClosureEdge
from grimace._south_star.reference_model import SouthStarTraversalClosureEndpoint
from grimace._south_star.reference_model import SouthStarTraversalEvent
from grimace._south_star.reference_model import SouthStarTraversalTreeEdge


def connected_graph_plan_from_events(
    *,
    root_atom_idx: int,
    events: tuple[SouthStarTraversalEvent, ...],
) -> SouthStarConnectedGraphTraversalPlan:
    closure_edges_by_id: dict[str, SouthStarTraversalClosureEdge] = {}
    closure_endpoints = []
    for event in events:
        if event.ring_closure is None:
            continue
        if event.edge is None or event.begin_atom_idx is None:
            raise ValueError("closure event must carry edge and begin atom")
        if event.end_atom_idx is None:
            raise ValueError("closure event must carry partner atom")
        closure = event.ring_closure
        edge = normalized_edge(event.edge)
        closure_edge = SouthStarTraversalClosureEdge(
            edge=edge,
            closure_id=closure.closure_id,
            label=closure.label,
        )
        existing = closure_edges_by_id.setdefault(
            closure.closure_id,
            closure_edge,
        )
        if existing != closure_edge:
            raise ValueError(
                f"closure id {closure.closure_id!r} has inconsistent edge or label"
            )
        closure_endpoints.append(
            SouthStarTraversalClosureEndpoint(
                closure_id=closure.closure_id,
                edge=edge,
                atom_idx=event.begin_atom_idx,
                partner_atom_idx=event.end_atom_idx,
                begin_parent_idx=event.begin_parent_idx,
                role=closure.role,
                label=closure.label,
                syntax_position=event.kind,
            )
        )
    _validate_closure_endpoint_pairs(tuple(closure_endpoints))
    return SouthStarConnectedGraphTraversalPlan(
        root_atom_idx=root_atom_idx,
        atom_order=tuple(
            event.atom_idx
            for event in events
            if event.kind == "atom" and event.atom_idx is not None
        ),
        tree_edges=tuple(
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
        ),
        closure_edges=tuple(closure_edges_by_id.values()),
        closure_endpoints=tuple(closure_endpoints),
    )


def _validate_closure_endpoint_pairs(
    closure_endpoints: tuple[SouthStarTraversalClosureEndpoint, ...],
) -> None:
    endpoints_by_id: dict[str, list[SouthStarTraversalClosureEndpoint]] = {}
    for endpoint in closure_endpoints:
        endpoints_by_id.setdefault(endpoint.closure_id, []).append(endpoint)
    for closure_id, endpoints in endpoints_by_id.items():
        if len(endpoints) != 2:
            raise ValueError(
                f"closure id {closure_id!r} must have exactly two endpoints"
            )
        if {endpoint.role for endpoint in endpoints} != {"open", "close"}:
            raise ValueError(
                f"closure id {closure_id!r} must have open and close endpoints"
            )
        if len({endpoint.label for endpoint in endpoints}) != 1:
            raise ValueError(
                f"closure id {closure_id!r} has inconsistent endpoint labels"
            )
        if len({endpoint.edge for endpoint in endpoints}) != 1:
            raise ValueError(
                f"closure id {closure_id!r} has inconsistent endpoint edges"
            )
