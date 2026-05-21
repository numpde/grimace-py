from __future__ import annotations

from dataclasses import dataclass

from grimace._south_star.annotation_policy import Edge
from grimace._south_star.constraint_vocabulary import SouthStarRendererInput


@dataclass(frozen=True, slots=True)
class SouthStarCarrierContext:
    center_atom_idx: int
    double_neighbor_idx: int


@dataclass(frozen=True, slots=True)
class SouthStarMarkerSlot:
    slot_id: str
    edge: Edge
    begin_atom_idx: int
    end_atom_idx: int
    begin_parent_idx: int | None
    syntax_position: str
    adjacent_contexts: tuple[SouthStarCarrierContext, ...]


@dataclass(frozen=True, slots=True)
class SouthStarMarkerSlotAssignment:
    slot_id: str
    marker: str


@dataclass(frozen=True, slots=True)
class SouthStarRingClosure:
    closure_id: str
    label: str
    role: str


@dataclass(frozen=True, slots=True)
class SouthStarTraversalTreeEdge:
    edge: Edge
    begin_atom_idx: int
    end_atom_idx: int
    begin_parent_idx: int | None
    syntax_position: str


@dataclass(frozen=True, slots=True)
class SouthStarTraversalClosureEdge:
    edge: Edge
    closure_id: str
    label: str


@dataclass(frozen=True, slots=True)
class SouthStarTraversalClosureEndpoint:
    closure_id: str
    edge: Edge
    atom_idx: int
    partner_atom_idx: int
    begin_parent_idx: int | None
    role: str
    label: str
    syntax_position: str


@dataclass(frozen=True, slots=True)
class SouthStarConnectedGraphTraversalPlan:
    root_atom_idx: int
    atom_order: tuple[int, ...]
    tree_edges: tuple[SouthStarTraversalTreeEdge, ...]
    closure_edges: tuple[SouthStarTraversalClosureEdge, ...]
    closure_endpoints: tuple[SouthStarTraversalClosureEndpoint, ...]


@dataclass(frozen=True, slots=True)
class SouthStarTraversalEvent:
    kind: str
    text: str
    atom_idx: int | None = None
    edge: Edge | None = None
    begin_atom_idx: int | None = None
    end_atom_idx: int | None = None
    begin_parent_idx: int | None = None
    syntax_position: str = ""
    marker_slot: SouthStarMarkerSlot | None = None
    ring_closure: SouthStarRingClosure | None = None
    renderer_input: SouthStarRendererInput | None = None


@dataclass(frozen=True, slots=True)
class SouthStarTraversalFragment:
    events: tuple[SouthStarTraversalEvent, ...]


@dataclass(frozen=True, slots=True)
class SouthStarTraversal:
    root_atom_idx: int
    events: tuple[SouthStarTraversalEvent, ...]
    marker_assignments: tuple[SouthStarMarkerSlotAssignment, ...]
    component_marker_assignments: tuple[object, ...]
