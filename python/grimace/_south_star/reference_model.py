from __future__ import annotations

from dataclasses import dataclass

from grimace._south_star.annotation_policy import Edge


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
class SouthStarTraversalEvent:
    kind: str
    text: str
    atom_idx: int | None = None
    edge: Edge | None = None
    begin_atom_idx: int | None = None
    end_atom_idx: int | None = None
    begin_parent_idx: int | None = None
    marker_slot: SouthStarMarkerSlot | None = None
    ring_closure: SouthStarRingClosure | None = None


@dataclass(frozen=True, slots=True)
class SouthStarTraversalFragment:
    events: tuple[SouthStarTraversalEvent, ...]


@dataclass(frozen=True, slots=True)
class SouthStarTraversal:
    root_atom_idx: int
    events: tuple[SouthStarTraversalEvent, ...]
    marker_assignments: tuple[SouthStarMarkerSlotAssignment, ...]
    component_marker_assignments: tuple[object, ...]
