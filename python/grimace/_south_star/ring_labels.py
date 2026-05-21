from __future__ import annotations

from dataclasses import dataclass

from grimace._south_star.annotation_policy import Edge, normalized_edge


_SUPPORTED_RING_CLOSURE_LABELS: tuple[str, ...] = (
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
)


@dataclass(frozen=True, slots=True)
class SouthStarRingClosureLabelAssignment:
    closure_id: str
    edge: Edge
    label: str


class SouthStarFirstEncounterRingClosureLabelPolicy:
    name = "first_encounter_ring_closure_labels"

    def assignments_for_edges(
        self,
        closure_edges: tuple[Edge, ...],
    ) -> tuple[SouthStarRingClosureLabelAssignment, ...]:
        normalized_edges = tuple(normalized_edge(edge) for edge in closure_edges)
        if len(set(normalized_edges)) != len(normalized_edges):
            raise ValueError("ring closure label policy requires unique edges")
        if len(normalized_edges) > len(_SUPPORTED_RING_CLOSURE_LABELS):
            raise NotImplementedError(
                "South Star ring closure label policy currently supports at most "
                f"{len(_SUPPORTED_RING_CLOSURE_LABELS)} simultaneous closure edges"
            )
        return tuple(
            SouthStarRingClosureLabelAssignment(
                closure_id=closure_id_for_edge(edge),
                edge=edge,
                label=_SUPPORTED_RING_CLOSURE_LABELS[position],
            )
            for position, edge in enumerate(normalized_edges)
        )


def closure_id_for_edge(edge: Edge) -> str:
    begin_atom_idx, end_atom_idx = normalized_edge(edge)
    return f"{begin_atom_idx}-{end_atom_idx}"


DEFAULT_RING_CLOSURE_LABEL_POLICY = SouthStarFirstEncounterRingClosureLabelPolicy()
