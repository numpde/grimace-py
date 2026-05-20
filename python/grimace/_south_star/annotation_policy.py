from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol


DIRECTIONAL_MARKERS: tuple[str, str] = ("/", "\\")
Edge = tuple[int, int]
MarkerOptionsByEdge = Mapping[Edge, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class SemanticCarrierOpportunity:
    edge: Edge


@dataclass(frozen=True, slots=True)
class EmittedEdgeBasis:
    edge: Edge


@dataclass(frozen=True, slots=True)
class SurvivingSemanticAssignment:
    assignment_id: str
    marker_options_by_edge: MarkerOptionsByEdge


@dataclass(frozen=True, slots=True)
class AnnotationPolicyDecision:
    edge: Edge
    marker_required: bool
    allowed_markers: tuple[str, ...]


class AnnotationPolicy(Protocol):
    name: str

    def decision(
        self,
        *,
        carrier_opportunities: tuple[SemanticCarrierOpportunity, ...],
        emitted_edge: EmittedEdgeBasis,
        surviving_assignments: tuple[SurvivingSemanticAssignment, ...],
    ) -> AnnotationPolicyDecision:
        ...


class MaximalEligibleCarrierAnnotationPolicy:
    name: str = "maximal_eligible_carrier"

    def decision(
        self,
        *,
        carrier_opportunities: tuple[SemanticCarrierOpportunity, ...],
        emitted_edge: EmittedEdgeBasis,
        surviving_assignments: tuple[SurvivingSemanticAssignment, ...],
    ) -> AnnotationPolicyDecision:
        if not surviving_assignments:
            raise ValueError("annotation policy requires at least one survivor")
        _validate_surviving_assignments(surviving_assignments)

        edge = normalized_edge(emitted_edge.edge)
        eligible_edges = {
            normalized_edge(opportunity.edge) for opportunity in carrier_opportunities
        }
        if edge not in eligible_edges:
            return AnnotationPolicyDecision(
                edge=edge,
                marker_required=False,
                allowed_markers=(),
            )

        allowed_markers = tuple(
            marker
            for marker in DIRECTIONAL_MARKERS
            if any(
                marker in assignment.marker_options_by_edge.get(edge, ())
                for assignment in surviving_assignments
            )
        )
        if not allowed_markers:
            raise ValueError(
                f"eligible carrier edge {edge!r} has no surviving marker option"
            )

        return AnnotationPolicyDecision(
            edge=edge,
            marker_required=True,
            allowed_markers=allowed_markers,
        )


def normalized_edge(edge: Edge) -> Edge:
    begin_atom_idx, end_atom_idx = edge
    if begin_atom_idx < 0 or end_atom_idx < 0:
        raise ValueError(f"edge atom indices must be nonnegative: {edge!r}")
    if begin_atom_idx == end_atom_idx:
        raise ValueError(f"edge cannot be a self-edge: {edge!r}")
    if begin_atom_idx < end_atom_idx:
        return (begin_atom_idx, end_atom_idx)
    return (end_atom_idx, begin_atom_idx)


def _validate_surviving_assignments(
    surviving_assignments: tuple[SurvivingSemanticAssignment, ...],
) -> None:
    for assignment in surviving_assignments:
        if not assignment.assignment_id:
            raise ValueError("surviving semantic assignment id must be nonempty")
        for edge, marker_options in assignment.marker_options_by_edge.items():
            normalized_edge(edge)
            if not marker_options:
                raise ValueError(
                    f"surviving assignment {assignment.assignment_id!r} has empty "
                    f"marker options for edge {edge!r}"
                )
            invalid_markers = tuple(
                marker for marker in marker_options if marker not in DIRECTIONAL_MARKERS
            )
            if invalid_markers:
                raise ValueError(
                    f"surviving assignment {assignment.assignment_id!r} has invalid "
                    f"marker options {invalid_markers!r} for edge {edge!r}"
                )
