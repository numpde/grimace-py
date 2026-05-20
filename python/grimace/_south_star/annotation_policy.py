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


@dataclass(frozen=True, slots=True)
class AnnotationPolicyCandidate:
    name: str
    status: str
    description: str


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


class NoMarkerAnnotationPolicyStub:
    """Diagnostic policy stub used to exercise policy modularity.

    This is not a proposed package policy. It proves callers can swap the
    annotation-policy layer without changing molecule facts, component
    extraction, traversal events, or marker-equation solving.
    """

    name: str = "no_marker_policy_stub"

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


SOUTH_STAR_ANNOTATION_POLICY_CANDIDATES: tuple[AnnotationPolicyCandidate, ...] = (
    AnnotationPolicyCandidate(
        name=MaximalEligibleCarrierAnnotationPolicy.name,
        status="default",
        description=(
            "Emit markers for every eligible carrier that can express a "
            "surviving semantic assignment."
        ),
    ),
    AnnotationPolicyCandidate(
        name="minimal_sufficient",
        status="deferred_candidate",
        description=(
            "Emit a sufficient subset of markers while preserving the same "
            "semantic assignment set."
        ),
    ),
    AnnotationPolicyCandidate(
        name="canonical_semantic",
        status="deferred_candidate",
        description=(
            "Choose one deterministic semantic spelling from the policy space "
            "without importing RDKit writer behavior."
        ),
    ),
    AnnotationPolicyCandidate(
        name="rdkit_writer_like",
        status="comparison_candidate",
        description=(
            "Model RDKit-like marker placement as an explicit comparison policy, "
            "not as South Star semantic authority."
        ),
    ),
    AnnotationPolicyCandidate(
        name=NoMarkerAnnotationPolicyStub.name,
        status="test_stub",
        description="Exercise policy injection without proposing public behavior.",
    ),
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
