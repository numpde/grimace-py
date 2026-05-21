from __future__ import annotations

from dataclasses import dataclass

from grimace._south_star.annotation_policy import Edge
from grimace._south_star.annotation_policy import normalized_edge
from grimace._south_star.component_support_state import (
    SouthStarComponentSupportState,
)
from grimace._south_star.components import SouthStarSemanticStereoComponent
from grimace._south_star.components import SouthStarSourceStereoFeature
from grimace._south_star.constraint_vocabulary import SouthStarConstraintEquation
from grimace._south_star.constraint_vocabulary import SouthStarConstraintFamily
from grimace._south_star.constraint_vocabulary import SouthStarConstraintObligation
from grimace._south_star.constraint_vocabulary import SouthStarConstraintSyntaxSlot
from grimace._south_star.reference_model import SouthStarMarkerSlot
from grimace._south_star.reference_model import SouthStarTraversal


DIRECTIONAL_MARKER_CONSTRAINT_FAMILY = SouthStarConstraintFamily(
    family_id="directional_double_bond_marker",
    description="Directional marker slots constrained by double-bond stereo facts.",
)


@dataclass(frozen=True, slots=True)
class SouthStarFeatureCarrierTerm:
    component_id: str
    feature_id: str
    central_bond: Edge
    carrier_side: str
    source_marker: str
    required_stereo_phase: str
    shared_carrier_incidence_count: int


@dataclass(frozen=True, slots=True)
class SouthStarMarkerSlotParityEquation:
    equation_id: str
    slot_id: str
    edge: Edge
    syntax_position: str
    begin_atom_idx: int
    end_atom_idx: int
    begin_parent_idx: int | None
    graph_marker: str
    emitted_marker: str
    traversal_orientation_flip: bool
    component_ids: tuple[str, ...]
    feature_terms: tuple[SouthStarFeatureCarrierTerm, ...]


@dataclass(frozen=True, slots=True)
class SouthStarDirectionalMarkerConstraintRecords:
    syntax_slot: SouthStarConstraintSyntaxSlot
    obligations: tuple[SouthStarConstraintObligation, ...]
    equation: SouthStarConstraintEquation


def marker_slot_parity_equations_for_traversal(
    state: SouthStarComponentSupportState,
    traversal: SouthStarTraversal,
) -> tuple[SouthStarMarkerSlotParityEquation, ...]:
    graph_marker_by_edge = _graph_marker_by_edge(traversal)
    components_by_edge = _components_by_edge(state)
    ring_stereo_closure_open_atom_by_edge = (
        _ring_stereo_closure_open_atom_by_edge(traversal)
    )

    equations = []
    for event in traversal.events:
        if event.marker_slot is None:
            continue
        slot = event.marker_slot
        edge = normalized_edge(slot.edge)
        graph_marker = graph_marker_by_edge[edge]
        traversal_orientation_flip = _traversal_orientation_flip(
            slot,
            ring_stereo_closure_open_atom_by_edge=(
                ring_stereo_closure_open_atom_by_edge
            ),
        )
        emitted_marker = _marker_with_orientation(
            graph_marker,
            traversal_orientation_flip=traversal_orientation_flip,
        )
        feature_terms = _feature_terms_for_edge(
            components_by_edge.get(edge, ()),
            edge=edge,
        )
        component_ids = tuple(
            dict.fromkeys(term.component_id for term in feature_terms)
        )
        equations.append(
            SouthStarMarkerSlotParityEquation(
                equation_id=_equation_id(
                    slot_id=slot.slot_id,
                    component_ids=component_ids,
                ),
                slot_id=slot.slot_id,
                edge=edge,
                syntax_position=slot.syntax_position,
                begin_atom_idx=slot.begin_atom_idx,
                end_atom_idx=slot.end_atom_idx,
                begin_parent_idx=slot.begin_parent_idx,
                graph_marker=graph_marker,
                emitted_marker=emitted_marker,
                traversal_orientation_flip=traversal_orientation_flip,
                component_ids=component_ids,
                feature_terms=feature_terms,
            )
        )
    return tuple(equations)


def expected_marker_from_equation(
    equation: SouthStarMarkerSlotParityEquation,
) -> str:
    if equation.traversal_orientation_flip:
        return _flipped_marker(equation.graph_marker)
    return equation.graph_marker


def directional_marker_constraint_records_for_equation(
    equation: SouthStarMarkerSlotParityEquation,
) -> SouthStarDirectionalMarkerConstraintRecords:
    obligations = tuple(
        constraint_obligation_for_feature_carrier_term(
            term,
            slot_id=equation.slot_id,
            carrier_edge=equation.edge,
        )
        for term in equation.feature_terms
    )
    return SouthStarDirectionalMarkerConstraintRecords(
        syntax_slot=constraint_syntax_slot_for_marker_equation(equation),
        obligations=obligations,
        equation=SouthStarConstraintEquation(
            family_id=DIRECTIONAL_MARKER_CONSTRAINT_FAMILY.family_id,
            equation_id=equation.equation_id,
            obligation_ids=tuple(
                obligation.obligation_id for obligation in obligations
            ),
            syntax_slot_ids=(equation.slot_id,),
        ),
    )


def constraint_syntax_slot_for_marker_equation(
    equation: SouthStarMarkerSlotParityEquation,
) -> SouthStarConstraintSyntaxSlot:
    return SouthStarConstraintSyntaxSlot(
        family_id=DIRECTIONAL_MARKER_CONSTRAINT_FAMILY.family_id,
        slot_id=equation.slot_id,
        slot_kind="directional_marker",
        syntax_position=equation.syntax_position,
        edge=equation.edge,
    )


def constraint_equation_for_marker_equation(
    equation: SouthStarMarkerSlotParityEquation,
) -> SouthStarConstraintEquation:
    return directional_marker_constraint_records_for_equation(equation).equation


def constraint_obligation_for_feature_carrier_term(
    term: SouthStarFeatureCarrierTerm,
    *,
    slot_id: str,
    carrier_edge: Edge,
) -> SouthStarConstraintObligation:
    return SouthStarConstraintObligation(
        family_id=DIRECTIONAL_MARKER_CONSTRAINT_FAMILY.family_id,
        obligation_id=(
            f"directional_marker:{term.component_id}:{term.feature_id}:{slot_id}"
        ),
        subject_id=_edge_subject_id(carrier_edge),
        required_fact_ids=(
            f"component:{term.component_id}",
            f"feature:{term.feature_id}",
            f"central_bond:{_edge_text(term.central_bond)}",
            f"carrier_side:{term.carrier_side}",
            f"source_marker:{term.source_marker}",
            f"required_stereo_phase:{term.required_stereo_phase}",
        ),
        syntax_slot_ids=(slot_id,),
    )


def _edge_subject_id(edge: Edge) -> str:
    return f"edge:{_edge_text(edge)}"


def _edge_text(edge: Edge) -> str:
    return f"{edge[0]}-{edge[1]}"


def _graph_marker_by_edge(traversal: SouthStarTraversal) -> dict[Edge, str]:
    marker_by_edge: dict[Edge, str] = {}
    for assignment in traversal.component_marker_assignments:
        for edge, marker in assignment.marker_by_edge:
            normalized = normalized_edge(edge)
            existing = marker_by_edge.setdefault(normalized, marker)
            if existing != marker:
                raise ValueError(
                    f"conflicting graph marker assignments for edge {normalized!r}"
                )
    return marker_by_edge


def _components_by_edge(
    state: SouthStarComponentSupportState,
) -> dict[Edge, tuple[SouthStarSemanticStereoComponent, ...]]:
    grouped: dict[Edge, list[SouthStarSemanticStereoComponent]] = {}
    for component in state.components:
        for edge in component.eligible_carrier_edges:
            grouped.setdefault(edge, []).append(component)
    return {edge: tuple(components) for edge, components in grouped.items()}


def _feature_terms_for_edge(
    components: tuple[SouthStarSemanticStereoComponent, ...],
    *,
    edge: Edge,
) -> tuple[SouthStarFeatureCarrierTerm, ...]:
    terms = []
    for component in components:
        shared_carrier_incidence_count = sum(
            edge in feature.eligible_carrier_edges
            for feature in component.source_features
        )
        for feature in component.source_features:
            side = _feature_carrier_side(feature, edge=edge)
            if side is None:
                continue
            source_marker_by_edge = dict(feature.source_marker_by_edge)
            terms.append(
                SouthStarFeatureCarrierTerm(
                    component_id=component.component_id,
                    feature_id=feature.feature_id,
                    central_bond=feature.central_bond,
                    carrier_side=side,
                    source_marker=source_marker_by_edge[edge],
                    required_stereo_phase=feature.rdkit_stereo,
                    shared_carrier_incidence_count=shared_carrier_incidence_count,
                )
            )
    if not terms:
        raise ValueError(f"marker edge {edge!r} has no feature-carrier terms")
    return tuple(terms)


def _feature_carrier_side(
    feature: SouthStarSourceStereoFeature,
    *,
    edge: Edge,
) -> str | None:
    if edge in feature.left_carrier_edges:
        return "left"
    if edge in feature.right_carrier_edges:
        return "right"
    return None


def _equation_id(*, slot_id: str, component_ids: tuple[str, ...]) -> str:
    return f"{slot_id}:{','.join(component_ids)}"


def _ring_stereo_closure_open_atom_by_edge(
    traversal: SouthStarTraversal,
) -> dict[Edge, int]:
    return {
        normalized_edge(event.edge): event.begin_atom_idx
        for event in traversal.events
        if event.kind == "ring_open"
        and event.ring_closure is not None
        and event.edge is not None
        and event.begin_atom_idx is not None
        and event.text == "="
    }


def _traversal_orientation_flip(
    slot: SouthStarMarkerSlot,
    *,
    ring_stereo_closure_open_atom_by_edge: dict[Edge, int],
) -> bool:
    if slot.syntax_position == "ring_open":
        return _ring_open_orientation_flip(slot)

    flip = False
    for context in slot.adjacent_contexts:
        if (
            slot.begin_atom_idx == context.center_atom_idx
            and slot.end_atom_idx != context.double_neighbor_idx
            and slot.begin_parent_idx != context.double_neighbor_idx
        ):
            flip = not flip
        central_edge = normalized_edge(
            (context.center_atom_idx, context.double_neighbor_idx)
        )
        open_atom_idx = ring_stereo_closure_open_atom_by_edge.get(central_edge)
        # A stereo double bond rendered as a ring closure reverses exactly one
        # endpoint's local marker phase relative to the tree-edge carrier basis.
        if open_atom_idx == context.center_atom_idx:
            flip = not flip
    return flip


def _ring_open_orientation_flip(slot: SouthStarMarkerSlot) -> bool:
    flip = False
    for context in slot.adjacent_contexts:
        if (
            slot.end_atom_idx == context.center_atom_idx
            and slot.begin_atom_idx != context.double_neighbor_idx
        ):
            flip = not flip
        elif (
            slot.begin_atom_idx == context.center_atom_idx
            and slot.end_atom_idx != context.double_neighbor_idx
            and slot.begin_parent_idx != context.double_neighbor_idx
        ):
            flip = not flip
    return flip


def _marker_with_orientation(
    graph_marker: str,
    *,
    traversal_orientation_flip: bool,
) -> str:
    if traversal_orientation_flip:
        return _flipped_marker(graph_marker)
    return graph_marker


def _flipped_marker(marker: str) -> str:
    if marker == "/":
        return "\\"
    if marker == "\\":
        return "/"
    raise ValueError(f"unsupported South Star directional marker {marker!r}")
