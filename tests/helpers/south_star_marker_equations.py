from __future__ import annotations

from dataclasses import dataclass

from tests.helpers.south_star_annotation_policy import Edge
from tests.helpers.south_star_annotation_policy import normalized_edge
from tests.helpers.south_star_component_support_state import (
    SouthStarComponentSupportState,
)
from tests.helpers.south_star_components import SouthStarSemanticStereoComponent
from tests.helpers.south_star_components import SouthStarSourceStereoFeature
from tests.helpers.south_star_enum_s import SouthStarMarkerSlotAssignment
from tests.helpers.south_star_enum_s import SouthStarTreeTraversal
from tests.helpers.south_star_enum_s import mol_to_smiles_enum_s_tree_traversals_for_case
from tests.helpers.south_star_semantic_oracle import parse_smiles
from tests.helpers.south_star_semantics import SouthStarSemanticCase


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


def marker_slot_parity_equations_for_case(
    case: SouthStarSemanticCase,
) -> tuple[tuple[SouthStarMarkerSlotParityEquation, ...], ...]:
    mol = parse_smiles(case.source_smiles)
    state = SouthStarComponentSupportState.from_mol(mol)
    return tuple(
        marker_slot_parity_equations_for_traversal(state, traversal)
        for traversal in mol_to_smiles_enum_s_tree_traversals_for_case(case)
    )


def marker_slot_parity_equations_for_traversal(
    state: SouthStarComponentSupportState,
    traversal: SouthStarTreeTraversal,
) -> tuple[SouthStarMarkerSlotParityEquation, ...]:
    marker_by_slot = _marker_by_slot(traversal.marker_assignments)
    graph_marker_by_edge = _graph_marker_by_edge(traversal)
    components_by_edge = _components_by_edge(state)

    equations = []
    for event in traversal.events:
        if event.marker_slot is None:
            continue
        slot = event.marker_slot
        edge = normalized_edge(slot.edge)
        graph_marker = graph_marker_by_edge[edge]
        emitted_marker = marker_by_slot[slot.slot_id]
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
                traversal_orientation_flip=graph_marker != emitted_marker,
                component_ids=component_ids,
                feature_terms=feature_terms,
            )
        )
    return tuple(equations)


def _marker_by_slot(
    marker_assignments: tuple[SouthStarMarkerSlotAssignment, ...],
) -> dict[str, str]:
    marker_by_slot = {}
    for assignment in marker_assignments:
        if assignment.slot_id in marker_by_slot:
            raise ValueError(
                f"duplicate marker assignment for slot {assignment.slot_id!r}"
            )
        marker_by_slot[assignment.slot_id] = assignment.marker
    return marker_by_slot


def _graph_marker_by_edge(traversal: SouthStarTreeTraversal) -> dict[Edge, str]:
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
