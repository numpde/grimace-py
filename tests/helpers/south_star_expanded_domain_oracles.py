from __future__ import annotations

"""Independent test oracles for expanded South Star fixture domains.

These helpers intentionally keep independent traversal searches and renderers
instead of calling `grimace._south_star.enum_s`. They consume the shared South
Star reference record vocabulary where possible so they remain witnesses, not
separate mini-worlds with their own event/slot types.
"""

from dataclasses import dataclass
from itertools import permutations
from itertools import product

from rdkit import Chem

from grimace._south_star.annotation_policy import Edge
from grimace._south_star.annotation_policy import normalized_edge
from grimace._south_star.fragments import SouthStarDisconnectedCompositionResult
from grimace._south_star.fragments import SouthStarFragmentSupport
from grimace._south_star.fragments import compose_disconnected_fragment_supports
from grimace._south_star.reference_model import SouthStarCarrierContext
from grimace._south_star.reference_model import SouthStarMarkerSlot
from grimace._south_star.reference_model import SouthStarTraversalEvent
from grimace._south_star.reference_model import SouthStarTraversalFragment
from grimace._south_star.tetrahedral import IMPLICIT_HYDROGEN_LIGAND
from grimace._south_star.tetrahedral import extract_tetrahedral_center_facts
from grimace._south_star.tetrahedral import preserving_tetrahedral_token
from tests.helpers.south_star_exact_support import (
    SouthStarExpandedSupportCase,
    load_south_star_exact_first_domain_cases,
    load_south_star_expanded_support_cases,
)
from tests.helpers.south_star_semantic_oracle import parse_smiles


@dataclass(frozen=True, slots=True)
class SouthStarRingStereoOracleEquation:
    """Witness-only projection of shared marker-slot equation fields."""

    equation_id: str
    slot_id: str
    edge: Edge
    syntax_position: str
    graph_marker: str
    emitted_marker: str
    traversal_orientation_flip: bool


@dataclass(frozen=True, slots=True)
class SouthStarRingStereoOracleResult:
    outputs: tuple[str, ...]
    equations: tuple[SouthStarRingStereoOracleEquation, ...]
    closure_edge_count: int
    marker_assignment_count: int


def independent_saturated_monocycle_support_for_case(
    case: SouthStarExpandedSupportCase,
) -> tuple[str, ...]:
    """Enumerate saturated-monocycle support without using EnumS traversal."""
    mol = parse_smiles(case.source_smiles)
    _assert_saturated_monocycle_domain(mol)

    outputs = []
    for root_idx in range(mol.GetNumAtoms()):
        for closure_edge in _single_ring_edges(mol):
            for fragment in _tree_fragments_with_blocked_edge(
                mol,
                atom_idx=root_idx,
                parent_idx=None,
                visited=frozenset(),
                blocked_edge=closure_edge,
            ):
                outputs.append(_render_with_ring_digit(fragment, closure_edge))
    return tuple(dict.fromkeys(outputs))


def independent_disconnected_composition_support_for_case(
    case: SouthStarExpandedSupportCase,
) -> SouthStarDisconnectedCompositionResult:
    fragment_supports = _independent_fragment_supports_for_case(case)
    return compose_disconnected_fragment_supports(
        tuple(
            SouthStarFragmentSupport(
                fragment_id=f"fragment:{fragment_idx}",
                outputs=support,
            )
            for fragment_idx, support in enumerate(fragment_supports)
        )
    )


def independent_ring_stereo_monocycle_support_for_case(
    case: SouthStarExpandedSupportCase,
) -> SouthStarRingStereoOracleResult:
    """Check the first ring-stereo domain without using EnumS traversal.

    This oracle intentionally has its own traversal, marker-slot, and parity
    equation model. It is test evidence for the fixture, not package runtime.
    """
    mol = parse_smiles(case.source_smiles)
    marker_by_edge = _ring_stereo_source_markers_by_edge(mol)
    carrier_contexts_by_edge = _ring_stereo_carrier_contexts_by_edge(
        mol,
        marker_by_edge=marker_by_edge,
    )
    closure_edges = _ring_stereo_supported_ring_closure_edges(mol)
    marker_assignments = (
        marker_by_edge,
        {edge: _flipped_marker(marker) for edge, marker in marker_by_edge.items()},
    )

    rendered = []
    equations = []
    for marker_assignment in marker_assignments:
        for root_idx in range(mol.GetNumAtoms()):
            for closure_edge in closure_edges:
                for fragment in _ring_stereo_tree_fragments(
                    mol,
                    atom_idx=root_idx,
                    parent_idx=None,
                    visited=frozenset(),
                    blocked_edge=closure_edge,
                    marker_by_edge=marker_assignment,
                    carrier_contexts_by_edge=carrier_contexts_by_edge,
                ):
                    events = _ring_stereo_with_closure_events(
                        mol,
                        fragment.events,
                        closure_edge=closure_edge,
                        marker_by_edge=marker_assignment,
                        carrier_contexts_by_edge=carrier_contexts_by_edge,
                    )
                    marker_by_slot, event_equations = (
                        _ring_stereo_marker_assignments_for_events(
                            events,
                            marker_by_edge=marker_assignment,
                        )
                    )
                    rendered.append(
                        _render_ring_stereo_events(
                            events,
                            marker_by_slot=marker_by_slot,
                        )
                    )
                    equations.extend(event_equations)

    return SouthStarRingStereoOracleResult(
        outputs=tuple(dict.fromkeys(rendered)),
        equations=tuple(dict.fromkeys(equations)),
        closure_edge_count=len(closure_edges),
        marker_assignment_count=len(marker_assignments),
    )


def independent_tetrahedral_atom_stereo_support_for_case(
    case: SouthStarExpandedSupportCase,
) -> tuple[str, ...]:
    """Enumerate current star-shaped tetrahedral support independently.

    This oracle is intentionally limited to one tetrahedral center whose
    explicit ligands are terminal atoms. It derives `@`/`@@` from emitted ligand
    order and does not call EnumS traversal or rendering.
    """
    mol = parse_smiles(case.source_smiles)
    fact = _single_star_tetrahedral_fact(mol)
    center_idx = fact.center_atom_idx
    neighbor_indices = fact.explicit_neighbor_atom_indices

    outputs: list[str] = []
    for root_idx in range(mol.GetNumAtoms()):
        if root_idx == center_idx:
            for ordered_neighbors in permutations(neighbor_indices):
                outputs.append(
                    _render_tetrahedral_center_root(
                        mol,
                        center_idx=center_idx,
                        ordered_neighbors=ordered_neighbors,
                    )
                )
            continue
        if root_idx not in neighbor_indices:
            raise NotImplementedError("tetrahedral oracle requires a star graph")
        child_indices = tuple(idx for idx in neighbor_indices if idx != root_idx)
        for ordered_children in permutations(child_indices):
            outputs.append(
                _render_tetrahedral_ligand_root(
                    mol,
                    root_idx=root_idx,
                    center_idx=center_idx,
                    ordered_children=ordered_children,
                )
            )
    return tuple(dict.fromkeys(outputs))


def _independent_fragment_supports_for_case(
    case: SouthStarExpandedSupportCase,
) -> tuple[tuple[str, ...], ...]:
    if case.case_id == "markerless_disconnected_ring_and_atom":
        return (
            _expanded_support("simple_saturated_monocycle_cyclohexane"),
            ("O",),
        )
    if case.case_id == "disconnected_stereo_fragment_and_atom":
        return (
            _exact_first_domain_support("isolated_alkene_z"),
            ("O",),
        )
    raise NotImplementedError(
        f"no disconnected-composition oracle fragment supports for {case.case_id!r}"
    )


def _single_star_tetrahedral_fact(mol: Chem.Mol):
    facts = extract_tetrahedral_center_facts(mol)
    if len(facts) != 1:
        raise NotImplementedError("tetrahedral oracle requires exactly one center")
    fact = facts[0]
    center_idx = fact.center_atom_idx
    for neighbor_idx in fact.explicit_neighbor_atom_indices:
        neighbor = mol.GetAtomWithIdx(neighbor_idx)
        if neighbor.GetDegree() != 1:
            raise NotImplementedError(
                "tetrahedral oracle currently requires terminal explicit ligands"
            )
        if mol.GetBondBetweenAtoms(center_idx, neighbor_idx) is None:
            raise ValueError("tetrahedral ligand is not bonded to the center")
    return fact


def _render_tetrahedral_center_root(
    mol: Chem.Mol,
    *,
    center_idx: int,
    ordered_neighbors: tuple[int, ...],
) -> str:
    center_text = _tetrahedral_center_text(
        mol,
        center_idx=center_idx,
        parent_idx=None,
        ordered_children=ordered_neighbors,
    )
    return _render_center_with_ordered_ligands(
        mol,
        center_text=center_text,
        ordered_ligands=ordered_neighbors,
    )


def _render_tetrahedral_ligand_root(
    mol: Chem.Mol,
    *,
    root_idx: int,
    center_idx: int,
    ordered_children: tuple[int, ...],
) -> str:
    center_text = _tetrahedral_center_text(
        mol,
        center_idx=center_idx,
        parent_idx=root_idx,
        ordered_children=ordered_children,
    )
    return _atom_text(mol.GetAtomWithIdx(root_idx)) + _render_center_with_ordered_ligands(
        mol,
        center_text=center_text,
        ordered_ligands=ordered_children,
    )


def _render_center_with_ordered_ligands(
    mol: Chem.Mol,
    *,
    center_text: str,
    ordered_ligands: tuple[int, ...],
) -> str:
    if not ordered_ligands:
        return center_text
    branch_text = "".join(
        f"({_atom_text(mol.GetAtomWithIdx(atom_idx))})"
        for atom_idx in ordered_ligands[:-1]
    )
    return center_text + branch_text + _atom_text(
        mol.GetAtomWithIdx(ordered_ligands[-1])
    )


def _tetrahedral_center_text(
    mol: Chem.Mol,
    *,
    center_idx: int,
    parent_idx: int | None,
    ordered_children: tuple[int, ...],
) -> str:
    fact = _single_star_tetrahedral_fact(mol)
    if fact.center_atom_idx != center_idx:
        raise ValueError("tetrahedral center index mismatch")
    emitted_ligand_order = _emitted_tetrahedral_ligand_order(
        parent_idx=parent_idx,
        ordered_children=ordered_children,
        implicit_hydrogen_count=fact.implicit_hydrogen_count,
    )
    token = preserving_tetrahedral_token(
        source_token=fact.source_token,
        source_ligand_order=fact.source_ligand_order,
        emitted_ligand_order=emitted_ligand_order,
    )
    hydrogen_text = "H" if fact.implicit_hydrogen_count else ""
    return f"[{mol.GetAtomWithIdx(center_idx).GetSymbol()}{token}{hydrogen_text}]"


def _emitted_tetrahedral_ligand_order(
    *,
    parent_idx: int | None,
    ordered_children: tuple[int, ...],
    implicit_hydrogen_count: int,
) -> tuple[str, ...]:
    emitted = []
    if parent_idx is None and implicit_hydrogen_count:
        emitted.append(IMPLICIT_HYDROGEN_LIGAND)
    if parent_idx is not None:
        emitted.append(f"atom:{parent_idx}")
    emitted.extend(f"atom:{child_idx}" for child_idx in ordered_children)
    if parent_idx is not None and implicit_hydrogen_count:
        emitted.append(IMPLICIT_HYDROGEN_LIGAND)
    return tuple(emitted)


def _ring_stereo_source_markers_by_edge(mol: Chem.Mol) -> dict[Edge, str]:
    _assert_ring_stereo_monocycle_domain(mol)
    marker_by_edge: dict[Edge, str] = {}
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        if bond.GetStereo() == Chem.BondStereo.STEREONONE:
            continue
        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        for center_atom_idx, excluded_atom_idx in (
            (begin_atom_idx, end_atom_idx),
            (end_atom_idx, begin_atom_idx),
        ):
            for neighbor in mol.GetAtomWithIdx(center_atom_idx).GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx == excluded_atom_idx:
                    continue
                carrier_bond = mol.GetBondBetweenAtoms(center_atom_idx, neighbor_idx)
                if carrier_bond is None:
                    raise ValueError("neighbor relationship has no carrier bond")
                if carrier_bond.GetBondType() != Chem.BondType.SINGLE:
                    continue
                edge = normalized_edge((center_atom_idx, neighbor_idx))
                marker_by_edge[edge] = _ring_stereo_directional_marker(mol, edge=edge)
    if not marker_by_edge:
        raise NotImplementedError("ring-stereo oracle requires directional carriers")
    return marker_by_edge


def _ring_stereo_carrier_contexts_by_edge(
    mol: Chem.Mol,
    *,
    marker_by_edge: dict[Edge, str],
) -> dict[Edge, tuple[SouthStarCarrierContext, ...]]:
    return {
        edge: _ring_stereo_carrier_contexts_for_edge(mol, edge=edge)
        for edge in marker_by_edge
    }


def _ring_stereo_carrier_contexts_for_edge(
    mol: Chem.Mol,
    *,
    edge: Edge,
) -> tuple[SouthStarCarrierContext, ...]:
    contexts = []
    for center_atom_idx, substituent_atom_idx in (edge, (edge[1], edge[0])):
        for neighbor in mol.GetAtomWithIdx(center_atom_idx).GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx == substituent_atom_idx:
                continue
            bond = mol.GetBondBetweenAtoms(center_atom_idx, neighbor_idx)
            if bond is None:
                raise ValueError("neighbor relationship has no double-bond candidate")
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                contexts.append(
                    SouthStarCarrierContext(
                        center_atom_idx=center_atom_idx,
                        double_neighbor_idx=neighbor_idx,
                    )
                )
    if not contexts:
        raise NotImplementedError(
            f"ring-stereo marker edge {edge!r} is not adjacent to a double bond"
        )
    return tuple(contexts)


def _ring_stereo_supported_ring_closure_edges(mol: Chem.Mol) -> tuple[Edge, ...]:
    return tuple(
        edge
        for edge in _single_ring_edges(mol)
        if _ring_stereo_closure_edge_supported(mol, edge=edge)
    )


def _ring_stereo_closure_edge_supported(mol: Chem.Mol, *, edge: Edge) -> bool:
    bond = mol.GetBondBetweenAtoms(*edge)
    if bond is None:
        raise ValueError(f"ring closure edge {edge!r} is not a bond")
    return bond.GetBondType() in {Chem.BondType.SINGLE, Chem.BondType.DOUBLE}


def _ring_stereo_tree_fragments(
    mol: Chem.Mol,
    *,
    atom_idx: int,
    parent_idx: int | None,
    visited: frozenset[int],
    blocked_edge: Edge,
    marker_by_edge: dict[Edge, str],
    carrier_contexts_by_edge: dict[Edge, tuple[SouthStarCarrierContext, ...]],
) -> tuple[SouthStarTraversalFragment, ...]:
    visited = visited | {atom_idx}
    children = tuple(
        neighbor.GetIdx()
        for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
        if neighbor.GetIdx() != parent_idx
        and neighbor.GetIdx() not in visited
        and normalized_edge((atom_idx, neighbor.GetIdx())) != blocked_edge
    )
    if any(
        not _ring_stereo_child_edge_allowed(
            begin_atom_idx=atom_idx,
            end_atom_idx=child_idx,
            begin_parent_idx=parent_idx,
            carrier_contexts_by_edge=carrier_contexts_by_edge,
        )
        for child_idx in children
    ):
        return ()

    atom_event = SouthStarTraversalEvent(
        kind="atom",
        text=_atom_text(mol.GetAtomWithIdx(atom_idx)),
        atom_idx=atom_idx,
    )
    if not children:
        return (SouthStarTraversalFragment(events=(atom_event,)),)

    fragments = []
    for ordered_children in permutations(sorted(children)):
        branch_children = ordered_children[:-1]
        main_child = ordered_children[-1]
        branch_groups = tuple(
            _ring_stereo_branch_fragments(
                mol,
                begin_atom_idx=atom_idx,
                end_atom_idx=child_idx,
                begin_parent_idx=parent_idx,
                visited=visited,
                blocked_edge=blocked_edge,
                marker_by_edge=marker_by_edge,
                carrier_contexts_by_edge=carrier_contexts_by_edge,
            )
            for child_idx in branch_children
        )
        main_fragments = _ring_stereo_child_fragments(
            mol,
            begin_atom_idx=atom_idx,
            end_atom_idx=main_child,
            begin_parent_idx=parent_idx,
            visited=visited,
            blocked_edge=blocked_edge,
            marker_by_edge=marker_by_edge,
            carrier_contexts_by_edge=carrier_contexts_by_edge,
        )
        for branch_group in product(*branch_groups):
            branch_events = tuple(
                event for branch in branch_group for event in branch.events
            )
            for main_fragment in main_fragments:
                fragments.append(
                    SouthStarTraversalFragment(
                        events=(atom_event,) + branch_events + main_fragment.events,
                    )
                )
    return tuple(dict.fromkeys(fragments))


def _ring_stereo_branch_fragments(
    mol: Chem.Mol,
    *,
    begin_atom_idx: int,
    end_atom_idx: int,
    begin_parent_idx: int | None,
    visited: frozenset[int],
    blocked_edge: Edge,
    marker_by_edge: dict[Edge, str],
    carrier_contexts_by_edge: dict[Edge, tuple[SouthStarCarrierContext, ...]],
) -> tuple[SouthStarTraversalFragment, ...]:
    bond_event = _ring_stereo_bond_event(
        mol,
        begin_atom_idx=begin_atom_idx,
        end_atom_idx=end_atom_idx,
        begin_parent_idx=begin_parent_idx,
        syntax_position="branch",
        marker_by_edge=marker_by_edge,
        carrier_contexts_by_edge=carrier_contexts_by_edge,
    )
    return tuple(
        SouthStarTraversalFragment(
            events=(
                SouthStarTraversalEvent(kind="branch_open", text="("),
                bond_event,
            )
            + child_fragment.events
            + (SouthStarTraversalEvent(kind="branch_close", text=")"),),
        )
        for child_fragment in _ring_stereo_tree_fragments(
            mol,
            atom_idx=end_atom_idx,
            parent_idx=begin_atom_idx,
            visited=visited,
            blocked_edge=blocked_edge,
            marker_by_edge=marker_by_edge,
            carrier_contexts_by_edge=carrier_contexts_by_edge,
        )
    )


def _ring_stereo_child_fragments(
    mol: Chem.Mol,
    *,
    begin_atom_idx: int,
    end_atom_idx: int,
    begin_parent_idx: int | None,
    visited: frozenset[int],
    blocked_edge: Edge,
    marker_by_edge: dict[Edge, str],
    carrier_contexts_by_edge: dict[Edge, tuple[SouthStarCarrierContext, ...]],
) -> tuple[SouthStarTraversalFragment, ...]:
    bond_event = _ring_stereo_bond_event(
        mol,
        begin_atom_idx=begin_atom_idx,
        end_atom_idx=end_atom_idx,
        begin_parent_idx=begin_parent_idx,
        syntax_position="main",
        marker_by_edge=marker_by_edge,
        carrier_contexts_by_edge=carrier_contexts_by_edge,
    )
    return tuple(
        SouthStarTraversalFragment(events=(bond_event,) + child_fragment.events)
        for child_fragment in _ring_stereo_tree_fragments(
            mol,
            atom_idx=end_atom_idx,
            parent_idx=begin_atom_idx,
            visited=visited,
            blocked_edge=blocked_edge,
            marker_by_edge=marker_by_edge,
            carrier_contexts_by_edge=carrier_contexts_by_edge,
        )
    )


def _ring_stereo_bond_event(
    mol: Chem.Mol,
    *,
    begin_atom_idx: int,
    end_atom_idx: int,
    begin_parent_idx: int | None,
    syntax_position: str,
    marker_by_edge: dict[Edge, str],
    carrier_contexts_by_edge: dict[Edge, tuple[SouthStarCarrierContext, ...]],
) -> SouthStarTraversalEvent:
    edge = normalized_edge((begin_atom_idx, end_atom_idx))
    if edge in marker_by_edge:
        return SouthStarTraversalEvent(
            kind="bond",
            text="",
            edge=edge,
            begin_atom_idx=begin_atom_idx,
            end_atom_idx=end_atom_idx,
            begin_parent_idx=begin_parent_idx,
            marker_slot=_ring_stereo_marker_slot(
                edge=edge,
                begin_atom_idx=begin_atom_idx,
                end_atom_idx=end_atom_idx,
                begin_parent_idx=begin_parent_idx,
                syntax_position=syntax_position,
                carrier_contexts_by_edge=carrier_contexts_by_edge,
            ),
        )
    bond = mol.GetBondBetweenAtoms(begin_atom_idx, end_atom_idx)
    if bond is None:
        raise ValueError(f"edge {edge!r} is not a bond")
    return SouthStarTraversalEvent(
        kind="bond",
        text=_bond_text(mol, begin_atom_idx, end_atom_idx),
        edge=edge,
        begin_atom_idx=begin_atom_idx,
        end_atom_idx=end_atom_idx,
        begin_parent_idx=begin_parent_idx,
    )


def _ring_stereo_with_closure_events(
    mol: Chem.Mol,
    events: tuple[SouthStarTraversalEvent, ...],
    *,
    closure_edge: Edge,
    marker_by_edge: dict[Edge, str],
    carrier_contexts_by_edge: dict[Edge, tuple[SouthStarCarrierContext, ...]],
) -> tuple[SouthStarTraversalEvent, ...]:
    endpoint_positions = tuple(
        (position, event.atom_idx)
        for position, event in enumerate(events)
        if event.kind == "atom" and event.atom_idx in closure_edge
    )
    if len(endpoint_positions) != 2:
        raise ValueError(
            f"ring closure edge {closure_edge!r} endpoints are not both emitted"
        )
    (_, open_atom_idx), (_, close_atom_idx) = endpoint_positions
    parent_by_atom = {
        event.end_atom_idx: event.begin_atom_idx
        for event in events
        if event.kind == "bond"
        and event.begin_atom_idx is not None
        and event.end_atom_idx is not None
    }
    open_event = SouthStarTraversalEvent(
        kind="ring_open",
        text=_bond_text(mol, *closure_edge),
        edge=closure_edge,
        begin_atom_idx=open_atom_idx,
        end_atom_idx=close_atom_idx,
        begin_parent_idx=parent_by_atom.get(open_atom_idx),
        marker_slot=_ring_stereo_closure_marker_slot(
            closure_edge=closure_edge,
            begin_atom_idx=open_atom_idx,
            end_atom_idx=close_atom_idx,
            begin_parent_idx=parent_by_atom.get(open_atom_idx),
            marker_by_edge=marker_by_edge,
            carrier_contexts_by_edge=carrier_contexts_by_edge,
        ),
    )
    close_event = SouthStarTraversalEvent(
        kind="ring_close",
        text="",
        edge=closure_edge,
        begin_atom_idx=close_atom_idx,
        end_atom_idx=open_atom_idx,
        begin_parent_idx=parent_by_atom.get(close_atom_idx),
    )

    with_closure_events = []
    for event in events:
        with_closure_events.append(event)
        if event.kind != "atom":
            continue
        if event.atom_idx == open_atom_idx:
            with_closure_events.append(open_event)
        elif event.atom_idx == close_atom_idx:
            with_closure_events.append(close_event)
    return tuple(with_closure_events)


def _ring_stereo_closure_marker_slot(
    *,
    closure_edge: Edge,
    begin_atom_idx: int,
    end_atom_idx: int,
    begin_parent_idx: int | None,
    marker_by_edge: dict[Edge, str],
    carrier_contexts_by_edge: dict[Edge, tuple[SouthStarCarrierContext, ...]],
) -> SouthStarMarkerSlot | None:
    if closure_edge not in marker_by_edge:
        return None
    return _ring_stereo_marker_slot(
        edge=closure_edge,
        begin_atom_idx=begin_atom_idx,
        end_atom_idx=end_atom_idx,
        begin_parent_idx=begin_parent_idx,
        syntax_position="ring_open",
        carrier_contexts_by_edge=carrier_contexts_by_edge,
    )


def _ring_stereo_marker_slot(
    *,
    edge: Edge,
    begin_atom_idx: int,
    end_atom_idx: int,
    begin_parent_idx: int | None,
    syntax_position: str,
    carrier_contexts_by_edge: dict[Edge, tuple[SouthStarCarrierContext, ...]],
) -> SouthStarMarkerSlot:
    parent = "root" if begin_parent_idx is None else str(begin_parent_idx)
    return SouthStarMarkerSlot(
        slot_id=f"{syntax_position}:{parent}->{begin_atom_idx}->{end_atom_idx}",
        edge=edge,
        begin_atom_idx=begin_atom_idx,
        end_atom_idx=end_atom_idx,
        begin_parent_idx=begin_parent_idx,
        syntax_position=syntax_position,
        adjacent_contexts=carrier_contexts_by_edge[edge],
    )


def _ring_stereo_marker_assignments_for_events(
    events: tuple[SouthStarTraversalEvent, ...],
    *,
    marker_by_edge: dict[Edge, str],
) -> tuple[dict[str, str], tuple[SouthStarRingStereoOracleEquation, ...]]:
    marker_by_slot = {}
    equations = []
    ring_stereo_closure_open_atom_by_edge = (
        _ring_stereo_closure_open_atom_by_edge(events)
    )
    for event in events:
        slot = event.marker_slot
        if slot is None:
            continue
        graph_marker = marker_by_edge[slot.edge]
        traversal_orientation_flip = _ring_stereo_traversal_orientation_flip(
            slot,
            ring_stereo_closure_open_atom_by_edge=(
                ring_stereo_closure_open_atom_by_edge
            ),
        )
        emitted_marker = (
            _flipped_marker(graph_marker)
            if traversal_orientation_flip
            else graph_marker
        )
        marker_by_slot[slot.slot_id] = emitted_marker
        equations.append(
            SouthStarRingStereoOracleEquation(
                equation_id=f"{slot.slot_id}:{slot.edge[0]}-{slot.edge[1]}",
                slot_id=slot.slot_id,
                edge=slot.edge,
                syntax_position=slot.syntax_position,
                graph_marker=graph_marker,
                emitted_marker=emitted_marker,
                traversal_orientation_flip=traversal_orientation_flip,
            )
        )
    return marker_by_slot, tuple(equations)


def _ring_stereo_closure_open_atom_by_edge(
    events: tuple[SouthStarTraversalEvent, ...],
) -> dict[Edge, int]:
    return {
        normalized_edge(event.edge): event.begin_atom_idx
        for event in events
        if event.kind == "ring_open"
        and event.edge is not None
        and event.begin_atom_idx is not None
        and event.text == "="
    }


def _render_ring_stereo_events(
    events: tuple[SouthStarTraversalEvent, ...],
    *,
    marker_by_slot: dict[str, str],
) -> str:
    rendered = []
    for event in events:
        if event.kind == "ring_open":
            marker = ""
            if event.marker_slot is not None:
                marker = marker_by_slot[event.marker_slot.slot_id]
            rendered.append(f"{marker}{event.text}1")
        elif event.kind == "ring_close":
            rendered.append(f"{event.text}1")
        elif event.marker_slot is not None:
            rendered.append(marker_by_slot[event.marker_slot.slot_id])
        else:
            rendered.append(event.text)
    return "".join(rendered)


def _ring_stereo_traversal_orientation_flip(
    slot: SouthStarMarkerSlot,
    *,
    ring_stereo_closure_open_atom_by_edge: dict[Edge, int],
) -> bool:
    if slot.syntax_position == "ring_open":
        return _ring_stereo_ring_open_orientation_flip(slot)

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
        # The independent oracle models the same semantic phase rule as the
        # runtime: closing the stereo double bond flips one endpoint basis.
        if open_atom_idx == context.center_atom_idx:
            flip = not flip
    return flip


def _ring_stereo_ring_open_orientation_flip(
    slot: SouthStarMarkerSlot,
) -> bool:
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


def _ring_stereo_child_edge_allowed(
    *,
    begin_atom_idx: int,
    end_atom_idx: int,
    begin_parent_idx: int | None,
    carrier_contexts_by_edge: dict[Edge, tuple[SouthStarCarrierContext, ...]],
) -> bool:
    edge = normalized_edge((begin_atom_idx, end_atom_idx))
    contexts = carrier_contexts_by_edge.get(edge, ())
    if len(contexts) <= 1:
        return True

    begin_contexts = tuple(
        context
        for context in contexts
        if context.center_atom_idx == begin_atom_idx
    )
    if len(begin_contexts) <= 1:
        return True
    return begin_parent_idx is not None and begin_parent_idx in {
        context.double_neighbor_idx for context in begin_contexts
    }


def _ring_stereo_directional_marker(mol: Chem.Mol, *, edge: Edge) -> str:
    bond = mol.GetBondBetweenAtoms(*edge)
    if bond is None:
        raise ValueError(f"carrier edge {edge!r} is not a bond")
    direction = bond.GetBondDir()
    if direction == Chem.BondDir.ENDUPRIGHT:
        return "/"
    if direction == Chem.BondDir.ENDDOWNRIGHT:
        return "\\"
    raise ValueError(
        f"ring-stereo carrier edge {edge!r} has unsupported direction {direction}"
    )


def _assert_ring_stereo_monocycle_domain(mol: Chem.Mol) -> None:
    if len(Chem.GetMolFrags(mol)) != 1:
        raise NotImplementedError("ring-stereo oracle requires one component")
    if len(mol.GetRingInfo().BondRings()) != 1:
        raise NotImplementedError("ring-stereo oracle requires one ring")


def _expanded_support(case_id: str) -> tuple[str, ...]:
    return next(
        case.expected_support
        for case in load_south_star_expanded_support_cases()
        if case.case_id == case_id
    )


def _exact_first_domain_support(case_id: str) -> tuple[str, ...]:
    return next(
        case.expected_support
        for case in load_south_star_exact_first_domain_cases()
        if case.case_id == case_id
    )


def _tree_fragments_with_blocked_edge(
    mol: Chem.Mol,
    *,
    atom_idx: int,
    parent_idx: int | None,
    visited: frozenset[int],
    blocked_edge: Edge,
) -> tuple[SouthStarTraversalFragment, ...]:
    visited = visited | {atom_idx}
    children = tuple(
        neighbor.GetIdx()
        for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
        if neighbor.GetIdx() != parent_idx
        and neighbor.GetIdx() not in visited
        and normalized_edge((atom_idx, neighbor.GetIdx())) != blocked_edge
    )
    atom_event = SouthStarTraversalEvent(
        kind="atom",
        text=_atom_text(mol.GetAtomWithIdx(atom_idx)),
        atom_idx=atom_idx,
    )
    if not children:
        return (SouthStarTraversalFragment(events=(atom_event,)),)

    fragments = []
    for ordered_children in permutations(sorted(children)):
        branch_children = ordered_children[:-1]
        main_child = ordered_children[-1]
        branch_groups = tuple(
            _branch_fragments(
                mol,
                begin_atom_idx=atom_idx,
                end_atom_idx=child_idx,
                visited=visited,
                blocked_edge=blocked_edge,
            )
            for child_idx in branch_children
        )
        main_fragments = _child_fragments(
            mol,
            begin_atom_idx=atom_idx,
            end_atom_idx=main_child,
            visited=visited,
            blocked_edge=blocked_edge,
        )
        for branch_group in product(*branch_groups):
            branch_events = tuple(
                event for branch in branch_group for event in branch.events
            )
            for main_fragment in main_fragments:
                fragments.append(
                    SouthStarTraversalFragment(
                        events=(atom_event,) + branch_events + main_fragment.events,
                    )
                )
    return tuple(dict.fromkeys(fragments))


def _branch_fragments(
    mol: Chem.Mol,
    *,
    begin_atom_idx: int,
    end_atom_idx: int,
    visited: frozenset[int],
    blocked_edge: Edge,
) -> tuple[SouthStarTraversalFragment, ...]:
    bond_event = _plain_ring_bond_event(
        mol,
        begin_atom_idx=begin_atom_idx,
        end_atom_idx=end_atom_idx,
    )
    return tuple(
        SouthStarTraversalFragment(
            events=(
                SouthStarTraversalEvent(kind="branch_open", text="("),
                bond_event,
            )
            + child_fragment.events
            + (SouthStarTraversalEvent(kind="branch_close", text=")"),)
        )
        for child_fragment in _tree_fragments_with_blocked_edge(
            mol,
            atom_idx=end_atom_idx,
            parent_idx=begin_atom_idx,
            visited=visited,
            blocked_edge=blocked_edge,
        )
    )


def _child_fragments(
    mol: Chem.Mol,
    *,
    begin_atom_idx: int,
    end_atom_idx: int,
    visited: frozenset[int],
    blocked_edge: Edge,
) -> tuple[SouthStarTraversalFragment, ...]:
    bond_event = _plain_ring_bond_event(
        mol,
        begin_atom_idx=begin_atom_idx,
        end_atom_idx=end_atom_idx,
    )
    return tuple(
        SouthStarTraversalFragment(events=(bond_event,) + child_fragment.events)
        for child_fragment in _tree_fragments_with_blocked_edge(
            mol,
            atom_idx=end_atom_idx,
            parent_idx=begin_atom_idx,
            visited=visited,
            blocked_edge=blocked_edge,
        )
    )


def _render_with_ring_digit(
    fragment: SouthStarTraversalFragment,
    closure_edge: Edge,
) -> str:
    rendered = []
    endpoint_count = 0
    for event in fragment.events:
        rendered.append(event.text)
        if event.atom_idx in closure_edge:
            endpoint_count += 1
            rendered.append("1")
    if endpoint_count != 2:
        raise ValueError(
            f"closure edge {closure_edge!r} endpoints were not both rendered"
        )
    return "".join(rendered)


def _plain_ring_bond_event(
    mol: Chem.Mol,
    *,
    begin_atom_idx: int,
    end_atom_idx: int,
) -> SouthStarTraversalEvent:
    edge = normalized_edge((begin_atom_idx, end_atom_idx))
    return SouthStarTraversalEvent(
        kind="bond",
        text=_bond_text(mol, begin_atom_idx, end_atom_idx),
        edge=edge,
        begin_atom_idx=begin_atom_idx,
        end_atom_idx=end_atom_idx,
    )


def _single_ring_edges(mol: Chem.Mol) -> tuple[Edge, ...]:
    bond_rings = mol.GetRingInfo().BondRings()
    if len(bond_rings) != 1:
        raise ValueError("saturated-monocycle oracle expects one ring")
    return tuple(
        normalized_edge(
            (
                mol.GetBondWithIdx(bond_idx).GetBeginAtomIdx(),
                mol.GetBondWithIdx(bond_idx).GetEndAtomIdx(),
            )
        )
        for bond_idx in bond_rings[0]
    )


def _assert_saturated_monocycle_domain(mol: Chem.Mol) -> None:
    if len(Chem.GetMolFrags(mol)) != 1:
        raise NotImplementedError("saturated-monocycle oracle requires one component")
    ring_edges = frozenset(_single_ring_edges(mol))
    if not ring_edges:
        raise NotImplementedError("saturated-monocycle oracle requires a ring")
    for atom in mol.GetAtoms():
        _atom_text(atom)
    for bond in mol.GetBonds():
        edge = normalized_edge((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        if bond.GetBondType() != Chem.BondType.SINGLE:
            raise NotImplementedError(
                "saturated-monocycle oracle supports only single bonds"
            )
        if bond.IsInRing() != (edge in ring_edges):
            raise ValueError(f"inconsistent ring membership for bond {edge!r}")


def _atom_text(atom: Chem.Atom) -> str:
    symbol = atom.GetSymbol()
    if symbol in {"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"}:
        return symbol
    raise NotImplementedError(f"unsupported saturated-monocycle atom {symbol!r}")


def _bond_text(mol: Chem.Mol, begin_atom_idx: int, end_atom_idx: int) -> str:
    bond = mol.GetBondBetweenAtoms(begin_atom_idx, end_atom_idx)
    if bond is None:
        raise ValueError(f"atoms {begin_atom_idx}, {end_atom_idx} are not bonded")
    if bond.GetBondType() == Chem.BondType.SINGLE:
        return ""
    if bond.GetBondType() == Chem.BondType.DOUBLE:
        return "="
    raise NotImplementedError(
        f"unsupported saturated-monocycle bond {bond.GetBondType()}"
    )


def _flipped_marker(marker: str) -> str:
    if marker == "/":
        return "\\"
    if marker == "\\":
        return "/"
    raise ValueError(f"unsupported directional marker {marker!r}")
