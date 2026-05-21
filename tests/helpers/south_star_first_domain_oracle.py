from __future__ import annotations

"""Independent spec oracle for the first South Star domain.

This module deliberately keeps an independent traversal search and renderer
from the graph-native EnumS prototype. The record vocabulary is shared with the
South Star reference spine so this helper remains a witness, not a separate
mini-world with its own carrier/slot/event types.

Do not import this module from package/runtime code. If the oracle and EnumS
prototype disagree, investigate the semantic model or fixtures instead of
making one call into the other. The only intended consumers are South Star
tests that need an implementation-independent support check.
"""

from itertools import permutations
from itertools import product

from rdkit import Chem

from grimace._south_star.annotation_policy import Edge
from grimace._south_star.annotation_policy import normalized_edge
from grimace._south_star.component_support_state import (
    SouthStarComponentMarkerAssignment,
)
from grimace._south_star.component_support_state import (
    SouthStarComponentSupportState,
)
from grimace._south_star.reference_model import SouthStarCarrierContext
from grimace._south_star.reference_model import SouthStarMarkerSlot
from grimace._south_star.reference_model import SouthStarTraversalEvent
from grimace._south_star.reference_model import SouthStarTraversalFragment
from tests.helpers.south_star_semantic_oracle import parse_smiles
from tests.helpers.south_star_semantics import SouthStarSemanticCase

def independent_first_domain_support_for_case(
    case: SouthStarSemanticCase,
) -> tuple[str, ...]:
    """Enumerate first-domain support independently of graph-native EnumS."""
    mol = parse_smiles(case.source_smiles)
    state = SouthStarComponentSupportState.from_mol(mol)
    _assert_first_domain_mol(mol)

    outputs = []
    for marker_by_edge in _combined_marker_by_edge_assignments(state):
        carrier_contexts_by_edge = _carrier_contexts_by_edge(
            mol,
            marker_by_edge=marker_by_edge,
        )
        for root_idx in range(mol.GetNumAtoms()):
            for fragment in _oracle_atom_variants(
                mol,
                atom_idx=root_idx,
                parent_idx=None,
                visited=frozenset(),
                marker_by_edge=marker_by_edge,
                carrier_contexts_by_edge=carrier_contexts_by_edge,
            ):
                outputs.append(
                    _render_fragment(
                        fragment,
                        marker_by_edge=marker_by_edge,
                    )
                )
    return tuple(dict.fromkeys(outputs))


def _combined_marker_by_edge_assignments(
    state: SouthStarComponentSupportState,
) -> tuple[dict[Edge, str], ...]:
    per_component = state.component_marker_assignments()
    if not per_component:
        return ({},)

    combined = []
    for assignment_group in product(*per_component):
        marker_by_edge: dict[Edge, str] = {}
        for assignment in assignment_group:
            _merge_assignment(marker_by_edge, assignment)
        combined.append(marker_by_edge)
    return tuple(combined)


def _merge_assignment(
    marker_by_edge: dict[Edge, str],
    assignment: SouthStarComponentMarkerAssignment,
) -> None:
    for edge, marker in assignment.marker_by_edge:
        existing = marker_by_edge.setdefault(edge, marker)
        if existing != marker:
            raise ValueError(f"conflicting graph marker assignment for {edge!r}")


def _oracle_atom_variants(
    mol: Chem.Mol,
    *,
    atom_idx: int,
    parent_idx: int | None,
    visited: frozenset[int],
    marker_by_edge: dict[Edge, str],
    carrier_contexts_by_edge: dict[Edge, tuple[SouthStarCarrierContext, ...]],
) -> tuple[SouthStarTraversalFragment, ...]:
    visited = visited | {atom_idx}
    children = tuple(
        neighbor.GetIdx()
        for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
        if neighbor.GetIdx() != parent_idx and neighbor.GetIdx() not in visited
    )
    if any(
        not _oracle_child_allowed(
            begin_atom_idx=atom_idx,
            end_atom_idx=child_idx,
            begin_parent_idx=parent_idx,
            carrier_contexts_by_edge=carrier_contexts_by_edge,
        )
        for child_idx in children
    ):
        return ()

    atom_token = _atom_token(mol.GetAtomWithIdx(atom_idx))
    if not children:
        return (
            SouthStarTraversalFragment(
                events=(_atom_event(atom_idx, atom_token),),
            ),
        )

    fragments = []
    for ordered_children in permutations(sorted(children)):
        branch_children = ordered_children[:-1]
        main_child = ordered_children[-1]
        branch_variants = tuple(
            _oracle_branch_variants(
                mol,
                begin_atom_idx=atom_idx,
                end_atom_idx=child_idx,
                begin_parent_idx=parent_idx,
                visited=visited,
                marker_by_edge=marker_by_edge,
                carrier_contexts_by_edge=carrier_contexts_by_edge,
            )
            for child_idx in branch_children
        )
        main_variants = _oracle_child_variants(
            mol,
            begin_atom_idx=atom_idx,
            end_atom_idx=main_child,
            begin_parent_idx=parent_idx,
            visited=visited,
            marker_by_edge=marker_by_edge,
            carrier_contexts_by_edge=carrier_contexts_by_edge,
        )
        for branch_group in product(*branch_variants):
            branch_events = tuple(
                event for branch in branch_group for event in branch.events
            )
            for main_fragment in main_variants:
                fragments.append(
                    SouthStarTraversalFragment(
                        events=(
                            _atom_event(atom_idx, atom_token),
                        )
                        + branch_events
                        + main_fragment.events,
                    )
                )
    return tuple(dict.fromkeys(fragments))


def _oracle_branch_variants(
    mol: Chem.Mol,
    *,
    begin_atom_idx: int,
    end_atom_idx: int,
    begin_parent_idx: int | None,
    visited: frozenset[int],
    marker_by_edge: dict[Edge, str],
    carrier_contexts_by_edge: dict[Edge, tuple[SouthStarCarrierContext, ...]],
) -> tuple[SouthStarTraversalFragment, ...]:
    bond_event = _bond_event(
        mol,
        begin_atom_idx,
        end_atom_idx,
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
        for child_fragment in _oracle_atom_variants(
            mol,
            atom_idx=end_atom_idx,
            parent_idx=begin_atom_idx,
            visited=visited,
            marker_by_edge=marker_by_edge,
            carrier_contexts_by_edge=carrier_contexts_by_edge,
        )
    )


def _oracle_child_variants(
    mol: Chem.Mol,
    *,
    begin_atom_idx: int,
    end_atom_idx: int,
    begin_parent_idx: int | None,
    visited: frozenset[int],
    marker_by_edge: dict[Edge, str],
    carrier_contexts_by_edge: dict[Edge, tuple[SouthStarCarrierContext, ...]],
) -> tuple[SouthStarTraversalFragment, ...]:
    bond_event = _bond_event(
        mol,
        begin_atom_idx,
        end_atom_idx,
        begin_parent_idx=begin_parent_idx,
        syntax_position="main",
        marker_by_edge=marker_by_edge,
        carrier_contexts_by_edge=carrier_contexts_by_edge,
    )
    return tuple(
        SouthStarTraversalFragment(
            events=(bond_event,) + child_fragment.events,
        )
        for child_fragment in _oracle_atom_variants(
            mol,
            atom_idx=end_atom_idx,
            parent_idx=begin_atom_idx,
            visited=visited,
            marker_by_edge=marker_by_edge,
            carrier_contexts_by_edge=carrier_contexts_by_edge,
        )
    )


def _bond_event(
    mol: Chem.Mol,
    begin_atom_idx: int,
    end_atom_idx: int,
    *,
    begin_parent_idx: int | None,
    syntax_position: str,
    marker_by_edge: dict[Edge, str],
    carrier_contexts_by_edge: dict[Edge, tuple[SouthStarCarrierContext, ...]],
) -> SouthStarTraversalEvent:
    edge = normalized_edge((begin_atom_idx, end_atom_idx))
    bond = mol.GetBondBetweenAtoms(begin_atom_idx, end_atom_idx)
    if bond is None:
        raise ValueError(f"edge {edge!r} is not a bond")
    if edge in marker_by_edge:
        slot = SouthStarMarkerSlot(
            slot_id=_slot_id(
                begin_atom_idx=begin_atom_idx,
                end_atom_idx=end_atom_idx,
                begin_parent_idx=begin_parent_idx,
                syntax_position=syntax_position,
            ),
            edge=edge,
            begin_atom_idx=begin_atom_idx,
            end_atom_idx=end_atom_idx,
            begin_parent_idx=begin_parent_idx,
            syntax_position=syntax_position,
            adjacent_contexts=carrier_contexts_by_edge[edge],
        )
        return _bond_traversal_event(
            text="",
            edge=edge,
            begin_atom_idx=begin_atom_idx,
            end_atom_idx=end_atom_idx,
            begin_parent_idx=begin_parent_idx,
            marker_slot=slot,
        )
    if bond.GetBondType() == Chem.BondType.DOUBLE:
        return _bond_traversal_event(
            text="=",
            edge=edge,
            begin_atom_idx=begin_atom_idx,
            end_atom_idx=end_atom_idx,
            begin_parent_idx=begin_parent_idx,
        )
    if bond.GetBondType() == Chem.BondType.SINGLE:
        return _bond_traversal_event(
            text="",
            edge=edge,
            begin_atom_idx=begin_atom_idx,
            end_atom_idx=end_atom_idx,
            begin_parent_idx=begin_parent_idx,
        )
    raise NotImplementedError(f"unsupported first-domain bond {bond.GetBondType()}")


def _render_fragment(
    fragment: SouthStarTraversalFragment,
    *,
    marker_by_edge: dict[Edge, str],
) -> str:
    marker_by_slot = {
        event.marker_slot.slot_id: _oriented_marker(
            marker_by_edge[event.marker_slot.edge],
            event.marker_slot,
        )
        for event in fragment.events
        if event.marker_slot is not None
    }
    tokens = []
    for event in fragment.events:
        if event.marker_slot is not None:
            tokens.append(marker_by_slot[event.marker_slot.slot_id])
        else:
            tokens.append(event.text)
    return "".join(tokens)


def _oriented_marker(graph_marker: str, slot: SouthStarMarkerSlot) -> str:
    flip = False
    for context in slot.adjacent_contexts:
        if (
            slot.begin_atom_idx == context.center_atom_idx
            and slot.end_atom_idx != context.double_neighbor_idx
            and slot.begin_parent_idx != context.double_neighbor_idx
        ):
            flip = not flip
    return _flipped_marker(graph_marker) if flip else graph_marker


def _carrier_contexts_by_edge(
    mol: Chem.Mol,
    *,
    marker_by_edge: dict[Edge, str],
) -> dict[Edge, tuple[SouthStarCarrierContext, ...]]:
    return {
        edge: _carrier_contexts_for_edge(mol, edge=edge)
        for edge in marker_by_edge
    }


def _carrier_contexts_for_edge(
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
            if bond is not None and bond.GetBondType() == Chem.BondType.DOUBLE:
                contexts.append(
                    SouthStarCarrierContext(
                        center_atom_idx=center_atom_idx,
                        double_neighbor_idx=neighbor_idx,
                    )
                )
    if not contexts:
        raise ValueError(f"marker edge {edge!r} has no double-bond context")
    return tuple(contexts)


def _oracle_child_allowed(
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
    if len(begin_contexts) != 1:
        raise NotImplementedError(f"ambiguous shared carrier edge {edge!r}")
    return begin_parent_idx == begin_contexts[0].double_neighbor_idx


def _atom_token(atom: Chem.Atom) -> str:
    symbol = atom.GetSymbol()
    if symbol in {"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"}:
        return symbol
    raise NotImplementedError(f"unsupported first-domain atom {symbol!r}")


def _atom_event(atom_idx: int, text: str) -> SouthStarTraversalEvent:
    return SouthStarTraversalEvent(kind="atom", text=text, atom_idx=atom_idx)


def _bond_traversal_event(
    *,
    text: str,
    edge: Edge,
    begin_atom_idx: int,
    end_atom_idx: int,
    begin_parent_idx: int | None,
    marker_slot: SouthStarMarkerSlot | None = None,
) -> SouthStarTraversalEvent:
    return SouthStarTraversalEvent(
        kind="bond",
        text=text,
        edge=edge,
        begin_atom_idx=begin_atom_idx,
        end_atom_idx=end_atom_idx,
        begin_parent_idx=begin_parent_idx,
        marker_slot=marker_slot,
    )


def _assert_first_domain_mol(mol: Chem.Mol) -> None:
    if len(Chem.GetMolFrags(mol)) != 1:
        raise NotImplementedError("first-domain oracle requires one component")
    if mol.GetNumBonds() != mol.GetNumAtoms() - 1:
        raise NotImplementedError("first-domain oracle requires an acyclic molecule")


def _slot_id(
    *,
    begin_atom_idx: int,
    end_atom_idx: int,
    begin_parent_idx: int | None,
    syntax_position: str,
) -> str:
    parent = "root" if begin_parent_idx is None else str(begin_parent_idx)
    return f"{syntax_position}:{parent}->{begin_atom_idx}->{end_atom_idx}"


def _flipped_marker(marker: str) -> str:
    if marker == "/":
        return "\\"
    if marker == "\\":
        return "/"
    raise ValueError(f"unsupported first-domain marker {marker!r}")
