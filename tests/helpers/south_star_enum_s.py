from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from itertools import product

from rdkit import Chem

from tests.helpers.south_star_component_support_state import (
    SouthStarComponentComplexitySnapshot,
    SouthStarComponentMarkerAssignment,
    SouthStarComponentSupportState,
)
from tests.helpers.south_star_annotation_policy import Edge, normalized_edge
from tests.helpers.south_star_semantic_oracle import semantic_oracle_accepts
from tests.helpers.south_star_semantic_oracle import parse_smiles
from tests.helpers.south_star_semantics import SouthStarSemanticCase


@dataclass(frozen=True, slots=True)
class _CarrierContext:
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
    selected_marker: str
    adjacent_contexts: tuple[_CarrierContext, ...]


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


@dataclass(frozen=True, slots=True)
class SouthStarTreeTraversal:
    root_atom_idx: int
    events: tuple[SouthStarTraversalEvent, ...]

    def render(self) -> str:
        return "".join(event.text for event in self.events)


@dataclass(frozen=True, slots=True)
class SouthStarEnumSPrototypeResult:
    case_id: str
    outputs: tuple[str, ...]
    complexity_snapshot: SouthStarComponentComplexitySnapshot
    generation_basis: str


def mol_to_smiles_enum_s_prototype_for_case(
    case: SouthStarSemanticCase,
) -> SouthStarEnumSPrototypeResult:
    state = SouthStarComponentSupportState.from_case(case)
    _assert_fixture_carriers_are_supported(case, state)

    outputs = tuple(
        output
        for output in case.positive_semantic_smiles
        if semantic_oracle_accepts(
            source_smiles=case.source_smiles,
            candidate_smiles=output,
        )
    )
    if len(outputs) != len(case.positive_semantic_smiles):
        raise AssertionError(
            f"South Star fixture {case.case_id!r} contains semantic-positive "
            "outputs that the semantic oracle rejects"
        )

    return SouthStarEnumSPrototypeResult(
        case_id=case.case_id,
        outputs=outputs,
        complexity_snapshot=state.complexity_snapshot(),
        generation_basis="south_star_semantic_fixture_witnesses",
    )


def mol_to_smiles_enum_s_graph_native_for_case(
    case: SouthStarSemanticCase,
) -> SouthStarEnumSPrototypeResult:
    mol = parse_smiles(case.source_smiles)
    state = SouthStarComponentSupportState.from_mol(mol)
    outputs = tuple(
        dict.fromkeys(
            traversal.render()
            for traversal in _tree_traversals_for_mol(mol, state=state)
        )
    )
    return SouthStarEnumSPrototypeResult(
        case_id=case.case_id,
        outputs=outputs,
        complexity_snapshot=state.complexity_snapshot(),
        generation_basis="south_star_graph_native_tree_traversal",
    )


def mol_to_smiles_enum_s_tree_traversals_for_case(
    case: SouthStarSemanticCase,
) -> tuple[SouthStarTreeTraversal, ...]:
    mol = parse_smiles(case.source_smiles)
    state = SouthStarComponentSupportState.from_mol(mol)
    return _tree_traversals_for_mol(mol, state=state)


def _tree_traversals_for_mol(
    mol: Chem.Mol,
    *,
    state: SouthStarComponentSupportState,
) -> tuple[SouthStarTreeTraversal, ...]:
    return tuple(
        traversal
        for marker_by_edge in _combined_marker_assignments(state)
        for traversal in _tree_traversals_for_marker_assignment(
            mol,
            marker_by_edge=marker_by_edge,
        )
    )


def _assert_fixture_carriers_are_supported(
    case: SouthStarSemanticCase,
    state: SouthStarComponentSupportState,
) -> None:
    for edge in case.eligible_carrier_edges:
        if not state.allowed_directional_markers(edge=edge):
            raise AssertionError(
                f"South Star fixture {case.case_id!r} carrier edge {edge!r} "
                "has no component-supported directional marker"
            )


def _combined_marker_assignments(
    state: SouthStarComponentSupportState,
) -> tuple[dict[Edge, str], ...]:
    per_component = state.component_marker_assignments()
    if not per_component:
        return ({},)

    combined = []
    for assignment_group in product(*per_component):
        marker_by_edge: dict[Edge, str] = {}
        for assignment in assignment_group:
            _merge_assignment_markers(marker_by_edge, assignment)
        combined.append(marker_by_edge)
    return tuple(combined)


def _merge_assignment_markers(
    marker_by_edge: dict[Edge, str],
    assignment: SouthStarComponentMarkerAssignment,
) -> None:
    for edge, marker in assignment.marker_by_edge:
        existing = marker_by_edge.setdefault(edge, marker)
        if existing != marker:
            raise ValueError(
                f"conflicting graph-native marker assignment for edge {edge!r}"
            )


def _tree_traversals_for_marker_assignment(
    mol: Chem.Mol,
    *,
    marker_by_edge: dict[Edge, str],
) -> tuple[SouthStarTreeTraversal, ...]:
    _assert_tree_traversal_supported(mol)
    carrier_contexts_by_edge = _carrier_contexts_by_edge(
        mol,
        marker_by_edge=marker_by_edge,
    )
    return tuple(
        SouthStarTreeTraversal(
            root_atom_idx=root_idx,
            events=events,
        )
        for root_idx in range(mol.GetNumAtoms())
        for events in _atom_subtree_event_variants(
            mol,
            atom_idx=root_idx,
            parent_idx=None,
            visited=frozenset(),
            marker_by_edge=marker_by_edge,
            carrier_contexts_by_edge=carrier_contexts_by_edge,
        )
    )


def _assert_tree_traversal_supported(mol: Chem.Mol) -> None:
    if mol.GetNumAtoms() == 0:
        raise ValueError("South Star graph-native tree traversal requires atoms")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise NotImplementedError(
            "South Star graph-native tree traversal currently requires one "
            "connected component"
        )
    if mol.GetNumBonds() != mol.GetNumAtoms() - 1:
        raise NotImplementedError(
            "South Star graph-native tree traversal currently requires one "
            "connected acyclic component"
        )


def _atom_subtree_event_variants(
    mol: Chem.Mol,
    *,
    atom_idx: int,
    parent_idx: int | None,
    visited: frozenset[int],
    marker_by_edge: dict[Edge, str],
    carrier_contexts_by_edge: dict[Edge, tuple[_CarrierContext, ...]],
) -> tuple[tuple[SouthStarTraversalEvent, ...], ...]:
    visited = visited | {atom_idx}
    atom_text = _atom_text(mol.GetAtomWithIdx(atom_idx))
    atom_event = SouthStarTraversalEvent(
        kind="atom",
        text=atom_text,
        atom_idx=atom_idx,
    )
    children = tuple(
        neighbor.GetIdx()
        for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
        if neighbor.GetIdx() != parent_idx and neighbor.GetIdx() not in visited
    )
    if any(
        not _traversal_child_edge_allowed(
            begin_atom_idx=atom_idx,
            end_atom_idx=child_idx,
            begin_parent_idx=parent_idx,
            carrier_contexts_by_edge=carrier_contexts_by_edge,
        )
        for child_idx in children
    ):
        return ()
    if not children:
        return ((atom_event,),)

    variants: list[tuple[SouthStarTraversalEvent, ...]] = []
    for ordered_children in permutations(sorted(children)):
        branch_children = ordered_children[:-1]
        main_child = ordered_children[-1]
        branch_variants = tuple(
            _branch_event_variants(
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
        main_variants = _child_event_variants(
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
                event for branch in branch_group for event in branch
            )
            for main_events in main_variants:
                variants.append((atom_event,) + branch_events + main_events)
    return tuple(dict.fromkeys(variants))


def _branch_event_variants(
    mol: Chem.Mol,
    *,
    begin_atom_idx: int,
    end_atom_idx: int,
    begin_parent_idx: int | None,
    visited: frozenset[int],
    marker_by_edge: dict[Edge, str],
    carrier_contexts_by_edge: dict[Edge, tuple[_CarrierContext, ...]],
) -> tuple[tuple[SouthStarTraversalEvent, ...], ...]:
    branch_open = SouthStarTraversalEvent(kind="branch_open", text="(")
    branch_close = SouthStarTraversalEvent(kind="branch_close", text=")")
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
        (branch_open, bond_event) + child_events + (branch_close,)
        for child_events in _atom_subtree_event_variants(
            mol,
            atom_idx=end_atom_idx,
            parent_idx=begin_atom_idx,
            visited=visited,
            marker_by_edge=marker_by_edge,
            carrier_contexts_by_edge=carrier_contexts_by_edge,
        )
    )


def _child_event_variants(
    mol: Chem.Mol,
    *,
    begin_atom_idx: int,
    end_atom_idx: int,
    begin_parent_idx: int | None,
    visited: frozenset[int],
    marker_by_edge: dict[Edge, str],
    carrier_contexts_by_edge: dict[Edge, tuple[_CarrierContext, ...]],
) -> tuple[tuple[SouthStarTraversalEvent, ...], ...]:
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
        (bond_event,) + child_events
        for child_events in _atom_subtree_event_variants(
            mol,
            atom_idx=end_atom_idx,
            parent_idx=begin_atom_idx,
            visited=visited,
            marker_by_edge=marker_by_edge,
            carrier_contexts_by_edge=carrier_contexts_by_edge,
        )
    )


def _atom_text(atom: Chem.Atom) -> str:
    symbol = atom.GetSymbol()
    if symbol in {"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"}:
        return symbol
    raise NotImplementedError(
        f"South Star graph-native seed traversal does not support atom {symbol!r}"
    )


def _bond_event(
    mol: Chem.Mol,
    begin_atom_idx: int,
    end_atom_idx: int,
    *,
    begin_parent_idx: int | None,
    syntax_position: str,
    marker_by_edge: dict[Edge, str],
    carrier_contexts_by_edge: dict[Edge, tuple[_CarrierContext, ...]],
) -> SouthStarTraversalEvent:
    edge = normalized_edge((begin_atom_idx, end_atom_idx))
    bond = mol.GetBondBetweenAtoms(begin_atom_idx, end_atom_idx)
    if bond is None:
        raise ValueError(f"edge {edge!r} is not a bond")
    marker_slot = None
    if edge in marker_by_edge:
        text = _marker_for_traversal_edge(
            edge=edge,
            base_marker=marker_by_edge[edge],
            begin_atom_idx=begin_atom_idx,
            end_atom_idx=end_atom_idx,
            begin_parent_idx=begin_parent_idx,
            carrier_contexts_by_edge=carrier_contexts_by_edge,
        )
        marker_slot = _marker_slot(
            edge=edge,
            begin_atom_idx=begin_atom_idx,
            end_atom_idx=end_atom_idx,
            begin_parent_idx=begin_parent_idx,
            syntax_position=syntax_position,
            selected_marker=text,
            carrier_contexts_by_edge=carrier_contexts_by_edge,
        )
    elif bond.GetBondType() == Chem.BondType.DOUBLE:
        text = "="
    elif bond.GetBondType() == Chem.BondType.SINGLE:
        text = ""
    else:
        raise NotImplementedError(
            f"South Star graph-native tree traversal does not support bond "
            f"{bond.GetBondType()}"
        )
    return SouthStarTraversalEvent(
        kind="bond",
        text=text,
        edge=edge,
        begin_atom_idx=begin_atom_idx,
        end_atom_idx=end_atom_idx,
        begin_parent_idx=begin_parent_idx,
        marker_slot=marker_slot,
    )


def _marker_slot(
    *,
    edge: Edge,
    begin_atom_idx: int,
    end_atom_idx: int,
    begin_parent_idx: int | None,
    syntax_position: str,
    selected_marker: str,
    carrier_contexts_by_edge: dict[Edge, tuple[_CarrierContext, ...]],
) -> SouthStarMarkerSlot:
    return SouthStarMarkerSlot(
        slot_id=_marker_slot_id(
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
        selected_marker=selected_marker,
        adjacent_contexts=carrier_contexts_by_edge[edge],
    )


def _marker_slot_id(
    *,
    begin_atom_idx: int,
    end_atom_idx: int,
    begin_parent_idx: int | None,
    syntax_position: str,
) -> str:
    parent = "root" if begin_parent_idx is None else str(begin_parent_idx)
    return f"{syntax_position}:{parent}->{begin_atom_idx}->{end_atom_idx}"


def _carrier_contexts_by_edge(
    mol: Chem.Mol,
    *,
    marker_by_edge: dict[Edge, str],
) -> dict[Edge, tuple[_CarrierContext, ...]]:
    return {
        edge: _carrier_contexts_for_edge(mol, edge=edge)
        for edge in marker_by_edge
    }


def _carrier_contexts_for_edge(
    mol: Chem.Mol,
    *,
    edge: Edge,
) -> tuple[_CarrierContext, ...]:
    contexts = []
    for center_atom_idx, substituent_atom_idx in (edge, (edge[1], edge[0])):
        double_neighbors = tuple(
            neighbor.GetIdx()
            for neighbor in mol.GetAtomWithIdx(center_atom_idx).GetNeighbors()
            if neighbor.GetIdx() != substituent_atom_idx
            and mol.GetBondBetweenAtoms(
                center_atom_idx,
                neighbor.GetIdx(),
            ).GetBondType()
            == Chem.BondType.DOUBLE
        )
        contexts.extend(
            _CarrierContext(
                center_atom_idx=center_atom_idx,
                double_neighbor_idx=double_neighbor_idx,
            )
            for double_neighbor_idx in double_neighbors
        )
    if not contexts:
        raise NotImplementedError(
            f"South Star marker edge {edge!r} is not adjacent to a double bond"
        )
    return tuple(contexts)


def _marker_for_traversal_edge(
    *,
    edge: Edge,
    base_marker: str,
    begin_atom_idx: int,
    end_atom_idx: int,
    begin_parent_idx: int | None,
    carrier_contexts_by_edge: dict[Edge, tuple[_CarrierContext, ...]],
) -> str:
    flip = False
    for context in carrier_contexts_by_edge[edge]:
        if (
            begin_atom_idx == context.center_atom_idx
            and end_atom_idx != context.double_neighbor_idx
            and begin_parent_idx != context.double_neighbor_idx
        ):
            flip = not flip
    return _flipped_marker(base_marker) if flip else base_marker


def _traversal_child_edge_allowed(
    *,
    begin_atom_idx: int,
    end_atom_idx: int,
    begin_parent_idx: int | None,
    carrier_contexts_by_edge: dict[Edge, tuple[_CarrierContext, ...]],
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
        raise NotImplementedError(
            f"South Star shared carrier edge {edge!r} has ambiguous traversal "
            f"contexts from atom {begin_atom_idx}"
        )
    return begin_parent_idx == begin_contexts[0].double_neighbor_idx


def _flipped_marker(marker: str) -> str:
    if marker == "/":
        return "\\"
    if marker == "\\":
        return "/"
    raise ValueError(f"unsupported South Star directional marker {marker!r}")
