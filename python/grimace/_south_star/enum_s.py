from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from functools import reduce
from itertools import permutations
from itertools import product
from operator import mul

from rdkit import Chem

from grimace._south_star.component_support_state import (
    SouthStarComponentComplexitySnapshot,
    SouthStarComponentMarkerAssignment,
    SouthStarComponentSupportState,
)
from grimace._south_star.annotation_policy import Edge, normalized_edge
from grimace._south_star.fragments import (
    SouthStarFragmentSupport,
    compose_disconnected_fragment_supports,
)
from grimace._south_star.policies import DEFAULT_SOUTH_STAR_POLICY_SET
from grimace._south_star.policies import SouthStarPolicySet
from grimace._south_star.support_gates import (
    is_supported_monocycle_with_acyclic_branches,
)
from grimace._south_star.tetrahedral import (
    IMPLICIT_HYDROGEN_LIGAND,
    SouthStarTetrahedralCenterFact,
    extract_tetrahedral_center_facts,
    preserving_tetrahedral_token,
)


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
    adjacent_contexts: tuple[_CarrierContext, ...]


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
class SouthStarTreeTraversal:
    root_atom_idx: int
    events: tuple[SouthStarTraversalEvent, ...]
    marker_assignments: tuple[SouthStarMarkerSlotAssignment, ...]
    component_marker_assignments: tuple[SouthStarComponentMarkerAssignment, ...]

    def render(self) -> str:
        return render_south_star_traversal(
            self.events,
            marker_assignments=self.marker_assignments,
        )


@dataclass(frozen=True, slots=True)
class SouthStarEnumSGenerationDiagnostics:
    fragment_count: int
    fragment_output_counts: tuple[int, ...]
    traversal_skeleton_count: int
    marker_slot_count: int
    local_assignment_count: int
    solved_assignment_count: int
    estimated_product_size: int


@dataclass(frozen=True, slots=True)
class SouthStarEnumSPrototypeResult:
    case_id: str
    outputs: tuple[str, ...]
    complexity_snapshot: SouthStarComponentComplexitySnapshot
    generation_basis: str
    generation_diagnostics: SouthStarEnumSGenerationDiagnostics | None = None
    annotation_policy: str = DEFAULT_SOUTH_STAR_POLICY_SET.annotation_policy.name
    fragment_order_policy: str = (
        DEFAULT_SOUTH_STAR_POLICY_SET.fragment_order_policy.name
    )
    output_order_policy: str = DEFAULT_SOUTH_STAR_POLICY_SET.output_order_policy.name


@dataclass(frozen=True, slots=True)
class _TraversalFragment:
    events: tuple[SouthStarTraversalEvent, ...]


@dataclass(frozen=True, slots=True)
class _CombinedMarkerAssignment:
    marker_by_edge: dict[Edge, str]
    component_marker_assignments: tuple[SouthStarComponentMarkerAssignment, ...]


@dataclass(frozen=True, slots=True)
class _SupportGeneration:
    outputs: tuple[str, ...]
    diagnostics: SouthStarEnumSGenerationDiagnostics


def mol_to_smiles_enum_s_graph_native_for_case(
    case: object,
) -> SouthStarEnumSPrototypeResult:
    return mol_to_smiles_enum_s_graph_native(
        case.source_smiles,
        case_id=case.case_id,
    )


def mol_to_smiles_enum_s_graph_native(
    source_smiles: str,
    *,
    case_id: str = "",
    policy_set: SouthStarPolicySet = DEFAULT_SOUTH_STAR_POLICY_SET,
) -> SouthStarEnumSPrototypeResult:
    mol = _parse_smiles(source_smiles)
    return _mol_to_smiles_enum_s_graph_native_for_mol(
        mol,
        case_id=case_id,
        policy_set=policy_set,
    )


def _mol_to_smiles_enum_s_graph_native_for_mol(
    mol: Chem.Mol,
    *,
    case_id: str = "",
    policy_set: SouthStarPolicySet,
) -> SouthStarEnumSPrototypeResult:
    state = SouthStarComponentSupportState.from_mol(
        mol,
        annotation_policy=policy_set.annotation_policy,
    )
    if len(Chem.GetMolFrags(mol)) > 1:
        generation = _disconnected_generation_for_mol(mol, policy_set=policy_set)
    else:
        generation = _connected_generation_for_mol(
            mol,
            state=state,
            policy_set=policy_set,
        )
    return SouthStarEnumSPrototypeResult(
        case_id=case_id,
        outputs=generation.outputs,
        complexity_snapshot=state.complexity_snapshot(),
        generation_basis="south_star_graph_native_equation_solved_tree_traversal",
        generation_diagnostics=generation.diagnostics,
        annotation_policy=policy_set.annotation_policy.name,
        fragment_order_policy=policy_set.fragment_order_policy.name,
        output_order_policy=policy_set.output_order_policy.name,
    )


def _connected_generation_for_mol(
    mol: Chem.Mol,
    *,
    state: SouthStarComponentSupportState,
    policy_set: SouthStarPolicySet,
) -> _SupportGeneration:
    traversals = _tree_traversals_for_mol(mol, state=state)
    raw_outputs = tuple(traversal.render() for traversal in traversals)
    outputs = policy_set.output_order_policy.deduplicate(raw_outputs)
    marker_slot_count = sum(
        1
        for traversal in traversals
        for event in traversal.events
        if event.marker_slot is not None
    )
    local_assignment_count = state.complexity_snapshot().estimated_product_size
    return _SupportGeneration(
        outputs=outputs,
        diagnostics=SouthStarEnumSGenerationDiagnostics(
            fragment_count=1,
            fragment_output_counts=(len(outputs),),
            traversal_skeleton_count=len(traversals),
            marker_slot_count=marker_slot_count,
            local_assignment_count=local_assignment_count,
            solved_assignment_count=len(traversals),
            estimated_product_size=len(raw_outputs),
        ),
    )


def _disconnected_generation_for_mol(
    mol: Chem.Mol,
    *,
    policy_set: SouthStarPolicySet,
) -> _SupportGeneration:
    fragment_generations = tuple(
        _connected_generation_for_mol(
            fragment,
            state=SouthStarComponentSupportState.from_mol(
                fragment,
                annotation_policy=policy_set.annotation_policy,
            ),
            policy_set=policy_set,
        )
        for fragment in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    )
    fragment_supports = tuple(
        SouthStarFragmentSupport(
            fragment_id=f"fragment:{fragment_idx}",
            outputs=generation.outputs,
        )
        for fragment_idx, generation in enumerate(fragment_generations)
    )
    composition = compose_disconnected_fragment_supports(
        fragment_supports,
        fragment_order_policy=policy_set.fragment_order_policy,
        output_order_policy=policy_set.output_order_policy,
    )
    return _SupportGeneration(
        outputs=composition.outputs,
        diagnostics=SouthStarEnumSGenerationDiagnostics(
            fragment_count=composition.fragment_count,
            fragment_output_counts=composition.fragment_output_counts,
            traversal_skeleton_count=sum(
                generation.diagnostics.traversal_skeleton_count
                for generation in fragment_generations
            ),
            marker_slot_count=sum(
                generation.diagnostics.marker_slot_count
                for generation in fragment_generations
            ),
            local_assignment_count=reduce(
                mul,
                (
                    generation.diagnostics.local_assignment_count
                    for generation in fragment_generations
                ),
                1,
            ),
            solved_assignment_count=sum(
                generation.diagnostics.solved_assignment_count
                for generation in fragment_generations
            ),
            estimated_product_size=composition.estimated_product_size,
        ),
    )


def mol_to_smiles_enum_s_tree_traversals_for_case(
    case: object,
) -> tuple[SouthStarTreeTraversal, ...]:
    mol = _parse_smiles(case.source_smiles)
    state = SouthStarComponentSupportState.from_mol(mol)
    return _tree_traversals_for_mol(mol, state=state)


def render_south_star_traversal(
    events: tuple[SouthStarTraversalEvent, ...],
    *,
    marker_assignments: tuple[SouthStarMarkerSlotAssignment, ...],
) -> str:
    markers_by_slot = _marker_assignments_by_slot(marker_assignments)
    required_slot_ids = {
        event.marker_slot.slot_id
        for event in events
        if event.marker_slot is not None
    }
    if set(markers_by_slot) != required_slot_ids:
        raise ValueError(
            "marker assignments must exactly cover traversal marker slots"
        )

    rendered: list[str] = []
    for event in events:
        rendered.append(_render_traversal_event(event, markers_by_slot))
    return "".join(rendered)


def _render_traversal_event(
    event: SouthStarTraversalEvent,
    markers_by_slot: dict[str, str],
) -> str:
    if event.ring_closure is not None:
        return _render_ring_closure_event(event, markers_by_slot)
    if event.marker_slot is None:
        return event.text
    if event.text:
        raise ValueError("marker-slot events must not carry rendered marker text")
    return markers_by_slot[event.marker_slot.slot_id]


def _render_ring_closure_event(
    event: SouthStarTraversalEvent,
    markers_by_slot: dict[str, str],
) -> str:
    closure = event.ring_closure
    if closure is None:
        raise ValueError("ring closure event requires ring_closure payload")
    if event.kind not in {"ring_open", "ring_close"}:
        raise ValueError("ring closure payload requires ring_open or ring_close event")
    if closure.role not in {"open", "close"}:
        raise ValueError("ring closure role must be open or close")
    if not closure.closure_id:
        raise ValueError("ring closure id must be nonempty")
    if not closure.label:
        raise ValueError("ring closure label must be nonempty")

    marker = ""
    if event.marker_slot is not None:
        marker = markers_by_slot[event.marker_slot.slot_id]
    return f"{marker}{event.text}{closure.label}"


def _marker_assignments_by_slot(
    marker_assignments: tuple[SouthStarMarkerSlotAssignment, ...],
) -> dict[str, str]:
    markers_by_slot = {}
    for assignment in marker_assignments:
        if assignment.marker not in {"/", "\\"}:
            raise ValueError(
                f"unsupported South Star marker assignment {assignment.marker!r}"
            )
        if assignment.slot_id in markers_by_slot:
            raise ValueError(
                f"duplicate marker assignment for slot {assignment.slot_id!r}"
            )
        markers_by_slot[assignment.slot_id] = assignment.marker
    return markers_by_slot


def _tree_traversals_for_mol(
    mol: Chem.Mol,
    *,
    state: SouthStarComponentSupportState,
) -> tuple[SouthStarTreeTraversal, ...]:
    return tuple(
        traversal
        for combined_assignment in _combined_marker_assignments(state)
        for traversal in _tree_traversals_for_marker_assignment(
            mol,
            state=state,
            marker_by_edge=combined_assignment.marker_by_edge,
            component_marker_assignments=(
                combined_assignment.component_marker_assignments
            ),
        )
    )


def _combined_marker_assignments(
    state: SouthStarComponentSupportState,
) -> tuple[_CombinedMarkerAssignment, ...]:
    per_component = state.component_marker_assignments()
    if not per_component:
        return (
            _CombinedMarkerAssignment(
                marker_by_edge={},
                component_marker_assignments=(),
            ),
        )

    combined = []
    for assignment_group in product(*per_component):
        marker_by_edge: dict[Edge, str] = {}
        for assignment in assignment_group:
            _merge_assignment_markers(marker_by_edge, assignment)
        combined.append(
            _CombinedMarkerAssignment(
                marker_by_edge=marker_by_edge,
                component_marker_assignments=assignment_group,
            )
        )
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
    state: SouthStarComponentSupportState,
    marker_by_edge: dict[Edge, str],
    component_marker_assignments: tuple[SouthStarComponentMarkerAssignment, ...],
) -> tuple[SouthStarTreeTraversal, ...]:
    _assert_tree_traversal_supported(mol)
    if is_supported_monocycle_with_acyclic_branches(mol):
        return _monocycle_traversals(
            mol,
            state=state,
            marker_by_edge=marker_by_edge,
            component_marker_assignments=component_marker_assignments,
        )

    carrier_contexts_by_edge = _carrier_contexts_by_edge(
        mol,
        marker_by_edge=marker_by_edge,
    )
    tetrahedral_facts_by_atom = _tetrahedral_facts_by_atom(mol)
    return tuple(
        _with_solved_marker_assignments(
            state,
            SouthStarTreeTraversal(
                root_atom_idx=root_idx,
                events=fragment.events,
                marker_assignments=(),
                component_marker_assignments=component_marker_assignments,
            ),
        )
        for root_idx in range(mol.GetNumAtoms())
        for fragment in _atom_subtree_event_variants(
            mol,
            atom_idx=root_idx,
            parent_idx=None,
            visited=frozenset(),
            blocked_edges=frozenset(),
            marker_by_edge=marker_by_edge,
            carrier_contexts_by_edge=carrier_contexts_by_edge,
            tetrahedral_facts_by_atom=tetrahedral_facts_by_atom,
        )
    )


def _with_solved_marker_assignments(
    state: SouthStarComponentSupportState,
    traversal: SouthStarTreeTraversal,
) -> SouthStarTreeTraversal:
    from grimace._south_star.marker_equations import (
        marker_slot_parity_equations_for_traversal,
    )
    from grimace._south_star.parity_solver import (
        solve_marker_slot_parity_equations,
    )

    equations = marker_slot_parity_equations_for_traversal(state, traversal)
    solver_result = solve_marker_slot_parity_equations(equations)
    if len(solver_result.assignments) != 1:
        raise ValueError(
            "South Star graph-native traversal expects exactly one solved "
            "marker assignment per traversal skeleton"
        )
    return replace(
        traversal,
        marker_assignments=tuple(
            SouthStarMarkerSlotAssignment(slot_id=slot_id, marker=marker)
            for slot_id, marker in solver_result.assignments[0].marker_by_slot
        ),
    )


def _assert_tree_traversal_supported(mol: Chem.Mol) -> None:
    if mol.GetNumAtoms() == 0:
        raise ValueError("South Star graph-native tree traversal requires atoms")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise NotImplementedError(
            "South Star graph-native tree traversal currently requires one "
            "connected component"
        )
    if mol.GetNumBonds() == mol.GetNumAtoms() - 1:
        return
    if is_supported_monocycle_with_acyclic_branches(mol):
        return
    if mol.GetNumBonds() != mol.GetNumAtoms() - 1:
        raise NotImplementedError(
            "South Star graph-native tree traversal currently requires one "
            "connected acyclic component or one nonstereo monocycle"
        )


def _parse_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"failed to parse SMILES {smiles!r}")
    return mol


def _atom_subtree_event_variants(
    mol: Chem.Mol,
    *,
    atom_idx: int,
    parent_idx: int | None,
    visited: frozenset[int],
    blocked_edges: frozenset[Edge],
    marker_by_edge: dict[Edge, str],
    carrier_contexts_by_edge: dict[Edge, tuple[_CarrierContext, ...]],
    tetrahedral_facts_by_atom: dict[int, SouthStarTetrahedralCenterFact],
) -> tuple[_TraversalFragment, ...]:
    visited = visited | {atom_idx}
    children = tuple(
        neighbor.GetIdx()
        for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
        if neighbor.GetIdx() != parent_idx and neighbor.GetIdx() not in visited
        and normalized_edge((atom_idx, neighbor.GetIdx())) not in blocked_edges
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
        atom_event = _atom_event(
            mol,
            atom_idx=atom_idx,
            parent_idx=parent_idx,
            ordered_children=(),
            tetrahedral_facts_by_atom=tetrahedral_facts_by_atom,
        )
        return (_TraversalFragment(events=(atom_event,)),)

    variants: list[_TraversalFragment] = []
    for ordered_children in permutations(sorted(children)):
        atom_event = _atom_event(
            mol,
            atom_idx=atom_idx,
            parent_idx=parent_idx,
            ordered_children=ordered_children,
            tetrahedral_facts_by_atom=tetrahedral_facts_by_atom,
        )
        branch_children = ordered_children[:-1]
        main_child = ordered_children[-1]
        branch_variants = tuple(
            _branch_event_variants(
                mol,
                begin_atom_idx=atom_idx,
                end_atom_idx=child_idx,
                begin_parent_idx=parent_idx,
                visited=visited,
                blocked_edges=blocked_edges,
                marker_by_edge=marker_by_edge,
                carrier_contexts_by_edge=carrier_contexts_by_edge,
                tetrahedral_facts_by_atom=tetrahedral_facts_by_atom,
            )
            for child_idx in branch_children
        )
        main_variants = _child_event_variants(
            mol,
            begin_atom_idx=atom_idx,
            end_atom_idx=main_child,
            begin_parent_idx=parent_idx,
            visited=visited,
            blocked_edges=blocked_edges,
            marker_by_edge=marker_by_edge,
            carrier_contexts_by_edge=carrier_contexts_by_edge,
            tetrahedral_facts_by_atom=tetrahedral_facts_by_atom,
        )
        for branch_group in product(*branch_variants):
            branch_events = tuple(
                event for branch in branch_group for event in branch.events
            )
            for main_fragment in main_variants:
                variants.append(
                    _TraversalFragment(
                        events=(atom_event,)
                        + branch_events
                        + main_fragment.events,
                    )
                )
    return tuple(dict.fromkeys(variants))


def _branch_event_variants(
    mol: Chem.Mol,
    *,
    begin_atom_idx: int,
    end_atom_idx: int,
    begin_parent_idx: int | None,
    visited: frozenset[int],
    blocked_edges: frozenset[Edge],
    marker_by_edge: dict[Edge, str],
    carrier_contexts_by_edge: dict[Edge, tuple[_CarrierContext, ...]],
    tetrahedral_facts_by_atom: dict[int, SouthStarTetrahedralCenterFact],
) -> tuple[_TraversalFragment, ...]:
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
        _TraversalFragment(
            events=(branch_open, bond_event)
            + child_fragment.events
            + (branch_close,),
        )
        for child_fragment in _atom_subtree_event_variants(
            mol,
            atom_idx=end_atom_idx,
            parent_idx=begin_atom_idx,
            visited=visited,
            blocked_edges=blocked_edges,
            marker_by_edge=marker_by_edge,
            carrier_contexts_by_edge=carrier_contexts_by_edge,
            tetrahedral_facts_by_atom=tetrahedral_facts_by_atom,
        )
    )


def _child_event_variants(
    mol: Chem.Mol,
    *,
    begin_atom_idx: int,
    end_atom_idx: int,
    begin_parent_idx: int | None,
    visited: frozenset[int],
    blocked_edges: frozenset[Edge],
    marker_by_edge: dict[Edge, str],
    carrier_contexts_by_edge: dict[Edge, tuple[_CarrierContext, ...]],
    tetrahedral_facts_by_atom: dict[int, SouthStarTetrahedralCenterFact],
) -> tuple[_TraversalFragment, ...]:
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
        _TraversalFragment(
            events=(bond_event,) + child_fragment.events,
        )
        for child_fragment in _atom_subtree_event_variants(
            mol,
            atom_idx=end_atom_idx,
            parent_idx=begin_atom_idx,
            visited=visited,
            blocked_edges=blocked_edges,
            marker_by_edge=marker_by_edge,
            carrier_contexts_by_edge=carrier_contexts_by_edge,
            tetrahedral_facts_by_atom=tetrahedral_facts_by_atom,
        )
    )


def _monocycle_traversals(
    mol: Chem.Mol,
    *,
    state: SouthStarComponentSupportState,
    marker_by_edge: dict[Edge, str],
    component_marker_assignments: tuple[SouthStarComponentMarkerAssignment, ...],
) -> tuple[SouthStarTreeTraversal, ...]:
    ring_edges = _supported_ring_closure_edges(mol)
    carrier_contexts_by_edge = _carrier_contexts_by_edge(
        mol,
        marker_by_edge=marker_by_edge,
    )
    return tuple(
        _with_solved_marker_assignments(
            state,
            SouthStarTreeTraversal(
                root_atom_idx=root_idx,
                events=_with_ring_closure_events(
                    mol,
                    fragment.events,
                    closure_edge=closure_edge,
                    marker_by_edge=marker_by_edge,
                    carrier_contexts_by_edge=carrier_contexts_by_edge,
                ),
                marker_assignments=(),
                component_marker_assignments=component_marker_assignments,
            ),
        )
        for root_idx in range(mol.GetNumAtoms())
        for closure_edge in ring_edges
        for fragment in _atom_subtree_event_variants(
            mol,
            atom_idx=root_idx,
            parent_idx=None,
            visited=frozenset(),
            blocked_edges=frozenset((closure_edge,)),
            marker_by_edge=marker_by_edge,
            carrier_contexts_by_edge=carrier_contexts_by_edge,
            tetrahedral_facts_by_atom={},
        )
    )


def _single_ring_edges(mol: Chem.Mol) -> tuple[Edge, ...]:
    bond_rings = mol.GetRingInfo().BondRings()
    if len(bond_rings) != 1:
        raise ValueError("South Star nonstereo-ring traversal expects one ring")
    return tuple(
        normalized_edge(
            (
                mol.GetBondWithIdx(bond_idx).GetBeginAtomIdx(),
                mol.GetBondWithIdx(bond_idx).GetEndAtomIdx(),
            )
        )
        for bond_idx in bond_rings[0]
    )


def _supported_ring_closure_edges(mol: Chem.Mol) -> tuple[Edge, ...]:
    return tuple(
        edge
        for edge in _single_ring_edges(mol)
        if _ring_closure_edge_supported(mol, edge=edge)
    )


def _ring_closure_edge_supported(mol: Chem.Mol, *, edge: Edge) -> bool:
    bond = mol.GetBondBetweenAtoms(*edge)
    if bond is None:
        raise ValueError(f"ring closure edge {edge!r} is not a bond")
    return bond.GetStereo() == Chem.BondStereo.STEREONONE


def _with_ring_closure_events(
    mol: Chem.Mol,
    events: tuple[SouthStarTraversalEvent, ...],
    *,
    closure_edge: Edge,
    marker_by_edge: dict[Edge, str],
    carrier_contexts_by_edge: dict[Edge, tuple[_CarrierContext, ...]],
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
    traversal_parent_by_atom = {
        event.end_atom_idx: event.begin_atom_idx
        for event in events
        if event.kind == "bond"
        and event.begin_atom_idx is not None
        and event.end_atom_idx is not None
    }
    open_event = SouthStarTraversalEvent(
        kind="ring_open",
        text=_ring_closure_bond_text(mol, closure_edge),
        edge=closure_edge,
        begin_atom_idx=open_atom_idx,
        end_atom_idx=close_atom_idx,
        begin_parent_idx=traversal_parent_by_atom.get(open_atom_idx),
        marker_slot=_ring_closure_marker_slot(
            closure_edge=closure_edge,
            begin_atom_idx=open_atom_idx,
            end_atom_idx=close_atom_idx,
            begin_parent_idx=traversal_parent_by_atom.get(open_atom_idx),
            marker_by_edge=marker_by_edge,
            carrier_contexts_by_edge=carrier_contexts_by_edge,
        ),
        ring_closure=_ring_closure(closure_edge, role="open"),
    )
    close_event = SouthStarTraversalEvent(
        kind="ring_close",
        text="",
        edge=closure_edge,
        begin_atom_idx=close_atom_idx,
        end_atom_idx=open_atom_idx,
        begin_parent_idx=traversal_parent_by_atom.get(close_atom_idx),
        ring_closure=_ring_closure(closure_edge, role="close"),
    )

    with_closure_events: list[SouthStarTraversalEvent] = []
    for event in events:
        with_closure_events.append(event)
        if event.kind != "atom":
            continue
        if event.atom_idx == open_atom_idx:
            with_closure_events.append(open_event)
        elif event.atom_idx == close_atom_idx:
            with_closure_events.append(close_event)
    return tuple(with_closure_events)


def _ring_closure_marker_slot(
    *,
    closure_edge: Edge,
    begin_atom_idx: int,
    end_atom_idx: int,
    begin_parent_idx: int | None,
    marker_by_edge: dict[Edge, str],
    carrier_contexts_by_edge: dict[Edge, tuple[_CarrierContext, ...]],
) -> SouthStarMarkerSlot | None:
    if closure_edge not in marker_by_edge:
        return None
    return _marker_slot(
        edge=closure_edge,
        begin_atom_idx=begin_atom_idx,
        end_atom_idx=end_atom_idx,
        begin_parent_idx=begin_parent_idx,
        syntax_position="ring_open",
        carrier_contexts_by_edge=carrier_contexts_by_edge,
    )


def _ring_closure(
    closure_edge: Edge,
    *,
    role: str,
) -> SouthStarRingClosure:
    return SouthStarRingClosure(
        closure_id=f"{closure_edge[0]}-{closure_edge[1]}",
        label="1",
        role=role,
    )


def _ring_closure_bond_text(mol: Chem.Mol, edge: Edge) -> str:
    bond = mol.GetBondBetweenAtoms(*edge)
    if bond is None:
        raise ValueError(f"ring closure edge {edge!r} is not a bond")
    if bond.GetBondType() == Chem.BondType.SINGLE:
        return ""
    if bond.GetBondType() == Chem.BondType.DOUBLE:
        return "="
    raise NotImplementedError(
        "South Star simple ring traversal currently supports only single- and "
        "double-bond ring closures"
    )


def _atom_event(
    mol: Chem.Mol,
    *,
    atom_idx: int,
    parent_idx: int | None,
    ordered_children: tuple[int, ...],
    tetrahedral_facts_by_atom: dict[int, SouthStarTetrahedralCenterFact],
) -> SouthStarTraversalEvent:
    return SouthStarTraversalEvent(
        kind="atom",
        text=_atom_text_for_traversal(
            mol.GetAtomWithIdx(atom_idx),
            parent_idx=parent_idx,
            ordered_children=ordered_children,
            tetrahedral_facts_by_atom=tetrahedral_facts_by_atom,
        ),
        atom_idx=atom_idx,
    )


def _atom_text_for_traversal(
    atom: Chem.Atom,
    *,
    parent_idx: int | None,
    ordered_children: tuple[int, ...],
    tetrahedral_facts_by_atom: dict[int, SouthStarTetrahedralCenterFact],
) -> str:
    fact = tetrahedral_facts_by_atom.get(atom.GetIdx())
    if fact is None:
        return _atom_text(atom)

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
    return f"[{atom.GetSymbol()}{token}{hydrogen_text}]"


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


def _tetrahedral_facts_by_atom(
    mol: Chem.Mol,
) -> dict[int, SouthStarTetrahedralCenterFact]:
    return {
        fact.center_atom_idx: fact
        for fact in extract_tetrahedral_center_facts(mol)
    }


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
        marker_slot = _marker_slot(
            edge=edge,
            begin_atom_idx=begin_atom_idx,
            end_atom_idx=end_atom_idx,
            begin_parent_idx=begin_parent_idx,
            syntax_position=syntax_position,
            carrier_contexts_by_edge=carrier_contexts_by_edge,
        )
        text = ""
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
