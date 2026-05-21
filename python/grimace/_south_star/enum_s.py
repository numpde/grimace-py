from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from functools import reduce
from itertools import combinations
from itertools import permutations
from itertools import product
from operator import mul

from rdkit import Chem

from grimace._south_star.atom_text import atom_text_for_supported_atom
from grimace._south_star.atom_text import tetrahedral_atom_text_obligation
from grimace._south_star.connected_traversal import (
    connected_graph_plan_from_events,
)
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
from grimace._south_star.molecule_facts import SouthStarMoleculeFacts
from grimace._south_star.policies import DEFAULT_SOUTH_STAR_POLICY_SET
from grimace._south_star.policies import SouthStarPolicySet
from grimace._south_star.reference_model import SouthStarCarrierContext
from grimace._south_star.reference_model import SouthStarConnectedGraphTraversalPlan
from grimace._south_star.reference_model import SouthStarMarkerSlot
from grimace._south_star.reference_model import SouthStarMarkerSlotAssignment
from grimace._south_star.reference_model import SouthStarRingClosure
from grimace._south_star.reference_model import SouthStarTraversal
from grimace._south_star.reference_model import SouthStarTraversalEvent
from grimace._south_star.reference_model import SouthStarTraversalFragment
from grimace._south_star.ring_labels import DEFAULT_RING_CLOSURE_LABEL_POLICY
from grimace._south_star.ring_labels import closure_id_for_edge
from grimace._south_star.support_gates import (
    is_supported_monocycle_with_acyclic_branches,
    is_supported_nonstereo_polycyclic_skeleton,
)
from grimace._south_star.tetrahedral import (
    SouthStarTetrahedralCenterFact,
    SouthStarTetrahedralTraversalObservation,
    emitted_tetrahedral_ligand_order_from_observation,
    preserving_tetrahedral_token,
)


@dataclass(frozen=True, slots=True)
class SouthStarTreeTraversal(SouthStarTraversal):
    connected_graph_plan: SouthStarConnectedGraphTraversalPlan | None = None

    def render(self) -> str:
        return render_south_star_traversal(
            self.events,
            marker_assignments=self.marker_assignments,
        )


_CarrierContext = SouthStarCarrierContext
_TraversalFragment = SouthStarTraversalFragment


@dataclass(frozen=True, slots=True)
class SouthStarFragmentGenerationRecord:
    fragment_id: str
    source_atom_indices: tuple[int, ...]
    source_fragment_smiles: str
    output_count: int


@dataclass(frozen=True, slots=True)
class SouthStarClosureEdgeSetRecord:
    root_atom_idx: int
    closure_edges: tuple[Edge, ...]
    closure_ids: tuple[str, ...]
    closure_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if not (
            len(self.closure_edges)
            == len(self.closure_ids)
            == len(self.closure_labels)
        ):
            raise ValueError("closure-edge set records require aligned fields")
        if len(set(self.closure_labels)) != len(self.closure_labels):
            raise ValueError("closure-edge set records require unique labels")


@dataclass(frozen=True, slots=True)
class SouthStarEnumSGenerationDiagnostics:
    fragment_count: int
    fragment_output_counts: tuple[int, ...]
    fragment_generation_records: tuple[SouthStarFragmentGenerationRecord, ...]
    fragment_order_count: int
    stereo_component_count: int
    traversal_skeleton_count: int
    marker_slot_count: int
    local_assignment_count: int
    solved_assignment_count: int
    estimated_product_size: int
    raw_output_count: int
    output_count: int
    spanning_tree_count: int = 0
    closure_edge_count: int = 0
    closure_label_count: int = 0
    closure_edge_set_records: tuple[SouthStarClosureEdgeSetRecord, ...] = ()

    def __post_init__(self) -> None:
        if self.fragment_count != len(self.fragment_output_counts):
            raise ValueError("fragment count must match fragment output counts")
        if self.fragment_count != len(self.fragment_generation_records):
            raise ValueError("fragment count must match fragment provenance records")
        record_output_counts = tuple(
            record.output_count for record in self.fragment_generation_records
        )
        if self.fragment_output_counts != record_output_counts:
            raise ValueError(
                "fragment output counts must match fragment provenance records"
            )

    @property
    def deduplication_drop_count(self) -> int:
        return self.raw_output_count - self.output_count

    @property
    def deduplicated_output_ratio(self) -> float:
        if self.raw_output_count == 0:
            return 0.0
        return self.output_count / self.raw_output_count


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
class _CombinedMarkerAssignment:
    marker_by_edge: dict[Edge, str]
    component_marker_assignments: tuple[SouthStarComponentMarkerAssignment, ...]


@dataclass(frozen=True, slots=True)
class _SupportGeneration:
    outputs: tuple[str, ...]
    diagnostics: SouthStarEnumSGenerationDiagnostics


@dataclass(frozen=True, slots=True)
class _ConnectedGraphPlanDiagnostics:
    spanning_tree_count: int
    closure_edge_count: int
    closure_label_count: int
    closure_edge_set_records: tuple[SouthStarClosureEdgeSetRecord, ...]


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
    molecule_facts: SouthStarMoleculeFacts | None = None,
) -> SouthStarEnumSPrototypeResult:
    facts = molecule_facts or SouthStarMoleculeFacts.from_mol(mol)
    state = SouthStarComponentSupportState.from_molecule_facts(
        facts,
        annotation_policy=policy_set.annotation_policy,
    )
    if facts.graph_topology.fragment_count > 1:
        generation = _disconnected_generation_for_mol(
            mol,
            molecule_facts=facts,
            policy_set=policy_set,
        )
    else:
        generation = _connected_generation_for_mol(
            mol,
            molecule_facts=facts,
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
    molecule_facts: SouthStarMoleculeFacts,
    state: SouthStarComponentSupportState,
    policy_set: SouthStarPolicySet,
) -> _SupportGeneration:
    traversals = _tree_traversals_for_mol(
        mol,
        molecule_facts=molecule_facts,
        state=state,
    )
    raw_outputs = tuple(traversal.render() for traversal in traversals)
    outputs = policy_set.output_order_policy.deduplicate(raw_outputs)
    marker_slot_count = sum(
        1
        for traversal in traversals
        for event in traversal.events
        if event.marker_slot is not None
    )
    complexity_snapshot = state.complexity_snapshot()
    local_assignment_count = complexity_snapshot.estimated_product_size
    graph_plan_diagnostics = _connected_graph_plan_diagnostics(traversals)
    return _SupportGeneration(
        outputs=outputs,
        diagnostics=SouthStarEnumSGenerationDiagnostics(
            fragment_count=1,
            fragment_output_counts=(len(outputs),),
            fragment_generation_records=(
                _whole_molecule_fragment_record(mol, output_count=len(outputs)),
            ),
            fragment_order_count=1,
            stereo_component_count=complexity_snapshot.component_count,
            traversal_skeleton_count=len(traversals),
            marker_slot_count=marker_slot_count,
            local_assignment_count=local_assignment_count,
            solved_assignment_count=len(traversals),
            estimated_product_size=len(raw_outputs),
            raw_output_count=len(raw_outputs),
            output_count=len(outputs),
            spanning_tree_count=graph_plan_diagnostics.spanning_tree_count,
            closure_edge_count=graph_plan_diagnostics.closure_edge_count,
            closure_label_count=graph_plan_diagnostics.closure_label_count,
            closure_edge_set_records=(
                graph_plan_diagnostics.closure_edge_set_records
            ),
        ),
    )


def _disconnected_generation_for_mol(
    mol: Chem.Mol,
    *,
    molecule_facts: SouthStarMoleculeFacts,
    policy_set: SouthStarPolicySet,
) -> _SupportGeneration:
    if molecule_facts.graph_topology.fragment_count <= 1:
        raise ValueError("disconnected generation requires multiple fragments")
    source_atom_fragments = tuple(tuple(fragment) for fragment in Chem.GetMolFrags(mol))
    fragments = tuple(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True))
    if len(fragments) != molecule_facts.graph_topology.fragment_count:
        raise AssertionError(
            "South Star disconnected generation requires fragment extraction "
            "to match molecule topology facts"
        )
    if len(source_atom_fragments) != len(fragments):
        raise AssertionError(
            "South Star disconnected generation requires source fragment "
            "provenance to match fragment molecule extraction"
        )
    fragment_generations = tuple(
        _connected_generation_for_mol(
            fragment,
            molecule_facts=fragment_facts,
            state=SouthStarComponentSupportState.from_molecule_facts(
                fragment_facts,
                annotation_policy=policy_set.annotation_policy,
            ),
            policy_set=policy_set,
        )
        for fragment_facts, fragment in (
            (SouthStarMoleculeFacts.from_mol(fragment), fragment)
            for fragment in fragments
        )
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
            fragment_generation_records=_fragment_generation_records(
                source_atom_fragments=source_atom_fragments,
                fragments=fragments,
                fragment_generations=fragment_generations,
            ),
            fragment_order_count=composition.fragment_order_count,
            stereo_component_count=sum(
                generation.diagnostics.stereo_component_count
                for generation in fragment_generations
            ),
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
            raw_output_count=composition.estimated_product_size,
            output_count=len(composition.outputs),
            spanning_tree_count=sum(
                generation.diagnostics.spanning_tree_count
                for generation in fragment_generations
            ),
            closure_edge_count=sum(
                generation.diagnostics.closure_edge_count
                for generation in fragment_generations
            ),
            closure_label_count=sum(
                generation.diagnostics.closure_label_count
                for generation in fragment_generations
            ),
            closure_edge_set_records=tuple(
                record
                for generation in fragment_generations
                for record in generation.diagnostics.closure_edge_set_records
            ),
        ),
    )


def _whole_molecule_fragment_record(
    mol: Chem.Mol,
    *,
    output_count: int,
) -> SouthStarFragmentGenerationRecord:
    return SouthStarFragmentGenerationRecord(
        fragment_id="fragment:0",
        source_atom_indices=tuple(atom.GetIdx() for atom in mol.GetAtoms()),
        source_fragment_smiles=_source_fragment_smiles(mol),
        output_count=output_count,
    )


def _fragment_generation_records(
    *,
    source_atom_fragments: tuple[tuple[int, ...], ...],
    fragments: tuple[Chem.Mol, ...],
    fragment_generations: tuple[_SupportGeneration, ...],
) -> tuple[SouthStarFragmentGenerationRecord, ...]:
    if not (
        len(source_atom_fragments)
        == len(fragments)
        == len(fragment_generations)
    ):
        raise AssertionError(
            "South Star fragment provenance requires source atoms, fragment "
            "molecules, and generation records to align"
        )
    return tuple(
        SouthStarFragmentGenerationRecord(
            fragment_id=f"fragment:{fragment_idx}",
            source_atom_indices=source_atom_indices,
            source_fragment_smiles=_source_fragment_smiles(fragment),
            output_count=len(generation.outputs),
        )
        for fragment_idx, (source_atom_indices, fragment, generation) in enumerate(
            zip(source_atom_fragments, fragments, fragment_generations, strict=True)
        )
    )


def _source_fragment_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=False, isomericSmiles=True)


def _connected_graph_plan_diagnostics(
    traversals: tuple[SouthStarTreeTraversal, ...],
) -> _ConnectedGraphPlanDiagnostics:
    plans = tuple(
        traversal.connected_graph_plan
        for traversal in traversals
        if traversal.connected_graph_plan is not None
    )
    if not plans:
        return _ConnectedGraphPlanDiagnostics(
            spanning_tree_count=0,
            closure_edge_count=0,
            closure_label_count=0,
            closure_edge_set_records=(),
        )
    return _ConnectedGraphPlanDiagnostics(
        spanning_tree_count=len(
            {
                frozenset(edge.edge for edge in plan.tree_edges)
                for plan in plans
            }
        ),
        closure_edge_count=max(len(plan.closure_edges) for plan in plans),
        closure_label_count=max(
            len({edge.label for edge in plan.closure_edges})
            for plan in plans
        ),
        closure_edge_set_records=tuple(
            SouthStarClosureEdgeSetRecord(
                root_atom_idx=plan.root_atom_idx,
                closure_edges=tuple(edge.edge for edge in plan.closure_edges),
                closure_ids=tuple(edge.closure_id for edge in plan.closure_edges),
                closure_labels=tuple(edge.label for edge in plan.closure_edges),
            )
            for plan in plans
        ),
    )


def mol_to_smiles_enum_s_tree_traversals_for_case(
    case: object,
) -> tuple[SouthStarTreeTraversal, ...]:
    mol = _parse_smiles(case.source_smiles)
    molecule_facts = SouthStarMoleculeFacts.from_mol(mol)
    state = SouthStarComponentSupportState.from_molecule_facts(molecule_facts)
    return _tree_traversals_for_mol(
        mol,
        molecule_facts=molecule_facts,
        state=state,
    )


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
    molecule_facts: SouthStarMoleculeFacts,
    state: SouthStarComponentSupportState,
) -> tuple[SouthStarTreeTraversal, ...]:
    _assert_state_uses_molecule_facts(state, molecule_facts)
    return tuple(
        traversal
        for combined_assignment in _combined_marker_assignments(state)
        for traversal in _tree_traversals_for_marker_assignment(
            mol,
            molecule_facts=molecule_facts,
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
    molecule_facts: SouthStarMoleculeFacts,
    state: SouthStarComponentSupportState,
    marker_by_edge: dict[Edge, str],
    component_marker_assignments: tuple[SouthStarComponentMarkerAssignment, ...],
) -> tuple[SouthStarTreeTraversal, ...]:
    _assert_tree_traversal_supported(mol, molecule_facts=molecule_facts)
    if is_supported_monocycle_with_acyclic_branches(mol):
        return _ring_system_traversals(
            mol,
            molecule_facts=molecule_facts,
            state=state,
            closure_edge_sets=tuple(
                (edge,) for edge in _supported_single_ring_edges(mol)
            ),
            marker_by_edge=marker_by_edge,
            component_marker_assignments=component_marker_assignments,
        )
    if is_supported_nonstereo_polycyclic_skeleton(mol):
        return _ring_system_traversals(
            mol,
            molecule_facts=molecule_facts,
            state=state,
            closure_edge_sets=_supported_polycyclic_closure_edge_sets(mol),
            marker_by_edge=marker_by_edge,
            component_marker_assignments=component_marker_assignments,
        )

    carrier_contexts_by_edge = _carrier_contexts_by_edge(
        mol,
        marker_by_edge=marker_by_edge,
    )
    tetrahedral_facts_by_atom = _tetrahedral_facts_by_atom(molecule_facts)
    return tuple(
        _with_solved_marker_assignments(
            state,
            _tree_traversal(
                root_atom_idx=root_idx,
                events=fragment.events,
                marker_assignments=(),
                component_marker_assignments=component_marker_assignments,
            ),
        )
        for root_idx in molecule_facts.graph_topology.atom_indices
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


def _tree_traversal(
    *,
    root_atom_idx: int,
    events: tuple[SouthStarTraversalEvent, ...],
    marker_assignments: tuple[SouthStarMarkerSlotAssignment, ...],
    component_marker_assignments: tuple[SouthStarComponentMarkerAssignment, ...],
) -> SouthStarTreeTraversal:
    return SouthStarTreeTraversal(
        root_atom_idx=root_atom_idx,
        events=events,
        marker_assignments=marker_assignments,
        component_marker_assignments=component_marker_assignments,
        connected_graph_plan=connected_graph_plan_from_events(
            root_atom_idx=root_atom_idx,
            events=events,
        ),
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


def _assert_tree_traversal_supported(
    mol: Chem.Mol,
    *,
    molecule_facts: SouthStarMoleculeFacts,
) -> None:
    topology = molecule_facts.graph_topology
    if topology.atom_count == 0:
        raise ValueError("South Star graph-native tree traversal requires atoms")
    if not topology.connected:
        raise NotImplementedError(
            "South Star graph-native tree traversal currently requires one "
            "connected component"
        )
    if topology.acyclic_connected_tree:
        return
    if is_supported_monocycle_with_acyclic_branches(mol):
        return
    if is_supported_nonstereo_polycyclic_skeleton(mol):
        return
    raise NotImplementedError(
        "South Star graph-native tree traversal currently requires one "
        "connected acyclic component, one supported monocycle, or one "
        "supported nonstereo polycyclic skeleton"
    )


def _assert_state_uses_molecule_facts(
    state: SouthStarComponentSupportState,
    molecule_facts: SouthStarMoleculeFacts,
) -> None:
    if state.molecule_facts is not molecule_facts:
        raise AssertionError(
            "South Star generation requires component support state to share "
            "the supplied molecule facts"
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


def _ring_system_traversals(
    mol: Chem.Mol,
    *,
    molecule_facts: SouthStarMoleculeFacts,
    state: SouthStarComponentSupportState,
    closure_edge_sets: tuple[tuple[Edge, ...], ...],
    marker_by_edge: dict[Edge, str],
    component_marker_assignments: tuple[SouthStarComponentMarkerAssignment, ...],
) -> tuple[SouthStarTreeTraversal, ...]:
    carrier_contexts_by_edge = _carrier_contexts_by_edge(
        mol,
        marker_by_edge=marker_by_edge,
    )
    return tuple(
        _with_solved_marker_assignments(
            state,
            _tree_traversal(
                root_atom_idx=root_idx,
                events=_with_ring_closure_events(
                    mol,
                    fragment.events,
                    closure_edges=closure_edges,
                    marker_by_edge=marker_by_edge,
                    carrier_contexts_by_edge=carrier_contexts_by_edge,
                ),
                marker_assignments=(),
                component_marker_assignments=component_marker_assignments,
            ),
        )
        for root_idx in molecule_facts.graph_topology.atom_indices
        for closure_edges in closure_edge_sets
        for fragment in _atom_subtree_event_variants(
            mol,
            atom_idx=root_idx,
            parent_idx=None,
            visited=frozenset(),
            blocked_edges=frozenset(closure_edges),
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


def _supported_single_ring_edges(mol: Chem.Mol) -> tuple[Edge, ...]:
    return tuple(
        edge
        for edge in _single_ring_edges(mol)
        if _ring_closure_edge_supported(mol, edge=edge)
    )


def _supported_polycyclic_closure_edge_sets(
    mol: Chem.Mol,
) -> tuple[tuple[Edge, ...], ...]:
    closure_edge_count = mol.GetNumBonds() - mol.GetNumAtoms() + 1
    if closure_edge_count <= 1:
        raise ValueError("polycyclic closure-edge sets require cyclomatic number > 1")
    candidate_edges = []
    for bond in mol.GetBonds():
        if not bond.IsInRing():
            continue
        edge = normalized_edge((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        if _ring_closure_edge_supported(mol, edge=edge):
            candidate_edges.append(edge)
    closure_edge_sets = tuple(
        closure_edges
        for closure_edges in combinations(tuple(candidate_edges), closure_edge_count)
        if _blocked_edges_form_spanning_tree(
            mol,
            blocked_edges=frozenset(closure_edges),
        )
    )
    if not closure_edge_sets:
        raise NotImplementedError(
            "South Star polycyclic traversal found no supported spanning-tree "
            "closure-edge choices"
        )
    return closure_edge_sets


def _blocked_edges_form_spanning_tree(
    mol: Chem.Mol,
    *,
    blocked_edges: frozenset[Edge],
) -> bool:
    remaining_edge_count = mol.GetNumBonds() - len(blocked_edges)
    if remaining_edge_count != mol.GetNumAtoms() - 1:
        return False
    if mol.GetNumAtoms() == 0:
        return False
    visited = {0}
    stack = [0]
    while stack:
        atom_idx = stack.pop()
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            edge = normalized_edge((atom_idx, neighbor_idx))
            if edge in blocked_edges or neighbor_idx in visited:
                continue
            visited.add(neighbor_idx)
            stack.append(neighbor_idx)
    return len(visited) == mol.GetNumAtoms()


def _ring_closure_edge_supported(mol: Chem.Mol, *, edge: Edge) -> bool:
    bond = mol.GetBondBetweenAtoms(*edge)
    if bond is None:
        raise ValueError(f"ring closure edge {edge!r} is not a bond")
    return bond.GetBondType() in {Chem.BondType.SINGLE, Chem.BondType.DOUBLE}


def _with_ring_closure_events(
    mol: Chem.Mol,
    events: tuple[SouthStarTraversalEvent, ...],
    *,
    closure_edges: tuple[Edge, ...],
    marker_by_edge: dict[Edge, str],
    carrier_contexts_by_edge: dict[Edge, tuple[_CarrierContext, ...]],
) -> tuple[SouthStarTraversalEvent, ...]:
    traversal_parent_by_atom = {
        event.end_atom_idx: event.begin_atom_idx
        for event in events
        if event.kind == "bond"
        and event.begin_atom_idx is not None
        and event.end_atom_idx is not None
    }
    closure_events_by_atom: dict[int, list[SouthStarTraversalEvent]] = {}
    label_assignments = DEFAULT_RING_CLOSURE_LABEL_POLICY.assignments_for_edges(
        closure_edges
    )
    for assignment in label_assignments:
        open_event, close_event = _ring_closure_event_pair(
            mol,
            events,
            closure_edge=assignment.edge,
            closure_label=assignment.label,
            traversal_parent_by_atom=traversal_parent_by_atom,
            marker_by_edge=marker_by_edge,
            carrier_contexts_by_edge=carrier_contexts_by_edge,
        )
        if open_event.begin_atom_idx is None or close_event.begin_atom_idx is None:
            raise AssertionError("ring closure events must carry begin atoms")
        closure_events_by_atom.setdefault(open_event.begin_atom_idx, []).append(
            open_event
        )
        closure_events_by_atom.setdefault(close_event.begin_atom_idx, []).append(
            close_event
        )
    with_closure_events: list[SouthStarTraversalEvent] = []
    for event in events:
        with_closure_events.append(event)
        if event.kind != "atom" or event.atom_idx is None:
            continue
        with_closure_events.extend(closure_events_by_atom.get(event.atom_idx, ()))
    return tuple(with_closure_events)


def _ring_closure_event_pair(
    mol: Chem.Mol,
    events: tuple[SouthStarTraversalEvent, ...],
    *,
    closure_edge: Edge,
    closure_label: str,
    traversal_parent_by_atom: dict[int, int],
    marker_by_edge: dict[Edge, str],
    carrier_contexts_by_edge: dict[Edge, tuple[_CarrierContext, ...]],
) -> tuple[SouthStarTraversalEvent, SouthStarTraversalEvent]:
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
        ring_closure=_ring_closure(
            closure_edge,
            label=closure_label,
            role="open",
        ),
    )
    close_event = SouthStarTraversalEvent(
        kind="ring_close",
        text="",
        edge=closure_edge,
        begin_atom_idx=close_atom_idx,
        end_atom_idx=open_atom_idx,
        begin_parent_idx=traversal_parent_by_atom.get(close_atom_idx),
        ring_closure=_ring_closure(
            closure_edge,
            label=closure_label,
            role="close",
        ),
    )
    return open_event, close_event


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
    label: str,
    role: str,
) -> SouthStarRingClosure:
    return SouthStarRingClosure(
        closure_id=closure_id_for_edge(closure_edge),
        label=label,
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
        center_atom_idx=fact.center_atom_idx,
        parent_idx=parent_idx,
        ordered_children=ordered_children,
        implicit_hydrogen_count=fact.implicit_hydrogen_count,
    )
    token = preserving_tetrahedral_token(
        source_token=fact.source_token,
        source_ligand_order=fact.source_ligand_order,
        emitted_ligand_order=emitted_ligand_order,
    )
    return tetrahedral_atom_text_obligation(
        atom,
        stereo_token=token,
        implicit_hydrogen_count=fact.implicit_hydrogen_count,
    ).emitted_text


def _emitted_tetrahedral_ligand_order(
    *,
    center_atom_idx: int,
    parent_idx: int | None,
    ordered_children: tuple[int, ...],
    implicit_hydrogen_count: int,
) -> tuple[str, ...]:
    return emitted_tetrahedral_ligand_order_from_observation(
        SouthStarTetrahedralTraversalObservation(
            center_atom_idx=center_atom_idx,
            parent_atom_idx=parent_idx,
            child_atom_indices=ordered_children,
            ring_closure_ligand_atom_indices=(),
            ring_closure_labels=(),
            implicit_hydrogen_count=implicit_hydrogen_count,
        )
    )


def _tetrahedral_facts_by_atom(
    molecule_facts: SouthStarMoleculeFacts,
) -> dict[int, SouthStarTetrahedralCenterFact]:
    return {
        fact.center_atom_idx: fact
        for fact in molecule_facts.tetrahedral_center_facts
    }


def _atom_text(atom: Chem.Atom) -> str:
    return atom_text_for_supported_atom(atom)


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
        syntax_position=syntax_position,
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
