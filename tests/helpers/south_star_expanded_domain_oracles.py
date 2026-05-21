from __future__ import annotations

"""Test evidence helpers for expanded South Star fixture domains.

The remaining `independent_*` helpers are temporary witnesses. Helpers without
that prefix intentionally consume shared EnumS traversal/equation records so
they do not grow into separate support universes.

The `TemporarySouthStar*Witness*` records below are fixture evidence envelopes,
not reference-model vocabulary. Shared constraint-family records live under
`grimace._south_star`.
"""

from dataclasses import dataclass
from functools import reduce
from operator import mul

from rdkit import Chem

from grimace._south_star.annotation_policy import Edge
from grimace._south_star.annotation_policy import normalized_edge
from grimace._south_star.components import SouthStarSemanticStereoComponent
from grimace._south_star.component_support_state import (
    SouthStarComponentSupportState,
)
from grimace._south_star.enum_s import (
    SouthStarFragmentGenerationRecord,
    mol_to_smiles_enum_s_graph_native,
    mol_to_smiles_enum_s_tree_traversals_for_case,
    render_south_star_tree_traversal,
)
from grimace._south_star.fragments import SouthStarFragmentSupport
from grimace._south_star.fragments import compose_disconnected_fragment_supports
from grimace._south_star.marker_equations import SouthStarMarkerSlotParityEquation
from grimace._south_star.marker_equations import (
    marker_slot_parity_equations_for_traversal,
)
from grimace._south_star.molecule_facts import SouthStarMoleculeFacts
from grimace._south_star.parity_solver import solve_marker_slot_parity_equations
from grimace._south_star.reference_model import SouthStarConnectedGraphTraversalPlan
from grimace._south_star.tetrahedral import SouthStarTetrahedralCenterFact
from grimace._south_star.tetrahedral import (
    SouthStarTetrahedralTraversalProofInput,
    SouthStarTetrahedralTraversalTokenDiagnostic,
)
from grimace._south_star.tetrahedral import extract_tetrahedral_center_facts
from grimace._south_star.tetrahedral import (
    tetrahedral_traversal_observation_from_connected_graph_plan,
)
from grimace._south_star.tetrahedral import tetrahedral_traversal_proof_input
from grimace._south_star.tetrahedral import tetrahedral_traversal_token_diagnostic
from tests.helpers.south_star_domain_manifest import (
    SOUTH_STAR_NONSTEREO_MONOCYCLE_UNIFIED_REFERENCE_AUTHORITY,
    SOUTH_STAR_SINGLE_ATOM_ATOM_TEXT_UNIFIED_REFERENCE_AUTHORITY,
    SOUTH_STAR_UNIFIED_REFERENCE_AUTHORITIES,
)
from tests.helpers.south_star_exact_support import (
    SouthStarExpandedSupportCase,
    load_south_star_exact_first_domain_cases,
)
from tests.helpers.south_star_first_domain_proof_inputs import (
    first_domain_renderer_proof_from_shared_spine,
)
from tests.helpers.south_star_semantic_oracle import parse_smiles
from tests.helpers.south_star_unified_reference import (
    is_nonstereo_monocycle_ring_traversal_domain,
    is_single_atom_atom_text_domain,
    nonstereo_monocycle_support_from_shared_spine,
    single_atom_atom_text_support_from_facts,
)


@dataclass(frozen=True, slots=True)
class SouthStarRingStereoSupportProof:
    outputs: tuple[str, ...]
    equations: tuple[SouthStarMarkerSlotParityEquation, ...]
    closure_edge_count: int
    closure_edge_set_count: int
    closure_edges_per_traversal: int
    closure_label_count: int
    marker_slot_count: int
    marker_assignment_count: int


@dataclass(frozen=True, slots=True)
class SouthStarTetrahedralAtomStereoProof:
    outputs: tuple[str, ...]
    diagnostics: tuple[SouthStarTetrahedralTraversalTokenDiagnostic, ...]
    proof_inputs: tuple[SouthStarTetrahedralTraversalProofInput, ...]
    output_count: int
    expected_output_count: int
    fixture_cross_check_passed: bool


@dataclass(frozen=True, slots=True)
class SouthStarIndependentDirectionalComponentProductProof:
    outputs: tuple[str, ...]
    component_ids: tuple[str, ...]
    component_local_assignment_counts: tuple[int, ...]
    component_assignment_product_size: int
    traversal_skeletons_per_component_assignment: int
    traversal_count: int
    marker_slot_count: int
    equation_count: int
    solver_assignment_count: int
    raw_output_count: int
    output_count: int
    disjoint_component_carriers: bool
    all_components_uncoupled: bool
    all_equations_component_local: bool
    all_traversals_have_component_product_assignment: bool
    all_solver_assignments_match_traversal: bool
    expected_support_strings_used: bool


@dataclass(frozen=True, slots=True)
class SouthStarDirectionalTetrahedralCompositionProof:
    outputs: tuple[str, ...]
    directional_component_ids: tuple[str, ...]
    tetrahedral_center_atom_indices: tuple[int, ...]
    component_assignment_product_size: int
    traversal_count: int
    marker_slot_count: int
    equation_count: int
    solver_assignment_count: int
    renderer_input_count: int
    tetrahedral_proof_input_count: int
    tetrahedral_diagnostic_count: int
    raw_output_count: int
    output_count: int
    all_traversals_have_directional_obligations: bool
    all_traversals_have_tetrahedral_obligations: bool
    all_solver_assignments_match_traversal: bool
    all_tetrahedral_tokens_preserve_orientation: bool
    all_tetrahedral_renderer_inputs_match_proof: bool
    expected_support_strings_used: bool


@dataclass(frozen=True, slots=True)
class TemporarySouthStarDisconnectedCompositionWitnessEvidence:
    outputs: tuple[str, ...]
    fragment_count: int
    fragment_output_counts: tuple[int, ...]
    fragment_generation_records: tuple[SouthStarFragmentGenerationRecord, ...]
    fragment_order_policy: str
    fragment_order_count: int
    estimated_product_size: int


@dataclass(frozen=True, slots=True)
class SouthStarFragmentAuthorityCase:
    case_id: str
    source_smiles: str


@dataclass(frozen=True, slots=True)
class SouthStarDisconnectedCompositionAlgebraProof:
    case_id: str
    support_authority: str
    fragment_count: int
    fragment_source_smiles: tuple[str, ...]
    fragment_support_authorities: tuple[str, ...]
    fragment_output_counts: tuple[int, ...]
    fragment_order_policy: str
    fragment_order_count: int
    output_order_policy: str
    estimated_product_size: int
    composed_outputs: tuple[str, ...]
    graph_native_outputs: tuple[str, ...]
    support_authority_promoted: bool


@dataclass(frozen=True, slots=True)
class SouthStarDisconnectedMixedStereoCompositionProof:
    outputs: tuple[str, ...]
    fragment_count: int
    fragment_source_smiles: tuple[str, ...]
    fragment_feature_classes: tuple[str, ...]
    fragment_output_counts: tuple[int, ...]
    fragment_order_policy: str
    fragment_order_count: int
    output_order_policy: str
    per_fragment_support_product_size: int
    estimated_product_size: int
    graph_native_outputs: tuple[str, ...]
    all_fragments_have_shared_spine_proofs: bool
    dot_rendering_in_all_outputs: bool
    first_occurrence_deduplication_preserved_product: bool
    expected_support_strings_used: bool


@dataclass(frozen=True, slots=True)
class SouthStarRingCoreProofRecord:
    case_id: str
    root_atom_idx: int
    closure_edges: tuple[Edge, ...]
    closure_ids: tuple[str, ...]
    closure_labels: tuple[str, ...]
    closure_endpoint_roles: tuple[str, ...]
    closure_endpoint_labels: tuple[str, ...]
    closure_event_kinds: tuple[str, ...]
    closure_event_roles: tuple[str, ...]
    closure_event_labels: tuple[str, ...]
    closure_open_bond_texts: tuple[str, ...]
    marker_slot_count: int
    renderer_input_count: int


def shared_saturated_monocycle_support_for_case(
    case: SouthStarExpandedSupportCase,
) -> tuple[str, ...]:
    mol = parse_smiles(case.source_smiles)
    _assert_saturated_monocycle_domain(mol)

    return _shared_traversal_support_for_case(case)


def shared_nonstereo_monocycle_support_for_case(
    case: SouthStarExpandedSupportCase,
) -> tuple[str, ...]:
    mol = parse_smiles(case.source_smiles)
    _assert_nonstereo_monocycle_domain(mol)

    return _shared_traversal_support_for_case(case)


def ring_core_proof_records_for_case(
    case: SouthStarExpandedSupportCase,
) -> tuple[SouthStarRingCoreProofRecord, ...]:
    traversals = mol_to_smiles_enum_s_tree_traversals_for_case(case)
    records = []
    for traversal in traversals:
        plan = traversal.connected_graph_plan
        if plan is None or not plan.closure_edges:
            continue
        closure_events = tuple(
            event for event in traversal.events if event.ring_closure is not None
        )
        records.append(
            SouthStarRingCoreProofRecord(
                case_id=case.case_id,
                root_atom_idx=traversal.root_atom_idx,
                closure_edges=tuple(edge.edge for edge in plan.closure_edges),
                closure_ids=tuple(edge.closure_id for edge in plan.closure_edges),
                closure_labels=tuple(edge.label for edge in plan.closure_edges),
                closure_endpoint_roles=tuple(
                    endpoint.role for endpoint in plan.closure_endpoints
                ),
                closure_endpoint_labels=tuple(
                    endpoint.label for endpoint in plan.closure_endpoints
                ),
                closure_event_kinds=tuple(event.kind for event in closure_events),
                closure_event_roles=tuple(
                    event.ring_closure.role
                    for event in closure_events
                    if event.ring_closure is not None
                ),
                closure_event_labels=tuple(
                    event.ring_closure.label
                    for event in closure_events
                    if event.ring_closure is not None
                ),
                closure_open_bond_texts=tuple(
                    event.text for event in closure_events if event.kind == "ring_open"
                ),
                marker_slot_count=sum(
                    1 for event in traversal.events if event.marker_slot is not None
                ),
                renderer_input_count=sum(
                    1 for event in traversal.events if event.renderer_input is not None
                ),
            )
        )
    return tuple(records)


def _shared_traversal_support_for_case(
    case: SouthStarExpandedSupportCase,
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            render_south_star_tree_traversal(traversal)
            for traversal in mol_to_smiles_enum_s_tree_traversals_for_case(case)
        )
    )


def shared_disconnected_composition_support_for_case(
    case: SouthStarExpandedSupportCase,
) -> TemporarySouthStarDisconnectedCompositionWitnessEvidence:
    result = mol_to_smiles_enum_s_graph_native(
        case.source_smiles,
        case_id=case.case_id,
    )
    diagnostics = result.generation_diagnostics
    if diagnostics is None:
        raise ValueError("disconnected composition evidence requires diagnostics")
    return TemporarySouthStarDisconnectedCompositionWitnessEvidence(
        outputs=result.outputs,
        fragment_count=diagnostics.fragment_count,
        fragment_output_counts=diagnostics.fragment_output_counts,
        fragment_generation_records=diagnostics.fragment_generation_records,
        fragment_order_policy=result.fragment_order_policy,
        fragment_order_count=diagnostics.fragment_order_count,
        estimated_product_size=diagnostics.estimated_product_size,
    )


def disconnected_composition_algebra_proof_for_case(
    case: SouthStarExpandedSupportCase,
) -> SouthStarDisconnectedCompositionAlgebraProof:
    """Recompose disconnected support from opaque fragment supports."""

    graph_native = shared_disconnected_composition_support_for_case(case)
    fragment_supports = []
    fragment_authorities = []
    for record in graph_native.fragment_generation_records:
        fragment_result = mol_to_smiles_enum_s_graph_native(
            record.source_fragment_smiles,
            case_id=f"{case.case_id}:{record.fragment_id}",
        )
        if len(fragment_result.outputs) != record.output_count:
            raise AssertionError(
                f"fragment {record.fragment_id!r} output count changed while "
                "building disconnected composition proof"
            )
        fragment_supports.append(
            SouthStarFragmentSupport(
                fragment_id=record.fragment_id,
                outputs=fragment_result.outputs,
            )
        )
        fragment_authorities.append(
            _fragment_support_authority(
                source_smiles=record.source_fragment_smiles,
                outputs=fragment_result.outputs,
            )
        )

    composition = compose_disconnected_fragment_supports(tuple(fragment_supports))
    return SouthStarDisconnectedCompositionAlgebraProof(
        case_id=case.case_id,
        support_authority=case.support_authority,
        fragment_count=composition.fragment_count,
        fragment_source_smiles=tuple(
            record.source_fragment_smiles
            for record in graph_native.fragment_generation_records
        ),
        fragment_support_authorities=tuple(fragment_authorities),
        fragment_output_counts=composition.fragment_output_counts,
        fragment_order_policy=composition.fragment_order_policy,
        fragment_order_count=composition.fragment_order_count,
        output_order_policy=composition.output_order_policy,
        estimated_product_size=composition.estimated_product_size,
        composed_outputs=composition.outputs,
        graph_native_outputs=graph_native.outputs,
        support_authority_promoted=(
            case.support_authority in SOUTH_STAR_UNIFIED_REFERENCE_AUTHORITIES
        ),
    )


def disconnected_mixed_stereo_composition_proof_for_case(
    case: SouthStarExpandedSupportCase,
) -> SouthStarDisconnectedMixedStereoCompositionProof:
    """Prove mixed-stereo disconnected support by fragment composition."""

    graph_native = shared_disconnected_composition_support_for_case(case)
    fragment_supports = []
    fragment_classes = []
    fragment_shared_spine_proofs = []
    for record in graph_native.fragment_generation_records:
        fragment_result = mol_to_smiles_enum_s_graph_native(
            record.source_fragment_smiles,
            case_id=f"{case.case_id}:{record.fragment_id}",
        )
        fragment_supports.append(
            SouthStarFragmentSupport(
                fragment_id=record.fragment_id,
                outputs=fragment_result.outputs,
            )
        )
        fragment_class, proof_passed = _mixed_stereo_fragment_class_and_proof(
            case_id=f"{case.case_id}:{record.fragment_id}",
            source_smiles=record.source_fragment_smiles,
            expected_outputs=fragment_result.outputs,
        )
        fragment_classes.append(fragment_class)
        fragment_shared_spine_proofs.append(proof_passed)

    composition = compose_disconnected_fragment_supports(tuple(fragment_supports))
    per_fragment_support_product_size = reduce(
        mul,
        composition.fragment_output_counts,
        1,
    )
    return SouthStarDisconnectedMixedStereoCompositionProof(
        outputs=composition.outputs,
        fragment_count=composition.fragment_count,
        fragment_source_smiles=tuple(
            record.source_fragment_smiles
            for record in graph_native.fragment_generation_records
        ),
        fragment_feature_classes=tuple(fragment_classes),
        fragment_output_counts=composition.fragment_output_counts,
        fragment_order_policy=composition.fragment_order_policy,
        fragment_order_count=composition.fragment_order_count,
        output_order_policy=composition.output_order_policy,
        per_fragment_support_product_size=per_fragment_support_product_size,
        estimated_product_size=composition.estimated_product_size,
        graph_native_outputs=graph_native.outputs,
        all_fragments_have_shared_spine_proofs=all(fragment_shared_spine_proofs),
        dot_rendering_in_all_outputs=all(
            output.count(".") == composition.fragment_count - 1
            for output in composition.outputs
        ),
        first_occurrence_deduplication_preserved_product=(
            len(composition.outputs) == composition.estimated_product_size
        ),
        expected_support_strings_used=False,
    )


def _mixed_stereo_fragment_class_and_proof(
    *,
    case_id: str,
    source_smiles: str,
    expected_outputs: tuple[str, ...],
) -> tuple[str, bool]:
    mol = parse_smiles(source_smiles)
    facts = SouthStarMoleculeFacts.from_mol(mol)
    tetrahedral_facts = extract_tetrahedral_center_facts(mol)
    fragment_case = SouthStarFragmentAuthorityCase(
        case_id=case_id,
        source_smiles=source_smiles,
    )
    if facts.components and not tetrahedral_facts:
        proof = first_domain_renderer_proof_from_shared_spine(fragment_case)
        return (
            "directional_marker_equation_fragment",
            proof.rendered_outputs == expected_outputs
            and proof.all_rendered_outputs_have_marker_equation_proofs
            and not proof.expected_support_strings_used,
        )
    if tetrahedral_facts and not facts.components:
        proof = _tetrahedral_fragment_shared_spine_proof(
            fragment_case,
            expected_outputs=expected_outputs,
        )
        return ("tetrahedral_renderer_obligation_fragment", proof)
    raise NotImplementedError(
        "mixed-stereo disconnected composition currently expects one "
        "directional fragment and one tetrahedral fragment"
    )


def _tetrahedral_fragment_shared_spine_proof(
    case: SouthStarFragmentAuthorityCase,
    *,
    expected_outputs: tuple[str, ...],
) -> bool:
    mol = parse_smiles(case.source_smiles)
    facts_by_atom = {
        fact.center_atom_idx: fact for fact in extract_tetrahedral_center_facts(mol)
    }
    traversals = mol_to_smiles_enum_s_tree_traversals_for_case(case)
    outputs = tuple(
        dict.fromkeys(
            render_south_star_tree_traversal(traversal)
            for traversal in traversals
        )
    )
    diagnostics = tuple(
        _tetrahedral_token_diagnostic_for_atom_event(
            traversal.connected_graph_plan,
            center_atom_idx=event.atom_idx,
            emitted_token=_tetrahedral_token_from_atom_text(event.text),
            facts_by_atom=facts_by_atom,
        )
        for traversal in traversals
        for event in traversal.events
        if event.kind == "atom"
        and event.atom_idx in facts_by_atom
        and event.atom_idx is not None
    )
    return (
        outputs == expected_outputs
        and len(diagnostics) == len(traversals) * len(facts_by_atom)
        and all(diagnostic.preserves_orientation for diagnostic in diagnostics)
    )


def _fragment_support_authority(*, source_smiles: str, outputs: tuple[str, ...]) -> str:
    facts = SouthStarMoleculeFacts.from_mol(parse_smiles(source_smiles))
    if is_single_atom_atom_text_domain(facts):
        support = single_atom_atom_text_support_from_facts(facts)
        if support.support == outputs:
            return SOUTH_STAR_SINGLE_ATOM_ATOM_TEXT_UNIFIED_REFERENCE_AUTHORITY

    if is_nonstereo_monocycle_ring_traversal_domain(facts):
        support = nonstereo_monocycle_support_from_shared_spine(
            SouthStarFragmentAuthorityCase(
                case_id=f"fragment_authority:{source_smiles}",
                source_smiles=source_smiles,
            )
        )
        if support.support == outputs:
            return SOUTH_STAR_NONSTEREO_MONOCYCLE_UNIFIED_REFERENCE_AUTHORITY

    for exact_case in load_south_star_exact_first_domain_cases():
        if (
            exact_case.source_smiles == source_smiles
            and exact_case.expected_support == outputs
        ):
            return exact_case.support_authority

    return "unproven_fragment_support"


def shared_ring_stereo_monocycle_support_for_case(
    case: SouthStarExpandedSupportCase,
) -> SouthStarRingStereoSupportProof:
    """Check ring-stereo fixtures through shared traversal/equation records."""
    mol = parse_smiles(case.source_smiles)
    _assert_ring_stereo_monocycle_domain(mol)
    return _shared_ring_stereo_support_for_case(case, mol=mol)


def shared_polycyclic_ring_stereo_support_for_case(
    case: SouthStarExpandedSupportCase,
) -> SouthStarRingStereoSupportProof:
    """Check polycyclic ring-stereo fixtures through shared equations."""
    mol = parse_smiles(case.source_smiles)
    if len(Chem.GetMolFrags(mol)) != 1:
        raise NotImplementedError("polycyclic ring-stereo proof requires one component")
    if len(mol.GetRingInfo().BondRings()) <= 1:
        raise NotImplementedError("polycyclic ring-stereo proof requires >1 ring")
    return _shared_ring_stereo_support_for_case(case, mol=mol)


def _shared_ring_stereo_support_for_case(
    case: SouthStarExpandedSupportCase,
    *,
    mol: Chem.Mol,
) -> SouthStarRingStereoSupportProof:
    state = SouthStarComponentSupportState.from_mol(mol)
    traversals = mol_to_smiles_enum_s_tree_traversals_for_case(case)
    plans = tuple(
        traversal.connected_graph_plan
        for traversal in traversals
        if traversal.connected_graph_plan is not None
    )
    closure_events = tuple(
        event
        for traversal in traversals
        for event in traversal.events
        if event.ring_closure is not None
    )
    equations = tuple(
        dict.fromkeys(
            equation
            for traversal in traversals
            for equation in marker_slot_parity_equations_for_traversal(
                state,
                traversal,
            )
        )
    )

    return SouthStarRingStereoSupportProof(
        outputs=tuple(
            dict.fromkeys(
                render_south_star_tree_traversal(traversal)
                for traversal in traversals
            )
        ),
        equations=equations,
        closure_edge_count=len(
            {
                normalized_edge(event.edge)
                for event in closure_events
                if event.ring_closure is not None
                and event.ring_closure.role == "open"
                and event.edge is not None
            }
        ),
        closure_edge_set_count=len(
            {
                frozenset(edge.edge for edge in plan.closure_edges)
                for plan in plans
            }
        ),
        closure_edges_per_traversal=max(
            (len(plan.closure_edges) for plan in plans),
            default=0,
        ),
        closure_label_count=len(
            {
                event.ring_closure.label
                for event in closure_events
                if event.ring_closure is not None
            }
        ),
        marker_slot_count=sum(
            1
            for traversal in traversals
            for event in traversal.events
            if event.marker_slot is not None
        ),
        marker_assignment_count=state.complexity_snapshot().estimated_product_size,
    )


def shared_tetrahedral_atom_stereo_support_for_case(
    case: SouthStarExpandedSupportCase,
) -> SouthStarTetrahedralAtomStereoProof:
    """Check tetrahedral fixtures through shared traversal-token proof inputs.

    This remains fixture evidence, not an independent support generator. Outputs
    come from the graph-native traversal path; the fixture is only a cross-check
    while tetrahedral authority is still temporary-witness-backed.
    """
    mol = parse_smiles(case.source_smiles)
    facts = extract_tetrahedral_center_facts(mol)
    if not facts:
        raise NotImplementedError("tetrahedral traversal check requires centers")
    facts_by_atom = {fact.center_atom_idx: fact for fact in facts}
    traversals = mol_to_smiles_enum_s_tree_traversals_for_case(case)
    outputs = tuple(
        dict.fromkeys(
            render_south_star_tree_traversal(traversal)
            for traversal in traversals
        )
    )
    return SouthStarTetrahedralAtomStereoProof(
        outputs=outputs,
        diagnostics=tuple(
            _tetrahedral_token_diagnostic_for_atom_event(
                traversal.connected_graph_plan,
                center_atom_idx=event.atom_idx,
                emitted_token=_tetrahedral_token_from_atom_text(event.text),
                facts_by_atom=facts_by_atom,
            )
            for traversal in traversals
            for event in traversal.events
            if event.kind == "atom"
            and event.atom_idx in facts_by_atom
            and event.atom_idx is not None
        ),
        proof_inputs=tuple(
            _tetrahedral_proof_input_for_atom_event(
                traversal.connected_graph_plan,
                center_atom_idx=event.atom_idx,
                facts_by_atom=facts_by_atom,
            )
            for traversal in traversals
            for event in traversal.events
            if event.kind == "atom"
            and event.atom_idx in facts_by_atom
            and event.atom_idx is not None
        ),
        output_count=len(outputs),
        expected_output_count=len(case.expected_support),
        fixture_cross_check_passed=outputs == case.expected_support,
    )


def _tetrahedral_token_diagnostic_for_atom_event(
    connected_graph_plan: SouthStarConnectedGraphTraversalPlan | None,
    *,
    center_atom_idx: int,
    emitted_token: str,
    facts_by_atom: dict[int, SouthStarTetrahedralCenterFact],
) -> SouthStarTetrahedralTraversalTokenDiagnostic:
    fact = facts_by_atom[center_atom_idx]
    if connected_graph_plan is None:
        raise ValueError("tetrahedral diagnostic requires connected graph plan")
    observation = tetrahedral_traversal_observation_from_connected_graph_plan(
        connected_graph_plan,
        center_atom_idx=center_atom_idx,
        implicit_hydrogen_count=fact.implicit_hydrogen_count,
    )
    diagnostic = tetrahedral_traversal_token_diagnostic(
        fact,
        observation,
        emitted_token=emitted_token,
    )
    if not diagnostic.preserves_orientation:
        raise ValueError(
            f"tetrahedral traversal emitted {diagnostic.emitted_token!r}, "
            f"expected {diagnostic.expected_token!r} for atom {center_atom_idx}"
        )
    return diagnostic


def _tetrahedral_proof_input_for_atom_event(
    connected_graph_plan: SouthStarConnectedGraphTraversalPlan | None,
    *,
    center_atom_idx: int,
    facts_by_atom: dict[int, SouthStarTetrahedralCenterFact],
) -> SouthStarTetrahedralTraversalProofInput:
    fact = facts_by_atom[center_atom_idx]
    if connected_graph_plan is None:
        raise ValueError("tetrahedral proof input requires connected graph plan")
    observation = tetrahedral_traversal_observation_from_connected_graph_plan(
        connected_graph_plan,
        center_atom_idx=center_atom_idx,
        implicit_hydrogen_count=fact.implicit_hydrogen_count,
    )
    return tetrahedral_traversal_proof_input(fact, observation)


def _tetrahedral_token_from_atom_text(atom_text: str) -> str:
    if "@@" in atom_text:
        return "@@"
    if "@" in atom_text:
        return "@"
    raise ValueError(f"tetrahedral atom text has no token: {atom_text!r}")


def independent_directional_component_product_proof_for_case(
    case: SouthStarExpandedSupportCase,
) -> SouthStarIndependentDirectionalComponentProductProof:
    """Explain multi-component directional support through shared equations."""

    mol = parse_smiles(case.source_smiles)
    facts = SouthStarMoleculeFacts.from_mol(mol)
    if (
        not facts.graph_topology.connected
        or not facts.graph_topology.acyclic_connected_tree
    ):
        raise NotImplementedError(
            "independent directional-component proof requires one acyclic component"
        )
    state = SouthStarComponentSupportState.from_molecule_facts(facts)
    if len(state.components) <= 1:
        raise NotImplementedError(
            "independent directional-component proof requires multiple components"
        )

    component_marker_assignments = state.component_marker_assignments()
    component_local_assignment_counts = tuple(
        len(assignments) for assignments in component_marker_assignments
    )
    component_assignment_product_size = reduce(
        mul,
        component_local_assignment_counts,
        1,
    )
    traversals = mol_to_smiles_enum_s_tree_traversals_for_case(case)
    equation_groups = tuple(
        marker_slot_parity_equations_for_traversal(state, traversal)
        for traversal in traversals
    )
    solver_results = tuple(
        solve_marker_slot_parity_equations(equations)
        for equations in equation_groups
    )
    raw_outputs = tuple(
        render_south_star_tree_traversal(traversal) for traversal in traversals
    )
    outputs = tuple(dict.fromkeys(raw_outputs))
    if len(traversals) % component_assignment_product_size:
        raise AssertionError(
            "independent directional-component traversals must factor into "
            "component assignments times traversal skeletons"
        )

    return SouthStarIndependentDirectionalComponentProductProof(
        outputs=outputs,
        component_ids=tuple(component.component_id for component in state.components),
        component_local_assignment_counts=component_local_assignment_counts,
        component_assignment_product_size=component_assignment_product_size,
        traversal_skeletons_per_component_assignment=(
            len(traversals) // component_assignment_product_size
        ),
        traversal_count=len(traversals),
        marker_slot_count=sum(len(equations) for equations in equation_groups),
        equation_count=sum(len(equations) for equations in equation_groups),
        solver_assignment_count=sum(
            len(result.assignments) for result in solver_results
        ),
        raw_output_count=len(raw_outputs),
        output_count=len(outputs),
        disjoint_component_carriers=_components_have_disjoint_carriers(
            state.components
        ),
        all_components_uncoupled=all(
            not component.coupling_causes for component in state.components
        ),
        all_equations_component_local=all(
            len(equation.component_ids) == 1
            for equations in equation_groups
            for equation in equations
        ),
        all_traversals_have_component_product_assignment=all(
            len(traversal.component_marker_assignments) == len(state.components)
            for traversal in traversals
        ),
        all_solver_assignments_match_traversal=all(
            tuple(sorted(result.assignments[0].marker_by_slot))
            == tuple(
                sorted(
                    (assignment.slot_id, assignment.marker)
                    for assignment in traversal.marker_assignments
                )
            )
            for traversal, result in zip(traversals, solver_results, strict=True)
        ),
        expected_support_strings_used=False,
    )


def _components_have_disjoint_carriers(
    components: tuple[SouthStarSemanticStereoComponent, ...],
) -> bool:
    seen: set[Edge] = set()
    for component in components:
        for edge in component.eligible_carrier_edges:
            if edge in seen:
                return False
            seen.add(edge)
    return True


def directional_tetrahedral_composition_proof_for_case(
    case: SouthStarExpandedSupportCase,
) -> SouthStarDirectionalTetrahedralCompositionProof:
    """Explain mixed directional/tetrahedral support through one event spine."""

    mol = parse_smiles(case.source_smiles)
    facts = SouthStarMoleculeFacts.from_mol(mol)
    if (
        not facts.graph_topology.connected
        or not facts.graph_topology.acyclic_connected_tree
    ):
        raise NotImplementedError(
            "directional/tetrahedral proof requires one acyclic component"
        )
    state = SouthStarComponentSupportState.from_molecule_facts(facts)
    tetrahedral_facts = extract_tetrahedral_center_facts(mol)
    if not state.components:
        raise NotImplementedError("directional/tetrahedral proof requires markers")
    if not tetrahedral_facts:
        raise NotImplementedError(
            "directional/tetrahedral proof requires tetrahedral centers"
        )

    facts_by_atom = {fact.center_atom_idx: fact for fact in tetrahedral_facts}
    traversals = mol_to_smiles_enum_s_tree_traversals_for_case(case)
    equation_groups = tuple(
        marker_slot_parity_equations_for_traversal(state, traversal)
        for traversal in traversals
    )
    solver_results = tuple(
        solve_marker_slot_parity_equations(equations)
        for equations in equation_groups
    )
    tetrahedral_proof_inputs = tuple(
        _tetrahedral_proof_input_for_atom_event(
            traversal.connected_graph_plan,
            center_atom_idx=event.atom_idx,
            facts_by_atom=facts_by_atom,
        )
        for traversal in traversals
        for event in traversal.events
        if event.kind == "atom"
        and event.atom_idx in facts_by_atom
        and event.atom_idx is not None
    )
    tetrahedral_diagnostics = tuple(
        _tetrahedral_token_diagnostic_for_atom_event(
            traversal.connected_graph_plan,
            center_atom_idx=event.atom_idx,
            emitted_token=_tetrahedral_token_from_atom_text(event.text),
            facts_by_atom=facts_by_atom,
        )
        for traversal in traversals
        for event in traversal.events
        if event.kind == "atom"
        and event.atom_idx in facts_by_atom
        and event.atom_idx is not None
    )
    raw_outputs = tuple(
        render_south_star_tree_traversal(traversal) for traversal in traversals
    )
    outputs = tuple(dict.fromkeys(raw_outputs))
    component_assignment_product_size = reduce(
        mul,
        (len(assignments) for assignments in state.component_marker_assignments()),
        1,
    )

    return SouthStarDirectionalTetrahedralCompositionProof(
        outputs=outputs,
        directional_component_ids=tuple(
            component.component_id for component in state.components
        ),
        tetrahedral_center_atom_indices=tuple(
            fact.center_atom_idx for fact in tetrahedral_facts
        ),
        component_assignment_product_size=component_assignment_product_size,
        traversal_count=len(traversals),
        marker_slot_count=sum(len(equations) for equations in equation_groups),
        equation_count=sum(len(equations) for equations in equation_groups),
        solver_assignment_count=sum(
            len(result.assignments) for result in solver_results
        ),
        renderer_input_count=sum(
            1
            for traversal in traversals
            for event in traversal.events
            if event.renderer_input is not None
        ),
        tetrahedral_proof_input_count=len(tetrahedral_proof_inputs),
        tetrahedral_diagnostic_count=len(tetrahedral_diagnostics),
        raw_output_count=len(raw_outputs),
        output_count=len(outputs),
        all_traversals_have_directional_obligations=all(
            any(event.marker_slot is not None for event in traversal.events)
            for traversal in traversals
        ),
        all_traversals_have_tetrahedral_obligations=all(
            any(
                event.renderer_input is not None
                and event.atom_idx in facts_by_atom
                for event in traversal.events
            )
            for traversal in traversals
        ),
        all_solver_assignments_match_traversal=all(
            tuple(sorted(result.assignments[0].marker_by_slot))
            == tuple(
                sorted(
                    (assignment.slot_id, assignment.marker)
                    for assignment in traversal.marker_assignments
                )
            )
            for traversal, result in zip(traversals, solver_results, strict=True)
        ),
        all_tetrahedral_tokens_preserve_orientation=all(
            diagnostic.preserves_orientation for diagnostic in tetrahedral_diagnostics
        ),
        all_tetrahedral_renderer_inputs_match_proof=all(
            proof_input.expected_token == proof_input.renderer_input.value
            for proof_input in tetrahedral_proof_inputs
        ),
        expected_support_strings_used=False,
    )


def _assert_ring_stereo_monocycle_domain(mol: Chem.Mol) -> None:
    if len(Chem.GetMolFrags(mol)) != 1:
        raise NotImplementedError("ring-stereo oracle requires one component")
    if len(mol.GetRingInfo().BondRings()) != 1:
        raise NotImplementedError("ring-stereo oracle requires one ring")


def _assert_saturated_monocycle_domain(mol: Chem.Mol) -> None:
    _assert_nonstereo_monocycle_domain(mol)
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            raise NotImplementedError(
                "saturated monocycle check supports only single bonds"
            )


def _assert_nonstereo_monocycle_domain(mol: Chem.Mol) -> None:
    if len(Chem.GetMolFrags(mol)) != 1:
        raise NotImplementedError("nonstereo monocycle check requires one component")
    if len(mol.GetRingInfo().BondRings()) != 1:
        raise NotImplementedError("nonstereo monocycle check requires one ring")
    for atom in mol.GetAtoms():
        _atom_text(atom)
    for bond in mol.GetBonds():
        if bond.GetBondType() not in {Chem.BondType.SINGLE, Chem.BondType.DOUBLE}:
            raise NotImplementedError(
                "nonstereo monocycle check supports only single and double bonds"
            )
        if bond.GetStereo() != Chem.BondStereo.STEREONONE:
            raise NotImplementedError(
                "nonstereo monocycle check does not support bond stereo"
            )
        if bond.GetBondDir() != Chem.BondDir.NONE:
            raise NotImplementedError(
                "nonstereo monocycle check does not support directional bonds"
            )


def _atom_text(atom: Chem.Atom) -> str:
    if atom.GetIsAromatic():
        raise NotImplementedError(
            "nonstereo-monocycle witness does not support aromatic atoms"
        )
    if atom.GetFormalCharge() != 0:
        raise NotImplementedError(
            "nonstereo-monocycle witness does not support charged atoms"
        )
    if atom.GetIsotope() != 0:
        raise NotImplementedError(
            "nonstereo-monocycle witness does not support isotopic atoms"
        )
    if atom.GetNumRadicalElectrons() != 0:
        raise NotImplementedError(
            "nonstereo-monocycle witness does not support radical atoms"
        )
    symbol = atom.GetSymbol()
    if symbol in {"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"}:
        return symbol
    raise NotImplementedError(f"unsupported nonstereo-monocycle atom {symbol!r}")
