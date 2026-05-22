from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

from rdkit import Chem

from grimace._south_star.component_support_state import SouthStarComponentSupportState
from grimace._south_star.components import _componentize_features
from grimace._south_star.components import _source_stereo_features
from grimace._south_star.enum_s import _combined_marker_assignments
from grimace._south_star.enum_s import _ring_system_traversals
from grimace._south_star.enum_s import _supported_polycyclic_closure_edge_sets
from grimace._south_star.enum_s import render_south_star_tree_traversal
from grimace._south_star.molecule_facts import SouthStarMoleculeFacts
from grimace._south_star.molecule_facts import _carrier_opportunities
from grimace._south_star.policies import DEFAULT_SOUTH_STAR_POLICY_SET
from grimace._south_star.tetrahedral import (
    extract_ring_tetrahedral_interaction_obligations,
)
from tests.helpers.south_star_ring_tetrahedral_proof_spine import (
    _tetrahedral_facts_from_ring_obligations,
)
from tests.helpers.south_star_semantic_oracle import graph_signature
from tests.helpers.south_star_semantic_oracle import parse_smiles
from tests.helpers.south_star_semantic_oracle import semantic_signature


@dataclass(frozen=True, slots=True)
class SouthStarMixedPolycyclicDirectionalProof:
    source_smiles: str
    unsupported_categories: tuple[str, ...]
    ring_tetrahedral_center_atom_indices: tuple[int, ...]
    directional_component_ids: tuple[str, ...]
    directional_feature_ids: tuple[str, ...]
    ring_tetrahedral_obligation_count: int
    directional_component_count: int
    directional_coupling_cause_count: int
    component_assignment_count: int
    traversal_count: int
    raw_output_count: int
    output_count: int
    closure_event_count: int
    marker_slot_count: int
    renderer_input_count: int
    semantic_parseback_passed: bool
    outputs: tuple[str, ...]


def mixed_polycyclic_directional_proof(
    source_smiles: str,
) -> SouthStarMixedPolycyclicDirectionalProof:
    mol = parse_smiles(source_smiles)
    molecule_facts = SouthStarMoleculeFacts.from_mol(mol)
    directional_components = _componentize_features(_source_stereo_features(mol))
    ring_tetrahedral_obligations = extract_ring_tetrahedral_interaction_obligations(
        mol
    )
    _assert_domain(
        mol,
        molecule_facts=molecule_facts,
        directional_component_count=len(directional_components),
        ring_tetrahedral_obligation_count=len(ring_tetrahedral_obligations),
    )
    proof_facts = replace(
        molecule_facts,
        components=directional_components,
        carrier_opportunities=_carrier_opportunities(directional_components),
        tetrahedral_center_facts=tuple(
            _tetrahedral_facts_from_ring_obligations(mol).values()
        ),
    )
    state = SouthStarComponentSupportState(
        molecule_facts=proof_facts,
        annotation_policy=DEFAULT_SOUTH_STAR_POLICY_SET.annotation_policy,
    )
    combined_assignments = _combined_marker_assignments(state)
    traversals = tuple(
        traversal
        for combined_assignment in combined_assignments
        for traversal in _ring_system_traversals(
            mol,
            molecule_facts=proof_facts,
            state=state,
            closure_edge_sets=_supported_polycyclic_closure_edge_sets(mol),
            marker_by_edge=combined_assignment.marker_by_edge,
            component_marker_assignments=(
                combined_assignment.component_marker_assignments
            ),
            tetrahedral_facts_by_atom_override=(
                _tetrahedral_facts_from_ring_obligations(mol)
            ),
        )
    )
    raw_outputs = tuple(render_south_star_tree_traversal(t) for t in traversals)
    outputs = tuple(dict.fromkeys(raw_outputs))
    semantic_parseback_passed = _outputs_preserve_semantics(source_smiles, outputs)
    return SouthStarMixedPolycyclicDirectionalProof(
        source_smiles=source_smiles,
        unsupported_categories=tuple(sorted(molecule_facts.unsupported_categories)),
        ring_tetrahedral_center_atom_indices=tuple(
            obligation.center_atom_idx for obligation in ring_tetrahedral_obligations
        ),
        directional_component_ids=tuple(
            component.component_id for component in directional_components
        ),
        directional_feature_ids=tuple(
            feature.feature_id
            for component in directional_components
            for feature in component.source_features
        ),
        ring_tetrahedral_obligation_count=len(ring_tetrahedral_obligations),
        directional_component_count=len(directional_components),
        directional_coupling_cause_count=sum(
            len(component.coupling_causes) for component in directional_components
        ),
        component_assignment_count=len(combined_assignments),
        traversal_count=len(traversals),
        raw_output_count=len(raw_outputs),
        output_count=len(outputs),
        closure_event_count=sum(
            1
            for traversal in traversals
            for event in traversal.events
            if event.ring_closure is not None
        ),
        marker_slot_count=sum(
            1
            for traversal in traversals
            for event in traversal.events
            if event.marker_slot is not None
        ),
        renderer_input_count=sum(
            1
            for traversal in traversals
            for event in traversal.events
            if event.renderer_input is not None
        ),
        semantic_parseback_passed=semantic_parseback_passed,
        outputs=outputs,
    )


def _assert_domain(
    mol: Chem.Mol,
    *,
    molecule_facts: SouthStarMoleculeFacts,
    directional_component_count: int,
    ring_tetrahedral_obligation_count: int,
) -> None:
    expected_categories = {
        "fused_or_polycyclic_ring",
        "ring_molecule",
        "ring_tetrahedral_interaction",
    }
    unexpected = set(molecule_facts.unsupported_categories) - expected_categories
    if unexpected:
        raise NotImplementedError(
            "mixed polycyclic directional proof only accepts the named "
            f"frontier categories; found {sorted(unexpected)!r}"
        )
    if not molecule_facts.graph_topology.connected:
        raise NotImplementedError(
            "mixed polycyclic directional proof currently requires one fragment"
        )
    if not molecule_facts.graph_topology.ring_system.fused_or_polycyclic:
        raise NotImplementedError(
            "mixed polycyclic directional proof requires a fused or polycyclic ring"
        )
    if ring_tetrahedral_obligation_count != 1:
        raise NotImplementedError(
            "mixed polycyclic directional proof currently requires exactly one "
            "ring/tetrahedral obligation"
        )
    if directional_component_count != 1:
        raise NotImplementedError(
            "mixed polycyclic directional proof currently requires exactly one "
            "directional component"
        )
    if not _tetrahedral_facts_from_ring_obligations(mol):
        raise NotImplementedError(
            "mixed polycyclic directional proof requires ring/tetrahedral facts"
        )


def _outputs_preserve_semantics(
    source_smiles: str,
    outputs: tuple[str, ...],
) -> bool:
    source_graph = graph_signature(source_smiles)
    source_semantics = semantic_signature(source_smiles)
    return all(
        graph_signature(output) == source_graph
        and semantic_signature(output) == source_semantics
        for output in outputs
    )
