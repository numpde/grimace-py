from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

from grimace._south_star.component_support_state import SouthStarComponentSupportState
from grimace._south_star.enum_s import _combined_marker_assignments
from grimace._south_star.enum_s import _ring_system_traversals
from grimace._south_star.enum_s import _supported_polycyclic_closure_edge_sets
from grimace._south_star.enum_s import render_south_star_tree_traversal
from grimace._south_star.molecule_facts import SouthStarMoleculeFacts
from grimace._south_star.policies import DEFAULT_SOUTH_STAR_POLICY_SET
from grimace._south_star.tetrahedral import (
    SouthStarTetrahedralCenterFact,
    extract_ring_tetrahedral_interaction_obligations,
)
from tests.helpers.south_star_semantic_oracle import graph_signature
from tests.helpers.south_star_semantic_oracle import parse_smiles
from tests.helpers.south_star_semantic_oracle import semantic_signature


POLYCYCLIC_RING_TETRAHEDRAL_PROOF_AUTHORITY = (
    "unified_reference_polycyclic_ring_tetrahedral_obligations"
)
_EXPECTED_PROOF_GATE_CATEGORIES = frozenset(
    {
        "fused_or_polycyclic_ring",
        "ring_molecule",
        "ring_tetrahedral_interaction",
    }
)


@dataclass(frozen=True, slots=True)
class SouthStarPolycyclicRingTetrahedralProofSpine:
    source_smiles: str
    support_authority: str
    traversal_count: int
    output_count: int
    closure_event_count: int
    renderer_input_count: int
    obligation_count: int
    outputs: tuple[str, ...]


def polycyclic_ring_tetrahedral_proof_spine(
    source_smiles: str,
) -> SouthStarPolycyclicRingTetrahedralProofSpine:
    mol = parse_smiles(source_smiles)
    molecule_facts = SouthStarMoleculeFacts.from_mol(mol)
    _assert_polycyclic_ring_tetrahedral_proof_domain(molecule_facts)
    state = SouthStarComponentSupportState(
        molecule_facts=molecule_facts,
        annotation_policy=DEFAULT_SOUTH_STAR_POLICY_SET.annotation_policy,
    )
    traversals = tuple(
        traversal
        for combined_assignment in _combined_marker_assignments(state)
        for traversal in _ring_system_traversals(
            mol,
            molecule_facts=molecule_facts,
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
    outputs = tuple(
        dict.fromkeys(
            render_south_star_tree_traversal(traversal) for traversal in traversals
        )
    )
    _assert_outputs_preserve_semantics(source_smiles, outputs)
    obligations = extract_ring_tetrahedral_interaction_obligations(mol)
    return SouthStarPolycyclicRingTetrahedralProofSpine(
        source_smiles=source_smiles,
        support_authority=POLYCYCLIC_RING_TETRAHEDRAL_PROOF_AUTHORITY,
        traversal_count=len(traversals),
        output_count=len(outputs),
        closure_event_count=sum(
            1
            for traversal in traversals
            for event in traversal.events
            if event.ring_closure is not None
        ),
        renderer_input_count=sum(
            1
            for traversal in traversals
            for event in traversal.events
            if event.renderer_input is not None
        ),
        obligation_count=len(obligations),
        outputs=outputs,
    )


def _assert_polycyclic_ring_tetrahedral_proof_domain(
    molecule_facts: SouthStarMoleculeFacts,
) -> None:
    unexpected = (
        set(molecule_facts.unsupported_categories)
        - _EXPECTED_PROOF_GATE_CATEGORIES
    )
    if unexpected:
        raise NotImplementedError(
            "polycyclic ring/tetrahedral proof spine only accepts the named "
            f"frontier categories; found {sorted(unexpected)!r}"
        )
    if not _EXPECTED_PROOF_GATE_CATEGORIES <= set(
        molecule_facts.unsupported_categories
    ):
        raise NotImplementedError(
            "polycyclic ring/tetrahedral proof spine requires the current "
            "fused/polycyclic ring-tetrahedral frontier"
        )
    if not molecule_facts.graph_topology.ring_system.fused_or_polycyclic:
        raise NotImplementedError(
            "polycyclic ring/tetrahedral proof spine requires a fused or "
            "polycyclic ring system"
        )


def _tetrahedral_facts_from_ring_obligations(
    mol: Chem.Mol,
) -> dict[int, SouthStarTetrahedralCenterFact]:
    return {
        obligation.center_atom_idx: SouthStarTetrahedralCenterFact(
            center_atom_idx=obligation.center_atom_idx,
            chiral_tag=str(
                mol.GetAtomWithIdx(obligation.center_atom_idx).GetChiralTag()
            ),
            source_token=obligation.source_token,
            explicit_neighbor_atom_indices=tuple(
                neighbor.GetIdx()
                for neighbor in mol.GetAtomWithIdx(
                    obligation.center_atom_idx
                ).GetNeighbors()
            ),
            implicit_hydrogen_count=obligation.implicit_hydrogen_count,
            source_ligand_order=obligation.source_ligand_order,
        )
        for obligation in extract_ring_tetrahedral_interaction_obligations(mol)
    }


def _assert_outputs_preserve_semantics(
    source_smiles: str,
    outputs: tuple[str, ...],
) -> None:
    source_graph_signature = graph_signature(source_smiles)
    source_semantic_signature = semantic_signature(source_smiles)
    for output in outputs:
        parse_smiles(output)
        if graph_signature(output) != source_graph_signature:
            raise AssertionError(f"output {output!r} changed source graph")
        if semantic_signature(output) != source_semantic_signature:
            raise AssertionError(f"output {output!r} changed source semantics")
