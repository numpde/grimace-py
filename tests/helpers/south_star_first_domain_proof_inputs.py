from __future__ import annotations

from dataclasses import dataclass

from grimace._south_star.enum_s import mol_to_smiles_enum_s_tree_traversals_for_case
from grimace._south_star.molecule_facts import SouthStarMoleculeFacts
from grimace._south_star.policies import DEFAULT_SOUTH_STAR_POLICY_SET
from tests.helpers.south_star_marker_equations import (
    marker_slot_parity_equations_for_case,
)
from tests.helpers.south_star_semantic_oracle import parse_smiles
from tests.helpers.south_star_semantics import SouthStarSemanticCase


@dataclass(frozen=True, slots=True)
class SouthStarFirstDomainProofInputs:
    case_id: str
    source_smiles: str
    atom_count: int
    bond_count: int
    component_count: int
    carrier_opportunity_count: int
    traversal_count: int
    traversal_event_count: int
    marker_slot_count: int
    carrier_context_count: int
    renderer_input_count: int
    equation_count: int
    annotation_policy_name: str
    expected_support_strings_used: bool


def first_domain_proof_inputs_from_shared_spine(
    case: SouthStarSemanticCase,
) -> SouthStarFirstDomainProofInputs:
    """Extract first-domain proof inputs without reading expected SMILES strings."""

    mol = parse_smiles(case.source_smiles)
    molecule_facts = SouthStarMoleculeFacts.from_mol(mol)
    traversals = mol_to_smiles_enum_s_tree_traversals_for_case(case)
    marker_slots = tuple(
        event.marker_slot
        for traversal in traversals
        for event in traversal.events
        if event.marker_slot is not None
    )
    return SouthStarFirstDomainProofInputs(
        case_id=case.case_id,
        source_smiles=case.source_smiles,
        atom_count=molecule_facts.graph_topology.atom_count,
        bond_count=molecule_facts.graph_topology.bond_count,
        component_count=len(molecule_facts.components),
        carrier_opportunity_count=len(molecule_facts.carrier_opportunities),
        traversal_count=len(traversals),
        traversal_event_count=sum(len(traversal.events) for traversal in traversals),
        marker_slot_count=len(marker_slots),
        carrier_context_count=sum(
            len(slot.adjacent_contexts) for slot in marker_slots
        ),
        renderer_input_count=sum(
            1
            for traversal in traversals
            for event in traversal.events
            if event.renderer_input is not None
        ),
        equation_count=sum(
            len(equations)
            for equations in marker_slot_parity_equations_for_case(case)
        ),
        annotation_policy_name=DEFAULT_SOUTH_STAR_POLICY_SET.annotation_policy.name,
        expected_support_strings_used=False,
    )
