from __future__ import annotations

from dataclasses import dataclass

from grimace._south_star.atom_text import atom_text_obligation_for_supported_fields
from grimace._south_star.enum_s import mol_to_smiles_enum_s_tree_traversals_for_case
from grimace._south_star.enum_s import render_south_star_tree_traversal
from grimace._south_star.molecule_facts import SouthStarMoleculeFacts
from tests.helpers.south_star_semantic_oracle import parse_smiles


@dataclass(frozen=True, slots=True)
class SouthStarSingleAtomAtomTextSupport:
    emitted_text: str
    support: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SouthStarMarkerlessAcyclicTreeSupportProof:
    case_id: str
    atom_count: int
    bond_count: int
    traversal_count: int
    bond_event_count: int
    branch_event_count: int
    raw_output_count: int
    output_count: int
    support: tuple[str, ...]
    expected_support_strings_used: bool


def is_single_atom_atom_text_domain(facts: SouthStarMoleculeFacts) -> bool:
    topology = facts.graph_topology
    return (
        facts.supported
        and topology.connected
        and topology.atom_count == 1
        and topology.bond_count == 0
        and not topology.ring_system.has_rings
        and len(facts.atom_text_facts) == 1
        and not facts.components
        and not facts.carrier_opportunities
        and not facts.tetrahedral_center_facts
    )


def single_atom_atom_text_support_from_facts(
    facts: SouthStarMoleculeFacts,
) -> SouthStarSingleAtomAtomTextSupport:
    if not is_single_atom_atom_text_domain(facts):
        raise NotImplementedError(
            "single-atom atom-text unified reference requires one supported "
            "atom, zero bonds, one fragment, and no stereo constraints"
        )
    [fields] = facts.atom_text_facts
    emitted_text = atom_text_obligation_for_supported_fields(fields).emitted_text
    return SouthStarSingleAtomAtomTextSupport(
        emitted_text=emitted_text,
        support=(emitted_text,),
    )


def is_two_atom_markerless_atom_text_domain(facts: SouthStarMoleculeFacts) -> bool:
    topology = facts.graph_topology
    return (
        facts.supported
        and topology.connected
        and topology.atom_count == 2
        and topology.bond_count == 1
        and not topology.ring_system.has_rings
        and len(facts.atom_text_facts) == 2
        and len(facts.bond_text_facts) == 1
        and not facts.components
        and not facts.carrier_opportunities
        and not facts.tetrahedral_center_facts
    )


def two_atom_markerless_atom_text_support_from_facts(
    facts: SouthStarMoleculeFacts,
) -> tuple[str, ...]:
    if not is_two_atom_markerless_atom_text_domain(facts):
        raise NotImplementedError(
            "two-atom markerless atom-text unified reference requires two "
            "supported atoms, one bond, one fragment, and no stereo constraints"
        )
    [bond_fact] = facts.bond_text_facts
    bond_text = _bond_text_from_fact(bond_fact.bond_type)
    atom_text_by_idx = {
        fields.atom_idx: atom_text_obligation_for_supported_fields(fields).emitted_text
        for fields in facts.atom_text_facts
    }
    begin_idx, end_idx = bond_fact.edge
    outputs = (
        f"{atom_text_by_idx[begin_idx]}{bond_text}{atom_text_by_idx[end_idx]}",
        f"{atom_text_by_idx[end_idx]}{bond_text}{atom_text_by_idx[begin_idx]}",
    )
    return tuple(dict.fromkeys(outputs))


def is_markerless_acyclic_tree_domain(facts: SouthStarMoleculeFacts) -> bool:
    topology = facts.graph_topology
    return (
        facts.supported
        and topology.connected
        and topology.acyclic_connected_tree
        and topology.atom_count >= 3
        and len(facts.atom_text_facts) == topology.atom_count
        and len(facts.bond_text_facts) == topology.bond_count
        and not topology.ring_system.has_rings
        and not facts.components
        and not facts.carrier_opportunities
        and not facts.tetrahedral_center_facts
    )


def markerless_acyclic_tree_support_from_shared_spine(
    case: object,
) -> SouthStarMarkerlessAcyclicTreeSupportProof:
    mol = parse_smiles(case.source_smiles)
    facts = SouthStarMoleculeFacts.from_mol(mol)
    if not is_markerless_acyclic_tree_domain(facts):
        raise NotImplementedError(
            "markerless acyclic-tree unified reference requires one connected "
            "acyclic molecule, at least three atoms, supported atom/bond text, "
            "and no stereo constraints"
        )
    traversals = mol_to_smiles_enum_s_tree_traversals_for_case(case)
    if any(
        event.marker_slot is not None or event.renderer_input is not None
        for traversal in traversals
        for event in traversal.events
    ):
        raise AssertionError(
            "markerless acyclic-tree proof must not use stereo marker slots or "
            "renderer-input obligations"
        )
    raw_outputs = tuple(
        render_south_star_tree_traversal(traversal) for traversal in traversals
    )
    support = tuple(dict.fromkeys(raw_outputs))
    return SouthStarMarkerlessAcyclicTreeSupportProof(
        case_id=case.case_id,
        atom_count=facts.graph_topology.atom_count,
        bond_count=facts.graph_topology.bond_count,
        traversal_count=len(traversals),
        bond_event_count=sum(
            1
            for traversal in traversals
            for event in traversal.events
            if event.kind == "bond"
        ),
        branch_event_count=sum(
            1
            for traversal in traversals
            for event in traversal.events
            if event.kind in {"branch_open", "branch_close"}
        ),
        raw_output_count=len(raw_outputs),
        output_count=len(support),
        support=support,
        expected_support_strings_used=False,
    )


def _bond_text_from_fact(bond_type: str) -> str:
    if bond_type == "SINGLE":
        return ""
    if bond_type == "DOUBLE":
        return "="
    if bond_type == "TRIPLE":
        return "#"
    raise NotImplementedError(f"unsupported markerless bond type {bond_type!r}")
