from __future__ import annotations

from dataclasses import dataclass

from grimace._south_star.atom_text import atom_text_obligation_for_supported_fields
from grimace._south_star.molecule_facts import SouthStarMoleculeFacts


@dataclass(frozen=True, slots=True)
class SouthStarSingleAtomAtomTextSupport:
    emitted_text: str
    support: tuple[str, ...]


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


def _bond_text_from_fact(bond_type: str) -> str:
    if bond_type == "SINGLE":
        return ""
    if bond_type == "DOUBLE":
        return "="
    if bond_type == "TRIPLE":
        return "#"
    raise NotImplementedError(f"unsupported markerless bond type {bond_type!r}")
