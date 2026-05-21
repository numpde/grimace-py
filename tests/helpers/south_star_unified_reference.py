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
