"""Derived graph indexes for South Star 1 molecule facts.

Indexes are caches over immutable facts. They must not perform chemistry
perception or call parser/writer libraries.
"""

from __future__ import annotations

from dataclasses import dataclass

from .facts import AtomFacts
from .facts import atom_pair_key
from .facts import BondFacts
from .facts import MoleculeFacts
from .ids import AtomId
from .ids import BondId


@dataclass(frozen=True, slots=True)
class GraphIndex:
    atom_by_id: dict[AtomId, AtomFacts]
    bond_by_id: dict[BondId, BondFacts]
    incident_bonds: dict[AtomId, tuple[BondId, ...]]
    neighbors: dict[AtomId, tuple[AtomId, ...]]
    bond_between: dict[tuple[AtomId, AtomId], BondId]


def build_graph_index(facts: MoleculeFacts) -> GraphIndex:
    facts.validate()

    atom_by_id = {atom.id: atom for atom in facts.atoms}
    bond_by_id = {bond.id: bond for bond in facts.bonds}
    incident: dict[AtomId, list[BondId]] = {atom.id: [] for atom in facts.atoms}
    neighbor_lists: dict[AtomId, list[AtomId]] = {atom.id: [] for atom in facts.atoms}
    bond_between: dict[tuple[AtomId, AtomId], BondId] = {}

    for bond in facts.bonds:
        incident[bond.a].append(bond.id)
        incident[bond.b].append(bond.id)
        neighbor_lists[bond.a].append(bond.b)
        neighbor_lists[bond.b].append(bond.a)
        bond_between[atom_pair_key(bond.a, bond.b)] = bond.id

    return GraphIndex(
        atom_by_id=atom_by_id,
        bond_by_id=bond_by_id,
        incident_bonds={
            atom_id: tuple(bond_ids)
            for atom_id, bond_ids in incident.items()
        },
        neighbors={
            atom_id: tuple(neighbors)
            for atom_id, neighbors in neighbor_lists.items()
        },
        bond_between=bond_between,
    )

__all__ = ("GraphIndex", "build_graph_index")
