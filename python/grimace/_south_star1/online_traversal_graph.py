"""Cached graph view used by online traversal and search."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .facts import MoleculeFacts
from .graph_index import GraphIndex
from .ids import AtomId
from .ids import BondId


@dataclass(frozen=True, slots=True)
class OnlineTraversalGraph:
    atoms: tuple[AtomId, ...]
    bonds: Mapping[BondId, tuple[AtomId, AtomId]]
    components: tuple[tuple[tuple[AtomId, ...], tuple[BondId, ...]], ...]
    incident_bonds: Mapping[AtomId, tuple[BondId, ...]]


def build_online_traversal_graph_from_index(
    facts: MoleculeFacts,
    index: GraphIndex,
) -> OnlineTraversalGraph:
    return OnlineTraversalGraph(
        atoms=tuple(atom.id for atom in facts.atoms),
        bonds=MappingProxyType(
            {
                bond_id: (bond.a, bond.b)
                for bond_id, bond in index.bond_by_id.items()
            }
        ),
        components=tuple(
            (component.atoms, component.bonds)
            for component in facts.components
        ),
        incident_bonds=MappingProxyType(dict(index.incident_bonds)),
    )


def build_online_traversal_graph_from_facts(
    facts: MoleculeFacts,
) -> OnlineTraversalGraph:
    incident: dict[AtomId, list[BondId]] = {atom.id: [] for atom in facts.atoms}
    bonds: dict[BondId, tuple[AtomId, AtomId]] = {}
    for bond in facts.bonds:
        bonds[bond.id] = (bond.a, bond.b)
        incident[bond.a].append(bond.id)
        incident[bond.b].append(bond.id)
    return OnlineTraversalGraph(
        atoms=tuple(atom.id for atom in facts.atoms),
        bonds=MappingProxyType(bonds),
        components=tuple(
            (component.atoms, component.bonds)
            for component in facts.components
        ),
        incident_bonds=MappingProxyType(
            {
                atom: tuple(items)
                for atom, items in incident.items()
            }
        ),
    )


__all__ = (
    "OnlineTraversalGraph",
    "build_online_traversal_graph_from_facts",
    "build_online_traversal_graph_from_index",
)
