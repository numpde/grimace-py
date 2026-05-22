"""Traversal-skeleton records and generators for the private proof kernel."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from itertools import permutations
from itertools import product
from types import MappingProxyType

from .facts import MoleculeFacts
from .graph_index import GraphIndex
from .ids import AtomId
from .ids import BondId
from .policy import SmilesPolicy


class ChildRole(Enum):
    BRANCH = "branch"
    CONTINUATION = "continuation"


@dataclass(frozen=True, slots=True)
class ChildEvent:
    bond: BondId
    parent: AtomId
    child: AtomId
    role: ChildRole


@dataclass(frozen=True, slots=True)
class TraversalSkeleton:
    roots: tuple[AtomId, ...]
    parent: Mapping[AtomId, AtomId | None]
    tree_bonds: frozenset[BondId]
    ring_bonds: frozenset[BondId]
    events_at: Mapping[AtomId, tuple[ChildEvent, ...]]


def enumerate_tree_skeletons(
    facts: MoleculeFacts,
    index: GraphIndex,
    policy: SmilesPolicy,
) -> tuple[TraversalSkeleton, ...]:
    """Enumerate restrictive acyclic traversal skeletons.

    This first proof-kernel slice deliberately supports only graph components
    that are already trees. Ring endpoints and ring-label choices enter in the
    next slice.
    """

    facts.validate()
    policy.validate_for_facts(facts)
    _require_tree_components(facts, index)

    skeletons: list[TraversalSkeleton] = []
    component_root_domains = tuple(component.atoms for component in facts.components)

    for roots in product(*component_root_domains):
        parent: dict[AtomId, AtomId | None] = {}
        child_bond_by_parent: dict[AtomId, list[tuple[BondId, AtomId]]] = {
            atom.id: [] for atom in facts.atoms
        }

        for root in roots:
            _orient_tree_component(index, root, parent, child_bond_by_parent)

        local_order_domains = tuple(
            (atom.id, tuple(_local_child_orders(atom.id, child_bond_by_parent[atom.id])))
            for atom in facts.atoms
        )
        for events_at_items in product(*(domain for _, domain in local_order_domains)):
            events_at = {
                atom: events
                for (atom, _), events in zip(local_order_domains, events_at_items)
            }
            skeletons.append(
                TraversalSkeleton(
                    roots=tuple(roots),
                    parent=MappingProxyType(dict(parent)),
                    tree_bonds=frozenset(bond.id for bond in facts.bonds),
                    ring_bonds=frozenset(),
                    events_at=MappingProxyType(events_at),
                )
            )

    return tuple(skeletons)


def _require_tree_components(facts: MoleculeFacts, index: GraphIndex) -> None:
    for component in facts.components:
        if len(component.bonds) != len(component.atoms) - 1:
            raise NotImplementedError(
                "South Star 1 tree skeletons currently require acyclic components"
            )
        component_atoms = set(component.atoms)
        seen = _reachable_atoms(index, component.atoms[0], component_atoms)
        if seen != component_atoms:
            raise NotImplementedError(
                "South Star 1 tree skeletons currently require connected components"
            )


def _reachable_atoms(
    index: GraphIndex,
    start: AtomId,
    allowed_atoms: set[AtomId],
) -> set[AtomId]:
    seen: set[AtomId] = set()
    stack = [start]
    while stack:
        atom = stack.pop()
        if atom in seen:
            continue
        seen.add(atom)
        for bond_id in index.incident_bonds[atom]:
            bond = index.bond_by_id[bond_id]
            neighbor = bond.b if bond.a == atom else bond.a
            if neighbor in allowed_atoms and neighbor not in seen:
                stack.append(neighbor)
    return seen


def _orient_tree_component(
    index: GraphIndex,
    root: AtomId,
    parent: dict[AtomId, AtomId | None],
    child_bond_by_parent: dict[AtomId, list[tuple[BondId, AtomId]]],
) -> None:
    parent[root] = None
    stack = [root]

    while stack:
        atom = stack.pop()
        for bond_id in reversed(index.incident_bonds[atom]):
            bond = index.bond_by_id[bond_id]
            neighbor = bond.b if bond.a == atom else bond.a
            if parent.get(atom) == neighbor:
                continue
            if neighbor in parent:
                raise NotImplementedError(
                    "South Star 1 tree skeletons currently reject cycles"
                )
            parent[neighbor] = atom
            child_bond_by_parent[atom].append((bond_id, neighbor))
            stack.append(neighbor)


def _local_child_orders(
    parent: AtomId,
    children: list[tuple[BondId, AtomId]],
) -> tuple[tuple[ChildEvent, ...], ...]:
    if not children:
        return ((),)

    orders: list[tuple[ChildEvent, ...]] = []
    for ordered_children in permutations(children):
        orders.append(
            tuple(
                ChildEvent(
                    bond=bond_id,
                    parent=parent,
                    child=child,
                    role=ChildRole.BRANCH,
                )
                for bond_id, child in ordered_children
            )
        )
        orders.append(
            tuple(
                ChildEvent(
                    bond=bond_id,
                    parent=parent,
                    child=child,
                    role=(
                        ChildRole.CONTINUATION
                        if i == len(ordered_children) - 1
                        else ChildRole.BRANCH
                    ),
                )
                for i, (bond_id, child) in enumerate(ordered_children)
            )
        )
    return tuple(orders)


__all__ = (
    "ChildEvent",
    "ChildRole",
    "TraversalSkeleton",
    "enumerate_tree_skeletons",
)
