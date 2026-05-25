"""Traversal-skeleton records and generators for the private proof kernel."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
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
class RingEvent:
    bond: BondId
    atom: AtomId
    other_atom: AtomId


LocalEvent = ChildEvent | RingEvent


@dataclass(frozen=True, slots=True)
class TraversalSkeleton:
    roots: tuple[AtomId, ...]
    parent: Mapping[AtomId, AtomId | None]
    tree_bonds: frozenset[BondId]
    ring_bonds: frozenset[BondId]
    events_at: Mapping[AtomId, tuple[LocalEvent, ...]]


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


def enumerate_traversal_skeletons(
    facts: MoleculeFacts,
    index: GraphIndex,
    policy: SmilesPolicy,
    rooted_at_atom: AtomId | None = None,
) -> tuple[TraversalSkeleton, ...]:
    """Enumerate traversal skeletons with explicit tree/ring partitions."""

    facts.validate()
    policy.validate_for_facts(facts)

    component_root_domains = _component_root_domains(facts, rooted_at_atom)
    component_tree_domains = tuple(
        _component_spanning_trees(index, component.atoms, component.bonds)
        for component in facts.components
    )

    skeletons: list[TraversalSkeleton] = []
    for roots in product(*component_root_domains):
        for tree_bond_sets in product(*component_tree_domains):
            tree_bonds = frozenset(
                bond_id
                for component_tree in tree_bond_sets
                for bond_id in component_tree
            )
            ring_bonds = frozenset(bond.id for bond in facts.bonds) - tree_bonds

            parent: dict[AtomId, AtomId | None] = {}
            child_bond_by_parent: dict[AtomId, list[tuple[BondId, AtomId]]] = {
                atom.id: [] for atom in facts.atoms
            }
            ring_events_by_atom: dict[AtomId, list[RingEvent]] = {
                atom.id: [] for atom in facts.atoms
            }

            for root, component_tree in zip(roots, tree_bond_sets):
                _orient_tree_component(
                    index,
                    root,
                    parent,
                    child_bond_by_parent,
                    allowed_tree_bonds=frozenset(component_tree),
                )

            for bond in facts.bonds:
                if bond.id not in ring_bonds:
                    continue
                ring_events_by_atom[bond.a].append(
                    RingEvent(bond=bond.id, atom=bond.a, other_atom=bond.b)
                )
                ring_events_by_atom[bond.b].append(
                    RingEvent(bond=bond.id, atom=bond.b, other_atom=bond.a)
                )

            local_order_domains = tuple(
                (
                    atom.id,
                    tuple(
                        _local_event_orders(
                            atom.id,
                            child_bond_by_parent[atom.id],
                            ring_events_by_atom[atom.id],
                        )
                    ),
                )
                for atom in facts.atoms
            )
            for events_at_items in product(
                *(domain for _, domain in local_order_domains)
            ):
                events_at = {
                    atom: events
                    for (atom, _), events in zip(local_order_domains, events_at_items)
                }
                skeletons.append(
                    TraversalSkeleton(
                        roots=tuple(roots),
                        parent=MappingProxyType(dict(parent)),
                        tree_bonds=tree_bonds,
                        ring_bonds=ring_bonds,
                        events_at=MappingProxyType(events_at),
                    )
                )

    return tuple(skeletons)


def _component_root_domains(
    facts: MoleculeFacts,
    rooted_at_atom: AtomId | None,
) -> tuple[tuple[AtomId, ...], ...]:
    if rooted_at_atom is None:
        return tuple(component.atoms for component in facts.components)
    domains: list[tuple[AtomId, ...]] = []
    found = False
    for component in facts.components:
        if rooted_at_atom in component.atoms:
            domains.append((rooted_at_atom,))
            found = True
        else:
            domains.append(component.atoms)
    if not found:
        raise ValueError(f"rooted atom is not present in any component: {rooted_at_atom!r}")
    return tuple(domains)


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
    *,
    allowed_tree_bonds: frozenset[BondId] | None = None,
) -> None:
    parent[root] = None
    stack = [root]

    while stack:
        atom = stack.pop()
        for bond_id in reversed(index.incident_bonds[atom]):
            if allowed_tree_bonds is not None and bond_id not in allowed_tree_bonds:
                continue
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


def _local_event_orders(
    parent: AtomId,
    children: list[tuple[BondId, AtomId]],
    ring_events: list[RingEvent],
) -> tuple[tuple[LocalEvent, ...], ...]:
    branch_children = tuple(
        ChildEvent(
            bond=bond_id,
            parent=parent,
            child=child,
            role=ChildRole.BRANCH,
        )
        for bond_id, child in children
    )
    ring_event_tuple = tuple(ring_events)
    orders = set(permutations(ring_event_tuple + branch_children))

    for ordered_children in permutations(children):
        if not ordered_children:
            continue
        continuation = ChildEvent(
            bond=ordered_children[-1][0],
            parent=parent,
            child=ordered_children[-1][1],
            role=ChildRole.CONTINUATION,
        )
        decoration_children = tuple(
            ChildEvent(
                bond=bond_id,
                parent=parent,
                child=child,
                role=ChildRole.BRANCH,
            )
            for bond_id, child in ordered_children[:-1]
        )
        for decorations in permutations(ring_event_tuple + decoration_children):
            orders.add(decorations + (continuation,))

    return tuple(sorted(orders, key=repr))


def _component_spanning_trees(
    index: GraphIndex,
    atoms: tuple[AtomId, ...],
    bonds: tuple[BondId, ...],
) -> tuple[frozenset[BondId], ...]:
    if len(atoms) == 1:
        if bonds:
            raise ValueError("single-atom component cannot have bonds")
        return (frozenset(),)

    atom_set = set(atoms)
    trees: list[frozenset[BondId]] = []
    for candidate in combinations(bonds, len(atoms) - 1):
        candidate_set = frozenset(candidate)
        if (
            _reachable_atoms_on_bonds(index, atoms[0], atom_set, candidate_set)
            == atom_set
        ):
            trees.append(candidate_set)

    if not trees:
        raise ValueError("component has no spanning tree")
    return tuple(trees)


def _reachable_atoms_on_bonds(
    index: GraphIndex,
    start: AtomId,
    allowed_atoms: set[AtomId],
    allowed_bonds: frozenset[BondId],
) -> set[AtomId]:
    seen: set[AtomId] = set()
    stack = [start]
    while stack:
        atom = stack.pop()
        if atom in seen:
            continue
        seen.add(atom)
        for bond_id in index.incident_bonds[atom]:
            if bond_id not in allowed_bonds:
                continue
            bond = index.bond_by_id[bond_id]
            neighbor = bond.b if bond.a == atom else bond.a
            if neighbor in allowed_atoms and neighbor not in seen:
                stack.append(neighbor)
    return seen


__all__ = (
    "ChildEvent",
    "ChildRole",
    "LocalEvent",
    "RingEvent",
    "TraversalSkeleton",
    "enumerate_tree_skeletons",
    "enumerate_traversal_skeletons",
)
