"""Independent finite traversal and prefix-space checks for support artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from itertools import permutations
from itertools import product


@dataclass(frozen=True, slots=True)
class FactGraph:
    atoms: tuple[int, ...]
    bonds: dict[int, tuple[int, int]]
    components: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    incident_bonds: dict[int, tuple[int, ...]]


def graph_from_facts_json(facts_json: dict[str, object]) -> FactGraph:
    atoms = tuple(int(atom["id"]) for atom in _require_list(facts_json["atoms"]))
    bonds = {
        int(bond["id"]): (int(bond["a"]), int(bond["b"]))
        for bond in _require_list(facts_json["bonds"])
    }
    incident: dict[int, list[int]] = {atom: [] for atom in atoms}
    for bond_id, (left, right) in bonds.items():
        incident[left].append(bond_id)
        incident[right].append(bond_id)
    components = tuple(
        (
            tuple(int(atom) for atom in _require_list(component["atoms"])),
            tuple(int(bond) for bond in _require_list(component["bonds"])),
        )
        for component in _require_list(facts_json["components"])
    )
    return FactGraph(
        atoms=atoms,
        bonds=bonds,
        components=components,
        incident_bonds={
            atom: tuple(sorted(items))
            for atom, items in incident.items()
        },
    )


def enumerate_root_tuples_from_policy_json(
    facts_json: dict[str, object],
    policy_json: dict[str, object],
) -> tuple[tuple[int, ...], ...]:
    del policy_json
    graph = graph_from_facts_json(facts_json)
    return tuple(product(*(atoms for atoms, _ in graph.components)))


def enumerate_spanning_forest_edge_sets(
    facts_json: dict[str, object],
) -> tuple[tuple[int, ...], ...]:
    graph = graph_from_facts_json(facts_json)
    component_trees = tuple(
        _component_spanning_tree_sets(graph, atoms, bonds)
        for atoms, bonds in graph.components
    )
    out: list[tuple[int, ...]] = []
    for chosen in product(*component_trees):
        out.append(tuple(sorted(bond for component in chosen for bond in component)))
    return tuple(out)


def orient_forest_from_roots(
    facts_json: dict[str, object],
    roots: tuple[int, ...],
    tree_bonds: tuple[int, ...],
) -> tuple[tuple[int, int | None], ...]:
    graph = graph_from_facts_json(facts_json)
    allowed = set(tree_bonds)
    parent: dict[int, int | None] = {}
    for root, (component_atoms, _) in zip(roots, graph.components, strict=True):
        if root not in component_atoms:
            raise ValueError("root is outside component")
        parent[root] = None
        stack = [root]
        while stack:
            atom = stack.pop()
            for bond in reversed(graph.incident_bonds[atom]):
                if bond not in allowed:
                    continue
                left, right = graph.bonds[bond]
                neighbor = right if left == atom else left
                if parent.get(atom) == neighbor:
                    continue
                if neighbor in parent:
                    raise ValueError("tree bond set is cyclic")
                parent[neighbor] = atom
                stack.append(neighbor)
        if not set(component_atoms) <= set(parent):
            raise ValueError("tree bond set does not connect component")
    return tuple(sorted(parent.items()))


def local_event_orders_for_oriented_forest(
    facts_json: dict[str, object],
    roots: tuple[int, ...],
    parent_items: tuple[tuple[int, int | None], ...],
    tree_bonds: tuple[int, ...],
    ring_bonds: tuple[int, ...],
) -> tuple[tuple[tuple[int, tuple[object, ...]], ...], ...]:
    del roots
    graph = graph_from_facts_json(facts_json)
    parent = dict(parent_items)
    children_by_parent: dict[int, list[tuple[int, int]]] = {
        atom: [] for atom in graph.atoms
    }
    for child, parent_atom in parent.items():
        if parent_atom is None:
            continue
        bond = _bond_between(graph.bonds, child, parent_atom)
        if bond is None or bond not in tree_bonds:
            raise ValueError("parent relation is not induced by tree bonds")
        children_by_parent[parent_atom].append((bond, child))

    rings_by_atom: dict[int, list[tuple[object, ...]]] = {atom: [] for atom in graph.atoms}
    for bond in ring_bonds:
        left, right = graph.bonds[bond]
        rings_by_atom[left].append(("ring", bond, left, right))
        rings_by_atom[right].append(("ring", bond, right, left))

    local_domains = []
    for atom in graph.atoms:
        orders = _local_event_orders(
            atom,
            sorted(children_by_parent[atom]),
            sorted(rings_by_atom[atom]),
        )
        local_domains.append((atom, orders))

    out = []
    for choices in product(*(orders for _, orders in local_domains)):
        out.append(
            tuple(
                (atom, events)
                for (atom, _), events in zip(local_domains, choices, strict=True)
            )
        )
    return tuple(out)


def expected_skeleton_keys_from_traversal_grammar(
    *,
    facts_json: dict[str, object],
    policy_json: dict[str, object],
) -> frozenset[tuple[object, ...]]:
    graph = graph_from_facts_json(facts_json)
    all_bonds = set(graph.bonds)
    out: set[tuple[object, ...]] = set()
    for roots in enumerate_root_tuples_from_policy_json(facts_json, policy_json):
        for tree_bonds in enumerate_spanning_forest_edge_sets(facts_json):
            ring_bonds = tuple(sorted(all_bonds - set(tree_bonds)))
            parent_items = orient_forest_from_roots(facts_json, roots, tree_bonds)
            for local_orders in local_event_orders_for_oriented_forest(
                facts_json,
                roots,
                parent_items,
                tree_bonds,
                ring_bonds,
            ):
                out.add(
                    (
                        tuple(roots),
                        parent_items,
                        tuple(tree_bonds),
                        ring_bonds,
                        local_orders,
                    )
                )
    return frozenset(out)


def expected_slot_bundle_key_from_traversal_decision(
    *,
    facts_json: dict[str, object],
    roots: tuple[int, ...],
    local_event_orders: tuple[tuple[int, tuple[object, ...]], ...],
) -> tuple[object, ...]:
    graph = graph_from_facts_json(facts_json)
    events_by_atom = dict(local_event_orders)
    atom_slots = tuple((index, atom) for index, atom in enumerate(graph.atoms))
    bond_slots: list[tuple[object, ...]] = []
    ring_endpoints: list[tuple[object, ...]] = []
    carrier_slots: list[tuple[object, ...]] = []
    seen_tree_bonds: set[int] = set()

    def walk(atom: int):
        for event in events_by_atom[atom]:
            yield event
            if event[0] == "child":
                yield from walk(int(event[3]))

    syntax_position = 0
    for root in roots:
        for event in walk(root):
            if event[0] == "child":
                _, bond, parent, child, _ = event
                bond = int(bond)
                parent = int(parent)
                child = int(child)
                if bond in seen_tree_bonds:
                    raise ValueError("tree bond has multiple syntax slots")
                seen_tree_bonds.add(bond)
                bond_slot = len(bond_slots)
                bond_slots.append(
                    (bond_slot, bond, "tree", parent, child, syntax_position, None)
                )
                carrier_slots.append(
                    (len(carrier_slots), bond_slot, bond, parent, child)
                )
                syntax_position += 1
                continue
            if event[0] == "ring":
                _, bond, atom, other_atom = event
                bond = int(bond)
                atom = int(atom)
                other_atom = int(other_atom)
                endpoint = len(ring_endpoints)
                bond_slot = len(bond_slots)
                bond_slots.append(
                    (
                        bond_slot,
                        bond,
                        "ring_endpoint",
                        atom,
                        other_atom,
                        syntax_position,
                        endpoint,
                    )
                )
                ring_endpoints.append(
                    (endpoint, bond, atom, other_atom, bond_slot, syntax_position)
                )
                carrier_slots.append(
                    (len(carrier_slots), bond_slot, bond, atom, other_atom)
                )
                syntax_position += 1
                continue
            raise ValueError("unknown local event kind")

    return (
        atom_slots,
        tuple(bond_slots),
        tuple(ring_endpoints),
        tuple(carrier_slots),
    )


def expected_prefix_keys_from_policy_product(
    *,
    facts_json: dict[str, object],
    policy_json: dict[str, object],
    skeleton_key: tuple[object, ...],
    slot_bundle_key: tuple[object, ...],
) -> frozenset[tuple[object, ...]]:
    del skeleton_key
    atom_domains = tuple(
        (
            int(domain["atom"]),
            tuple(str(choice["name"]) for choice in _require_list(domain["choices"])),
        )
        for domain in _require_list(policy_json["atom_text_domains"])
    )
    bond_policy = _bond_policy_domains(policy_json)
    bond_slots = tuple(slot_bundle_key[1])
    ring_endpoints = tuple(slot_bundle_key[2])
    bond_domains = tuple(
        (
            int(slot[0]),
            bond_policy[(int(slot[1]), str(slot[2]))],
        )
        for slot in bond_slots
    )
    ring_assignments = enumerate_ring_label_assignments_from_slots(
        ring_endpoints=ring_endpoints,
        ring_labels=tuple(int(label) for label in _require_list(policy_json["ring_labels"])),
        least_free=bool(policy_json["least_free_ring_labels"]),
    )

    out: set[tuple[object, ...]] = set()
    for atom_values in product(*(values for _, values in atom_domains)):
        atom_part = tuple(
            sorted(
                zip((atom for atom, _ in atom_domains), atom_values, strict=True)
            )
        )
        for bond_values in product(*(values for _, values in bond_domains)):
            bond_part = tuple(
                sorted(
                    zip((slot for slot, _ in bond_domains), bond_values, strict=True)
                )
            )
            for ring_assignment in ring_assignments:
                out.add((atom_part, bond_part, ring_assignment))
    return frozenset(out)


def enumerate_ring_label_assignments_from_slots(
    *,
    ring_endpoints: tuple[object, ...],
    ring_labels: tuple[int, ...],
    least_free: bool,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    if not ring_endpoints:
        return ((),)
    intervals = _ring_intervals(ring_endpoints)
    assignments: list[tuple[tuple[int, int], ...]] = []
    chosen: list[tuple[int, int, int]] = []
    out: dict[int, int] = {}

    def active_labels_at(position: int) -> set[int]:
        return {
            label
            for start, end, label in chosen
            if start < position < end
        }

    def rec(index: int) -> None:
        if index == len(intervals):
            assignments.append(tuple(sorted(out.items())))
            return
        bond, endpoint_1, endpoint_2, start, end = intervals[index]
        del bond
        active = active_labels_at(start)
        candidates = tuple(label for label in ring_labels if label not in active)
        if least_free:
            candidates = () if not candidates else (min(candidates),)
        for label in candidates:
            out[endpoint_1] = label
            out[endpoint_2] = label
            chosen.append((start, end, label))
            rec(index + 1)
            chosen.pop()
            del out[endpoint_1]
            del out[endpoint_2]

    rec(0)
    return tuple(assignments)


def _component_spanning_tree_sets(
    graph: FactGraph,
    atoms: tuple[int, ...],
    bonds: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    if len(atoms) == 1:
        return ((),) if not bonds else ()
    out: list[tuple[int, ...]] = []
    for candidate in combinations(bonds, len(atoms) - 1):
        if _reachable_on_bonds(graph, atoms[0], set(atoms), set(candidate)) == set(atoms):
            out.append(tuple(sorted(candidate)))
    if not out:
        raise ValueError("component has no spanning tree")
    return tuple(out)


def _reachable_on_bonds(
    graph: FactGraph,
    start: int,
    allowed_atoms: set[int],
    allowed_bonds: set[int],
) -> set[int]:
    seen: set[int] = set()
    stack = [start]
    while stack:
        atom = stack.pop()
        if atom in seen:
            continue
        seen.add(atom)
        for bond in graph.incident_bonds[atom]:
            if bond not in allowed_bonds:
                continue
            left, right = graph.bonds[bond]
            neighbor = right if left == atom else left
            if neighbor in allowed_atoms and neighbor not in seen:
                stack.append(neighbor)
    return seen


def _local_event_orders(
    atom: int,
    children: list[tuple[int, int]],
    ring_events: list[tuple[object, ...]],
) -> tuple[tuple[object, ...], ...]:
    branch_children = tuple(
        ("child", bond, atom, child, "branch")
        for bond, child in children
    )
    ring_event_tuple = tuple(ring_events)
    orders = set(permutations(ring_event_tuple + branch_children))
    for ordered_children in permutations(children):
        if not ordered_children:
            continue
        continuation = (
            "child",
            ordered_children[-1][0],
            atom,
            ordered_children[-1][1],
            "continuation",
        )
        decorations = tuple(
            ("child", bond, atom, child, "branch")
            for bond, child in ordered_children[:-1]
        )
        for ordered_decorations in permutations(ring_event_tuple + decorations):
            orders.add(ordered_decorations + (continuation,))
    return tuple(sorted(orders, key=repr))


def _bond_between(
    bonds: Mapping[int, tuple[int, int]],
    left: int,
    right: int,
) -> int | None:
    pair = tuple(sorted((left, right)))
    for bond, endpoints in bonds.items():
        if tuple(sorted(endpoints)) == pair:
            return bond
    return None


def _bond_policy_domains(policy_json: dict[str, object]) -> dict[tuple[int, str], tuple[str, ...]]:
    out = {}
    for domain in _require_list(policy_json["bond_text_domains"]):
        out[(int(domain["bond"]), str(domain["slot_kind"]))] = tuple(
            str(choice["name"])
            for choice in _require_list(domain["choices"])
        )
    return out


def _ring_intervals(
    ring_endpoints: tuple[object, ...],
) -> tuple[tuple[int, int, int, int, int], ...]:
    by_bond: dict[int, list[tuple[int, int]]] = {}
    for endpoint in ring_endpoints:
        endpoint_id, bond, _, _, _, syntax_position = endpoint
        by_bond.setdefault(int(bond), []).append((int(endpoint_id), int(syntax_position)))
    intervals: list[tuple[int, int, int, int, int]] = []
    for bond, endpoints in by_bond.items():
        if len(endpoints) != 2:
            raise ValueError("ring bond does not have exactly two endpoints")
        (endpoint_1, position_1), (endpoint_2, position_2) = sorted(
            endpoints,
            key=lambda item: item[1],
        )
        start, end = sorted((position_1, position_2))
        intervals.append((bond, endpoint_1, endpoint_2, start, end))
    return tuple(sorted(intervals, key=lambda item: (item[3], item[0])))


def _require_list(value: object) -> list:
    if not isinstance(value, list):
        raise TypeError(f"expected list: {value!r}")
    return value


__all__ = (
    "FactGraph",
    "enumerate_ring_label_assignments_from_slots",
    "enumerate_root_tuples_from_policy_json",
    "enumerate_spanning_forest_edge_sets",
    "expected_prefix_keys_from_policy_product",
    "expected_skeleton_keys_from_traversal_grammar",
    "expected_slot_bundle_key_from_traversal_decision",
    "graph_from_facts_json",
    "local_event_orders_for_oriented_forest",
    "orient_forest_from_roots",
)
