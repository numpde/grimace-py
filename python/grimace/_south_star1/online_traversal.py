"""Lazy online traversal/event streams for South Star runtime work."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations
from itertools import permutations
from typing import TYPE_CHECKING

from .facts import MoleculeFacts
from .graph_index import GraphIndex
from .ids import AtomId
from .ids import BondId
from .online_traversal_graph import OnlineTraversalGraph
from .online_traversal_graph import build_online_traversal_graph_from_facts
from .online_traversal_graph import build_online_traversal_graph_from_index
from .policy import BranchPresentationMode
from .policy import SmilesPolicy
from .root_domains import component_root_domains_for_facts
from .skeleton import ChildRole

if TYPE_CHECKING:
    from .prepared_runtime import SouthStarPreparedMol


@dataclass(frozen=True, slots=True)
class OnlineAtomEvent:
    atom: AtomId
    parent: AtomId | None


@dataclass(frozen=True, slots=True)
class OnlineTreeBondEvent:
    bond: BondId
    written_from: AtomId
    written_to: AtomId
    role: ChildRole


@dataclass(frozen=True, slots=True)
class OnlineRingEndpointEvent:
    bond: BondId
    at: AtomId
    other_atom: AtomId
    syntax_position: int


@dataclass(frozen=True, slots=True)
class OnlineBranchOpen:
    pass


@dataclass(frozen=True, slots=True)
class OnlineBranchClose:
    pass


@dataclass(frozen=True, slots=True)
class OnlineDotEvent:
    pass


OnlineTraversalEvent = (
    OnlineAtomEvent
    | OnlineTreeBondEvent
    | OnlineRingEndpointEvent
    | OnlineBranchOpen
    | OnlineBranchClose
    | OnlineDotEvent
)


@dataclass(frozen=True, slots=True)
class OnlineTraversalTrace:
    roots: tuple[AtomId, ...]
    parent: tuple[tuple[AtomId, AtomId | None], ...]
    tree_bonds: frozenset[BondId]
    ring_bonds: frozenset[BondId]
    events: tuple[OnlineTraversalEvent, ...]


@dataclass(frozen=True, slots=True)
class _ChildLocalEvent:
    bond: BondId
    parent: AtomId
    child: AtomId
    role: ChildRole


@dataclass(frozen=True, slots=True)
class _RingLocalEvent:
    bond: BondId
    atom: AtomId
    other_atom: AtomId


_LocalEvent = _ChildLocalEvent | _RingLocalEvent


@dataclass(slots=True)
class _OnlineTraversalState:
    parent: dict[AtomId, AtomId | None]
    tree_bonds: frozenset[BondId]
    ring_bonds: frozenset[BondId]
    events_at: dict[AtomId, tuple[_LocalEvent, ...]]
    event_buffer: list[OnlineTraversalEvent]
    syntax_position: int = 0


def iter_online_traversal_traces(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    rooted_at_atom: AtomId | None = None,
    index: GraphIndex | None = None,
    component_root_domains: tuple[tuple[AtomId, ...], ...] | None = None,
) -> Iterator[OnlineTraversalTrace]:
    """Yield traversal traces lazily without materializing skeleton space."""

    facts.validate()
    policy.validate_for_facts(facts)
    graph = _graph_from_facts(facts, index=index)
    yield from _iter_online_traversal_traces_on_graph(
        facts=facts,
        graph=graph,
        rooted_at_atom=rooted_at_atom,
        component_root_domains=component_root_domains,
        branch_presentation_mode=policy.branch_presentation_mode,
    )


def iter_prepared_online_traversal_traces(
    *,
    prepared: SouthStarPreparedMol,
    rooted_at_atom: AtomId | None,
    component_root_domains: tuple[tuple[AtomId, ...], ...],
) -> Iterator[OnlineTraversalTrace]:
    """Yield prepared traversal traces without replaying raw validation."""

    yield from _iter_online_traversal_traces_on_graph(
        facts=prepared.facts,
        graph=prepared.online_traversal_graph,
        rooted_at_atom=rooted_at_atom,
        component_root_domains=component_root_domains,
        branch_presentation_mode=prepared.policy.branch_presentation_mode,
    )


def _iter_online_traversal_traces_on_graph(
    *,
    facts: MoleculeFacts,
    graph: OnlineTraversalGraph,
    rooted_at_atom: AtomId | None,
    component_root_domains: tuple[tuple[AtomId, ...], ...] | None,
    branch_presentation_mode: BranchPresentationMode,
) -> Iterator[OnlineTraversalTrace]:
    all_bonds = frozenset(graph.bonds)

    for roots in _iter_root_choices(
        graph,
        facts=facts,
        rooted_at_atom=rooted_at_atom,
        component_root_domains=component_root_domains,
    ):
        for tree_bonds in _iter_spanning_forest_choices_lazy(graph):
            ring_bonds = all_bonds - tree_bonds
            parent, children_by_parent = _orient_forest_from_roots(
                graph=graph,
                roots=roots,
                tree_bonds=tree_bonds,
            )
            ring_events_by_atom = _ring_events_by_atom(graph, ring_bonds)
            local_domains = tuple(
                (
                    atom,
                    tuple(
                        _local_event_orders_lazy(
                            atom,
                            children_by_parent[atom],
                            ring_events_by_atom[atom],
                            branch_presentation_mode=branch_presentation_mode,
                        )
                    ),
                )
                for atom in graph.atoms
            )
            yield from _iter_local_order_products(
                roots=roots,
                parent=parent,
                tree_bonds=tree_bonds,
                ring_bonds=ring_bonds,
                local_domains=local_domains,
            )


def online_trace_key(trace: OnlineTraversalTrace) -> tuple[object, ...]:
    return trace_to_skeleton_like_key(trace)


def trace_to_skeleton_like_key(trace: OnlineTraversalTrace) -> tuple[object, ...]:
    events_by_atom: dict[AtomId, list[tuple[object, ...]]] = {
        atom: [] for atom, _ in trace.parent
    }
    for event in trace.events:
        if isinstance(event, OnlineTreeBondEvent):
            events_by_atom[event.written_from].append(
                (
                    "child",
                    int(event.bond),
                    int(event.written_from),
                    int(event.written_to),
                    event.role.value,
                )
            )
            continue
        if isinstance(event, OnlineRingEndpointEvent):
            events_by_atom[event.at].append(
                (
                    "ring",
                    int(event.bond),
                    int(event.at),
                    int(event.other_atom),
                )
            )

    return (
        tuple(int(root) for root in trace.roots),
        tuple(
            sorted(
                (
                    int(atom),
                    None if parent is None else int(parent),
                )
                for atom, parent in trace.parent
            )
        ),
        tuple(sorted(int(bond) for bond in trace.tree_bonds)),
        tuple(sorted(int(bond) for bond in trace.ring_bonds)),
        tuple(
            sorted(
                (int(atom), tuple(events))
                for atom, events in events_by_atom.items()
            )
        ),
    )


def _graph_from_facts(
    facts: MoleculeFacts,
    *,
    index: GraphIndex | None,
) -> OnlineTraversalGraph:
    if index is None:
        return build_online_traversal_graph_from_facts(facts)
    return build_online_traversal_graph_from_index(facts, index)


def _iter_root_choices(
    graph: OnlineTraversalGraph,
    *,
    facts: MoleculeFacts,
    rooted_at_atom: AtomId | None = None,
    component_root_domains: tuple[tuple[AtomId, ...], ...] | None = None,
) -> Iterator[tuple[AtomId, ...]]:
    root_domains = _component_root_domains(
        graph,
        facts,
        rooted_at_atom,
        component_root_domains,
    )
    roots: list[AtomId] = []

    def rec(index: int) -> Iterator[tuple[AtomId, ...]]:
        if index == len(root_domains):
            yield tuple(roots)
            return
        for atom in root_domains[index]:
            roots.append(atom)
            yield from rec(index + 1)
            roots.pop()

    yield from rec(0)


def _component_root_domains(
    graph: OnlineTraversalGraph,
    facts: MoleculeFacts,
    rooted_at_atom: AtomId | None,
    component_root_domains: tuple[tuple[AtomId, ...], ...] | None,
) -> tuple[tuple[AtomId, ...], ...]:
    del graph
    if component_root_domains is not None:
        if len(component_root_domains) != len(facts.components):
            raise ValueError("component root domain count does not match molecule components")
        return component_root_domains
    return tuple(
        atoms
        for _, atoms in component_root_domains_for_facts(facts, rooted_at_atom)
    )


def _iter_spanning_forest_choices_lazy(
    graph: OnlineTraversalGraph,
) -> Iterator[frozenset[BondId]]:
    chosen: list[BondId] = []

    def rec(index: int) -> Iterator[frozenset[BondId]]:
        if index == len(graph.components):
            yield frozenset(chosen)
            return
        atoms, bonds = graph.components[index]
        for tree in _iter_component_spanning_trees(graph, atoms, bonds):
            checkpoint = len(chosen)
            chosen.extend(tree)
            yield from rec(index + 1)
            del chosen[checkpoint:]

    yield from rec(0)


def _iter_component_spanning_trees(
    graph: OnlineTraversalGraph,
    atoms: tuple[AtomId, ...],
    bonds: tuple[BondId, ...],
) -> Iterator[tuple[BondId, ...]]:
    if len(atoms) == 1:
        if bonds:
            raise ValueError("single-atom component cannot have bonds")
        yield ()
        return

    atom_set = set(atoms)
    for candidate in combinations(bonds, len(atoms) - 1):
        candidate_set = frozenset(candidate)
        if _reachable_atoms_on_bonds(graph, atoms[0], atom_set, candidate_set) == atom_set:
            yield tuple(candidate)


def _orient_forest_from_roots(
    *,
    graph: OnlineTraversalGraph,
    roots: tuple[AtomId, ...],
    tree_bonds: frozenset[BondId],
) -> tuple[dict[AtomId, AtomId | None], dict[AtomId, list[tuple[BondId, AtomId]]]]:
    parent: dict[AtomId, AtomId | None] = {}
    children_by_parent: dict[AtomId, list[tuple[BondId, AtomId]]] = {
        atom: [] for atom in graph.atoms
    }
    for root, (component_atoms, _) in zip(roots, graph.components, strict=True):
        if root not in component_atoms:
            raise ValueError("root is outside component")
        parent[root] = None
        stack = [root]
        while stack:
            atom = stack.pop()
            for bond in reversed(graph.incident_bonds[atom]):
                if bond not in tree_bonds:
                    continue
                left, right = graph.bonds[bond]
                neighbor = right if left == atom else left
                if parent.get(atom) == neighbor:
                    continue
                if neighbor in parent:
                    raise ValueError("tree bond set is cyclic")
                parent[neighbor] = atom
                children_by_parent[atom].append((bond, neighbor))
                stack.append(neighbor)
        if not set(component_atoms) <= set(parent):
            raise ValueError("tree bond set does not connect component")
    return parent, children_by_parent


def _ring_events_by_atom(
    graph: OnlineTraversalGraph,
    ring_bonds: frozenset[BondId],
) -> dict[AtomId, list[_RingLocalEvent]]:
    out: dict[AtomId, list[_RingLocalEvent]] = {atom: [] for atom in graph.atoms}
    for bond in graph.bonds:
        if bond not in ring_bonds:
            continue
        left, right = graph.bonds[bond]
        out[left].append(_RingLocalEvent(bond=bond, atom=left, other_atom=right))
        out[right].append(_RingLocalEvent(bond=bond, atom=right, other_atom=left))
    return out


def _iter_local_order_products(
    *,
    roots: tuple[AtomId, ...],
    parent: dict[AtomId, AtomId | None],
    tree_bonds: frozenset[BondId],
    ring_bonds: frozenset[BondId],
    local_domains: tuple[tuple[AtomId, tuple[tuple[_LocalEvent, ...], ...]], ...],
) -> Iterator[OnlineTraversalTrace]:
    events_at: dict[AtomId, tuple[_LocalEvent, ...]] = {}

    def rec(index: int) -> Iterator[OnlineTraversalTrace]:
        if index == len(local_domains):
            state = _OnlineTraversalState(
                parent=parent,
                tree_bonds=tree_bonds,
                ring_bonds=ring_bonds,
                events_at=events_at,
                event_buffer=[],
            )
            for root_index, root in enumerate(roots):
                if root_index:
                    state.event_buffer.append(OnlineDotEvent())
                _emit_atom_subtree(state, root)
            yield OnlineTraversalTrace(
                roots=roots,
                parent=tuple(sorted(parent.items())),
                tree_bonds=tree_bonds,
                ring_bonds=ring_bonds,
                events=tuple(state.event_buffer),
            )
            return

        atom, orders = local_domains[index]
        for order in orders:
            events_at[atom] = order
            yield from rec(index + 1)
            del events_at[atom]

    yield from rec(0)


def _local_event_orders_lazy(
    parent: AtomId,
    children: list[tuple[BondId, AtomId]],
    ring_events: list[_RingLocalEvent],
    *,
    branch_presentation_mode: BranchPresentationMode = BranchPresentationMode.EXHAUSTIVE,
) -> Iterator[tuple[_LocalEvent, ...]]:
    branch_children = tuple(
        _ChildLocalEvent(
            bond=bond,
            parent=parent,
            child=child,
            role=ChildRole.BRANCH,
        )
        for bond, child in children
    )
    ring_event_tuple = tuple(ring_events)
    seen: set[tuple[_LocalEvent, ...]] = set()

    if branch_presentation_mode is BranchPresentationMode.EXHAUSTIVE or not children:
        for order in permutations(ring_event_tuple + branch_children):
            if order not in seen:
                seen.add(order)
                yield order

    for ordered_children in permutations(children):
        if not ordered_children:
            continue
        continuation = _ChildLocalEvent(
            bond=ordered_children[-1][0],
            parent=parent,
            child=ordered_children[-1][1],
            role=ChildRole.CONTINUATION,
        )
        decoration_children = tuple(
            _ChildLocalEvent(
                bond=bond,
                parent=parent,
                child=child,
                role=ChildRole.BRANCH,
            )
            for bond, child in ordered_children[:-1]
        )
        for decorations in permutations(ring_event_tuple + decoration_children):
            order = decorations + (continuation,)
            if order not in seen:
                seen.add(order)
                yield order


def _emit_atom_subtree(state: _OnlineTraversalState, atom: AtomId) -> None:
    state.event_buffer.append(OnlineAtomEvent(atom=atom, parent=state.parent[atom]))
    for event in state.events_at[atom]:
        if isinstance(event, _RingLocalEvent):
            state.event_buffer.append(
                OnlineRingEndpointEvent(
                    bond=event.bond,
                    at=event.atom,
                    other_atom=event.other_atom,
                    syntax_position=state.syntax_position,
                )
            )
            state.syntax_position += 1
            continue
        if event.role is ChildRole.BRANCH:
            state.event_buffer.append(OnlineBranchOpen())
        state.event_buffer.append(
            OnlineTreeBondEvent(
                bond=event.bond,
                written_from=event.parent,
                written_to=event.child,
                role=event.role,
            )
        )
        state.syntax_position += 1
        _emit_atom_subtree(state, event.child)
        if event.role is ChildRole.BRANCH:
            state.event_buffer.append(OnlineBranchClose())


def _reachable_atoms_on_bonds(
    graph: OnlineTraversalGraph,
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
        for bond in graph.incident_bonds[atom]:
            if bond not in allowed_bonds:
                continue
            left, right = graph.bonds[bond]
            neighbor = right if left == atom else left
            if neighbor in allowed_atoms and neighbor not in seen:
                stack.append(neighbor)
    return seen


__all__ = (
    "OnlineAtomEvent",
    "OnlineBranchClose",
    "OnlineBranchOpen",
    "OnlineDotEvent",
    "OnlineRingEndpointEvent",
    "OnlineTraversalEvent",
    "OnlineTraversalTrace",
    "OnlineTreeBondEvent",
    "iter_prepared_online_traversal_traces",
    "iter_online_traversal_traces",
    "online_trace_key",
    "trace_to_skeleton_like_key",
)
