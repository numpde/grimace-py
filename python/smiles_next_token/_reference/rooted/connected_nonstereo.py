from __future__ import annotations

import random
from bisect import insort
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import permutations, product
from typing import Iterable, Iterator

from rdkit import Chem

from smiles_next_token._reference.prepared_graph import (
    CONNECTED_NONSTEREO_SURFACE,
    PreparedSmilesGraph,
    build_atom_tokens as _build_atom_tokens_from_mol,
    check_supported_smiles_graph_surface,
    prepare_smiles_graph,
    ring_label_text,
)
from smiles_next_token._reference.policy import ReferencePolicy


@dataclass(frozen=True)
class PendingRing:
    label: int
    bond_token: str


@dataclass(frozen=True)
class SearchResult:
    smiles: str
    visited: frozenset[int]
    pending: tuple[tuple[int, tuple[PendingRing, ...]], ...]
    free_labels: tuple[int, ...]
    next_label: int


@dataclass(frozen=True)
class EmitTokenAction:
    token: str


@dataclass(frozen=True)
class EnterAtomAction:
    atom_idx: int


@dataclass(frozen=True)
class AfterAtomAction:
    atom_idx: int
    closures_here: tuple[PendingRing, ...]
    neighbor_groups: tuple[tuple[int, ...], ...]
    group_sizes: tuple[int, ...]
    opening_count: int
    ring_action_count: int
    linear_child_idx: int | None


@dataclass(slots=True)
class RootedConnectedNonStereoWalkerState:
    prefix_tokens: list[str]
    visited: set[int]
    pending: dict[int, tuple[PendingRing, ...]]
    free_labels: tuple[int, ...]
    next_label: int
    action_stack: list[object]

    @property
    def prefix(self) -> str:
        return "".join(self.prefix_tokens)

    @property
    def tokens(self) -> tuple[str, ...]:
        return tuple(self.prefix_tokens)


def check_supported_surface(mol: Chem.Mol) -> None:
    check_supported_smiles_graph_surface(mol, surface_kind=CONNECTED_NONSTEREO_SURFACE)


def build_atom_tokens(
    mol: Chem.Mol | PreparedSmilesGraph,
    policy: ReferencePolicy | None = None,
) -> tuple[str, ...]:
    if isinstance(mol, PreparedSmilesGraph):
        if policy is not None:
            mol.validate_policy(policy)
        return mol.atom_tokens
    if policy is None:
        raise TypeError("policy is required when building atom tokens from an RDKit molecule")
    return _build_atom_tokens_from_mol(mol, policy)


def pending_to_tuple(
    pending: dict[int, tuple[PendingRing, ...]],
) -> tuple[tuple[int, tuple[PendingRing, ...]], ...]:
    return tuple(sorted(pending.items()))


def tuple_to_pending(
    pending: tuple[tuple[int, tuple[PendingRing, ...]], ...],
) -> dict[int, tuple[PendingRing, ...]]:
    return dict(pending)


def ordered_neighbor_groups(
    prepared: PreparedSmilesGraph,
    atom_idx: int,
    visited: set[int] | frozenset[int],
) -> tuple[tuple[int, ...], ...]:
    remaining_neighbors = {
        neighbor_idx
        for neighbor_idx in prepared.neighbors_of(atom_idx)
        if neighbor_idx not in visited
    }
    if not remaining_neighbors:
        return ()

    blocked = set(visited)
    blocked.add(atom_idx)
    groups_with_mins: list[tuple[int, tuple[int, ...]]] = []

    while remaining_neighbors:
        seed = min(remaining_neighbors)
        remaining_neighbors.remove(seed)
        queue = deque([seed])
        seen = {seed}
        component_min = seed
        group = [seed]

        while queue:
            current = queue.popleft()
            if current < component_min:
                component_min = current
            for neighbor_idx in prepared.neighbors_of(current):
                if neighbor_idx in blocked or neighbor_idx in seen:
                    continue
                seen.add(neighbor_idx)
                if neighbor_idx in remaining_neighbors:
                    remaining_neighbors.remove(neighbor_idx)
                    group.append(neighbor_idx)
                queue.append(neighbor_idx)

        groups_with_mins.append((component_min, tuple(sorted(group))))

    return tuple(
        group
        for _, group in sorted(groups_with_mins, key=lambda item: item[0])
    )


def add_pending(
    pending: dict[int, tuple[PendingRing, ...]],
    target_atom: int,
    ring: PendingRing,
) -> dict[int, tuple[PendingRing, ...]]:
    updated = dict(pending)
    current = list(updated.get(target_atom, ()))
    current.append(ring)
    updated[target_atom] = tuple(sorted(current, key=lambda item: (item.label, item.bond_token)))
    return updated


def free_label(
    free_labels: tuple[int, ...],
    label: int,
) -> tuple[int, ...]:
    labels = list(free_labels)
    insort(labels, label)
    return tuple(labels)


def allocate_label(
    free_labels: tuple[int, ...],
    next_label: int,
) -> tuple[int, tuple[int, ...], int]:
    if free_labels:
        return free_labels[0], free_labels[1:], next_label
    return next_label, free_labels, next_label + 1


def unique_permutations(items: Iterable[object]) -> Iterator[tuple[object, ...]]:
    values = tuple(items)
    if not values:
        yield ()
        return

    if all(
        values[index] != values[other_index]
        for index in range(len(values))
        for other_index in range(index)
    ):
        yield from permutations(values)
        return

    current: list[object] = []
    used = [False] * len(values)

    def recurse() -> Iterator[tuple[object, ...]]:
        if len(current) == len(values):
            yield tuple(current)
            return

        seen_at_depth: list[object] = []
        for index, value in enumerate(values):
            if used[index]:
                continue
            if any(value == seen for seen in seen_at_depth):
                continue
            seen_at_depth.append(value)
            used[index] = True
            current.append(value)
            yield from recurse()
            current.pop()
            used[index] = False

    yield from recurse()


def enumerate_from_atom(
    prepared: PreparedSmilesGraph,
    atom_idx: int,
    visited: frozenset[int],
    pending_state: tuple[tuple[int, tuple[PendingRing, ...]], ...],
    free_labels: tuple[int, ...],
    next_label: int,
) -> Iterator[SearchResult]:
    visited_now = visited | {atom_idx}
    token = prepared.atom_tokens[atom_idx]

    pending = tuple_to_pending(pending_state)
    closures_here = pending.pop(atom_idx, ())
    ordered_groups = list(ordered_neighbor_groups(prepared, atom_idx, visited_now))
    child_choice_space = [group for group in ordered_groups]

    for chosen_children in product(*child_choice_space) if child_choice_space else [()]:
        child_set = set(chosen_children)
        opening_targets = [
            neighbor_idx
            for group in ordered_groups
            for neighbor_idx in group
            if neighbor_idx not in child_set
        ]
        indexed_closures = list(enumerate(closures_here))
        ring_actions = tuple(
            [("close", index) for index, _ in indexed_closures]
            + [("open", target_idx) for target_idx in opening_targets]
        )

        for ring_action_order in unique_permutations(ring_actions):
            current_pending = dict(pending)
            current_free = free_labels
            current_next_label = next_label
            current_ring_text: list[str] = []
            labels_freed_after_atom: list[int] = []

            for action_kind, payload in ring_action_order:
                if action_kind == "close":
                    closure = closures_here[payload]
                    current_ring_text.append(f"{closure.bond_token}{ring_label_text(closure.label)}")
                    labels_freed_after_atom.append(closure.label)
                    continue

                target_idx = int(payload)
                label, current_free, current_next_label = allocate_label(
                    current_free,
                    current_next_label,
                )
                current_ring_text.append(ring_label_text(label))
                current_pending = add_pending(
                    current_pending,
                    target_idx,
                    PendingRing(label=label, bond_token=prepared.bond_token(atom_idx, target_idx)),
                )

            for label in labels_freed_after_atom:
                current_free = free_label(current_free, label)

            child_orders = unique_permutations(chosen_children)
            for child_order in child_orders:
                yield from expand_children(
                    prepared=prepared,
                    parent_idx=atom_idx,
                    child_order=child_order,
                    prefix=token + "".join(current_ring_text),
                    visited=visited_now,
                    pending_state=pending_to_tuple(current_pending),
                    free_labels=current_free,
                    next_label=current_next_label,
                )


def expand_children(
    prepared: PreparedSmilesGraph,
    parent_idx: int,
    child_order: tuple[object, ...],
    prefix: str,
    visited: frozenset[int],
    pending_state: tuple[tuple[int, tuple[PendingRing, ...]], ...],
    free_labels: tuple[int, ...],
    next_label: int,
) -> Iterator[SearchResult]:
    if not child_order:
        yield SearchResult(
            smiles=prefix,
            visited=visited,
            pending=pending_state,
            free_labels=free_labels,
            next_label=next_label,
        )
        return

    branch_children = child_order[:-1]
    main_child = int(child_order[-1])
    edge_prefix = prepared.bond_token(parent_idx, main_child)

    def recurse_branch_children(
        branch_index: int,
        partial: SearchResult,
    ) -> Iterator[SearchResult]:
        if branch_index == len(branch_children):
            for main_result in enumerate_from_atom(
                prepared=prepared,
                atom_idx=main_child,
                visited=partial.visited,
                pending_state=partial.pending,
                free_labels=partial.free_labels,
                next_label=partial.next_label,
            ):
                yield SearchResult(
                    smiles=f"{partial.smiles}{edge_prefix}{main_result.smiles}",
                    visited=main_result.visited,
                    pending=main_result.pending,
                    free_labels=main_result.free_labels,
                    next_label=main_result.next_label,
                )
            return

        child_idx = int(branch_children[branch_index])
        branch_prefix = prepared.bond_token(parent_idx, child_idx)
        for branch_result in enumerate_from_atom(
            prepared=prepared,
            atom_idx=child_idx,
            visited=partial.visited,
            pending_state=partial.pending,
            free_labels=partial.free_labels,
            next_label=partial.next_label,
        ):
            yield from recurse_branch_children(
                branch_index + 1,
                SearchResult(
                    smiles=f"{partial.smiles}({branch_prefix}{branch_result.smiles})",
                    visited=branch_result.visited,
                    pending=branch_result.pending,
                    free_labels=branch_result.free_labels,
                    next_label=branch_result.next_label,
                ),
            )

    yield from recurse_branch_children(
        0,
        SearchResult(
            smiles=prefix,
            visited=visited,
            pending=pending_state,
            free_labels=free_labels,
            next_label=next_label,
        ),
    )


class RootedConnectedNonStereoWalker:
    def __init__(
        self,
        mol: Chem.Mol | PreparedSmilesGraph,
        root_idx: int,
        policy: ReferencePolicy | None = None,
    ) -> None:
        prepared = _coerce_prepared_graph(mol, policy)
        if prepared.atom_count and (root_idx < 0 or root_idx >= prepared.atom_count):
            raise IndexError("root_idx out of range")
        self.prepared = prepared
        self.root_idx = root_idx

    @classmethod
    def from_mol(
        cls,
        mol: Chem.Mol,
        root_idx: int,
        policy: ReferencePolicy,
    ) -> "RootedConnectedNonStereoWalker":
        return cls(mol, root_idx, policy)

    def initial_state(self) -> RootedConnectedNonStereoWalkerState:
        action_stack: list[object] = []
        if self.prepared.atom_count:
            action_stack.append(EnterAtomAction(self.root_idx))
        return RootedConnectedNonStereoWalkerState(
            prefix_tokens=[],
            visited=set(),
            pending={},
            free_labels=(),
            next_label=1,
            action_stack=action_stack,
        )

    def clone_state(
        self,
        state: RootedConnectedNonStereoWalkerState,
    ) -> RootedConnectedNonStereoWalkerState:
        return RootedConnectedNonStereoWalkerState(
            prefix_tokens=list(state.prefix_tokens),
            visited=set(state.visited),
            pending={atom_idx: tuple(rings) for atom_idx, rings in state.pending.items()},
            free_labels=state.free_labels,
            next_label=state.next_label,
            action_stack=list(state.action_stack),
        )

    def normalize_state(self, state: RootedConnectedNonStereoWalkerState) -> None:
        while state.action_stack:
            action = state.action_stack[-1]
            if not isinstance(action, AfterAtomAction):
                return
            if action.ring_action_count > 0 or action.neighbor_groups:
                return
            state.action_stack.pop()

    def is_terminal(self, state: RootedConnectedNonStereoWalkerState) -> bool:
        self.normalize_state(state)
        return not state.action_stack

    def next_token_support(
        self,
        state: RootedConnectedNonStereoWalkerState,
    ) -> tuple[str, ...]:
        action = self._normalized_action(state)
        if action is None:
            return ()
        if isinstance(action, EmitTokenAction):
            return (action.token,)
        if isinstance(action, EnterAtomAction):
            return (self.prepared.atom_tokens[action.atom_idx],)
        if not isinstance(action, AfterAtomAction):
            raise TypeError(f"Unsupported action: {type(action)!r}")

        if action.ring_action_count > 0:
            tokens = {
                closure.bond_token or ring_label_text(closure.label)
                for closure in action.closures_here
            }
            if action.opening_count:
                tokens.add(self._next_open_label_token(state))
            return tuple(sorted(tokens))

        if action.linear_child_idx is not None:
            return (self._edge_prefix_or_atom(action.atom_idx, action.linear_child_idx),)

        return ("(",)

    def advance_token(
        self,
        state: RootedConnectedNonStereoWalkerState,
        chosen_token: str,
        *,
        rng: random.Random | None = None,
    ) -> RootedConnectedNonStereoWalkerState:
        successors_by_token = self._successors_by_token(state)
        if chosen_token not in successors_by_token:
            available = tuple(sorted(successors_by_token))
            raise KeyError(f"Token {chosen_token!r} is not available; choices={available!r}")

        candidates = successors_by_token[chosen_token]
        chosen_successor = candidates[0] if rng is None else rng.choice(candidates)
        self._replace_state(state, chosen_successor)
        return state

    def _replace_state(
        self,
        target: RootedConnectedNonStereoWalkerState,
        source: RootedConnectedNonStereoWalkerState,
    ) -> None:
        target.prefix_tokens = list(source.prefix_tokens)
        target.visited = set(source.visited)
        target.pending = {atom_idx: tuple(rings) for atom_idx, rings in source.pending.items()}
        target.free_labels = source.free_labels
        target.next_label = source.next_label
        target.action_stack = list(source.action_stack)

    def _normalized_action(
        self,
        state: RootedConnectedNonStereoWalkerState,
    ) -> object | None:
        self.normalize_state(state)
        if not state.action_stack:
            return None
        return state.action_stack[-1]

    def _successors_by_token(
        self,
        state: RootedConnectedNonStereoWalkerState,
    ) -> dict[str, list[RootedConnectedNonStereoWalkerState]]:
        action = self._normalized_action(state)
        if action is None:
            return {}

        if isinstance(action, EmitTokenAction):
            successor = self.clone_state(state)
            successor.action_stack.pop()
            successor.prefix_tokens.append(action.token)
            self.normalize_state(successor)
            return {action.token: [successor]}

        if isinstance(action, EnterAtomAction):
            successor = self.clone_state(state)
            successor.action_stack.pop()
            successor.prefix_tokens.append(self.prepared.atom_tokens[action.atom_idx])
            self._consume_enter_atom(successor, action.atom_idx)
            self.normalize_state(successor)
            return {self.prepared.atom_tokens[action.atom_idx]: [successor]}

        if not isinstance(action, AfterAtomAction):
            raise TypeError(f"Unsupported action: {type(action)!r}")

        if action.ring_action_count > 0:
            return self._ring_action_successors(state, action)
        if not action.neighbor_groups:
            return {}
        if action.linear_child_idx is not None:
            successor = self.clone_state(state)
            successor.action_stack.pop()
            child_idx = action.linear_child_idx
            token = self._edge_prefix_or_atom(action.atom_idx, child_idx)
            successor.prefix_tokens.append(token)
            edge_prefix = self.prepared.bond_token(action.atom_idx, child_idx)
            if edge_prefix:
                successor.action_stack.append(EnterAtomAction(child_idx))
            else:
                self._consume_enter_atom(successor, child_idx)
            self.normalize_state(successor)
            return {token: [successor]}

        successors: defaultdict[str, list[RootedConnectedNonStereoWalkerState]] = defaultdict(list)
        child_order_seed = tuple(group[0] for group in action.neighbor_groups)
        for child_order in unique_permutations(child_order_seed):
            successor = self.clone_state(state)
            successor.action_stack.pop()
            successor.prefix_tokens.append("(")
            self._push_child_actions(
                successor.action_stack,
                action.atom_idx,
                tuple(int(child_idx) for child_idx in child_order),
                first_branch_open_consumed=True,
            )
            self.normalize_state(successor)
            successors["("].append(successor)
        return dict(successors)

    def _ring_action_successors(
        self,
        state: RootedConnectedNonStereoWalkerState,
        action: AfterAtomAction,
    ) -> dict[str, list[RootedConnectedNonStereoWalkerState]]:
        successors: defaultdict[str, list[RootedConnectedNonStereoWalkerState]] = defaultdict(list)

        child_choice_space = [group for group in action.neighbor_groups]
        child_choices = product(*child_choice_space) if child_choice_space else [()]
        for chosen_children in child_choices:
            chosen_children_tuple = tuple(int(child_idx) for child_idx in chosen_children)
            opening_targets = self._opening_targets_from_choices(
                action.neighbor_groups,
                chosen_children_tuple,
            )
            ring_actions = tuple(
                [("close", index) for index in range(len(action.closures_here))]
                + [("open", target_idx) for target_idx in opening_targets]
            )
            for ring_action_order in unique_permutations(ring_actions):
                first_kind, first_payload = ring_action_order[0]
                if first_kind == "close":
                    closure = action.closures_here[int(first_payload)]
                    first_token = closure.bond_token or ring_label_text(closure.label)
                else:
                    first_token = self._next_open_label_token(state)

                for child_order in unique_permutations(chosen_children_tuple):
                    successor = self.clone_state(state)
                    successor.action_stack.pop()
                    successor.prefix_tokens.append(first_token)
                    self._apply_exact_ring_plan(
                        successor,
                        action,
                        ring_action_order=tuple(
                            (kind, int(payload))
                            for kind, payload in ring_action_order
                        ),
                        child_order=tuple(int(child_idx) for child_idx in child_order),
                    )
                    self.normalize_state(successor)
                    successors[first_token].append(successor)

        return dict(successors)

    def _consume_enter_atom(
        self,
        state: RootedConnectedNonStereoWalkerState,
        atom_idx: int,
    ) -> None:
        if atom_idx in state.visited:
            raise ValueError(f"Atom {atom_idx} is already visited")

        state.visited.add(atom_idx)
        closures_here = state.pending.pop(atom_idx, ())
        neighbor_groups = ordered_neighbor_groups(self.prepared, atom_idx, state.visited)
        state.action_stack.append(
            self._make_after_atom_action(atom_idx, closures_here, neighbor_groups)
        )

    def _make_after_atom_action(
        self,
        atom_idx: int,
        closures_here: tuple[PendingRing, ...],
        neighbor_groups: tuple[tuple[int, ...], ...],
    ) -> AfterAtomAction:
        group_sizes = tuple(len(group) for group in neighbor_groups)
        opening_count = sum(size - 1 for size in group_sizes)
        ring_action_count = len(closures_here) + opening_count
        linear_child_idx = None
        if ring_action_count == 0 and len(neighbor_groups) == 1 and len(neighbor_groups[0]) == 1:
            linear_child_idx = neighbor_groups[0][0]
        return AfterAtomAction(
            atom_idx=atom_idx,
            closures_here=closures_here,
            neighbor_groups=neighbor_groups,
            group_sizes=group_sizes,
            opening_count=opening_count,
            ring_action_count=ring_action_count,
            linear_child_idx=linear_child_idx,
        )

    def _next_open_label_token(self, state: RootedConnectedNonStereoWalkerState) -> str:
        label = state.free_labels[0] if state.free_labels else state.next_label
        return ring_label_text(label)

    def _edge_prefix_or_atom(self, parent_idx: int, child_idx: int) -> str:
        token = self.prepared.bond_token(parent_idx, child_idx)
        if token is None:
            raise KeyError(f"No bond between atoms {parent_idx} and {child_idx}")
        return token or self.prepared.atom_tokens[child_idx]

    def _apply_exact_ring_plan(
        self,
        state: RootedConnectedNonStereoWalkerState,
        action: AfterAtomAction,
        *,
        ring_action_order: tuple[tuple[str, int], ...],
        child_order: tuple[int, ...],
    ) -> None:
        current_pending = {
            target_atom: list(rings)
            for target_atom, rings in state.pending.items()
        }
        current_free = list(state.free_labels)
        current_next = state.next_label
        freed_labels: list[int] = []
        emitted_suffix_tokens: list[str] = []

        for index, (kind, payload) in enumerate(ring_action_order):
            is_first = index == 0
            if kind == "close":
                closure = action.closures_here[payload]
                if closure.bond_token:
                    if not is_first:
                        emitted_suffix_tokens.append(closure.bond_token)
                    emitted_suffix_tokens.append(ring_label_text(closure.label))
                elif not is_first:
                    emitted_suffix_tokens.append(ring_label_text(closure.label))
                freed_labels.append(closure.label)
                continue

            target_idx = payload
            label, current_free, current_next = allocate_label(current_free, current_next)
            current_pending = add_pending(
                {
                    target_atom: tuple(rings)
                    for target_atom, rings in current_pending.items()
                },
                target_idx,
                PendingRing(
                    label=label,
                    bond_token=self.prepared.bond_token(action.atom_idx, target_idx) or "",
                ),
            )
            current_pending = {
                target_atom: list(rings)
                for target_atom, rings in current_pending.items()
            }
            if not is_first:
                emitted_suffix_tokens.append(ring_label_text(label))

        for label in freed_labels:
            insort(current_free, label)

        state.pending = {
            target_atom: tuple(rings)
            for target_atom, rings in current_pending.items()
        }
        state.free_labels = tuple(current_free)
        state.next_label = current_next
        self._push_child_actions(state.action_stack, action.atom_idx, child_order)
        for token in reversed(emitted_suffix_tokens):
            state.action_stack.append(EmitTokenAction(token))

    def _opening_targets_from_choices(
        self,
        neighbor_groups: tuple[tuple[int, ...], ...],
        chosen_children: tuple[int, ...],
    ) -> list[int]:
        chosen_child_by_group = dict(enumerate(chosen_children))
        opening_targets: list[int] = []
        for group_index, group in enumerate(neighbor_groups):
            chosen_child = chosen_child_by_group[group_index]
            for neighbor_idx in group:
                if neighbor_idx != chosen_child:
                    opening_targets.append(neighbor_idx)
        return opening_targets

    def _push_child_actions(
        self,
        action_stack: list[object],
        parent_idx: int,
        child_order: tuple[int, ...],
        *,
        first_branch_open_consumed: bool = False,
    ) -> None:
        if not child_order:
            return

        branch_children = child_order[:-1]
        main_child = child_order[-1]

        main_prefix = self.prepared.bond_token(parent_idx, main_child) or ""
        action_stack.append(EnterAtomAction(main_child))
        if main_prefix:
            action_stack.append(EmitTokenAction(main_prefix))

        if not branch_children:
            return

        for child_idx in reversed(branch_children[1:]):
            action_stack.append(EmitTokenAction(")"))
            action_stack.append(EnterAtomAction(child_idx))
            edge_prefix = self.prepared.bond_token(parent_idx, child_idx) or ""
            if edge_prefix:
                action_stack.append(EmitTokenAction(edge_prefix))
            action_stack.append(EmitTokenAction("("))

        first_branch_child = branch_children[0]
        action_stack.append(EmitTokenAction(")"))
        action_stack.append(EnterAtomAction(first_branch_child))
        first_edge_prefix = self.prepared.bond_token(parent_idx, first_branch_child) or ""
        if first_edge_prefix:
            action_stack.append(EmitTokenAction(first_edge_prefix))
        if not first_branch_open_consumed:
            action_stack.append(EmitTokenAction("("))


def _coerce_prepared_graph(
    mol_or_prepared: Chem.Mol | PreparedSmilesGraph,
    policy: ReferencePolicy | None,
) -> PreparedSmilesGraph:
    if isinstance(mol_or_prepared, PreparedSmilesGraph):
        if policy is not None:
            mol_or_prepared.validate_policy(policy)
        return mol_or_prepared
    if hasattr(mol_or_prepared, "to_dict"):
        prepared = PreparedSmilesGraph.from_dict(mol_or_prepared.to_dict())
        if policy is not None:
            prepared.validate_policy(policy)
        return prepared
    if policy is None:
        raise TypeError("policy is required when preparing a graph from an RDKit molecule")
    return prepare_smiles_graph(
        mol_or_prepared,
        policy,
        surface_kind=CONNECTED_NONSTEREO_SURFACE,
    )


def enumerate_rooted_connected_nonstereo_smiles_support(
    mol: Chem.Mol | PreparedSmilesGraph,
    root_idx: int,
    policy: ReferencePolicy | None = None,
) -> set[str]:
    prepared = _coerce_prepared_graph(mol, policy)
    if prepared.atom_count == 0:
        return {""}
    if root_idx < 0 or root_idx >= prepared.atom_count:
        raise IndexError("root_idx out of range")

    results: set[str] = set()
    for result in enumerate_from_atom(
        prepared=prepared,
        atom_idx=root_idx,
        visited=frozenset(),
        pending_state=(),
        free_labels=(),
        next_label=1,
    ):
        if len(result.visited) != prepared.atom_count:
            continue
        if result.pending:
            continue
        results.add(result.smiles)
    return results


def validate_rooted_connected_nonstereo_smiles_support(
    mol: Chem.Mol | PreparedSmilesGraph,
    root_idx: int,
    policy: ReferencePolicy | None = None,
    support: Iterable[str] | None = None,
) -> list[tuple[str, str]]:
    prepared = _coerce_prepared_graph(mol, policy)
    target_identity = prepared.identity_smiles
    candidate_support = (
        support
        if support is not None
        else enumerate_rooted_connected_nonstereo_smiles_support(prepared, root_idx)
    )
    issues: list[tuple[str, str]] = []

    for smiles in sorted(candidate_support):
        parsed = Chem.MolFromSmiles(smiles)
        if parsed is None:
            issues.append((smiles, "failed to parse"))
            continue

        parsed_identity = prepared.identity_smiles_for(parsed)
        if parsed_identity != target_identity:
            issues.append((smiles, parsed_identity))

    return issues
