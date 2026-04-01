from __future__ import annotations

from bisect import insort
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import permutations, product
from typing import Iterable, Iterator

from rdkit import Chem

from grimace._reference.policy import ReferencePolicy
from grimace._reference.prepared_graph import (
    AROMATIC_SUBSET,
    CONNECTED_STEREO_SURFACE,
    PreparedSmilesGraph,
    format_charge,
    format_hydrogen_count,
    prepare_smiles_graph,
    ring_label_text,
)


_PERIODIC_TABLE = Chem.GetPeriodicTable()
_HYDROGEN_NEIGHBOR = -1
_UNKNOWN_COMPONENT_PHASE = -1
_STORED_COMPONENT_PHASE = 0
_FLIPPED_COMPONENT_PHASE = 1
_UNKNOWN_EDGE_ORIENTATION = -1
_BEFORE_ATOM_EDGE_ORIENTATION = 0
_AFTER_ATOM_EDGE_ORIENTATION = 1

_SUPPORTED_CHIRAL_TAGS = {
    "CHI_UNSPECIFIED",
    "CHI_TETRAHEDRAL_CCW",
    "CHI_TETRAHEDRAL_CW",
}
_CIS_STEREO_BOND_KINDS = {
    "STEREOCIS",
    "STEREOZ",
}
_TRANS_STEREO_BOND_KINDS = {
    "STEREOE",
    "STEREOTRANS",
}


@dataclass(frozen=True)
class PendingRing:
    label: int
    other_atom_idx: int


@dataclass(frozen=True)
class DeferredDirectionalToken:
    component_idx: int
    stored_token: str


@dataclass(frozen=True)
class SearchResult:
    parts: tuple[str | DeferredDirectionalToken, ...]
    visited: frozenset[int]
    pending: tuple[tuple[int, tuple[PendingRing, ...]], ...]
    free_labels: tuple[int, ...]
    next_label: int
    stereo_component_phases: tuple[int, ...]
    stereo_selected_neighbors: tuple[int, ...]
    stereo_selected_orientations: tuple[int, ...]
    stereo_component_begin_atoms: tuple[int, ...]


@dataclass(frozen=True)
class StereoSideInfo:
    component_idx: int
    endpoint_atom_idx: int
    other_endpoint_atom_idx: int
    candidate_neighbors: tuple[int, ...]
    candidate_base_tokens: tuple[str, ...]


def _flip_direction_token(token: str) -> str:
    if token == "/":
        return "\\"
    if token == "\\":
        return "/"
    raise ValueError(f"Unsupported directional token: {token!r}")


def _part_tuple(part: str | DeferredDirectionalToken) -> tuple[str | DeferredDirectionalToken, ...]:
    if isinstance(part, str) and not part:
        return ()
    return (part,)


def _resolve_part(part: str | DeferredDirectionalToken, component_phases: tuple[int, ...]) -> str:
    if isinstance(part, str):
        return part
    phase = component_phases[part.component_idx]
    if phase in {_UNKNOWN_COMPONENT_PHASE, _STORED_COMPONENT_PHASE}:
        return part.stored_token
    return _flip_direction_token(part.stored_token)


def _resolve_parts(
    parts: tuple[str | DeferredDirectionalToken, ...],
    component_phases: tuple[int, ...],
    *,
    component_flips: tuple[bool, ...] | None = None,
) -> str:
    resolved_parts: list[str] = []
    for part in parts:
        resolved = _resolve_part(part, component_phases)
        if (
            component_flips is not None
            and isinstance(part, DeferredDirectionalToken)
            and component_flips[part.component_idx]
        ):
            resolved = _flip_direction_token(resolved)
        resolved_parts.append(resolved)
    return "".join(resolved_parts)


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
    updated[target_atom] = tuple(
        sorted(current, key=lambda item: (item.label, item.other_atom_idx))
    )
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


def _prepared_atom_symbol(prepared: PreparedSmilesGraph, atom_idx: int) -> str:
    symbol = _PERIODIC_TABLE.GetElementSymbol(prepared.atom_atomic_numbers[atom_idx])
    if prepared.atom_is_aromatic[atom_idx] and not prepared.writer_kekule_smiles:
        lowered = symbol.lower()
        if lowered in AROMATIC_SUBSET:
            return lowered
    return symbol


def _permutation_parity(reference_order: tuple[int, ...], emitted_order: tuple[int, ...]) -> int:
    if len(reference_order) != len(emitted_order):
        raise ValueError("Stereo neighbor order length mismatch")
    if set(reference_order) != set(emitted_order):
        raise ValueError("Stereo neighbor order membership mismatch")

    reference_index = {
        neighbor: index
        for index, neighbor in enumerate(reference_order)
    }
    permutation = [reference_index[neighbor] for neighbor in emitted_order]
    inversions = 0
    for index, left in enumerate(permutation):
        for right in permutation[index + 1:]:
            if left > right:
                inversions += 1
    return inversions % 2


def _stereo_neighbor_order(
    prepared: PreparedSmilesGraph,
    atom_idx: int,
    *,
    parent_idx: int | None,
    ring_neighbor_order: tuple[int, ...],
    child_order: tuple[int, ...],
) -> tuple[int, ...]:
    hydrogen_count = prepared.atom_explicit_h_counts[atom_idx] + prepared.atom_implicit_h_counts[atom_idx]
    if hydrogen_count > 1:
        raise ValueError("Tetrahedral stereo currently supports at most one hydrogen ligand")

    emitted: list[int] = []
    if parent_idx is not None:
        emitted.append(parent_idx)
    if hydrogen_count == 1:
        emitted.append(_HYDROGEN_NEIGHBOR)
    emitted.extend(ring_neighbor_order)
    emitted.extend(child_order)
    return tuple(emitted)


def _stereo_atom_token(
    prepared: PreparedSmilesGraph,
    atom_idx: int,
    *,
    emitted_neighbor_order: tuple[int, ...],
) -> str:
    chiral_tag = prepared.atom_chiral_tags[atom_idx]
    if chiral_tag == "CHI_UNSPECIFIED":
        return prepared.atom_tokens[atom_idx]
    if chiral_tag not in _SUPPORTED_CHIRAL_TAGS:
        raise NotImplementedError(f"Unsupported chiral tag for rooted stereo emission: {chiral_tag}")

    hydrogen_count = prepared.atom_explicit_h_counts[atom_idx] + prepared.atom_implicit_h_counts[atom_idx]
    reference_order = list(prepared.atom_stereo_neighbor_orders[atom_idx])
    if hydrogen_count == 1:
        reference_order.append(_HYDROGEN_NEIGHBOR)

    parity = _permutation_parity(tuple(reference_order), emitted_neighbor_order)
    use_single_at = (
        (chiral_tag == "CHI_TETRAHEDRAL_CCW" and parity == 0)
        or (chiral_tag == "CHI_TETRAHEDRAL_CW" and parity == 1)
    )
    stereo_mark = "@" if use_single_at else "@@"

    parts = ["["]
    isotope = prepared.atom_isotopes[atom_idx]
    if isotope:
        parts.append(str(isotope))
    parts.append(_prepared_atom_symbol(prepared, atom_idx))
    parts.append(stereo_mark)
    parts.append(format_hydrogen_count(hydrogen_count))
    parts.append(format_charge(prepared.atom_formal_charges[atom_idx]))
    atom_map_number = 0 if prepared.writer_ignore_atom_map_numbers else prepared.atom_map_numbers[atom_idx]
    if atom_map_number:
        parts.append(f":{atom_map_number}")
    parts.append("]")
    return "".join(parts)


def _check_supported_stereo_writer_surface(prepared: PreparedSmilesGraph) -> None:
    if prepared.surface_kind != CONNECTED_STEREO_SURFACE:
        raise ValueError(f"Expected surface_kind={CONNECTED_STEREO_SURFACE!r}, got {prepared.surface_kind!r}")
    for atom_idx, chiral_tag in enumerate(prepared.atom_chiral_tags):
        if chiral_tag not in _SUPPORTED_CHIRAL_TAGS:
            raise NotImplementedError(f"Unsupported chiral tag at atom {atom_idx}: {chiral_tag}")


def _is_stereo_double_bond(prepared: PreparedSmilesGraph, bond_idx: int) -> bool:
    if prepared.bond_kinds[bond_idx] != "DOUBLE":
        return False
    stereo_kind = prepared.bond_stereo_kinds[bond_idx]
    return stereo_kind in _CIS_STEREO_BOND_KINDS or stereo_kind in _TRANS_STEREO_BOND_KINDS


def _canonical_edge(begin_idx: int, end_idx: int) -> tuple[int, int]:
    if begin_idx < end_idx:
        return (begin_idx, end_idx)
    return (end_idx, begin_idx)


def _stereo_component_ids(prepared: PreparedSmilesGraph) -> tuple[int, ...]:
    stereo_bond_indices = [
        bond_idx
        for bond_idx in range(prepared.bond_count)
        if _is_stereo_double_bond(prepared, bond_idx)
    ]
    if not stereo_bond_indices:
        return tuple(-1 for _ in range(prepared.bond_count))

    parents = {bond_idx: bond_idx for bond_idx in stereo_bond_indices}

    def find(bond_idx: int) -> int:
        root = bond_idx
        while parents[root] != root:
            root = parents[root]
        while parents[bond_idx] != bond_idx:
            next_idx = parents[bond_idx]
            parents[bond_idx] = root
            bond_idx = next_idx
        return root

    def union(left_idx: int, right_idx: int) -> None:
        left_root = find(left_idx)
        right_root = find(right_idx)
        if left_root != right_root:
            parents[right_root] = left_root

    edge_to_bonds: dict[tuple[int, int], list[int]] = defaultdict(list)
    for bond_idx in stereo_bond_indices:
        stored_begin_idx = prepared.bond_begin_atom_indices[bond_idx]
        stored_end_idx = prepared.bond_end_atom_indices[bond_idx]
        stereo_begin_atom, stereo_end_atom = prepared.bond_stereo_atoms[bond_idx]
        if stereo_begin_atom >= 0:
            edge_to_bonds[_canonical_edge(stored_begin_idx, stereo_begin_atom)].append(bond_idx)
        if stereo_end_atom >= 0:
            edge_to_bonds[_canonical_edge(stored_end_idx, stereo_end_atom)].append(bond_idx)

    for connected_bonds in edge_to_bonds.values():
        head_idx = connected_bonds[0]
        for other_idx in connected_bonds[1:]:
            union(head_idx, other_idx)

    component_lookup: dict[int, int] = {}
    component_ids = [-1 for _ in range(prepared.bond_count)]
    next_component_id = 0
    for bond_idx in stereo_bond_indices:
        root_idx = find(bond_idx)
        component_id = component_lookup.get(root_idx)
        if component_id is None:
            component_id = next_component_id
            component_lookup[root_idx] = component_id
            next_component_id += 1
        component_ids[bond_idx] = component_id

    return tuple(component_ids)


def _component_sizes(stereo_component_ids: tuple[int, ...]) -> tuple[int, ...]:
    component_count = max(stereo_component_ids, default=-1) + 1
    counts = [0 for _ in range(component_count)]
    for component_idx in stereo_component_ids:
        if component_idx >= 0:
            counts[component_idx] += 1
    return tuple(counts)


def _with_component_phase(
    component_phases: tuple[int, ...],
    *,
    component_idx: int,
    phase: int,
) -> tuple[int, ...]:
    existing = component_phases[component_idx]
    if existing == phase:
        return component_phases
    if existing != _UNKNOWN_COMPONENT_PHASE:
        raise ValueError("Stereo component phase was committed inconsistently")
    updated = list(component_phases)
    updated[component_idx] = phase
    return tuple(updated)


def _with_component_begin_atom(
    component_begin_atoms: tuple[int, ...],
    *,
    component_idx: int,
    atom_idx: int,
) -> tuple[int, ...]:
    existing = component_begin_atoms[component_idx]
    if existing == atom_idx:
        return component_begin_atoms
    if existing != -1:
        raise ValueError("Stereo component begin atom was committed inconsistently")
    updated = list(component_begin_atoms)
    updated[component_idx] = atom_idx
    return tuple(updated)


def _component_phases_after_edge(
    prepared: PreparedSmilesGraph,
    stereo_component_ids: tuple[int, ...],
    component_phases: tuple[int, ...],
    component_begin_atoms: tuple[int, ...],
    *,
    begin_idx: int,
    end_idx: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    bond_idx = prepared.bond_index(begin_idx, end_idx)
    component_idx = stereo_component_ids[bond_idx]
    if component_idx < 0:
        return component_phases, component_begin_atoms
    if not _is_stereo_double_bond(prepared, bond_idx):
        return component_phases, component_begin_atoms
    if component_phases[component_idx] != _UNKNOWN_COMPONENT_PHASE:
        return component_phases, component_begin_atoms

    stored_begin_idx = prepared.bond_begin_atom_indices[bond_idx]
    stored_end_idx = prepared.bond_end_atom_indices[bond_idx]
    stereo_kind = prepared.bond_stereo_kinds[bond_idx]
    if (begin_idx, end_idx) == (stored_begin_idx, stored_end_idx):
        phase = _STORED_COMPONENT_PHASE
    elif stereo_kind in _CIS_STEREO_BOND_KINDS:
        phase = _STORED_COMPONENT_PHASE
    elif stereo_kind in _TRANS_STEREO_BOND_KINDS:
        phase = _FLIPPED_COMPONENT_PHASE
    else:
        raise NotImplementedError(f"Unsupported stereo bond kind: {stereo_kind}")
    return (
        _with_component_phase(
            component_phases,
            component_idx=component_idx,
            phase=phase,
        ),
        _with_component_begin_atom(
            component_begin_atoms,
            component_idx=component_idx,
            atom_idx=begin_idx,
        ),
    )


def _stereo_side_infos(
    prepared: PreparedSmilesGraph,
    stereo_component_ids: tuple[int, ...],
) -> tuple[tuple[StereoSideInfo, ...], dict[tuple[int, int], tuple[int, ...]]]:
    side_candidates: list[tuple[int, int, int, tuple[int, ...]]] = []
    oriented_nodes: set[tuple[int, int]] = set()
    parity_edges: defaultdict[tuple[int, int], list[tuple[tuple[int, int], bool]]] = defaultdict(list)
    seed_tokens: dict[tuple[int, int], str] = {}

    for bond_idx in range(prepared.bond_count):
        component_idx = stereo_component_ids[bond_idx]
        if component_idx < 0 or not _is_stereo_double_bond(prepared, bond_idx):
            continue

        begin_idx = prepared.bond_begin_atom_indices[bond_idx]
        end_idx = prepared.bond_end_atom_indices[bond_idx]
        for endpoint_idx, other_idx in ((begin_idx, end_idx), (end_idx, begin_idx)):
            candidate_neighbors = tuple(
                neighbor_idx
                for neighbor_idx in prepared.neighbors_of(endpoint_idx)
                if neighbor_idx != other_idx
                and prepared.bond_kinds[prepared.bond_index(endpoint_idx, neighbor_idx)] == "SINGLE"
            )
            if not candidate_neighbors:
                continue
            if len(candidate_neighbors) > 2:
                raise NotImplementedError(
                    "Unsupported stereo endpoint with more than two eligible carrier edges"
                )

            side_candidates.append((component_idx, endpoint_idx, other_idx, candidate_neighbors))
            oriented_for_side = [(endpoint_idx, neighbor_idx) for neighbor_idx in candidate_neighbors]
            oriented_nodes.update(oriented_for_side)

            if len(oriented_for_side) == 2:
                left_node, right_node = oriented_for_side
                parity_edges[left_node].append((right_node, True))
                parity_edges[right_node].append((left_node, True))

            for node in oriented_for_side:
                reverse_node = (node[1], node[0])
                if reverse_node in oriented_nodes:
                    parity_edges[node].append((reverse_node, True))
                    parity_edges[reverse_node].append((node, True))

                stored_token = prepared.directed_bond_token(node[0], node[1])
                if stored_token in {"/", "\\"}:
                    existing = seed_tokens.get(node)
                    if existing is not None and existing != stored_token:
                        raise ValueError("Inconsistent stored directional token assignment")
                    seed_tokens[node] = stored_token

    for _, endpoint_idx, _, candidate_neighbors in side_candidates:
        if len(candidate_neighbors) != 2:
            continue
        known_neighbors = [
            neighbor_idx
            for neighbor_idx in candidate_neighbors
            if (endpoint_idx, neighbor_idx) in seed_tokens
        ]
        if len(known_neighbors) == 1:
            known_neighbor = known_neighbors[0]
            other_neighbor = candidate_neighbors[0] if candidate_neighbors[1] == known_neighbor else candidate_neighbors[1]
            seed_tokens[(endpoint_idx, other_neighbor)] = _flip_direction_token(
                seed_tokens[(endpoint_idx, known_neighbor)]
            )

    assignments: dict[tuple[int, int], str] = {}
    queue = deque(seed_tokens)
    while queue:
        node = queue.popleft()
        token = seed_tokens[node]
        assigned = assignments.get(node)
        if assigned is not None:
            if assigned != token:
                raise ValueError("Conflicting stereo carrier token constraints")
            continue
        assignments[node] = token
        for other_node, flipped in parity_edges.get(node, ()):
            other_token = _flip_direction_token(token) if flipped else token
            existing = seed_tokens.get(other_node)
            if existing is not None and existing != other_token:
                raise ValueError("Conflicting stereo carrier token propagation")
            if existing is None:
                seed_tokens[other_node] = other_token
                queue.append(other_node)

    for node in sorted(oriented_nodes):
        if node in assignments:
            continue
        seed_tokens[node] = "/"
        queue.append(node)
        while queue:
            current_node = queue.popleft()
            current_token = seed_tokens[current_node]
            assigned = assignments.get(current_node)
            if assigned is not None:
                if assigned != current_token:
                    raise ValueError("Conflicting stereo carrier token constraints")
                continue
            assignments[current_node] = current_token
            for other_node, flipped in parity_edges.get(current_node, ()):
                other_token = _flip_direction_token(current_token) if flipped else current_token
                existing = seed_tokens.get(other_node)
                if existing is not None and existing != other_token:
                    raise ValueError("Conflicting stereo carrier token propagation")
                if existing is None:
                    seed_tokens[other_node] = other_token
                    queue.append(other_node)

    side_infos = tuple(
        StereoSideInfo(
            component_idx=component_idx,
            endpoint_atom_idx=endpoint_idx,
            other_endpoint_atom_idx=other_idx,
            candidate_neighbors=candidate_neighbors,
            candidate_base_tokens=tuple(
                assignments[(endpoint_idx, neighbor_idx)]
                for neighbor_idx in candidate_neighbors
            ),
        )
        for component_idx, endpoint_idx, other_idx, candidate_neighbors in side_candidates
    )

    edge_to_side_ids: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    for side_idx, side_info in enumerate(side_infos):
        for neighbor_idx in side_info.candidate_neighbors:
            edge_to_side_ids[_canonical_edge(side_info.endpoint_atom_idx, neighbor_idx)].append(side_idx)

    return side_infos, {
        edge: tuple(side_ids)
        for edge, side_ids in edge_to_side_ids.items()
    }


def _candidate_base_token(side_info: StereoSideInfo, neighbor_idx: int) -> str:
    for offset, candidate_neighbor in enumerate(side_info.candidate_neighbors):
        if candidate_neighbor == neighbor_idx:
            return side_info.candidate_base_tokens[offset]
    raise KeyError(
        f"Neighbor {neighbor_idx} is not a stereo carrier candidate for endpoint {side_info.endpoint_atom_idx}"
    )


def _emitted_candidate_token(
    side_info: StereoSideInfo,
    *,
    begin_idx: int,
    end_idx: int,
) -> str:
    if begin_idx == side_info.endpoint_atom_idx:
        return _candidate_base_token(side_info, end_idx)
    if end_idx == side_info.endpoint_atom_idx:
        return _flip_direction_token(_candidate_base_token(side_info, begin_idx))
    raise KeyError("Emitted edge does not match the stereo side")


def _emitted_edge_part_generic(
    prepared: PreparedSmilesGraph,
    side_infos: tuple[StereoSideInfo, ...],
    edge_to_side_ids: dict[tuple[int, int], tuple[int, ...]],
    component_phases: tuple[int, ...],
    selected_neighbors: tuple[int, ...],
    selected_orientations: tuple[int, ...],
    *,
    begin_idx: int,
    end_idx: int,
) -> tuple[str | DeferredDirectionalToken, tuple[int, ...], tuple[int, ...]]:
    side_ids = edge_to_side_ids.get(_canonical_edge(begin_idx, end_idx), ())
    if not side_ids:
        return prepared.bond_token(begin_idx, end_idx), selected_neighbors, selected_orientations

    updated_neighbors = list(selected_neighbors)
    updated_orientations = list(selected_orientations)
    stored_tokens: list[tuple[int, str]] = []

    for side_idx in side_ids:
        side_info = side_infos[side_idx]
        if begin_idx == side_info.endpoint_atom_idx:
            neighbor_idx = end_idx
            edge_orientation = _AFTER_ATOM_EDGE_ORIENTATION
        elif end_idx == side_info.endpoint_atom_idx:
            neighbor_idx = begin_idx
            edge_orientation = _BEFORE_ATOM_EDGE_ORIENTATION
        else:
            continue

        selected_neighbor = updated_neighbors[side_idx]
        if selected_neighbor < 0:
            updated_neighbors[side_idx] = neighbor_idx
            updated_orientations[side_idx] = edge_orientation
            selected_neighbor = neighbor_idx
        if selected_neighbor != neighbor_idx:
            continue

        stored_tokens.append(
            (
                side_info.component_idx,
                _emitted_candidate_token(side_info, begin_idx=begin_idx, end_idx=end_idx),
            )
        )

    updated_state = tuple(updated_neighbors)
    updated_orientation_state = tuple(updated_orientations)
    if not stored_tokens:
        return prepared.bond_token(begin_idx, end_idx), updated_state, updated_orientation_state

    component_idx = stored_tokens[0][0]
    stored_token = stored_tokens[0][1]
    for other_component_idx, other_stored_token in stored_tokens[1:]:
        if other_component_idx != component_idx:
            raise ValueError("Carrier edge unexpectedly spans multiple stereo components")
        if other_stored_token != stored_token:
            raise ValueError("Carrier edge received conflicting stereo token assignments")

    phase = component_phases[component_idx]
    if phase == _UNKNOWN_COMPONENT_PHASE:
        return (
            DeferredDirectionalToken(component_idx=component_idx, stored_token=stored_token),
            updated_state,
            updated_orientation_state,
        )
    if phase == _STORED_COMPONENT_PHASE:
        return stored_token, updated_state, updated_orientation_state
    return _flip_direction_token(stored_token), updated_state, updated_orientation_state


def _emitted_isolated_edge_part(
    prepared: PreparedSmilesGraph,
    side_infos: tuple[StereoSideInfo, ...],
    edge_to_side_ids: dict[tuple[int, int], tuple[int, ...]],
    component_phases: tuple[int, ...],
    selected_neighbors: tuple[int, ...],
    selected_orientations: tuple[int, ...],
    *,
    begin_idx: int,
    end_idx: int,
) -> tuple[str | DeferredDirectionalToken, tuple[int, ...], tuple[int, ...]]:
    side_ids = edge_to_side_ids.get(_canonical_edge(begin_idx, end_idx), ())
    if not side_ids:
        return prepared.bond_token(begin_idx, end_idx), selected_neighbors, selected_orientations

    updated_neighbors = list(selected_neighbors)
    updated_orientations = list(selected_orientations)
    stored_tokens: list[tuple[int, str]] = []

    for side_idx in side_ids:
        side_info = side_infos[side_idx]
        if begin_idx == side_info.endpoint_atom_idx:
            neighbor_idx = end_idx
            edge_orientation = _AFTER_ATOM_EDGE_ORIENTATION
        elif end_idx == side_info.endpoint_atom_idx:
            neighbor_idx = begin_idx
            edge_orientation = _BEFORE_ATOM_EDGE_ORIENTATION
        else:
            continue

        selected_neighbor = updated_neighbors[side_idx]
        if selected_neighbor < 0:
            updated_neighbors[side_idx] = neighbor_idx
            updated_orientations[side_idx] = edge_orientation
            selected_neighbor = neighbor_idx
        if selected_neighbor != neighbor_idx:
            continue

        stored_tokens.append(
            (
                side_info.component_idx,
                _emitted_candidate_token(side_info, begin_idx=begin_idx, end_idx=end_idx),
            )
        )

    updated_state = tuple(updated_neighbors)
    updated_orientation_state = tuple(updated_orientations)
    if not stored_tokens:
        return prepared.bond_token(begin_idx, end_idx), updated_state, updated_orientation_state

    component_idx = stored_tokens[0][0]
    stored_token = stored_tokens[0][1]
    for other_component_idx, other_stored_token in stored_tokens[1:]:
        if other_component_idx != component_idx:
            raise ValueError("Carrier edge unexpectedly spans multiple stereo components")
        if other_stored_token != stored_token:
            raise ValueError("Carrier edge received conflicting stereo token assignments")

    return (
        DeferredDirectionalToken(component_idx=component_idx, stored_token=stored_token),
        updated_state,
        updated_orientation_state,
    )


def _emitted_coupled_edge_part(
    prepared: PreparedSmilesGraph,
    side_infos: tuple[StereoSideInfo, ...],
    edge_to_side_ids: dict[tuple[int, int], tuple[int, ...]],
    component_phases: tuple[int, ...],
    selected_neighbors: tuple[int, ...],
    selected_orientations: tuple[int, ...],
    *,
    begin_idx: int,
    end_idx: int,
) -> tuple[str | DeferredDirectionalToken, tuple[int, ...], tuple[int, ...]]:
    return _emitted_edge_part_generic(
        prepared,
        side_infos,
        edge_to_side_ids,
        component_phases,
        selected_neighbors,
        selected_orientations,
        begin_idx=begin_idx,
        end_idx=end_idx,
    )


def _emitted_edge_part(
    prepared: PreparedSmilesGraph,
    side_infos: tuple[StereoSideInfo, ...],
    edge_to_side_ids: dict[tuple[int, int], tuple[int, ...]],
    component_phases: tuple[int, ...],
    selected_neighbors: tuple[int, ...],
    selected_orientations: tuple[int, ...],
    isolated_components: tuple[bool, ...],
    *,
    begin_idx: int,
    end_idx: int,
) -> tuple[str | DeferredDirectionalToken, tuple[int, ...], tuple[int, ...]]:
    edge = _canonical_edge(begin_idx, end_idx)
    side_ids = edge_to_side_ids.get(edge, ())
    if not side_ids:
        return prepared.bond_token(begin_idx, end_idx), selected_neighbors, selected_orientations

    uses_isolated_component = any(
        isolated_components[side_infos[side_idx].component_idx]
        for side_idx in side_ids
        if side_infos[side_idx].component_idx >= 0
    )
    if uses_isolated_component:
        return _emitted_isolated_edge_part(
            prepared,
            side_infos,
            edge_to_side_ids,
            component_phases,
            selected_neighbors,
            selected_orientations,
            begin_idx=begin_idx,
            end_idx=end_idx,
        )
    return _emitted_coupled_edge_part(
        prepared,
        side_infos,
        edge_to_side_ids,
        component_phases,
        selected_neighbors,
        selected_orientations,
        begin_idx=begin_idx,
        end_idx=end_idx,
    )


def _isolated_component_flips(
    prepared: PreparedSmilesGraph,
    stereo_component_ids: tuple[int, ...],
    side_infos: tuple[StereoSideInfo, ...],
    isolated_components: tuple[bool, ...],
    result: SearchResult,
) -> tuple[bool, ...]:
    if not isolated_components:
        return ()

    side_ids_by_component: defaultdict[int, list[int]] = defaultdict(list)
    for side_idx, side_info in enumerate(side_infos):
        side_ids_by_component[side_info.component_idx].append(side_idx)

    flips = [False for _ in range(len(isolated_components))]
    for component_idx, isolated in enumerate(isolated_components):
        if not isolated:
            continue

        side_ids = side_ids_by_component.get(component_idx, [])
        if not side_ids:
            continue
        if all(len(side_infos[side_idx].candidate_neighbors) == 1 for side_idx in side_ids):
            continue

        begin_atom_idx = result.stereo_component_begin_atoms[component_idx]
        if begin_atom_idx < 0:
            continue

        begin_side_idx = next(
            (
                side_idx
                for side_idx in side_ids
                if side_infos[side_idx].endpoint_atom_idx == begin_atom_idx
            ),
            None,
        )
        if begin_side_idx is None:
            continue

        selected_neighbor_idx = result.stereo_selected_neighbors[begin_side_idx]
        if selected_neighbor_idx < 0:
            continue

        side_info = side_infos[begin_side_idx]
        selected_base_token = _candidate_base_token(side_info, selected_neighbor_idx)
        phase = result.stereo_component_phases[component_idx]
        if phase == _UNKNOWN_COMPONENT_PHASE:
            continue
        flips[component_idx] = (
            selected_base_token
            == ("/" if phase == _STORED_COMPONENT_PHASE else "\\")
        )

    return tuple(flips)


def enumerate_from_atom(
    prepared: PreparedSmilesGraph,
    stereo_component_ids: tuple[int, ...],
    side_infos: tuple[StereoSideInfo, ...],
    edge_to_side_ids: dict[tuple[int, int], tuple[int, ...]],
    isolated_components: tuple[bool, ...],
    atom_idx: int,
    *,
    parent_idx: int | None,
    visited: frozenset[int],
    pending_state: tuple[tuple[int, tuple[PendingRing, ...]], ...],
    free_labels: tuple[int, ...],
    next_label: int,
    component_phases: tuple[int, ...],
    selected_neighbors: tuple[int, ...],
    selected_orientations: tuple[int, ...],
    component_begin_atoms: tuple[int, ...],
) -> Iterator[SearchResult]:
    visited_now = visited | {atom_idx}

    pending = tuple_to_pending(pending_state)
    closures_here = pending.pop(atom_idx, ())
    ordered_groups = list(ordered_neighbor_groups(prepared, atom_idx, visited_now))
    child_choice_space = [group for group in ordered_groups]

    for chosen_children in product(*child_choice_space) if child_choice_space else [()]:
        child_order_seed = tuple(int(child_idx) for child_idx in chosen_children)
        child_set = set(child_order_seed)
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
            current_component_phases = component_phases
            current_selected_neighbors = selected_neighbors
            current_selected_orientations = selected_orientations
            current_component_begin_atoms = component_begin_atoms
            current_ring_parts: list[str | DeferredDirectionalToken] = []
            labels_freed_after_atom: list[int] = []
            ring_neighbor_order: list[int] = []

            for action_kind, payload in ring_action_order:
                if action_kind == "close":
                    closure = closures_here[payload]
                    bond_part, current_selected_neighbors, current_selected_orientations = _emitted_edge_part(
                        prepared,
                        side_infos,
                        edge_to_side_ids,
                        current_component_phases,
                        current_selected_neighbors,
                        current_selected_orientations,
                        isolated_components,
                        begin_idx=atom_idx,
                        end_idx=closure.other_atom_idx,
                    )
                    current_ring_parts.extend(_part_tuple(bond_part))
                    current_ring_parts.append(ring_label_text(closure.label))
                    labels_freed_after_atom.append(closure.label)
                    ring_neighbor_order.append(closure.other_atom_idx)
                    continue

                target_idx = int(payload)
                label, current_free, current_next_label = allocate_label(
                    current_free,
                    current_next_label,
                )
                current_ring_parts.append(ring_label_text(label))
                current_component_phases, current_component_begin_atoms = _component_phases_after_edge(
                    prepared,
                    stereo_component_ids,
                    current_component_phases,
                    current_component_begin_atoms,
                    begin_idx=atom_idx,
                    end_idx=target_idx,
                )
                current_pending = add_pending(
                    current_pending,
                    target_idx,
                    PendingRing(
                        label=label,
                        other_atom_idx=atom_idx,
                    ),
                )
                ring_neighbor_order.append(target_idx)

            for label in labels_freed_after_atom:
                current_free = free_label(current_free, label)

            child_orders = unique_permutations(chosen_children)
            for child_order in child_orders:
                child_order_tuple = tuple(int(child_idx) for child_idx in child_order)
                if prepared.atom_chiral_tags[atom_idx] == "CHI_UNSPECIFIED":
                    atom_token = prepared.atom_tokens[atom_idx]
                else:
                    atom_token = _stereo_atom_token(
                        prepared,
                        atom_idx,
                        emitted_neighbor_order=_stereo_neighbor_order(
                            prepared,
                            atom_idx,
                            parent_idx=parent_idx,
                            ring_neighbor_order=tuple(ring_neighbor_order),
                            child_order=child_order_tuple,
                        ),
                    )
                yield from expand_children(
                    prepared=prepared,
                    stereo_component_ids=stereo_component_ids,
                    side_infos=side_infos,
                    edge_to_side_ids=edge_to_side_ids,
                    isolated_components=isolated_components,
                    parent_idx=atom_idx,
                    child_order=child_order_tuple,
                    prefix_parts=(atom_token, *current_ring_parts),
                    visited=visited_now,
                    pending_state=pending_to_tuple(current_pending),
                    free_labels=current_free,
                    next_label=current_next_label,
                    component_phases=current_component_phases,
                    selected_neighbors=current_selected_neighbors,
                    selected_orientations=current_selected_orientations,
                    component_begin_atoms=current_component_begin_atoms,
                )


def expand_children(
    prepared: PreparedSmilesGraph,
    stereo_component_ids: tuple[int, ...],
    side_infos: tuple[StereoSideInfo, ...],
    edge_to_side_ids: dict[tuple[int, int], tuple[int, ...]],
    isolated_components: tuple[bool, ...],
    *,
    parent_idx: int,
    child_order: tuple[int, ...],
    prefix_parts: tuple[str | DeferredDirectionalToken, ...],
    visited: frozenset[int],
    pending_state: tuple[tuple[int, tuple[PendingRing, ...]], ...],
    free_labels: tuple[int, ...],
    next_label: int,
    component_phases: tuple[int, ...],
    selected_neighbors: tuple[int, ...],
    selected_orientations: tuple[int, ...],
    component_begin_atoms: tuple[int, ...],
) -> Iterator[SearchResult]:
    if not child_order:
        yield SearchResult(
            parts=prefix_parts,
            visited=visited,
            pending=pending_state,
            free_labels=free_labels,
            next_label=next_label,
            stereo_component_phases=component_phases,
            stereo_selected_neighbors=selected_neighbors,
            stereo_selected_orientations=selected_orientations,
            stereo_component_begin_atoms=component_begin_atoms,
        )
        return

    branch_children = child_order[:-1]
    main_child = int(child_order[-1])
    def recurse_branch_children(
        branch_index: int,
        partial: SearchResult,
    ) -> Iterator[SearchResult]:
        if branch_index == len(branch_children):
            edge_part, main_selected_neighbors, main_selected_orientations = _emitted_edge_part(
                prepared,
                side_infos,
                edge_to_side_ids,
                partial.stereo_component_phases,
                partial.stereo_selected_neighbors,
                partial.stereo_selected_orientations,
                isolated_components,
                begin_idx=parent_idx,
                end_idx=main_child,
            )
            main_component_phases, main_component_begin_atoms = _component_phases_after_edge(
                prepared,
                stereo_component_ids,
                partial.stereo_component_phases,
                partial.stereo_component_begin_atoms,
                begin_idx=parent_idx,
                end_idx=main_child,
            )
            for main_result in enumerate_from_atom(
                prepared=prepared,
                stereo_component_ids=stereo_component_ids,
                side_infos=side_infos,
                edge_to_side_ids=edge_to_side_ids,
                isolated_components=isolated_components,
                atom_idx=main_child,
                parent_idx=parent_idx,
                visited=partial.visited,
                pending_state=partial.pending,
                free_labels=partial.free_labels,
                next_label=partial.next_label,
                component_phases=main_component_phases,
                selected_neighbors=main_selected_neighbors,
                selected_orientations=main_selected_orientations,
                component_begin_atoms=main_component_begin_atoms,
            ):
                yield SearchResult(
                    parts=partial.parts + _part_tuple(edge_part) + main_result.parts,
                    visited=main_result.visited,
                    pending=main_result.pending,
                    free_labels=main_result.free_labels,
                    next_label=main_result.next_label,
                    stereo_component_phases=main_result.stereo_component_phases,
                    stereo_selected_neighbors=main_result.stereo_selected_neighbors,
                    stereo_selected_orientations=main_result.stereo_selected_orientations,
                    stereo_component_begin_atoms=main_result.stereo_component_begin_atoms,
                )
            return

        child_idx = int(branch_children[branch_index])
        branch_part, branch_selected_neighbors, branch_selected_orientations = _emitted_edge_part(
            prepared,
            side_infos,
            edge_to_side_ids,
            partial.stereo_component_phases,
            partial.stereo_selected_neighbors,
            partial.stereo_selected_orientations,
            isolated_components,
            begin_idx=parent_idx,
            end_idx=child_idx,
        )
        child_component_phases, child_component_begin_atoms = _component_phases_after_edge(
            prepared,
            stereo_component_ids,
            partial.stereo_component_phases,
            partial.stereo_component_begin_atoms,
            begin_idx=parent_idx,
            end_idx=child_idx,
        )
        for branch_result in enumerate_from_atom(
            prepared=prepared,
            stereo_component_ids=stereo_component_ids,
            side_infos=side_infos,
            edge_to_side_ids=edge_to_side_ids,
            isolated_components=isolated_components,
            atom_idx=child_idx,
            parent_idx=parent_idx,
            visited=partial.visited,
            pending_state=partial.pending,
            free_labels=partial.free_labels,
            next_label=partial.next_label,
            component_phases=child_component_phases,
            selected_neighbors=branch_selected_neighbors,
            selected_orientations=branch_selected_orientations,
            component_begin_atoms=child_component_begin_atoms,
        ):
            yield from recurse_branch_children(
                branch_index + 1,
                SearchResult(
                    parts=(
                        partial.parts
                        + ("(",)
                        + _part_tuple(branch_part)
                        + branch_result.parts
                        + (")",)
                    ),
                    visited=branch_result.visited,
                    pending=branch_result.pending,
                    free_labels=branch_result.free_labels,
                    next_label=branch_result.next_label,
                    stereo_component_phases=branch_result.stereo_component_phases,
                    stereo_selected_neighbors=branch_result.stereo_selected_neighbors,
                    stereo_selected_orientations=branch_result.stereo_selected_orientations,
                    stereo_component_begin_atoms=branch_result.stereo_component_begin_atoms,
                ),
            )

    yield from recurse_branch_children(
        0,
        SearchResult(
            parts=prefix_parts,
            visited=visited,
            pending=pending_state,
            free_labels=free_labels,
            next_label=next_label,
            stereo_component_phases=component_phases,
            stereo_selected_neighbors=selected_neighbors,
            stereo_selected_orientations=selected_orientations,
            stereo_component_begin_atoms=component_begin_atoms,
        ),
    )


def _coerce_prepared_graph(
    mol_or_prepared: Chem.Mol | PreparedSmilesGraph,
    policy: ReferencePolicy | None,
) -> PreparedSmilesGraph:
    if isinstance(mol_or_prepared, PreparedSmilesGraph):
        if policy is not None:
            mol_or_prepared.validate_policy(policy)
        prepared = mol_or_prepared
    elif hasattr(mol_or_prepared, "to_dict"):
        prepared = PreparedSmilesGraph.from_dict(mol_or_prepared.to_dict())
        if policy is not None:
            prepared.validate_policy(policy)
    else:
        if policy is None:
            raise TypeError("policy is required when preparing a graph from an RDKit molecule")
        prepared = prepare_smiles_graph(
            mol_or_prepared,
            policy,
            surface_kind=CONNECTED_STEREO_SURFACE,
        )
    _check_supported_stereo_writer_surface(prepared)
    return prepared


def enumerate_rooted_connected_stereo_smiles_support(
    mol: Chem.Mol | PreparedSmilesGraph,
    root_idx: int,
    policy: ReferencePolicy | None = None,
) -> set[str]:
    prepared = _coerce_prepared_graph(mol, policy)
    if prepared.atom_count == 0:
        return {""}
    if root_idx < 0 or root_idx >= prepared.atom_count:
        raise IndexError("root_idx out of range")

    stereo_component_ids = _stereo_component_ids(prepared)
    component_count = max(stereo_component_ids, default=-1) + 1
    isolated_components = tuple(size == 1 for size in _component_sizes(stereo_component_ids))
    initial_component_phases = tuple(_UNKNOWN_COMPONENT_PHASE for _ in range(component_count))
    side_infos, edge_to_side_ids = _stereo_side_infos(prepared, stereo_component_ids)
    initial_selected_neighbors = tuple(-1 for _ in range(len(side_infos)))
    initial_selected_orientations = tuple(_UNKNOWN_EDGE_ORIENTATION for _ in range(len(side_infos)))
    initial_component_begin_atoms = tuple(-1 for _ in range(component_count))

    results: set[str] = set()
    for result in enumerate_from_atom(
        prepared=prepared,
        stereo_component_ids=stereo_component_ids,
        side_infos=side_infos,
        edge_to_side_ids=edge_to_side_ids,
        isolated_components=isolated_components,
        atom_idx=root_idx,
        parent_idx=None,
        visited=frozenset(),
        pending_state=(),
        free_labels=(),
        next_label=1,
        component_phases=initial_component_phases,
        selected_neighbors=initial_selected_neighbors,
        selected_orientations=initial_selected_orientations,
        component_begin_atoms=initial_component_begin_atoms,
    ):
        if len(result.visited) != prepared.atom_count:
            continue
        if result.pending:
            continue
        results.add(
            _resolve_parts(
                result.parts,
                result.stereo_component_phases,
                component_flips=_isolated_component_flips(
                    prepared,
                    stereo_component_ids,
                    side_infos,
                    isolated_components,
                    result,
                ),
            )
        )
    return results


def validate_rooted_connected_stereo_smiles_support(
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
        else enumerate_rooted_connected_stereo_smiles_support(prepared, root_idx)
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


__all__ = [
    "enumerate_rooted_connected_stereo_smiles_support",
    "validate_rooted_connected_stereo_smiles_support",
]
