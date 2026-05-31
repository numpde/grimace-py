"""Residual graph-obligation classification for writer-shaped states."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .ids import AtomId
from .ids import BondId
from .writer_state import WriterStateKey

if TYPE_CHECKING:
    from .prepared_runtime import SouthStarPreparedMol


class WriterBoundaryOwnerKind(Enum):
    ACTIVE_ATOM = "active_atom"
    BRANCH_RETURN = "branch_return"
    PENDING_PARENT = "pending_parent"
    OPEN_RING_ENDPOINT = "open_ring_endpoint"
    UNOWNED = "unowned"


class WriterEdgeObligationKind(Enum):
    TREE_ENTRY = "tree_entry"
    PENDING_ENTRY = "pending_entry"
    LATENT_RESIDUAL = "latent_residual"
    BOUNDARY_INCIDENCE = "boundary_incidence"
    CLOSURE_CANDIDATE = "closure_candidate"
    OPEN_CLOSURE_ENDPOINT = "open_closure_endpoint"
    CLOSED_CLOSURE = "closed_closure"


@dataclass(frozen=True, slots=True)
class WriterEdgeObligation:
    bond: BondId
    kind: WriterEdgeObligationKind
    a: AtomId
    b: AtomId


@dataclass(frozen=True, slots=True)
class WriterEdgeObligationPartition:
    obligations: tuple[WriterEdgeObligation, ...]


@dataclass(frozen=True, slots=True)
class WriterBoundaryIncidence:
    bond: BondId
    written_atom: AtomId
    residual_atom: AtomId
    owner_kind: WriterBoundaryOwnerKind


@dataclass(frozen=True, slots=True)
class WriterResidualAttachment:
    attachment_id: int
    atoms: frozenset[AtomId]
    latent_bonds: frozenset[BondId]
    boundary: tuple[WriterBoundaryIncidence, ...]
    cyclic_rank: int
    block_ids: frozenset[int]


@dataclass(frozen=True, slots=True)
class WriterResidualAttachmentState:
    attachments: tuple[WriterResidualAttachment, ...]


@dataclass(frozen=True, slots=True)
class WriterGraphObligationSummary:
    attachments: WriterResidualAttachmentState
    boundary_by_owner_atom: tuple[tuple[AtomId, tuple[int, ...]], ...]
    boundary_by_pending_parent: tuple[tuple[AtomId, tuple[int, ...]], ...]
    has_cyclic_attachment: bool


@dataclass(frozen=True, slots=True)
class WriterBlockCutMetadata:
    bridge_bonds: frozenset[BondId]
    biconnected_block_by_bond: tuple[tuple[BondId, int], ...]
    cyclic_blocks: frozenset[int]


@dataclass(frozen=True, slots=True)
class WriterComponentGraphSurface:
    component_index: int
    atoms: frozenset[AtomId]
    bonds: frozenset[BondId]
    connected: bool
    tree: bool
    cyclic_rank: int
    cyclic_block_ids: frozenset[int]
    unsupported_reason: str | None


@dataclass(frozen=True, slots=True)
class WriterGraphPreparedMetadata:
    block_cut: WriterBlockCutMetadata
    component_surfaces: tuple[WriterComponentGraphSurface, ...]


@dataclass(frozen=True, slots=True)
class WriterGraphObligationContext:
    edge_partition: WriterEdgeObligationPartition
    residual_summary: WriterGraphObligationSummary
    prepared_metadata: WriterGraphPreparedMetadata


def build_writer_block_cut_metadata(
    prepared: SouthStarPreparedMol,
) -> WriterBlockCutMetadata:
    return _build_writer_block_cut_metadata_from_graph(prepared.atom_ids, prepared.graph_index)


def build_writer_graph_prepared_metadata(
    prepared: SouthStarPreparedMol,
) -> WriterGraphPreparedMetadata:
    return build_writer_graph_prepared_metadata_from_facts(
        prepared.facts,
        prepared.graph_index,
        prepared.atom_ids,
    )


def build_writer_graph_prepared_metadata_from_facts(
    facts,
    graph_index,
    atom_ids: tuple[AtomId, ...],
) -> WriterGraphPreparedMetadata:
    block_cut = _build_writer_block_cut_metadata_from_graph(atom_ids, graph_index)
    return WriterGraphPreparedMetadata(
        block_cut=block_cut,
        component_surfaces=_component_graph_surfaces(facts, graph_index, block_cut),
    )


def _build_writer_block_cut_metadata_from_graph(
    atom_ids: tuple[AtomId, ...],
    graph,
) -> WriterBlockCutMetadata:
    time = 0
    seen: set[AtomId] = set()
    discovery: dict[AtomId, int] = {}
    low: dict[AtomId, int] = {}
    edge_stack: list[BondId] = []
    blocks: list[tuple[BondId, ...]] = []

    def dfs(atom: AtomId, parent_bond: BondId | None) -> None:
        nonlocal time
        seen.add(atom)
        discovery[atom] = time
        low[atom] = time
        time += 1
        for neighbor in graph.neighbors[atom]:
            bond = graph.bond_between[(min(atom, neighbor), max(atom, neighbor))]
            if bond == parent_bond:
                continue
            if neighbor not in seen:
                edge_stack.append(bond)
                dfs(neighbor, bond)
                low[atom] = min(low[atom], low[neighbor])
                if low[neighbor] >= discovery[atom]:
                    block_edges = []
                    while edge_stack:
                        popped = edge_stack.pop()
                        block_edges.append(popped)
                        if popped == bond:
                            break
                    blocks.append(tuple(block_edges))
            elif discovery[neighbor] < discovery[atom]:
                edge_stack.append(bond)
                low[atom] = min(low[atom], discovery[neighbor])

    for atom in atom_ids:
        if atom in seen:
            continue
        dfs(atom, None)
        if edge_stack:
            blocks.append(tuple(edge_stack))
            edge_stack.clear()

    bridge_bonds: set[BondId] = set()
    cyclic_blocks: set[int] = set()
    block_by_bond: list[tuple[BondId, int]] = []
    for block_id, block_edges in enumerate(blocks):
        unique_edges = frozenset(block_edges)
        if len(unique_edges) == 1:
            bridge_bonds.update(unique_edges)
        else:
            cyclic_blocks.add(block_id)
        for bond in unique_edges:
            block_by_bond.append((bond, block_id))

    return WriterBlockCutMetadata(
        bridge_bonds=frozenset(bridge_bonds),
        biconnected_block_by_bond=tuple(
            sorted(block_by_bond, key=lambda item: int(item[0]))
        ),
        cyclic_blocks=frozenset(cyclic_blocks),
    )


def build_writer_graph_obligation_context(
    prepared: SouthStarPreparedMol,
    key: WriterStateKey,
) -> WriterGraphObligationContext:
    _current_component(prepared, key)
    partition = classify_writer_edge_obligations(prepared, key)
    validate_writer_edge_obligation_partition(prepared, key, partition)
    metadata = prepared.writer_graph_metadata
    summary = classify_writer_residual_attachments(
        prepared,
        key,
        metadata.block_cut,
        partition=partition,
    )
    return WriterGraphObligationContext(
        edge_partition=partition,
        residual_summary=summary,
        prepared_metadata=metadata,
    )


def validate_writer_supported_graph_surface(
    prepared: SouthStarPreparedMol,
) -> None:
    for surface in prepared.writer_graph_metadata.component_surfaces:
        if surface.unsupported_reason is not None:
            raise SouthStarError(
                SouthStarErrorKind.UNSUPPORTED_POLICY,
                surface.unsupported_reason,
            )


def classify_writer_residual_attachments(
    prepared: SouthStarPreparedMol,
    key: WriterStateKey,
    block_cut: WriterBlockCutMetadata,
    *,
    partition: WriterEdgeObligationPartition | None = None,
) -> WriterGraphObligationSummary:
    component = _current_component(prepared, key)
    component_atoms = frozenset(component.atoms)
    visited = frozenset(atom for atom in key.visited_atoms if atom in component_atoms)
    residual_atoms = component_atoms - visited
    block_by_bond = dict(block_cut.biconnected_block_by_bond)
    if partition is None:
        partition = classify_writer_edge_obligations(prepared, key)
        validate_writer_edge_obligation_partition(prepared, key, partition)
    latent_obligations = tuple(
        obligation
        for obligation in partition.obligations
        if obligation.kind is WriterEdgeObligationKind.LATENT_RESIDUAL
    )
    boundary_obligations = tuple(
        obligation
        for obligation in partition.obligations
        if obligation.kind is WriterEdgeObligationKind.BOUNDARY_INCIDENCE
    )
    has_closure_candidate = any(
        obligation.kind is WriterEdgeObligationKind.CLOSURE_CANDIDATE
        for obligation in partition.obligations
    )
    attachments: list[WriterResidualAttachment] = []

    for atoms in _residual_atom_components(prepared, residual_atoms, latent_obligations):
        latent_bonds = frozenset(
            obligation.bond
            for obligation in latent_obligations
            if obligation.a in atoms and obligation.b in atoms
        )
        boundary = tuple(
            sorted(
                _boundary_incidences(prepared, key, atoms, boundary_obligations),
                key=writer_boundary_incidence_sort_tuple,
            )
        )
        block_ids = frozenset(
            block_by_bond[bond]
            for bond in (*latent_bonds, *(incidence.bond for incidence in boundary))
            if bond in block_by_bond
        )
        cyclic_rank = len(latent_bonds) - len(atoms) + 1
        attachments.append(
            WriterResidualAttachment(
                attachment_id=0,
                atoms=frozenset(atoms),
                latent_bonds=latent_bonds,
                boundary=boundary,
                cyclic_rank=cyclic_rank,
                block_ids=block_ids,
            )
        )

    sorted_attachments = []
    for attachment_id, attachment in enumerate(
        sorted(attachments, key=writer_residual_attachment_sort_tuple)
    ):
        sorted_attachments.append(
            WriterResidualAttachment(
                attachment_id=attachment_id,
                atoms=attachment.atoms,
                latent_bonds=attachment.latent_bonds,
                boundary=attachment.boundary,
                cyclic_rank=attachment.cyclic_rank,
                block_ids=attachment.block_ids,
            )
        )

    boundary_by_owner: dict[AtomId, set[int]] = {}
    boundary_by_pending: dict[AtomId, set[int]] = {}
    for attachment in sorted_attachments:
        for incidence in attachment.boundary:
            if incidence.owner_kind in (
                WriterBoundaryOwnerKind.ACTIVE_ATOM,
                WriterBoundaryOwnerKind.BRANCH_RETURN,
            ):
                boundary_by_owner.setdefault(incidence.written_atom, set()).add(
                    attachment.attachment_id
                )
            if incidence.owner_kind is WriterBoundaryOwnerKind.PENDING_PARENT:
                boundary_by_pending.setdefault(incidence.written_atom, set()).add(
                    attachment.attachment_id
                )

    has_cyclic_attachment = any(
        attachment.cyclic_rank > 0
        or len(attachment.boundary) > 1
        or bool(attachment.block_ids & block_cut.cyclic_blocks)
        for attachment in sorted_attachments
    ) or has_closure_candidate
    return WriterGraphObligationSummary(
        attachments=WriterResidualAttachmentState(
            attachments=tuple(sorted_attachments)
        ),
        boundary_by_owner_atom=_canonical_boundary_index(boundary_by_owner),
        boundary_by_pending_parent=_canonical_boundary_index(boundary_by_pending),
        has_cyclic_attachment=has_cyclic_attachment,
    )


def classify_writer_edge_obligations(
    prepared: SouthStarPreparedMol,
    key: WriterStateKey,
) -> WriterEdgeObligationPartition:
    component = _current_component(prepared, key)
    pending = key.obligations.pending_entry
    obligations = []
    for bond in sorted(component.bonds, key=int):
        fact = prepared.graph_index.bond_by_id[bond]
        if bond in key.written_bonds:
            kind = WriterEdgeObligationKind.TREE_ENTRY
        elif pending is not None and pending.bond == bond:
            kind = WriterEdgeObligationKind.PENDING_ENTRY
        else:
            left_visited = fact.a in key.visited_atoms
            right_visited = fact.b in key.visited_atoms
            if left_visited and right_visited:
                kind = WriterEdgeObligationKind.CLOSURE_CANDIDATE
            elif left_visited or right_visited:
                kind = WriterEdgeObligationKind.BOUNDARY_INCIDENCE
            else:
                kind = WriterEdgeObligationKind.LATENT_RESIDUAL
        obligations.append(
            WriterEdgeObligation(
                bond=bond,
                kind=kind,
                a=fact.a,
                b=fact.b,
            )
        )
    return WriterEdgeObligationPartition(obligations=tuple(obligations))


def validate_writer_edge_obligation_partition(
    prepared: SouthStarPreparedMol,
    key: WriterStateKey,
    partition: WriterEdgeObligationPartition,
) -> None:
    component = _current_component(prepared, key)
    component_bonds = frozenset(component.bonds)
    seen_bonds = [obligation.bond for obligation in partition.obligations]
    if len(set(seen_bonds)) != len(seen_bonds):
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "writer edge obligation partition contains duplicate bonds",
        )
    if frozenset(seen_bonds) != component_bonds:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "writer edge obligation partition does not cover current component bonds",
        )
    pending = key.obligations.pending_entry
    if pending is not None and pending.bond in key.written_bonds:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "writer pending bond is already written",
        )
    pending_entries = [
        obligation
        for obligation in partition.obligations
        if obligation.kind is WriterEdgeObligationKind.PENDING_ENTRY
    ]
    if pending is None:
        if pending_entries:
            raise SouthStarError(
                SouthStarErrorKind.INTERNAL_INVARIANT,
                "writer edge partition has pending entry without pending state",
            )
    elif len(pending_entries) != 1 or pending_entries[0].bond != pending.bond:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "writer edge partition does not match pending entry",
        )
    for obligation in partition.obligations:
        fact = prepared.graph_index.bond_by_id[obligation.bond]
        if (obligation.a, obligation.b) != (fact.a, fact.b):
            raise SouthStarError(
                SouthStarErrorKind.INTERNAL_INVARIANT,
                "writer edge obligation endpoints do not match prepared bond",
            )
        left_visited = obligation.a in key.visited_atoms
        right_visited = obligation.b in key.visited_atoms
        if obligation.kind is WriterEdgeObligationKind.TREE_ENTRY:
            if obligation.bond not in key.written_bonds:
                _invalid_edge_partition("tree-entry obligation is not written")
            if not left_visited or not right_visited:
                _invalid_edge_partition("tree-entry obligation has unvisited endpoint")
        elif obligation.kind is WriterEdgeObligationKind.PENDING_ENTRY:
            if pending is None or obligation.bond != pending.bond:
                _invalid_edge_partition("pending obligation does not match pending state")
            _require_pending_orientation(pending, obligation)
            if obligation.bond in key.written_bonds:
                _invalid_edge_partition("pending obligation is already written")
            if pending.parent not in key.visited_atoms or pending.child in key.visited_atoms:
                _invalid_edge_partition("pending obligation has invalid visited endpoints")
        elif obligation.kind is WriterEdgeObligationKind.BOUNDARY_INCIDENCE:
            if obligation.bond in key.written_bonds:
                _invalid_edge_partition("boundary obligation is already written")
            if pending is not None and obligation.bond == pending.bond:
                _invalid_edge_partition("boundary obligation is also pending")
            if left_visited == right_visited:
                _invalid_edge_partition("boundary obligation must have one visited endpoint")
        elif obligation.kind is WriterEdgeObligationKind.LATENT_RESIDUAL:
            if obligation.bond in key.written_bonds:
                _invalid_edge_partition("latent obligation is already written")
            if pending is not None and obligation.bond == pending.bond:
                _invalid_edge_partition("latent obligation is also pending")
            if left_visited or right_visited:
                _invalid_edge_partition("latent obligation has visited endpoint")
        elif obligation.kind is WriterEdgeObligationKind.CLOSURE_CANDIDATE:
            if obligation.bond in key.written_bonds:
                _invalid_edge_partition("closure candidate is already written")
            if pending is not None and obligation.bond == pending.bond:
                _invalid_edge_partition("closure candidate is also pending")
            if not left_visited or not right_visited:
                _invalid_edge_partition("closure candidate must have visited endpoints")
        elif obligation.kind in (
            WriterEdgeObligationKind.OPEN_CLOSURE_ENDPOINT,
            WriterEdgeObligationKind.CLOSED_CLOSURE,
        ):
            raise SouthStarError(
                SouthStarErrorKind.UNSUPPORTED_POLICY,
                "WRITER_SHAPED ring closure edge obligations are not supported yet",
            )
        else:
            _invalid_edge_partition("unknown writer edge obligation kind")


def writer_residual_attachment_sort_tuple(
    attachment: WriterResidualAttachment,
) -> tuple[object, ...]:
    return (
        tuple(sorted(int(atom) for atom in attachment.atoms)),
        tuple(sorted(int(bond) for bond in attachment.latent_bonds)),
        tuple(writer_boundary_incidence_sort_tuple(item) for item in attachment.boundary),
        int(attachment.cyclic_rank),
        tuple(sorted(attachment.block_ids)),
    )


def writer_boundary_incidence_sort_tuple(
    incidence: WriterBoundaryIncidence,
) -> tuple[object, ...]:
    return (
        int(incidence.bond),
        int(incidence.written_atom),
        int(incidence.residual_atom),
        incidence.owner_kind.value,
    )


def writer_edge_obligation_sort_tuple(
    obligation: WriterEdgeObligation,
) -> tuple[object, ...]:
    return (
        int(obligation.bond),
        obligation.kind.value,
        int(obligation.a),
        int(obligation.b),
    )


def writer_edge_obligation_partition_sort_tuple(
    partition: WriterEdgeObligationPartition,
) -> tuple[object, ...]:
    return tuple(
        writer_edge_obligation_sort_tuple(obligation)
        for obligation in partition.obligations
    )


def _component_graph_surfaces(
    facts,
    graph_index,
    block_cut: WriterBlockCutMetadata,
) -> tuple[WriterComponentGraphSurface, ...]:
    block_by_bond = dict(block_cut.biconnected_block_by_bond)
    surfaces = []
    for index, component in enumerate(facts.components):
        atoms = frozenset(component.atoms)
        bonds = frozenset(component.bonds)
        connected_components = _component_connected_count(atoms, bonds, graph_index)
        connected = connected_components == 1
        cyclic_rank = len(bonds) - len(atoms) + connected_components
        cyclic_block_ids = frozenset(
            block_by_bond[bond]
            for bond in bonds
            if bond in block_by_bond and block_by_bond[bond] in block_cut.cyclic_blocks
        )
        tree = (
            bool(atoms)
            and connected
            and len(bonds) == len(atoms) - 1
            and cyclic_rank == 0
            and not cyclic_block_ids
        )
        surfaces.append(
            WriterComponentGraphSurface(
                component_index=index,
                atoms=atoms,
                bonds=bonds,
                connected=connected,
                tree=tree,
                cyclic_rank=cyclic_rank,
                cyclic_block_ids=cyclic_block_ids,
                unsupported_reason=(
                    None
                    if tree
                    else "WRITER_SHAPED writer-state runtime supports connected tree components only"
                ),
            )
        )
    return tuple(surfaces)


def _component_connected_count(
    atoms: frozenset[AtomId],
    bonds: frozenset[BondId],
    graph_index,
) -> int:
    remaining = set(atoms)
    count = 0
    while remaining:
        count += 1
        start = min(remaining)
        remaining.remove(start)
        stack = [start]
        while stack:
            atom = stack.pop()
            for neighbor in graph_index.neighbors[atom]:
                if neighbor not in remaining:
                    continue
                bond = graph_index.bond_between[(min(atom, neighbor), max(atom, neighbor))]
                if bond not in bonds:
                    continue
                remaining.remove(neighbor)
                stack.append(neighbor)
    return count


def _residual_atom_components(
    prepared: SouthStarPreparedMol,
    residual_atoms: frozenset[AtomId],
    latent_obligations: tuple[WriterEdgeObligation, ...],
) -> tuple[frozenset[AtomId], ...]:
    adjacency: dict[AtomId, list[AtomId]] = {}
    for obligation in latent_obligations:
        adjacency.setdefault(obligation.a, []).append(obligation.b)
        adjacency.setdefault(obligation.b, []).append(obligation.a)
    remaining = set(residual_atoms)
    components = []
    while remaining:
        start = min(remaining)
        seen = {start}
        stack = [start]
        remaining.remove(start)
        while stack:
            atom = stack.pop()
            for neighbor in adjacency.get(atom, ()):
                if neighbor not in remaining:
                    continue
                remaining.remove(neighbor)
                seen.add(neighbor)
                stack.append(neighbor)
        components.append(frozenset(seen))
    return tuple(components)


def _boundary_incidences(
    prepared: SouthStarPreparedMol,
    key: WriterStateKey,
    residual_atoms: frozenset[AtomId],
    boundary_obligations: tuple[WriterEdgeObligation, ...],
) -> tuple[WriterBoundaryIncidence, ...]:
    incidences = []
    for obligation in boundary_obligations:
        if obligation.a in residual_atoms:
            written_atom, residual_atom = obligation.b, obligation.a
        elif obligation.b in residual_atoms:
            written_atom, residual_atom = obligation.a, obligation.b
        else:
            continue
        incidences.append(
            WriterBoundaryIncidence(
                bond=obligation.bond,
                written_atom=written_atom,
                residual_atom=residual_atom,
                owner_kind=_owner_kind_for_boundary(key, written_atom),
            )
        )
    return tuple(incidences)


def _current_component(
    prepared: SouthStarPreparedMol,
    key: WriterStateKey,
):
    current = key.component_cursor.component_index
    if current < 0 or current >= len(prepared.facts.components):
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "writer component index is outside prepared components",
        )
    return prepared.facts.components[current]


def _require_pending_orientation(
    pending,
    obligation: WriterEdgeObligation,
) -> None:
    if pending.bond != obligation.bond:
        _invalid_edge_partition("pending obligation has wrong bond")
    if {pending.parent, pending.child} != {obligation.a, obligation.b}:
        _invalid_edge_partition("pending obligation has wrong endpoints")


def _invalid_edge_partition(message: str) -> None:
    raise SouthStarError(SouthStarErrorKind.INTERNAL_INVARIANT, message)


def _owner_kind_for_boundary(
    key: WriterStateKey,
    written_atom: AtomId,
) -> WriterBoundaryOwnerKind:
    pending = key.obligations.pending_entry
    if pending is not None and pending.parent == written_atom:
        return WriterBoundaryOwnerKind.PENDING_PARENT
    active = key.active
    if active.atom == written_atom:
        return WriterBoundaryOwnerKind.ACTIVE_ATOM
    if any(frame.return_atom.atom == written_atom for frame in key.branch_stack):
        return WriterBoundaryOwnerKind.BRANCH_RETURN
    return WriterBoundaryOwnerKind.UNOWNED


def _bond_endpoints_in(
    prepared: SouthStarPreparedMol,
    bond: BondId,
    atoms: frozenset[AtomId],
) -> bool:
    fact = prepared.graph_index.bond_by_id[bond]
    return fact.a in atoms and fact.b in atoms


def _canonical_boundary_index(
    items: dict[AtomId, set[int]],
) -> tuple[tuple[AtomId, tuple[int, ...]], ...]:
    return tuple(
        (atom, tuple(sorted(attachment_ids)))
        for atom, attachment_ids in sorted(items.items(), key=lambda item: int(item[0]))
    )


__all__ = (
    "WriterBlockCutMetadata",
    "WriterBoundaryIncidence",
    "WriterBoundaryOwnerKind",
    "WriterComponentGraphSurface",
    "WriterEdgeObligation",
    "WriterEdgeObligationKind",
    "WriterEdgeObligationPartition",
    "WriterGraphObligationContext",
    "WriterGraphObligationSummary",
    "WriterGraphPreparedMetadata",
    "WriterResidualAttachment",
    "WriterResidualAttachmentState",
    "build_writer_graph_obligation_context",
    "build_writer_graph_prepared_metadata",
    "build_writer_graph_prepared_metadata_from_facts",
    "build_writer_block_cut_metadata",
    "classify_writer_edge_obligations",
    "classify_writer_residual_attachments",
    "validate_writer_supported_graph_surface",
    "validate_writer_edge_obligation_partition",
    "writer_boundary_incidence_sort_tuple",
    "writer_edge_obligation_partition_sort_tuple",
    "writer_edge_obligation_sort_tuple",
    "writer_residual_attachment_sort_tuple",
)
