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


def build_writer_block_cut_metadata(
    prepared: SouthStarPreparedMol,
) -> WriterBlockCutMetadata:
    graph = prepared.graph_index
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

    for atom in prepared.atom_ids:
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


def classify_writer_residual_attachments(
    prepared: SouthStarPreparedMol,
    key: WriterStateKey,
    block_cut: WriterBlockCutMetadata,
) -> WriterGraphObligationSummary:
    current = key.component_cursor.component_index
    if current < 0 or current >= len(prepared.facts.components):
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "writer component index is outside prepared components",
        )
    component = prepared.facts.components[current]
    component_atoms = frozenset(component.atoms)
    component_bonds = frozenset(component.bonds)
    visited = frozenset(atom for atom in key.visited_atoms if atom in component_atoms)
    residual_atoms = component_atoms - visited
    block_by_bond = dict(block_cut.biconnected_block_by_bond)
    attachments: list[WriterResidualAttachment] = []

    for atoms in _residual_atom_components(prepared, residual_atoms, component_bonds):
        latent_bonds = frozenset(
            bond
            for bond in component_bonds
            if _bond_endpoints_in(prepared, bond, atoms)
        )
        boundary = tuple(
            sorted(
                _boundary_incidences(prepared, key, atoms, visited, component_bonds),
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
    )
    return WriterGraphObligationSummary(
        attachments=WriterResidualAttachmentState(
            attachments=tuple(sorted_attachments)
        ),
        boundary_by_owner_atom=_canonical_boundary_index(boundary_by_owner),
        boundary_by_pending_parent=_canonical_boundary_index(boundary_by_pending),
        has_cyclic_attachment=has_cyclic_attachment,
    )


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


def _residual_atom_components(
    prepared: SouthStarPreparedMol,
    residual_atoms: frozenset[AtomId],
    component_bonds: frozenset[BondId],
) -> tuple[frozenset[AtomId], ...]:
    remaining = set(residual_atoms)
    components = []
    while remaining:
        start = min(remaining)
        seen = {start}
        stack = [start]
        remaining.remove(start)
        while stack:
            atom = stack.pop()
            for neighbor in prepared.graph_index.neighbors[atom]:
                if neighbor not in remaining:
                    continue
                bond = prepared.graph_index.bond_between[
                    (min(atom, neighbor), max(atom, neighbor))
                ]
                if bond not in component_bonds:
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
    visited: frozenset[AtomId],
    component_bonds: frozenset[BondId],
) -> tuple[WriterBoundaryIncidence, ...]:
    incidences = []
    for bond in component_bonds:
        fact = prepared.graph_index.bond_by_id[bond]
        if fact.a in visited and fact.b in residual_atoms:
            written_atom, residual_atom = fact.a, fact.b
        elif fact.b in visited and fact.a in residual_atoms:
            written_atom, residual_atom = fact.b, fact.a
        else:
            continue
        incidences.append(
            WriterBoundaryIncidence(
                bond=bond,
                written_atom=written_atom,
                residual_atom=residual_atom,
                owner_kind=_owner_kind_for_boundary(key, written_atom),
            )
        )
    return tuple(incidences)


def _owner_kind_for_boundary(
    key: WriterStateKey,
    written_atom: AtomId,
) -> WriterBoundaryOwnerKind:
    pending = key.obligations.pending_entry
    if pending is not None and pending.parent == written_atom:
        return WriterBoundaryOwnerKind.PENDING_PARENT
    active = key.active
    if active is not None and active.atom == written_atom:
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
    "WriterGraphObligationSummary",
    "WriterResidualAttachment",
    "WriterResidualAttachmentState",
    "build_writer_block_cut_metadata",
    "classify_writer_residual_attachments",
    "writer_boundary_incidence_sort_tuple",
    "writer_residual_attachment_sort_tuple",
)
