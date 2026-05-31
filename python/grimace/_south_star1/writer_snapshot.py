"""Writer-shaped frontier snapshots."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .facts import LigandKind
from .ids import AtomId
from .ids import BondId
from .ids import OccurrenceId
from .ids import SiteId
from .policy import SerializationLanguageMode
from .policy import DirectionMark
from .policy import TetraToken
from .prepared_runtime import SouthStarPreparedMol
from .prepared_runtime import SouthStarRuntimeOptions
from .prepared_runtime import require_writer_shaped_runtime_options
from .residual_constraints import DirectionalCarrierResidual
from .residual_constraints import DirectionalResidualFactor
from .residual_constraints import DirectionalResidualFactorValueSnapshot
from .residual_constraints import ResidualStore
from .residual_constraints import ResidualStoreValueSnapshot
from .residual_constraints import TetraResidualFactor
from .residual_constraints import TetraResidualFactorValueSnapshot
from .residual_constraints import VarId
from .residual_constraints import direction_var
from .residual_constraints import tetra_var
from .writer_graph_obligations import WriterBoundaryOwnerKind
from .writer_graph_obligations import WriterGraphObligationSummary
from .writer_graph_obligations import build_writer_block_cut_metadata
from .writer_graph_obligations import classify_writer_residual_attachments
from .writer_frontier import WriterFrontierChoices
from .writer_frontier import WriterFrontierCursor
from .writer_frontier import writer_frontier_choices
from .writer_state import ComponentCursor
from .writer_state import ObligationStateKey
from .writer_state import PendingEntryPhase
from .writer_state import PendingWriterEntry
from .writer_state import WriterAtomFrame
from .writer_state import WriterBranchFrame
from .writer_state import WriterRingStateKey
from .writer_state import WriterStateKey
from .writer_state import WriterStereoStateKey


@dataclass(frozen=True, slots=True)
class WriterPreparedIdentity:
    runtime: tuple[object, ...]
    atoms: tuple[tuple[object, ...], ...]
    bonds: tuple[tuple[object, ...], ...]
    components: tuple[tuple[object, ...], ...]
    ligand_occurrences: tuple[tuple[object, ...], ...]
    tetra_templates: tuple[tuple[object, ...], ...]
    directional_templates: tuple[tuple[object, ...], ...]
    policy: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class WriterDecoderBoundary:
    consumed_token_count: int = 0


@dataclass(frozen=True, slots=True)
class WriterFrontierFrame:
    cursor: WriterFrontierCursor


WriterSnapshotFrame = WriterFrontierFrame


@dataclass(frozen=True, slots=True)
class WriterSearchSnapshot:
    serialization_language: SerializationLanguageMode
    prepared_identity: WriterPreparedIdentity
    runtime_options: SouthStarRuntimeOptions
    cursor: WriterFrontierCursor
    decoder_boundary: WriterDecoderBoundary
    frame_stack: tuple[WriterSnapshotFrame, ...]


def capture_writer_frontier_snapshot(
    *,
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions,
    cursor: WriterFrontierCursor,
    decoder_boundary: WriterDecoderBoundary = WriterDecoderBoundary(),
) -> WriterSearchSnapshot:
    require_writer_shaped_runtime_options(runtime_options)
    snapshot = WriterSearchSnapshot(
        serialization_language=SerializationLanguageMode.WRITER_SHAPED,
        prepared_identity=_prepared_identity(prepared, runtime_options),
        runtime_options=runtime_options,
        cursor=cursor,
        decoder_boundary=decoder_boundary,
        frame_stack=(WriterFrontierFrame(cursor),),
    )
    validate_writer_search_snapshot(snapshot, prepared=prepared)
    return snapshot


def writer_frontier_cursor_from_snapshot(
    snapshot: WriterSearchSnapshot,
    *,
    prepared: SouthStarPreparedMol,
) -> WriterFrontierCursor:
    validate_writer_search_snapshot(snapshot, prepared=prepared)
    return snapshot.cursor


def resume_writer_frontier_choices_from_snapshot(
    snapshot: WriterSearchSnapshot,
    *,
    prepared: SouthStarPreparedMol,
) -> WriterFrontierChoices:
    cursor = writer_frontier_cursor_from_snapshot(snapshot, prepared=prepared)
    return writer_frontier_choices(prepared, cursor)


def validate_writer_search_snapshot(
    snapshot: WriterSearchSnapshot,
    *,
    prepared: SouthStarPreparedMol,
) -> None:
    if snapshot.serialization_language is not SerializationLanguageMode.WRITER_SHAPED:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_POLICY,
            "writer snapshot requires serialization_language=WRITER_SHAPED",
        )
    require_writer_shaped_runtime_options(snapshot.runtime_options)
    if snapshot.prepared_identity != _prepared_identity(
        prepared,
        snapshot.runtime_options,
    ):
        raise SouthStarError(
            SouthStarErrorKind.INVALID_FACTS,
            "writer snapshot prepared identity does not match prepared molecule",
        )
    if snapshot.cursor != WriterFrontierCursor(snapshot.cursor.weighted_states):
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "writer snapshot cursor is not canonical",
        )
    _validate_frames(snapshot.frame_stack, snapshot.cursor)
    validate_writer_cursor_against_prepared(
        prepared,
        snapshot.cursor,
        runtime_options=snapshot.runtime_options,
    )


def validate_writer_cursor_against_prepared(
    prepared: SouthStarPreparedMol,
    cursor: WriterFrontierCursor,
    *,
    runtime_options: SouthStarRuntimeOptions | None = None,
) -> None:
    atom_ids = frozenset(prepared.atom_ids)
    bond_ids = frozenset(bond.id for bond in prepared.facts.bonds)
    allowed_roots = _allowed_component_roots(prepared, runtime_options)
    atom_component = _atom_component_index(prepared)
    bond_component = _bond_component_index(prepared)
    for key, weight in cursor.weighted_states:
        if weight <= 0:
            _invalid_snapshot("writer cursor contains nonpositive weight")
        _validate_component_cursor(key.component_cursor, allowed_roots)
        summary = _writer_obligation_summary(prepared, key)
        _validate_atom_frame(key.active, atom_ids, bond_ids, prepared)
        for frame in key.branch_stack:
            _validate_branch_frame(frame, atom_ids, bond_ids, prepared)
        _validate_known_atoms("visited_atoms", key.visited_atoms, atom_ids)
        _validate_known_bonds("written_bonds", key.written_bonds, bond_ids)
        _validate_active_coherence(key)
        _validate_component_membership(prepared, key, atom_component, bond_component)
        _validate_current_component_tree_fragment(prepared, key)
        _validate_writer_frame_tree_path(prepared, key)
        _validate_written_bond_coherence(prepared, key)
        _validate_obligations(
            key.obligations,
            key,
            atom_ids,
            bond_ids,
            prepared,
            summary,
        )
        _validate_live_frontier_ownership(prepared, key, summary)
        _validate_stereo_occurrences_bound_to_graph_state(prepared, key)
        _validate_ring_state_empty(key.ring_state)
        _validate_policy_state(key, atom_ids, bond_ids)
        _validate_stereo_state(prepared, key.stereo_state)


def _validate_frames(
    frame_stack: tuple[object, ...],
    cursor: WriterFrontierCursor,
) -> None:
    if len(frame_stack) != 1:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "writer snapshot currently requires exactly one frontier frame",
        )
    frame = frame_stack[0]
    if not isinstance(frame, WriterFrontierFrame):
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "writer snapshot top frame must be a frontier frame",
        )
    if frame.cursor != cursor:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "writer snapshot frontier frame cursor must match snapshot cursor",
        )


def _round_trip_residual_snapshot(snapshot: ResidualStoreValueSnapshot) -> None:
    round_tripped = ResidualStore.from_value_snapshot(snapshot).value_snapshot()
    if round_tripped != snapshot:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "writer residual snapshot does not round-trip",
        )


def _writer_obligation_summary(
    prepared: SouthStarPreparedMol,
    key: WriterStateKey,
) -> WriterGraphObligationSummary:
    return classify_writer_residual_attachments(
        prepared,
        key,
        build_writer_block_cut_metadata(prepared),
    )


def _allowed_component_roots(
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions | None,
) -> tuple[frozenset[AtomId], ...]:
    if runtime_options is None or runtime_options.rooted_at_atom < 0:
        domains = prepared.all_root_domains
    else:
        try:
            domains = prepared.component_root_domains_by_explicit_root[
                AtomId(runtime_options.rooted_at_atom)
            ]
        except KeyError as exc:
            raise SouthStarError(
                SouthStarErrorKind.INVALID_FACTS,
                "writer snapshot runtime root is not in prepared molecule",
            ) from exc
    return tuple(frozenset(atoms) for _, atoms in domains)


def _validate_component_cursor(
    cursor: ComponentCursor,
    allowed_roots: tuple[frozenset[AtomId], ...],
) -> None:
    if len(cursor.component_roots) != len(allowed_roots):
        _invalid_snapshot("writer component root count does not match prepared domains")
    if cursor.component_index < 0 or cursor.component_index >= len(cursor.component_roots):
        _invalid_snapshot("writer component index is outside component roots")
    for index, root in enumerate(cursor.component_roots):
        if root not in allowed_roots[index]:
            _invalid_snapshot("writer component root is outside runtime root domain")


def _validate_atom_frame(
    frame: WriterAtomFrame | None,
    atom_ids: frozenset[AtomId],
    bond_ids: frozenset[BondId],
    prepared: SouthStarPreparedMol,
) -> None:
    if frame is None:
        return
    if frame.atom not in atom_ids:
        _invalid_snapshot("writer atom frame references unknown atom")
    if frame.parent is None or frame.incoming_bond is None:
        if frame.parent is not None or frame.incoming_bond is not None:
            _invalid_snapshot("writer atom frame has partial incoming edge")
        return
    if frame.parent not in atom_ids or frame.incoming_bond not in bond_ids:
        _invalid_snapshot("writer atom frame references unknown incoming edge")
    _require_graph_bond(prepared, frame.parent, frame.atom, frame.incoming_bond)


def _validate_branch_frame(
    frame: WriterBranchFrame,
    atom_ids: frozenset[AtomId],
    bond_ids: frozenset[BondId],
    prepared: SouthStarPreparedMol,
) -> None:
    _validate_atom_frame(frame.return_atom, atom_ids, bond_ids, prepared)
    if not frame.return_atom.atom_emitted:
        _invalid_snapshot("writer branch return frame must be emitted")


def _validate_known_atoms(
    label: str,
    atoms: frozenset[AtomId],
    atom_ids: frozenset[AtomId],
) -> None:
    if not atoms.issubset(atom_ids):
        _invalid_snapshot(f"writer {label} references unknown atom")


def _validate_known_bonds(
    label: str,
    bonds: frozenset[BondId],
    bond_ids: frozenset[BondId],
) -> None:
    if not bonds.issubset(bond_ids):
        _invalid_snapshot(f"writer {label} references unknown bond")


def _validate_active_coherence(key: WriterStateKey) -> None:
    active = key.active
    if active is None:
        _invalid_snapshot("writer snapshot cannot use active=None before terminal-state support")
    if active.parent is None:
        if active.atom != key.component_cursor.component_roots[
            key.component_cursor.component_index
        ]:
            _invalid_snapshot("writer root active frame does not match component root")
        if active.incoming_bond is not None:
            _invalid_snapshot("writer root active frame has incoming bond")
    elif active.incoming_bond is None:
        _invalid_snapshot("writer non-root active frame lacks incoming bond")
    if active.atom_emitted:
        if active.atom not in key.visited_atoms:
            _invalid_snapshot("writer emitted active atom is not visited")
    elif active.atom in key.visited_atoms:
        _invalid_snapshot("writer un-emitted active atom is already visited")
    if active.parent is None:
        if active.incoming_bond is not None:
            _invalid_snapshot("writer root active frame has incoming bond")
    elif active.parent not in key.visited_atoms:
        _invalid_snapshot("writer active parent is not visited")
    if active.incoming_bond is not None and active.atom_emitted:
        if active.incoming_bond not in key.written_bonds:
            _invalid_snapshot("writer emitted child lacks written incoming bond")
    for frame in key.branch_stack:
        if frame.return_atom.atom not in key.visited_atoms:
            _invalid_snapshot("writer branch return atom is not visited")


def _validate_component_membership(
    prepared: SouthStarPreparedMol,
    key: WriterStateKey,
    atom_component: dict[AtomId, int],
    bond_component: dict[BondId, int],
) -> None:
    current = key.component_cursor.component_index
    allowed_components = set(range(current + 1))
    active = key.active
    if active is not None and atom_component[active.atom] != current:
        _invalid_snapshot("writer active atom is outside current component")
    for atom in key.visited_atoms:
        if atom_component[atom] not in allowed_components:
            _invalid_snapshot("writer visited atom is outside completed/current components")
    for bond in key.written_bonds:
        if bond_component[bond] not in allowed_components:
            _invalid_snapshot("writer written bond is outside completed/current components")
    pending = key.obligations.pending_entry
    if pending is not None:
        if (
            atom_component[pending.parent] != current
            or atom_component[pending.child] != current
            or bond_component[pending.bond] != current
        ):
            _invalid_snapshot("writer pending entry is outside current component")
    for frame in key.branch_stack:
        if atom_component[frame.return_atom.atom] != current:
            _invalid_snapshot("writer branch return atom is outside current component")
    for index, component in enumerate(prepared.facts.components):
        if index >= current:
            break
        if not frozenset(component.atoms).issubset(key.visited_atoms):
            _invalid_snapshot("writer completed component has unvisited atoms")
        if not frozenset(component.bonds).issubset(key.written_bonds):
            _invalid_snapshot("writer completed component has unwritten bonds")


def _validate_current_component_tree_fragment(
    prepared: SouthStarPreparedMol,
    key: WriterStateKey,
) -> None:
    current = key.component_cursor.component_index
    component = prepared.facts.components[current]
    component_atoms = frozenset(component.atoms)
    component_bonds = frozenset(component.bonds)
    root = key.component_cursor.component_roots[current]
    visited = frozenset(atom for atom in key.visited_atoms if atom in component_atoms)
    written = frozenset(bond for bond in key.written_bonds if bond in component_bonds)
    if not visited:
        if written:
            _invalid_snapshot("writer current component has written bonds before root")
        return
    if root not in visited:
        _invalid_snapshot("writer current component visited atoms do not include root")
    if len(written) != len(visited) - 1:
        _invalid_snapshot("writer current component written graph is not a tree fragment")
    reachable = _reachable_written_atoms(prepared, root, written)
    if reachable != visited:
        _invalid_snapshot("writer current component visited atoms are not root-reachable")
    active = key.active
    if active is not None and active.atom_emitted and active.atom not in reachable:
        _invalid_snapshot("writer active atom is not in reachable written graph")
    for frame in key.branch_stack:
        if frame.return_atom.atom not in reachable:
            _invalid_snapshot("writer branch return atom is not in reachable written graph")
    pending = key.obligations.pending_entry
    if pending is not None:
        if pending.parent not in reachable:
            _invalid_snapshot("writer pending parent is not in reachable written graph")
        if pending.phase is PendingEntryPhase.NEEDS_ATOM_AFTER_BOND:
            if pending.bond in written or pending.child in visited:
                _invalid_snapshot("writer pending post-bond edge is already materialized")


def _reachable_written_atoms(
    prepared: SouthStarPreparedMol,
    root: AtomId,
    written_bonds: frozenset[BondId],
) -> frozenset[AtomId]:
    adjacency: dict[AtomId, set[AtomId]] = {}
    for bond in written_bonds:
        fact = prepared.graph_index.bond_by_id[bond]
        adjacency.setdefault(fact.a, set()).add(fact.b)
        adjacency.setdefault(fact.b, set()).add(fact.a)
    seen = {root}
    stack = [root]
    while stack:
        atom = stack.pop()
        for neighbor in adjacency.get(atom, ()):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            stack.append(neighbor)
    return frozenset(seen)


def _validate_writer_frame_tree_path(
    prepared: SouthStarPreparedMol,
    key: WriterStateKey,
) -> None:
    parent_links = _written_tree_parent_links(prepared, key)
    root = key.component_cursor.component_roots[key.component_cursor.component_index]
    _validate_atom_frame_tree_edge(key.active, root, parent_links)
    for frame in key.branch_stack:
        _validate_atom_frame_tree_edge(frame.return_atom, root, parent_links)
    if key.active is None or not key.active.atom_emitted:
        if key.branch_stack:
            _invalid_snapshot("writer branch stack requires emitted active atom")
        return
    active_path = _root_to_atom_path(root, key.active.atom, parent_links)
    ancestor_positions = {atom: index for index, atom in enumerate(active_path[:-1])}
    previous_position = -1
    for frame in key.branch_stack:
        position = ancestor_positions.get(frame.return_atom.atom)
        if position is None:
            _invalid_snapshot("writer branch return atom is not an active ancestor")
        if position <= previous_position:
            _invalid_snapshot("writer branch stack does not follow root-to-active path")
        previous_position = position


def _validate_atom_frame_tree_edge(
    frame: WriterAtomFrame | None,
    root: AtomId,
    parent_links: dict[AtomId, tuple[AtomId, BondId]],
) -> None:
    if frame is None or not frame.atom_emitted:
        return
    if frame.atom == root:
        if frame.parent is not None or frame.incoming_bond is not None:
            _invalid_snapshot("writer root frame disagrees with written-tree root")
        return
    expected = parent_links.get(frame.atom)
    if expected is None:
        _invalid_snapshot("writer atom frame is missing from written-tree parent links")
    if (frame.parent, frame.incoming_bond) != expected:
        _invalid_snapshot("writer atom frame disagrees with written-tree orientation")


def _root_to_atom_path(
    root: AtomId,
    atom: AtomId,
    parent_links: dict[AtomId, tuple[AtomId, BondId]],
) -> tuple[AtomId, ...]:
    reversed_path = [atom]
    current = atom
    seen = {atom}
    while current != root:
        parent = parent_links.get(current)
        if parent is None:
            _invalid_snapshot("writer active atom is not connected to written-tree root")
        current = parent[0]
        if current in seen:
            _invalid_snapshot("writer written-tree parent links contain a cycle")
        seen.add(current)
        reversed_path.append(current)
    return tuple(reversed(reversed_path))


def _validate_written_bond_coherence(
    prepared: SouthStarPreparedMol,
    key: WriterStateKey,
) -> None:
    for bond in key.written_bonds:
        fact = prepared.graph_index.bond_by_id[bond]
        left, right = fact.a, fact.b
        if left not in key.visited_atoms or right not in key.visited_atoms:
            _invalid_snapshot("writer written bond has unvisited endpoint")


def _validate_obligations(
    obligations: ObligationStateKey,
    key: WriterStateKey,
    atom_ids: frozenset[AtomId],
    bond_ids: frozenset[BondId],
    prepared: SouthStarPreparedMol,
    summary: WriterGraphObligationSummary,
) -> None:
    pending = obligations.pending_entry
    if pending is None:
        return
    _validate_pending_entry(pending, atom_ids, bond_ids, prepared)
    _validate_pending_entry_role(summary, pending)
    if key.active is None or key.active.atom != pending.parent:
        _invalid_snapshot("writer pending entry parent is not active")
    if not key.active.atom_emitted:
        _invalid_snapshot("writer pending entry parent is not emitted")
    if pending.parent not in key.visited_atoms:
        _invalid_snapshot("writer pending parent is not visited")
    if pending.child in key.visited_atoms or pending.bond in key.written_bonds:
        _invalid_snapshot("writer pending entry is already written")
    has_bond_record = _has_bond_occurrence_record(
        key.stereo_state,
        pending.bond,
        pending.parent,
        pending.child,
    )
    if pending.phase is PendingEntryPhase.NEEDS_ATOM_AFTER_BOND:
        if not has_bond_record:
            _invalid_snapshot("writer pending post-bond entry lacks bond occurrence")
    elif pending.phase is PendingEntryPhase.NEEDS_BOND_OR_ATOM:
        if has_bond_record:
            _invalid_snapshot("writer pending pre-bond entry already has bond occurrence")
    else:
        _invalid_snapshot("writer pending entry has unknown phase")


def _validate_pending_entry(
    pending: PendingWriterEntry,
    atom_ids: frozenset[AtomId],
    bond_ids: frozenset[BondId],
    prepared: SouthStarPreparedMol,
) -> None:
    if pending.parent not in atom_ids or pending.child not in atom_ids:
        _invalid_snapshot("writer pending entry references unknown atom")
    if pending.bond not in bond_ids:
        _invalid_snapshot("writer pending entry references unknown bond")
    _require_graph_bond(prepared, pending.parent, pending.child, pending.bond)


def _validate_pending_entry_role(
    summary: WriterGraphObligationSummary,
    pending: PendingWriterEntry,
) -> None:
    children = _boundary_children_for_atom(summary, pending.parent)
    if (pending.bond, pending.child) not in children:
        _invalid_snapshot("writer pending entry is not a live child obligation")
    if pending.branch:
        if len(children) <= 1:
            _invalid_snapshot("writer pending branch entry has no sibling obligations")
    elif children != ((pending.bond, pending.child),):
        _invalid_snapshot("writer pending inline entry is not the final child")


def _validate_live_frontier_ownership(
    prepared: SouthStarPreparedMol,
    key: WriterStateKey,
    summary: WriterGraphObligationSummary,
) -> None:
    current = key.component_cursor.component_index
    component = prepared.facts.components[current]
    component_atoms = frozenset(component.atoms)
    visited = frozenset(atom for atom in key.visited_atoms if atom in component_atoms)
    unvisited = component_atoms - visited
    if not visited:
        return
    boundary_edges = [
        incidence
        for attachment in summary.attachments.attachments
        for incidence in attachment.boundary
    ]
    if any(not attachment.boundary for attachment in summary.attachments.attachments):
        _invalid_snapshot("writer residual attachment has no boundary incidence")
    branch_return_atoms = tuple(frame.return_atom.atom for frame in key.branch_stack)
    branch_owned_atoms = {
        incidence.written_atom
        for incidence in boundary_edges
        if incidence.owner_kind is WriterBoundaryOwnerKind.BRANCH_RETURN
    }
    if any(
        incidence.owner_kind is WriterBoundaryOwnerKind.UNOWNED
        for incidence in boundary_edges
    ):
        _invalid_snapshot("writer live frontier does not own unvisited obligation")
    if unvisited and not boundary_edges:
        _invalid_snapshot("writer current component has unvisited atoms without frontier")
    if key.branch_stack and not unvisited:
        _invalid_snapshot("writer branch stack has no unresolved return obligation")
    if any(atom not in branch_owned_atoms for atom in branch_return_atoms):
        _invalid_snapshot("writer branch return frame owns no unresolved obligation")
    if (
        not unvisited
        and key.obligations.pending_entry is None
        and not key.branch_stack
        and not _active_is_terminal_leaf(prepared, key)
    ):
        _invalid_snapshot("writer completed component active frame is not terminal")


def _boundary_children_for_atom(
    summary: WriterGraphObligationSummary,
    atom: AtomId,
) -> tuple[tuple[BondId, AtomId], ...]:
    children = []
    for attachment in summary.attachments.attachments:
        boundary = tuple(
            incidence
            for incidence in attachment.boundary
            if incidence.written_atom == atom
        )
        if not boundary:
            continue
        if len(boundary) != 1:
            _invalid_snapshot("writer residual attachment has multiple incidences")
        incidence = boundary[0]
        children.append((incidence.bond, incidence.residual_atom))
    return tuple(sorted(children, key=lambda item: (int(item[0]), int(item[1]))))


def _active_is_terminal_leaf(
    prepared: SouthStarPreparedMol,
    key: WriterStateKey,
) -> bool:
    active = key.active
    if active is None or not active.atom_emitted:
        return False
    current = key.component_cursor.component_index
    component = prepared.facts.components[current]
    if len(component.atoms) == 1:
        return active.atom == key.component_cursor.component_roots[current]
    parent_links = _written_tree_parent_links(prepared, key)
    children = {
        child
        for child, (parent, _) in parent_links.items()
        if parent == active.atom
    }
    return not children


def _validate_stereo_occurrences_bound_to_graph_state(
    prepared: SouthStarPreparedMol,
    key: WriterStateKey,
) -> None:
    atom_occurrence_atoms = frozenset(
        record.atom for record in key.stereo_state.atom_occurrences
    )
    if atom_occurrence_atoms != key.visited_atoms:
        _invalid_snapshot("writer atom occurrences do not cover visited atoms")
    pending_bond = _pending_post_bond_edge(key)
    expected_bonds = set(key.written_bonds)
    if pending_bond is not None:
        expected_bonds.add(pending_bond.bond)
    bond_occurrence_bonds = frozenset(
        record.bond for record in key.stereo_state.bond_occurrences
    )
    if bond_occurrence_bonds != frozenset(expected_bonds):
        _invalid_snapshot("writer bond occurrences do not cover emitted bonds")
    parent_by_child = _written_tree_parent_links(prepared, key)
    for record in key.stereo_state.atom_occurrences:
        if record.atom not in key.visited_atoms:
            _invalid_snapshot("writer atom occurrence is not backed by visited atom")
    for record in key.stereo_state.local_orders:
        if record.atom not in key.visited_atoms:
            _invalid_snapshot("writer local-order record is not backed by visited atom")
    for record in key.stereo_state.bond_occurrences:
        if record.bond in key.written_bonds:
            if record.parent not in key.visited_atoms or record.child not in key.visited_atoms:
                _invalid_snapshot("writer bond occurrence has unvisited written endpoint")
            expected = parent_by_child.get(record.child)
            if expected != (record.parent, record.bond):
                _invalid_snapshot("writer bond occurrence has wrong writer orientation")
            continue
        if (
            pending_bond is not None
            and pending_bond.bond == record.bond
            and pending_bond.parent == record.parent
            and pending_bond.child == record.child
        ):
            if record.parent not in key.visited_atoms:
                _invalid_snapshot("writer pending bond occurrence has unvisited parent")
            if record.child in key.visited_atoms or record.bond in key.written_bonds:
                _invalid_snapshot("writer pending bond occurrence is already materialized")
            continue
        _invalid_snapshot("writer bond occurrence is not backed by emitted graph state")


def _pending_post_bond_edge(key: WriterStateKey) -> PendingWriterEntry | None:
    pending = key.obligations.pending_entry
    if pending is None or pending.phase is not PendingEntryPhase.NEEDS_ATOM_AFTER_BOND:
        return None
    return pending


def _written_tree_parent_links(
    prepared: SouthStarPreparedMol,
    key: WriterStateKey,
) -> dict[AtomId, tuple[AtomId, BondId]]:
    parent_by_child: dict[AtomId, tuple[AtomId, BondId]] = {}
    for index in range(key.component_cursor.component_index + 1):
        component = prepared.facts.components[index]
        component_bonds = frozenset(component.bonds)
        written = frozenset(bond for bond in key.written_bonds if bond in component_bonds)
        root = key.component_cursor.component_roots[index]
        adjacency: dict[AtomId, list[tuple[AtomId, BondId]]] = {}
        for bond in written:
            fact = prepared.graph_index.bond_by_id[bond]
            adjacency.setdefault(fact.a, []).append((fact.b, bond))
            adjacency.setdefault(fact.b, []).append((fact.a, bond))
        seen = {root}
        stack = [root]
        while stack:
            parent = stack.pop()
            for child, bond in adjacency.get(parent, ()):
                if child in seen:
                    continue
                seen.add(child)
                parent_by_child[child] = (parent, bond)
                stack.append(child)
    return parent_by_child


def _validate_ring_state_empty(ring_state: WriterRingStateKey) -> None:
    if (
        ring_state.open_endpoints
        or ring_state.active_spines
        or ring_state.closed_bonds
    ):
        _invalid_snapshot("writer snapshot has nonempty ring state before cyclic support")


def _validate_policy_state(
    key: WriterStateKey,
    atom_ids: frozenset[AtomId],
    bond_ids: frozenset[BondId],
) -> None:
    if any(atom not in atom_ids for atom, _ in key.policy_state.atom_text):
        _invalid_snapshot("writer policy atom text references unknown atom")
    if any(bond not in bond_ids for bond, _ in key.policy_state.bond_text):
        _invalid_snapshot("writer policy bond text references unknown bond")


def _validate_stereo_state(
    prepared: SouthStarPreparedMol,
    stereo_state: WriterStereoStateKey,
) -> None:
    _round_trip_residual_snapshot(stereo_state.residual_snapshot)
    _validate_unique_stereo_records(stereo_state)
    occurrence_by_id = {item.id: item for item in prepared.facts.ligand_occurrences}
    atom_ids = frozenset(prepared.atom_ids)
    bond_ids = frozenset(bond.id for bond in prepared.facts.bonds)
    assignments = dict(stereo_state.residual_snapshot.assignments)
    tetra_by_center = {template.center: template for template in prepared.tetra_templates}
    directional_sites_by_bond = _directional_sites_by_carrier_bond(prepared)
    _validate_atom_occurrence_records(
        stereo_state,
        atom_ids,
        tetra_by_center,
        assignments,
    )
    _validate_bond_occurrence_records(
        stereo_state,
        atom_ids,
        bond_ids,
        prepared,
        directional_sites_by_bond,
        assignments,
    )
    _validate_local_order_records(
        prepared,
        stereo_state,
        occurrence_by_id,
        atom_ids,
        tetra_by_center,
    )
    _validate_delayed_factor_records(prepared, stereo_state)
    _validate_reverse_stereo_coverage(
        prepared,
        stereo_state,
        tetra_by_center,
        directional_sites_by_bond,
    )


def _validate_unique_stereo_records(stereo_state: WriterStereoStateKey) -> None:
    _reject_duplicate_items(
        (
            ("tetra", record.var)
            if record.var is not None
            else ("atom", record.atom)
            for record in stereo_state.atom_occurrences
        ),
        "writer atom occurrence records contain duplicates",
    )
    _reject_duplicate_items(
        ((record.bond, record.parent, record.child) for record in stereo_state.bond_occurrences),
        "writer bond occurrence records contain duplicate orientations",
    )
    _reject_duplicate_items(
        (record.bond for record in stereo_state.bond_occurrences),
        "writer bond occurrence records contain duplicate bonds",
    )
    _reject_duplicate_items(
        (record.atom for record in stereo_state.local_orders),
        "writer local-order records contain duplicate atoms",
    )
    _reject_duplicate_items(
        (
            (factor.kind, factor.site)
            for factor in stereo_state.delayed_factors
        ),
        "writer delayed factors contain duplicates",
    )
    _reject_duplicate_items(
        stereo_state.residual_snapshot.factors,
        "writer residual factor snapshots contain duplicates",
    )
    _reject_duplicate_items(
        (var for var, _ in stereo_state.residual_snapshot.domains),
        "writer residual domains contain duplicate variables",
    )
    _reject_duplicate_items(
        (var for var, _ in stereo_state.residual_snapshot.assignments),
        "writer residual assignments contain duplicate variables",
    )


def _validate_atom_occurrence_records(
    stereo_state: WriterStereoStateKey,
    atom_ids: frozenset[AtomId],
    tetra_by_center,
    assignments: dict[VarId, object],
) -> None:
    domain_vars = _residual_domain_vars(stereo_state)
    for record in stereo_state.atom_occurrences:
        if record.atom not in atom_ids:
            _invalid_snapshot("writer atom occurrence references unknown atom")
        template = tetra_by_center.get(record.atom)
        if template is None:
            if record.var is not None:
                _invalid_snapshot("writer atom occurrence has unexpected tetra variable")
            if record.token is not TetraToken.NONE:
                _invalid_snapshot("writer atom occurrence has unexpected tetra token")
            continue
        expected_var = tetra_var(("writer", int(template.site)))
        if record.var != expected_var:
            _invalid_snapshot("writer atom occurrence has wrong tetra variable")
        if expected_var not in domain_vars:
            _invalid_snapshot("writer atom occurrence variable is missing from residual store")
        if assignments.get(expected_var) is not record.token:
            _invalid_snapshot("writer atom occurrence token does not match residual assignment")


def _validate_bond_occurrence_records(
    stereo_state: WriterStereoStateKey,
    atom_ids: frozenset[AtomId],
    bond_ids: frozenset[BondId],
    prepared: SouthStarPreparedMol,
    directional_sites_by_bond: dict[BondId, tuple[SiteId, ...]],
    assignments: dict[VarId, object],
) -> None:
    domain_vars = _residual_domain_vars(stereo_state)
    for record in stereo_state.bond_occurrences:
        if record.bond not in bond_ids or record.parent not in atom_ids or record.child not in atom_ids:
            _invalid_snapshot("writer bond occurrence references unknown graph item")
        _require_graph_bond(prepared, record.parent, record.child, record.bond)
        eligible_sites = directional_sites_by_bond.get(record.bond, ())
        if not eligible_sites:
            if record.var is not None:
                _invalid_snapshot("writer bond occurrence has unexpected directional variable")
            if record.mark is not DirectionMark.ABSENT:
                _invalid_snapshot("writer bond occurrence has unexpected direction mark")
            continue
        expected_var = direction_var(("writer", int(record.bond)))
        if record.var != expected_var:
            _invalid_snapshot("writer bond occurrence has wrong directional variable")
        if expected_var not in domain_vars:
            _invalid_snapshot("writer bond occurrence variable is missing from residual store")
        if assignments.get(expected_var) is not record.mark:
            _invalid_snapshot("writer bond occurrence mark does not match residual assignment")


def _validate_local_order_records(
    prepared: SouthStarPreparedMol,
    stereo_state: WriterStereoStateKey,
    occurrence_by_id,
    atom_ids: frozenset[AtomId],
    tetra_by_center,
) -> None:
    for record in stereo_state.local_orders:
        if record.atom not in atom_ids:
            _invalid_snapshot("writer local-order record references unknown atom")
        if len(set(record.order)) != len(record.order):
            _invalid_snapshot("writer local-order record repeats ligand occurrence")
        template = tetra_by_center.get(record.atom)
        allowed = _allowed_local_order_occurrences(prepared, record.atom, template)
        for occurrence_id in record.order:
            occurrence = occurrence_by_id.get(occurrence_id)
            if occurrence is None:
                _invalid_snapshot("writer local-order record references unknown ligand occurrence")
            if occurrence_id not in allowed:
                _invalid_snapshot("writer local-order occurrence belongs to another site")
            if occurrence.kind is LigandKind.IMPLICIT_H:
                if occurrence.atom != record.atom:
                    _invalid_snapshot("writer local-order implicit-H occurrence is on another atom")
            elif occurrence.kind is LigandKind.NEIGHBOR_ATOM:
                if occurrence.atom not in atom_ids or occurrence.bond is None:
                    _invalid_snapshot("writer local-order neighbor occurrence references unknown atom")
                _require_graph_bond(prepared, record.atom, occurrence.atom, occurrence.bond)
            else:
                _invalid_snapshot("writer local-order pseudo occurrence is unsupported")
        if record.closed and template is not None:
            if set(record.order) != set(template.ligand_occurrences):
                _invalid_snapshot("writer closed tetra local order is incomplete")


def _validate_delayed_factor_records(
    prepared: SouthStarPreparedMol,
    stereo_state: WriterStereoStateKey,
) -> None:
    domain_vars = _residual_domain_vars(stereo_state)
    assignment_vars = _residual_assignment_vars(stereo_state)
    factor_snapshots = stereo_state.residual_snapshot.factors
    tetra_by_site = {template.site: template for template in prepared.tetra_templates}
    directional_by_site = {
        template.site: template for template in prepared.directional_templates
    }
    for factor in stereo_state.delayed_factors:
        if not factor.scope:
            _invalid_snapshot("writer delayed factor has empty scope")
        for var in factor.scope:
            if var not in domain_vars:
                _invalid_snapshot("writer delayed factor variable is missing from residual store")
            if var not in assignment_vars:
                _invalid_snapshot("writer delayed factor variable is unassigned")
        _validate_delayed_factor_shape(
            prepared,
            stereo_state,
            factor,
            tetra_by_site,
            directional_by_site,
        )
        if factor.closed:
            expected = _expected_residual_factor_snapshot(prepared, stereo_state, factor)
            if expected not in factor_snapshots:
                _invalid_snapshot("writer closed delayed factor lacks matching residual factor")
    for snapshot in factor_snapshots:
        if not _has_matching_closed_delayed_factor(
            prepared,
            stereo_state,
            snapshot,
            stereo_state.delayed_factors,
        ):
            _invalid_snapshot("writer residual factor snapshot lacks closed delayed factor")


def _has_matching_closed_delayed_factor(
    prepared: SouthStarPreparedMol,
    stereo_state: WriterStereoStateKey,
    snapshot: object,
    factors,
) -> bool:
    if isinstance(snapshot, TetraResidualFactorValueSnapshot):
        kind = "tetra"
    elif isinstance(snapshot, DirectionalResidualFactorValueSnapshot):
        kind = "directional"
    else:
        _invalid_snapshot("writer residual snapshot has unknown factor type")
    for factor in factors:
        if factor.kind != kind or not factor.closed:
            continue
        if _expected_residual_factor_snapshot(prepared, stereo_state, factor) == snapshot:
            return True
    return False


def _validate_reverse_stereo_coverage(
    prepared: SouthStarPreparedMol,
    stereo_state: WriterStereoStateKey,
    tetra_by_center,
    directional_sites_by_bond: dict[BondId, tuple[SiteId, ...]],
) -> None:
    occurrence_vars: set[VarId] = set()
    delayed_vars: set[VarId] = set()
    for record in stereo_state.atom_occurrences:
        if record.var is None:
            continue
        occurrence_vars.add(record.var)
        template = tetra_by_center.get(record.atom)
        if template is None:
            _invalid_snapshot("writer tetra occurrence lacks prepared template")
        if not _has_delayed_factor(
            stereo_state,
            kind="tetra",
            site=template.site,
            var=record.var,
        ):
            _invalid_snapshot("writer tetra occurrence lacks delayed factor")
    for record in stereo_state.bond_occurrences:
        if record.var is None:
            continue
        occurrence_vars.add(record.var)
        for site in directional_sites_by_bond.get(record.bond, ()):
            if not _has_delayed_factor(
                stereo_state,
                kind="directional",
                site=site,
                var=record.var,
            ):
                _invalid_snapshot("writer directional occurrence lacks delayed factor")
    for factor in stereo_state.delayed_factors:
        for var in factor.scope:
            delayed_vars.add(var)
            if var not in occurrence_vars:
                _invalid_snapshot("writer delayed factor variable lacks occurrence record")
    assignment_vars = _residual_assignment_vars(stereo_state)
    domain_vars = _residual_domain_vars(stereo_state)
    if not assignment_vars.issubset(occurrence_vars):
        _invalid_snapshot("writer residual assignment lacks occurrence record")
    if not domain_vars.issubset(occurrence_vars):
        _invalid_snapshot("writer residual domain lacks occurrence record")
    if not occurrence_vars.issubset(delayed_vars):
        _invalid_snapshot("writer occurrence variable lacks delayed factor")
    factor_vars = frozenset(
        var
        for snapshot in stereo_state.residual_snapshot.factors
        for var in snapshot.scope
    )
    closed_delayed_vars = frozenset(
        var
        for factor in stereo_state.delayed_factors
        if factor.closed
        for var in factor.scope
    )
    if not factor_vars.issubset(closed_delayed_vars):
        _invalid_snapshot("writer residual factor variable lacks closed delayed factor")


def _has_delayed_factor(
    stereo_state: WriterStereoStateKey,
    *,
    kind: str,
    site: SiteId,
    var: VarId,
) -> bool:
    return any(
        factor.kind == kind
        and factor.site == site
        and var in factor.scope
        for factor in stereo_state.delayed_factors
    )


def _validate_delayed_factor_shape(
    prepared: SouthStarPreparedMol,
    stereo_state: WriterStereoStateKey,
    factor,
    tetra_by_site,
    directional_by_site,
) -> None:
    if factor.kind == "tetra":
        template = tetra_by_site.get(factor.site)
        if template is None:
            _invalid_snapshot("writer tetra delayed factor references unknown site")
        expected_var = tetra_var(("writer", int(template.site)))
        if factor.scope != (expected_var,):
            _invalid_snapshot("writer tetra delayed factor has unexpected scope")
        if factor.evidence != (("atom", int(template.center)),):
            _invalid_snapshot("writer tetra delayed factor has unexpected evidence")
        if factor.closed:
            record = _local_order_record(stereo_state, template.center)
            if record is None or not record.closed:
                _invalid_snapshot("writer closed tetra delayed factor lacks closed local order")
        else:
            record = _local_order_record(stereo_state, template.center)
            if record is not None and record.closed:
                _invalid_snapshot("writer pending tetra delayed factor is already complete")
        return
    if factor.kind == "directional":
        template = directional_by_site.get(factor.site)
        if template is None:
            _invalid_snapshot("writer directional delayed factor references unknown site")
        expected_scope, expected_evidence = _expected_directional_scope_and_evidence(
            prepared,
            stereo_state,
            template,
        )
        carrier_bonds = _directional_template_substituent_bonds(prepared, template)
        emitted_bonds = frozenset(bond for _, bond in expected_evidence)
        if factor.scope != expected_scope or factor.evidence != expected_evidence:
            _invalid_snapshot("writer directional delayed factor has unexpected scope/evidence")
        if factor.closed:
            if emitted_bonds != carrier_bonds:
                _invalid_snapshot("writer closed directional delayed factor is incomplete")
        elif emitted_bonds == carrier_bonds:
            _invalid_snapshot("writer pending directional delayed factor is already complete")
        return
    _invalid_snapshot("writer delayed factor has unknown kind")


def _expected_residual_factor_snapshot(
    prepared: SouthStarPreparedMol,
    stereo_state: WriterStereoStateKey,
    factor,
) -> object:
    assignments = dict(stereo_state.residual_snapshot.assignments)
    if factor.kind == "tetra":
        template = {item.site: item for item in prepared.tetra_templates}.get(factor.site)
        if template is None:
            _invalid_snapshot("writer tetra delayed factor references unknown site")
        record = _local_order_record(stereo_state, template.center)
        if record is None or not record.closed:
            _invalid_snapshot("writer closed tetra delayed factor lacks closed local order")
        expected_var = tetra_var(("writer", int(template.site)))
        expected = TetraResidualFactor(
            scope=(expected_var,),
            status=template.status,
            target=template.target,
            reference_order=template.reference_order,
            local_order=record.order,
        )
        if not expected.assign(expected_var, assignments.get(expected_var)):
            _invalid_snapshot("writer tetra residual assignment is invalid")
        if not expected.close():
            _invalid_snapshot("writer tetra residual factor is not closed")
        return expected.value_snapshot()
    if factor.kind == "directional":
        template = {item.site: item for item in prepared.directional_templates}.get(
            factor.site
        )
        if template is None:
            _invalid_snapshot("writer directional delayed factor references unknown site")
        models = _directional_models(prepared, template, stereo_state)
        expected = DirectionalResidualFactor(
            scope=tuple(sorted(models, key=_var_sort_tuple)),
            status=template.status,
            target=template.target,
            carrier_models=models,
        )
        for var in expected.scope:
            if not expected.assign(var, assignments.get(var)):
                _invalid_snapshot("writer directional residual assignment is invalid")
        if not expected.close():
            _invalid_snapshot("writer directional residual factor is not closed")
        return expected.value_snapshot()
    _invalid_snapshot("writer delayed factor has unknown kind")


def _expected_directional_scope_and_evidence(
    prepared: SouthStarPreparedMol,
    stereo_state: WriterStereoStateKey,
    template,
) -> tuple[tuple[VarId, ...], tuple[tuple[str, int], ...]]:
    carrier_bonds = _directional_template_substituent_bonds(prepared, template)
    emitted_bonds = tuple(
        record.bond
        for record in stereo_state.bond_occurrences
        if record.bond in carrier_bonds
    )
    scope = tuple(
        sorted(
            (direction_var(("writer", int(bond))) for bond in set(emitted_bonds)),
            key=_var_sort_tuple,
        )
    )
    evidence = tuple(sorted(("bond", int(bond)) for bond in set(emitted_bonds)))
    return scope, evidence


def _directional_models(
    prepared: SouthStarPreparedMol,
    template,
    stereo_state: WriterStereoStateKey,
) -> dict[VarId, DirectionalCarrierResidual]:
    bond_records = {record.bond: record for record in stereo_state.bond_occurrences}
    occurrence_by_id = {item.id: item for item in prepared.facts.ligand_occurrences}
    left_reference, right_reference = _directional_reference_pair(template)
    left_by_bond = _neighbor_ligands_by_bond(occurrence_by_id, template.left_ligands)
    right_by_bond = _neighbor_ligands_by_bond(occurrence_by_id, template.right_ligands)
    models: dict[VarId, DirectionalCarrierResidual] = {}
    for bond, occurrence in left_by_bond.items():
        record = bond_records.get(bond)
        if record is None:
            _invalid_snapshot("writer directional residual factor lacks carrier record")
        var = direction_var(("writer", int(bond)))
        models[var] = DirectionalCarrierResidual(
            var=var,
            side="left",
            orientation=_carrier_orientation(record, template.left_endpoint),
            ligand_factor=_ligand_factor(
                occurrence,
                reference=left_reference,
                side_ligands=template.left_ligands,
            ),
        )
    for bond, occurrence in right_by_bond.items():
        record = bond_records.get(bond)
        if record is None:
            _invalid_snapshot("writer directional residual factor lacks carrier record")
        var = direction_var(("writer", int(bond)))
        models[var] = DirectionalCarrierResidual(
            var=var,
            side="right",
            orientation=_carrier_orientation(record, template.right_endpoint),
            ligand_factor=_ligand_factor(
                occurrence,
                reference=right_reference,
                side_ligands=template.right_ligands,
            ),
        )
    return models


def _carrier_orientation(record, endpoint: AtomId) -> int:
    if record.parent == endpoint:
        return 1
    if record.child == endpoint:
        return -1
    _invalid_snapshot("writer directional carrier is not incident to endpoint")


def _ligand_factor(
    occurrence: OccurrenceId,
    *,
    reference: OccurrenceId,
    side_ligands: tuple[OccurrenceId, ...],
) -> int:
    if occurrence == reference:
        return 1
    if occurrence not in side_ligands:
        _invalid_snapshot("writer directional occurrence is not on template side")
    return -1


def _directional_reference_pair(template) -> tuple[OccurrenceId, OccurrenceId]:
    if template.reference_pair is not None:
        return template.reference_pair
    return (min(template.left_ligands, key=int), min(template.right_ligands, key=int))


def _neighbor_ligands_by_bond(
    occurrence_by_id,
    ligand_ids: tuple[OccurrenceId, ...],
) -> dict[BondId, OccurrenceId]:
    out = {}
    for ligand_id in ligand_ids:
        occurrence = occurrence_by_id[ligand_id]
        if occurrence.kind is not LigandKind.NEIGHBOR_ATOM:
            continue
        if occurrence.bond is None:
            _invalid_snapshot("writer directional neighbor occurrence lacks bond")
        out[occurrence.bond] = ligand_id
    return out


def _directional_template_substituent_bonds(
    prepared: SouthStarPreparedMol,
    template,
) -> frozenset[BondId]:
    occurrence_by_id = {item.id: item for item in prepared.facts.ligand_occurrences}
    bonds: set[BondId] = set()
    for occurrence_id in template.left_ligands + template.right_ligands:
        occurrence = occurrence_by_id[occurrence_id]
        if occurrence.kind is not LigandKind.NEIGHBOR_ATOM:
            continue
        if occurrence.bond is None:
            _invalid_snapshot("writer directional neighbor occurrence lacks bond")
        bonds.add(occurrence.bond)
    return frozenset(bonds)


def _directional_sites_by_carrier_bond(
    prepared: SouthStarPreparedMol,
) -> dict[BondId, tuple[SiteId, ...]]:
    by_bond: dict[BondId, list[SiteId]] = {}
    for template in prepared.directional_templates:
        for bond in _directional_template_substituent_bonds(prepared, template):
            by_bond.setdefault(bond, []).append(template.site)
    return {
        bond: tuple(sorted(sites, key=int))
        for bond, sites in by_bond.items()
    }


def _allowed_local_order_occurrences(
    prepared: SouthStarPreparedMol,
    atom: AtomId,
    template,
) -> frozenset[OccurrenceId]:
    if template is not None:
        return frozenset(template.ligand_occurrences)
    return frozenset(
        occurrence.id
        for occurrence in prepared.facts.ligand_occurrences
        if occurrence.kind is LigandKind.IMPLICIT_H and occurrence.atom == atom
    )


def _local_order_record(
    stereo_state: WriterStereoStateKey,
    atom: AtomId,
):
    for record in stereo_state.local_orders:
        if record.atom == atom:
            return record
    return None


def _has_bond_occurrence_record(
    stereo_state: WriterStereoStateKey,
    bond: BondId,
    parent: AtomId,
    child: AtomId,
) -> bool:
    return any(
        record.bond == bond
        and record.parent == parent
        and record.child == child
        for record in stereo_state.bond_occurrences
    )


def _reject_duplicate_items(items, message: str) -> None:
    seen = set()
    for item in items:
        if item in seen:
            _invalid_snapshot(message)
        seen.add(item)


def _var_sort_tuple(var: VarId) -> tuple[object, ...]:
    return (var.kind, tuple(_value_sort_tuple(item) for item in var.key))


def _value_sort_tuple(value: object) -> tuple[object, ...]:
    if isinstance(value, (int, str)):
        return (type(value).__name__, value)
    if isinstance(value, (TetraToken, DirectionMark)):
        return (value.__class__.__name__, value.value)
    if isinstance(value, tuple):
        return ("tuple", tuple(_value_sort_tuple(item) for item in value))
    return (value.__class__.__name__, str(value))


def _residual_domain_vars(stereo_state: WriterStereoStateKey) -> frozenset[VarId]:
    return frozenset(var for var, _ in stereo_state.residual_snapshot.domains)


def _residual_assignment_vars(stereo_state: WriterStereoStateKey) -> frozenset[VarId]:
    return frozenset(var for var, _ in stereo_state.residual_snapshot.assignments)


def _require_graph_bond(
    prepared: SouthStarPreparedMol,
    left: AtomId,
    right: AtomId,
    bond: BondId,
) -> None:
    actual = prepared.graph_index.bond_between.get((min(left, right), max(left, right)))
    if actual != bond:
        _invalid_snapshot("writer state contains graph-invalid atom/bond triple")


def _atom_component_index(prepared: SouthStarPreparedMol) -> dict[AtomId, int]:
    out: dict[AtomId, int] = {}
    for index, component in enumerate(prepared.facts.components):
        for atom in component.atoms:
            out[atom] = index
    return out


def _bond_component_index(prepared: SouthStarPreparedMol) -> dict[BondId, int]:
    out: dict[BondId, int] = {}
    for index, component in enumerate(prepared.facts.components):
        for bond in component.bonds:
            out[bond] = index
    return out


def _invalid_snapshot(message: str) -> None:
    raise SouthStarError(SouthStarErrorKind.INTERNAL_INVARIANT, message)


def _prepared_identity(
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions,
) -> WriterPreparedIdentity:
    return WriterPreparedIdentity(
        runtime=(
            runtime_options.serialization_language.value,
            runtime_options.rooted_at_atom,
            runtime_options.canonical,
            runtime_options.do_random,
        ),
        atoms=tuple(
            (
                int(atom.id),
                atom.atomic_num,
                atom.symbol,
                atom.isotope,
                atom.formal_charge,
                atom.is_aromatic,
                atom.explicit_h_count,
                atom.implicit_h_count,
                atom.no_implicit,
            )
            for atom in prepared.facts.atoms
        ),
        bonds=tuple(
            (
                int(bond.id),
                int(bond.a),
                int(bond.b),
                bond.order.value,
                bond.is_aromatic,
                bond.is_conjugated,
            )
            for bond in prepared.facts.bonds
        ),
        components=tuple(
            (
                int(component.id),
                tuple(int(atom) for atom in component.atoms),
                tuple(int(bond) for bond in component.bonds),
            )
            for component in prepared.facts.components
        ),
        ligand_occurrences=tuple(
            (
                int(occurrence.id),
                int(occurrence.site),
                occurrence.kind.value,
                None if occurrence.atom is None else int(occurrence.atom),
                None if occurrence.bond is None else int(occurrence.bond),
                occurrence.ordinal,
            )
            for occurrence in prepared.facts.ligand_occurrences
        ),
        tetra_templates=tuple(
            (
                int(template.site),
                int(template.center),
                template.status.value,
                template.target.value,
                tuple(int(item) for item in template.reference_order),
                tuple(int(item) for item in template.ligand_occurrences),
            )
            for template in prepared.tetra_templates
        ),
        directional_templates=tuple(
            (
                int(template.site),
                int(template.center_bond),
                int(template.left_endpoint),
                int(template.right_endpoint),
                template.status.value,
                template.target.value,
                tuple(int(item) for item in template.left_ligands),
                tuple(int(item) for item in template.right_ligands),
                None
                if template.reference_pair is None
                else tuple(int(item) for item in template.reference_pair),
            )
            for template in prepared.directional_templates
        ),
        policy=(
            tuple(int(label.value) for label in prepared.policy.ring_labels),
            prepared.policy.annotation_mode.value,
            prepared.policy.least_free_ring_labels,
            tuple(
                (
                    int(domain.atom),
                    tuple(
                        (
                            choice.name,
                            tuple((token.value, text) for token, text in choice.text_by_tetra),
                        )
                        for choice in domain.choices
                    ),
                )
                for domain in prepared.policy.atom_text_domains
            ),
            tuple(
                (
                    int(domain.bond),
                    domain.slot_kind,
                    tuple(
                        (choice.name, choice.base_text, choice.permits_direction)
                        for choice in domain.choices
                    ),
                )
                for domain in prepared.policy.bond_text_domains
            ),
        ),
    )


__all__ = (
    "WriterDecoderBoundary",
    "WriterFrontierFrame",
    "WriterPreparedIdentity",
    "WriterSearchSnapshot",
    "WriterSnapshotFrame",
    "capture_writer_frontier_snapshot",
    "resume_writer_frontier_choices_from_snapshot",
    "validate_writer_cursor_against_prepared",
    "validate_writer_search_snapshot",
    "writer_frontier_cursor_from_snapshot",
)
