"""Writer-shaped frontier snapshots."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .facts import LigandKind
from .ids import AtomId
from .ids import BondId
from .policy import SerializationLanguageMode
from .prepared_runtime import SouthStarPreparedMol
from .prepared_runtime import SouthStarRuntimeOptions
from .prepared_runtime import require_writer_shaped_runtime_options
from .residual_constraints import DirectionalResidualFactorValueSnapshot
from .residual_constraints import ResidualStore
from .residual_constraints import ResidualStoreValueSnapshot
from .residual_constraints import TetraResidualFactorValueSnapshot
from .residual_constraints import VarId
from .writer_frontier import WriterFrontierChoices
from .writer_frontier import WriterFrontierCursor
from .writer_frontier import writer_frontier_choices
from .writer_state import ComponentCursor
from .writer_state import ObligationStateKey
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
    for key, weight in cursor.weighted_states:
        if weight <= 0:
            _invalid_snapshot("writer cursor contains nonpositive weight")
        _validate_component_cursor(key.component_cursor, allowed_roots)
        _validate_atom_frame(key.active, atom_ids, bond_ids, prepared)
        for frame in key.branch_stack:
            _validate_branch_frame(frame, atom_ids, bond_ids, prepared)
        _validate_known_atoms("visited_atoms", key.visited_atoms, atom_ids)
        _validate_known_bonds("written_bonds", key.written_bonds, bond_ids)
        _validate_active_coherence(key)
        _validate_written_bond_coherence(prepared, key)
        _validate_obligations(key.obligations, key, atom_ids, bond_ids, prepared)
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
        if key.obligations.pending_entry is not None or key.branch_stack:
            _invalid_snapshot("writer inactive state has active obligations")
        return
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
) -> None:
    pending = obligations.pending_entry
    if pending is None:
        return
    _validate_pending_entry(pending, atom_ids, bond_ids, prepared)
    if key.active is None or key.active.atom != pending.parent:
        _invalid_snapshot("writer pending entry parent is not active")
    if not key.active.atom_emitted:
        _invalid_snapshot("writer pending entry parent is not emitted")
    if pending.parent not in key.visited_atoms:
        _invalid_snapshot("writer pending parent is not visited")
    if pending.child in key.visited_atoms or pending.bond in key.written_bonds:
        _invalid_snapshot("writer pending entry is already written")


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
    occurrence_by_id = {item.id: item for item in prepared.facts.ligand_occurrences}
    atom_ids = frozenset(prepared.atom_ids)
    bond_ids = frozenset(bond.id for bond in prepared.facts.bonds)
    _validate_atom_occurrence_records(stereo_state, atom_ids)
    _validate_bond_occurrence_records(stereo_state, atom_ids, bond_ids, prepared)
    _validate_local_order_records(stereo_state, occurrence_by_id, atom_ids)
    _validate_delayed_factor_records(stereo_state)


def _validate_atom_occurrence_records(
    stereo_state: WriterStereoStateKey,
    atom_ids: frozenset[AtomId],
) -> None:
    domain_vars = _residual_domain_vars(stereo_state)
    for record in stereo_state.atom_occurrences:
        if record.atom not in atom_ids:
            _invalid_snapshot("writer atom occurrence references unknown atom")
        if record.var is not None and record.var not in domain_vars:
            _invalid_snapshot("writer atom occurrence variable is missing from residual store")


def _validate_bond_occurrence_records(
    stereo_state: WriterStereoStateKey,
    atom_ids: frozenset[AtomId],
    bond_ids: frozenset[BondId],
    prepared: SouthStarPreparedMol,
) -> None:
    domain_vars = _residual_domain_vars(stereo_state)
    for record in stereo_state.bond_occurrences:
        if record.bond not in bond_ids or record.parent not in atom_ids or record.child not in atom_ids:
            _invalid_snapshot("writer bond occurrence references unknown graph item")
        _require_graph_bond(prepared, record.parent, record.child, record.bond)
        if record.var is not None and record.var not in domain_vars:
            _invalid_snapshot("writer bond occurrence variable is missing from residual store")


def _validate_local_order_records(
    stereo_state: WriterStereoStateKey,
    occurrence_by_id,
    atom_ids: frozenset[AtomId],
) -> None:
    for record in stereo_state.local_orders:
        if record.atom not in atom_ids:
            _invalid_snapshot("writer local-order record references unknown atom")
        if len(set(record.order)) != len(record.order):
            _invalid_snapshot("writer local-order record repeats ligand occurrence")
        for occurrence_id in record.order:
            occurrence = occurrence_by_id.get(occurrence_id)
            if occurrence is None:
                _invalid_snapshot("writer local-order record references unknown ligand occurrence")
            if occurrence.kind is LigandKind.IMPLICIT_H:
                if occurrence.atom != record.atom:
                    _invalid_snapshot("writer local-order implicit-H occurrence is on another atom")
            elif occurrence.kind is LigandKind.NEIGHBOR_ATOM:
                if occurrence.atom not in atom_ids:
                    _invalid_snapshot("writer local-order neighbor occurrence references unknown atom")
            else:
                _invalid_snapshot("writer local-order pseudo occurrence is unsupported")


def _validate_delayed_factor_records(stereo_state: WriterStereoStateKey) -> None:
    domain_vars = _residual_domain_vars(stereo_state)
    assignment_vars = _residual_assignment_vars(stereo_state)
    factor_snapshots = stereo_state.residual_snapshot.factors
    for factor in stereo_state.delayed_factors:
        if not factor.scope:
            _invalid_snapshot("writer delayed factor has empty scope")
        for var in factor.scope:
            if var not in domain_vars:
                _invalid_snapshot("writer delayed factor variable is missing from residual store")
            if var not in assignment_vars:
                _invalid_snapshot("writer delayed factor variable is unassigned")
        if factor.closed and not _has_matching_residual_factor(factor, factor_snapshots):
            _invalid_snapshot("writer closed delayed factor lacks residual factor snapshot")
    for snapshot in factor_snapshots:
        if not _has_matching_closed_delayed_factor(snapshot, stereo_state.delayed_factors):
            _invalid_snapshot("writer residual factor snapshot lacks closed delayed factor")


def _has_matching_residual_factor(
    factor,
    snapshots: tuple[object, ...],
) -> bool:
    scope = frozenset(factor.scope)
    if factor.kind == "tetra":
        return any(
            isinstance(snapshot, TetraResidualFactorValueSnapshot)
            and frozenset(snapshot.scope) == scope
            for snapshot in snapshots
        )
    if factor.kind == "directional":
        return any(
            isinstance(snapshot, DirectionalResidualFactorValueSnapshot)
            and frozenset(snapshot.scope) == scope
            for snapshot in snapshots
        )
    _invalid_snapshot("writer delayed factor has unknown kind")


def _has_matching_closed_delayed_factor(
    snapshot: object,
    factors,
) -> bool:
    if isinstance(snapshot, TetraResidualFactorValueSnapshot):
        kind = "tetra"
    elif isinstance(snapshot, DirectionalResidualFactorValueSnapshot):
        kind = "directional"
    else:
        _invalid_snapshot("writer residual snapshot has unknown factor type")
    scope = frozenset(snapshot.scope)
    return any(
        factor.kind == kind
        and factor.closed
        and frozenset(factor.scope) == scope
        for factor in factors
    )


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
