"""Writer-shaped frontier snapshots."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .policy import SerializationLanguageMode
from .prepared_runtime import SouthStarPreparedMol
from .prepared_runtime import SouthStarRuntimeOptions
from .prepared_runtime import require_writer_shaped_runtime_options
from .residual_constraints import ResidualStore
from .residual_constraints import ResidualStoreValueSnapshot
from .writer_events import WriterEvent
from .writer_frontier import WriterFrontierChoices
from .writer_frontier import WriterFrontierCursor
from .writer_frontier import writer_frontier_choices
from .writer_stereo import WriterDelayedStereoFactor


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


@dataclass(frozen=True, slots=True)
class WriterTransitionFrame:
    events: tuple[WriterEvent, ...]


@dataclass(frozen=True, slots=True)
class WriterStereoResidualFrame:
    residual_snapshot: ResidualStoreValueSnapshot


@dataclass(frozen=True, slots=True)
class WriterDelayedFactorFrame:
    delayed_factors: tuple[WriterDelayedStereoFactor, ...]


WriterSnapshotFrame = (
    WriterFrontierFrame
    | WriterTransitionFrame
    | WriterStereoResidualFrame
    | WriterDelayedFactorFrame
)


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
    _validate_cursor_residual_snapshots(snapshot.cursor)


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


def _validate_cursor_residual_snapshots(cursor: WriterFrontierCursor) -> None:
    for key, _ in cursor.weighted_states:
        _round_trip_residual_snapshot(key.stereo_state.residual_snapshot)


def _round_trip_residual_snapshot(snapshot: ResidualStoreValueSnapshot) -> None:
    round_tripped = ResidualStore.from_value_snapshot(snapshot).value_snapshot()
    if round_tripped != snapshot:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "writer residual snapshot does not round-trip",
        )


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
    "WriterDelayedFactorFrame",
    "WriterFrontierFrame",
    "WriterPreparedIdentity",
    "WriterSearchSnapshot",
    "WriterSnapshotFrame",
    "WriterStereoResidualFrame",
    "WriterTransitionFrame",
    "capture_writer_frontier_snapshot",
    "resume_writer_frontier_choices_from_snapshot",
    "validate_writer_search_snapshot",
    "writer_frontier_cursor_from_snapshot",
)
