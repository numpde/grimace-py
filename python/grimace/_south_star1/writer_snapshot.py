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
    atom_ids: tuple[int, ...]
    component_ids: tuple[int, ...]
    bond_ids: tuple[int, ...]


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
        prepared_identity=_prepared_identity(prepared),
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
    if snapshot.prepared_identity != _prepared_identity(prepared):
        raise SouthStarError(
            SouthStarErrorKind.INVALID_FACTS,
            "writer snapshot prepared identity does not match prepared molecule",
        )
    if snapshot.cursor != WriterFrontierCursor(snapshot.cursor.weighted_states):
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "writer snapshot cursor is not canonical",
        )
    _validate_frames(snapshot.frame_stack)
    _validate_cursor_residual_snapshots(snapshot.cursor)


def _validate_frames(frame_stack: tuple[object, ...]) -> None:
    if not frame_stack:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "writer snapshot frame stack must include a frontier frame",
        )
    has_frontier = False
    for frame in frame_stack:
        if isinstance(frame, WriterFrontierFrame):
            has_frontier = True
            continue
        if isinstance(frame, WriterTransitionFrame):
            if not frame.events:
                raise SouthStarError(
                    SouthStarErrorKind.INTERNAL_INVARIANT,
                    "writer transition snapshot frame must carry events",
                )
            continue
        if isinstance(frame, WriterStereoResidualFrame):
            _round_trip_residual_snapshot(frame.residual_snapshot)
            continue
        if isinstance(frame, WriterDelayedFactorFrame):
            continue
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            f"unknown writer snapshot frame payload: {type(frame).__name__}",
        )
    if not has_frontier:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "writer snapshot cannot contain only context frames",
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


def _prepared_identity(prepared: SouthStarPreparedMol) -> WriterPreparedIdentity:
    return WriterPreparedIdentity(
        atom_ids=tuple(int(atom) for atom in prepared.atom_ids),
        component_ids=tuple(int(component) for component in prepared.component_ids),
        bond_ids=tuple(int(bond.id) for bond in prepared.facts.bonds),
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
