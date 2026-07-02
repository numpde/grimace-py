"""Certificates for replaying emitted text through writer snapshots."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import SouthStarError
from .errors import SouthStarErrorKind


@dataclass(frozen=True, slots=True)
class WriterSnapshotStepCertificate:
    source_snapshot: object
    emitted_text: str
    text_projection_certificate: object
    source_cursor: object
    successor_cursor: object
    advanced_snapshot: object
    decoder_boundary_before: object
    decoder_boundary_after: object
    branch_certificates: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class WriterSnapshotReplayCertificate:
    source_snapshot: object
    emitted_texts: tuple[str, ...]
    step_certificates: tuple[WriterSnapshotStepCertificate, ...]
    final_snapshot: object


def writer_snapshot_step_certificate(
    *,
    source_snapshot,
    emitted_text: str,
    text_projection_certificate,
    advanced_snapshot,
) -> WriterSnapshotStepCertificate:
    if not emitted_text:
        _step_violation("missing_emitted_text")
    if text_projection_certificate.emitted_text != emitted_text:
        _step_violation("emitted_text_mismatch")
    if not text_projection_certificate.branch_certificates:
        _step_violation("missing_branch_certificates")
    if advanced_snapshot.cursor != text_projection_certificate.successor_cursor:
        _step_violation("successor_cursor_mismatch")

    expected_count = (
        source_snapshot.decoder_boundary.consumed_token_count + 1
    )
    if advanced_snapshot.decoder_boundary.consumed_token_count != expected_count:
        _step_violation("decoder_boundary_mismatch")

    frame_cursors = tuple(
        frame.cursor
        for frame in advanced_snapshot.frame_stack
    )
    if frame_cursors != (text_projection_certificate.successor_cursor,):
        _step_violation("frame_stack_cursor_mismatch")

    return WriterSnapshotStepCertificate(
        source_snapshot=source_snapshot,
        emitted_text=emitted_text,
        text_projection_certificate=text_projection_certificate,
        source_cursor=source_snapshot.cursor,
        successor_cursor=text_projection_certificate.successor_cursor,
        advanced_snapshot=advanced_snapshot,
        decoder_boundary_before=source_snapshot.decoder_boundary,
        decoder_boundary_after=advanced_snapshot.decoder_boundary,
        branch_certificates=text_projection_certificate.branch_certificates,
    )


def writer_snapshot_replay_certificate(
    *,
    source_snapshot,
    emitted_texts: tuple[str, ...],
    step_certificates: tuple[WriterSnapshotStepCertificate, ...],
    final_snapshot,
) -> WriterSnapshotReplayCertificate:
    if len(step_certificates) != len(emitted_texts):
        _replay_violation("step_count_mismatch")

    current = source_snapshot
    for index, step in enumerate(step_certificates):
        if step.emitted_text != emitted_texts[index]:
            _replay_violation("step_emitted_text_mismatch")
        if step.source_snapshot != current:
            _replay_violation("step_source_snapshot_mismatch")
        current = step.advanced_snapshot

    if final_snapshot != current:
        _replay_violation("final_snapshot_mismatch")

    return WriterSnapshotReplayCertificate(
        source_snapshot=source_snapshot,
        emitted_texts=emitted_texts,
        step_certificates=step_certificates,
        final_snapshot=final_snapshot,
    )


def _step_violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer snapshot step certificate violation: {kind}",
    )


def _replay_violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer snapshot replay certificate violation: {kind}",
    )


__all__ = (
    "WriterSnapshotReplayCertificate",
    "WriterSnapshotStepCertificate",
    "writer_snapshot_replay_certificate",
    "writer_snapshot_step_certificate",
)
