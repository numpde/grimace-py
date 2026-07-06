"""Certificates for replaying emitted text through writer snapshots."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import SouthStarError
from .errors import SouthStarErrorKind


@dataclass(frozen=True, slots=True)
class WriterSnapshotStepCertificate:
    source_snapshot: object
    emitted_text: str
    frontier_projection_certificate: object
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
    frontier_projection_certificates: tuple[object, ...] = ()


@dataclass(frozen=True, slots=True)
class WriterSnapshotPrefixReadCertificate:
    source_snapshot: object
    emitted_texts: tuple[str, ...]
    replay_certificate: object
    final_snapshot: object
    final_frontier_projection_certificate: object
    checked_frontier_certificate: object | None
    support_count_certificate: object | None
    completion_count_certificate: object | None
    support_count: int | None
    completion_count: int | None


def writer_snapshot_step_certificate(
    *,
    source_snapshot,
    emitted_text: str,
    frontier_projection_certificate,
    text_projection_certificate,
    advanced_snapshot,
) -> WriterSnapshotStepCertificate:
    if frontier_projection_certificate is None:
        _step_violation("missing_frontier_projection_certificate")
    if frontier_projection_certificate.cursor != source_snapshot.cursor:
        _step_violation("frontier_projection_cursor_mismatch")
    if not any(
        projection is text_projection_certificate
        for projection in (
            frontier_projection_certificate
            .text_choice_projection_certificates
        )
    ):
        _step_violation("text_projection_not_in_frontier_projection")
    if not emitted_text:
        _step_violation("missing_emitted_text")
    if text_projection_certificate.emitted_text != emitted_text:
        _step_violation("emitted_text_mismatch")
    if text_projection_certificate.source_cursor != source_snapshot.cursor:
        _step_violation("projection_source_cursor_mismatch")
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
        frontier_projection_certificate=frontier_projection_certificate,
        text_projection_certificate=text_projection_certificate,
        source_cursor=text_projection_certificate.source_cursor,
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
        if step.source_cursor != current.cursor:
            _replay_violation("step_source_cursor_mismatch")
        if step.frontier_projection_certificate.cursor != current.cursor:
            _replay_violation("step_frontier_projection_cursor_mismatch")
        if step.successor_cursor != step.advanced_snapshot.cursor:
            _replay_violation("step_successor_cursor_mismatch")
        current = step.advanced_snapshot

    if final_snapshot != current:
        _replay_violation("final_snapshot_mismatch")

    return WriterSnapshotReplayCertificate(
        source_snapshot=source_snapshot,
        emitted_texts=emitted_texts,
        step_certificates=step_certificates,
        final_snapshot=final_snapshot,
        frontier_projection_certificates=tuple(
            step.frontier_projection_certificate
            for step in step_certificates
        ),
    )


def writer_snapshot_prefix_read_certificate(
    *,
    source_snapshot,
    emitted_texts: tuple[str, ...],
    replay_certificate,
    final_snapshot,
    final_frontier_product,
) -> WriterSnapshotPrefixReadCertificate:
    if replay_certificate.source_snapshot != source_snapshot:
        _prefix_violation("replay_source_snapshot_mismatch")
    if replay_certificate.emitted_texts != emitted_texts:
        _prefix_violation("replay_emitted_texts_mismatch")
    if replay_certificate.final_snapshot != final_snapshot:
        _prefix_violation("replay_final_snapshot_mismatch")
    if final_frontier_product.cursor != final_snapshot.cursor:
        _prefix_violation("frontier_product_cursor_mismatch")

    projection = final_frontier_product.projection_certificate
    if projection is None:
        _prefix_violation("missing_final_frontier_projection_certificate")
    if projection.cursor != final_snapshot.cursor:
        _prefix_violation("final_projection_cursor_mismatch")

    checked = final_frontier_product.checked_frontier_certificate
    support_count_certificate = final_frontier_product.support_count_certificate
    count_certificate = final_frontier_product.count_certificate
    if checked is not None:
        if checked.cursor != final_snapshot.cursor:
            _prefix_violation("checked_frontier_cursor_mismatch")
        if checked.projection_certificate is not projection:
            _prefix_violation("checked_frontier_projection_mismatch")
        if checked.support_count_certificate is not support_count_certificate:
            _prefix_violation("checked_frontier_support_count_mismatch")
        if checked.count_certificate is not count_certificate:
            _prefix_violation("checked_frontier_count_mismatch")

    support_count = (
        None
        if support_count_certificate is None
        else support_count_certificate.support_count
    )
    completion_count = (
        None if count_certificate is None else count_certificate.completion_count
    )
    return WriterSnapshotPrefixReadCertificate(
        source_snapshot=source_snapshot,
        emitted_texts=emitted_texts,
        replay_certificate=replay_certificate,
        final_snapshot=final_snapshot,
        final_frontier_projection_certificate=projection,
        checked_frontier_certificate=checked,
        support_count_certificate=support_count_certificate,
        completion_count_certificate=count_certificate,
        support_count=support_count,
        completion_count=completion_count,
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


def _prefix_violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer snapshot prefix read certificate violation: {kind}",
    )


__all__ = (
    "WriterSnapshotPrefixReadCertificate",
    "WriterSnapshotReplayCertificate",
    "WriterSnapshotStepCertificate",
    "writer_snapshot_prefix_read_certificate",
    "writer_snapshot_replay_certificate",
    "writer_snapshot_step_certificate",
)
