"""Certificates for materialized writer support strings and images."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import SouthStarError
from .errors import SouthStarErrorKind


@dataclass(frozen=True, slots=True)
class WriterSupportStringCertificate:
    string: str
    emitted_texts: tuple[str, ...]
    replay_certificate: object
    final_snapshot: object
    terminal_projection_certificate: object
    terminal_certificates: tuple[object, ...]
    text_projection_certificates: tuple[object, ...] = ()


@dataclass(frozen=True, slots=True)
class WriterFrontierSupportStringCertificate:
    source_cursor: object
    string: str
    emitted_texts: tuple[str, ...]
    text_projection_certificates: tuple[object, ...]
    terminal_projection_certificate: object
    terminal_certificates: tuple[object, ...]
    final_cursor: object


@dataclass(frozen=True, slots=True)
class WriterSupportImageCertificate:
    source_snapshot: object
    strings: tuple[str, ...]
    string_certificates: tuple[WriterSupportStringCertificate, ...]
    distinct_count: int
    witness_count: int
    support_count_certificate: object | None = None
    witness_count_certificate: object | None = None


def writer_support_string_certificate(
    *,
    source_snapshot,
    string: str,
    emitted_texts: tuple[str, ...],
    replay_certificate,
    terminal_projection_certificate,
    text_projection_certificates: tuple[object, ...] = (),
) -> WriterSupportStringCertificate:
    if string != "".join(emitted_texts):
        _string_violation("string_emitted_texts_mismatch")
    if replay_certificate.source_snapshot != source_snapshot:
        _string_violation("replay_source_snapshot_mismatch")
    if replay_certificate.emitted_texts != emitted_texts:
        _string_violation("replay_emitted_texts_mismatch")
    if terminal_projection_certificate is None:
        _string_violation("missing_terminal_projection_certificate")
    if terminal_projection_certificate.terminal is None:
        _string_violation("terminal_projection_lacks_terminal")
    if not terminal_projection_certificate.terminal_certificates:
        _string_violation("terminal_projection_lacks_certificates")
    if terminal_projection_certificate.terminal.finalized_cursor != (
        terminal_projection_certificate.finalized_cursor
    ):
        _string_violation("terminal_finalized_cursor_mismatch")
    if terminal_projection_certificate.source_cursor != (
        replay_certificate.final_snapshot.cursor
    ):
        _string_violation("terminal_projection_final_cursor_mismatch")
    if len(text_projection_certificates) != len(emitted_texts):
        _string_violation("support_string_projection_count_mismatch")
    if tuple(
        certificate.emitted_text
        for certificate in text_projection_certificates
    ) != emitted_texts:
        _string_violation("support_string_projection_text_mismatch")
    if replay_certificate.emitted_texts != emitted_texts:
        _string_violation("support_string_replay_text_mismatch")
    if tuple(
        id(step.text_projection_certificate)
        for step in replay_certificate.step_certificates
    ) != tuple(id(certificate) for certificate in text_projection_certificates):
        _string_violation(
            "support_string_replay_projection_chain_mismatch"
        )

    return WriterSupportStringCertificate(
        string=string,
        emitted_texts=emitted_texts,
        replay_certificate=replay_certificate,
        final_snapshot=replay_certificate.final_snapshot,
        terminal_projection_certificate=terminal_projection_certificate,
        terminal_certificates=terminal_projection_certificate.terminal_certificates,
        text_projection_certificates=text_projection_certificates,
    )


def writer_frontier_support_string_certificate(
    *,
    source_cursor,
    string: str,
    emitted_texts: tuple[str, ...],
    text_projection_certificates: tuple[object, ...],
    terminal_projection_certificate,
) -> WriterFrontierSupportStringCertificate:
    if string != "".join(emitted_texts):
        _string_violation("frontier_string_text_mismatch")
    if len(emitted_texts) != len(text_projection_certificates):
        _string_violation("frontier_projection_count_mismatch")

    current = source_cursor
    for emitted_text, projection in zip(
        emitted_texts,
        text_projection_certificates,
    ):
        if projection.emitted_text != emitted_text:
            _string_violation("frontier_projection_text_mismatch")
        if projection.source_cursor != current:
            _string_violation("frontier_projection_source_cursor_mismatch")
        if projection.choice.successor != projection.successor_cursor:
            _string_violation("frontier_projection_successor_mismatch")
        current = projection.successor_cursor

    if terminal_projection_certificate is None:
        _string_violation("frontier_support_lacks_terminal_projection")
    if terminal_projection_certificate.terminal is None:
        _string_violation("frontier_terminal_projection_lacks_terminal")
    if not terminal_projection_certificate.terminal_certificates:
        _string_violation("frontier_terminal_projection_lacks_certificates")
    if terminal_projection_certificate.source_cursor != current:
        _string_violation(
            "frontier_terminal_projection_source_cursor_mismatch"
        )

    return WriterFrontierSupportStringCertificate(
        source_cursor=source_cursor,
        string=string,
        emitted_texts=emitted_texts,
        text_projection_certificates=text_projection_certificates,
        terminal_projection_certificate=terminal_projection_certificate,
        terminal_certificates=terminal_projection_certificate.terminal_certificates,
        final_cursor=current,
    )


def writer_support_image_certificate(
    *,
    source_snapshot,
    string_certificates: tuple[WriterSupportStringCertificate, ...],
    witness_count: int,
    support_count_certificate: object | None = None,
    witness_count_certificate: object | None = None,
) -> WriterSupportImageCertificate:
    strings = tuple(certificate.string for certificate in string_certificates)
    if len(set(strings)) != len(strings):
        _image_violation("duplicate_support_string_certificate")
    for certificate in string_certificates:
        if certificate.replay_certificate.source_snapshot != source_snapshot:
            _image_violation("string_certificate_source_snapshot_mismatch")

    if support_count_certificate is not None:
        if getattr(support_count_certificate, "source_snapshot", None) != (
            source_snapshot
        ):
            _image_violation("support_count_source_snapshot_mismatch")
        if support_count_certificate.support_count != len(strings):
            _image_violation("support_count_certificate_mismatch")

    if witness_count_certificate is not None:
        source_cursor = getattr(source_snapshot, "cursor", None)
        if (
            source_cursor is not None
            and witness_count_certificate.cursor != source_cursor
        ):
            _image_violation("witness_count_cursor_mismatch")
        if witness_count_certificate.completion_count != witness_count:
            _image_violation("witness_count_certificate_mismatch")

    return WriterSupportImageCertificate(
        source_snapshot=source_snapshot,
        strings=strings,
        string_certificates=string_certificates,
        distinct_count=len(strings),
        witness_count=witness_count,
        support_count_certificate=support_count_certificate,
        witness_count_certificate=witness_count_certificate,
    )


def _string_violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer support string certificate violation: {kind}",
    )


def _image_violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer support image certificate violation: {kind}",
    )


__all__ = (
    "WriterFrontierSupportStringCertificate",
    "WriterSupportImageCertificate",
    "WriterSupportStringCertificate",
    "writer_frontier_support_string_certificate",
    "writer_support_image_certificate",
    "writer_support_string_certificate",
)
