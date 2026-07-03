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


@dataclass(frozen=True, slots=True)
class WriterSupportImageCertificate:
    source_snapshot: object
    strings: tuple[str, ...]
    string_certificates: tuple[WriterSupportStringCertificate, ...]
    distinct_count: int
    witness_count: int
    witness_count_certificate: object | None = None


def writer_support_string_certificate(
    *,
    source_snapshot,
    string: str,
    emitted_texts: tuple[str, ...],
    replay_certificate,
    terminal_projection_certificate,
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

    return WriterSupportStringCertificate(
        string=string,
        emitted_texts=emitted_texts,
        replay_certificate=replay_certificate,
        final_snapshot=replay_certificate.final_snapshot,
        terminal_projection_certificate=terminal_projection_certificate,
        terminal_certificates=terminal_projection_certificate.terminal_certificates,
    )


def writer_support_image_certificate(
    *,
    source_snapshot,
    string_certificates: tuple[WriterSupportStringCertificate, ...],
    witness_count: int,
    witness_count_certificate: object | None = None,
) -> WriterSupportImageCertificate:
    strings = tuple(certificate.string for certificate in string_certificates)
    if len(set(strings)) != len(strings):
        _image_violation("duplicate_support_string_certificate")

    if witness_count_certificate is not None:
        if witness_count_certificate.completion_count != witness_count:
            _image_violation("witness_count_certificate_mismatch")

    return WriterSupportImageCertificate(
        source_snapshot=source_snapshot,
        strings=strings,
        string_certificates=string_certificates,
        distinct_count=len(strings),
        witness_count=witness_count,
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
    "WriterSupportImageCertificate",
    "WriterSupportStringCertificate",
    "writer_support_image_certificate",
    "writer_support_string_certificate",
)
