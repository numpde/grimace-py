"""Support-count certificates for distinct writer-frontier strings."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import SouthStarError
from .errors import SouthStarErrorKind


@dataclass(frozen=True, slots=True)
class WriterTextChoiceSupportCountTermCertificate:
    text_projection_certificate: object
    successor_support_count_certificate: object
    support_count: int


@dataclass(frozen=True, slots=True)
class WriterTextStateSupportCountCertificate:
    cursor: object
    terminal_projection_certificate: object | None
    terminal_count: int
    choice_terms: tuple[WriterTextChoiceSupportCountTermCertificate, ...]
    support_count: int


@dataclass(frozen=True, slots=True)
class WriterTextSupportCountCertificate:
    source_snapshot: object
    cursor: object
    state_support_count_certificate: WriterTextStateSupportCountCertificate
    support_count: int



def writer_text_choice_support_count_term_certificate(
    *,
    text_projection_certificate,
    successor_support_count_certificate,
) -> WriterTextChoiceSupportCountTermCertificate:
    if text_projection_certificate is None:
        _support_count_violation("missing_text_projection_certificate")
    if successor_support_count_certificate is None:
        _support_count_violation("missing_successor_support_count_certificate")

    successor_cursor = getattr(
        text_projection_certificate,
        "successor_cursor",
        None,
    )
    if successor_cursor is None:
        _support_count_violation("missing_projection_successor_cursor")
    if successor_support_count_certificate.cursor != successor_cursor:
        _support_count_violation("projection_successor_cursor_mismatch")

    support_count = getattr(
        successor_support_count_certificate,
        "support_count",
        None,
    )
    if support_count is None:
        _support_count_violation("successor_support_count_missing")
    if support_count < 0:
        _support_count_violation("negative_successor_support_count")

    return WriterTextChoiceSupportCountTermCertificate(
        text_projection_certificate=text_projection_certificate,
        successor_support_count_certificate=successor_support_count_certificate,
        support_count=support_count,
    )



def writer_text_state_support_count_certificate(
    *,
    cursor,
    terminal_projection_certificate,
    terminal_count: int,
    choice_terms: tuple[WriterTextChoiceSupportCountTermCertificate, ...],
) -> WriterTextStateSupportCountCertificate:
    if terminal_count < 0:
        _support_count_violation("negative_terminal_count")

    if terminal_projection_certificate is None and terminal_count != 0:
        _support_count_violation("terminal_count_without_terminal_projection")

    if terminal_projection_certificate is not None and terminal_count != 1:
        _support_count_violation("terminal_count_mismatch")

    if terminal_projection_certificate is not None:
        terminal = terminal_projection_certificate.terminal
        if terminal is None:
            _support_count_violation("terminal_projection_lacks_terminal")

    for term in choice_terms:
        if term is None:
            _support_count_violation("missing_choice_term")

    support_count = terminal_count + sum(term.support_count for term in choice_terms)
    if support_count < 0:
        _support_count_violation("negative_support_count")

    return WriterTextStateSupportCountCertificate(
        cursor=cursor,
        terminal_projection_certificate=terminal_projection_certificate,
        terminal_count=terminal_count,
        choice_terms=choice_terms,
        support_count=support_count,
    )



def writer_text_support_count_certificate(
    *,
    source_snapshot,
    cursor,
    state_support_count_certificate,
) -> WriterTextSupportCountCertificate:
    if state_support_count_certificate is None:
        _support_count_violation("missing_state_support_count_certificate")

    if state_support_count_certificate.cursor != cursor:
        _support_count_violation("state_support_count_cursor_mismatch")

    return WriterTextSupportCountCertificate(
        source_snapshot=source_snapshot,
        cursor=cursor,
        state_support_count_certificate=state_support_count_certificate,
        support_count=state_support_count_certificate.support_count,
    )



def _support_count_violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer support count certificate violation: {kind}",
    )


__all__ = (
    "WriterTextChoiceSupportCountTermCertificate",
    "WriterTextStateSupportCountCertificate",
    "WriterTextSupportCountCertificate",
    "writer_text_choice_support_count_term_certificate",
    "writer_text_state_support_count_certificate",
    "writer_text_support_count_certificate",
)
