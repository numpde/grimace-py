"""Per-choice support/completion-count certificates."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import SouthStarError
from .errors import SouthStarErrorKind


@dataclass(frozen=True, slots=True)
class WriterTextChoiceCountCertificate:
    text_projection_certificate: object
    support_count_certificate: object
    completion_count_certificate: object
    emitted_text: str
    support_count: int
    completion_count: int


@dataclass(frozen=True, slots=True)
class WriterTerminalChoiceCountCertificate:
    terminal_projection_certificate: object
    support_count: int
    completion_count: int


def writer_text_choice_count_certificate(
    *,
    text_projection_certificate,
    support_count_certificate,
    completion_count_certificate,
) -> WriterTextChoiceCountCertificate:
    if text_projection_certificate is None:
        _choice_count_violation("missing_text_projection_certificate")

    if support_count_certificate is None:
        _choice_count_violation("missing_support_count_certificate")

    if completion_count_certificate is None:
        _choice_count_violation("missing_completion_count_certificate")

    projection_successor_cursor = getattr(
        text_projection_certificate,
        "successor_cursor",
        None,
    )
    if projection_successor_cursor is None:
        _choice_count_violation("text_projection_missing_successor_cursor")

    if getattr(
        support_count_certificate,
        "cursor",
        None,
    ) != projection_successor_cursor:
        _choice_count_violation("support_count_successor_cursor_mismatch")

    if getattr(
        completion_count_certificate,
        "cursor",
        None,
    ) != projection_successor_cursor:
        _choice_count_violation("completion_count_successor_cursor_mismatch")

    projection_support_count = getattr(
        text_projection_certificate,
        "support_count",
        None,
    )
    projected_successor_support_count = getattr(
        support_count_certificate,
        "support_count",
        None,
    )
    if projected_successor_support_count is None:
        _choice_count_violation("support_count_certificate_missing_count")

    if projection_support_count is not None and (
        projection_support_count != projected_successor_support_count
    ):
        _choice_count_violation("support_count_mismatch")

    projection_completion_count = getattr(
        text_projection_certificate,
        "completion_count",
        None,
    )
    projected_successor_completion_count = getattr(
        completion_count_certificate,
        "completion_count",
        None,
    )
    if projected_successor_completion_count is None:
        _choice_count_violation("completion_count_certificate_missing_count")

    if projection_completion_count is not None and (
        projection_completion_count != projected_successor_completion_count
    ):
        _choice_count_violation("completion_count_mismatch")

    return WriterTextChoiceCountCertificate(
        text_projection_certificate=text_projection_certificate,
        support_count_certificate=support_count_certificate,
        completion_count_certificate=completion_count_certificate,
        emitted_text=getattr(text_projection_certificate, "emitted_text", ""),
        support_count=projected_successor_support_count,
        completion_count=projected_successor_completion_count,
    )



def writer_terminal_choice_count_certificate(
    *,
    terminal_projection_certificate,
) -> WriterTerminalChoiceCountCertificate | None:
    if terminal_projection_certificate is None:
        return None

    terminal = getattr(terminal_projection_certificate, "terminal", None)
    if terminal is None:
        _choice_count_violation("terminal_projection_lacks_terminal")

    if terminal_projection_certificate.support_count != terminal.support_count:
        _choice_count_violation("terminal_support_count_mismatch")

    if terminal_projection_certificate.completion_count != (
        terminal.completion_count
    ):
        _choice_count_violation("terminal_completion_count_mismatch")

    return WriterTerminalChoiceCountCertificate(
        terminal_projection_certificate=terminal_projection_certificate,
        support_count=terminal_projection_certificate.support_count,
        completion_count=terminal_projection_certificate.completion_count,
    )



def _choice_count_violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer choice count certificate violation: {kind}",
    )


__all__ = (
    "WriterTextChoiceCountCertificate",
    "WriterTerminalChoiceCountCertificate",
    "writer_text_choice_count_certificate",
    "writer_terminal_choice_count_certificate",
)
