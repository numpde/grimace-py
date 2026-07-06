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


@dataclass(frozen=True, slots=True)
class WriterFrontierTextSupportCountCoverageTerm:
    text_projection_certificate: object
    support_count_term_certificate: object
    successor_support_count_certificate: object
    support_count: int


@dataclass(frozen=True, slots=True)
class WriterFrontierTerminalSupportCountCoverageTerm:
    terminal_projection_certificate: object
    terminal_count: int


@dataclass(frozen=True, slots=True)
class WriterFrontierSupportCountTermCoverageCertificate:
    projection_certificate: object
    support_count_certificate: object
    text_terms: tuple[WriterFrontierTextSupportCountCoverageTerm, ...]
    terminal_term: WriterFrontierTerminalSupportCountCoverageTerm | None
    text_support_count: int
    terminal_support_count: int
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
        if terminal_projection_certificate.source_cursor != cursor:
            _support_count_violation(
                "terminal_projection_source_cursor_mismatch"
            )
        terminal = terminal_projection_certificate.terminal
        if terminal is None:
            _support_count_violation("terminal_projection_lacks_terminal")

    for term in choice_terms:
        if term is None:
            _support_count_violation("missing_choice_term")
        if term.text_projection_certificate.source_cursor != cursor:
            _support_count_violation(
                "choice_projection_source_cursor_mismatch"
            )
        if term.support_count != (
            term.successor_support_count_certificate.support_count
        ):
            _support_count_violation("choice_term_support_count_mismatch")

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
    source_cursor = getattr(source_snapshot, "cursor", None)
    if source_cursor is not None and source_cursor != cursor:
        _support_count_violation("source_snapshot_cursor_mismatch")

    return WriterTextSupportCountCertificate(
        source_snapshot=source_snapshot,
        cursor=cursor,
        state_support_count_certificate=state_support_count_certificate,
        support_count=state_support_count_certificate.support_count,
    )


def writer_frontier_support_count_term_coverage_certificate(
    *,
    projection_certificate,
    support_count_certificate,
) -> WriterFrontierSupportCountTermCoverageCertificate:
    if projection_certificate is None:
        _support_count_violation("missing_projection_certificate")
    if support_count_certificate is None:
        _support_count_violation("missing_support_count_certificate")
    if support_count_certificate.cursor != projection_certificate.cursor:
        _support_count_violation("support_count_cursor_mismatch")

    state_certificate = support_count_certificate.state_support_count_certificate
    if state_certificate.cursor != projection_certificate.cursor:
        _support_count_violation("state_support_count_cursor_mismatch")

    text_terms = _frontier_text_support_coverage_terms(
        projection_certificate=projection_certificate,
        support_terms=tuple(state_certificate.choice_terms),
    )
    terminal_term, terminal_support_count = (
        _frontier_terminal_support_coverage_term(
            projection_certificate=projection_certificate,
            state_certificate=state_certificate,
        )
    )
    text_support_count = sum(term.support_count for term in text_terms)
    support_count = terminal_support_count + text_support_count
    if state_certificate.support_count != support_count:
        _support_count_violation(
            "support_count_term_coverage_state_total_mismatch"
        )
    if support_count_certificate.support_count != support_count:
        _support_count_violation("support_count_term_coverage_total_mismatch")

    return WriterFrontierSupportCountTermCoverageCertificate(
        projection_certificate=projection_certificate,
        support_count_certificate=support_count_certificate,
        text_terms=text_terms,
        terminal_term=terminal_term,
        text_support_count=text_support_count,
        terminal_support_count=terminal_support_count,
        support_count=support_count,
    )


def _frontier_text_support_coverage_terms(
    *,
    projection_certificate,
    support_terms: tuple[object, ...],
) -> tuple[WriterFrontierTextSupportCountCoverageTerm, ...]:
    projections = tuple(
        projection_certificate.text_choice_projection_certificates
    )
    projected_by_key = {
        _text_projection_support_key(projection): projection
        for projection in projections
    }
    support_by_key: dict[tuple[object, ...], object] = {}

    for term in support_terms:
        projection = term.text_projection_certificate
        if (
            getattr(projection, "source_cursor", None)
            != projection_certificate.cursor
        ):
            _support_count_violation("choice_projection_source_cursor_mismatch")
        projection_id = _text_projection_support_key(projection)
        if projection_id in support_by_key:
            _support_count_violation("support_count_choice_term_duplicate")
        if projection_id not in projected_by_key:
            if term.support_count != 0:
                _support_count_violation(
                    "support_count_choice_term_projection_partition_mismatch"
                )
            continue
        support_by_key[projection_id] = term

    if set(projected_by_key) - set(support_by_key):
        _support_count_violation(
            "support_count_choice_term_projection_partition_mismatch"
        )

    terms = []
    for projection in projections:
        projection_key = _text_projection_support_key(projection)
        term = support_by_key[projection_key]
        if _text_projection_support_key(term.text_projection_certificate) != (
            projection_key
        ):
            _support_count_violation("support_count_choice_projection_mismatch")
        if (
            term.successor_support_count_certificate.cursor
            != projection.successor_cursor
        ):
            _support_count_violation("support_count_successor_cursor_mismatch")
        if term.support_count != (
            term.successor_support_count_certificate.support_count
        ):
            _support_count_violation("support_count_choice_term_total_mismatch")
        terms.append(
            WriterFrontierTextSupportCountCoverageTerm(
                text_projection_certificate=projection,
                support_count_term_certificate=term,
                successor_support_count_certificate=(
                    term.successor_support_count_certificate
                ),
                support_count=term.support_count,
            )
        )
    return tuple(terms)


def _frontier_terminal_support_coverage_term(
    *,
    projection_certificate,
    state_certificate,
):
    terminal_projection = projection_certificate.terminal_projection_certificate
    state_terminal_projection = state_certificate.terminal_projection_certificate

    if terminal_projection is None:
        if state_terminal_projection is not None:
            _support_count_violation(
                "terminal_support_count_projection_mismatch"
            )
        if state_certificate.terminal_count != 0:
            _support_count_violation("terminal_support_count_without_projection")
        return None, 0

    if _terminal_projection_support_key(state_terminal_projection) != (
        _terminal_projection_support_key(terminal_projection)
    ):
        _support_count_violation("terminal_support_count_projection_mismatch")
    if state_certificate.terminal_count != 1:
        _support_count_violation("terminal_support_count_mismatch")
    return (
        WriterFrontierTerminalSupportCountCoverageTerm(
            terminal_projection_certificate=terminal_projection,
            terminal_count=state_certificate.terminal_count,
        ),
        1,
    )


def _text_projection_support_key(projection) -> tuple[object, ...]:
    return (
        projection.source_cursor,
        projection.emitted_text,
        projection.successor_cursor,
        projection.immediate_multiplicity,
    )


def _terminal_projection_support_key(projection) -> tuple[object, ...]:
    if projection is None:
        return ()
    return (
        projection.source_cursor,
        projection.finalized_cursor,
        projection.multiplicity,
        projection.support_count,
    )



def _support_count_violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer support count certificate violation: {kind}",
    )


__all__ = (
    "WriterFrontierSupportCountTermCoverageCertificate",
    "WriterFrontierTerminalSupportCountCoverageTerm",
    "WriterFrontierTextSupportCountCoverageTerm",
    "WriterTextChoiceSupportCountTermCertificate",
    "WriterTextStateSupportCountCertificate",
    "WriterTextSupportCountCertificate",
    "writer_frontier_support_count_term_coverage_certificate",
    "writer_text_choice_support_count_term_certificate",
    "writer_text_state_support_count_certificate",
    "writer_text_support_count_certificate",
)
