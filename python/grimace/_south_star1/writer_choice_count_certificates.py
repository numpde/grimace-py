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


@dataclass(frozen=True, slots=True)
class WriterTextChoiceCountCoverageCertificate:
    text_projection_certificate: object
    text_choice_count_certificate: object
    support_coverage_term: object
    completion_coverage_terms: tuple[object, ...]
    support_count: int
    completion_count: int


@dataclass(frozen=True, slots=True)
class WriterTerminalChoiceCountCoverageCertificate:
    terminal_projection_certificate: object
    terminal_choice_count_certificate: object
    terminal_support_coverage_term: object | None
    terminal_completion_coverage_terms: tuple[object, ...]
    support_count: int
    completion_count: int


@dataclass(frozen=True, slots=True)
class WriterFrontierChoiceCountCoverageCertificate:
    projection_certificate: object
    text_choice_count_certificates: tuple[object, ...]
    terminal_choice_count_certificate: object | None
    support_count_term_coverage_certificate: object
    completion_count_term_coverage_certificate: object
    text_choice_terms: tuple[WriterTextChoiceCountCoverageCertificate, ...]
    terminal_choice_term: WriterTerminalChoiceCountCoverageCertificate | None
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


def writer_frontier_choice_count_coverage_certificate(
    *,
    projection_certificate,
    text_choice_count_certificates: tuple[object, ...],
    terminal_choice_count_certificate,
    support_count_term_coverage_certificate,
    completion_count_term_coverage_certificate,
) -> WriterFrontierChoiceCountCoverageCertificate:
    if projection_certificate is None:
        _choice_count_violation("missing_projection_certificate")
    if support_count_term_coverage_certificate is None:
        _choice_count_violation("missing_support_count_term_coverage")
    if completion_count_term_coverage_certificate is None:
        _choice_count_violation("missing_completion_count_term_coverage")
    if (
        support_count_term_coverage_certificate.projection_certificate
        is not projection_certificate
    ):
        _choice_count_violation("support_coverage_projection_mismatch")
    if (
        completion_count_term_coverage_certificate.projection_certificate
        is not projection_certificate
    ):
        _choice_count_violation("completion_coverage_projection_mismatch")

    projections = tuple(
        projection_certificate.text_choice_projection_certificates
    )
    if len(text_choice_count_certificates) != len(projections):
        _choice_count_violation("text_choice_count_certificate_count_mismatch")

    support_terms_by_key = {
        _text_projection_key(term.text_projection_certificate): term
        for term in support_count_term_coverage_certificate.text_terms
    }
    text_terms = tuple(
        _text_choice_count_coverage_certificate(
            projection=projection,
            choice_count=choice_count,
            support_terms_by_key=support_terms_by_key,
            completion_terms=(
                completion_count_term_coverage_certificate.branch_terms
            ),
        )
        for projection, choice_count in zip(
            projections,
            text_choice_count_certificates,
        )
    )
    terminal_term = _terminal_choice_count_coverage_certificate(
        projection_certificate=projection_certificate,
        terminal_choice_count_certificate=terminal_choice_count_certificate,
        support_count_term_coverage_certificate=(
            support_count_term_coverage_certificate
        ),
        completion_count_term_coverage_certificate=(
            completion_count_term_coverage_certificate
        ),
    )
    support_count = sum(term.support_count for term in text_terms)
    completion_count = sum(term.completion_count for term in text_terms)
    if terminal_term is not None:
        support_count += terminal_term.support_count
        completion_count += terminal_term.completion_count
    if support_count != support_count_term_coverage_certificate.support_count:
        _choice_count_violation("choice_coverage_support_total_mismatch")
    if completion_count != (
        completion_count_term_coverage_certificate.completion_count
    ):
        _choice_count_violation("choice_coverage_completion_total_mismatch")

    return WriterFrontierChoiceCountCoverageCertificate(
        projection_certificate=projection_certificate,
        text_choice_count_certificates=text_choice_count_certificates,
        terminal_choice_count_certificate=terminal_choice_count_certificate,
        support_count_term_coverage_certificate=(
            support_count_term_coverage_certificate
        ),
        completion_count_term_coverage_certificate=(
            completion_count_term_coverage_certificate
        ),
        text_choice_terms=text_terms,
        terminal_choice_term=terminal_term,
        support_count=support_count,
        completion_count=completion_count,
    )


def _text_choice_count_coverage_certificate(
    *,
    projection,
    choice_count,
    support_terms_by_key,
    completion_terms: tuple[object, ...],
) -> WriterTextChoiceCountCoverageCertificate:
    if choice_count.text_projection_certificate is not projection:
        _choice_count_violation("text_choice_count_projection_mismatch")

    support_term = support_terms_by_key.get(_text_projection_key(projection))
    if support_term is None:
        _choice_count_violation("text_choice_support_coverage_missing")
    if support_term.support_count != choice_count.support_count:
        _choice_count_violation("text_choice_support_count_coverage_mismatch")

    branch_ids = _branch_certificate_identity_set(projection)
    matched_completion_terms = tuple(
        term
        for term in completion_terms
        if id(term.projection_branch_certificate) in branch_ids
    )
    covered_branch_ids = frozenset(
        id(term.projection_branch_certificate)
        for term in matched_completion_terms
    )
    if covered_branch_ids != branch_ids:
        _choice_count_violation(
            "text_choice_completion_branch_partition_mismatch"
        )
    completion_count = sum(
        term.weighted_completion_count for term in matched_completion_terms
    )
    if completion_count != choice_count.completion_count:
        _choice_count_violation(
            "text_choice_completion_count_coverage_mismatch"
        )

    return WriterTextChoiceCountCoverageCertificate(
        text_projection_certificate=projection,
        text_choice_count_certificate=choice_count,
        support_coverage_term=support_term,
        completion_coverage_terms=matched_completion_terms,
        support_count=support_term.support_count,
        completion_count=completion_count,
    )


def _terminal_choice_count_coverage_certificate(
    *,
    projection_certificate,
    terminal_choice_count_certificate,
    support_count_term_coverage_certificate,
    completion_count_term_coverage_certificate,
) -> WriterTerminalChoiceCountCoverageCertificate | None:
    terminal_projection = projection_certificate.terminal_projection_certificate
    if terminal_projection is None:
        if terminal_choice_count_certificate is not None:
            _choice_count_violation("terminal_choice_count_without_projection")
        return None

    if terminal_choice_count_certificate is None:
        _choice_count_violation("terminal_choice_count_missing")
    if (
        terminal_choice_count_certificate.terminal_projection_certificate
        is not terminal_projection
    ):
        _choice_count_violation("terminal_choice_projection_mismatch")

    terminal_support_term = support_count_term_coverage_certificate.terminal_term
    if terminal_support_term is None:
        _choice_count_violation("terminal_support_coverage_missing")
    if terminal_support_term.terminal_count != (
        terminal_choice_count_certificate.support_count
    ):
        _choice_count_violation("terminal_support_count_coverage_mismatch")

    terminal_completion_terms = tuple(
        completion_count_term_coverage_certificate.terminal_terms
    )
    terminal_completion_count = sum(
        term.weighted_completion_count for term in terminal_completion_terms
    )
    if terminal_completion_count != (
        terminal_choice_count_certificate.completion_count
    ):
        _choice_count_violation("terminal_completion_count_coverage_mismatch")

    return WriterTerminalChoiceCountCoverageCertificate(
        terminal_projection_certificate=terminal_projection,
        terminal_choice_count_certificate=terminal_choice_count_certificate,
        terminal_support_coverage_term=terminal_support_term,
        terminal_completion_coverage_terms=terminal_completion_terms,
        support_count=terminal_support_term.terminal_count,
        completion_count=terminal_completion_count,
    )


def _text_projection_key(projection) -> tuple[object, ...]:
    return (
        projection.source_cursor,
        projection.emitted_text,
        projection.successor_cursor,
        projection.immediate_multiplicity,
    )


def _branch_certificate_identity_set(projection) -> frozenset[int]:
    return frozenset(
        id(certificate) for certificate in projection.branch_certificates
    )



def _choice_count_violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer choice count certificate violation: {kind}",
    )


__all__ = (
    "WriterFrontierChoiceCountCoverageCertificate",
    "WriterTerminalChoiceCountCoverageCertificate",
    "WriterTextChoiceCountCertificate",
    "WriterTextChoiceCountCoverageCertificate",
    "WriterTerminalChoiceCountCertificate",
    "writer_frontier_choice_count_coverage_certificate",
    "writer_text_choice_count_certificate",
    "writer_terminal_choice_count_certificate",
)
