"""Aggregate frontier certificates for checked frontier cursors."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .errors import SouthStarError
from .errors import SouthStarErrorKind


@dataclass(frozen=True, slots=True)
class WriterCheckedFrontierCertificate:
    cursor: object
    choices: object
    branch_certificates: tuple[object, ...]
    terminal_certificates: tuple[object, ...]
    text_choice_projection_certificates: tuple[object, ...]
    terminal_projection_certificate: object | None
    count_certificate: object
    diagnostic_certificate: object | None = None


def writer_checked_frontier_certificate(
    *,
    cursor,
    choices,
    branch_supports: tuple[object, ...],
    terminal_supports: tuple[object, ...],
    text_choice_projection_certificates: tuple[object, ...],
    terminal_projection_certificate,
    count_certificate,
    diagnostic_certificate=None,
) -> WriterCheckedFrontierCertificate:
    branch_certificates = tuple(
        support.checked_branch_certificate for support in branch_supports
    )
    if any(certificate is None for certificate in branch_certificates):
        _frontier_violation("missing_branch_certificate")

    for support, certificate in zip(branch_supports, branch_certificates):
        if certificate.source_state != support.source_state:
            _frontier_violation("branch_certificate_source_mismatch")
        if certificate.successor_state != support.successor_state:
            _frontier_violation("branch_certificate_successor_mismatch")
        if certificate.emitted_text != support.emitted_text:
            _frontier_violation("branch_certificate_text_mismatch")

    terminal_certificates = tuple(
        support.checked_terminal_certificate for support in terminal_supports
    )
    if any(certificate is None for certificate in terminal_certificates):
        _frontier_violation("missing_terminal_certificate")

    if choices.terminal is None:
        if terminal_supports:
            _frontier_violation("terminal_supports_without_terminal_choice")
        if terminal_projection_certificate is not None:
            _frontier_violation("terminal_projection_without_terminal_choice")
    else:
        if terminal_projection_certificate is None:
            _frontier_violation("terminal_choice_lacks_projection_certificate")
        for terminal_support, terminal_certificate in zip(
            terminal_supports, terminal_certificates
        ):
            if terminal_certificate.source_state != terminal_support.source_state:
                _frontier_violation(
                    "terminal_certificate_source_state_mismatch"
                )
            if terminal_certificate.finalized_state != (
                terminal_support.finalized_state
            ):
                _frontier_violation(
                    "terminal_certificate_finalized_state_mismatch"
                )

    projection_texts = tuple(
        cert.emitted_text for cert in text_choice_projection_certificates
    )
    choice_texts = tuple(choice.emitted_text for choice in choices.choices)
    if projection_texts != choice_texts:
        _frontier_violation("projection_choice_text_mismatch")

    projected_certs = tuple(
        branch_certificate
        for projection in text_choice_projection_certificates
        for branch_certificate in projection.branch_certificates
    )
    branch_certificate_identity = tuple(id(certificate) for certificate in branch_certificates)
    projected_certificate_identity = tuple(
        id(branch_certificate)
        for branch_certificate in projected_certs
    )

    projected_counts = Counter(projected_certificate_identity)
    branch_counts = Counter(branch_certificate_identity)
    if projected_counts != branch_counts:
        _frontier_violation("projection_branch_certificate_partition_mismatch")

    if any(count > 1 for count in projected_counts.values()):
        _frontier_violation("projection_branch_certificate_duplicate")

    if any(count > 1 for count in branch_counts.values()):
        _frontier_violation("branch_certificate_duplicates")

    if not all(
        projected_counts.get(id(certificate), 0) == 1
        for certificate in branch_certificates
    ):
        _frontier_violation("projection_branch_certificate_partition_mismatch")

    if count_certificate is None:
        _frontier_violation("missing_count_certificate")
    if count_certificate.cursor != cursor:
        _frontier_violation("count_certificate_cursor_mismatch")

    if diagnostic_certificate is not None:
        if diagnostic_certificate.cursor != cursor:
            _frontier_violation("diagnostic_cursor_mismatch")
        if (
            getattr(diagnostic_certificate, "count_certificate", None)
            is not count_certificate
        ):
            _frontier_violation("diagnostic_count_certificate_mismatch")
        if (
            getattr(diagnostic_certificate, "text_choice_projection_certificates", ())
            != text_choice_projection_certificates
        ):
            _frontier_violation("diagnostic_projection_certificate_mismatch")

    return WriterCheckedFrontierCertificate(
        cursor=cursor,
        choices=choices,
        branch_certificates=branch_certificates,
        terminal_certificates=terminal_certificates,
        text_choice_projection_certificates=text_choice_projection_certificates,
        terminal_projection_certificate=terminal_projection_certificate,
        count_certificate=count_certificate,
        diagnostic_certificate=diagnostic_certificate,
    )


def _frontier_violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer checked frontier certificate violation: {kind}",
    )


__all__ = ("WriterCheckedFrontierCertificate", "writer_checked_frontier_certificate")
