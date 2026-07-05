"""Aggregate frontier certificates for checked frontier cursors."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .errors import SouthStarError
from .errors import SouthStarErrorKind


@dataclass(frozen=True, slots=True)
class WriterFrontierProjectionCertificate:
    cursor: object
    choices: object
    branch_certificates: tuple[object, ...]
    terminal_certificates: tuple[object, ...]
    text_choice_projection_certificates: tuple[object, ...]
    terminal_projection_certificate: object | None


@dataclass(frozen=True, slots=True)
class WriterCheckedFrontierCertificate:
    projection_certificate: object
    cursor: object
    choices: object
    branch_certificates: tuple[object, ...]
    terminal_certificates: tuple[object, ...]
    text_choice_projection_certificates: tuple[object, ...]
    terminal_projection_certificate: object | None
    count_certificate: object
    text_choice_count_certificates: tuple[object, ...] = ()
    terminal_choice_count_certificate: object | None = None
    support_count_certificate: object | None = None
    diagnostic_certificate: object | None = None


def writer_frontier_projection_certificate(
    *,
    cursor,
    choices,
    branch_supports: tuple[object, ...],
    terminal_supports: tuple[object, ...],
    text_choice_projection_certificates: tuple[object, ...],
    terminal_projection_certificate,
) -> WriterFrontierProjectionCertificate:
    branch_certificates = tuple(
        support.checked_branch_certificate for support in branch_supports
    )
    if any(certificate is None for certificate in branch_certificates):
        _frontier_violation("missing_branch_certificate")

    for support, certificate in zip(branch_supports, branch_certificates):
        if getattr(certificate, "successor_state_certificate", None) is None:
            _frontier_violation(
                "branch_certificate_lacks_successor_state_certificate"
            )
        if certificate.source_state != support.source_state:
            _frontier_violation("branch_certificate_source_mismatch")
        if certificate.successor_state != support.successor_state:
            _frontier_violation("branch_certificate_successor_mismatch")
        if certificate.emitted_text != support.emitted_text:
            _frontier_violation("branch_certificate_text_mismatch")
        if certificate.parent_weight != support.parent_weight:
            _frontier_violation("branch_certificate_parent_weight_mismatch")
        if certificate.branch_ordinal != support.branch_ordinal:
            _frontier_violation("branch_certificate_ordinal_mismatch")

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
        if terminal_projection_certificate.source_cursor != cursor:
            _frontier_violation(
                "terminal_projection_source_cursor_mismatch"
            )
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
            if terminal_certificate.parent_weight != terminal_support.parent_weight:
                _frontier_violation(
                    "terminal_certificate_parent_weight_mismatch"
                )

    projection_texts = tuple(
        cert.emitted_text for cert in text_choice_projection_certificates
    )
    choice_texts = tuple(choice.emitted_text for choice in choices.choices)
    if projection_texts != choice_texts:
        _frontier_violation("projection_choice_text_mismatch")
    for projection in text_choice_projection_certificates:
        if getattr(projection, "source_cursor", None) != cursor:
            _frontier_violation("projection_source_cursor_mismatch")

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

    if terminal_projection_certificate is not None and tuple(
        terminal_projection_certificate.terminal_certificates
    ) != terminal_certificates:
        _frontier_violation("terminal_certificate_partition_mismatch")

    return WriterFrontierProjectionCertificate(
        cursor=cursor,
        choices=choices,
        branch_certificates=branch_certificates,
        terminal_certificates=terminal_certificates,
        text_choice_projection_certificates=text_choice_projection_certificates,
        terminal_projection_certificate=terminal_projection_certificate,
    )


def writer_checked_frontier_certificate(
    *,
    cursor=None,
    choices=None,
    branch_supports: tuple[object, ...] = (),
    terminal_supports: tuple[object, ...] = (),
    text_choice_projection_certificates: tuple[object, ...] = (),
    projection_certificate=None,
    text_choice_count_certificates: tuple[object, ...] = (),
    terminal_choice_count_certificate: object | None = None,
    support_count_certificate: object | None = None,
    terminal_projection_certificate=None,
    count_certificate=None,
    diagnostic_certificate=None,
) -> WriterCheckedFrontierCertificate:
    if projection_certificate is None:
        projection_certificate = writer_frontier_projection_certificate(
            cursor=cursor,
            choices=choices,
            branch_supports=branch_supports,
            terminal_supports=terminal_supports,
            text_choice_projection_certificates=(
                text_choice_projection_certificates
            ),
            terminal_projection_certificate=terminal_projection_certificate,
        )
    cursor = projection_certificate.cursor
    choices = projection_certificate.choices
    branch_certificates = projection_certificate.branch_certificates
    terminal_certificates = projection_certificate.terminal_certificates
    text_choice_projection_certificates = (
        projection_certificate.text_choice_projection_certificates
    )
    terminal_projection_certificate = (
        projection_certificate.terminal_projection_certificate
    )

    if text_choice_count_certificates:
        if len(text_choice_count_certificates) != len(
            text_choice_projection_certificates
        ):
            _frontier_violation(
                "choice_count_certificate_count_mismatch"
            )

        for projection, choice_count_certificate in zip(
            text_choice_projection_certificates,
            text_choice_count_certificates,
        ):
            if getattr(
                choice_count_certificate,
                "text_projection_certificate",
                None,
            ) is not projection:
                _frontier_violation(
                    "choice_count_certificate_projection_mismatch"
                )

        projected_choice_count_projection_identities = tuple(
            id(certificate.text_projection_certificate)
            for certificate in text_choice_count_certificates
        )
        if len(set(projected_choice_count_projection_identities)) != len(
            projected_choice_count_projection_identities
        ):
            _frontier_violation("choice_count_certificate_duplicate")

        projected_choice_count_projections = tuple(
            cert.text_projection_certificate
            for cert in text_choice_count_certificates
        )
        if sorted(projected_choice_count_projections, key=id) != sorted(
            text_choice_projection_certificates,
            key=id,
        ):
            _frontier_violation(
                "choice_count_certificate_projection_partition"
            )

    if terminal_projection_certificate is None and (
        terminal_choice_count_certificate is not None
    ):
        _frontier_violation("terminal_choice_count_without_terminal")
    if terminal_projection_certificate is not None and (
        terminal_choice_count_certificate is None
    ):
        _frontier_violation("terminal_choice_count_missing")
    if terminal_projection_certificate is not None:
        if terminal_projection_certificate.source_cursor != cursor:
            _frontier_violation(
                "terminal_projection_source_cursor_mismatch"
            )
    if terminal_choice_count_certificate is not None:
        if terminal_choice_count_certificate.terminal_projection_certificate is not (
            terminal_projection_certificate
        ):
            _frontier_violation(
                "terminal_choice_count_projection_mismatch"
            )
        if (
            terminal_choice_count_certificate
            .terminal_projection_certificate
            .source_cursor
            != cursor
        ):
            _frontier_violation(
                "terminal_choice_count_source_cursor_mismatch"
            )

    if support_count_certificate is not None:
        if support_count_certificate.cursor != cursor:
            _frontier_violation(
                "support_count_certificate_cursor_mismatch"
            )

        terminal_support_count = 0
        if terminal_choice_count_certificate is not None:
            terminal_support_count = (
                terminal_choice_count_certificate.support_count
            )

        if support_count_certificate.support_count != (
            terminal_support_count
            + sum(
                cert.support_count
                for cert in text_choice_count_certificates
            )
        ):
            _frontier_violation(
                "support_count_certificate_total_mismatch"
            )

    if count_certificate is not None:
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
        projection_certificate=projection_certificate,
        cursor=cursor,
        choices=choices,
        branch_certificates=branch_certificates,
        terminal_certificates=terminal_certificates,
        text_choice_projection_certificates=text_choice_projection_certificates,
        text_choice_count_certificates=text_choice_count_certificates,
        terminal_choice_count_certificate=terminal_choice_count_certificate,
        support_count_certificate=support_count_certificate,
        terminal_projection_certificate=terminal_projection_certificate,
        count_certificate=count_certificate,
        diagnostic_certificate=diagnostic_certificate,
    )


def _frontier_violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer checked frontier certificate violation: {kind}",
    )


__all__ = (
    "WriterCheckedFrontierCertificate",
    "WriterFrontierProjectionCertificate",
    "writer_checked_frontier_certificate",
    "writer_frontier_projection_certificate",
)
