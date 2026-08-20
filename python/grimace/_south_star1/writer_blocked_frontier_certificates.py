"""Certificates for blocked writer frontier products."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import SouthStarError
from .errors import SouthStarErrorKind


@dataclass(frozen=True, slots=True)
class WriterBlockedFrontierCertificate:
    cursor: object
    blocked: bool
    graph_policy_blocker_certificates: tuple[object, ...]
    stereo_policy_blocker_certificates: tuple[object, ...]
    unsupported_execution_capability_certificates: tuple[object, ...]
    unsupported_terminal_execution_capability_certificates: tuple[object, ...]
    work_envelope_violation_certificates: tuple[object, ...]
    diagnostic_certificate: object


def writer_blocked_frontier_certificate(
    *,
    cursor,
    diagnostic_certificate,
) -> WriterBlockedFrontierCertificate:
    if diagnostic_certificate.cursor != cursor:
        _blocked_frontier_violation("diagnostic_cursor_mismatch")

    if not diagnostic_certificate.blocked:
        _blocked_frontier_violation("diagnostic_not_blocked")

    if diagnostic_certificate.text_choice_projection_certificates:
        _blocked_frontier_violation("blocked_frontier_has_text_projections")

    if diagnostic_certificate.terminal_projection_certificate is not None:
        _blocked_frontier_violation("blocked_frontier_has_terminal_projection")

    if diagnostic_certificate.branch_certificates:
        _blocked_frontier_violation("blocked_frontier_has_branch_certificates")

    if diagnostic_certificate.terminal_certificates:
        _blocked_frontier_violation("blocked_frontier_has_terminal_certificates")

    if diagnostic_certificate.count_certificate is not None:
        _blocked_frontier_violation("blocked_frontier_has_count_certificate")

    graph_policy_blocker_certificates = tuple(
        diagnostic_certificate.graph_policy_blocker_certificates
    )
    stereo_policy_blocker_certificates = tuple(
        diagnostic_certificate.stereo_policy_blocker_certificates
    )
    unsupported_execution_capability_certificates = tuple(
        diagnostic_certificate.unsupported_execution_capability_certificates
    )
    unsupported_terminal_execution_capability_certificates = tuple(
        diagnostic_certificate
        .unsupported_terminal_execution_capability_certificates
    )
    work_envelope_violation_certificates = tuple(
        diagnostic_certificate.work_envelope_violation_certificates
    )

    if not (
        graph_policy_blocker_certificates
        or stereo_policy_blocker_certificates
        or unsupported_execution_capability_certificates
        or unsupported_terminal_execution_capability_certificates
        or work_envelope_violation_certificates
    ):
        _blocked_frontier_violation(
            "blocked_frontier_lacks_negative_evidence"
        )

    return WriterBlockedFrontierCertificate(
        cursor=cursor,
        blocked=True,
        graph_policy_blocker_certificates=graph_policy_blocker_certificates,
        stereo_policy_blocker_certificates=stereo_policy_blocker_certificates,
        unsupported_execution_capability_certificates=(
            unsupported_execution_capability_certificates
        ),
        unsupported_terminal_execution_capability_certificates=(
            unsupported_terminal_execution_capability_certificates
        ),
        work_envelope_violation_certificates=(
            work_envelope_violation_certificates
        ),
        diagnostic_certificate=diagnostic_certificate,
    )


def _blocked_frontier_violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer blocked frontier certificate violation: {kind}",
    )


__all__ = (
    "WriterBlockedFrontierCertificate",
    "writer_blocked_frontier_certificate",
)
