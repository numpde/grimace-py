"""Online statistics certificates for writer-shaped runtime choice results."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import SouthStarError
from .errors import SouthStarErrorKind


@dataclass(frozen=True, slots=True)
class WriterOnlineStatsCertificate:
    prefix: str
    stats: object
    choice_result_certificate: object
    checked_frontier_certificate: object
    support_count_certificate: object
    completion_count_certificate: object



def writer_online_stats_certificate(
    *,
    prefix: str,
    stats,
    choice_result_certificate,
    checked_frontier_certificate,
    support_count_certificate,
    completion_count_certificate,
) -> WriterOnlineStatsCertificate:
    if stats is None:
        _stats_violation("missing_stats")
    if choice_result_certificate is None:
        _stats_violation("missing_choice_result_certificate")
    if checked_frontier_certificate is None:
        _stats_violation("missing_checked_frontier_certificate")
    if support_count_certificate is None:
        _stats_violation("missing_support_count_certificate")
    if completion_count_certificate is None:
        _stats_violation("missing_completion_count_certificate")

    if getattr(choice_result_certificate, "prefix", None) != prefix:
        _stats_violation("choice_result_prefix_mismatch")

    if checked_frontier_certificate != getattr(
        choice_result_certificate,
        "checked_frontier_certificate",
        None,
    ):
        _stats_violation("choice_result_checked_frontier_certificate_mismatch")

    if completion_count_certificate != getattr(
        checked_frontier_certificate,
        "count_certificate",
        None,
    ):
        _stats_violation("frontier_certificate_completion_count_mismatch")

    frontier_cursor = checked_frontier_certificate.cursor
    if getattr(support_count_certificate, "cursor", None) != frontier_cursor:
        _stats_violation("support_count_certificate_cursor_mismatch")
    if getattr(completion_count_certificate, "cursor", None) != frontier_cursor:
        _stats_violation("completion_count_certificate_cursor_mismatch")

    if getattr(stats, "support_count", None) != getattr(
        support_count_certificate,
        "support_count",
        None,
    ):
        _stats_violation("support_count_mismatch")

    if getattr(stats, "completion_count", None) != getattr(
        completion_count_certificate,
        "completion_count",
        None,
    ):
        _stats_violation("completion_count_mismatch")

    choices = getattr(choice_result_certificate, "choices", ())
    if len(choices) != getattr(stats, "choice_count", None):
        _stats_violation("choice_count_mismatch")

    if getattr(stats, "has_eos", False) != any(
        getattr(choice, "is_eos", False) for choice in choices
    ):
        _stats_violation("has_eos_mismatch")

    eos_available = (
        checked_frontier_certificate.terminal_projection_certificate is not None
    )
    if getattr(stats, "eos_available", False) != eos_available:
        _stats_violation("eos_available_mismatch")

    choice_coverage = getattr(
        checked_frontier_certificate,
        "choice_count_coverage_certificate",
        None,
    )
    if choice_coverage is None:
        _stats_violation("missing_choice_count_coverage_certificate")
    if getattr(stats, "support_count", None) != choice_coverage.support_count:
        _stats_violation("choice_coverage_support_count_mismatch")
    if (
        getattr(stats, "completion_count", None)
        != choice_coverage.completion_count
    ):
        _stats_violation("choice_coverage_completion_count_mismatch")

    return WriterOnlineStatsCertificate(
        prefix=prefix,
        stats=stats,
        choice_result_certificate=choice_result_certificate,
        checked_frontier_certificate=checked_frontier_certificate,
        support_count_certificate=support_count_certificate,
        completion_count_certificate=completion_count_certificate,
    )



def _stats_violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer online stats certificate violation: {kind}",
    )


__all__ = (
    "WriterOnlineStatsCertificate",
    "writer_online_stats_certificate",
)
