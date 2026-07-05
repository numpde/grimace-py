"""Completion-count certificates for branch-support-backed witness counting."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import SouthStarError
from .errors import SouthStarErrorKind


@dataclass(frozen=True, slots=True)
class WriterBranchCompletionTermCertificate:
    branch_certificate: object
    successor_count_certificate: object
    successor_count: int


@dataclass(frozen=True, slots=True)
class WriterStateCompletionCountCertificate:
    state_key: object
    terminal_projection_certificate: object | None
    terminal_count: int
    branch_terms: tuple[WriterBranchCompletionTermCertificate, ...]
    completion_count: int


@dataclass(frozen=True, slots=True)
class WriterCursorCompletionCountCertificate:
    cursor: object
    state_count_certificates: tuple[
        tuple[object, int, WriterStateCompletionCountCertificate],
        ...,
    ]
    completion_count: int


def writer_cursor_completion_count_certificate(
    *,
    cursor,
    state_count_certificates: tuple[
        tuple[object, int, WriterStateCompletionCountCertificate],
        ...,
    ],
) -> WriterCursorCompletionCountCertificate:
    total = 0
    cursor_weighted_states = tuple(cursor.weighted_states)
    observed = tuple(
        (state_key, weight) for state_key, weight, _certificate in state_count_certificates
    )
    if observed != cursor_weighted_states:
        _count_violation("cursor_weighted_states_mismatch")

    seen = frozenset()
    for state_key, weight, certificate in state_count_certificates:
        if weight <= 0:
            _count_violation("nonpositive_cursor_weight")
        if certificate.state_key != state_key:
            _count_violation("state_count_certificate_key_mismatch")
        if state_key in seen:
            _count_violation("duplicate_state_in_cursor")
        seen = seen | frozenset((state_key,))
        total += weight * certificate.completion_count

    return WriterCursorCompletionCountCertificate(
        cursor=cursor,
        state_count_certificates=state_count_certificates,
        completion_count=total,
    )


def writer_state_completion_count_certificate(
    *,
    state_key,
    terminal_projection_certificate,
    terminal_count: int,
    branch_terms: tuple[WriterBranchCompletionTermCertificate, ...],
) -> WriterStateCompletionCountCertificate:
    if terminal_count < 0:
        _count_violation("negative_terminal_count")

    if terminal_projection_certificate is None and terminal_count != 0:
        _count_violation("terminal_count_without_terminal_projection")

    if terminal_projection_certificate is not None:
        terminal = terminal_projection_certificate.terminal
        if terminal is None:
            _count_violation("terminal_projection_lacks_terminal")
        if tuple(terminal_projection_certificate.source_cursor.weighted_states) != (
            (state_key, 1),
        ):
            _count_violation("terminal_projection_source_state_mismatch")
        if terminal.completion_count != terminal_count:
            _count_violation("terminal_count_mismatch")

    for term in branch_terms:
        if term.branch_certificate.source_state != state_key:
            _count_violation("branch_term_source_state_mismatch")
        if term.successor_count != (
            term.successor_count_certificate.completion_count
        ):
            _count_violation("branch_term_successor_count_mismatch")

    branch_total = sum(term.successor_count for term in branch_terms)
    completion_count = terminal_count + branch_total
    if completion_count < 0:
        _count_violation("negative_completion_count")

    return WriterStateCompletionCountCertificate(
        state_key=state_key,
        terminal_projection_certificate=terminal_projection_certificate,
        terminal_count=terminal_count,
        branch_terms=branch_terms,
        completion_count=completion_count,
    )


def writer_branch_completion_term_certificate(
    *,
    branch_certificate,
    successor_count_certificate,
) -> WriterBranchCompletionTermCertificate:
    if branch_certificate is None:
        _count_violation("missing_branch_certificate")

    if (
        not hasattr(successor_count_certificate, "cursor")
        or not successor_count_certificate.cursor.weighted_states
    ):
        _count_violation("invalid_successor_count_certificate")

    weighted_states = tuple(successor_count_certificate.cursor.weighted_states)
    if len(weighted_states) != 1:
        _count_violation("branch_successor_cursor_not_singleton")

    if weighted_states[0][1] != 1:
        _count_violation("branch_successor_cursor_not_singleton")

    successor_state = weighted_states[0][0]
    if branch_certificate.successor_state != successor_state:
        _count_violation("branch_successor_count_mismatch")
    if (
        len(successor_count_certificate.state_count_certificates) != 1
        or successor_count_certificate.state_count_certificates[0][0]
        != successor_state
    ):
        _count_violation("branch_successor_count_state_mismatch")

    return WriterBranchCompletionTermCertificate(
        branch_certificate=branch_certificate,
        successor_count_certificate=successor_count_certificate,
        successor_count=successor_count_certificate.completion_count,
    )


def _count_violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer branch completion count certificate violation: {kind}",
    )


__all__ = (
    "WriterBranchCompletionTermCertificate",
    "WriterCursorCompletionCountCertificate",
    "WriterStateCompletionCountCertificate",
    "writer_branch_completion_term_certificate",
    "writer_cursor_completion_count_certificate",
    "writer_state_completion_count_certificate",
)
