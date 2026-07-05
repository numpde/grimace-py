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


@dataclass(frozen=True, slots=True)
class WriterFrontierBranchCompletionCoverageTerm:
    projection_branch_certificate: object
    count_branch_term_certificate: object
    cursor_weight: int
    projection_parent_weight: int
    count_parent_weight: int
    successor_completion_count: int
    weighted_completion_count: int


@dataclass(frozen=True, slots=True)
class WriterFrontierTerminalCompletionCoverageTerm:
    projection_terminal_certificate: object
    state_terminal_projection_certificate: object
    cursor_weight: int
    projection_parent_weight: int
    count_parent_weight: int
    terminal_completion_count: int
    weighted_completion_count: int


@dataclass(frozen=True, slots=True)
class WriterFrontierCompletionTermCoverageCertificate:
    projection_certificate: object
    count_certificate: object
    branch_terms: tuple[WriterFrontierBranchCompletionCoverageTerm, ...]
    terminal_terms: tuple[WriterFrontierTerminalCompletionCoverageTerm, ...]
    branch_completion_count: int
    terminal_completion_count: int
    completion_count: int


@dataclass(frozen=True, slots=True)
class WriterFrontierCompletionCountCertificate:
    projection_certificate: object
    count_certificate: object
    text_choice_count_certificates: tuple[object, ...]
    terminal_choice_count_certificate: object | None
    terminal_completion_count: int
    text_completion_count: int
    completion_count: int
    term_coverage_certificate: object | None = None


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


def writer_frontier_completion_count_certificate(
    *,
    projection_certificate,
    count_certificate,
    text_choice_count_certificates: tuple[object, ...],
    terminal_choice_count_certificate,
) -> WriterFrontierCompletionCountCertificate:
    if projection_certificate is None:
        _frontier_count_violation("missing_projection_certificate")
    if count_certificate is None:
        _frontier_count_violation("missing_count_certificate")
    if count_certificate.cursor != projection_certificate.cursor:
        _frontier_count_violation("count_certificate_cursor_mismatch")

    projections = tuple(
        projection_certificate.text_choice_projection_certificates
    )
    if len(text_choice_count_certificates) != len(projections):
        _frontier_count_violation(
            "text_choice_count_certificate_count_mismatch"
        )

    for projection, choice_count in zip(
        projections,
        text_choice_count_certificates,
    ):
        if choice_count.text_projection_certificate is not projection:
            _frontier_count_violation(
                "text_choice_count_projection_mismatch"
            )
        if choice_count.completion_count != (
            choice_count.completion_count_certificate.completion_count
        ):
            _frontier_count_violation(
                "text_choice_completion_count_nested_mismatch"
            )

    terminal_projection = (
        projection_certificate.terminal_projection_certificate
    )
    if terminal_projection is None:
        if terminal_choice_count_certificate is not None:
            _frontier_count_violation(
                "terminal_choice_count_without_terminal_projection"
            )
        terminal_completion_count = 0
    else:
        if terminal_choice_count_certificate is None:
            _frontier_count_violation("terminal_choice_count_missing")
        if (
            terminal_choice_count_certificate.terminal_projection_certificate
            is not terminal_projection
        ):
            _frontier_count_violation(
                "terminal_choice_count_projection_mismatch"
            )
        if (
            terminal_choice_count_certificate.completion_count
            != terminal_projection.completion_count
        ):
            _frontier_count_violation(
                "terminal_choice_completion_count_mismatch"
            )
        terminal_completion_count = (
            terminal_choice_count_certificate.completion_count
        )

    text_completion_count = sum(
        certificate.completion_count
        for certificate in text_choice_count_certificates
    )
    completion_count = terminal_completion_count + text_completion_count
    if completion_count != count_certificate.completion_count:
        _frontier_count_violation("frontier_completion_count_total_mismatch")

    term_coverage_certificate = (
        writer_frontier_completion_term_coverage_certificate(
            projection_certificate=projection_certificate,
            count_certificate=count_certificate,
        )
    )
    if term_coverage_certificate.completion_count != completion_count:
        _frontier_count_violation(
            "completion_term_coverage_aggregate_mismatch"
        )

    return WriterFrontierCompletionCountCertificate(
        projection_certificate=projection_certificate,
        count_certificate=count_certificate,
        text_choice_count_certificates=text_choice_count_certificates,
        terminal_choice_count_certificate=terminal_choice_count_certificate,
        terminal_completion_count=terminal_completion_count,
        text_completion_count=text_completion_count,
        completion_count=completion_count,
        term_coverage_certificate=term_coverage_certificate,
    )


def writer_frontier_completion_term_coverage_certificate(
    *,
    projection_certificate,
    count_certificate,
) -> WriterFrontierCompletionTermCoverageCertificate:
    if projection_certificate is None:
        _frontier_count_violation("missing_projection_certificate")
    if count_certificate is None:
        _frontier_count_violation("missing_count_certificate")
    if count_certificate.cursor != projection_certificate.cursor:
        _frontier_count_violation("count_certificate_cursor_mismatch")

    cursor_weights = dict(projection_certificate.cursor.weighted_states)
    projected_branch_by_key = _group_by_semantic_key(
        projection_certificate.branch_certificates,
        _branch_semantic_key,
    )
    count_branch_by_key: dict[tuple[object, ...], list[tuple[int, object]]] = {}
    count_terminal_by_key: dict[
        tuple[object, ...],
        list[tuple[int, object, object]],
    ] = {}

    for state_key, cursor_weight, state_certificate in (
        count_certificate.state_count_certificates
    ):
        if cursor_weights.get(state_key) != cursor_weight:
            _frontier_count_violation("state_cursor_weight_mismatch")
        if state_certificate.state_key != state_key:
            _frontier_count_violation("state_count_key_mismatch")
        for branch_term in state_certificate.branch_terms:
            key = _branch_semantic_key(branch_term.branch_certificate)
            count_branch_by_key.setdefault(key, []).append(
                (cursor_weight, branch_term)
            )
        terminal_projection = state_certificate.terminal_projection_certificate
        if terminal_projection is not None:
            for terminal_certificate in terminal_projection.terminal_certificates:
                key = _terminal_semantic_key(terminal_certificate)
                count_terminal_by_key.setdefault(key, []).append(
                    (cursor_weight, terminal_projection, terminal_certificate)
                )

    branch_terms = _frontier_branch_coverage_terms(
        projected_branch_by_key=projected_branch_by_key,
        count_branch_by_key=count_branch_by_key,
    )
    terminal_terms = _frontier_terminal_coverage_terms(
        projection_certificate=projection_certificate,
        count_terminal_by_key=count_terminal_by_key,
    )
    branch_completion_count = sum(
        term.weighted_completion_count for term in branch_terms
    )
    terminal_completion_count = sum(
        term.weighted_completion_count for term in terminal_terms
    )
    completion_count = branch_completion_count + terminal_completion_count
    if completion_count != count_certificate.completion_count:
        _frontier_count_violation("completion_term_coverage_total_mismatch")

    return WriterFrontierCompletionTermCoverageCertificate(
        projection_certificate=projection_certificate,
        count_certificate=count_certificate,
        branch_terms=branch_terms,
        terminal_terms=terminal_terms,
        branch_completion_count=branch_completion_count,
        terminal_completion_count=terminal_completion_count,
        completion_count=completion_count,
    )


def _branch_semantic_key(certificate) -> tuple[object, ...]:
    return (
        certificate.source_state,
        certificate.successor_state,
        certificate.emitted_text,
        certificate.transition_kind,
        certificate.graph_action_surface,
        certificate.policy_family,
        certificate.events,
        certificate.transition_evidence,
        certificate.execution_capabilities,
        certificate.graph_obligation_work_evidence,
        certificate.residual_work_evidence,
        certificate.finite_relation_work_evidence,
        certificate.closure_candidate_lifecycle_evidence,
        certificate.residual_attachment_lifecycle_evidence,
        certificate.stereo_lifecycle_evidence,
    )


def _terminal_semantic_key(certificate) -> tuple[object, ...]:
    return (
        certificate.source_state,
        certificate.finalized_state,
        certificate.terminal_execution_capabilities,
        certificate.terminal_residual_work_evidence,
        certificate.terminal_stereo_lifecycle_evidence,
        certificate.graph_obligation_work_evidence,
        certificate.terminal_certificates,
    )


def _group_by_semantic_key(certificates, key_function):
    grouped: dict[tuple[object, ...], list[object]] = {}
    for certificate in certificates:
        grouped.setdefault(key_function(certificate), []).append(certificate)
    return grouped


def _frontier_branch_coverage_terms(
    *,
    projected_branch_by_key,
    count_branch_by_key,
) -> tuple[WriterFrontierBranchCompletionCoverageTerm, ...]:
    if not set(projected_branch_by_key) <= set(count_branch_by_key):
        _frontier_count_violation(
            "branch_completion_term_key_partition_mismatch"
        )
    extra_count_keys = set(count_branch_by_key) - set(projected_branch_by_key)
    if any(
        term.successor_count != 0
        for key in extra_count_keys
        for _cursor_weight, term in count_branch_by_key[key]
    ):
        _frontier_count_violation(
            "branch_completion_term_key_partition_mismatch"
        )

    terms = []
    for key, projection_certificates in projected_branch_by_key.items():
        count_terms = count_branch_by_key[key]
        if len(projection_certificates) != len(count_terms):
            _frontier_count_violation("branch_completion_term_count_mismatch")
        for projection_branch, (cursor_weight, count_term) in zip(
            projection_certificates,
            count_terms,
        ):
            count_branch = count_term.branch_certificate
            if count_branch.parent_weight <= 0:
                _frontier_count_violation(
                    "count_branch_nonpositive_parent_weight"
                )
            expected_weight = cursor_weight * count_branch.parent_weight
            if projection_branch.parent_weight != expected_weight:
                _frontier_count_violation(
                    "branch_completion_parent_weight_scale_mismatch"
                )
            terms.append(
                WriterFrontierBranchCompletionCoverageTerm(
                    projection_branch_certificate=projection_branch,
                    count_branch_term_certificate=count_term,
                    cursor_weight=cursor_weight,
                    projection_parent_weight=projection_branch.parent_weight,
                    count_parent_weight=count_branch.parent_weight,
                    successor_completion_count=count_term.successor_count,
                    weighted_completion_count=(
                        cursor_weight * count_term.successor_count
                    ),
                )
            )
    return tuple(terms)


def _frontier_terminal_coverage_terms(
    *,
    projection_certificate,
    count_terminal_by_key,
) -> tuple[WriterFrontierTerminalCompletionCoverageTerm, ...]:
    projected = {}
    terminal_projection = projection_certificate.terminal_projection_certificate
    if terminal_projection is not None:
        projected = _group_by_semantic_key(
            terminal_projection.terminal_certificates,
            _terminal_semantic_key,
        )
    if set(projected) != set(count_terminal_by_key):
        _frontier_count_violation(
            "terminal_completion_term_key_partition_mismatch"
        )

    terms = []
    for key, projection_certificates in projected.items():
        count_entries = count_terminal_by_key[key]
        if len(projection_certificates) != len(count_entries):
            _frontier_count_violation("terminal_completion_term_count_mismatch")
        for projection_terminal, (
            cursor_weight,
            state_terminal_projection,
            count_terminal,
        ) in zip(projection_certificates, count_entries):
            expected_weight = cursor_weight * count_terminal.parent_weight
            if projection_terminal.parent_weight != expected_weight:
                _frontier_count_violation(
                    "terminal_completion_parent_weight_scale_mismatch"
                )
            terms.append(
                WriterFrontierTerminalCompletionCoverageTerm(
                    projection_terminal_certificate=projection_terminal,
                    state_terminal_projection_certificate=(
                        state_terminal_projection
                    ),
                    cursor_weight=cursor_weight,
                    projection_parent_weight=projection_terminal.parent_weight,
                    count_parent_weight=count_terminal.parent_weight,
                    terminal_completion_count=(
                        state_terminal_projection.completion_count
                    ),
                    weighted_completion_count=(
                        cursor_weight
                        * state_terminal_projection.completion_count
                    ),
                )
            )
    return tuple(terms)


def _count_violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer branch completion count certificate violation: {kind}",
    )


def _frontier_count_violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer frontier completion count certificate violation: {kind}",
    )


__all__ = (
    "WriterBranchCompletionTermCertificate",
    "WriterCursorCompletionCountCertificate",
    "WriterFrontierBranchCompletionCoverageTerm",
    "WriterFrontierCompletionCountCertificate",
    "WriterFrontierCompletionTermCoverageCertificate",
    "WriterFrontierTerminalCompletionCoverageTerm",
    "WriterStateCompletionCountCertificate",
    "writer_branch_completion_term_certificate",
    "writer_cursor_completion_count_certificate",
    "writer_frontier_completion_count_certificate",
    "writer_frontier_completion_term_coverage_certificate",
    "writer_state_completion_count_certificate",
)
