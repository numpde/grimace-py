"""Certificates for public text/EOS projection over checked supports."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .errors import SouthStarError
from .errors import SouthStarErrorKind


@dataclass(frozen=True, slots=True)
class WriterTextChoiceProjectionCertificate:
    source_cursor: object
    emitted_text: str
    choice: object
    branch_certificates: tuple[object, ...]
    successor_cursor: object
    immediate_multiplicity: int
    support_count: int | None = None
    completion_count: int | None = None


@dataclass(frozen=True, slots=True)
class WriterTerminalProjectionCertificate:
    source_cursor: object
    terminal: object
    terminal_certificates: tuple[object, ...]
    finalized_cursor: object
    multiplicity: int
    support_count: int
    completion_count: int


def writer_text_choice_projection_certificates(
    *,
    source_cursor,
    choices,
    branch_supports: tuple[object, ...],
) -> tuple[WriterTextChoiceProjectionCertificate, ...]:
    source_weights = dict(getattr(source_cursor, "weighted_states", ()))
    certificates: list[WriterTextChoiceProjectionCertificate] = []
    for choice in choices.choices:
        supports = tuple(
            support
            for support in branch_supports
            if support.emitted_text == choice.emitted_text
        )
        if not supports:
            _choice_violation("choice_lacks_branch_support")
        for support in supports:
            if support.source_state not in source_weights:
                _choice_violation("branch_support_source_not_in_cursor")
            if support.parent_weight != source_weights[support.source_state]:
                _choice_violation("branch_support_parent_weight_mismatch")

        branch_certificates = tuple(
            support.checked_branch_certificate
            for support in supports
        )
        if any(certificate is None for certificate in branch_certificates):
            _choice_violation("branch_support_lacks_checked_certificate")

        expected_successor = _cursor_like(
            choice.successor,
            _weighted_successor_states(supports),
        )
        if choice.successor != expected_successor:
            _choice_violation("choice_successor_cursor_mismatch")

        immediate_multiplicity = sum(
            support.parent_weight
            for support in supports
        )
        if choice.immediate_multiplicity != immediate_multiplicity:
            _choice_violation("choice_immediate_multiplicity_mismatch")

        certificates.append(
            WriterTextChoiceProjectionCertificate(
                source_cursor=source_cursor,
                emitted_text=choice.emitted_text,
                choice=choice,
                branch_certificates=branch_certificates,
                successor_cursor=expected_successor,
                immediate_multiplicity=immediate_multiplicity,
                support_count=choice.support_count,
                completion_count=choice.completion_count,
            )
        )

    return tuple(certificates)


def writer_terminal_projection_certificate(
    *,
    source_cursor,
    terminal,
    terminal_supports: tuple[object, ...],
) -> WriterTerminalProjectionCertificate | None:
    if terminal is None:
        if terminal_supports:
            _terminal_violation("terminal_supports_without_terminal")
        return None

    if not terminal_supports:
        _terminal_violation("terminal_lacks_terminal_supports")
    source_weights = dict(getattr(source_cursor, "weighted_states", ()))
    for support in terminal_supports:
        if support.source_state not in source_weights:
            _terminal_violation("terminal_support_source_not_in_cursor")
        if support.parent_weight != source_weights[support.source_state]:
            _terminal_violation("terminal_support_parent_weight_mismatch")

    terminal_certificates = tuple(
        support.checked_terminal_certificate
        for support in terminal_supports
    )
    if any(certificate is None for certificate in terminal_certificates):
        _terminal_violation("terminal_support_lacks_checked_certificate")

    expected_finalized_cursor = _cursor_like(
        terminal.finalized_cursor,
        _weighted_finalized_states(terminal_supports),
    )
    if terminal.finalized_cursor != expected_finalized_cursor:
        _terminal_violation("terminal_finalized_cursor_mismatch")

    multiplicity = sum(
        support.parent_weight
        for support in terminal_supports
    )
    if terminal.multiplicity != multiplicity:
        _terminal_violation("terminal_multiplicity_mismatch")

    return WriterTerminalProjectionCertificate(
        source_cursor=source_cursor,
        terminal=terminal,
        terminal_certificates=terminal_certificates,
        finalized_cursor=expected_finalized_cursor,
        multiplicity=multiplicity,
        support_count=terminal.support_count,
        completion_count=terminal.completion_count,
    )


def _weighted_successor_states(
    supports: tuple[object, ...],
) -> tuple[tuple[object, int], ...]:
    weighted: Counter[object] = Counter()
    for support in supports:
        weighted[support.successor_state] += support.parent_weight
    return tuple(weighted.items())


def _weighted_finalized_states(
    supports: tuple[object, ...],
) -> tuple[tuple[object, int], ...]:
    weighted: Counter[object] = Counter()
    for support in supports:
        weighted[support.finalized_state] += support.parent_weight
    return tuple(weighted.items())


def _cursor_like(cursor, weighted_states: tuple[tuple[object, int], ...]):
    return cursor.__class__(weighted_states=weighted_states)


def _choice_violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer text choice projection certificate violation: {kind}",
    )


def _terminal_violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer terminal projection certificate violation: {kind}",
    )


__all__ = (
    "WriterTerminalProjectionCertificate",
    "WriterTextChoiceProjectionCertificate",
    "writer_terminal_projection_certificate",
    "writer_text_choice_projection_certificates",
)
