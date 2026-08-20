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
    terminal_support_identities: tuple[object, ...]
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
        for support, certificate in zip(supports, branch_certificates):
            if certificate.parent_weight != support.parent_weight:
                _choice_violation("branch_certificate_parent_weight_mismatch")
            if certificate.branch_ordinal != support.branch_ordinal:
                _choice_violation("branch_certificate_ordinal_mismatch")
            if certificate.source_state != support.source_state:
                _choice_violation("branch_certificate_source_mismatch")
            if certificate.successor_state != support.successor_state:
                _choice_violation("branch_certificate_successor_mismatch")

        expected_successor = _cursor_like(
            choice.successor,
            _weighted_successor_states_from_branch_certificates(
                branch_certificates
            ),
        )
        support_expected_successor = _cursor_like(
            choice.successor,
            _weighted_successor_states(supports),
        )
        if expected_successor != support_expected_successor:
            _choice_violation("branch_certificate_successor_weight_mismatch")
        if choice.successor != expected_successor:
            _choice_violation("choice_successor_cursor_mismatch")

        immediate_multiplicity = sum(
            certificate.parent_weight
            for certificate in branch_certificates
        )
        support_multiplicity = sum(support.parent_weight for support in supports)
        if immediate_multiplicity != support_multiplicity:
            _choice_violation(
                "branch_certificate_support_weight_total_mismatch"
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
    for support, certificate in zip(terminal_supports, terminal_certificates):
        if certificate.source_state != support.source_state:
            _terminal_violation("terminal_certificate_source_mismatch")
        if certificate.finalized_state != support.finalized_state:
            _terminal_violation("terminal_certificate_finalized_mismatch")
        if certificate.parent_weight != support.parent_weight:
            _terminal_violation("terminal_certificate_parent_weight_mismatch")
        if certificate.terminal_ordinal != support.terminal_ordinal:
            _terminal_violation("terminal_certificate_ordinal_mismatch")
        if certificate.terminal_support_key != support.terminal_support_key:
            _terminal_violation("terminal_certificate_key_mismatch")

    expected_finalized_cursor = _cursor_like(
        terminal.finalized_cursor,
        _weighted_finalized_states_from_terminal_certificates(
            terminal_certificates
        ),
    )
    support_expected_finalized_cursor = _cursor_like(
        terminal.finalized_cursor,
        _weighted_finalized_states(terminal_supports),
    )
    if expected_finalized_cursor != support_expected_finalized_cursor:
        _terminal_violation("terminal_certificate_finalized_weight_mismatch")
    if terminal.finalized_cursor != expected_finalized_cursor:
        _terminal_violation("terminal_finalized_cursor_mismatch")

    multiplicity = sum(
        certificate.parent_weight
        for certificate in terminal_certificates
    )
    support_multiplicity = sum(
        support.parent_weight for support in terminal_supports
    )
    if multiplicity != support_multiplicity:
        _terminal_violation("terminal_certificate_support_weight_total_mismatch")
    if terminal.multiplicity != multiplicity:
        _terminal_violation("terminal_multiplicity_mismatch")

    return WriterTerminalProjectionCertificate(
        source_cursor=source_cursor,
        terminal=terminal,
        terminal_certificates=terminal_certificates,
        terminal_support_identities=tuple(
            certificate.terminal_support_key
            for certificate in terminal_certificates
        ),
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


def _weighted_successor_states_from_branch_certificates(
    certificates: tuple[object, ...],
) -> tuple[tuple[object, int], ...]:
    weighted: Counter[object] = Counter()
    for certificate in certificates:
        weighted[certificate.successor_state] += certificate.parent_weight
    return tuple(weighted.items())


def _weighted_finalized_states(
    supports: tuple[object, ...],
) -> tuple[tuple[object, int], ...]:
    weighted: Counter[object] = Counter()
    for support in supports:
        weighted[support.finalized_state] += support.parent_weight
    return tuple(weighted.items())


def _weighted_finalized_states_from_terminal_certificates(
    certificates: tuple[object, ...],
) -> tuple[tuple[object, int], ...]:
    weighted: Counter[object] = Counter()
    for certificate in certificates:
        weighted[certificate.finalized_state] += certificate.parent_weight
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
