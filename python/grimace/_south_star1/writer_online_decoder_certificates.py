"""Certificates for writer-shaped online decoder choices and outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .errors import SouthStarError
from .errors import SouthStarErrorKind


class WriterOnlineChoiceCertificateKind(Enum):
    TEXT = "text"
    EOS = "eos"


@dataclass(frozen=True, slots=True)
class WriterOnlineChoiceCertificate:
    kind: WriterOnlineChoiceCertificateKind
    prefix_before: str
    text: str
    prefix_after: str | None
    choice: object | None
    text_projection_certificate: object | None
    snapshot_step_certificate: object | None
    terminal_projection_certificate: object | None
    frontier_projection_certificate: object | None
    checked_frontier_certificate: object | None
    count_certificate: object | None
    choice_count_coverage_term: object | None = None
    multiplicity: int = 1
    support_count: int = 0
    completion_count: int = 0
    branch_certificates: tuple[object, ...] = ()
    terminal_certificates: tuple[object, ...] = ()


@dataclass(frozen=True, slots=True)
class WriterOnlineChoiceResultCertificate:
    prefix: str
    choices: tuple[object, ...]
    choice_certificates: tuple[WriterOnlineChoiceCertificate, ...]
    checked_frontier_certificate: object | None
    count_certificate: object | None


def writer_online_text_choice_certificate(
    *,
    prefix: str,
    choice,
    next_state,
    snapshot_step_certificate,
    text_projection_certificate,
    frontier_projection_certificate,
    checked_frontier_certificate,
    count_certificate,
    choice_count_coverage_term=None,
    online_multiplicity: int | None = None,
    online_support_count: int | None = None,
    online_completion_count: int | None = None,
) -> WriterOnlineChoiceCertificate:
    text = getattr(choice, "emitted_text", "")
    if not text:
        _violation("text_choice_empty_emitted_text")
    if snapshot_step_certificate is None:
        _violation("missing_snapshot_step_certificate")
    if snapshot_step_certificate.emitted_text != text:
        _violation("snapshot_step_text_mismatch")
    if snapshot_step_certificate.text_projection_certificate != (
        text_projection_certificate
    ):
        _violation("snapshot_step_projection_mismatch")
    if text_projection_certificate is None:
        _violation("missing_text_projection_certificate")
    if text_projection_certificate.emitted_text != text:
        _violation("text_projection_text_mismatch")
    if text_projection_certificate.source_cursor != (
        snapshot_step_certificate.source_cursor
    ):
        _violation("text_projection_step_source_cursor_mismatch")
    if not text_projection_certificate.branch_certificates:
        _violation("text_projection_missing_branch_certificates")
    if frontier_projection_certificate is None:
        _violation("missing_frontier_projection_certificate")
    if frontier_projection_certificate.cursor != (
        text_projection_certificate.source_cursor
    ):
        _violation("frontier_projection_cursor_mismatch")
    if not any(
        projection is text_projection_certificate
        for projection in (
            frontier_projection_certificate
            .text_choice_projection_certificates
        )
    ):
        _violation("text_projection_not_in_frontier_projection")
    if (
        snapshot_step_certificate.advanced_snapshot
        != next_state.raw_state.snapshot
    ):
        _violation("text_choice_advanced_snapshot_mismatch")
    if checked_frontier_certificate is not None:
        if text_projection_certificate.source_cursor != (
            checked_frontier_certificate.cursor
        ):
            _violation("text_projection_frontier_cursor_mismatch")
        if not any(
            projection is text_projection_certificate
            for projection in (
                checked_frontier_certificate
                .text_choice_projection_certificates
            )
        ):
            _violation("text_projection_not_in_frontier_certificate")
    if online_multiplicity is None:
        online_multiplicity = text_projection_certificate.immediate_multiplicity
    if online_support_count is None:
        online_support_count = getattr(choice, "support_count", 0) or 0
    if online_completion_count is None:
        online_completion_count = getattr(choice, "completion_count", 0) or 0
    _validate_text_choice_count_coverage(
        checked_frontier_certificate=checked_frontier_certificate,
        text_projection_certificate=text_projection_certificate,
        choice_count_coverage_term=choice_count_coverage_term,
        online_multiplicity=online_multiplicity,
        online_support_count=online_support_count,
        online_completion_count=online_completion_count,
    )

    return WriterOnlineChoiceCertificate(
        kind=WriterOnlineChoiceCertificateKind.TEXT,
        prefix_before=prefix,
        text=text,
        prefix_after=prefix + text,
        choice=choice,
        text_projection_certificate=text_projection_certificate,
        snapshot_step_certificate=snapshot_step_certificate,
        terminal_projection_certificate=None,
        frontier_projection_certificate=frontier_projection_certificate,
        checked_frontier_certificate=checked_frontier_certificate,
        count_certificate=count_certificate,
        choice_count_coverage_term=choice_count_coverage_term,
        multiplicity=online_multiplicity,
        support_count=online_support_count,
        completion_count=online_completion_count,
        branch_certificates=tuple(
            text_projection_certificate.branch_certificates
        ),
        terminal_certificates=(),
    )


def writer_online_eos_choice_certificate(
    *,
    prefix: str,
    eos_text: str,
    terminal,
    terminal_projection_certificate,
    frontier_projection_certificate,
    checked_frontier_certificate,
    count_certificate,
    terminal_choice_count_coverage_term=None,
    online_multiplicity: int | None = None,
    online_support_count: int | None = None,
    online_completion_count: int | None = None,
) -> WriterOnlineChoiceCertificate:
    if eos_text != "<EOS>":
        _violation("eos_choice_text_mismatch")
    if terminal is None:
        _violation("missing_terminal")
    if terminal_projection_certificate is None:
        _violation("missing_terminal_projection_certificate")
    if terminal_projection_certificate.terminal is not terminal:
        _violation("terminal_projection_terminal_mismatch")
    if not terminal_projection_certificate.terminal_certificates:
        _violation("terminal_projection_missing_terminal_certificates")
    if frontier_projection_certificate is None:
        _violation("missing_frontier_projection_certificate")
    if frontier_projection_certificate.cursor != (
        terminal_projection_certificate.source_cursor
    ):
        _violation("frontier_projection_cursor_mismatch")
    if terminal_projection_certificate is not (
        frontier_projection_certificate.terminal_projection_certificate
    ):
        _violation("terminal_projection_not_in_frontier_projection")
    if checked_frontier_certificate is not None:
        if terminal_projection_certificate.source_cursor != (
            checked_frontier_certificate.cursor
        ):
            _violation("terminal_projection_frontier_cursor_mismatch")
        if terminal_projection_certificate is not (
            checked_frontier_certificate.terminal_projection_certificate
        ):
            _violation("terminal_projection_not_in_frontier_certificate")
    if online_multiplicity is None:
        online_multiplicity = terminal_projection_certificate.multiplicity
    if online_support_count is None:
        online_support_count = terminal_projection_certificate.support_count
    if online_completion_count is None:
        online_completion_count = terminal_projection_certificate.completion_count
    _validate_terminal_choice_count_coverage(
        checked_frontier_certificate=checked_frontier_certificate,
        terminal_projection_certificate=terminal_projection_certificate,
        terminal_choice_count_coverage_term=terminal_choice_count_coverage_term,
        online_multiplicity=online_multiplicity,
        online_support_count=online_support_count,
        online_completion_count=online_completion_count,
    )

    return WriterOnlineChoiceCertificate(
        kind=WriterOnlineChoiceCertificateKind.EOS,
        prefix_before=prefix,
        text=eos_text,
        prefix_after=None,
        choice=None,
        text_projection_certificate=None,
        snapshot_step_certificate=None,
        terminal_projection_certificate=terminal_projection_certificate,
        frontier_projection_certificate=frontier_projection_certificate,
        checked_frontier_certificate=checked_frontier_certificate,
        count_certificate=count_certificate,
        choice_count_coverage_term=terminal_choice_count_coverage_term,
        multiplicity=online_multiplicity,
        support_count=online_support_count,
        completion_count=online_completion_count,
        branch_certificates=(),
        terminal_certificates=terminal_projection_certificate.terminal_certificates,
    )


def writer_online_choice_result_certificate(
    *,
    prefix: str,
    choices: tuple[object, ...],
    choice_certificates: tuple[WriterOnlineChoiceCertificate, ...],
    checked_frontier_certificate,
    count_certificate,
) -> WriterOnlineChoiceResultCertificate:
    if len(choices) != len(choice_certificates):
        _violation("choice_count_mismatch")
    for choice, certificate in zip(choices, choice_certificates):
        if certificate.choice is not None and getattr(choice, "is_eos", False):
            if certificate.choice is not None:
                _choice_violation("choice_type_mismatch")
        elif (
            not getattr(choice, "is_eos", False)
            and certificate.choice is None
        ):
            _choice_violation("choice_type_mismatch")
        if not getattr(choice, "is_eos", False):
            if getattr(certificate, "choice", None) is not None:
                if (
                    not hasattr(certificate.choice, "emitted_text")
                    or certificate.choice.emitted_text
                    != choice.text
                ):
                    _choice_violation("choice_mismatch")
            elif getattr(certificate, "text_projection_certificate", None) is None:
                _choice_violation("choice_mismatch")
        if getattr(choice, "is_eos", False) and certificate.choice is not None:
            if certificate.kind is not WriterOnlineChoiceCertificateKind.EOS:
                _choice_violation("choice_type_mismatch")
        elif not getattr(choice, "is_eos", False) and certificate.choice is None:
            _choice_violation("choice_type_mismatch")
        if (
            hasattr(choice, "text")
            and getattr(certificate, "text", None) != choice.text
        ):
            _choice_violation("choice_mismatch")
        if certificate.prefix_before != prefix:
            _violation("choice_prefix_before_mismatch")
        if (
            checked_frontier_certificate is not None
            and certificate.checked_frontier_certificate is not (
                checked_frontier_certificate
            )
        ):
            _violation("checked_frontier_certificate_mismatch")
        if (
            count_certificate is not None
            and certificate.count_certificate is not count_certificate
        ):
            _violation("count_certificate_mismatch")
        if getattr(choice, "multiplicity", None) != certificate.multiplicity:
            _choice_violation("choice_multiplicity_certificate_mismatch")
        if getattr(choice, "support_count", None) != certificate.support_count:
            _choice_violation("choice_support_count_certificate_mismatch")
        if (
            getattr(choice, "completion_count", None)
            != certificate.completion_count
        ):
            _choice_violation("choice_completion_count_certificate_mismatch")
        _validate_result_choice_coverage_term(
            certificate=certificate,
            checked_frontier_certificate=checked_frontier_certificate,
        )

    return WriterOnlineChoiceResultCertificate(
        prefix=prefix,
        choices=choices,
        choice_certificates=choice_certificates,
        checked_frontier_certificate=checked_frontier_certificate,
        count_certificate=count_certificate,
    )


def _validate_text_choice_count_coverage(
    *,
    checked_frontier_certificate,
    text_projection_certificate,
    choice_count_coverage_term,
    online_multiplicity: int,
    online_support_count: int,
    online_completion_count: int,
) -> None:
    if checked_frontier_certificate is None:
        return
    choice_coverage = getattr(
        checked_frontier_certificate,
        "choice_count_coverage_certificate",
        None,
    )
    if choice_coverage is None:
        return
    if choice_count_coverage_term is None:
        _violation("missing_choice_count_coverage_term")
    matches = tuple(
        term
        for term in choice_coverage.text_choice_terms
        if term.text_projection_certificate is text_projection_certificate
    )
    if len(matches) != 1 or choice_count_coverage_term is not matches[0]:
        _violation("choice_coverage_term_mismatch")
    if online_support_count != choice_count_coverage_term.support_count:
        _violation("online_support_count_coverage_mismatch")
    if online_completion_count != choice_count_coverage_term.completion_count:
        _violation("online_completion_count_coverage_mismatch")
    if online_multiplicity != text_projection_certificate.immediate_multiplicity:
        _violation("online_multiplicity_projection_mismatch")


def _validate_terminal_choice_count_coverage(
    *,
    checked_frontier_certificate,
    terminal_projection_certificate,
    terminal_choice_count_coverage_term,
    online_multiplicity: int,
    online_support_count: int,
    online_completion_count: int,
) -> None:
    if checked_frontier_certificate is None:
        return
    choice_coverage = getattr(
        checked_frontier_certificate,
        "choice_count_coverage_certificate",
        None,
    )
    if choice_coverage is None:
        return
    if terminal_choice_count_coverage_term is None:
        _violation("missing_terminal_choice_count_coverage_term")
    if (
        terminal_choice_count_coverage_term
        is not choice_coverage.terminal_choice_term
    ):
        _violation("terminal_choice_coverage_term_mismatch")
    if online_support_count != terminal_choice_count_coverage_term.support_count:
        _violation("online_terminal_support_count_coverage_mismatch")
    if (
        online_completion_count
        != terminal_choice_count_coverage_term.completion_count
    ):
        _violation("online_terminal_completion_count_coverage_mismatch")
    if online_multiplicity != terminal_projection_certificate.multiplicity:
        _violation("online_terminal_multiplicity_projection_mismatch")


def _validate_result_choice_coverage_term(
    *,
    certificate,
    checked_frontier_certificate,
) -> None:
    if checked_frontier_certificate is None:
        return
    choice_coverage = getattr(
        checked_frontier_certificate,
        "choice_count_coverage_certificate",
        None,
    )
    if choice_coverage is None:
        return
    if certificate.kind is WriterOnlineChoiceCertificateKind.TEXT:
        matches = tuple(
            term
            for term in choice_coverage.text_choice_terms
            if term.text_projection_certificate
            is certificate.text_projection_certificate
        )
        if (
            len(matches) != 1
            or certificate.choice_count_coverage_term is not matches[0]
        ):
            _choice_violation("choice_coverage_term_mismatch")
    elif (
        certificate.choice_count_coverage_term
        is not choice_coverage.terminal_choice_term
    ):
        _choice_violation("terminal_choice_coverage_term_mismatch")


def _choice_violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer online choice certificate violation: {kind}",
    )


def _violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer online choice certificate violation: {kind}",
    )


__all__ = (
    "WriterOnlineChoiceCertificateKind",
    "WriterOnlineChoiceCertificate",
    "WriterOnlineChoiceResultCertificate",
    "writer_online_text_choice_certificate",
    "writer_online_eos_choice_certificate",
    "writer_online_choice_result_certificate",
)
