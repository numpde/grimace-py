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
    checked_frontier_certificate: object | None
    count_certificate: object | None
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
    checked_frontier_certificate,
    count_certificate,
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

    return WriterOnlineChoiceCertificate(
        kind=WriterOnlineChoiceCertificateKind.TEXT,
        prefix_before=prefix,
        text=text,
        prefix_after=prefix + text,
        choice=choice,
        text_projection_certificate=text_projection_certificate,
        snapshot_step_certificate=snapshot_step_certificate,
        terminal_projection_certificate=None,
        checked_frontier_certificate=checked_frontier_certificate,
        count_certificate=count_certificate,
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
    checked_frontier_certificate,
    count_certificate,
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
    if checked_frontier_certificate is not None:
        if terminal_projection_certificate.source_cursor != (
            checked_frontier_certificate.cursor
        ):
            _violation("terminal_projection_frontier_cursor_mismatch")
        if terminal_projection_certificate is not (
            checked_frontier_certificate.terminal_projection_certificate
        ):
            _violation("terminal_projection_not_in_frontier_certificate")

    return WriterOnlineChoiceCertificate(
        kind=WriterOnlineChoiceCertificateKind.EOS,
        prefix_before=prefix,
        text=eos_text,
        prefix_after=None,
        choice=None,
        text_projection_certificate=None,
        snapshot_step_certificate=None,
        terminal_projection_certificate=terminal_projection_certificate,
        checked_frontier_certificate=checked_frontier_certificate,
        count_certificate=count_certificate,
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

    return WriterOnlineChoiceResultCertificate(
        prefix=prefix,
        choices=choices,
        choice_certificates=choice_certificates,
        checked_frontier_certificate=checked_frontier_certificate,
        count_certificate=count_certificate,
    )


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
