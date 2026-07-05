"""Diagnostic certificates for frontier-owned writer diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import SouthStarError
from .errors import SouthStarErrorKind


@dataclass(frozen=True, slots=True)
class WriterWorkEnvelopeViolationCertificate:
    category: str
    violation: object
    evidence: object


@dataclass(frozen=True, slots=True)
class WriterUnsupportedCapabilityCertificate:
    capability: object
    source: str
    source_certificates: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class WriterDiagnosticBlockerCertificate:
    category: str
    blocker: object


@dataclass(frozen=True, slots=True)
class WriterDiagnosticsCertificate:
    cursor: object
    blocked: bool
    graph_policy_blocker_certificates: tuple[WriterDiagnosticBlockerCertificate, ...]
    stereo_policy_blocker_certificates: tuple[WriterDiagnosticBlockerCertificate, ...]
    execution_capabilities: frozenset[object]
    terminal_execution_capabilities: frozenset[object]
    unsupported_execution_capability_certificates: tuple[
        WriterUnsupportedCapabilityCertificate,
        ...,
    ]
    unsupported_terminal_execution_capability_certificates: tuple[
        WriterUnsupportedCapabilityCertificate,
        ...,
    ]
    work_envelope_violation_certificates: tuple[
        WriterWorkEnvelopeViolationCertificate,
        ...,
    ]
    text_choice_projection_certificates: tuple[object, ...]
    terminal_projection_certificate: object | None
    branch_certificates: tuple[object, ...]
    terminal_certificates: tuple[object, ...]
    count_certificate: object | None = None


def writer_diagnostics_certificate(
    *,
    cursor,
    diagnostics,
    branch_batch,
    count_certificate=None,
) -> WriterDiagnosticsCertificate:
    choice_texts = tuple(
        cert.emitted_text
        for cert in getattr(branch_batch, "text_choice_projection_certificates", ())
    )
    if diagnostics.choice_texts != choice_texts:
        _diagnostic_violation("choice_texts_mismatch")

    for projection in getattr(
        branch_batch,
        "text_choice_projection_certificates",
        (),
    ):
        if getattr(projection, "source_cursor", None) != cursor:
            _diagnostic_violation("projection_source_cursor_mismatch")

    terminal_projection_certificate = (
        getattr(branch_batch, "terminal_projection_certificate", None)
    )
    if (
        terminal_projection_certificate is not None
        and getattr(terminal_projection_certificate, "source_cursor", None)
        != cursor
    ):
        _diagnostic_violation("terminal_projection_source_cursor_mismatch")

    if bool(terminal_projection_certificate) != diagnostics.has_eos:
        _diagnostic_violation("has_eos_mismatch")

    graph_blockers = tuple(
        WriterDiagnosticBlockerCertificate(
            category="graph",
            blocker=blocker,
        )
        for blocker in diagnostics.graph_policy_blockers
    )
    stereo_blockers = tuple(
        WriterDiagnosticBlockerCertificate(
            category="stereo",
            blocker=blocker,
        )
        for blocker in diagnostics.stereo_policy_blockers
    )

    graph_obligation_evidences = tuple(diagnostics.graph_obligation_work_evidence)
    residual_evidences = tuple(diagnostics.residual_work_evidence)
    terminal_residual_evidences = tuple(
        diagnostics.terminal_residual_work_evidence
    )
    finite_relation_evidences = tuple(diagnostics.finite_relation_work_evidence)

    evidence_pool = (
        *graph_obligation_evidences,
        *residual_evidences,
        *terminal_residual_evidences,
        *finite_relation_evidences,
    )

    work_envelope_violation_certificates = (
        *_work_violation_certificates(
            category="graph_obligation",
            violations=diagnostics.graph_obligation_work_envelope_violations,
            evidence_pool=evidence_pool,
        ),
        *_work_violation_certificates(
            category="residual_work",
            violations=diagnostics.residual_work_envelope_violations,
            evidence_pool=evidence_pool,
        ),
        *_work_violation_certificates(
            category="terminal_residual_work",
            violations=diagnostics.terminal_residual_work_envelope_violations,
            evidence_pool=evidence_pool,
        ),
        *_work_violation_certificates(
            category="finite_relation_work",
            violations=diagnostics.finite_relation_work_envelope_violations,
            evidence_pool=evidence_pool,
        ),
    )

    blocked = bool(
        graph_blockers
        or stereo_blockers
        or work_envelope_violation_certificates
        or diagnostics.unsupported_execution_capabilities
        or diagnostics.unsupported_terminal_execution_capabilities
    )
    if diagnostics.blocked != blocked:
        _diagnostic_violation("blocked_mismatch")

    if blocked and choice_texts:
        _diagnostic_violation("blocked_has_projection_choice_texts")

    if diagnostics.blocked and not (
        graph_blockers
        or stereo_blockers
        or work_envelope_violation_certificates
        or diagnostics.unsupported_execution_capabilities
        or diagnostics.unsupported_terminal_execution_capabilities
    ):
        _diagnostic_violation("blocked_without_blockers_or_violations")

    branch_certificates = _branch_certificates(branch_batch)
    terminal_certificates = _terminal_certificates(branch_batch)

    unsupported_execution_capability_certificates = (
        _unsupported_capability_certificates(
            unsupported=diagnostics.unsupported_execution_capabilities,
            source="branch_support",
            branch_certificates=branch_certificates,
            terminal_certificates=terminal_certificates,
        )
    )
    unsupported_terminal_execution_capability_certificates = (
        _unsupported_capability_certificates(
            unsupported=(
                diagnostics.unsupported_terminal_execution_capabilities
            ),
            source="terminal_support",
            branch_certificates=branch_certificates,
            terminal_certificates=terminal_certificates,
        )
    )

    if count_certificate is not None and count_certificate.cursor != cursor:
        _diagnostic_violation("count_certificate_cursor_mismatch")

    return WriterDiagnosticsCertificate(
        cursor=cursor,
        blocked=diagnostics.blocked,
        graph_policy_blocker_certificates=graph_blockers,
        stereo_policy_blocker_certificates=stereo_blockers,
        execution_capabilities=diagnostics.execution_capabilities,
        terminal_execution_capabilities=diagnostics.terminal_execution_capabilities,
        unsupported_execution_capability_certificates=(
            unsupported_execution_capability_certificates
        ),
        unsupported_terminal_execution_capability_certificates=(
            unsupported_terminal_execution_capability_certificates
        ),
        work_envelope_violation_certificates=work_envelope_violation_certificates,
        text_choice_projection_certificates=tuple(
            cert
            for cert in getattr(
                branch_batch,
                "text_choice_projection_certificates",
                (),
            )
        ),
        terminal_projection_certificate=terminal_projection_certificate,
        branch_certificates=branch_certificates,
        terminal_certificates=terminal_certificates,
        count_certificate=count_certificate,
    )


def _branch_certificates(branch_batch) -> tuple[object, ...]:
    return tuple(
        certificate
        for support in getattr(branch_batch, "supports", ())
        for certificate in (support.checked_branch_certificate,)
        if certificate is not None
    )


def _terminal_certificates(branch_batch) -> tuple[object, ...]:
    return tuple(
        certificate
        for support in getattr(branch_batch, "terminal_supports", ())
        for certificate in (support.checked_terminal_certificate,)
        if certificate is not None
    )


def _work_violation_certificates(
    *,
    category: str,
    violations: tuple[object, ...],
    evidence_pool: tuple[object, ...],
) -> tuple[WriterWorkEnvelopeViolationCertificate, ...]:
    certificates: list[WriterWorkEnvelopeViolationCertificate] = []
    for violation in violations:
        evidence = getattr(violation, "evidence", None)
        if evidence not in evidence_pool:
            _diagnostic_violation(f"{category}_violation_evidence_missing")
        certificates.append(
            WriterWorkEnvelopeViolationCertificate(
                category=category,
                violation=violation,
                evidence=evidence,
            )
        )

    return tuple(certificates)


def _unsupported_capability_certificates(
    *,
    unsupported: frozenset[object],
    source: str,
    branch_certificates: tuple[object, ...],
    terminal_certificates: tuple[object, ...],
) -> tuple[WriterUnsupportedCapabilityCertificate, ...]:
    certificates: list[WriterUnsupportedCapabilityCertificate] = []
    for capability in unsupported:
        source_certificates = tuple(
            certificate
            for certificate in (*branch_certificates, *terminal_certificates)
            if capability in getattr(
                certificate,
                "execution_capabilities",
                (),
            )
            or capability in getattr(
                certificate,
                "terminal_execution_capabilities",
                (),
            )
        )
        certificates.append(
            WriterUnsupportedCapabilityCertificate(
                capability=capability,
                source=source,
                source_certificates=source_certificates,
            )
        )
    return tuple(certificates)


def _diagnostic_violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer diagnostic certificate violation: {kind}",
    )


__all__ = (
    "WriterDiagnosticsCertificate",
    "WriterDiagnosticBlockerCertificate",
    "WriterWorkEnvelopeViolationCertificate",
    "WriterUnsupportedCapabilityCertificate",
    "writer_diagnostics_certificate",
)
