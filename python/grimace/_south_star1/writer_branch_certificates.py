"""Aggregate certificates for checked writer branch and terminal supports."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .writer_events import WriterRingEndpointEmitted
from .writer_events import WriterRingEndpointPaired
from .writer_events import WriterRingLabelAllocated
from .writer_events import WriterRingLabelReleased
from .writer_terminal_certificates import WriterTerminalCertificateKind


@dataclass(frozen=True, slots=True)
class WriterCheckedBranchSupportCertificate:
    source_state: object
    successor_state: object
    emitted_text: str
    transition_kind: object
    graph_action_surface: object | None
    policy_family: object | None
    events: tuple[object, ...]
    transition_evidence: object
    execution_capabilities: frozenset[object]
    graph_obligation_work_evidence: tuple[object, ...]
    residual_work_evidence: tuple[object, ...]
    finite_relation_work_evidence: tuple[object, ...]
    ring_lifecycle_events: tuple[object, ...]
    closure_candidate_resolution_evidence: tuple[object, ...]
    closure_candidate_lifecycle_evidence: tuple[object, ...]
    closure_candidate_branch_certificates: tuple[object, ...]
    residual_attachment_lifecycle_evidence: tuple[object, ...]
    residual_attachment_branch_certificates: tuple[object, ...]
    stereo_lifecycle_evidence: tuple[object, ...]
    stereo_branch_certificates: tuple[object, ...]
    residual_attachment_policy_evidence: tuple[object, ...]
    capability_coverage_certificate: object
    successor_state_certificate: object


@dataclass(frozen=True, slots=True)
class WriterCheckedTerminalSupportCertificate:
    source_state: object
    finalized_state: object
    parent_weight: int
    terminal_execution_capabilities: frozenset[object]
    terminal_residual_work_evidence: tuple[object, ...]
    terminal_stereo_lifecycle_evidence: tuple[object, ...]
    graph_obligation_work_evidence: tuple[object, ...]
    terminal_certificates: tuple[object, ...]


def writer_checked_branch_support_certificate(
    *,
    source_state,
    successor_state,
    emitted_text: str,
    transition_kind,
    graph_action_surface,
    policy_family,
    events: tuple[object, ...],
    transition_evidence,
    execution_capabilities: frozenset[object],
    graph_obligation_work_evidence: tuple[object, ...],
    residual_work_evidence: tuple[object, ...],
    finite_relation_work_evidence: tuple[object, ...],
    closure_candidate_resolution_evidence: tuple[object, ...],
    closure_candidate_lifecycle_evidence: tuple[object, ...],
    closure_candidate_branch_certificates: tuple[object, ...],
    residual_attachment_lifecycle_evidence: tuple[object, ...],
    residual_attachment_branch_certificates: tuple[object, ...],
    stereo_lifecycle_evidence: tuple[object, ...],
    stereo_branch_certificates: tuple[object, ...],
    residual_attachment_policy_evidence: tuple[object, ...],
    capability_coverage_certificate,
    successor_state_certificate,
) -> WriterCheckedBranchSupportCertificate:
    if not emitted_text:
        _branch_violation("missing_emitted_text")

    ring_lifecycle_events = _ring_lifecycle_events(events)
    _validate_ring_lifecycle_events(events, ring_lifecycle_events)
    _validate_graph_action_surface(
        graph_action_surface=graph_action_surface,
        policy_family=policy_family,
    )

    if (
        capability_coverage_certificate is None
        or capability_coverage_certificate.execution_capabilities
        != execution_capabilities
    ):
        _branch_violation(
            "capability_coverage_execution_mismatch"
        )
    if (
        capability_coverage_certificate.covered_capabilities
        != execution_capabilities
    ):
        _branch_violation("capability_coverage_incomplete")
    _validate_successor_state_certificate(
        successor_state_certificate=successor_state_certificate,
        source_state=source_state,
        successor_state=successor_state,
        emitted_text=emitted_text,
        transition_kind=transition_kind,
        graph_action_surface=graph_action_surface,
        policy_family=policy_family,
        events=events,
        transition_evidence=transition_evidence,
    )

    _validate_closure_candidate_certificates(
        execution_capabilities=execution_capabilities,
        resolution_evidence=closure_candidate_resolution_evidence,
        lifecycle_evidence=closure_candidate_lifecycle_evidence,
        certificates=closure_candidate_branch_certificates,
    )
    _validate_residual_attachment_certificates(
        execution_capabilities=execution_capabilities,
        lifecycle_evidence=residual_attachment_lifecycle_evidence,
        certificates=residual_attachment_branch_certificates,
    )
    _validate_stereo_certificates(
        execution_capabilities=execution_capabilities,
        lifecycle_evidence=stereo_lifecycle_evidence,
        certificates=stereo_branch_certificates,
    )
    _validate_stereo_work_evidence(
        stereo_lifecycle_evidence=stereo_lifecycle_evidence,
        residual_work_evidence=residual_work_evidence,
    )
    _validate_closure_finite_relation_evidence(
        transition_kind=transition_kind,
        finite_relation_work_evidence=finite_relation_work_evidence,
    )

    return WriterCheckedBranchSupportCertificate(
        source_state=source_state,
        successor_state=successor_state,
        emitted_text=emitted_text,
        transition_kind=transition_kind,
        graph_action_surface=graph_action_surface,
        policy_family=policy_family,
        events=events,
        transition_evidence=transition_evidence,
        execution_capabilities=execution_capabilities,
        graph_obligation_work_evidence=graph_obligation_work_evidence,
        residual_work_evidence=residual_work_evidence,
        finite_relation_work_evidence=finite_relation_work_evidence,
        ring_lifecycle_events=ring_lifecycle_events,
        closure_candidate_resolution_evidence=(
            closure_candidate_resolution_evidence
        ),
        closure_candidate_lifecycle_evidence=(
            closure_candidate_lifecycle_evidence
        ),
        closure_candidate_branch_certificates=(
            closure_candidate_branch_certificates
        ),
        residual_attachment_lifecycle_evidence=(
            residual_attachment_lifecycle_evidence
        ),
        residual_attachment_branch_certificates=(
            residual_attachment_branch_certificates
        ),
        stereo_lifecycle_evidence=stereo_lifecycle_evidence,
        stereo_branch_certificates=stereo_branch_certificates,
        residual_attachment_policy_evidence=residual_attachment_policy_evidence,
        capability_coverage_certificate=capability_coverage_certificate,
        successor_state_certificate=successor_state_certificate,
    )


def writer_checked_terminal_support_certificate(
    *,
    source_state,
    finalized_state,
    parent_weight: int,
    terminal_execution_capabilities: frozenset[object],
    terminal_residual_work_evidence: tuple[object, ...],
    terminal_stereo_lifecycle_evidence: tuple[object, ...],
    graph_obligation_work_evidence: tuple[object, ...],
    terminal_certificates: tuple[object, ...],
) -> WriterCheckedTerminalSupportCertificate:
    if parent_weight <= 0:
        _terminal_violation("nonpositive_parent_weight")

    kinds = frozenset(certificate.kind for certificate in terminal_certificates)
    if WriterTerminalCertificateKind.GRAPH_COMPLETE not in kinds:
        _terminal_violation("missing_graph_complete_certificate")
    if WriterTerminalCertificateKind.FINALIZED_STATE not in kinds:
        _terminal_violation("missing_finalized_state_certificate")
    if (
        terminal_execution_capabilities
        or terminal_residual_work_evidence
        or terminal_stereo_lifecycle_evidence
    ) and WriterTerminalCertificateKind.STEREO_TERMINALIZED not in kinds:
        _terminal_violation("missing_stereo_terminalized_certificate")

    for certificate in terminal_certificates:
        if certificate.source_state != source_state:
            _terminal_violation("certificate_source_state_mismatch")
        if certificate.finalized_state != finalized_state:
            _terminal_violation("certificate_finalized_state_mismatch")

    lifecycle_work = tuple(
        item
        for evidence in terminal_stereo_lifecycle_evidence
        for item in evidence.residual_work_evidence
    )
    if terminal_residual_work_evidence != lifecycle_work:
        _terminal_violation("terminal_residual_work_lifecycle_mismatch")

    return WriterCheckedTerminalSupportCertificate(
        source_state=source_state,
        finalized_state=finalized_state,
        parent_weight=parent_weight,
        terminal_execution_capabilities=terminal_execution_capabilities,
        terminal_residual_work_evidence=terminal_residual_work_evidence,
        terminal_stereo_lifecycle_evidence=terminal_stereo_lifecycle_evidence,
        graph_obligation_work_evidence=graph_obligation_work_evidence,
        terminal_certificates=terminal_certificates,
    )


def _ring_lifecycle_events(events: tuple[object, ...]) -> tuple[object, ...]:
    return tuple(
        event
        for event in events
        if isinstance(
            event,
            (
                WriterRingLabelAllocated,
                WriterRingEndpointEmitted,
                WriterRingEndpointPaired,
                WriterRingLabelReleased,
            ),
        )
    )


def _validate_ring_lifecycle_events(
    events: tuple[object, ...],
    ring_lifecycle_events: tuple[object, ...],
) -> None:
    has_endpoint = any(
        isinstance(event, (WriterRingEndpointEmitted, WriterRingEndpointPaired))
        for event in events
    )
    if has_endpoint and not ring_lifecycle_events:
        _branch_violation("ring_endpoint_lacks_lifecycle_events")


def _validate_graph_action_surface(
    *,
    graph_action_surface,
    policy_family,
) -> None:
    if graph_action_surface is None:
        return
    surface_policy_family = getattr(graph_action_surface, "policy_family", None)
    if (
        surface_policy_family is not None
        and surface_policy_family is not policy_family
    ):
        _branch_violation("graph_action_surface_policy_family_mismatch")


def _validate_closure_candidate_certificates(
    *,
    execution_capabilities: frozenset[object],
    resolution_evidence: tuple[object, ...],
    lifecycle_evidence: tuple[object, ...],
    certificates: tuple[object, ...],
) -> None:
    for certificate in certificates:
        if certificate.capability not in execution_capabilities:
            _branch_violation("closure_certificate_capability_missing")
        if certificate.lifecycle_evidence not in lifecycle_evidence:
            _branch_violation("closure_certificate_lifecycle_missing")
        if certificate.resolution_evidence not in resolution_evidence:
            _branch_violation("closure_certificate_resolution_missing")


def _validate_residual_attachment_certificates(
    *,
    execution_capabilities: frozenset[object],
    lifecycle_evidence: tuple[object, ...],
    certificates: tuple[object, ...],
) -> None:
    for certificate in certificates:
        if certificate.capability not in execution_capabilities:
            _branch_violation(
                "residual_attachment_certificate_capability_missing"
            )
        if certificate.lifecycle_evidence not in lifecycle_evidence:
            _branch_violation(
                "residual_attachment_certificate_lifecycle_missing"
            )


def _validate_stereo_certificates(
    *,
    execution_capabilities: frozenset[object],
    lifecycle_evidence: tuple[object, ...],
    certificates: tuple[object, ...],
) -> None:
    for certificate in certificates:
        if certificate.capability not in execution_capabilities:
            _branch_violation("stereo_certificate_capability_missing")
        if certificate.lifecycle_evidence not in lifecycle_evidence:
            _branch_violation("stereo_certificate_lifecycle_missing")


def _validate_stereo_work_evidence(
    *,
    stereo_lifecycle_evidence: tuple[object, ...],
    residual_work_evidence: tuple[object, ...],
) -> None:
    for evidence in stereo_lifecycle_evidence:
        for item in evidence.residual_work_evidence:
            if item not in residual_work_evidence:
                _branch_violation("stereo_work_evidence_missing")


def _validate_closure_finite_relation_evidence(
    *,
    transition_kind,
    finite_relation_work_evidence: tuple[object, ...],
) -> None:
    if (
        getattr(transition_kind, "name", None)
        in {"OPEN_CLOSURE_ENDPOINT", "PAIR_CLOSURE_ENDPOINT"}
        and not finite_relation_work_evidence
    ):
        _branch_violation("closure_transition_lacks_finite_relation_evidence")


def _validate_successor_state_certificate(
    *,
    successor_state_certificate,
    source_state,
    successor_state,
    emitted_text: str,
    transition_kind,
    graph_action_surface,
    policy_family,
    events: tuple[object, ...],
    transition_evidence,
) -> None:
    if successor_state_certificate is None:
        _branch_violation("missing_successor_state_certificate")
    if successor_state_certificate.source_state != source_state:
        _branch_violation("successor_certificate_source_mismatch")
    if successor_state_certificate.successor_state != successor_state:
        _branch_violation("successor_certificate_successor_mismatch")
    if successor_state_certificate.emitted_text != emitted_text:
        _branch_violation("successor_certificate_text_mismatch")
    if successor_state_certificate.transition_kind != transition_kind:
        _branch_violation("successor_certificate_transition_mismatch")
    if successor_state_certificate.graph_action_surface != graph_action_surface:
        _branch_violation("successor_certificate_graph_surface_mismatch")
    if successor_state_certificate.policy_family != policy_family:
        _branch_violation("successor_certificate_policy_family_mismatch")
    if successor_state_certificate.events != events:
        _branch_violation("successor_certificate_events_mismatch")
    if successor_state_certificate.transition_evidence != transition_evidence:
        _branch_violation("successor_certificate_evidence_mismatch")


def _branch_violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer checked branch certificate violation: {kind}",
    )


def _terminal_violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer checked terminal certificate violation: {kind}",
    )


__all__ = (
    "WriterCheckedBranchSupportCertificate",
    "WriterCheckedTerminalSupportCertificate",
    "writer_checked_branch_support_certificate",
    "writer_checked_terminal_support_certificate",
)
