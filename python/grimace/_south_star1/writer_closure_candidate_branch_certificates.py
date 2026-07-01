"""Branch-local certificates for closure-candidate capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .writer_capabilities import _WriterExecutionCapabilityKind
from .writer_closure_candidate_lifecycle import (
    WriterClosureCandidateLifecycleOutcomeKind,
)
from .writer_events import WriterRingEndpointEmitted
from .writer_events import WriterRingLabelAllocated
from .writer_graph_obligations import WriterClosureCandidateResolutionKind
from .writer_transitions import _WriterClosureOpenObligationSourceKind
from .writer_transitions import _WriterGraphPolicyActionFamily
from .writer_transitions import WriterTransitionKind


class WriterClosureCandidateBranchCertificateKind(Enum):
    LIVE_BRANCH_RETURN_OPENED = "live_branch_return_opened"
    DEFERRED_BRANCH_RETURN_RETAINED = "deferred_branch_return_retained"
    DEFERRED_CONTROL_LIVE_RETAINED = "deferred_control_live_retained"


@dataclass(frozen=True, slots=True)
class WriterClosureCandidateBranchCertificate:
    kind: WriterClosureCandidateBranchCertificateKind
    capability: _WriterExecutionCapabilityKind
    bond: object
    resolution_evidence: object
    lifecycle_evidence: object
    graph_action_surface: object | None = None
    graph_obligation_work_evidence: tuple[object, ...] = ()


def writer_closure_candidate_branch_certificates(
    *,
    execution_capabilities: frozenset[object],
    transition_kind: object,
    graph_action_surface: object | None,
    graph_obligation_work_evidence: tuple[object, ...],
    closure_candidate_resolution_evidence: tuple[object, ...],
    closure_candidate_lifecycle_evidence: tuple[object, ...],
    events: tuple[object, ...],
) -> tuple[WriterClosureCandidateBranchCertificate, ...]:
    certificates: list[WriterClosureCandidateBranchCertificate] = []

    if (
        _WriterExecutionCapabilityKind.LIVE_BRANCH_RETURN_CLOSURE_CANDIDATE_OPEN
        in execution_capabilities
    ):
        certificates.append(
            _live_branch_return_opened_certificate(
                transition_kind=transition_kind,
                graph_action_surface=graph_action_surface,
                graph_obligation_work_evidence=graph_obligation_work_evidence,
                resolution_evidence=closure_candidate_resolution_evidence,
                lifecycle_evidence=closure_candidate_lifecycle_evidence,
                events=events,
            )
        )

    if (
        _WriterExecutionCapabilityKind.DEFERRED_BRANCH_RETURN_CLOSURE_CANDIDATE
        in execution_capabilities
    ):
        certificates.append(
            _deferred_certificate(
                certificate_kind=(
                    WriterClosureCandidateBranchCertificateKind
                    .DEFERRED_BRANCH_RETURN_RETAINED
                ),
                capability=(
                    _WriterExecutionCapabilityKind
                    .DEFERRED_BRANCH_RETURN_CLOSURE_CANDIDATE
                ),
                expected_metric="deferred_branch_return_closure_candidate_count",
                expected_resolution_kind=(
                    WriterClosureCandidateResolutionKind.DEFERRED_BRANCH_RETURN
                ),
                resolution_evidence=closure_candidate_resolution_evidence,
                lifecycle_evidence=closure_candidate_lifecycle_evidence,
                graph_obligation_work_evidence=graph_obligation_work_evidence,
                graph_action_surface=graph_action_surface,
                transition_kind=transition_kind,
            )
        )

    if (
        _WriterExecutionCapabilityKind.DEFERRED_CONTROL_LIVE_CLOSURE_CANDIDATE
        in execution_capabilities
    ):
        certificates.append(
            _deferred_certificate(
                certificate_kind=(
                    WriterClosureCandidateBranchCertificateKind
                    .DEFERRED_CONTROL_LIVE_RETAINED
                ),
                capability=(
                    _WriterExecutionCapabilityKind
                    .DEFERRED_CONTROL_LIVE_CLOSURE_CANDIDATE
                ),
                expected_metric=(
                    "deferred_control_live_closure_candidate_count"
                ),
                expected_resolution_kind=(
                    WriterClosureCandidateResolutionKind.DEFERRED_CONTROL_LIVE
                ),
                resolution_evidence=closure_candidate_resolution_evidence,
                lifecycle_evidence=closure_candidate_lifecycle_evidence,
                graph_obligation_work_evidence=graph_obligation_work_evidence,
                graph_action_surface=graph_action_surface,
                transition_kind=transition_kind,
            )
        )

    return tuple(certificates)


def _live_branch_return_opened_certificate(
    *,
    transition_kind: object,
    graph_action_surface: object | None,
    graph_obligation_work_evidence: tuple[object, ...],
    resolution_evidence: tuple[object, ...],
    lifecycle_evidence: tuple[object, ...],
    events: tuple[object, ...],
) -> WriterClosureCandidateBranchCertificate:
    if transition_kind is not WriterTransitionKind.OPEN_CLOSURE_ENDPOINT:
        _violation("live_open_capability_requires_open_transition")
    if graph_action_surface is None:
        _violation("live_open_capability_lacks_graph_action_surface")
    if graph_action_surface.policy_family is not _WriterGraphPolicyActionFamily.CLOSURE_OPEN:
        _violation("live_open_capability_lacks_closure_open_surface")
    if (
        graph_action_surface.closure_open_source_kind
        is not (
            _WriterClosureOpenObligationSourceKind
            .LIVE_BRANCH_RETURN_CLOSURE_CANDIDATE
        )
    ):
        _violation("live_open_capability_lacks_live_source")

    resolution = _single_matching_resolution(
        resolution_evidence,
        kind=WriterClosureCandidateResolutionKind.LIVE_BRANCH_RETURN,
        bond=graph_action_surface.bond,
    )
    lifecycle = _single_matching_lifecycle(
        lifecycle_evidence,
        outcome=WriterClosureCandidateLifecycleOutcomeKind.OPENED,
        bond=graph_action_surface.bond,
    )
    _require_open_ring_events_for_bond(events, graph_action_surface.bond)

    return WriterClosureCandidateBranchCertificate(
        kind=(
            WriterClosureCandidateBranchCertificateKind
            .LIVE_BRANCH_RETURN_OPENED
        ),
        capability=(
            _WriterExecutionCapabilityKind
            .LIVE_BRANCH_RETURN_CLOSURE_CANDIDATE_OPEN
        ),
        bond=graph_action_surface.bond,
        resolution_evidence=resolution,
        lifecycle_evidence=lifecycle,
        graph_action_surface=graph_action_surface,
        graph_obligation_work_evidence=graph_obligation_work_evidence,
    )


def _deferred_certificate(
    *,
    certificate_kind: WriterClosureCandidateBranchCertificateKind,
    capability: _WriterExecutionCapabilityKind,
    expected_metric: str,
    expected_resolution_kind: WriterClosureCandidateResolutionKind,
    resolution_evidence: tuple[object, ...],
    lifecycle_evidence: tuple[object, ...],
    graph_obligation_work_evidence: tuple[object, ...],
    graph_action_surface: object | None,
    transition_kind: object,
) -> WriterClosureCandidateBranchCertificate:
    if transition_kind is WriterTransitionKind.OPEN_CLOSURE_ENDPOINT:
        _violation("deferred_candidate_opened_early")
    if not any(
        getattr(evidence, expected_metric, 0)
        for evidence in graph_obligation_work_evidence
    ):
        _violation("deferred_capability_lacks_graph_metric")

    retained = tuple(
        evidence
        for evidence in lifecycle_evidence
        if (
            evidence.outcome_kind
            is WriterClosureCandidateLifecycleOutcomeKind.RETAINED_SUPPORTED
            and evidence.source_resolution.resolution_kind
            is expected_resolution_kind
        )
    )
    if not retained:
        _violation("deferred_capability_lacks_retained_lifecycle")

    lifecycle = retained[0]
    if any(
        (
            evidence.bond == lifecycle.bond
            and evidence.outcome_kind
            is WriterClosureCandidateLifecycleOutcomeKind.OPENED
        )
        for evidence in lifecycle_evidence
    ):
        _violation("deferred_candidate_opened_early")
    resolution = _single_matching_resolution(
        resolution_evidence,
        kind=expected_resolution_kind,
        bond=lifecycle.bond,
    )

    return WriterClosureCandidateBranchCertificate(
        kind=certificate_kind,
        capability=capability,
        bond=lifecycle.bond,
        resolution_evidence=resolution,
        lifecycle_evidence=lifecycle,
        graph_action_surface=graph_action_surface,
        graph_obligation_work_evidence=graph_obligation_work_evidence,
    )


def _single_matching_resolution(
    evidence: tuple[object, ...],
    *,
    kind: WriterClosureCandidateResolutionKind,
    bond,
):
    matches = tuple(
        item
        for item in evidence
        if item.resolution_kind is kind and item.bond == bond
    )
    if len(matches) != 1:
        _violation("capability_lacks_exact_resolution")
    return matches[0]


def _single_matching_lifecycle(
    evidence: tuple[object, ...],
    *,
    outcome: WriterClosureCandidateLifecycleOutcomeKind,
    bond,
):
    matches = tuple(
        item
        for item in evidence
        if item.outcome_kind is outcome and item.bond == bond
    )
    if len(matches) != 1:
        _violation("capability_lacks_exact_lifecycle")
    return matches[0]


def _require_open_ring_events_for_bond(
    events: tuple[object, ...],
    bond,
) -> None:
    for endpoint_index, event in enumerate(events):
        if not isinstance(event, WriterRingEndpointEmitted):
            continue
        if event.bond != bond:
            continue
        if any(
            isinstance(candidate, WriterRingLabelAllocated)
            and candidate.label == event.label
            for candidate in events[:endpoint_index]
        ):
            return
        _violation("live_open_lacks_prior_label_allocation")

    _violation("live_open_lacks_endpoint_event")


def _violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer closure-candidate branch certificate violation: {kind}",
    )


__all__ = (
    "WriterClosureCandidateBranchCertificate",
    "WriterClosureCandidateBranchCertificateKind",
    "writer_closure_candidate_branch_certificates",
)
