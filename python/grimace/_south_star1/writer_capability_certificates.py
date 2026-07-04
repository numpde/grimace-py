"""Capability-coverage certificates for checked writer branch supports."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .writer_capabilities import _WriterExecutionCapabilityKind
from .writer_events import WriterBondEmitted
from .writer_events import WriterRingEndpointEmitted
from .writer_events import WriterRingEndpointPaired
from .writer_events import WriterRingLabelAllocated
from .writer_events import WriterRingLabelReleased
from .writer_transitions import _WriterGraphPolicyActionFamily
from .writer_transitions import WriterTransitionKind
from .policy import DirectionMark


class WriterCapabilityCertificateKind(Enum):
    TREE_CHILD_ENTRY = "tree_child_entry"
    CYCLIC_TREE_ENTRY = "cyclic_tree_entry"
    TREE_BOND_SLOT = "tree_bond_slot"
    VISIBLE_TREE_BOND_TEXT = "visible_tree_bond_text"
    CLOSURE_ENDPOINT_OPEN = "closure_endpoint_open"
    CLOSURE_ENDPOINT_PAIR = "closure_endpoint_pair"
    CONCURRENT_CLOSURE_ENDPOINT_OPEN = "concurrent_closure_endpoint_open"
    VISIBLE_CLOSURE_BOND_TEXT = "visible_closure_bond_text"

    CLOSURE_CANDIDATE_BRANCH = "closure_candidate_branch"
    RESIDUAL_ATTACHMENT_BRANCH = "residual_attachment_branch"
    STEREO_BRANCH = "stereo_branch"
    OPEN_RING_ENDPOINT_RESIDUAL_ATTACHMENT = (
        "open_ring_endpoint_residual_attachment"
    )
    RESIDUAL_PROPAGATION = "residual_propagation"
    DIRECTIONAL_SITE_COMPATIBILITY = "directional_site_compatibility"
    SHARED_DIRECTIONAL_CARRIER_RESTRICTION = (
        "shared_directional_carrier_restriction"
    )


@dataclass(frozen=True, slots=True)
class WriterCapabilityCertificate:
    capability: _WriterExecutionCapabilityKind
    kind: WriterCapabilityCertificateKind
    transition_kind: object
    graph_action_surface: object | None
    events: tuple[object, ...]
    transition_evidence: object
    supporting_certificates: tuple[object, ...] = ()
    work_evidence: tuple[object, ...] = ()


@dataclass(frozen=True, slots=True)
class WriterCapabilityCoverageCertificate:
    execution_capabilities: frozenset[object]
    capability_certificates: tuple[WriterCapabilityCertificate, ...]

    @property
    def covered_capabilities(self) -> frozenset[object]:
        return frozenset(
            certificate.capability
            for certificate in self.capability_certificates
        )


def writer_capability_coverage_certificate(
    *,
    execution_capabilities: frozenset[object],
    transition_kind,
    graph_action_surface,
    policy_family,
    events: tuple[object, ...],
    transition_evidence,
    finite_relation_work_evidence: tuple[object, ...],
    residual_work_evidence: tuple[object, ...],
    residual_attachment_policy_evidence: tuple[object, ...],
    closure_candidate_branch_certificates: tuple[object, ...],
    residual_attachment_branch_certificates: tuple[object, ...],
    stereo_branch_certificates: tuple[object, ...],
    successor_state=None,
) -> WriterCapabilityCoverageCertificate:
    certificates: list[WriterCapabilityCertificate] = []

    if _WriterExecutionCapabilityKind.TREE_CHILD_ENTRY in execution_capabilities:
        _tree_child_entry_certificate(
            transition_kind=transition_kind,
            graph_action_surface=graph_action_surface,
            policy_family=policy_family,
            transition_evidence=transition_evidence,
            events=events,
            certificates=certificates,
        )

    if _WriterExecutionCapabilityKind.CYCLIC_TREE_ENTRY in execution_capabilities:
        _cyclic_tree_entry_certificate(
            transition_kind=transition_kind,
            graph_action_surface=graph_action_surface,
            policy_family=policy_family,
            transition_evidence=transition_evidence,
            certificates=certificates,
        )

    if _WriterExecutionCapabilityKind.TREE_BOND_SLOT in execution_capabilities:
        _tree_bond_slot_certificate(
            transition_kind=transition_kind,
            events=events,
            transition_evidence=transition_evidence,
            certificates=certificates,
        )

    if (
        _WriterExecutionCapabilityKind.VISIBLE_TREE_BOND_TEXT
        in execution_capabilities
    ):
        _visible_tree_bond_text_certificate(
            transition_kind=transition_kind,
            events=events,
            transition_evidence=transition_evidence,
            certificates=certificates,
        )

    if _WriterExecutionCapabilityKind.CLOSURE_ENDPOINT_OPEN in execution_capabilities:
        _closure_endpoint_open_certificate(
            transition_kind=transition_kind,
            events=events,
            finite_relation_work_evidence=finite_relation_work_evidence,
            certificates=certificates,
        )

    if _WriterExecutionCapabilityKind.CLOSURE_ENDPOINT_PAIR in execution_capabilities:
        _closure_endpoint_pair_certificate(
            transition_kind=transition_kind,
            events=events,
            finite_relation_work_evidence=finite_relation_work_evidence,
            certificates=certificates,
        )

    if (
        _WriterExecutionCapabilityKind.VISIBLE_CLOSURE_BOND_TEXT
        in execution_capabilities
    ):
        _visible_closure_bond_text_certificate(
            transition_kind=transition_kind,
            events=events,
            certificates=certificates,
        )

    if (
        _WriterExecutionCapabilityKind.CONCURRENT_CLOSURE_ENDPOINT_OPEN
        in execution_capabilities
    ):
        _concurrent_closure_open_certificate(
            transition_kind=transition_kind,
            successor_state=successor_state,
            events=events,
            certificates=certificates,
        )

    if (
        _WriterExecutionCapabilityKind
        .OPEN_RING_ENDPOINT_RESIDUAL_ATTACHMENT_RESOLUTION
        in execution_capabilities
    ):
        _open_ring_endpoint_residual_attachment_certificate(
            capability=(
                _WriterExecutionCapabilityKind
                .OPEN_RING_ENDPOINT_RESIDUAL_ATTACHMENT_RESOLUTION
            ),
            graph_action_surface=graph_action_surface,
            residual_attachment_policy_evidence=residual_attachment_policy_evidence,
            transition_kind=transition_kind,
            events=events,
            certificates=certificates,
        )

    if (
        _WriterExecutionCapabilityKind
        .LIVE_BRANCH_RETURN_CLOSURE_CANDIDATE_OPEN
        in execution_capabilities
    ):
        _capability_from_subcertificates(
            capability=(
                _WriterExecutionCapabilityKind
                .LIVE_BRANCH_RETURN_CLOSURE_CANDIDATE_OPEN
            ),
            subcertificates=closure_candidate_branch_certificates,
            certificate_kind=WriterCapabilityCertificateKind.CLOSURE_CANDIDATE_BRANCH,
            transition_kind=transition_kind,
            graph_action_surface=graph_action_surface,
            events=events,
            transition_evidence=transition_evidence,
            certificates=certificates,
        )

    if (
        _WriterExecutionCapabilityKind.DEFERRED_BRANCH_RETURN_CLOSURE_CANDIDATE
        in execution_capabilities
    ):
        _capability_from_subcertificates(
            capability=(
                _WriterExecutionCapabilityKind
                .DEFERRED_BRANCH_RETURN_CLOSURE_CANDIDATE
            ),
            subcertificates=closure_candidate_branch_certificates,
            certificate_kind=WriterCapabilityCertificateKind.CLOSURE_CANDIDATE_BRANCH,
            transition_kind=transition_kind,
            graph_action_surface=graph_action_surface,
            events=events,
            transition_evidence=transition_evidence,
            certificates=certificates,
        )

    if (
        _WriterExecutionCapabilityKind.DEFERRED_CONTROL_LIVE_CLOSURE_CANDIDATE
        in execution_capabilities
    ):
        _capability_from_subcertificates(
            capability=(
                _WriterExecutionCapabilityKind
                .DEFERRED_CONTROL_LIVE_CLOSURE_CANDIDATE
            ),
            subcertificates=closure_candidate_branch_certificates,
            certificate_kind=WriterCapabilityCertificateKind.CLOSURE_CANDIDATE_BRANCH,
            transition_kind=transition_kind,
            graph_action_surface=graph_action_surface,
            events=events,
            transition_evidence=transition_evidence,
            certificates=certificates,
        )

    if (
        _WriterExecutionCapabilityKind.COUPLED_CYCLIC_ATTACHMENT_RESTRICTION
        in execution_capabilities
    ):
        _capability_from_subcertificates(
            capability=(
                _WriterExecutionCapabilityKind
                .COUPLED_CYCLIC_ATTACHMENT_RESTRICTION
            ),
            subcertificates=residual_attachment_branch_certificates,
            certificate_kind=WriterCapabilityCertificateKind.RESIDUAL_ATTACHMENT_BRANCH,
            transition_kind=transition_kind,
            graph_action_surface=graph_action_surface,
            events=events,
            transition_evidence=transition_evidence,
            certificates=certificates,
        )

    _stereo_capability_mappings(
        execution_capabilities=execution_capabilities,
        stereo_branch_certificates=stereo_branch_certificates,
        transition_kind=transition_kind,
        graph_action_surface=graph_action_surface,
        events=events,
        transition_evidence=transition_evidence,
        certificates=certificates,
    )

    if (
        _WriterExecutionCapabilityKind.RESIDUAL_PROPAGATION in execution_capabilities
    ):
        _residual_propagation_certificate(
            capability=_WriterExecutionCapabilityKind.RESIDUAL_PROPAGATION,
            transition_kind=transition_kind,
            graph_action_surface=graph_action_surface,
            events=events,
            residual_work_evidence=residual_work_evidence,
            stereo_branch_certificates=stereo_branch_certificates,
            transition_evidence=transition_evidence,
            certificates=certificates,
        )

    if (
        _WriterExecutionCapabilityKind.DIRECTIONAL_SITE_COMPATIBILITY
        in execution_capabilities
    ):
        _directional_site_compatibility_certificate(
            capability=_WriterExecutionCapabilityKind.DIRECTIONAL_SITE_COMPATIBILITY,
            transition_kind=transition_kind,
            graph_action_surface=graph_action_surface,
            events=events,
            stereo_branch_certificates=stereo_branch_certificates,
            residual_work_evidence=residual_work_evidence,
            transition_evidence=transition_evidence,
            certificates=certificates,
        )

    if (
        _WriterExecutionCapabilityKind.SHARED_DIRECTIONAL_CARRIER_RESTRICTION
        in execution_capabilities
    ):
        _shared_directional_carrier_certificate(
            capability=(_WriterExecutionCapabilityKind.SHARED_DIRECTIONAL_CARRIER_RESTRICTION),
            transition_kind=transition_kind,
            graph_action_surface=graph_action_surface,
            events=events,
            stereo_branch_certificates=stereo_branch_certificates,
            residual_work_evidence=residual_work_evidence,
            transition_evidence=transition_evidence,
            certificates=certificates,
        )

    coverage = WriterCapabilityCoverageCertificate(
        execution_capabilities=frozenset(execution_capabilities),
        capability_certificates=tuple(certificates),
    )

    if coverage.execution_capabilities != execution_capabilities:
        _capability_violation("execution_capability_mismatch")

    if coverage.covered_capabilities != execution_capabilities:
        _capability_violation("coverage_gap")

    return coverage


def _capability_from_subcertificates(
    *,
    capability: _WriterExecutionCapabilityKind,
    subcertificates,
    certificate_kind: WriterCapabilityCertificateKind,
    transition_kind,
    graph_action_surface,
    events: tuple[object, ...],
    transition_evidence,
    certificates: list[WriterCapabilityCertificate],
) -> None:
    matches = tuple(
        certificate
        for certificate in subcertificates
        if getattr(certificate, "capability", None) is capability
    )
    if len(matches) != 1:
        _capability_violation(
            f"{capability.value}_capability_lacks_sub_certificate"
        )
    certificates.append(
        WriterCapabilityCertificate(
            capability=capability,
            kind=certificate_kind,
            transition_kind=transition_kind,
            graph_action_surface=graph_action_surface,
            events=events,
            transition_evidence=transition_evidence,
            supporting_certificates=matches,
        )
    )


def _tree_child_entry_certificate(
    *,
    transition_kind,
    graph_action_surface,
    policy_family,
    transition_evidence,
    events,
    certificates: list[WriterCapabilityCertificate],
) -> None:
    if policy_family not in (
        _WriterGraphPolicyActionFamily.TREE_ENTRY,
        _WriterGraphPolicyActionFamily.ACYCLIC_TREE_ENTRY,
        _WriterGraphPolicyActionFamily.CYCLIC_TREE_ENTRY,
    ):
        _capability_violation("tree_child_capability_policy_family_mismatch")

    certificates.append(
        WriterCapabilityCertificate(
            capability=_WriterExecutionCapabilityKind.TREE_CHILD_ENTRY,
            kind=WriterCapabilityCertificateKind.TREE_CHILD_ENTRY,
            transition_kind=transition_kind,
            graph_action_surface=graph_action_surface,
            events=events,
            transition_evidence=transition_evidence,
        )
    )


def _cyclic_tree_entry_certificate(
    *,
    transition_kind,
    graph_action_surface,
    policy_family,
    transition_evidence,
    certificates: list[WriterCapabilityCertificate],
) -> None:
    if policy_family is not _WriterGraphPolicyActionFamily.CYCLIC_TREE_ENTRY:
        _capability_violation("cyclic_tree_entry_policy_family_mismatch")
    if getattr(graph_action_surface, "attachment_action_kind", None) is None:
        _capability_violation(
            "cyclic_tree_entry_missing_surface_attachment_kind"
        )

    certificates.append(
        WriterCapabilityCertificate(
            capability=_WriterExecutionCapabilityKind.CYCLIC_TREE_ENTRY,
            kind=WriterCapabilityCertificateKind.CYCLIC_TREE_ENTRY,
            transition_kind=transition_kind,
            graph_action_surface=graph_action_surface,
            events=tuple(),
            transition_evidence=transition_evidence,
        )
    )


def _tree_bond_slot_certificate(
    *,
    transition_kind,
    events,
    transition_evidence,
    certificates: list[WriterCapabilityCertificate],
) -> None:
    bond_event = _single_event_of_type(events, WriterBondEmitted)
    if bond_event is None:
        _capability_violation("tree_bond_slot_lacks_bond_event")
    if getattr(transition_evidence, "bond", None) != bond_event.bond:
        _capability_violation("tree_bond_slot_bond_mismatch")

    certificates.append(
        WriterCapabilityCertificate(
            capability=_WriterExecutionCapabilityKind.TREE_BOND_SLOT,
            kind=WriterCapabilityCertificateKind.TREE_BOND_SLOT,
            transition_kind=transition_kind,
            graph_action_surface=None,
            events=events,
            transition_evidence=transition_evidence,
        )
    )


def _visible_tree_bond_text_certificate(
    *,
    transition_kind,
    events,
    transition_evidence,
    certificates: list[WriterCapabilityCertificate],
) -> None:
    bond_event = _single_event_of_type(events, WriterBondEmitted)
    if bond_event is None:
        _capability_violation(
            "visible_tree_bond_text_missing_bond_event"
        )
    if not bond_event.text:
        _capability_violation("visible_tree_bond_text_not_visible")

    certificates.append(
        WriterCapabilityCertificate(
            capability=(_WriterExecutionCapabilityKind.VISIBLE_TREE_BOND_TEXT),
            kind=WriterCapabilityCertificateKind.VISIBLE_TREE_BOND_TEXT,
            transition_kind=transition_kind,
            graph_action_surface=None,
            events=events,
            transition_evidence=transition_evidence,
        )
    )


def _closure_endpoint_open_certificate(
    *,
    transition_kind,
    events,
    finite_relation_work_evidence,
    certificates: list[WriterCapabilityCertificate],
) -> None:
    if transition_kind is not WriterTransitionKind.OPEN_CLOSURE_ENDPOINT:
        _capability_violation("closure_open_transition_type_mismatch")
    if not finite_relation_work_evidence:
        _capability_violation("closure_open_lacks_finite_relation_work")
    if not _single_event_of_type(events, WriterRingEndpointEmitted):
        _capability_violation("closure_open_missing_endpoint_event")
    if not _single_event_of_type(events, WriterRingLabelAllocated):
        _capability_violation("closure_open_missing_prior_allocation")

    certificates.append(
        WriterCapabilityCertificate(
            capability=_WriterExecutionCapabilityKind.CLOSURE_ENDPOINT_OPEN,
            kind=WriterCapabilityCertificateKind.CLOSURE_ENDPOINT_OPEN,
            transition_kind=transition_kind,
            graph_action_surface=None,
            events=events,
            transition_evidence=None,
            work_evidence=finite_relation_work_evidence,
        )
    )


def _closure_endpoint_pair_certificate(
    *,
    transition_kind,
    events,
    finite_relation_work_evidence,
    certificates: list[WriterCapabilityCertificate],
) -> None:
    if transition_kind is not WriterTransitionKind.PAIR_CLOSURE_ENDPOINT:
        _capability_violation("closure_pair_transition_type_mismatch")
    if not finite_relation_work_evidence:
        _capability_violation("closure_pair_lacks_finite_relation_work")
    if not _single_event_of_type(events, WriterRingEndpointPaired):
        _capability_violation("closure_pair_missing_endpoint_event")
    if not _single_event_of_type(events, WriterRingLabelReleased):
        _capability_violation("closure_pair_missing_label_release")

    certificates.append(
        WriterCapabilityCertificate(
            capability=_WriterExecutionCapabilityKind.CLOSURE_ENDPOINT_PAIR,
            kind=WriterCapabilityCertificateKind.CLOSURE_ENDPOINT_PAIR,
            transition_kind=transition_kind,
            graph_action_surface=None,
            events=events,
            transition_evidence=None,
            work_evidence=finite_relation_work_evidence,
        )
    )


def _visible_closure_bond_text_certificate(
    *,
    transition_kind,
    events,
    certificates: list[WriterCapabilityCertificate],
) -> None:
    if transition_kind not in (
        WriterTransitionKind.OPEN_CLOSURE_ENDPOINT,
        WriterTransitionKind.PAIR_CLOSURE_ENDPOINT,
    ):
        _capability_violation(
            "visible_closure_bond_text_transition_type_mismatch"
        )
    event = _single_event_of_type(
        events,
        (WriterRingEndpointEmitted, WriterRingEndpointPaired),
    )
    if event is None:
        _capability_violation(
            "visible_closure_bond_text_missing_closure_endpoint_event"
        )
    if not event.bond_text and event.direction_mark is DirectionMark.ABSENT:
        _capability_violation("visible_closure_bond_text_invisible")

    certificates.append(
        WriterCapabilityCertificate(
            capability=(_WriterExecutionCapabilityKind.VISIBLE_CLOSURE_BOND_TEXT),
            kind=WriterCapabilityCertificateKind.VISIBLE_CLOSURE_BOND_TEXT,
            transition_kind=transition_kind,
            graph_action_surface=None,
            events=events,
            transition_evidence=None,
        )
    )


def _concurrent_closure_open_certificate(
    *,
    transition_kind,
    successor_state,
    events,
    certificates: list[WriterCapabilityCertificate],
) -> None:
    if transition_kind is not WriterTransitionKind.OPEN_CLOSURE_ENDPOINT:
        _capability_violation("concurrent_closure_open_transition_type_mismatch")
    if not _single_event_of_type(events, WriterRingEndpointEmitted):
        _capability_violation("concurrent_closure_open_missing_endpoint_event")
    open_endpoints = tuple(
        getattr(getattr(successor_state, "ring_state", object()), "open_endpoints", ())
        if successor_state is not None
        else ()
    )
    if not open_endpoints:
        _capability_violation("concurrent_closure_open_missing_successor_state")
    if len(open_endpoints) <= 1:
        _capability_violation("concurrent_closure_open_not_concurrent")

    certificates.append(
        WriterCapabilityCertificate(
            capability=(
                _WriterExecutionCapabilityKind.CONCURRENT_CLOSURE_ENDPOINT_OPEN
            ),
            kind=WriterCapabilityCertificateKind.CONCURRENT_CLOSURE_ENDPOINT_OPEN,
            transition_kind=transition_kind,
            graph_action_surface=None,
            events=events,
            transition_evidence=None,
            work_evidence=(open_endpoints,),
        )
    )


def _open_ring_endpoint_residual_attachment_certificate(
    *,
    capability,
    graph_action_surface,
    residual_attachment_policy_evidence,
    transition_kind,
    events,
    certificates: list[WriterCapabilityCertificate],
) -> None:
    if graph_action_surface is None:
        _capability_violation(
            "residual_attachment_capability_lacks_graph_action_surface"
        )
    if not residual_attachment_policy_evidence:
        _capability_violation(
            "residual_attachment_capability_lacks_policy_evidence"
        )

    certificates.append(
        WriterCapabilityCertificate(
            capability=capability,
            kind=WriterCapabilityCertificateKind.OPEN_RING_ENDPOINT_RESIDUAL_ATTACHMENT,
            transition_kind=transition_kind,
            graph_action_surface=graph_action_surface,
            events=events,
            transition_evidence=None,
            supporting_certificates=residual_attachment_policy_evidence,
            work_evidence=tuple(residual_attachment_policy_evidence),
        )
    )


def _stereo_capability_mappings(
    *,
    execution_capabilities,
    stereo_branch_certificates,
    transition_kind,
    graph_action_surface,
    events,
    transition_evidence,
    certificates: list[WriterCapabilityCertificate],
) -> None:
    mapping = (
        (
            _WriterExecutionCapabilityKind.TETRA_TOKEN_RESTRICTION,
            WriterCapabilityCertificateKind.STEREO_BRANCH,
        ),
        (
            _WriterExecutionCapabilityKind.TETRA_LOCAL_ORDER_RESTRICTION,
            WriterCapabilityCertificateKind.STEREO_BRANCH,
        ),
        (
            _WriterExecutionCapabilityKind.TETRA_RING_ENDPOINT_ORDER_OCCURRENCE,
            WriterCapabilityCertificateKind.STEREO_BRANCH,
        ),
        (
            _WriterExecutionCapabilityKind.DIRECTIONAL_CARRIER_RESTRICTION,
            WriterCapabilityCertificateKind.STEREO_BRANCH,
        ),
        (
            _WriterExecutionCapabilityKind.DIRECTIONAL_RING_PAIR_COMPATIBILITY,
            WriterCapabilityCertificateKind.STEREO_BRANCH,
        ),
        (
            _WriterExecutionCapabilityKind.RESIDUAL_FACTOR_DISCHARGE,
            WriterCapabilityCertificateKind.STEREO_BRANCH,
        ),
    )

    for capability, kind in mapping:
        if capability not in execution_capabilities:
            continue
        matches = tuple(
            certificate
            for certificate in stereo_branch_certificates
            if getattr(certificate, "capability", None) is capability
        )
        if len(matches) != 1:
            _capability_violation(f"{capability.value}_lacks_stereo_certificate")
        certificates.append(
            WriterCapabilityCertificate(
                capability=capability,
                kind=kind,
                transition_kind=transition_kind,
                graph_action_surface=graph_action_surface,
                events=events,
                transition_evidence=transition_evidence,
                supporting_certificates=matches,
            )
        )


def _residual_propagation_certificate(
    *,
    capability,
    transition_kind,
    graph_action_surface,
    events,
    residual_work_evidence,
    stereo_branch_certificates,
    transition_evidence,
    certificates: list[WriterCapabilityCertificate],
) -> None:
    matching_stereo = tuple(
        certificate
        for certificate in stereo_branch_certificates
        if getattr(certificate, "capability", None)
        in (
            _WriterExecutionCapabilityKind.TETRA_TOKEN_RESTRICTION,
            _WriterExecutionCapabilityKind.TETRA_LOCAL_ORDER_RESTRICTION,
            _WriterExecutionCapabilityKind.TETRA_RING_ENDPOINT_ORDER_OCCURRENCE,
            _WriterExecutionCapabilityKind.DIRECTIONAL_CARRIER_RESTRICTION,
            _WriterExecutionCapabilityKind.DIRECTIONAL_RING_PAIR_COMPATIBILITY,
            _WriterExecutionCapabilityKind.RESIDUAL_FACTOR_DISCHARGE,
        )
    )

    if not residual_work_evidence and not any(
        getattr(certificate, "residual_work_evidence", None)
        for certificate in matching_stereo
    ):
        _capability_violation(
            "residual_propagation_capability_lacks_work_evidence"
        )

    certificates.append(
        WriterCapabilityCertificate(
            capability=capability,
            kind=WriterCapabilityCertificateKind.RESIDUAL_PROPAGATION,
            transition_kind=transition_kind,
            graph_action_surface=graph_action_surface,
            events=events,
            transition_evidence=transition_evidence,
            supporting_certificates=matching_stereo,
            work_evidence=tuple(residual_work_evidence),
        )
    )


def _directional_site_compatibility_certificate(
    *,
    capability,
    transition_kind,
    graph_action_surface,
    events,
    stereo_branch_certificates,
    residual_work_evidence,
    transition_evidence,
    certificates: list[WriterCapabilityCertificate],
) -> None:
    directional = tuple(
        certificate
        for certificate in stereo_branch_certificates
        if getattr(certificate, "capability", None)
        in (
            _WriterExecutionCapabilityKind.DIRECTIONAL_CARRIER_RESTRICTION,
            _WriterExecutionCapabilityKind.DIRECTIONAL_RING_PAIR_COMPATIBILITY,
        )
    )
    if not directional:
        _capability_violation(
            "directional_site_compatibility_lacks_directional_certificate"
        )
    if not (
        residual_work_evidence
        or any(
            getattr(certificate, "residual_work_evidence", None)
            for certificate in directional
        )
    ):
        _capability_violation(
            "directional_site_compatibility_lacks_residual_work"
        )

    certificates.append(
        WriterCapabilityCertificate(
            capability=capability,
            kind=WriterCapabilityCertificateKind.DIRECTIONAL_SITE_COMPATIBILITY,
            transition_kind=transition_kind,
            graph_action_surface=graph_action_surface,
            events=events,
            transition_evidence=transition_evidence,
            supporting_certificates=directional,
            work_evidence=residual_work_evidence,
        )
    )


def _shared_directional_carrier_certificate(
    *,
    capability,
    transition_kind,
    graph_action_surface,
    events,
    stereo_branch_certificates,
    residual_work_evidence,
    transition_evidence,
    certificates: list[WriterCapabilityCertificate],
) -> None:
    directional = tuple(
        certificate
        for certificate in stereo_branch_certificates
        if getattr(certificate, "capability", None)
        in (
            _WriterExecutionCapabilityKind.DIRECTIONAL_CARRIER_RESTRICTION,
            _WriterExecutionCapabilityKind.DIRECTIONAL_RING_PAIR_COMPATIBILITY,
        )
    )
    if not directional:
        _capability_violation(
            "shared_directional_carrier_capability_lacks_directional_cert"
        )
    if not (
        residual_work_evidence
        or any(
            getattr(certificate, "residual_work_evidence", None)
            for certificate in directional
        )
    ):
        _capability_violation(
            "shared_directional_carrier_capability_lacks_residual_work"
        )

    certificates.append(
        WriterCapabilityCertificate(
            capability=capability,
            kind=(
                WriterCapabilityCertificateKind
                .SHARED_DIRECTIONAL_CARRIER_RESTRICTION
            ),
            transition_kind=transition_kind,
            graph_action_surface=graph_action_surface,
            events=events,
            transition_evidence=transition_evidence,
            supporting_certificates=directional,
            work_evidence=tuple(residual_work_evidence),
        )
    )


def _require_transition_kind(*, transition_kind, expected, failure):
    if transition_kind not in expected:
        _capability_violation(failure)


def _single_event_of_type(events: tuple[object, ...], event_type):
    for event in events:
        if isinstance(event, event_type):
            return event
    return None


def _capability_violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer capability coverage violation: {kind}",
    )


__all__ = (
    "WriterCapabilityCertificate",
    "WriterCapabilityCertificateKind",
    "WriterCapabilityCoverageCertificate",
    "writer_capability_coverage_certificate",
)
