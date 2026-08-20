"""Branch-local certificates for stereo capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .writer_capabilities import _WriterExecutionCapabilityKind
from .writer_events import WriterAtomEmitted
from .writer_events import WriterBondEmitted
from .writer_events import WriterLocalOrderClosed
from .writer_events import WriterRingEndpointEmitted
from .writer_events import WriterRingEndpointPaired
from .writer_stereo import WriterStereoLifecycleOutcomeKind


class WriterStereoBranchCertificateKind(Enum):
    TETRA_TOKEN_RESTRICTED = "tetra_token_restricted"
    TETRA_LOCAL_ORDER_RESTRICTED = "tetra_local_order_restricted"
    TETRA_RING_ENDPOINT_RECORDED = "tetra_ring_endpoint_recorded"
    DIRECTIONAL_CARRIER_RESTRICTED = "directional_carrier_restricted"
    DIRECTIONAL_RING_PAIR_RESTRICTED = "directional_ring_pair_restricted"
    RESIDUAL_FACTOR_DISCHARGED = "residual_factor_discharged"


@dataclass(frozen=True, slots=True)
class WriterStereoBranchCertificate:
    kind: WriterStereoBranchCertificateKind
    capability: _WriterExecutionCapabilityKind
    lifecycle_evidence: object
    event: object
    residual_work_evidence: tuple[object, ...] = ()


def writer_stereo_branch_certificates(
    *,
    execution_capabilities: frozenset[object],
    stereo_lifecycle_evidence: tuple[object, ...],
    events: tuple[object, ...],
) -> tuple[WriterStereoBranchCertificate, ...]:
    del events
    certificates: list[WriterStereoBranchCertificate] = []

    if (
        _WriterExecutionCapabilityKind.TETRA_TOKEN_RESTRICTION
        in execution_capabilities
    ):
        certificates.append(
            _certificate_for_capability(
                kind=(
                    WriterStereoBranchCertificateKind
                    .TETRA_TOKEN_RESTRICTED
                ),
                capability=(
                    _WriterExecutionCapabilityKind.TETRA_TOKEN_RESTRICTION
                ),
                lifecycle_evidence=stereo_lifecycle_evidence,
                event_type=WriterAtomEmitted,
                operation="tetrahedral atom-token restriction",
                require_residual_delta=True,
            )
        )

    if (
        _WriterExecutionCapabilityKind.TETRA_LOCAL_ORDER_RESTRICTION
        in execution_capabilities
    ):
        certificates.append(
            _certificate_for_capability(
                kind=(
                    WriterStereoBranchCertificateKind
                    .TETRA_LOCAL_ORDER_RESTRICTED
                ),
                capability=(
                    _WriterExecutionCapabilityKind
                    .TETRA_LOCAL_ORDER_RESTRICTION
                ),
                lifecycle_evidence=stereo_lifecycle_evidence,
                event_type=WriterLocalOrderClosed,
                operation="tetrahedral local-order factor closure",
                require_residual_delta=True,
            )
        )

    if (
        _WriterExecutionCapabilityKind
        .TETRA_RING_ENDPOINT_ORDER_OCCURRENCE
    ) in execution_capabilities:
        certificates.append(
            _certificate_for_capability(
                kind=(
                    WriterStereoBranchCertificateKind
                    .TETRA_RING_ENDPOINT_RECORDED
                ),
                capability=(
                    _WriterExecutionCapabilityKind
                    .TETRA_RING_ENDPOINT_ORDER_OCCURRENCE
                ),
                lifecycle_evidence=stereo_lifecycle_evidence,
                event_type=(WriterRingEndpointEmitted, WriterRingEndpointPaired),
                operation=None,
                require_local_order_delta=True,
            )
        )

    if (
        _WriterExecutionCapabilityKind.DIRECTIONAL_CARRIER_RESTRICTION
        in execution_capabilities
    ):
        certificates.append(
            _certificate_for_capability(
                kind=(
                    WriterStereoBranchCertificateKind
                    .DIRECTIONAL_CARRIER_RESTRICTED
                ),
                capability=(
                    _WriterExecutionCapabilityKind
                    .DIRECTIONAL_CARRIER_RESTRICTION
                ),
                lifecycle_evidence=stereo_lifecycle_evidence,
                event_type=(WriterBondEmitted, WriterRingEndpointPaired),
                operation=(
                    "directional carrier-mark restriction",
                    "directional ring pair restriction",
                ),
                require_residual_delta=True,
            )
        )

    if (
        _WriterExecutionCapabilityKind.DIRECTIONAL_RING_PAIR_COMPATIBILITY
        in execution_capabilities
    ):
        certificates.append(
            _certificate_for_capability(
                kind=(
                    WriterStereoBranchCertificateKind
                    .DIRECTIONAL_RING_PAIR_RESTRICTED
                ),
                capability=(
                    _WriterExecutionCapabilityKind
                    .DIRECTIONAL_RING_PAIR_COMPATIBILITY
                ),
                lifecycle_evidence=stereo_lifecycle_evidence,
                event_type=(WriterRingEndpointEmitted, WriterRingEndpointPaired),
                operation=(
                    "directional ring endpoint projection",
                    "directional ring pair restriction",
                ),
            )
        )

    if (
        _WriterExecutionCapabilityKind.RESIDUAL_FACTOR_DISCHARGE
        in execution_capabilities
    ):
        certificates.append(
            _residual_factor_discharge_certificate(stereo_lifecycle_evidence)
        )

    return tuple(certificates)


def _certificate_for_capability(
    *,
    kind: WriterStereoBranchCertificateKind,
    capability: _WriterExecutionCapabilityKind,
    lifecycle_evidence: tuple[object, ...],
    event_type,
    operation: str | tuple[str, ...] | None,
    require_residual_delta: bool = False,
    require_local_order_delta: bool = False,
) -> WriterStereoBranchCertificate:
    matches = tuple(
        evidence
        for evidence in lifecycle_evidence
        if (
            capability in evidence.capabilities
            and isinstance(evidence.event, event_type)
            and _has_operation(evidence, operation)
            and (
                not require_residual_delta
                or _has_residual_snapshot_delta(evidence)
            )
            and (
                not require_local_order_delta
                or evidence.source_local_orders
                != evidence.successor_local_orders
            )
        )
    )
    if len(matches) != 1:
        _violation(f"{capability.value}_lacks_exact_lifecycle")
    evidence = matches[0]
    return WriterStereoBranchCertificate(
        kind=kind,
        capability=capability,
        lifecycle_evidence=evidence,
        event=evidence.event,
        residual_work_evidence=tuple(evidence.residual_work_evidence),
    )


def _residual_factor_discharge_certificate(
    lifecycle_evidence: tuple[object, ...],
) -> WriterStereoBranchCertificate:
    matches = tuple(
        evidence
        for evidence in lifecycle_evidence
        if (
            _WriterExecutionCapabilityKind.RESIDUAL_FACTOR_DISCHARGE
            in evidence.capabilities
            and _has_residual_snapshot_delta(evidence)
            and evidence.outcome_kind
            in (
                WriterStereoLifecycleOutcomeKind.RESIDUAL_DISCHARGED,
                WriterStereoLifecycleOutcomeKind.RESIDUAL_RESTRICTED,
                WriterStereoLifecycleOutcomeKind.RECORD_AND_RESTRICT,
            )
        )
    )
    if not matches:
        _violation("residual_factor_discharge_lacks_lifecycle")
    evidence = matches[0]
    return WriterStereoBranchCertificate(
        kind=(
            WriterStereoBranchCertificateKind
            .RESIDUAL_FACTOR_DISCHARGED
        ),
        capability=_WriterExecutionCapabilityKind.RESIDUAL_FACTOR_DISCHARGE,
        lifecycle_evidence=evidence,
        event=evidence.event,
        residual_work_evidence=tuple(evidence.residual_work_evidence),
    )


def _has_operation(
    lifecycle_evidence,
    operation: str | tuple[str, ...] | None,
) -> bool:
    if operation is None:
        return True
    operations = (operation,) if isinstance(operation, str) else operation
    return any(
        item.operation in operations
        for item in lifecycle_evidence.residual_work_evidence
    )


def _has_residual_snapshot_delta(lifecycle_evidence) -> bool:
    return (
        lifecycle_evidence.source_residual_snapshot
        != lifecycle_evidence.successor_residual_snapshot
    )


def _violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer stereo branch certificate violation: {kind}",
    )


__all__ = (
    "WriterStereoBranchCertificate",
    "WriterStereoBranchCertificateKind",
    "writer_stereo_branch_certificates",
)
