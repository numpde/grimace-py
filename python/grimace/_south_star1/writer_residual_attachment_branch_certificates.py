"""Branch-local certificates for residual-attachment capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .writer_capabilities import _WriterExecutionCapabilityKind
from .writer_residual_attachment_lifecycle import (
    WriterResidualAttachmentLifecycleOutcomeKind,
)


class WriterResidualAttachmentBranchCertificateKind(Enum):
    COUPLED_CYCLIC_ATTACHMENT_DISCHARGED = (
        "coupled_cyclic_attachment_discharged"
    )


@dataclass(frozen=True, slots=True)
class WriterResidualAttachmentBranchCertificate:
    kind: WriterResidualAttachmentBranchCertificateKind
    capability: _WriterExecutionCapabilityKind
    attachment_id: int
    bond: object
    lifecycle_evidence: object
    graph_action_surface: object


def writer_residual_attachment_branch_certificates(
    *,
    execution_capabilities: frozenset[object],
    graph_action_surface: object | None,
    residual_attachment_lifecycle_evidence: tuple[object, ...],
) -> tuple[WriterResidualAttachmentBranchCertificate, ...]:
    if (
        _WriterExecutionCapabilityKind.COUPLED_CYCLIC_ATTACHMENT_RESTRICTION
        not in execution_capabilities
    ):
        return ()

    lifecycle = _single_lifecycle_evidence(
        residual_attachment_lifecycle_evidence
    )
    if (
        lifecycle.outcome_kind
        is not (
            WriterResidualAttachmentLifecycleOutcomeKind
            .CLOSURE_OPEN_DISCHARGED
        )
    ):
        _violation("coupled_capability_lacks_discharge")
    if (
        lifecycle.source_closure_deficit != 2
        or lifecycle.successor_closure_deficit != 1
    ):
        _violation("coupled_capability_lacks_deficit_delta")
    if len(lifecycle.source_attachment.block_ids) != 1:
        _violation("coupled_capability_lacks_single_block")
    if graph_action_surface is None:
        _violation("coupled_capability_lacks_graph_action_surface")

    return (
        WriterResidualAttachmentBranchCertificate(
            kind=(
                WriterResidualAttachmentBranchCertificateKind
                .COUPLED_CYCLIC_ATTACHMENT_DISCHARGED
            ),
            capability=(
                _WriterExecutionCapabilityKind
                .COUPLED_CYCLIC_ATTACHMENT_RESTRICTION
            ),
            attachment_id=lifecycle.attachment_id,
            bond=lifecycle.bond,
            lifecycle_evidence=lifecycle,
            graph_action_surface=graph_action_surface,
        ),
    )


def _single_lifecycle_evidence(evidence: tuple[object, ...]):
    if not evidence:
        _violation("coupled_capability_lacks_lifecycle")
    if len(evidence) != 1:
        _violation("coupled_capability_has_multiple_lifecycles")
    return evidence[0]


def _violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer residual attachment branch certificate violation: {kind}",
    )


__all__ = (
    "WriterResidualAttachmentBranchCertificate",
    "WriterResidualAttachmentBranchCertificateKind",
    "writer_residual_attachment_branch_certificates",
)
