"""Checked terminal/EOS certificates for writer frontier states."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .writer_events import WriterLocalOrderClosed
from .writer_graph_obligations import build_writer_graph_obligation_context
from .writer_graph_obligations import writer_graph_completion_status
from .writer_stereo import EMPTY_RESIDUAL_SNAPSHOT


class WriterTerminalCertificateKind(Enum):
    GRAPH_COMPLETE = "graph_complete"
    STEREO_TERMINALIZED = "stereo_terminalized"
    FINALIZED_STATE = "finalized_state"


@dataclass(frozen=True, slots=True)
class WriterTerminalCertificate:
    kind: WriterTerminalCertificateKind
    source_state: object
    finalized_state: object
    graph_completion_status: object | None = None
    graph_obligation_work_evidence: tuple[object, ...] = ()
    terminal_stereo_lifecycle_evidence: tuple[object, ...] = ()
    terminal_execution_capabilities: frozenset[object] = frozenset()
    terminal_residual_work_evidence: tuple[object, ...] = ()


def writer_terminal_certificates(
    *,
    prepared,
    source_state,
    finalized_state,
    graph_obligation_work_evidence: tuple[object, ...],
    terminal_stereo_lifecycle_evidence: tuple[object, ...],
    terminal_execution_capabilities: frozenset[object],
    terminal_residual_work_evidence: tuple[object, ...],
) -> tuple[WriterTerminalCertificate, ...]:
    graph_certificate = _graph_completion_certificate(
        prepared=prepared,
        source_state=source_state,
        finalized_state=finalized_state,
        graph_obligation_work_evidence=graph_obligation_work_evidence,
    )
    stereo_certificate = _stereo_terminal_certificate(
        source_state=source_state,
        finalized_state=finalized_state,
        terminal_stereo_lifecycle_evidence=(
            terminal_stereo_lifecycle_evidence
        ),
        terminal_execution_capabilities=terminal_execution_capabilities,
        terminal_residual_work_evidence=terminal_residual_work_evidence,
    )
    finalized_certificate = _finalized_state_certificate(
        prepared=prepared,
        source_state=source_state,
        finalized_state=finalized_state,
        graph_obligation_work_evidence=graph_obligation_work_evidence,
        terminal_stereo_lifecycle_evidence=(
            terminal_stereo_lifecycle_evidence
        ),
        terminal_execution_capabilities=terminal_execution_capabilities,
        terminal_residual_work_evidence=terminal_residual_work_evidence,
    )
    return (graph_certificate, stereo_certificate, finalized_certificate)


def _graph_completion_certificate(
    *,
    prepared,
    source_state,
    finalized_state,
    graph_obligation_work_evidence: tuple[object, ...],
) -> WriterTerminalCertificate:
    context = build_writer_graph_obligation_context(prepared, source_state)
    completion = writer_graph_completion_status(
        prepared,
        source_state,
        context,
    )
    if not completion.complete:
        _violation("terminal graph is not complete")
    if completion.unresolved_kinds or completion.unresolved_bonds:
        _violation("terminal graph has unresolved obligations")
    if context.residual_summary.attachment_actions:
        _violation("terminal graph has residual attachment actions")
    if any(
        evidence.unsupported_closure_candidate_count
        for evidence in graph_obligation_work_evidence
    ):
        _violation("terminal graph has unsupported closure candidates")
    if any(
        evidence.open_closure_count
        for evidence in graph_obligation_work_evidence
    ):
        _violation("terminal graph has open closures")

    return WriterTerminalCertificate(
        kind=WriterTerminalCertificateKind.GRAPH_COMPLETE,
        source_state=source_state,
        finalized_state=finalized_state,
        graph_completion_status=completion,
        graph_obligation_work_evidence=graph_obligation_work_evidence,
    )


def _stereo_terminal_certificate(
    *,
    source_state,
    finalized_state,
    terminal_stereo_lifecycle_evidence: tuple[object, ...],
    terminal_execution_capabilities: frozenset[object],
    terminal_residual_work_evidence: tuple[object, ...],
) -> WriterTerminalCertificate:
    if finalized_state.stereo_state.residual_snapshot != EMPTY_RESIDUAL_SNAPSHOT:
        _violation("terminal stereo residual snapshot is not empty")
    lifecycle_capabilities = frozenset(
        capability
        for evidence in terminal_stereo_lifecycle_evidence
        for capability in evidence.capabilities
    )
    if not terminal_execution_capabilities.issubset(lifecycle_capabilities):
        _violation("terminal stereo capabilities lack lifecycle evidence")
    lifecycle_work_evidence = tuple(
        item
        for evidence in terminal_stereo_lifecycle_evidence
        for item in evidence.residual_work_evidence
    )
    if terminal_residual_work_evidence != lifecycle_work_evidence:
        _violation("terminal stereo residual work does not match lifecycle")
    if any(
        not isinstance(evidence.event, WriterLocalOrderClosed)
        for evidence in terminal_stereo_lifecycle_evidence
    ):
        _violation("terminal stereo lifecycle is not local-order closure")

    return WriterTerminalCertificate(
        kind=WriterTerminalCertificateKind.STEREO_TERMINALIZED,
        source_state=source_state,
        finalized_state=finalized_state,
        terminal_stereo_lifecycle_evidence=(
            terminal_stereo_lifecycle_evidence
        ),
        terminal_execution_capabilities=terminal_execution_capabilities,
        terminal_residual_work_evidence=terminal_residual_work_evidence,
    )


def _finalized_state_certificate(
    *,
    prepared,
    source_state,
    finalized_state,
    graph_obligation_work_evidence: tuple[object, ...],
    terminal_stereo_lifecycle_evidence: tuple[object, ...],
    terminal_execution_capabilities: frozenset[object],
    terminal_residual_work_evidence: tuple[object, ...],
) -> WriterTerminalCertificate:
    if finalized_state.active != source_state.active:
        _violation("terminal finalized active frame changed")
    if finalized_state.branch_stack:
        _violation("terminal finalized branch stack is not empty")
    if finalized_state.obligations.pending_entry is not None:
        _violation("terminal finalized state has pending entry")
    if finalized_state.ring_state.open_endpoints:
        _violation("terminal finalized state has open ring endpoints")
    if finalized_state.ring_state.label_state.allocated:
        _violation("terminal finalized state has allocated ring labels")

    context = build_writer_graph_obligation_context(prepared, finalized_state)
    completion = writer_graph_completion_status(
        prepared,
        finalized_state,
        context,
    )
    if not completion.complete:
        _violation("terminal finalized graph is not complete")
    if finalized_state.stereo_state.residual_snapshot != EMPTY_RESIDUAL_SNAPSHOT:
        _violation("terminal finalized stereo residual snapshot is not empty")

    return WriterTerminalCertificate(
        kind=WriterTerminalCertificateKind.FINALIZED_STATE,
        source_state=source_state,
        finalized_state=finalized_state,
        graph_obligation_work_evidence=graph_obligation_work_evidence,
        terminal_stereo_lifecycle_evidence=(
            terminal_stereo_lifecycle_evidence
        ),
        terminal_execution_capabilities=terminal_execution_capabilities,
        terminal_residual_work_evidence=terminal_residual_work_evidence,
    )


def _violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer terminal certificate violation: {kind}",
    )


__all__ = (
    "WriterTerminalCertificate",
    "WriterTerminalCertificateKind",
    "writer_terminal_certificates",
)
