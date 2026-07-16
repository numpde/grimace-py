"""Coupled lifecycle evidence for directional non-single ring closures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .ids import AtomId
from .ids import BondId

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .writer_capabilities import _WriterExecutionCapabilityKind
from .writer_closure_bond_text_lifecycle import (
    WriterClosureBondTextLifecycleEvidence,
)
from .writer_closure_bond_text_lifecycle import (
    validate_writer_closure_bond_text_lifecycle_transition,
)
from .writer_events import WriterRingEndpointEmitted
from .writer_events import WriterRingEndpointPaired


@dataclass(frozen=True, slots=True)
class WriterDirectionalRingClosureBondTextLifecycleEvidence:
    closure_bond_text_lifecycle: WriterClosureBondTextLifecycleEvidence
    directional_stereo_lifecycle: object
    event: object
    source_ring_state: object
    successor_ring_state: object
    source_stereo_residual_snapshot: object
    successor_stereo_residual_snapshot: object
    closed_closure_record: object | None


@dataclass(frozen=True, slots=True)
class DirectionalRingClosureCouplingTerm:
    event_kind: Literal["ring_endpoint_emitted", "ring_endpoint_paired"]
    bond: BondId
    bond_order: Literal["double"]
    label_value: int
    label_text: str
    opening_atom: AtomId
    closing_atom: AtomId
    opening_marker: str
    closing_marker: str
    marker_side: Literal["opening", "closing"]
    source_state_digest: str
    successor_state_digest: str
    source_ring_state_digest: str
    successor_ring_state_digest: str
    source_residual_snapshot_digest: str
    successor_residual_snapshot_digest: str
    closure_manifest_digest: str
    stereo_lifecycle_digest: str
    residual_work_digests: tuple[str, ...]
    closed_closure_record_digest: str | None


def directional_ring_closure_bond_text_lifecycle_evidence(
    *,
    closure_bond_text_lifecycle_evidence: tuple[object, ...],
    stereo_lifecycle_evidence: tuple[object, ...],
) -> tuple[WriterDirectionalRingClosureBondTextLifecycleEvidence, ...]:
    coupled = []
    for closure in closure_bond_text_lifecycle_evidence:
        match = _matching_directional_lifecycle(
            closure,
            stereo_lifecycle_evidence,
        )
        if match is None:
            continue
        evidence = WriterDirectionalRingClosureBondTextLifecycleEvidence(
            closure_bond_text_lifecycle=closure,
            directional_stereo_lifecycle=match,
            event=closure.event,
            source_ring_state=closure.source_ring_state,
            successor_ring_state=closure.successor_ring_state,
            source_stereo_residual_snapshot=match.source_residual_snapshot,
            successor_stereo_residual_snapshot=match.successor_residual_snapshot,
            closed_closure_record=closure.closed_closure_record,
        )
        validate_writer_directional_ring_closure_bond_text_lifecycle_transition(
            evidence
        )
        coupled.append(evidence)
    return tuple(coupled)


def validate_writer_directional_ring_closure_bond_text_lifecycle_transition(
    evidence: WriterDirectionalRingClosureBondTextLifecycleEvidence,
) -> None:
    closure = evidence.closure_bond_text_lifecycle
    stereo = evidence.directional_stereo_lifecycle

    validate_writer_closure_bond_text_lifecycle_transition(closure)
    if evidence.event is not closure.event:
        _coupled_violation("coupled_event_mismatch")
    if evidence.event is not getattr(stereo, "event", None):
        _coupled_violation("stereo_event_mismatch")
    if evidence.source_ring_state != closure.source_ring_state:
        _coupled_violation("source_ring_state_mismatch")
    if evidence.successor_ring_state != closure.successor_ring_state:
        _coupled_violation("successor_ring_state_mismatch")
    if (
        evidence.source_stereo_residual_snapshot
        != getattr(stereo, "source_residual_snapshot", None)
    ):
        _coupled_violation("source_stereo_residual_snapshot_mismatch")
    if (
        evidence.successor_stereo_residual_snapshot
        != getattr(stereo, "successor_residual_snapshot", None)
    ):
        _coupled_violation("successor_stereo_residual_snapshot_mismatch")
    if evidence.closed_closure_record != closure.closed_closure_record:
        _coupled_violation("closed_closure_record_mismatch")
    if _directional_capability(closure) not in getattr(stereo, "capabilities", ()):
        _coupled_violation("directional_capability_missing")
    _validate_event_touches_closure_carrier(closure, stereo)


def _matching_directional_lifecycle(
    closure: WriterClosureBondTextLifecycleEvidence,
    stereo_lifecycle_evidence: tuple[object, ...],
):
    for evidence in stereo_lifecycle_evidence:
        if getattr(evidence, "event", None) is not closure.event:
            continue
        if _directional_capability(closure) not in getattr(
            evidence,
            "capabilities",
            (),
        ):
            continue
        return evidence
    return None


def _directional_capability(closure: WriterClosureBondTextLifecycleEvidence):
    if isinstance(closure.event, WriterRingEndpointEmitted):
        return _WriterExecutionCapabilityKind.DIRECTIONAL_RING_PAIR_COMPATIBILITY
    if isinstance(closure.event, WriterRingEndpointPaired):
        return _WriterExecutionCapabilityKind.DIRECTIONAL_CARRIER_RESTRICTION
    _coupled_violation("unsupported_ring_event")


def _validate_event_touches_closure_carrier(
    closure: WriterClosureBondTextLifecycleEvidence,
    stereo,
) -> None:
    event = closure.event
    if getattr(event, "bond", None) != closure.bond:
        _coupled_violation("event_bond_mismatch")
    if isinstance(event, WriterRingEndpointEmitted):
        if event.endpoint_atom != closure.opening_atom:
            _coupled_violation("event_opening_atom_mismatch")
        if event.partner_atom != closure.closing_atom:
            _coupled_violation("event_closing_atom_mismatch")
        return
    if isinstance(event, WriterRingEndpointPaired):
        if event.partner_atom != closure.opening_atom:
            _coupled_violation("event_opening_atom_mismatch")
        if event.endpoint_atom != closure.closing_atom:
            _coupled_violation("event_closing_atom_mismatch")
        if not getattr(stereo, "residual_work_evidence", ()):
            _coupled_violation("paired_event_lacks_directional_work")
        return
    _coupled_violation("unsupported_ring_event")


def _coupled_violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        "writer directional ring closure bond-text lifecycle violation: "
        f"{kind}",
    )


__all__ = (
    "DirectionalRingClosureCouplingTerm",
    "WriterDirectionalRingClosureBondTextLifecycleEvidence",
    "directional_ring_closure_bond_text_lifecycle_evidence",
    "validate_writer_directional_ring_closure_bond_text_lifecycle_transition",
)
