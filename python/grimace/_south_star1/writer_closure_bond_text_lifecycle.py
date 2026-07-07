"""Lifecycle evidence for split non-single ring-closure bond text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .ids import AtomId
from .ids import BondId
from .policy import DirectionMark


@dataclass(frozen=True, slots=True)
class WriterClosureBondTextLifecycleEvidence:
    bond: BondId
    bond_order: Literal["double", "triple"]
    label: object
    opening_atom: AtomId
    closing_atom: AtomId
    opening_marker: str
    closing_marker: str
    marker_side: Literal["opening", "closing"]
    source_ring_state: object
    successor_ring_state: object
    event_kind: Literal["endpoint_emitted", "endpoint_paired"]
    event: object
    closed_closure_record: object | None = None


def validate_writer_closure_bond_text_lifecycle_transition(
    evidence: WriterClosureBondTextLifecycleEvidence,
) -> None:
    marker = _marker_for_order(evidence.bond_order)
    _validate_marker_pair(
        opening_marker=evidence.opening_marker,
        closing_marker=evidence.closing_marker,
        marker=marker,
        marker_side=evidence.marker_side,
    )
    if evidence.event_kind == "endpoint_emitted":
        _validate_endpoint_emitted(evidence, marker=marker)
        return
    if evidence.event_kind == "endpoint_paired":
        _validate_endpoint_paired(evidence, marker=marker)
        return
    _lifecycle_violation("unknown_event_kind")


def closure_bond_text_lifecycle_evidence_from_transition(
    *,
    source_ring_state,
    successor_ring_state,
    event,
) -> WriterClosureBondTextLifecycleEvidence | None:
    event_type = event.__class__.__name__
    if event_type == "WriterRingEndpointEmitted":
        marker = _non_single_marker(getattr(event, "bond_text", ""))
        if marker is None:
            return None
        evidence = WriterClosureBondTextLifecycleEvidence(
            bond=event.bond,
            bond_order=_order_for_marker(marker),
            label=event.label,
            opening_atom=event.endpoint_atom,
            closing_atom=event.partner_atom,
            opening_marker=event.bond_text,
            closing_marker="",
            marker_side="opening",
            source_ring_state=source_ring_state,
            successor_ring_state=successor_ring_state,
            event_kind="endpoint_emitted",
            event=event,
            closed_closure_record=None,
        )
        validate_writer_closure_bond_text_lifecycle_transition(evidence)
        return evidence
    if event_type == "WriterRingEndpointPaired":
        opening_marker = getattr(event, "first_endpoint_bond_text", "")
        closing_marker = getattr(event, "bond_text", "")
        marker = _non_single_marker(opening_marker) or _non_single_marker(
            closing_marker
        )
        if marker is None:
            return None
        evidence = WriterClosureBondTextLifecycleEvidence(
            bond=event.bond,
            bond_order=_order_for_marker(marker),
            label=event.label,
            opening_atom=event.partner_atom,
            closing_atom=event.endpoint_atom,
            opening_marker=opening_marker,
            closing_marker=closing_marker,
            marker_side="opening"
            if opening_marker == marker
            else "closing",
            source_ring_state=source_ring_state,
            successor_ring_state=successor_ring_state,
            event_kind="endpoint_paired",
            event=event,
            closed_closure_record=_closed_closure_for_event(
                successor_ring_state,
                event,
            ),
        )
        validate_writer_closure_bond_text_lifecycle_transition(evidence)
        return evidence
    return None


def _validate_endpoint_emitted(
    evidence: WriterClosureBondTextLifecycleEvidence,
    *,
    marker: str,
) -> None:
    event = evidence.event
    if getattr(event, "bond", None) != evidence.bond:
        _lifecycle_violation("event_bond_mismatch")
    if getattr(event, "endpoint_atom", None) != evidence.opening_atom:
        _lifecycle_violation("event_opening_atom_mismatch")
    if getattr(event, "partner_atom", None) != evidence.closing_atom:
        _lifecycle_violation("event_closing_atom_mismatch")
    if getattr(event, "label", None) != evidence.label:
        _lifecycle_violation("event_label_mismatch")
    if getattr(event, "bond_text", None) != marker:
        _lifecycle_violation("event_marker_mismatch")
    if getattr(event, "direction_mark", None) is not DirectionMark.ABSENT:
        _lifecycle_violation("directional_marker_not_supported")
    if evidence.closed_closure_record is not None:
        _lifecycle_violation("open_event_has_closed_closure_record")
    endpoint = _matching_open_endpoint(
        evidence.successor_ring_state,
        bond=evidence.bond,
        label=evidence.label,
        first_atom=evidence.opening_atom,
        second_atom=evidence.closing_atom,
    )
    if endpoint is None:
        _lifecycle_violation("successor_open_endpoint_missing")
    if _matching_open_endpoint(
        evidence.source_ring_state,
        bond=evidence.bond,
        label=evidence.label,
        first_atom=evidence.opening_atom,
        second_atom=evidence.closing_atom,
    ) is not None:
        _lifecycle_violation("source_already_has_open_endpoint")
    if endpoint.first_endpoint_bond_text != marker:
        _lifecycle_violation("successor_open_marker_mismatch")


def _validate_endpoint_paired(
    evidence: WriterClosureBondTextLifecycleEvidence,
    *,
    marker: str,
) -> None:
    event = evidence.event
    if getattr(event, "bond", None) != evidence.bond:
        _lifecycle_violation("event_bond_mismatch")
    if getattr(event, "partner_atom", None) != evidence.opening_atom:
        _lifecycle_violation("event_opening_atom_mismatch")
    if getattr(event, "endpoint_atom", None) != evidence.closing_atom:
        _lifecycle_violation("event_closing_atom_mismatch")
    if getattr(event, "label", None) != evidence.label:
        _lifecycle_violation("event_label_mismatch")
    if getattr(event, "first_endpoint_bond_text", None) != evidence.opening_marker:
        _lifecycle_violation("event_opening_marker_mismatch")
    if getattr(event, "bond_text", None) != evidence.closing_marker:
        _lifecycle_violation("event_closing_marker_mismatch")
    if getattr(event, "direction_mark", None) is not DirectionMark.ABSENT:
        _lifecycle_violation("directional_marker_not_supported")
    if getattr(event, "first_endpoint_direction_mark", None) is not DirectionMark.ABSENT:
        _lifecycle_violation("directional_marker_not_supported")
    if _matching_open_endpoint(
        evidence.source_ring_state,
        bond=evidence.bond,
        label=evidence.label,
        first_atom=evidence.opening_atom,
        second_atom=evidence.closing_atom,
    ) is None:
        _lifecycle_violation("source_open_endpoint_missing")
    if _matching_open_endpoint(
        evidence.successor_ring_state,
        bond=evidence.bond,
        label=evidence.label,
        first_atom=evidence.opening_atom,
        second_atom=evidence.closing_atom,
    ) is not None:
        _lifecycle_violation("successor_open_endpoint_still_live")
    closure = evidence.closed_closure_record
    if closure is None:
        _lifecycle_violation("paired_event_lacks_closed_closure_record")
    if _closed_closure_for_event(evidence.successor_ring_state, event) != closure:
        _lifecycle_violation("successor_closed_closure_mismatch")
    if closure.first_endpoint_bond_text != evidence.opening_marker:
        _lifecycle_violation("closed_opening_marker_mismatch")
    if closure.second_endpoint_bond_text != evidence.closing_marker:
        _lifecycle_violation("closed_closing_marker_mismatch")
    if marker not in (closure.first_endpoint_bond_text, closure.second_endpoint_bond_text):
        _lifecycle_violation("closed_closure_marker_missing")


def _validate_marker_pair(
    *,
    opening_marker: str,
    closing_marker: str,
    marker: str,
    marker_side: str,
) -> None:
    if opening_marker not in ("", marker):
        _lifecycle_violation("wrong_opening_marker")
    if closing_marker not in ("", marker):
        _lifecycle_violation("wrong_closing_marker")
    marker_count = int(opening_marker == marker) + int(closing_marker == marker)
    if marker_count == 0:
        _lifecycle_violation("closure_bond_text_marker_missing")
    if marker_count > 1:
        _lifecycle_violation("closure_bond_text_marker_duplicate")
    expected_side = "opening" if opening_marker == marker else "closing"
    if marker_side != expected_side:
        _lifecycle_violation("marker_side_mismatch")


def _matching_open_endpoint(state, *, bond, label, first_atom, second_atom):
    for endpoint in getattr(state, "open_endpoints", ()):
        if (
            endpoint.bond == bond
            and endpoint.label == label
            and endpoint.first_atom == first_atom
            and endpoint.second_atom == second_atom
        ):
            return endpoint
    return None


def _closed_closure_for_event(state, event):
    matches = tuple(
        closure
        for closure in getattr(state, "closed_closures", ())
        if (
            closure.bond == event.bond
            and closure.label == event.label
            and closure.first_atom == event.partner_atom
            and closure.second_atom == event.endpoint_atom
        )
    )
    if not matches:
        return None
    if len(matches) > 1:
        _lifecycle_violation("duplicate_closed_closure_record")
    return matches[0]


def _non_single_marker(text: str) -> str | None:
    if text in {"=", "#"}:
        return text
    return None


def _marker_for_order(order: str) -> str:
    if order == "double":
        return "="
    if order == "triple":
        return "#"
    _lifecycle_violation("unsupported_bond_order")


def _order_for_marker(marker: str) -> Literal["double", "triple"]:
    if marker == "=":
        return "double"
    if marker == "#":
        return "triple"
    _lifecycle_violation("unsupported_marker")


def _lifecycle_violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer closure bond-text lifecycle violation: {kind}",
    )


__all__ = (
    "WriterClosureBondTextLifecycleEvidence",
    "closure_bond_text_lifecycle_evidence_from_transition",
    "validate_writer_closure_bond_text_lifecycle_transition",
)
