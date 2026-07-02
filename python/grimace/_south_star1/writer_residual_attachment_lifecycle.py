"""Residual-attachment graph lifecycle validation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .writer_graph_obligations import build_writer_graph_obligation_context
from .writer_graph_obligations import writer_residual_attachment_closure_deficit


class WriterResidualAttachmentLifecycleOutcomeKind(Enum):
    CLOSURE_OPEN_DISCHARGED = "closure_open_discharged"


@dataclass(frozen=True, slots=True)
class WriterResidualAttachmentLifecycleEvidence:
    attachment_id: int
    bond: object
    outcome_kind: WriterResidualAttachmentLifecycleOutcomeKind
    source_attachment: object
    successor_attachment: object
    source_closure_deficit: int
    successor_closure_deficit: int
    removed_boundary_bonds: tuple[object, ...]


def validate_writer_residual_attachment_lifecycle_transition(
    *,
    prepared,
    source_state,
    successor_state,
    graph_action_surface,
) -> None:
    violations = writer_residual_attachment_lifecycle_transition_violations(
        prepared=prepared,
        source_state=source_state,
        successor_state=successor_state,
        graph_action_surface=graph_action_surface,
    )
    if violations:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            (
                "writer residual attachment lifecycle transition violation: "
                f"{violations[0]}"
            ),
        )


def writer_residual_attachment_lifecycle_transition_violations(
    *,
    prepared,
    source_state,
    successor_state,
    graph_action_surface,
) -> tuple[str, ...]:
    try:
        writer_residual_attachment_lifecycle_evidence_for_transition(
            prepared=prepared,
            source_state=source_state,
            successor_state=successor_state,
            graph_action_surface=graph_action_surface,
        )
    except _ResidualAttachmentLifecycleViolation as caught:
        return (caught.kind,)

    return ()


def writer_residual_attachment_lifecycle_evidence_for_transition(
    *,
    prepared,
    source_state,
    successor_state,
    graph_action_surface,
) -> tuple[WriterResidualAttachmentLifecycleEvidence, ...]:
    if graph_action_surface is None:
        return ()
    if not _surface_is_closure_open(graph_action_surface):
        return ()

    attachment_id = getattr(graph_action_surface, "attachment_id", None)
    if attachment_id is None:
        return ()

    bond = getattr(graph_action_surface, "bond", None)
    if bond is None:
        _violation("closure_open_attachment_lacks_bond")

    source_context = build_writer_graph_obligation_context(
        prepared,
        source_state,
    )
    successor_context = build_writer_graph_obligation_context(
        prepared,
        successor_state,
    )
    source = _attachment_by_id(
        source_context.residual_summary.attachments.attachments,
        attachment_id,
    )
    successor = _matching_successor_attachment(
        source,
        successor_context.residual_summary.attachments.attachments,
    )
    expected_boundary = tuple(
        incidence for incidence in source.boundary if incidence.bond != bond
    )
    if len(expected_boundary) != len(source.boundary) - 1:
        _violation("selected_boundary_bond_missing")
    if successor.boundary != expected_boundary:
        _violation("successor_boundary_mismatch")

    source_deficit = writer_residual_attachment_closure_deficit(source)
    successor_deficit = writer_residual_attachment_closure_deficit(successor)
    if source_deficit != successor_deficit + 1:
        _violation("closure_deficit_delta_mismatch")

    return (
        WriterResidualAttachmentLifecycleEvidence(
            attachment_id=attachment_id,
            bond=bond,
            outcome_kind=(
                WriterResidualAttachmentLifecycleOutcomeKind
                .CLOSURE_OPEN_DISCHARGED
            ),
            source_attachment=source,
            successor_attachment=successor,
            source_closure_deficit=source_deficit,
            successor_closure_deficit=successor_deficit,
            removed_boundary_bonds=(bond,),
        ),
    )


def _surface_is_closure_open(graph_action_surface) -> bool:
    policy_family = getattr(graph_action_surface, "policy_family", None)
    return getattr(policy_family, "value", None) == "closure_open"


def _attachment_by_id(attachments, attachment_id: int):
    matches = tuple(
        attachment
        for attachment in attachments
        if attachment.attachment_id == attachment_id
    )
    if not matches:
        _violation("source_attachment_not_found")
    if len(matches) != 1:
        _violation("duplicate_source_attachment_id")
    return matches[0]


def _matching_successor_attachment(source, attachments):
    matches = tuple(
        attachment
        for attachment in attachments
        if (
            attachment.atoms == source.atoms
            and attachment.latent_bonds == source.latent_bonds
            and attachment.cyclic_rank == source.cyclic_rank
            and attachment.block_ids == source.block_ids
        )
    )
    if not matches:
        _violation("successor_attachment_not_found")
    if len(matches) != 1:
        _violation("duplicate_successor_attachment")
    return matches[0]


class _ResidualAttachmentLifecycleViolation(Exception):
    def __init__(self, kind: str) -> None:
        super().__init__(kind)
        self.kind = kind


def _violation(kind: str) -> None:
    raise _ResidualAttachmentLifecycleViolation(kind)


__all__ = (
    "WriterResidualAttachmentLifecycleEvidence",
    "WriterResidualAttachmentLifecycleOutcomeKind",
    "validate_writer_residual_attachment_lifecycle_transition",
    "writer_residual_attachment_lifecycle_evidence_for_transition",
    "writer_residual_attachment_lifecycle_transition_violations",
)
