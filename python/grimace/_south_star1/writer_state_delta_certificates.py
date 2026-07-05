"""Successor-state delta certificates for checked writer branches."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import SouthStarError
from .errors import SouthStarErrorKind


@dataclass(frozen=True, slots=True)
class WriterStateFieldDelta:
    field: str
    source_value: object
    successor_value: object
    changed: bool


@dataclass(frozen=True, slots=True)
class WriterBranchSuccessorStateCertificate:
    source_state: object
    successor_state: object
    emitted_text: str
    transition_kind: object
    graph_action_surface: object | None
    policy_family: object | None
    events: tuple[object, ...]
    transition_evidence: object

    component_cursor_delta: WriterStateFieldDelta
    active_delta: WriterStateFieldDelta
    branch_stack_delta: WriterStateFieldDelta
    visited_atoms_delta: WriterStateFieldDelta
    written_bonds_delta: WriterStateFieldDelta
    obligations_delta: WriterStateFieldDelta
    ring_state_delta: WriterStateFieldDelta
    stereo_state_delta: WriterStateFieldDelta
    policy_state_delta: WriterStateFieldDelta

    graph_obligation_work_evidence: tuple[object, ...]
    residual_work_evidence: tuple[object, ...]
    finite_relation_work_evidence: tuple[object, ...]
    closure_candidate_lifecycle_evidence: tuple[object, ...]
    residual_attachment_lifecycle_evidence: tuple[object, ...]
    stereo_lifecycle_evidence: tuple[object, ...]


def writer_branch_successor_state_certificate(
    *,
    source_state,
    successor_state,
    emitted_text: str,
    transition_kind,
    graph_action_surface,
    policy_family,
    events: tuple[object, ...],
    transition_evidence,
    graph_obligation_work_evidence: tuple[object, ...],
    residual_work_evidence: tuple[object, ...],
    finite_relation_work_evidence: tuple[object, ...],
    closure_candidate_lifecycle_evidence: tuple[object, ...],
    residual_attachment_lifecycle_evidence: tuple[object, ...],
    stereo_lifecycle_evidence: tuple[object, ...],
) -> WriterBranchSuccessorStateCertificate:
    if not emitted_text:
        _delta_violation("missing_emitted_text")

    deltas = _state_field_deltas(source_state, successor_state)
    _validate_ring_delta(
        source_state=source_state,
        successor_state=successor_state,
        events=events,
    )
    _validate_policy_delta(
        source_state=source_state,
        successor_state=successor_state,
        events=events,
    )
    _validate_stereo_delta(
        source_state=source_state,
        successor_state=successor_state,
        stereo_lifecycle_evidence=stereo_lifecycle_evidence,
        residual_work_evidence=residual_work_evidence,
    )
    _validate_obligation_delta(
        source_state=source_state,
        successor_state=successor_state,
        graph_obligation_work_evidence=graph_obligation_work_evidence,
        graph_action_surface=graph_action_surface,
    )
    _validate_closure_candidate_delta(
        closure_candidate_lifecycle_evidence=(
            closure_candidate_lifecycle_evidence
        ),
        graph_action_surface=graph_action_surface,
    )
    _validate_residual_attachment_delta(
        residual_attachment_lifecycle_evidence=(
            residual_attachment_lifecycle_evidence
        ),
        graph_action_surface=graph_action_surface,
    )

    return WriterBranchSuccessorStateCertificate(
        source_state=source_state,
        successor_state=successor_state,
        emitted_text=emitted_text,
        transition_kind=transition_kind,
        graph_action_surface=graph_action_surface,
        policy_family=policy_family,
        events=events,
        transition_evidence=transition_evidence,
        graph_obligation_work_evidence=graph_obligation_work_evidence,
        residual_work_evidence=residual_work_evidence,
        finite_relation_work_evidence=finite_relation_work_evidence,
        closure_candidate_lifecycle_evidence=(
            closure_candidate_lifecycle_evidence
        ),
        residual_attachment_lifecycle_evidence=(
            residual_attachment_lifecycle_evidence
        ),
        stereo_lifecycle_evidence=stereo_lifecycle_evidence,
        **deltas,
    )


def _delta(field: str, source, successor) -> WriterStateFieldDelta:
    return WriterStateFieldDelta(
        field=field,
        source_value=source,
        successor_value=successor,
        changed=source != successor,
    )


def _state_field_deltas(source_state, successor_state) -> dict[str, object]:
    return dict(
        component_cursor_delta=_delta(
            "component_cursor",
            source_state.component_cursor,
            successor_state.component_cursor,
        ),
        active_delta=_delta(
            "active",
            source_state.active,
            successor_state.active,
        ),
        branch_stack_delta=_delta(
            "branch_stack",
            source_state.branch_stack,
            successor_state.branch_stack,
        ),
        visited_atoms_delta=_delta(
            "visited_atoms",
            source_state.visited_atoms,
            successor_state.visited_atoms,
        ),
        written_bonds_delta=_delta(
            "written_bonds",
            source_state.written_bonds,
            successor_state.written_bonds,
        ),
        obligations_delta=_delta(
            "obligations",
            source_state.obligations,
            successor_state.obligations,
        ),
        ring_state_delta=_delta(
            "ring_state",
            source_state.ring_state,
            successor_state.ring_state,
        ),
        stereo_state_delta=_delta(
            "stereo_state",
            source_state.stereo_state,
            successor_state.stereo_state,
        ),
        policy_state_delta=_delta(
            "policy_state",
            source_state.policy_state,
            successor_state.policy_state,
        ),
    )


def _validate_policy_delta(
    *,
    source_state,
    successor_state,
    events: tuple[object, ...],
) -> None:
    if source_state.policy_state == successor_state.policy_state:
        return

    policy_event_names = {
        "WriterAtomEmitted",
        "WriterBondEmitted",
        "WriterRingEndpointEmitted",
        "WriterRingEndpointPaired",
    }
    if not any(event.__class__.__name__ in policy_event_names for event in events):
        _delta_violation("policy_delta_without_emission_event")


def _validate_ring_delta(
    *,
    source_state,
    successor_state,
    events: tuple[object, ...],
) -> None:
    if source_state.ring_state == successor_state.ring_state:
        return

    ring_event_names = {
        "WriterRingLabelAllocated",
        "WriterRingEndpointEmitted",
        "WriterRingEndpointPaired",
        "WriterRingLabelReleased",
    }
    if not any(event.__class__.__name__ in ring_event_names for event in events):
        _delta_violation("ring_delta_without_ring_lifecycle_event")


def _validate_stereo_delta(
    *,
    source_state,
    successor_state,
    stereo_lifecycle_evidence: tuple[object, ...],
    residual_work_evidence: tuple[object, ...],
) -> None:
    if source_state.stereo_state == successor_state.stereo_state:
        return

    if not stereo_lifecycle_evidence:
        _delta_violation("stereo_delta_without_lifecycle_evidence")

    lifecycle_work = tuple(
        item
        for evidence in stereo_lifecycle_evidence
        for item in getattr(evidence, "residual_work_evidence", ())
    )
    for item in lifecycle_work:
        if item not in residual_work_evidence:
            _delta_violation("stereo_delta_work_evidence_missing")


def _validate_obligation_delta(
    *,
    source_state,
    successor_state,
    graph_obligation_work_evidence: tuple[object, ...],
    graph_action_surface,
) -> None:
    if source_state.obligations == successor_state.obligations:
        return

    if graph_action_surface is None and not graph_obligation_work_evidence:
        _delta_violation("obligation_delta_without_graph_evidence")


def _validate_closure_candidate_delta(
    *,
    closure_candidate_lifecycle_evidence: tuple[object, ...],
    graph_action_surface,
) -> None:
    if closure_candidate_lifecycle_evidence and graph_action_surface is None:
        _delta_violation("closure_lifecycle_without_graph_action_surface")


def _validate_residual_attachment_delta(
    *,
    residual_attachment_lifecycle_evidence: tuple[object, ...],
    graph_action_surface,
) -> None:
    if residual_attachment_lifecycle_evidence and graph_action_surface is None:
        _delta_violation(
            "residual_attachment_lifecycle_without_graph_action_surface"
        )


def _delta_violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer branch successor-state certificate violation: {kind}",
    )


__all__ = (
    "WriterBranchSuccessorStateCertificate",
    "WriterStateFieldDelta",
    "writer_branch_successor_state_certificate",
)
