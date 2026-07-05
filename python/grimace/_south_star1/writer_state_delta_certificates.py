"""Successor-state delta certificates for checked writer branches."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .writer_events import WriterAtomEmitted
from .writer_events import WriterBondEmitted
from .writer_events import WriterBranchClosed
from .writer_events import WriterBranchOpened
from .writer_events import WriterComponentBoundaryEmitted
from .writer_events import WriterLocalOrderClosed
from .writer_events import WriterRingEndpointEmitted
from .writer_events import WriterRingEndpointPaired
from .writer_events import WriterRingLabelAllocated
from .writer_events import WriterRingLabelReleased
from .writer_state import WriterClosedClosure
from .writer_state import WriterAtomFrame
from .writer_state import WriterBranchFrame
from .writer_state import PendingWriterEntry
from .writer_state import WriterRingLabelState
from .writer_state import WriterOpenClosureEndpoint


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
    policy_delta_certificate: object | None = None
    ring_delta_certificate: object | None = None
    graph_delta_certificate: object | None = None
    stereo_delta_certificate: object | None = None
    policy_replay_certificate: object | None = None
    ring_replay_certificate: object | None = None
    graph_replay_certificate: object | None = None
    stereo_replay_certificate: object | None = None


@dataclass(frozen=True, slots=True)
class WriterPolicyStateDeltaCertificate:
    source_policy_state: object
    successor_policy_state: object
    events: tuple[object, ...]
    atom_text_added: tuple[tuple[object, str], ...]
    bond_text_added: tuple[tuple[object, str], ...]


@dataclass(frozen=True, slots=True)
class WriterRingStateDeltaCertificate:
    source_ring_state: object
    successor_ring_state: object
    events: tuple[object, ...]
    allocated_labels: tuple[object, ...]
    emitted_endpoints: tuple[object, ...]
    paired_endpoints: tuple[object, ...]
    released_labels: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class WriterGraphStateDeltaCertificate:
    source_state: object
    successor_state: object
    active_changed: bool
    branch_stack_changed: bool
    visited_atoms_added: frozenset[object]
    written_bonds_added: frozenset[object]
    component_cursor_changed: bool
    obligations_changed: bool
    graph_action_surface: object | None
    graph_obligation_work_evidence: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class WriterStereoStateDeltaCertificate:
    source_stereo_state: object
    successor_stereo_state: object
    stereo_lifecycle_evidence: tuple[object, ...]
    residual_work_evidence: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class WriterPolicyStateReplayCertificate:
    source_policy_state: object
    expected_successor_policy_state: object
    actual_successor_policy_state: object
    atom_text_events: tuple[object, ...]
    bond_text_events: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class WriterRingStateReplayCertificate:
    source_ring_state: object
    expected_successor_ring_state: object
    actual_successor_ring_state: object
    ring_events: tuple[object, ...]
    replayed_open_endpoints: tuple[object, ...] = ()
    replayed_closed_closures: tuple[object, ...] = ()
    replayed_label_state: object | None = None


@dataclass(frozen=True, slots=True)
class WriterGraphSuccessorProjection:
    component_cursor: object
    active: object
    branch_stack: tuple[object, ...]
    visited_atoms: frozenset[object]
    written_bonds: frozenset[object]
    obligations: object


class WriterGraphObligationReplayKind(Enum):
    UNCHANGED = "unchanged"
    PENDING_ENTRY_CREATED = "pending_entry_created"
    PENDING_ENTRY_DISCHARGED = "pending_entry_discharged"
    PENDING_ENTRY_TRANSFORMED = "pending_entry_transformed"
    EVIDENCE_BOUND_INCOMPLETE = "evidence_bound_incomplete"


@dataclass(frozen=True, slots=True)
class WriterGraphObligationReplayCertificate:
    kind: WriterGraphObligationReplayKind
    source_obligations: object
    expected_successor_obligations: object
    actual_successor_obligations: object
    graph_action_surface: object | None
    graph_obligation_work_evidence: tuple[object, ...]
    replay_complete: bool
    event_view: object | None = None


@dataclass(frozen=True, slots=True)
class WriterGraphStateReplayCertificate:
    source_state: object
    expected_successor_projection: WriterGraphSuccessorProjection
    actual_successor_state: object
    graph_action_surface: object | None
    graph_obligation_work_evidence: tuple[object, ...]
    obligation_replay_certificate: object | None = None


@dataclass(frozen=True, slots=True)
class WriterStereoStateReplayCertificate:
    source_stereo_state: object
    expected_successor_stereo_state: object
    actual_successor_stereo_state: object
    stereo_lifecycle_evidence: tuple[object, ...]
    residual_work_evidence: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class WriterEventDeltaView:
    atom_events: tuple[WriterAtomEmitted, ...]
    bond_events: tuple[WriterBondEmitted, ...]
    branch_open_events: tuple[WriterBranchOpened, ...]
    branch_close_events: tuple[WriterBranchClosed, ...]
    component_boundary_events: tuple[WriterComponentBoundaryEmitted, ...]
    local_order_events: tuple[WriterLocalOrderClosed, ...]
    ring_label_allocated_events: tuple[WriterRingLabelAllocated, ...]
    ring_label_released_events: tuple[WriterRingLabelReleased, ...]
    ring_endpoint_emitted_events: tuple[WriterRingEndpointEmitted, ...]
    ring_endpoint_paired_events: tuple[WriterRingEndpointPaired, ...]


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
    policy_delta_certificate = _policy_state_delta_certificate(
        source_state=source_state,
        successor_state=successor_state,
        events=events,
    )
    ring_delta_certificate = _ring_state_delta_certificate(
        source_state=source_state,
        successor_state=successor_state,
        events=events,
    )
    graph_delta_certificate = _graph_state_delta_certificate(
        source_state=source_state,
        successor_state=successor_state,
        graph_action_surface=graph_action_surface,
        graph_obligation_work_evidence=graph_obligation_work_evidence,
    )
    stereo_delta_certificate = _stereo_state_delta_certificate(
        source_state=source_state,
        successor_state=successor_state,
        stereo_lifecycle_evidence=stereo_lifecycle_evidence,
        residual_work_evidence=residual_work_evidence,
    )
    _validate_changed_field_coverage(
        deltas=deltas,
        policy_delta_certificate=policy_delta_certificate,
        ring_delta_certificate=ring_delta_certificate,
        graph_delta_certificate=graph_delta_certificate,
        stereo_delta_certificate=stereo_delta_certificate,
    )
    policy_replay_certificate = _policy_state_replay_certificate(
        source_state=source_state,
        successor_state=successor_state,
        events=events,
    )
    ring_replay_certificate = _ring_state_replay_certificate(
        source_state=source_state,
        successor_state=successor_state,
        events=events,
    )
    graph_replay_certificate = _graph_state_replay_certificate(
        source_state=source_state,
        successor_state=successor_state,
        events=events,
        graph_action_surface=graph_action_surface,
        graph_obligation_work_evidence=graph_obligation_work_evidence,
    )
    stereo_replay_certificate = _stereo_state_replay_certificate(
        source_state=source_state,
        successor_state=successor_state,
        stereo_lifecycle_evidence=stereo_lifecycle_evidence,
        residual_work_evidence=residual_work_evidence,
    )
    _validate_replay_coverage(
        policy_delta_certificate=policy_delta_certificate,
        ring_delta_certificate=ring_delta_certificate,
        graph_delta_certificate=graph_delta_certificate,
        stereo_delta_certificate=stereo_delta_certificate,
        policy_replay_certificate=policy_replay_certificate,
        ring_replay_certificate=ring_replay_certificate,
        graph_replay_certificate=graph_replay_certificate,
        stereo_replay_certificate=stereo_replay_certificate,
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

    certificate = WriterBranchSuccessorStateCertificate(
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
        policy_delta_certificate=policy_delta_certificate,
        ring_delta_certificate=ring_delta_certificate,
        graph_delta_certificate=graph_delta_certificate,
        stereo_delta_certificate=stereo_delta_certificate,
        policy_replay_certificate=policy_replay_certificate,
        ring_replay_certificate=ring_replay_certificate,
        graph_replay_certificate=graph_replay_certificate,
        stereo_replay_certificate=stereo_replay_certificate,
        **deltas,
    )
    validate_writer_branch_successor_state_certificate(certificate)
    return certificate


def writer_event_delta_view(events: tuple[object, ...]) -> WriterEventDeltaView:
    return WriterEventDeltaView(
        atom_events=tuple(
            event for event in events if isinstance(event, WriterAtomEmitted)
        ),
        bond_events=tuple(
            event for event in events if isinstance(event, WriterBondEmitted)
        ),
        branch_open_events=tuple(
            event for event in events if isinstance(event, WriterBranchOpened)
        ),
        branch_close_events=tuple(
            event for event in events if isinstance(event, WriterBranchClosed)
        ),
        component_boundary_events=tuple(
            event
            for event in events
            if isinstance(event, WriterComponentBoundaryEmitted)
        ),
        local_order_events=tuple(
            event for event in events if isinstance(event, WriterLocalOrderClosed)
        ),
        ring_label_allocated_events=tuple(
            event
            for event in events
            if isinstance(event, WriterRingLabelAllocated)
        ),
        ring_label_released_events=tuple(
            event
            for event in events
            if isinstance(event, WriterRingLabelReleased)
        ),
        ring_endpoint_emitted_events=tuple(
            event
            for event in events
            if isinstance(event, WriterRingEndpointEmitted)
        ),
        ring_endpoint_paired_events=tuple(
            event
            for event in events
            if isinstance(event, WriterRingEndpointPaired)
        ),
    )


def validate_writer_branch_successor_state_certificate(certificate) -> None:
    validate_writer_state_field_deltas(certificate)

    _validate_domain_certificate_values(certificate)
    _validate_changed_field_coverage(
        deltas=_observed_field_deltas(certificate),
        policy_delta_certificate=certificate.policy_delta_certificate,
        ring_delta_certificate=certificate.ring_delta_certificate,
        graph_delta_certificate=certificate.graph_delta_certificate,
        stereo_delta_certificate=certificate.stereo_delta_certificate,
    )
    _validate_replay_coverage(
        policy_delta_certificate=certificate.policy_delta_certificate,
        ring_delta_certificate=certificate.ring_delta_certificate,
        graph_delta_certificate=certificate.graph_delta_certificate,
        stereo_delta_certificate=certificate.stereo_delta_certificate,
        policy_replay_certificate=certificate.policy_replay_certificate,
        ring_replay_certificate=certificate.ring_replay_certificate,
        graph_replay_certificate=certificate.graph_replay_certificate,
        stereo_replay_certificate=certificate.stereo_replay_certificate,
    )
    _validate_replay_certificate_values(certificate)


def validate_writer_state_field_deltas(certificate) -> None:
    expected = _state_field_deltas(
        certificate.source_state,
        certificate.successor_state,
    )
    observed = _observed_field_deltas(certificate)

    for name, expected_delta in expected.items():
        observed_delta = observed[name]
        if observed_delta.field != expected_delta.field:
            _delta_violation(f"{name}_field_mismatch")
        if observed_delta.source_value != expected_delta.source_value:
            _delta_violation(f"{name}_source_value_mismatch")
        if observed_delta.successor_value != expected_delta.successor_value:
            _delta_violation(f"{name}_successor_value_mismatch")
        if observed_delta.changed != expected_delta.changed:
            _delta_violation(f"{name}_changed_mismatch")


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


def _observed_field_deltas(certificate) -> dict[str, object]:
    return dict(
        component_cursor_delta=certificate.component_cursor_delta,
        active_delta=certificate.active_delta,
        branch_stack_delta=certificate.branch_stack_delta,
        visited_atoms_delta=certificate.visited_atoms_delta,
        written_bonds_delta=certificate.written_bonds_delta,
        obligations_delta=certificate.obligations_delta,
        ring_state_delta=certificate.ring_state_delta,
        stereo_state_delta=certificate.stereo_state_delta,
        policy_state_delta=certificate.policy_state_delta,
    )


def _policy_state_delta_certificate(
    *,
    source_state,
    successor_state,
    events: tuple[object, ...],
) -> WriterPolicyStateDeltaCertificate | None:
    if source_state.policy_state == successor_state.policy_state:
        return None

    source_atoms = dict(source_state.policy_state.atom_text)
    successor_atoms = dict(successor_state.policy_state.atom_text)
    source_bonds = dict(source_state.policy_state.bond_text)
    successor_bonds = dict(successor_state.policy_state.bond_text)

    atom_text_added = tuple(
        (atom, text)
        for atom, text in successor_atoms.items()
        if source_atoms.get(atom) != text
    )
    bond_text_added = tuple(
        (bond, text)
        for bond, text in successor_bonds.items()
        if source_bonds.get(bond) != text
    )

    if not atom_text_added and not bond_text_added:
        _delta_violation("policy_delta_without_additions")

    event_view = writer_event_delta_view(events)
    event_payloads = _policy_event_payloads(event_view)
    for payload in atom_text_added:
        if payload not in event_payloads.atom_text:
            _delta_violation("atom_policy_delta_lacks_event")
    for payload in bond_text_added:
        if payload not in event_payloads.bond_text:
            _delta_violation("bond_policy_delta_lacks_event")

    if not (
        event_view.atom_events
        or event_view.bond_events
        or event_view.ring_endpoint_emitted_events
        or event_view.ring_endpoint_paired_events
    ):
        _delta_violation("policy_delta_without_emission_event")

    return WriterPolicyStateDeltaCertificate(
        source_policy_state=source_state.policy_state,
        successor_policy_state=successor_state.policy_state,
        events=events,
        atom_text_added=atom_text_added,
        bond_text_added=bond_text_added,
    )


def _policy_event_payloads(event_view: WriterEventDeltaView):
    atom_text = tuple(
        (event.atom, event.text)
        for event in event_view.atom_events
    )
    bond_text = tuple(
        (event.bond, event.text)
        for event in event_view.bond_events
    ) + tuple(
        (event.bond, event.bond_text)
        for event in (
            *event_view.ring_endpoint_emitted_events,
            *event_view.ring_endpoint_paired_events,
        )
    )
    return SimpleNamespace(
        atom_text=atom_text,
        bond_text=bond_text,
    )


def _ring_events_from_view(event_view: WriterEventDeltaView) -> tuple[object, ...]:
    return (
        event_view.ring_label_allocated_events
        + event_view.ring_endpoint_emitted_events
        + event_view.ring_endpoint_paired_events
        + event_view.ring_label_released_events
    )


def _open_endpoint_from_event(
    event: WriterRingEndpointEmitted,
) -> WriterOpenClosureEndpoint:
    return WriterOpenClosureEndpoint(
        bond=event.bond,
        first_atom=event.endpoint_atom,
        second_atom=event.partner_atom,
        label=event.label,
        first_endpoint_text=event.endpoint_text,
        first_endpoint_bond_text=event.bond_text,
        first_endpoint_direction_mark=event.direction_mark,
    )


def _closed_closure_from_pair_event(
    event: WriterRingEndpointPaired,
    *,
    source_open: frozenset[WriterOpenClosureEndpoint],
) -> WriterClosedClosure:
    matches = tuple(
        endpoint
        for endpoint in source_open
        if endpoint.bond == event.bond and endpoint.label == event.label
    )
    if len(matches) != 1:
        _delta_violation("ring_replay_pair_lacks_matching_open_endpoint")

    endpoint = matches[0]
    return WriterClosedClosure(
        bond=event.bond,
        first_atom=endpoint.first_atom,
        second_atom=endpoint.second_atom,
        label=event.label,
        first_endpoint_text=endpoint.first_endpoint_text,
        second_endpoint_text=event.endpoint_text,
        first_endpoint_bond_text=event.first_endpoint_bond_text,
        second_endpoint_bond_text=event.bond_text,
        first_endpoint_direction_mark=event.first_endpoint_direction_mark,
        second_endpoint_direction_mark=event.direction_mark,
    )


def _label_sort_key(label) -> tuple[object, ...]:
    return (label.value, label.text)


def _open_endpoint_sort_key(endpoint) -> tuple[object, ...]:
    return (
        int(endpoint.bond),
        int(endpoint.first_atom),
        int(endpoint.second_atom),
        endpoint.label.value,
        endpoint.label.text,
    )


def _closed_closure_sort_key(closure) -> tuple[object, ...]:
    return (
        int(closure.bond),
        int(closure.first_atom),
        int(closure.second_atom),
        closure.label.value,
        closure.label.text,
    )


def _ring_state_delta_certificate(
    *,
    source_state,
    successor_state,
    events: tuple[object, ...],
) -> WriterRingStateDeltaCertificate | None:
    if source_state.ring_state == successor_state.ring_state:
        return None

    event_view = writer_event_delta_view(events)
    ring_events = _ring_events_from_view(event_view)
    if not ring_events:
        _delta_violation("ring_delta_without_ring_lifecycle_event")

    source_open = frozenset(source_state.ring_state.open_endpoints)
    successor_open = frozenset(successor_state.ring_state.open_endpoints)
    source_closed = frozenset(source_state.ring_state.closed_closures)
    successor_closed = frozenset(successor_state.ring_state.closed_closures)
    if source_open != successor_open and not (
        event_view.ring_endpoint_emitted_events
        or event_view.ring_endpoint_paired_events
    ):
        _delta_violation("open_endpoint_delta_without_endpoint_event")
    if source_closed != successor_closed and not event_view.ring_endpoint_paired_events:
        _delta_violation("closed_closure_delta_without_pair_event")

    return WriterRingStateDeltaCertificate(
        source_ring_state=source_state.ring_state,
        successor_ring_state=successor_state.ring_state,
        events=ring_events,
        allocated_labels=event_view.ring_label_allocated_events,
        emitted_endpoints=event_view.ring_endpoint_emitted_events,
        paired_endpoints=event_view.ring_endpoint_paired_events,
        released_labels=event_view.ring_label_released_events,
    )


def _stereo_state_delta_certificate(
    *,
    source_state,
    successor_state,
    stereo_lifecycle_evidence: tuple[object, ...],
    residual_work_evidence: tuple[object, ...],
) -> WriterStereoStateDeltaCertificate | None:
    if source_state.stereo_state == successor_state.stereo_state:
        return None

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

    return WriterStereoStateDeltaCertificate(
        source_stereo_state=source_state.stereo_state,
        successor_stereo_state=successor_state.stereo_state,
        stereo_lifecycle_evidence=stereo_lifecycle_evidence,
        residual_work_evidence=residual_work_evidence,
    )


def _graph_state_delta_certificate(
    *,
    source_state,
    successor_state,
    graph_action_surface,
    graph_obligation_work_evidence: tuple[object, ...],
) -> WriterGraphStateDeltaCertificate | None:
    visited_atoms_added = (
        successor_state.visited_atoms - source_state.visited_atoms
    )
    written_bonds_added = (
        successor_state.written_bonds - source_state.written_bonds
    )
    changed = any(
        (
            source_state.component_cursor != successor_state.component_cursor,
            source_state.active != successor_state.active,
            source_state.branch_stack != successor_state.branch_stack,
            source_state.visited_atoms != successor_state.visited_atoms,
            source_state.written_bonds != successor_state.written_bonds,
            source_state.obligations != successor_state.obligations,
        )
    )
    if not changed:
        return None

    if not source_state.visited_atoms <= successor_state.visited_atoms:
        _delta_violation("visited_atoms_not_monotone")
    if not source_state.written_bonds <= successor_state.written_bonds:
        _delta_violation("written_bonds_not_monotone")

    if graph_action_surface is None and not graph_obligation_work_evidence:
        _delta_violation("graph_delta_without_graph_surface_or_evidence")

    return WriterGraphStateDeltaCertificate(
        source_state=source_state,
        successor_state=successor_state,
        active_changed=source_state.active != successor_state.active,
        branch_stack_changed=(
            source_state.branch_stack != successor_state.branch_stack
        ),
        visited_atoms_added=frozenset(visited_atoms_added),
        written_bonds_added=frozenset(written_bonds_added),
        component_cursor_changed=(
            source_state.component_cursor != successor_state.component_cursor
        ),
        obligations_changed=(
            source_state.obligations != successor_state.obligations
        ),
        graph_action_surface=graph_action_surface,
        graph_obligation_work_evidence=graph_obligation_work_evidence,
    )


def _policy_state_replay_certificate(
    *,
    source_state,
    successor_state,
    events: tuple[object, ...],
) -> WriterPolicyStateReplayCertificate | None:
    if source_state.policy_state == successor_state.policy_state:
        return None

    atom_text = dict(source_state.policy_state.atom_text)
    bond_text = dict(source_state.policy_state.bond_text)
    event_view = writer_event_delta_view(events)

    for event in event_view.atom_events:
        atom_text[event.atom] = event.text
    for event in event_view.bond_events:
        bond_text[event.bond] = event.text
    for event in (
        *event_view.ring_endpoint_emitted_events,
        *event_view.ring_endpoint_paired_events,
    ):
        bond_text[event.bond] = event.bond_text

    expected = source_state.policy_state.__class__(
        atom_text=tuple(sorted(atom_text.items(), key=lambda item: int(item[0]))),
        bond_text=tuple(sorted(bond_text.items(), key=lambda item: int(item[0]))),
    )
    if expected != successor_state.policy_state:
        _delta_violation("policy_replay_successor_mismatch")

    return WriterPolicyStateReplayCertificate(
        source_policy_state=source_state.policy_state,
        expected_successor_policy_state=expected,
        actual_successor_policy_state=successor_state.policy_state,
        atom_text_events=event_view.atom_events,
        bond_text_events=(
            event_view.bond_events
            + event_view.ring_endpoint_emitted_events
            + event_view.ring_endpoint_paired_events
        ),
    )


def _ring_state_replay_certificate(
    *,
    source_state,
    successor_state,
    events: tuple[object, ...],
) -> WriterRingStateReplayCertificate | None:
    if source_state.ring_state == successor_state.ring_state:
        return None

    event_view = writer_event_delta_view(events)
    ring_events = _ring_events_from_view(event_view)
    if not ring_events:
        _delta_violation("ring_replay_lacks_events")

    expected_open = set(source_state.ring_state.open_endpoints)
    expected_closed = set(source_state.ring_state.closed_closures)
    expected_allocated = list(source_state.ring_state.label_state.allocated)
    expected_reusable = list(source_state.ring_state.label_state.reusable)

    for event in event_view.ring_label_allocated_events:
        if event.source == "reused":
            if event.label not in expected_reusable:
                _delta_violation("ring_replay_reused_label_not_available")
            expected_reusable.remove(event.label)
        if event.label not in expected_allocated:
            expected_allocated.append(event.label)

    for event in event_view.ring_endpoint_emitted_events:
        if event.side == "open":
            expected_open.add(_open_endpoint_from_event(event))

    for event in event_view.ring_endpoint_paired_events:
        closed = _closed_closure_from_pair_event(
            event,
            source_open=frozenset(expected_open),
        )
        matches = tuple(
            endpoint
            for endpoint in expected_open
            if endpoint.bond == event.bond and endpoint.label == event.label
        )
        if len(matches) != 1:
            _delta_violation("ring_replay_pair_lacks_matching_open_endpoint")
        expected_open.remove(matches[0])
        expected_closed.add(closed)

    for event in event_view.ring_label_released_events:
        if event.label in expected_allocated:
            expected_allocated.remove(event.label)
        if event.label not in expected_reusable:
            expected_reusable.append(event.label)

    expected_ring_state = successor_state.ring_state.__class__(
        open_endpoints=tuple(sorted(expected_open, key=_open_endpoint_sort_key)),
        closed_closures=tuple(
            sorted(expected_closed, key=_closed_closure_sort_key)
        ),
        label_state=WriterRingLabelState(
            allocated=tuple(sorted(expected_allocated, key=_label_sort_key)),
            reusable=tuple(sorted(expected_reusable, key=_label_sort_key)),
        ),
    )

    source_open = frozenset(source_state.ring_state.open_endpoints)
    successor_open = frozenset(successor_state.ring_state.open_endpoints)
    source_closed = frozenset(source_state.ring_state.closed_closures)
    successor_closed = frozenset(successor_state.ring_state.closed_closures)
    added_open = successor_open - source_open
    removed_open = source_open - successor_open
    added_closed = successor_closed - source_closed
    expected_added_open = frozenset(expected_open) - source_open
    expected_added_closed = frozenset(expected_closed) - source_closed
    if added_open != expected_added_open:
        _delta_violation("ring_replay_added_open_mismatch")
    if added_closed != expected_added_closed:
        _delta_violation("ring_replay_added_closed_mismatch")
    if added_open and not event_view.ring_endpoint_emitted_events:
        _delta_violation("ring_replay_added_open_without_emit_event")
    if removed_open and not event_view.ring_endpoint_paired_events:
        _delta_violation("ring_replay_removed_open_without_pair_event")
    if added_closed and not event_view.ring_endpoint_paired_events:
        _delta_violation("ring_replay_added_closed_without_pair_event")
    if expected_ring_state != successor_state.ring_state:
        _delta_violation("ring_replay_successor_state_mismatch")

    return WriterRingStateReplayCertificate(
        source_ring_state=source_state.ring_state,
        expected_successor_ring_state=expected_ring_state,
        actual_successor_ring_state=successor_state.ring_state,
        ring_events=ring_events,
        replayed_open_endpoints=expected_ring_state.open_endpoints,
        replayed_closed_closures=expected_ring_state.closed_closures,
        replayed_label_state=expected_ring_state.label_state,
    )


def _graph_state_replay_certificate(
    *,
    source_state,
    successor_state,
    events: tuple[object, ...],
    graph_action_surface,
    graph_obligation_work_evidence: tuple[object, ...],
) -> WriterGraphStateReplayCertificate | None:
    changed = any(
        (
            source_state.component_cursor != successor_state.component_cursor,
            source_state.active != successor_state.active,
            source_state.branch_stack != successor_state.branch_stack,
            source_state.visited_atoms != successor_state.visited_atoms,
            source_state.written_bonds != successor_state.written_bonds,
            source_state.obligations != successor_state.obligations,
        )
    )
    if not changed:
        return None

    if not source_state.visited_atoms <= successor_state.visited_atoms:
        _delta_violation("graph_replay_visited_atoms_not_monotone")
    if not source_state.written_bonds <= successor_state.written_bonds:
        _delta_violation("graph_replay_written_bonds_not_monotone")
    if graph_action_surface is None and not graph_obligation_work_evidence:
        _delta_violation("graph_replay_lacks_action_or_evidence")

    expected_visited = set(source_state.visited_atoms)
    expected_written = set(source_state.written_bonds)
    expected_active = source_state.active
    expected_branch_stack = tuple(source_state.branch_stack)
    expected_component_cursor = source_state.component_cursor

    for event in events:
        if isinstance(event, WriterAtomEmitted):
            pending = getattr(source_state.obligations, "pending_entry", None)
            if pending is not None:
                if pending.parent != event.parent:
                    _delta_violation("obligation_replay_pending_parent_mismatch")
                if pending.child != event.atom:
                    _delta_violation("obligation_replay_pending_child_mismatch")
                if pending.bond != event.incoming_bond:
                    _delta_violation("obligation_replay_pending_bond_mismatch")
                if pending.branch:
                    expected_branch_stack = (
                        *expected_branch_stack,
                        WriterBranchFrame(
                            return_atom=WriterAtomFrame(
                                atom=pending.parent,
                                parent=source_state.active.parent,
                                incoming_bond=source_state.active.incoming_bond,
                                atom_emitted=True,
                            )
                        ),
                    )

            expected_visited.add(event.atom)
            if event.incoming_bond is not None:
                expected_written.add(event.incoming_bond)
            expected_active = WriterAtomFrame(
                atom=event.atom,
                parent=event.parent,
                incoming_bond=event.incoming_bond,
                atom_emitted=True,
            )
        elif isinstance(event, WriterRingEndpointEmitted):
            expected_written.add(event.bond)
        elif isinstance(event, WriterRingEndpointPaired):
            expected_written.add(event.bond)
        elif isinstance(event, WriterBranchClosed):
            if not expected_branch_stack:
                _delta_violation("graph_replay_branch_close_without_stack")
            if event.atom != expected_active.atom:
                _delta_violation("graph_replay_branch_close_atom_mismatch")
            expected_active = expected_branch_stack[-1].return_atom
            expected_branch_stack = expected_branch_stack[:-1]
        elif isinstance(event, WriterComponentBoundaryEmitted):
            roots = expected_component_cursor.component_roots
            try:
                component_index = roots.index(event.next_root)
            except ValueError:
                _delta_violation("graph_replay_component_boundary_root_mismatch")
            expected_component_cursor = expected_component_cursor.__class__(
                component_index=component_index,
                component_roots=roots,
            )
            expected_active = WriterAtomFrame(
                atom=event.next_root,
                parent=None,
                incoming_bond=None,
                atom_emitted=False,
            )

    if frozenset(expected_visited) != successor_state.visited_atoms:
        _delta_violation("graph_replay_visited_atoms_mismatch")
    if frozenset(expected_written) != successor_state.written_bonds:
        _delta_violation("graph_replay_written_bonds_mismatch")
    if expected_active != successor_state.active:
        _delta_violation("graph_replay_active_mismatch")
    if expected_branch_stack != successor_state.branch_stack:
        _delta_violation("graph_replay_branch_stack_mismatch")
    if expected_component_cursor != successor_state.component_cursor:
        _delta_violation("graph_replay_component_cursor_mismatch")

    obligation_replay_certificate = _graph_obligation_replay_certificate(
        source_state=source_state,
        successor_state=successor_state,
        event_view=writer_event_delta_view(events),
        graph_action_surface=graph_action_surface,
        graph_obligation_work_evidence=graph_obligation_work_evidence,
    )
    if graph_action_surface is not None:
        surface_atom = getattr(graph_action_surface, "partner_atom", None)
        surface_bond = getattr(graph_action_surface, "bond", None)
        added_atoms = successor_state.visited_atoms - source_state.visited_atoms
        added_bonds = successor_state.written_bonds - source_state.written_bonds
        if added_atoms and surface_atom is not None and surface_atom not in added_atoms:
            _delta_violation("graph_replay_added_atom_surface_mismatch")
        if added_bonds and surface_bond is not None and surface_bond not in added_bonds:
            _delta_violation("graph_replay_added_bond_surface_mismatch")

    return WriterGraphStateReplayCertificate(
        source_state=source_state,
        expected_successor_projection=WriterGraphSuccessorProjection(
            visited_atoms=frozenset(expected_visited),
            written_bonds=frozenset(expected_written),
            active=expected_active,
            branch_stack=expected_branch_stack,
            component_cursor=expected_component_cursor,
            obligations=successor_state.obligations,
        ),
        actual_successor_state=successor_state,
        graph_action_surface=graph_action_surface,
        graph_obligation_work_evidence=graph_obligation_work_evidence,
        obligation_replay_certificate=obligation_replay_certificate,
    )


def _graph_obligation_replay_certificate(
    *,
    source_state,
    successor_state,
    event_view: WriterEventDeltaView,
    graph_action_surface,
    graph_obligation_work_evidence: tuple[object, ...],
) -> WriterGraphObligationReplayCertificate:
    if source_state.obligations == successor_state.obligations:
        return WriterGraphObligationReplayCertificate(
            kind=WriterGraphObligationReplayKind.UNCHANGED,
            source_obligations=source_state.obligations,
            expected_successor_obligations=source_state.obligations,
            actual_successor_obligations=successor_state.obligations,
            graph_action_surface=graph_action_surface,
            graph_obligation_work_evidence=graph_obligation_work_evidence,
            replay_complete=True,
            event_view=event_view,
        )
    if graph_action_surface is None and not graph_obligation_work_evidence:
        _delta_violation("obligation_replay_lacks_graph_evidence")

    expected, kind, complete = _expected_obligations_from_graph_replay(
        source_state=source_state,
        successor_state=successor_state,
        event_view=event_view,
        graph_action_surface=graph_action_surface,
    )
    if complete and expected != successor_state.obligations:
        _delta_violation("obligation_replay_successor_mismatch")
    return WriterGraphObligationReplayCertificate(
        kind=kind,
        source_obligations=source_state.obligations,
        expected_successor_obligations=expected,
        actual_successor_obligations=successor_state.obligations,
        graph_action_surface=graph_action_surface,
        graph_obligation_work_evidence=graph_obligation_work_evidence,
        replay_complete=complete,
        event_view=event_view,
    )


def _expected_obligations_from_graph_replay(
    *,
    source_state,
    successor_state,
    event_view: WriterEventDeltaView,
    graph_action_surface,
) -> tuple[object, WriterGraphObligationReplayKind, bool]:
    expected = source_state.obligations
    pending = getattr(expected, "pending_entry", None)
    discharged = False

    for event in event_view.atom_events:
        if pending is None:
            continue
        if pending.parent != event.parent:
            _delta_violation("obligation_replay_pending_parent_mismatch")
        if pending.child != event.atom:
            _delta_violation("obligation_replay_pending_child_mismatch")
        if pending.bond != event.incoming_bond:
            _delta_violation("obligation_replay_pending_bond_mismatch")
        expected = expected.__class__(pending_entry=None)
        pending = None
        discharged = True

    if discharged:
        return (
            expected,
            WriterGraphObligationReplayKind.PENDING_ENTRY_DISCHARGED,
            True,
        )

    if getattr(source_state.obligations, "pending_entry", None) is None:
        pending_entry = _pending_entry_from_graph_action_surface(graph_action_surface)
        if pending_entry is not None:
            expected = source_state.obligations.__class__(
                pending_entry=pending_entry,
            )
            return (
                expected,
                WriterGraphObligationReplayKind.PENDING_ENTRY_CREATED,
                True,
            )

    return (
        successor_state.obligations,
        WriterGraphObligationReplayKind.EVIDENCE_BOUND_INCOMPLETE,
        False,
    )


def _pending_entry_from_graph_action_surface(surface) -> PendingWriterEntry | None:
    if surface is None:
        return None
    kind = getattr(getattr(surface, "kind", None), "value", None)
    if kind != "open_branch":
        return None
    parent = getattr(surface, "active_atom", None)
    child = getattr(surface, "partner_atom", None)
    bond = getattr(surface, "bond", None)
    if parent is None or child is None or bond is None:
        return None
    return PendingWriterEntry(
        parent=parent,
        child=child,
        bond=bond,
        branch=True,
    )


def _stereo_state_replay_certificate(
    *,
    source_state,
    successor_state,
    stereo_lifecycle_evidence: tuple[object, ...],
    residual_work_evidence: tuple[object, ...],
) -> WriterStereoStateReplayCertificate | None:
    if source_state.stereo_state == successor_state.stereo_state:
        return None

    if not stereo_lifecycle_evidence:
        _delta_violation("stereo_replay_lacks_lifecycle")

    lifecycle_work = tuple(
        item
        for evidence in stereo_lifecycle_evidence
        for item in getattr(evidence, "residual_work_evidence", ())
    )
    for item in lifecycle_work:
        if item not in residual_work_evidence:
            _delta_violation("stereo_replay_work_evidence_missing")

    return WriterStereoStateReplayCertificate(
        source_stereo_state=source_state.stereo_state,
        expected_successor_stereo_state=successor_state.stereo_state,
        actual_successor_stereo_state=successor_state.stereo_state,
        stereo_lifecycle_evidence=stereo_lifecycle_evidence,
        residual_work_evidence=residual_work_evidence,
    )


def _validate_changed_field_coverage(
    *,
    deltas: dict[str, object],
    policy_delta_certificate,
    ring_delta_certificate,
    graph_delta_certificate,
    stereo_delta_certificate,
) -> None:
    if deltas["policy_state_delta"].changed and policy_delta_certificate is None:
        _delta_violation("policy_delta_lacks_certificate")
    if deltas["ring_state_delta"].changed and ring_delta_certificate is None:
        _delta_violation("ring_delta_lacks_certificate")
    graph_delta_names = (
        "component_cursor_delta",
        "active_delta",
        "branch_stack_delta",
        "visited_atoms_delta",
        "written_bonds_delta",
        "obligations_delta",
    )
    if (
        any(deltas[name].changed for name in graph_delta_names)
        and graph_delta_certificate is None
    ):
        _delta_violation("graph_delta_lacks_certificate")
    if deltas["stereo_state_delta"].changed and stereo_delta_certificate is None:
        _delta_violation("stereo_delta_lacks_certificate")


def _validate_replay_coverage(
    *,
    policy_delta_certificate,
    ring_delta_certificate,
    graph_delta_certificate,
    stereo_delta_certificate,
    policy_replay_certificate,
    ring_replay_certificate,
    graph_replay_certificate,
    stereo_replay_certificate,
) -> None:
    if policy_delta_certificate is not None and policy_replay_certificate is None:
        _delta_violation("policy_delta_lacks_replay_certificate")
    if ring_delta_certificate is not None and ring_replay_certificate is None:
        _delta_violation("ring_delta_lacks_replay_certificate")
    if graph_delta_certificate is not None and graph_replay_certificate is None:
        _delta_violation("graph_delta_lacks_replay_certificate")
    if stereo_delta_certificate is not None and stereo_replay_certificate is None:
        _delta_violation("stereo_delta_lacks_replay_certificate")


def _validate_domain_certificate_values(certificate) -> None:
    policy = certificate.policy_delta_certificate
    if policy is not None:
        if policy.source_policy_state != certificate.source_state.policy_state:
            _delta_violation("policy_certificate_source_mismatch")
        if policy.successor_policy_state != certificate.successor_state.policy_state:
            _delta_violation("policy_certificate_successor_mismatch")
        if policy.events != certificate.events:
            _delta_violation("policy_certificate_events_mismatch")

    ring = certificate.ring_delta_certificate
    if ring is not None:
        if ring.source_ring_state != certificate.source_state.ring_state:
            _delta_violation("ring_certificate_source_mismatch")
        if ring.successor_ring_state != certificate.successor_state.ring_state:
            _delta_violation("ring_certificate_successor_mismatch")

    graph = certificate.graph_delta_certificate
    if graph is not None:
        if graph.source_state != certificate.source_state:
            _delta_violation("graph_certificate_source_mismatch")
        if graph.successor_state != certificate.successor_state:
            _delta_violation("graph_certificate_successor_mismatch")
        if graph.graph_action_surface != certificate.graph_action_surface:
            _delta_violation("graph_certificate_surface_mismatch")
        if (
            graph.graph_obligation_work_evidence
            != certificate.graph_obligation_work_evidence
        ):
            _delta_violation("graph_certificate_evidence_mismatch")

    stereo = certificate.stereo_delta_certificate
    if stereo is not None:
        if stereo.source_stereo_state != certificate.source_state.stereo_state:
            _delta_violation("stereo_certificate_source_mismatch")
        if stereo.successor_stereo_state != certificate.successor_state.stereo_state:
            _delta_violation("stereo_certificate_successor_mismatch")
        if stereo.stereo_lifecycle_evidence != certificate.stereo_lifecycle_evidence:
            _delta_violation("stereo_certificate_lifecycle_mismatch")
        if stereo.residual_work_evidence != certificate.residual_work_evidence:
            _delta_violation("stereo_certificate_residual_work_mismatch")


def _validate_replay_certificate_values(certificate) -> None:
    policy = certificate.policy_replay_certificate
    if policy is not None:
        if policy.source_policy_state != certificate.source_state.policy_state:
            _delta_violation("policy_replay_source_mismatch")
        if (
            policy.expected_successor_policy_state
            != certificate.successor_state.policy_state
        ):
            _delta_violation("policy_replay_expected_successor_mismatch")
        if (
            policy.actual_successor_policy_state
            != certificate.successor_state.policy_state
        ):
            _delta_violation("policy_replay_successor_mismatch")

    ring = certificate.ring_replay_certificate
    if ring is not None:
        if ring.source_ring_state != certificate.source_state.ring_state:
            _delta_violation("ring_replay_source_mismatch")
        if (
            ring.expected_successor_ring_state
            != certificate.successor_state.ring_state
        ):
            _delta_violation("ring_replay_expected_successor_mismatch")
        if (
            ring.actual_successor_ring_state
            != certificate.successor_state.ring_state
        ):
            _delta_violation("ring_replay_successor_mismatch")
        if tuple(ring.replayed_open_endpoints) != tuple(
            ring.expected_successor_ring_state.open_endpoints
        ):
            _delta_violation("ring_replay_open_endpoints_mismatch")
        if tuple(ring.replayed_closed_closures) != tuple(
            ring.expected_successor_ring_state.closed_closures
        ):
            _delta_violation("ring_replay_closed_closures_mismatch")
        if ring.replayed_label_state != ring.expected_successor_ring_state.label_state:
            _delta_violation("ring_replay_label_state_mismatch")

    graph = certificate.graph_replay_certificate
    if graph is not None:
        if graph.source_state != certificate.source_state:
            _delta_violation("graph_replay_source_mismatch")
        if graph.actual_successor_state != certificate.successor_state:
            _delta_violation("graph_replay_successor_mismatch")
        if graph.graph_action_surface != certificate.graph_action_surface:
            _delta_violation("graph_replay_surface_mismatch")
        if (
            graph.graph_obligation_work_evidence
            != certificate.graph_obligation_work_evidence
        ):
            _delta_violation("graph_replay_evidence_mismatch")
        projection = graph.expected_successor_projection
        actual = graph.actual_successor_state
        if projection.visited_atoms != actual.visited_atoms:
            _delta_violation("graph_replay_projection_visited_atoms_mismatch")
        if projection.written_bonds != actual.written_bonds:
            _delta_violation("graph_replay_projection_written_bonds_mismatch")
        if projection.active != actual.active:
            _delta_violation("graph_replay_projection_active_mismatch")
        if projection.branch_stack != actual.branch_stack:
            _delta_violation("graph_replay_projection_branch_stack_mismatch")
        if projection.component_cursor != actual.component_cursor:
            _delta_violation("graph_replay_projection_component_cursor_mismatch")
        if projection.obligations != actual.obligations:
            _delta_violation("graph_replay_projection_obligations_mismatch")
        obligation_replay = graph.obligation_replay_certificate
        if obligation_replay is None:
            if certificate.source_state.obligations != certificate.successor_state.obligations:
                _delta_violation("graph_replay_lacks_obligation_certificate")
        else:
            if obligation_replay.source_obligations != certificate.source_state.obligations:
                _delta_violation("obligation_replay_source_mismatch")
            if (
                obligation_replay.actual_successor_obligations
                != certificate.successor_state.obligations
            ):
                _delta_violation("obligation_replay_actual_successor_mismatch")
            if (
                obligation_replay.replay_complete
                and obligation_replay.expected_successor_obligations
                != obligation_replay.actual_successor_obligations
            ):
                _delta_violation("obligation_replay_expected_successor_mismatch")
            if (
                obligation_replay.kind
                is WriterGraphObligationReplayKind.EVIDENCE_BOUND_INCOMPLETE
                and obligation_replay.replay_complete
            ):
                _delta_violation("obligation_replay_false_completion")
            if (
                obligation_replay.kind
                is not WriterGraphObligationReplayKind.EVIDENCE_BOUND_INCOMPLETE
                and not obligation_replay.replay_complete
            ):
                _delta_violation("obligation_replay_incomplete_kind_mismatch")
            if obligation_replay.graph_action_surface != certificate.graph_action_surface:
                _delta_violation("obligation_replay_surface_mismatch")
            if (
                obligation_replay.graph_obligation_work_evidence
                != certificate.graph_obligation_work_evidence
            ):
                _delta_violation("obligation_replay_evidence_mismatch")

    stereo = certificate.stereo_replay_certificate
    if stereo is not None:
        if stereo.source_stereo_state != certificate.source_state.stereo_state:
            _delta_violation("stereo_replay_source_mismatch")
        if (
            stereo.expected_successor_stereo_state
            != certificate.successor_state.stereo_state
        ):
            _delta_violation("stereo_replay_expected_successor_mismatch")
        if (
            stereo.actual_successor_stereo_state
            != certificate.successor_state.stereo_state
        ):
            _delta_violation("stereo_replay_successor_mismatch")
        if stereo.stereo_lifecycle_evidence != certificate.stereo_lifecycle_evidence:
            _delta_violation("stereo_replay_lifecycle_mismatch")
        if stereo.residual_work_evidence != certificate.residual_work_evidence:
            _delta_violation("stereo_replay_residual_work_mismatch")


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
    "WriterEventDeltaView",
    "WriterGraphObligationReplayCertificate",
    "WriterGraphObligationReplayKind",
    "WriterGraphStateDeltaCertificate",
    "WriterGraphStateReplayCertificate",
    "WriterGraphSuccessorProjection",
    "WriterPolicyStateDeltaCertificate",
    "WriterPolicyStateReplayCertificate",
    "WriterRingStateDeltaCertificate",
    "WriterRingStateReplayCertificate",
    "WriterStereoStateDeltaCertificate",
    "WriterStereoStateReplayCertificate",
    "WriterStateFieldDelta",
    "validate_writer_branch_successor_state_certificate",
    "validate_writer_state_field_deltas",
    "writer_branch_successor_state_certificate",
    "writer_event_delta_view",
)
