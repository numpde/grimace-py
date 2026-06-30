"""Ring-label lifecycle validation for writer transition streams."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .writer_capabilities import _WriterExecutionCapabilityKind
from .writer_events import WriterRingEndpointEmitted
from .writer_events import WriterRingEndpointPaired
from .writer_events import WriterRingLabelAllocated
from .writer_events import WriterRingLabelReleased
from .writer_state import WriterClosureLabel
from .writer_state import WriterOpenClosureEndpoint


WriterRingLabelAllocationSource = Literal["fresh", "reused"]
_TRANSITION_LIFECYCLE_INSTALLED_MARKER = "_ring_label_lifecycle_installed"


def writer_ring_label_allocation_source(
    *,
    source_state: object,
    label: WriterClosureLabel,
) -> WriterRingLabelAllocationSource:
    return "reused" if label in _labels(source_state, "reusable") else "fresh"


def install_writer_transition_lifecycle() -> None:
    """Install direct lifecycle construction on raw closure transitions."""

    from . import writer_transitions

    for name, factory in (
        (
            "_open_closure_endpoint_transition_from_obligation",
            _open_closure_endpoint_transition_from_obligation,
        ),
        (
            "_pair_closure_endpoint_transition_from_obligation",
            _pair_closure_endpoint_transition_from_obligation,
        ),
    ):
        _install_lifecycle_factory(writer_transitions, name, factory)


def _install_lifecycle_factory(writer_transitions, name: str, factory) -> None:
    current = getattr(writer_transitions, name)
    if getattr(current, _TRANSITION_LIFECYCLE_INSTALLED_MARKER, False):
        return

    setattr(factory, _TRANSITION_LIFECYCLE_INSTALLED_MARKER, True)
    setattr(writer_transitions, name, factory)


def _open_closure_endpoint_transition_from_obligation(
    prepared,
    state,
    context,
    closure_obligation,
    label: WriterClosureLabel,
    first_endpoint_choice,
    relation_evidence,
):
    from . import writer_transitions

    endpoint = WriterOpenClosureEndpoint(
        bond=closure_obligation.bond,
        first_atom=closure_obligation.first_atom,
        second_atom=closure_obligation.second_atom,
        label=label,
        first_endpoint_text=label.text,
        first_endpoint_bond_text=first_endpoint_choice.bond_text,
        first_endpoint_direction_mark=first_endpoint_choice.direction_mark,
    )

    transition = writer_transitions._transition(
        prepared,
        state,
        emitted_text=f"{first_endpoint_choice.rendered_text}{label.text}",
        successor=replace(
            state,
            ring_state=writer_transitions._ring_state_after_open_endpoint(
                state.ring_state,
                endpoint,
            ),
        ),
        kind=writer_transitions.WriterTransitionKind.OPEN_CLOSURE_ENDPOINT,
        events=(
            WriterRingLabelAllocated(
                label=endpoint.label,
                source=writer_ring_label_allocation_source(
                    source_state=state,
                    label=endpoint.label,
                ),
            ),
            WriterRingEndpointEmitted(
                bond=endpoint.bond,
                endpoint_atom=endpoint.first_atom,
                partner_atom=endpoint.second_atom,
                label=endpoint.label,
                endpoint_text=endpoint.first_endpoint_text,
                bond_text=endpoint.first_endpoint_bond_text,
                direction_mark=endpoint.first_endpoint_direction_mark,
            ),
        ),
        evidence=writer_transitions.WriterTransitionEvidence(
            bond=endpoint.bond,
            parent=endpoint.first_atom,
            child=endpoint.second_atom,
        ),
        finite_relation_work_evidence=(relation_evidence,),
    )

    if transition is None:
        return None

    successor_graph = writer_transitions._validated_closure_open_successor_graph(
        prepared,
        transition.successor,
        endpoint,
    )
    if successor_graph is None:
        return None

    if not writer_transitions._closure_open_attachment_restriction_is_exact(
        obligation=closure_obligation,
        successor_graph=successor_graph,
    ):
        return None

    capabilities = set(transition.semantic_execution_capabilities)
    if writer_transitions._closure_open_emits_coupled_attachment_capability(
        obligation=closure_obligation,
        successor_graph=successor_graph,
    ):
        capabilities.add(
            _WriterExecutionCapabilityKind
            .COUPLED_CYCLIC_ATTACHMENT_RESTRICTION
        )

    return replace(
        transition,
        semantic_execution_capabilities=frozenset(capabilities),
    )


def _pair_closure_endpoint_transition_from_obligation(
    prepared,
    state,
    context,
    pair_obligation,
):
    from . import writer_transitions

    endpoint = pair_obligation.endpoint
    closure = pair_obligation.closure
    second_endpoint_text = writer_transitions._closure_endpoint_rendered_text(
        closure.second_endpoint_bond_text,
        closure.second_endpoint_direction_mark,
    )

    transition = writer_transitions._transition(
        prepared,
        state,
        emitted_text=f"{second_endpoint_text}{endpoint.label.text}",
        successor=replace(
            state,
            ring_state=writer_transitions._ring_state_after_pair_endpoint(
                state.ring_state,
                endpoint,
                closure,
            ),
        ),
        kind=writer_transitions.WriterTransitionKind.PAIR_CLOSURE_ENDPOINT,
        events=(
            WriterRingEndpointPaired(
                bond=closure.bond,
                endpoint_atom=closure.second_atom,
                partner_atom=closure.first_atom,
                label=closure.label,
                endpoint_text=closure.second_endpoint_text,
                bond_text=closure.second_endpoint_bond_text,
                direction_mark=closure.second_endpoint_direction_mark,
                first_endpoint_bond_text=closure.first_endpoint_bond_text,
                first_endpoint_direction_mark=closure.first_endpoint_direction_mark,
            ),
            WriterRingLabelReleased(label=closure.label),
        ),
        evidence=writer_transitions.WriterTransitionEvidence(
            bond=closure.bond,
            parent=closure.first_atom,
            child=closure.second_atom,
        ),
        finite_relation_work_evidence=pair_obligation.relation_evidence,
    )

    if transition is None:
        return None

    if not writer_transitions._closure_pair_successor_is_supported(
        prepared,
        transition.successor,
        closure,
    ):
        return None

    return transition


def validate_writer_ring_lifecycle_transition(
    *,
    source_state: object,
    successor_state: object,
    events: tuple[object, ...],
) -> None:
    violations = writer_ring_lifecycle_transition_violations(
        source_state=source_state,
        successor_state=successor_state,
        events=events,
    )
    if violations:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            f"writer ring lifecycle transition violation: {violations[0]}",
        )


def writer_ring_lifecycle_transition_violations(
    *,
    source_state: object,
    successor_state: object,
    events: tuple[object, ...],
) -> tuple[str, ...]:
    indexed = tuple(enumerate(events))
    opens = _indexed(indexed, WriterRingEndpointEmitted)
    pairs = _indexed(indexed, WriterRingEndpointPaired)
    allocations = _indexed(indexed, WriterRingLabelAllocated)
    releases = _indexed(indexed, WriterRingLabelReleased)
    violations: list[str] = []

    for index, event in opens:
        prior = _matching(allocations, event.label, before=index)
        _require_one(
            violations,
            prior,
            missing_kind="missing_open_label_allocation",
            duplicate_kind="duplicate_open_label_allocation",
        )
        if len(prior) == 1:
            expected = writer_ring_label_allocation_source(
                source_state=source_state,
                label=event.label,
            )
            if prior[0].source != expected:
                violations.append("allocation_source_mismatch")
        _require_open_state(violations, source_state, successor_state, event)

    for index, event in pairs:
        following = _matching(releases, event.label, after=index)
        _require_one(
            violations,
            following,
            missing_kind="missing_paired_label_release",
            duplicate_kind="duplicate_paired_label_release",
        )
        _require_pair_state(violations, source_state, successor_state, event)

    _require_order(
        violations,
        lifecycle_events=allocations,
        transition_events=opens,
        missing_kind="allocation_without_open_endpoint",
        order_kind="allocation_after_open_endpoint",
        ordered=lambda lifecycle_index, transition_index: (
            lifecycle_index < transition_index
        ),
    )
    _require_order(
        violations,
        lifecycle_events=releases,
        transition_events=pairs,
        missing_kind="release_without_paired_endpoint",
        order_kind="release_before_paired_endpoint",
        ordered=lambda lifecycle_index, transition_index: (
            transition_index < lifecycle_index
        ),
    )
    return tuple(violations)


def _require_open_state(
    violations: list[str],
    source_state: object,
    successor_state: object,
    event: WriterRingEndpointEmitted,
) -> None:
    _require_all(
        violations,
        (
            (
                event.label not in _labels(source_state, "allocated"),
                "open_source_label_already_allocated",
            ),
            (
                _has_open_endpoint(successor_state, event),
                "successor_open_endpoint_missing",
            ),
            (
                event.label in _labels(successor_state, "allocated"),
                "successor_open_label_not_allocated",
            ),
            (
                event.label not in _labels(successor_state, "reusable"),
                "successor_open_label_still_reusable",
            ),
        ),
    )


def _require_pair_state(
    violations: list[str],
    source_state: object,
    successor_state: object,
    event: WriterRingEndpointPaired,
) -> None:
    _require_all(
        violations,
        (
            (
                _has_open_endpoint_for_pair(source_state, event),
                "source_pair_open_endpoint_missing",
            ),
            (
                event.label in _labels(source_state, "allocated"),
                "pair_source_label_not_allocated",
            ),
            (
                not _has_open_endpoint_for_bond(successor_state, event.bond),
                "successor_pair_open_endpoint_retained",
            ),
            (
                _has_closed_closure(successor_state, event),
                "successor_closed_closure_missing",
            ),
            (
                event.label not in _labels(successor_state, "allocated"),
                "successor_paired_label_still_allocated",
            ),
            (
                event.label in _labels(successor_state, "reusable"),
                "successor_paired_label_not_reusable",
            ),
        ),
    )


def _require_one(
    violations: list[str],
    matches: tuple[object, ...],
    *,
    missing_kind: str,
    duplicate_kind: str,
) -> None:
    if not matches:
        violations.append(missing_kind)
    elif len(matches) > 1:
        violations.append(duplicate_kind)


def _require_all(
    violations: list[str],
    checks: tuple[tuple[bool, str], ...],
) -> None:
    violations.extend(kind for condition, kind in checks if not condition)


def _require_order(
    violations: list[str],
    *,
    lifecycle_events,
    transition_events,
    missing_kind: str,
    order_kind: str,
    ordered,
) -> None:
    for lifecycle_index, lifecycle_event in lifecycle_events:
        positions = tuple(
            transition_index
            for transition_index, transition_event in transition_events
            if transition_event.label == lifecycle_event.label
        )
        if not positions:
            violations.append(missing_kind)
        elif not any(ordered(lifecycle_index, position) for position in positions):
            violations.append(order_kind)


def _matching(
    indexed_events,
    label: WriterClosureLabel,
    *,
    before: int | None = None,
    after: int | None = None,
) -> tuple[object, ...]:
    return tuple(
        event
        for index, event in indexed_events
        if event.label == label
        and (before is None or index < before)
        and (after is None or index > after)
    )


def _indexed(indexed_events, event_type):
    return tuple(
        (index, event)
        for index, event in indexed_events
        if isinstance(event, event_type)
    )


def _labels(state: object, label_state_field: str) -> tuple[WriterClosureLabel, ...]:
    label_state = getattr(getattr(state, "ring_state", None), "label_state", None)
    if label_state is None:
        return ()
    return tuple(getattr(label_state, label_state_field, ()))


def _open_endpoints(state: object) -> tuple[object, ...]:
    return tuple(getattr(getattr(state, "ring_state", None), "open_endpoints", ()))


def _closed_closures(state: object) -> tuple[object, ...]:
    return tuple(getattr(getattr(state, "ring_state", None), "closed_closures", ()))


def _has_open_endpoint_for_bond(state: object, bond) -> bool:
    return any(endpoint.bond == bond for endpoint in _open_endpoints(state))


def _has_open_endpoint(state: object, event: WriterRingEndpointEmitted) -> bool:
    return any(
        endpoint.bond == event.bond
        and endpoint.first_atom == event.endpoint_atom
        and endpoint.second_atom == event.partner_atom
        and endpoint.label == event.label
        and endpoint.first_endpoint_text == event.endpoint_text
        and endpoint.first_endpoint_bond_text == event.bond_text
        and endpoint.first_endpoint_direction_mark == event.direction_mark
        for endpoint in _open_endpoints(state)
    )


def _has_open_endpoint_for_pair(
    state: object,
    event: WriterRingEndpointPaired,
) -> bool:
    return any(
        endpoint.bond == event.bond
        and endpoint.first_atom == event.partner_atom
        and endpoint.second_atom == event.endpoint_atom
        and endpoint.label == event.label
        and endpoint.first_endpoint_bond_text == event.first_endpoint_bond_text
        and endpoint.first_endpoint_direction_mark == event.first_endpoint_direction_mark
        for endpoint in _open_endpoints(state)
    )


def _has_closed_closure(state: object, event: WriterRingEndpointPaired) -> bool:
    return any(
        closure.bond == event.bond
        and closure.first_atom == event.partner_atom
        and closure.second_atom == event.endpoint_atom
        and closure.label == event.label
        and closure.second_endpoint_text == event.endpoint_text
        and closure.first_endpoint_bond_text == event.first_endpoint_bond_text
        and closure.second_endpoint_bond_text == event.bond_text
        and closure.first_endpoint_direction_mark == event.first_endpoint_direction_mark
        and closure.second_endpoint_direction_mark == event.direction_mark
        for closure in _closed_closures(state)
    )


__all__ = (
    "WriterRingLabelAllocationSource",
    "install_writer_transition_lifecycle",
    "validate_writer_ring_lifecycle_transition",
    "writer_ring_label_allocation_source",
    "writer_ring_lifecycle_transition_violations",
)
