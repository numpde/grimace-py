"""Closure-transition factories with direct ring-label lifecycle evidence."""

from __future__ import annotations

from dataclasses import replace

from .writer_capabilities import _WriterExecutionCapabilityKind
from .writer_events import WriterRingEndpointEmitted
from .writer_events import WriterRingEndpointPaired
from .writer_events import WriterRingLabelAllocated
from .writer_events import WriterRingLabelReleased
from .writer_ring_lifecycle import writer_ring_label_allocation_source
from .writer_state import WriterClosureLabel
from .writer_state import WriterOpenClosureEndpoint


_TRANSITION_LIFECYCLE_INSTALLED_MARKER = "_ring_label_lifecycle_installed"


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


__all__ = ("install_writer_transition_lifecycle",)
