"""Raw writer-shaped transitions for South Star 1."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from enum import Enum
from typing import TYPE_CHECKING

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .ids import AtomId
from .ids import BondId
from .writer_graph_obligations import WriterEdgeObligationKind
from .writer_graph_obligations import WriterGraphObligationContext
from .writer_graph_obligations import WriterResidualAttachmentActionKind
from .writer_graph_obligations import build_writer_graph_obligation_context
from .writer_graph_obligations import validate_writer_initial_support_graph_surface
from .writer_graph_obligations import validate_writer_snapshot_graph_surface
from .writer_graph_obligations import validate_writer_transition_graph_surface
from .writer_graph_obligations import writer_graph_completion_status
from .writer_graph_obligations import writer_residual_attachment_action_is_blocked
from .writer_graph_obligations import writer_residual_attachment_action_incidences_for_atom
from .writer_state import ComponentCursor
from .writer_state import ObligationState
from .writer_state import PendingEntryPhase
from .writer_state import PendingWriterEntry
from .writer_state import WriterAtomFrame
from .writer_state import WriterBranchFrame
from .writer_state import WriterClosedClosure
from .writer_state import WriterClosureLabel
from .writer_state import WriterOpenClosureEndpoint
from .writer_state import WriterRingLabelState
from .writer_state import WriterRingState
from .writer_state import WriterState
from .writer_state import WriterStateKey
from .writer_state import writer_state_key
from .writer_events import WriterAtomEmitted
from .writer_events import WriterBondEmitted
from .writer_events import WriterBranchClosed
from .writer_events import WriterBranchOpened
from .writer_events import WriterComponentBoundaryEmitted
from .writer_events import WriterEvent
from .writer_events import WriterLocalOrderClosed
from .writer_events import WriterRingEndpointEmitted
from .writer_events import WriterRingEndpointPaired
from .writer_stereo import WriterAtomTextChoice
from .writer_stereo import WriterBondTextChoice
from .writer_stereo import advance_writer_stereo_state
from .writer_stereo import terminal_writer_stereo_state
from .writer_stereo import validate_writer_stereo_supported_prepared
from .writer_stereo import writer_atom_text_choices
from .writer_stereo import writer_bond_text_choices

if TYPE_CHECKING:
    from .prepared_runtime import SouthStarPreparedMol


class WriterTransitionKind(Enum):
    ATOM = "atom"
    ENTER_CHILD_BOND = "enter_child_bond"
    ENTER_INLINE_CHILD = "enter_inline_child"
    OPEN_BRANCH = "open_branch"
    ENTER_BRANCH_CHILD = "enter_branch_child"
    CLOSE_BRANCH = "close_branch"
    DOT = "dot"
    OPEN_CLOSURE_ENDPOINT = "open_closure_endpoint"
    PAIR_CLOSURE_ENDPOINT = "pair_closure_endpoint"


@dataclass(frozen=True, slots=True)
class WriterTransitionEvidence:
    atom: AtomId | None = None
    bond: BondId | None = None
    parent: AtomId | None = None
    child: AtomId | None = None


@dataclass(frozen=True, slots=True)
class WriterTransition:
    emitted_text: str
    successor: WriterState
    kind: WriterTransitionKind
    events: tuple[WriterEvent, ...]
    evidence: WriterTransitionEvidence

    def __post_init__(self) -> None:
        if not self.emitted_text:
            raise ValueError("writer transitions must emit nonempty text")
        if not self.events:
            raise ValueError("writer transitions must carry semantic events")


@dataclass(frozen=True, slots=True)
class WriterTransitionExpansionContext:
    state_key: WriterStateKey
    graph: WriterGraphObligationContext


@dataclass(frozen=True, slots=True)
class _WriterChildObligation:
    bond: BondId
    child: AtomId
    attachment_id: int | None = None
    attachment_action_kind: WriterResidualAttachmentActionKind | None = None
    pending_entry: bool = False


@dataclass(frozen=True, slots=True)
class _WriterClosureOpenObligation:
    bond: BondId
    first_atom: AtomId
    second_atom: AtomId
    attachment_id: int


@dataclass(frozen=True, slots=True)
class _WriterClosurePairObligation:
    endpoint: WriterOpenClosureEndpoint
    closure: WriterClosedClosure


class _WriterScheduledActionKind(Enum):
    FINISH_ACTIVE = "finish_active"
    ENTER_INLINE_CHILD = "enter_inline_child"
    OPEN_BRANCH = "open_branch"
    OPEN_CLOSURE_ENDPOINT = "open_closure_endpoint"
    PAIR_CLOSURE_ENDPOINT = "pair_closure_endpoint"


@dataclass(frozen=True, slots=True)
class _WriterScheduledAction:
    kind: _WriterScheduledActionKind
    parent: AtomId
    child_obligation: _WriterChildObligation | None = None
    closure_open_obligation: _WriterClosureOpenObligation | None = None
    closure_open_label: WriterClosureLabel | None = None
    closure_pair_obligation: _WriterClosurePairObligation | None = None


def build_writer_transition_expansion_context(
    prepared: SouthStarPreparedMol,
    state: WriterState,
) -> WriterTransitionExpansionContext:
    if state.active is None:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "writer state requires an active writer frame",
        )
    validate_writer_transition_prepared(prepared)
    key = writer_state_key(state)
    graph = build_writer_graph_obligation_context(prepared, key)
    return WriterTransitionExpansionContext(state_key=key, graph=graph)


def validate_writer_transition_prepared(prepared: SouthStarPreparedMol) -> None:
    validate_writer_transition_graph_surface(prepared)
    validate_writer_stereo_supported_prepared(prepared)


def _open_branch_transition_from_child_obligation(
    prepared: SouthStarPreparedMol,
    state: WriterState,
    parent: AtomId,
    child_obligation: _WriterChildObligation,
) -> WriterTransition | None:
    return _transition(
        prepared,
        state,
        emitted_text="(",
        successor=replace(
            state,
            obligations=ObligationState(
                pending_entry=PendingWriterEntry(
                    parent=parent,
                    child=child_obligation.child,
                    bond=child_obligation.bond,
                    branch=True,
                )
            ),
        ),
        kind=WriterTransitionKind.OPEN_BRANCH,
        events=(
            WriterBranchOpened(
                parent=parent,
                child=child_obligation.child,
                bond=child_obligation.bond,
            ),
        ),
        evidence=WriterTransitionEvidence(
            bond=child_obligation.bond,
            parent=parent,
            child=child_obligation.child,
        ),
    )


def _enter_inline_child_transitions_from_child_obligation(
    prepared: SouthStarPreparedMol,
    state: WriterState,
    parent: AtomId,
    child_obligation: _WriterChildObligation,
    context: WriterTransitionExpansionContext,
) -> tuple[WriterTransition, ...]:
    return _enter_child_transitions(
        prepared,
        state,
        PendingWriterEntry(
            parent=parent,
            child=child_obligation.child,
            bond=child_obligation.bond,
            branch=False,
        ),
        kind=WriterTransitionKind.ENTER_INLINE_CHILD,
        context=context,
    )


def _active_child_scheduled_actions(
    parent: AtomId,
    children: tuple[_WriterChildObligation, ...],
) -> tuple[_WriterScheduledAction, ...]:
    if not children:
        return (
            _WriterScheduledAction(
                kind=_WriterScheduledActionKind.FINISH_ACTIVE,
                parent=parent,
            ),
        )

    if len(children) == 1:
        return (
            _WriterScheduledAction(
                kind=_WriterScheduledActionKind.ENTER_INLINE_CHILD,
                parent=parent,
                child_obligation=children[0],
            ),
        )

    return tuple(
        _WriterScheduledAction(
            kind=_WriterScheduledActionKind.OPEN_BRANCH,
            parent=parent,
            child_obligation=child_obligation,
        )
        for child_obligation in children
    )


def _active_child_transitions_from_scheduled_action(
    prepared: SouthStarPreparedMol,
    state: WriterState,
    context: WriterTransitionExpansionContext,
    action: _WriterScheduledAction,
) -> tuple[WriterTransition, ...]:
    if action.kind is _WriterScheduledActionKind.FINISH_ACTIVE:
        return _finish_active_transitions(prepared, state, context)

    child_obligation = action.child_obligation

    if child_obligation is None:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "scheduled child action requires a child obligation",
        )

    if action.kind is _WriterScheduledActionKind.ENTER_INLINE_CHILD:
        return _enter_inline_child_transitions_from_child_obligation(
            prepared,
            state,
            action.parent,
            child_obligation,
            context,
        )

    if action.kind is _WriterScheduledActionKind.OPEN_BRANCH:
        transition = _open_branch_transition_from_child_obligation(
            prepared,
            state,
            action.parent,
            child_obligation,
        )

        if transition is None:
            return ()

        return (transition,)

    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"unsupported scheduled child action: {action.kind!r}",
    )


def _active_child_transitions_from_obligations(
    prepared: SouthStarPreparedMol,
    state: WriterState,
    context: WriterTransitionExpansionContext,
    parent: AtomId,
    children: tuple[_WriterChildObligation, ...],
) -> tuple[WriterTransition, ...]:
    transitions: list[WriterTransition] = []

    for action in _active_child_scheduled_actions(parent, children):
        transitions.extend(
            _transitions_from_scheduled_action(
                prepared,
                state,
                context,
                action,
            )
        )

    return tuple(transitions)


def _active_emitted_transitions(
    prepared: SouthStarPreparedMol,
    state: WriterState,
    context: WriterTransitionExpansionContext,
    active_atom: AtomId,
) -> tuple[WriterTransition, ...]:
    closure_actions = _closure_endpoint_scheduled_actions(
        prepared,
        state,
        context,
    )

    closure_transitions = _transitions_from_scheduled_actions(
        prepared,
        state,
        context,
        closure_actions,
    )

    if closure_transitions:
        return closure_transitions

    children = _child_obligations_from_context(
        context,
        state,
        active_atom,
    )

    return _transitions_from_scheduled_actions(
        prepared,
        state,
        context,
        _active_child_scheduled_actions(
            active_atom,
            children,
        ),
    )


def _scheduled_writer_transitions(
    prepared: SouthStarPreparedMol,
    state: WriterState,
    context: WriterTransitionExpansionContext,
) -> tuple[WriterTransition, ...]:
    if state.obligations.pending_entry is not None:
        return _pending_entry_transitions(prepared, state, context)

    active = state.active

    if not active.atom_emitted:
        return _root_atom_transitions(prepared, state, active)

    return _active_emitted_transitions(
        prepared,
        state,
        context,
        active.atom,
    )


def legal_writer_transitions(
    prepared: SouthStarPreparedMol,
    state: WriterState,
) -> tuple[WriterTransition, ...]:
    context = build_writer_transition_expansion_context(prepared, state)

    return _scheduled_writer_transitions(
        prepared,
        state,
        context,
    )


def _root_atom_transitions(
    prepared: SouthStarPreparedMol,
    state: WriterState,
    active: WriterAtomFrame,
) -> tuple[WriterTransition, ...]:
    transitions = []
    for choice in writer_atom_text_choices(prepared, active.atom):
        transition = _transition(
            prepared,
            state,
            emitted_text=choice.text,
            successor=replace(
                state,
                active=replace(active, atom_emitted=True),
                visited_atoms=frozenset((*state.visited_atoms, active.atom)),
            ),
            kind=WriterTransitionKind.ATOM,
            events=(
                WriterAtomEmitted(
                    atom=active.atom,
                    text=choice.text,
                    tetra_token=choice.tetra_token,
                ),
            ),
            evidence=WriterTransitionEvidence(atom=active.atom),
        )
        if transition is not None:
            transitions.append(transition)
    return tuple(transitions)


def _pending_entry_transitions(
    prepared: SouthStarPreparedMol,
    state: WriterState,
    context: WriterTransitionExpansionContext,
) -> tuple[WriterTransition, ...]:
    pending = state.obligations.pending_entry
    if pending is None:
        return ()
    kind = (
        WriterTransitionKind.ENTER_BRANCH_CHILD
        if pending.branch
        else WriterTransitionKind.ENTER_INLINE_CHILD
    )
    if pending.phase is PendingEntryPhase.NEEDS_ATOM_AFTER_BOND:
        return _enter_child_atom_transitions(
            prepared,
            state,
            pending,
            kind=kind,
            context=context,
        )
    return _enter_child_transitions(
        prepared,
        state,
        pending,
        kind=kind,
        context=context,
    )


def _enter_child_transitions(
    prepared: SouthStarPreparedMol,
    state: WriterState,
    pending: PendingWriterEntry,
    *,
    kind: WriterTransitionKind,
    context: WriterTransitionExpansionContext,
) -> tuple[WriterTransition, ...]:
    parent_frame = state.active
    child_frame = WriterAtomFrame(
        atom=pending.child,
        parent=pending.parent,
        incoming_bond=pending.bond,
        atom_emitted=True,
    )
    branch_stack = state.branch_stack
    if pending.branch:
        branch_stack = (*branch_stack, WriterBranchFrame(return_atom=parent_frame))

    transitions: list[WriterTransition] = []
    for bond_choice in writer_bond_text_choices(prepared, pending.bond):
        if bond_choice.text:
            transition = _transition(
                prepared,
                state,
                emitted_text=bond_choice.text,
                successor=replace(
                    state,
                    obligations=ObligationState(
                        pending_entry=replace(
                            pending,
                            phase=PendingEntryPhase.NEEDS_ATOM_AFTER_BOND,
                        )
                    ),
                ),
                kind=WriterTransitionKind.ENTER_CHILD_BOND,
                events=(
                    WriterBondEmitted(
                        bond=pending.bond,
                        parent=pending.parent,
                        child=pending.child,
                        text=bond_choice.text,
                        direction_mark=bond_choice.direction_mark,
                    ),
                ),
                evidence=WriterTransitionEvidence(
                    bond=pending.bond,
                    parent=pending.parent,
                    child=pending.child,
                ),
            )
            if transition is not None:
                transitions.append(transition)
            continue
        for atom_choice in writer_atom_text_choices(prepared, pending.child):
            transition = _enter_child_atom_transition(
                prepared,
                state,
                pending,
                atom_choice=atom_choice,
                bond_choice=bond_choice,
                child_frame=child_frame,
                branch_stack=branch_stack,
                kind=kind,
                context=context,
            )
            if transition is not None:
                transitions.append(transition)
    return tuple(transitions)


def _enter_child_atom_transitions(
    prepared: SouthStarPreparedMol,
    state: WriterState,
    pending: PendingWriterEntry,
    *,
    kind: WriterTransitionKind,
    context: WriterTransitionExpansionContext,
) -> tuple[WriterTransition, ...]:
    parent_frame = state.active
    child_frame = WriterAtomFrame(
        atom=pending.child,
        parent=pending.parent,
        incoming_bond=pending.bond,
        atom_emitted=True,
    )
    branch_stack = state.branch_stack
    if pending.branch:
        branch_stack = (*branch_stack, WriterBranchFrame(return_atom=parent_frame))
    transitions = []
    for atom_choice in writer_atom_text_choices(prepared, pending.child):
        transition = _enter_child_atom_transition(
            prepared,
            state,
            pending,
            atom_choice=atom_choice,
            bond_choice=None,
            child_frame=child_frame,
            branch_stack=branch_stack,
            kind=kind,
            context=context,
        )
        if transition is not None:
            transitions.append(transition)
    return tuple(transitions)


def _enter_child_atom_transition(
    prepared: SouthStarPreparedMol,
    state: WriterState,
    pending: PendingWriterEntry,
    *,
    atom_choice: WriterAtomTextChoice,
    bond_choice: WriterBondTextChoice | None,
    child_frame: WriterAtomFrame,
    branch_stack: tuple[WriterBranchFrame, ...],
    kind: WriterTransitionKind,
    context: WriterTransitionExpansionContext,
) -> WriterTransition | None:
    events: list[WriterEvent] = []
    if bond_choice is not None:
        events.append(
            WriterBondEmitted(
                bond=pending.bond,
                parent=pending.parent,
                child=pending.child,
                text=bond_choice.text,
                direction_mark=bond_choice.direction_mark,
            )
        )
    events.append(
        WriterAtomEmitted(
            atom=pending.child,
            text=atom_choice.text,
            tetra_token=atom_choice.tetra_token,
            parent=pending.parent,
            incoming_bond=pending.bond,
        )
    )
    if not pending.branch and _is_final_child_for_parent(context, state, pending):
        events.append(WriterLocalOrderClosed(atom=pending.parent))
    return _transition(
        prepared,
        state,
        emitted_text=atom_choice.text,
        successor=replace(
            state,
            active=child_frame,
            branch_stack=branch_stack,
            visited_atoms=frozenset((*state.visited_atoms, pending.child)),
            written_bonds=frozenset((*state.written_bonds, pending.bond)),
            obligations=ObligationState(),
        ),
        kind=kind,
        events=tuple(events),
        evidence=WriterTransitionEvidence(
            atom=pending.child,
            bond=pending.bond,
            parent=pending.parent,
            child=pending.child,
        ),
    )


def _finish_active_transitions(
    prepared: SouthStarPreparedMol,
    state: WriterState,
    context: WriterTransitionExpansionContext,
) -> tuple[WriterTransition, ...]:
    active_atom = state.active.atom
    if state.branch_stack:
        frame = state.branch_stack[-1]
        transition = _transition(
            prepared,
            state,
            emitted_text=")",
            successor=replace(
                state,
                active=frame.return_atom,
                branch_stack=state.branch_stack[:-1],
            ),
            kind=WriterTransitionKind.CLOSE_BRANCH,
            events=(
                WriterLocalOrderClosed(atom=active_atom),
                WriterBranchClosed(atom=active_atom),
            ),
            evidence=WriterTransitionEvidence(atom=active_atom),
        )
        return (() if transition is None else (transition,))

    next_component_index = state.component_cursor.component_index + 1
    if next_component_index >= len(state.component_cursor.component_roots):
        return ()
    completion = writer_graph_completion_status(prepared, context.state_key, context.graph)
    if not completion.complete:
        return ()
    root = state.component_cursor.component_roots[next_component_index]
    transition = _transition(
        prepared,
        state,
        emitted_text=".",
        successor=replace(
            state,
            component_cursor=ComponentCursor(
                component_index=next_component_index,
                component_roots=state.component_cursor.component_roots,
            ),
            active=WriterAtomFrame(
                atom=root,
                parent=None,
                incoming_bond=None,
                atom_emitted=False,
            ),
        ),
        kind=WriterTransitionKind.DOT,
        events=(
            WriterLocalOrderClosed(atom=active_atom),
            WriterComponentBoundaryEmitted(next_root=root),
        ),
        evidence=WriterTransitionEvidence(atom=root),
    )
    return (() if transition is None else (transition,))


def _closure_endpoint_scheduled_actions(
    prepared: SouthStarPreparedMol,
    state: WriterState,
    context: WriterTransitionExpansionContext,
) -> tuple[_WriterScheduledAction, ...]:
    active_atom = state.active.atom
    actions: list[_WriterScheduledAction] = []

    actions.extend(
        _closure_pair_scheduled_actions(
            state,
            active_atom,
        )
    )

    labels = _available_closure_labels_for_open(
        prepared,
        state.ring_state,
    )

    if labels:
        actions.extend(
            _closure_open_scheduled_actions(
                context,
                active_atom,
                labels,
            )
        )

    return tuple(actions)


def _closure_endpoint_transitions_from_scheduled_action(
    prepared: SouthStarPreparedMol,
    state: WriterState,
    context: WriterTransitionExpansionContext,
    action: _WriterScheduledAction,
) -> tuple[WriterTransition, ...]:
    if action.kind is _WriterScheduledActionKind.PAIR_CLOSURE_ENDPOINT:
        return _closure_pair_transitions_from_scheduled_action(
            prepared,
            state,
            context,
            action,
        )

    if action.kind is _WriterScheduledActionKind.OPEN_CLOSURE_ENDPOINT:
        return _closure_open_transitions_from_scheduled_action(
            prepared,
            state,
            context,
            action,
        )

    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"unsupported closure-endpoint scheduled action: {action.kind!r}",
    )


def _transitions_from_scheduled_action(
    prepared: SouthStarPreparedMol,
    state: WriterState,
    context: WriterTransitionExpansionContext,
    action: _WriterScheduledAction,
) -> tuple[WriterTransition, ...]:
    if action.kind in (
        _WriterScheduledActionKind.FINISH_ACTIVE,
        _WriterScheduledActionKind.ENTER_INLINE_CHILD,
        _WriterScheduledActionKind.OPEN_BRANCH,
    ):
        return _active_child_transitions_from_scheduled_action(
            prepared,
            state,
            context,
            action,
        )

    if action.kind in (
        _WriterScheduledActionKind.OPEN_CLOSURE_ENDPOINT,
        _WriterScheduledActionKind.PAIR_CLOSURE_ENDPOINT,
    ):
        return _closure_endpoint_transitions_from_scheduled_action(
            prepared,
            state,
            context,
            action,
        )

    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"unsupported scheduled action: {action.kind!r}",
    )


def _transitions_from_scheduled_actions(
    prepared: SouthStarPreparedMol,
    state: WriterState,
    context: WriterTransitionExpansionContext,
    actions: tuple[_WriterScheduledAction, ...],
) -> tuple[WriterTransition, ...]:
    transitions: list[WriterTransition] = []

    for action in actions:
        transitions.extend(
            _transitions_from_scheduled_action(
                prepared,
                state,
                context,
                action,
            )
        )

    return tuple(transitions)


def _closure_endpoint_transitions(
    prepared: SouthStarPreparedMol,
    state: WriterState,
    context: WriterTransitionExpansionContext,
) -> tuple[WriterTransition, ...]:
    transitions: list[WriterTransition] = []

    for action in _closure_endpoint_scheduled_actions(
        prepared,
        state,
        context,
    ):
        transitions.extend(
            _transitions_from_scheduled_action(
                prepared,
                state,
                context,
                action,
            )
        )

    return tuple(transitions)


def _open_closure_endpoint_transition_from_obligation(
    prepared: SouthStarPreparedMol,
    state: WriterState,
    context: WriterTransitionExpansionContext,
    closure_obligation: _WriterClosureOpenObligation,
    label: WriterClosureLabel,
) -> WriterTransition | None:
    endpoint = WriterOpenClosureEndpoint(
        bond=closure_obligation.bond,
        first_atom=closure_obligation.first_atom,
        second_atom=closure_obligation.second_atom,
        label=label,
        first_endpoint_text=label.text,
        first_endpoint_bond_text="",
    )

    transition = _transition(
        prepared,
        state,
        emitted_text=label.text,
        successor=replace(
            state,
            ring_state=_ring_state_after_open_endpoint(
                state.ring_state,
                endpoint,
            ),
        ),
        kind=WriterTransitionKind.OPEN_CLOSURE_ENDPOINT,
        events=(
            WriterRingEndpointEmitted(
                bond=endpoint.bond,
                endpoint_atom=endpoint.first_atom,
                partner_atom=endpoint.second_atom,
                label=endpoint.label,
                endpoint_text=endpoint.first_endpoint_text,
                bond_text=endpoint.first_endpoint_bond_text,
            ),
        ),
        evidence=WriterTransitionEvidence(
            bond=endpoint.bond,
            parent=endpoint.first_atom,
            child=endpoint.second_atom,
        ),
    )

    if transition is None:
        return None

    if not _closure_open_successor_is_supported(
        prepared,
        transition.successor,
        endpoint,
    ):
        return None

    return transition


def _closure_open_transitions_from_scheduled_action(
    prepared: SouthStarPreparedMol,
    state: WriterState,
    context: WriterTransitionExpansionContext,
    action: _WriterScheduledAction,
) -> tuple[WriterTransition, ...]:
    if action.kind is not _WriterScheduledActionKind.OPEN_CLOSURE_ENDPOINT:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            f"unsupported closure-open scheduled action: {action.kind!r}",
        )

    closure_obligation = action.closure_open_obligation

    if closure_obligation is None:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "scheduled closure-open action requires an open obligation",
        )

    label = action.closure_open_label

    if label is None:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "scheduled closure-open action requires a closure label",
        )

    transition = _open_closure_endpoint_transition_from_obligation(
        prepared,
        state,
        context,
        closure_obligation,
        label,
    )

    if transition is None:
        return ()

    return (transition,)


def _closure_open_obligations_from_context(
    context: WriterTransitionExpansionContext,
    atom: AtomId,
) -> tuple[_WriterClosureOpenObligation, ...]:
    obligations: list[_WriterClosureOpenObligation] = []

    for action_incidence in writer_residual_attachment_action_incidences_for_atom(
        context.graph.residual_summary,
        atom,
    ):
        action = action_incidence.action

        if action.kind is not WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY:
            continue

        incidence = action_incidence.incidence

        obligations.append(
            _WriterClosureOpenObligation(
                bond=incidence.bond,
                first_atom=atom,
                second_atom=incidence.residual_atom,
                attachment_id=action.attachment_id,
            )
        )

    return tuple(obligations)


def _closure_open_scheduled_actions(
    context: WriterTransitionExpansionContext,
    active_atom: AtomId,
    labels: tuple[WriterClosureLabel, ...],
) -> tuple[_WriterScheduledAction, ...]:
    actions: list[_WriterScheduledAction] = []

    for closure_obligation in _closure_open_obligations_from_context(
        context,
        active_atom,
    ):
        for label in labels:
            actions.append(
                _WriterScheduledAction(
                    kind=_WriterScheduledActionKind.OPEN_CLOSURE_ENDPOINT,
                    parent=active_atom,
                    closure_open_obligation=closure_obligation,
                    closure_open_label=label,
                )
            )

    return tuple(actions)


def _closure_pair_obligations_from_state(
    state: WriterState,
    atom: AtomId,
) -> tuple[_WriterClosurePairObligation, ...]:
    obligations: list[_WriterClosurePairObligation] = []

    for endpoint in state.ring_state.open_endpoints:
        if endpoint.second_atom != atom:
            continue

        closure = WriterClosedClosure(
            bond=endpoint.bond,
            first_atom=endpoint.first_atom,
            second_atom=endpoint.second_atom,
            label=endpoint.label,
            first_endpoint_text=endpoint.first_endpoint_text,
            second_endpoint_text=endpoint.label.text,
            first_endpoint_bond_text=endpoint.first_endpoint_bond_text,
            second_endpoint_bond_text="",
        )

        obligations.append(
            _WriterClosurePairObligation(
                endpoint=endpoint,
                closure=closure,
            )
        )

    return tuple(obligations)


def _closure_pair_scheduled_actions(
    state: WriterState,
    active_atom: AtomId,
) -> tuple[_WriterScheduledAction, ...]:
    return tuple(
        _WriterScheduledAction(
            kind=_WriterScheduledActionKind.PAIR_CLOSURE_ENDPOINT,
            parent=active_atom,
            closure_pair_obligation=pair_obligation,
        )
        for pair_obligation in _closure_pair_obligations_from_state(
            state,
            active_atom,
        )
    )


def _pair_closure_endpoint_transition_from_obligation(
    prepared: SouthStarPreparedMol,
    state: WriterState,
    context: WriterTransitionExpansionContext,
    pair_obligation: _WriterClosurePairObligation,
) -> WriterTransition | None:
    endpoint = pair_obligation.endpoint
    closure = pair_obligation.closure

    transition = _transition(
        prepared,
        state,
        emitted_text=endpoint.label.text,
        successor=replace(
            state,
            ring_state=_ring_state_after_pair_endpoint(
                state.ring_state,
                endpoint,
                closure,
            ),
        ),
        kind=WriterTransitionKind.PAIR_CLOSURE_ENDPOINT,
        events=(
            WriterRingEndpointPaired(
                bond=closure.bond,
                endpoint_atom=closure.second_atom,
                partner_atom=closure.first_atom,
                label=closure.label,
                endpoint_text=closure.second_endpoint_text,
                bond_text=closure.second_endpoint_bond_text,
            ),
        ),
        evidence=WriterTransitionEvidence(
            bond=closure.bond,
            parent=closure.first_atom,
            child=closure.second_atom,
        ),
    )

    if transition is None:
        return None

    if not _closure_pair_successor_is_supported(
        prepared,
        transition.successor,
        closure,
    ):
        return None

    return transition


def _closure_pair_transitions_from_scheduled_action(
    prepared: SouthStarPreparedMol,
    state: WriterState,
    context: WriterTransitionExpansionContext,
    action: _WriterScheduledAction,
) -> tuple[WriterTransition, ...]:
    if action.kind is not _WriterScheduledActionKind.PAIR_CLOSURE_ENDPOINT:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            f"unsupported closure-pair scheduled action: {action.kind!r}",
        )

    pair_obligation = action.closure_pair_obligation

    if pair_obligation is None:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "scheduled closure-pair action requires a pair obligation",
        )

    transition = _pair_closure_endpoint_transition_from_obligation(
        prepared,
        state,
        context,
        pair_obligation,
    )

    if transition is None:
        return ()

    return (transition,)


def _available_closure_labels_for_open(
    prepared: SouthStarPreparedMol,
    ring_state: WriterRingState,
) -> tuple[WriterClosureLabel, ...]:
    allocated = set(ring_state.label_state.allocated)
    available = []
    for label in prepared.policy.ring_labels:
        candidate = WriterClosureLabel(value=label.value, text=label.text())
        if candidate not in allocated:
            available.append(candidate)
    if prepared.policy.least_free_ring_labels:
        if not available:
            return ()
        return (min(available, key=lambda label: label.value),)
    return tuple(available)


def _ring_state_after_open_endpoint(
    ring_state: WriterRingState,
    endpoint: WriterOpenClosureEndpoint,
) -> WriterRingState:
    label = endpoint.label
    return replace(
        ring_state,
        open_endpoints=_sorted_open_endpoints((*ring_state.open_endpoints, endpoint)),
        label_state=WriterRingLabelState(
            allocated=_sorted_closure_labels((*ring_state.label_state.allocated, label)),
            reusable=_sorted_closure_labels(
                item for item in ring_state.label_state.reusable if item != label
            ),
        ),
    )


def _ring_state_after_pair_endpoint(
    ring_state: WriterRingState,
    endpoint: WriterOpenClosureEndpoint,
    closure: WriterClosedClosure,
) -> WriterRingState:
    label = endpoint.label
    return replace(
        ring_state,
        open_endpoints=_sorted_open_endpoints(
            item for item in ring_state.open_endpoints if item != endpoint
        ),
        closed_closures=_sorted_closed_closures((*ring_state.closed_closures, closure)),
        label_state=WriterRingLabelState(
            allocated=_sorted_closure_labels(
                item for item in ring_state.label_state.allocated if item != label
            ),
            reusable=_sorted_closure_labels((*ring_state.label_state.reusable, label)),
        ),
    )


def _closure_open_successor_is_supported(
    prepared: SouthStarPreparedMol,
    state: WriterState,
    endpoint: WriterOpenClosureEndpoint,
) -> bool:
    try:
        key = writer_state_key(state)
        graph = build_writer_graph_obligation_context(prepared, key)
        if _edge_kind(graph, endpoint.bond) is not WriterEdgeObligationKind.OPEN_CLOSURE_ENDPOINT:
            return False
        if _has_closure_candidate(graph):
            return False
        if _has_blocked_attachment_action(graph):
            return False
        validate_writer_snapshot_graph_surface(prepared, key, graph)
    except SouthStarError:
        return False
    return True


def _closure_pair_successor_is_supported(
    prepared: SouthStarPreparedMol,
    state: WriterState,
    closure: WriterClosedClosure,
) -> bool:
    try:
        key = writer_state_key(state)
        graph = build_writer_graph_obligation_context(prepared, key)
        if _edge_kind(graph, closure.bond) is not WriterEdgeObligationKind.CLOSED_CLOSURE:
            return False
        if any(endpoint.bond == closure.bond for endpoint in key.ring_state.open_endpoints):
            return False
        validate_writer_snapshot_graph_surface(prepared, key, graph)
    except SouthStarError:
        return False
    return True


def _edge_kind(
    graph: WriterGraphObligationContext,
    bond: BondId,
) -> WriterEdgeObligationKind | None:
    for obligation in graph.edge_partition.obligations:
        if obligation.bond == bond:
            return obligation.kind
    return None


def _has_closure_candidate(graph: WriterGraphObligationContext) -> bool:
    return any(
        obligation.kind is WriterEdgeObligationKind.CLOSURE_CANDIDATE
        for obligation in graph.edge_partition.obligations
    )


def _has_blocked_attachment_action(graph: WriterGraphObligationContext) -> bool:
    return any(
        writer_residual_attachment_action_is_blocked(action)
        for action in graph.residual_summary.attachment_actions
    )


def _sorted_closure_labels(labels) -> tuple[WriterClosureLabel, ...]:
    return tuple(sorted(labels, key=lambda item: (item.value, item.text)))


def _sorted_open_endpoints(endpoints) -> tuple[WriterOpenClosureEndpoint, ...]:
    return tuple(
        sorted(
            endpoints,
            key=lambda item: (
                int(item.bond),
                int(item.first_atom),
                int(item.second_atom),
                item.label.value,
                item.label.text,
            ),
        )
    )


def _sorted_closed_closures(closures) -> tuple[WriterClosedClosure, ...]:
    return tuple(
        sorted(
            closures,
            key=lambda item: (
                int(item.bond),
                int(item.first_atom),
                int(item.second_atom),
                item.label.value,
                item.label.text,
            ),
        )
    )


def writer_state_is_eos(
    prepared: SouthStarPreparedMol,
    state: WriterState,
) -> bool:
    return finalize_writer_terminal_state(prepared, state) is not None


def finalize_writer_terminal_state(
    prepared: SouthStarPreparedMol,
    state: WriterState,
) -> WriterState | None:
    context = build_writer_transition_expansion_context(prepared, state)
    if state.obligations.pending_entry is not None or state.branch_stack:
        return None
    if state.ring_state.open_endpoints:
        return None
    if not state.active.atom_emitted:
        return None
    completion = writer_graph_completion_status(prepared, context.state_key, context.graph)
    if not completion.complete:
        return None
    if _child_obligations_from_context(context, state, state.active.atom):
        return None
    if state.component_cursor.component_index + 1 < len(
        state.component_cursor.component_roots
    ):
        return None
    stereo_state = terminal_writer_stereo_state(
        prepared,
        state.stereo_state,
        state.active.atom,
    )
    if stereo_state is None:
        return None
    return replace(state, stereo_state=stereo_state)


def _transition(
    prepared: SouthStarPreparedMol,
    state: WriterState,
    *,
    emitted_text: str,
    successor: WriterState,
    kind: WriterTransitionKind,
    events: tuple[WriterEvent, ...],
    evidence: WriterTransitionEvidence,
) -> WriterTransition | None:
    stereo_state = advance_writer_stereo_state(
        prepared,
        state.stereo_state,
        events,
    )
    if stereo_state is None:
        return None
    return WriterTransition(
        emitted_text=emitted_text,
        successor=replace(successor, stereo_state=stereo_state),
        kind=kind,
        events=events,
        evidence=evidence,
    )


def validate_writer_supported_prepared(prepared: SouthStarPreparedMol) -> None:
    validate_writer_initial_support_graph_surface(prepared)
    validate_writer_stereo_supported_prepared(prepared)


def _child_obligations_from_context(
    context: WriterTransitionExpansionContext,
    state: WriterState,
    atom: AtomId,
) -> tuple[_WriterChildObligation, ...]:
    partition = context.graph.edge_partition

    if any(
        obligation.kind is WriterEdgeObligationKind.CLOSURE_CANDIDATE
        for obligation in partition.obligations
    ):
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_POLICY,
            "WRITER_SHAPED closure-candidate edge obligations are not supported yet",
        )

    summary = context.graph.residual_summary
    action_incidences_for_atom = writer_residual_attachment_action_incidences_for_atom(
        summary,
        atom,
    )

    children: list[_WriterChildObligation] = []

    for action in summary.attachment_actions:
        if action.kind not in (
            WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY,
            WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY,
        ):
            continue

        boundary = tuple(
            action_incidence.incidence
            for action_incidence in action_incidences_for_atom
            if action_incidence.action is action
        )

        if not boundary:
            continue

        if len(boundary) != 1:
            raise SouthStarError(
                SouthStarErrorKind.UNSUPPORTED_POLICY,
                "WRITER_SHAPED multi-incidence residual attachments are not supported yet",
            )

        incidence = boundary[0]
        children.append(
            _WriterChildObligation(
                bond=incidence.bond,
                child=incidence.residual_atom,
                attachment_id=action.attachment_id,
                attachment_action_kind=action.kind,
            )
        )

    pending = state.obligations.pending_entry

    if pending is not None and pending.parent == atom:
        children.append(
            _WriterChildObligation(
                bond=pending.bond,
                child=pending.child,
                pending_entry=True,
            )
        )

    return tuple(
        sorted(
            children,
            key=lambda item: (int(item.bond), int(atom), int(item.child)),
        )
    )


def _is_final_child_for_parent(
    context: WriterTransitionExpansionContext,
    state: WriterState,
    pending: PendingWriterEntry,
) -> bool:
    children = _child_obligations_from_context(context, state, pending.parent)
    return (
        len(children) == 1
        and children[0].bond == pending.bond
        and children[0].child == pending.child
    )


__all__ = (
    "WriterTransition",
    "WriterTransitionEvidence",
    "WriterTransitionExpansionContext",
    "WriterTransitionKind",
    "build_writer_transition_expansion_context",
    "finalize_writer_terminal_state",
    "legal_writer_transitions",
    "validate_writer_supported_prepared",
    "validate_writer_transition_prepared",
    "writer_state_is_eos",
)
