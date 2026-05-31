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
from .writer_graph_obligations import build_writer_graph_obligation_context
from .writer_graph_obligations import validate_writer_initial_support_graph_surface
from .writer_graph_obligations import validate_writer_snapshot_graph_surface
from .writer_state import ComponentCursor
from .writer_state import ObligationState
from .writer_state import PendingEntryPhase
from .writer_state import PendingWriterEntry
from .writer_state import WriterAtomFrame
from .writer_state import WriterBranchFrame
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


def build_writer_transition_expansion_context(
    prepared: SouthStarPreparedMol,
    state: WriterState,
) -> WriterTransitionExpansionContext:
    if state.active is None:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "writer state requires an active writer frame",
        )
    if (
        not state.ring_state.open_endpoints
        and not state.ring_state.closed_closures
    ):
        validate_writer_initial_support_graph_surface(prepared)
    key = writer_state_key(state)
    graph = build_writer_graph_obligation_context(prepared, key)
    validate_writer_stereo_supported_prepared(prepared)
    validate_writer_snapshot_graph_surface(prepared, key, graph)
    return WriterTransitionExpansionContext(state_key=key, graph=graph)


def legal_writer_transitions(
    prepared: SouthStarPreparedMol,
    state: WriterState,
) -> tuple[WriterTransition, ...]:
    context = build_writer_transition_expansion_context(prepared, state)
    if state.obligations.pending_entry is not None:
        return _pending_entry_transitions(prepared, state, context)

    active = state.active
    if not active.atom_emitted:
        return _root_atom_transitions(prepared, state, active)

    children = _child_obligations_from_context(context, state, active.atom)
    if not children:
        return _finish_active_transitions(prepared, state, context)
    if len(children) == 1:
        bond, child = children[0]
        return _enter_child_transitions(
            prepared,
            state,
            PendingWriterEntry(
                parent=active.atom,
                child=child,
                bond=bond,
                branch=False,
            ),
            kind=WriterTransitionKind.ENTER_INLINE_CHILD,
            context=context,
        )

    transitions = []
    for bond, child in children:
        transition = _transition(
            prepared,
            state,
            emitted_text="(",
            successor=replace(
                state,
                obligations=ObligationState(
                    pending_entry=PendingWriterEntry(
                        parent=active.atom,
                        child=child,
                        bond=bond,
                        branch=True,
                    )
                ),
            ),
            kind=WriterTransitionKind.OPEN_BRANCH,
            events=(
                WriterBranchOpened(
                    parent=active.atom,
                    child=child,
                    bond=bond,
                ),
            ),
            evidence=WriterTransitionEvidence(
                bond=bond,
                parent=active.atom,
                child=child,
            ),
        )
        if transition is not None:
            transitions.append(transition)
    return tuple(transitions)


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
    if not state.active.atom_emitted:
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
) -> tuple[tuple[BondId, AtomId], ...]:
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
    if summary.has_cyclic_attachment:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_POLICY,
            "WRITER_SHAPED cyclic residual attachments are not supported yet",
        )
    children = []
    for attachment in summary.attachments.attachments:
        boundary = tuple(
            incidence
            for incidence in attachment.boundary
            if incidence.written_atom == atom
        )
        if not boundary:
            continue
        if len(boundary) != 1:
            raise SouthStarError(
                SouthStarErrorKind.UNSUPPORTED_POLICY,
                "WRITER_SHAPED multi-incidence residual attachments are not supported yet",
            )
        incidence = boundary[0]
        children.append((incidence.bond, incidence.residual_atom))
    pending = state.obligations.pending_entry
    if pending is not None and pending.parent == atom:
        children.append((pending.bond, pending.child))
    return tuple(
        sorted(
            children,
            key=lambda item: (int(item[0]), int(atom), int(item[1])),
        )
    )


def _is_final_child_for_parent(
    context: WriterTransitionExpansionContext,
    state: WriterState,
    pending: PendingWriterEntry,
) -> bool:
    children = _child_obligations_from_context(context, state, pending.parent)
    return children == ((pending.bond, pending.child),)


__all__ = (
    "WriterTransition",
    "WriterTransitionEvidence",
    "WriterTransitionExpansionContext",
    "WriterTransitionKind",
    "build_writer_transition_expansion_context",
    "finalize_writer_terminal_state",
    "legal_writer_transitions",
    "validate_writer_supported_prepared",
    "writer_state_is_eos",
)
