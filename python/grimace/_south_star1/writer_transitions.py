"""Raw writer-shaped transitions for South Star 1."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from enum import Enum
from typing import TYPE_CHECKING

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .facts import MoleculeFacts
from .ids import AtomId
from .ids import BondId
from .policy import TetraToken
from .writer_state import ComponentCursor
from .writer_state import ObligationState
from .writer_state import PendingEntryPhase
from .writer_state import PendingWriterEntry
from .writer_state import WriterAtomFrame
from .writer_state import WriterBranchFrame
from .writer_state import WriterState

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
    evidence: WriterTransitionEvidence

    def __post_init__(self) -> None:
        if not self.emitted_text:
            raise ValueError("writer transitions must emit nonempty text")


def legal_writer_transitions(
    prepared: SouthStarPreparedMol,
    state: WriterState,
) -> tuple[WriterTransition, ...]:
    if state.obligations.pending_entry is not None:
        return _pending_entry_transitions(prepared, state)

    active = state.active
    if active is None:
        return ()
    if not active.atom_emitted:
        return _root_atom_transitions(prepared, state, active)

    children = _child_obligations(prepared, state, active.atom)
    if not children:
        return _finish_active_transitions(state)
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
        )

    return tuple(
        WriterTransition(
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
            evidence=WriterTransitionEvidence(
                bond=bond,
                parent=active.atom,
                child=child,
            ),
        )
        for bond, child in children
    )


def _root_atom_transitions(
    prepared: SouthStarPreparedMol,
    state: WriterState,
    active: WriterAtomFrame,
) -> tuple[WriterTransition, ...]:
    return tuple(
        WriterTransition(
            emitted_text=text,
            successor=replace(
                state,
                active=replace(active, atom_emitted=True),
                visited_atoms=frozenset((*state.visited_atoms, active.atom)),
            ),
            kind=WriterTransitionKind.ATOM,
            evidence=WriterTransitionEvidence(atom=active.atom),
        )
        for text in _atom_texts(prepared, active.atom)
    )


def _pending_entry_transitions(
    prepared: SouthStarPreparedMol,
    state: WriterState,
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
        return _enter_child_atom_transitions(prepared, state, pending, kind=kind)
    return _enter_child_transitions(prepared, state, pending, kind=kind)


def _enter_child_transitions(
    prepared: SouthStarPreparedMol,
    state: WriterState,
    pending: PendingWriterEntry,
    *,
    kind: WriterTransitionKind,
) -> tuple[WriterTransition, ...]:
    parent_frame = state.active
    if parent_frame is None:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "writer child transition requires an active parent frame",
        )
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
    for bond_text in _tree_bond_texts(prepared, pending.bond):
        if bond_text:
            transitions.append(
                WriterTransition(
                    emitted_text=bond_text,
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
                    evidence=WriterTransitionEvidence(
                        bond=pending.bond,
                        parent=pending.parent,
                        child=pending.child,
                    ),
                )
            )
            continue
        for atom_text in _atom_texts(prepared, pending.child):
            transitions.append(
                _enter_child_atom_transition(
                    state,
                    pending,
                    atom_text=atom_text,
                    child_frame=child_frame,
                    branch_stack=branch_stack,
                    kind=kind,
                )
            )
    return tuple(transitions)


def _enter_child_atom_transitions(
    prepared: SouthStarPreparedMol,
    state: WriterState,
    pending: PendingWriterEntry,
    *,
    kind: WriterTransitionKind,
) -> tuple[WriterTransition, ...]:
    parent_frame = state.active
    if parent_frame is None:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            "writer child atom transition requires an active parent frame",
        )
    child_frame = WriterAtomFrame(
        atom=pending.child,
        parent=pending.parent,
        incoming_bond=pending.bond,
        atom_emitted=True,
    )
    branch_stack = state.branch_stack
    if pending.branch:
        branch_stack = (*branch_stack, WriterBranchFrame(return_atom=parent_frame))
    return tuple(
        _enter_child_atom_transition(
            state,
            pending,
            atom_text=atom_text,
            child_frame=child_frame,
            branch_stack=branch_stack,
            kind=kind,
        )
        for atom_text in _atom_texts(prepared, pending.child)
    )


def _enter_child_atom_transition(
    state: WriterState,
    pending: PendingWriterEntry,
    *,
    atom_text: str,
    child_frame: WriterAtomFrame,
    branch_stack: tuple[WriterBranchFrame, ...],
    kind: WriterTransitionKind,
) -> WriterTransition:
    return WriterTransition(
        emitted_text=atom_text,
        successor=replace(
            state,
            active=child_frame,
            branch_stack=branch_stack,
            visited_atoms=frozenset((*state.visited_atoms, pending.child)),
            written_bonds=frozenset((*state.written_bonds, pending.bond)),
            obligations=ObligationState(),
        ),
        kind=kind,
        evidence=WriterTransitionEvidence(
            atom=pending.child,
            bond=pending.bond,
            parent=pending.parent,
            child=pending.child,
        ),
    )


def _finish_active_transitions(state: WriterState) -> tuple[WriterTransition, ...]:
    if state.branch_stack:
        frame = state.branch_stack[-1]
        return (
            WriterTransition(
                emitted_text=")",
                successor=replace(
                    state,
                    active=frame.return_atom,
                    branch_stack=state.branch_stack[:-1],
                ),
                kind=WriterTransitionKind.CLOSE_BRANCH,
                evidence=WriterTransitionEvidence(atom=state.active.atom if state.active else None),
            ),
        )

    next_component_index = state.component_cursor.component_index + 1
    if next_component_index >= len(state.component_cursor.component_roots):
        return ()
    root = state.component_cursor.component_roots[next_component_index]
    return (
        WriterTransition(
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
            evidence=WriterTransitionEvidence(atom=root),
        ),
    )


def writer_state_is_eos(
    prepared: SouthStarPreparedMol,
    state: WriterState,
) -> bool:
    if state.active is None:
        return True
    if state.obligations.pending_entry is not None or state.branch_stack:
        return False
    if not state.active.atom_emitted:
        return False
    return (
        not _child_obligations(prepared, state, state.active.atom)
        and state.component_cursor.component_index + 1
        >= len(state.component_cursor.component_roots)
    )


def validate_writer_supported_prepared(prepared: SouthStarPreparedMol) -> None:
    _reject_stereo(prepared.facts)
    _reject_non_tree_components(prepared)


def _reject_stereo(facts: MoleculeFacts) -> None:
    if facts.stereo.tetrahedral or facts.stereo.directional or facts.ligand_occurrences:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_STEREO,
            "WRITER_SHAPED writer-state MVP does not support stereo facts yet",
        )


def _reject_non_tree_components(prepared: SouthStarPreparedMol) -> None:
    graph = prepared.graph_index
    for component in prepared.facts.components:
        component_atoms = frozenset(component.atoms)
        component_bonds = frozenset(component.bonds)
        if not component_atoms or len(component_bonds) != len(component_atoms) - 1:
            raise SouthStarError(
                SouthStarErrorKind.UNSUPPORTED_POLICY,
                "WRITER_SHAPED writer-state MVP supports tree components only",
            )

        start = component.atoms[0]
        seen = {start}
        stack = [start]
        while stack:
            atom = stack.pop()
            for neighbor in graph.neighbors[atom]:
                if neighbor not in component_atoms:
                    continue
                bond = graph.bond_between[(min(atom, neighbor), max(atom, neighbor))]
                if bond not in component_bonds or neighbor in seen:
                    continue
                seen.add(neighbor)
                stack.append(neighbor)

        if frozenset(seen) != component_atoms:
            raise SouthStarError(
                SouthStarErrorKind.UNSUPPORTED_POLICY,
                "WRITER_SHAPED writer-state MVP supports connected tree components only",
            )


def _child_obligations(
    prepared: SouthStarPreparedMol,
    state: WriterState,
    atom: AtomId,
) -> tuple[tuple[BondId, AtomId], ...]:
    return _child_obligations_from_facts(state, atom, prepared)


def _child_obligations_from_facts(
    state: WriterState,
    atom: AtomId,
    prepared: SouthStarPreparedMol,
) -> tuple[tuple[BondId, AtomId], ...]:
    graph = prepared.graph_index
    children = []
    for neighbor in graph.neighbors[atom]:
        bond = graph.bond_between[(min(atom, neighbor), max(atom, neighbor))]
        if bond in state.written_bonds or neighbor in state.visited_atoms:
            continue
        children.append((bond, neighbor))
    return tuple(children)


def _atom_texts(prepared: SouthStarPreparedMol, atom: AtomId) -> tuple[str, ...]:
    texts = []
    for choice in prepared.policy.atom_text_domain_unchecked(atom):
        if choice.permits(TetraToken.NONE):
            texts.append(choice.render(TetraToken.NONE))
    if not texts:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_POLICY,
            f"WRITER_SHAPED has no non-stereo atom text for {atom!r}",
        )
    return tuple(texts)


def _tree_bond_texts(prepared: SouthStarPreparedMol, bond: BondId) -> tuple[str, ...]:
    return tuple(
        choice.base_text
        for choice in prepared.policy.bond_text_domain_unchecked(
            bond,
            slot_kind="tree",
        )
    )


__all__ = (
    "WriterTransition",
    "WriterTransitionEvidence",
    "WriterTransitionKind",
    "legal_writer_transitions",
    "validate_writer_supported_prepared",
    "writer_state_is_eos",
)
