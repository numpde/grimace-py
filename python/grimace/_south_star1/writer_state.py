"""Writer-shaped state records for the South Star 1 writer kernel."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .ids import AtomId
from .ids import BondId


@dataclass(frozen=True, slots=True)
class ComponentCursor:
    component_index: int
    component_roots: tuple[AtomId, ...]


@dataclass(frozen=True, slots=True)
class WriterAtomFrame:
    atom: AtomId
    parent: AtomId | None
    incoming_bond: BondId | None
    atom_emitted: bool


@dataclass(frozen=True, slots=True)
class WriterBranchFrame:
    return_atom: WriterAtomFrame


class PendingEntryPhase(Enum):
    NEEDS_BOND_OR_ATOM = "needs_bond_or_atom"
    NEEDS_ATOM_AFTER_BOND = "needs_atom_after_bond"


@dataclass(frozen=True, slots=True)
class PendingWriterEntry:
    parent: AtomId
    child: AtomId
    bond: BondId
    branch: bool
    phase: PendingEntryPhase = PendingEntryPhase.NEEDS_BOND_OR_ATOM


@dataclass(frozen=True, slots=True)
class ObligationState:
    pending_entry: PendingWriterEntry | None = None


@dataclass(frozen=True, slots=True)
class WriterRingEndpoint:
    pass


@dataclass(frozen=True, slots=True)
class WriterRingSpineFrame:
    pass


@dataclass(frozen=True, slots=True)
class WriterRingState:
    open_endpoints: tuple[WriterRingEndpoint, ...] = ()
    active_spines: tuple[WriterRingSpineFrame, ...] = ()
    closed_bonds: frozenset[BondId] = frozenset()


@dataclass(frozen=True, slots=True)
class WriterStereoResidualKey:
    pass


EMPTY_WRITER_STEREO_KEY = WriterStereoResidualKey()


@dataclass(frozen=True, slots=True)
class WriterStereoState:
    residual_key: WriterStereoResidualKey = EMPTY_WRITER_STEREO_KEY


@dataclass(frozen=True, slots=True)
class WriterPolicyState:
    atom_text: tuple[tuple[AtomId, str], ...] = ()
    bond_text: tuple[tuple[BondId, str], ...] = ()


@dataclass(frozen=True, slots=True)
class WriterState:
    component_cursor: ComponentCursor
    active: WriterAtomFrame | None
    branch_stack: tuple[WriterBranchFrame, ...]
    visited_atoms: frozenset[AtomId]
    written_bonds: frozenset[BondId]
    obligations: ObligationState
    ring_state: WriterRingState
    stereo_state: WriterStereoState
    policy_state: WriterPolicyState


@dataclass(frozen=True, slots=True)
class ObligationStateKey:
    pending_entry: PendingWriterEntry | None = None


@dataclass(frozen=True, slots=True)
class WriterRingStateKey:
    open_endpoints: tuple[WriterRingEndpoint, ...] = ()
    active_spines: tuple[WriterRingSpineFrame, ...] = ()
    closed_bonds: frozenset[BondId] = frozenset()


@dataclass(frozen=True, slots=True)
class WriterStereoStateKey:
    residual_key: WriterStereoResidualKey = EMPTY_WRITER_STEREO_KEY


@dataclass(frozen=True, slots=True)
class WriterPolicyStateKey:
    atom_text: tuple[tuple[AtomId, str], ...] = ()
    bond_text: tuple[tuple[BondId, str], ...] = ()


@dataclass(frozen=True, slots=True)
class WriterStateKey:
    component_cursor: ComponentCursor
    active: WriterAtomFrame | None
    branch_stack: tuple[WriterBranchFrame, ...]
    visited_atoms: frozenset[AtomId]
    written_bonds: frozenset[BondId]
    obligations: ObligationStateKey
    ring_state: WriterRingStateKey
    stereo_state: WriterStereoStateKey
    policy_state: WriterPolicyStateKey


def writer_state_key(state: WriterState) -> WriterStateKey:
    return WriterStateKey(
        component_cursor=state.component_cursor,
        active=state.active,
        branch_stack=state.branch_stack,
        visited_atoms=state.visited_atoms,
        written_bonds=state.written_bonds,
        obligations=ObligationStateKey(
            pending_entry=state.obligations.pending_entry,
        ),
        ring_state=WriterRingStateKey(
            open_endpoints=state.ring_state.open_endpoints,
            active_spines=state.ring_state.active_spines,
            closed_bonds=state.ring_state.closed_bonds,
        ),
        stereo_state=WriterStereoStateKey(
            residual_key=state.stereo_state.residual_key,
        ),
        policy_state=WriterPolicyStateKey(
            atom_text=state.policy_state.atom_text,
            bond_text=state.policy_state.bond_text,
        ),
    )


def writer_state_from_key(key: WriterStateKey) -> WriterState:
    return WriterState(
        component_cursor=key.component_cursor,
        active=key.active,
        branch_stack=key.branch_stack,
        visited_atoms=key.visited_atoms,
        written_bonds=key.written_bonds,
        obligations=ObligationState(
            pending_entry=key.obligations.pending_entry,
        ),
        ring_state=WriterRingState(
            open_endpoints=key.ring_state.open_endpoints,
            active_spines=key.ring_state.active_spines,
            closed_bonds=key.ring_state.closed_bonds,
        ),
        stereo_state=WriterStereoState(
            residual_key=key.stereo_state.residual_key,
        ),
        policy_state=WriterPolicyState(
            atom_text=key.policy_state.atom_text,
            bond_text=key.policy_state.bond_text,
        ),
    )


__all__ = (
    "ComponentCursor",
    "EMPTY_WRITER_STEREO_KEY",
    "ObligationState",
    "ObligationStateKey",
    "PendingEntryPhase",
    "PendingWriterEntry",
    "WriterAtomFrame",
    "WriterBranchFrame",
    "WriterPolicyStateKey",
    "WriterPolicyState",
    "WriterRingEndpoint",
    "WriterRingSpineFrame",
    "WriterRingStateKey",
    "WriterRingState",
    "WriterStereoResidualKey",
    "WriterStereoStateKey",
    "WriterState",
    "WriterStateKey",
    "WriterStereoState",
    "writer_state_from_key",
    "writer_state_key",
)
