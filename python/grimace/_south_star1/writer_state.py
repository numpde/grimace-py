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
class WriterRingState:
    open_endpoints: tuple[object, ...] = ()
    active_spines: tuple[object, ...] = ()
    closed_bonds: frozenset[BondId] = frozenset()


@dataclass(frozen=True, slots=True)
class WriterStereoState:
    residual_snapshot: object | None = None


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


WriterStateKey = WriterState


def writer_state_key(state: WriterState) -> WriterStateKey:
    return state


def writer_state_from_key(key: WriterStateKey) -> WriterState:
    return key


__all__ = (
    "ComponentCursor",
    "ObligationState",
    "PendingEntryPhase",
    "PendingWriterEntry",
    "WriterAtomFrame",
    "WriterBranchFrame",
    "WriterPolicyState",
    "WriterRingState",
    "WriterState",
    "WriterStateKey",
    "WriterStereoState",
    "writer_state_from_key",
    "writer_state_key",
)
