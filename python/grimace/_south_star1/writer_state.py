"""Writer-shaped state records for the South Star 1 writer kernel."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .ids import AtomId
from .ids import BondId
from .residual_constraints import ResidualStoreValueSnapshot
from .writer_stereo import EMPTY_RESIDUAL_SNAPSHOT
from .writer_stereo import WriterAtomOccurrenceRecord
from .writer_stereo import WriterBondOccurrenceRecord
from .writer_stereo import WriterDelayedStereoFactor
from .writer_stereo import WriterLocalOrderRecord
from .writer_stereo import writer_stereo_state_sort_tuple


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
class WriterStereoState:
    residual_snapshot: ResidualStoreValueSnapshot = EMPTY_RESIDUAL_SNAPSHOT
    atom_occurrences: tuple[WriterAtomOccurrenceRecord, ...] = ()
    bond_occurrences: tuple[WriterBondOccurrenceRecord, ...] = ()
    local_orders: tuple[WriterLocalOrderRecord, ...] = ()
    delayed_factors: tuple[WriterDelayedStereoFactor, ...] = ()


@dataclass(frozen=True, slots=True)
class WriterPolicyState:
    atom_text: tuple[tuple[AtomId, str], ...] = ()
    bond_text: tuple[tuple[BondId, str], ...] = ()


@dataclass(frozen=True, slots=True)
class WriterState:
    component_cursor: ComponentCursor
    active: WriterAtomFrame
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
    residual_snapshot: ResidualStoreValueSnapshot = EMPTY_RESIDUAL_SNAPSHOT
    atom_occurrences: tuple[WriterAtomOccurrenceRecord, ...] = ()
    bond_occurrences: tuple[WriterBondOccurrenceRecord, ...] = ()
    local_orders: tuple[WriterLocalOrderRecord, ...] = ()
    delayed_factors: tuple[WriterDelayedStereoFactor, ...] = ()


@dataclass(frozen=True, slots=True)
class WriterPolicyStateKey:
    atom_text: tuple[tuple[AtomId, str], ...] = ()
    bond_text: tuple[tuple[BondId, str], ...] = ()


@dataclass(frozen=True, slots=True)
class WriterStateKey:
    component_cursor: ComponentCursor
    active: WriterAtomFrame
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
            residual_snapshot=state.stereo_state.residual_snapshot,
            atom_occurrences=state.stereo_state.atom_occurrences,
            bond_occurrences=state.stereo_state.bond_occurrences,
            local_orders=state.stereo_state.local_orders,
            delayed_factors=state.stereo_state.delayed_factors,
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
            residual_snapshot=key.stereo_state.residual_snapshot,
            atom_occurrences=key.stereo_state.atom_occurrences,
            bond_occurrences=key.stereo_state.bond_occurrences,
            local_orders=key.stereo_state.local_orders,
            delayed_factors=key.stereo_state.delayed_factors,
        ),
        policy_state=WriterPolicyState(
            atom_text=key.policy_state.atom_text,
            bond_text=key.policy_state.bond_text,
        ),
    )


def writer_state_key_sort_tuple(key: WriterStateKey) -> tuple[object, ...]:
    if key.ring_state.open_endpoints or key.ring_state.active_spines:
        raise AssertionError(
            "writer_state_key_sort_tuple must be extended before nonempty ring state"
        )
    return (
        int(key.component_cursor.component_index),
        tuple(int(atom) for atom in key.component_cursor.component_roots),
        _atom_frame_sort_tuple(key.active),
        tuple(_branch_frame_sort_tuple(frame) for frame in key.branch_stack),
        tuple(sorted(int(atom) for atom in key.visited_atoms)),
        tuple(sorted(int(bond) for bond in key.written_bonds)),
        _obligation_sort_tuple(key.obligations),
        (
            tuple(sorted(int(bond) for bond in key.ring_state.closed_bonds)),
        ),
        writer_stereo_state_sort_tuple(
            WriterStereoState(
                residual_snapshot=key.stereo_state.residual_snapshot,
                atom_occurrences=key.stereo_state.atom_occurrences,
                bond_occurrences=key.stereo_state.bond_occurrences,
                local_orders=key.stereo_state.local_orders,
                delayed_factors=key.stereo_state.delayed_factors,
            )
        ),
        (
            tuple((int(atom), text) for atom, text in key.policy_state.atom_text),
            tuple((int(bond), text) for bond, text in key.policy_state.bond_text),
        ),
    )


def _atom_frame_sort_tuple(frame: WriterAtomFrame) -> tuple[object, ...]:
    return (
        "atom",
        int(frame.atom),
        None if frame.parent is None else int(frame.parent),
        None if frame.incoming_bond is None else int(frame.incoming_bond),
        bool(frame.atom_emitted),
    )


def _branch_frame_sort_tuple(frame: WriterBranchFrame) -> tuple[object, ...]:
    return (_atom_frame_sort_tuple(frame.return_atom),)


def _obligation_sort_tuple(obligations: ObligationStateKey) -> tuple[object, ...]:
    pending = obligations.pending_entry
    if pending is None:
        return ("no_pending",)
    return (
        "pending",
        int(pending.parent),
        int(pending.child),
        int(pending.bond),
        bool(pending.branch),
        pending.phase.value,
    )


__all__ = (
    "ComponentCursor",
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
    "WriterStereoStateKey",
    "WriterState",
    "WriterStateKey",
    "WriterStereoState",
    "writer_state_from_key",
    "writer_state_key",
    "writer_state_key_sort_tuple",
)
