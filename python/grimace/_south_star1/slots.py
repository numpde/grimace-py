"""Typed syntax-slot records for the private proof kernel."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .facts import MoleculeFacts
from .ids import AtomId
from .ids import AtomSlotId
from .ids import BondId
from .ids import BondSlotId
from .ids import CarrierSlotId
from .ids import RingEndpointId
from .skeleton import ChildEvent
from .skeleton import TraversalSkeleton


class BondSlotKind(Enum):
    TREE = "tree"
    RING_ENDPOINT = "ring_endpoint"


@dataclass(frozen=True, slots=True)
class AtomSlot:
    id: AtomSlotId
    atom: AtomId


@dataclass(frozen=True, slots=True)
class BondSlot:
    id: BondSlotId
    bond: BondId
    kind: BondSlotKind
    written_from: AtomId
    written_to: AtomId | None
    ring_endpoint: RingEndpointId | None = None


@dataclass(frozen=True, slots=True)
class CarrierSlot:
    id: CarrierSlotId
    bond_slot: BondSlotId
    bond: BondId
    written_from: AtomId
    written_to: AtomId | None


@dataclass(frozen=True, slots=True)
class SlotBundle:
    atom_slots: tuple[AtomSlot, ...]
    bond_slots: tuple[BondSlot, ...]
    carrier_slots: tuple[CarrierSlot, ...] = ()
    ring_endpoints: tuple[RingEndpointId, ...] = ()


def allocate_tree_slots(
    facts: MoleculeFacts,
    skeleton: TraversalSkeleton,
) -> SlotBundle:
    facts.validate()
    if skeleton.ring_bonds:
        raise NotImplementedError("tree slot allocation does not yet support rings")

    atom_slots = tuple(
        AtomSlot(id=AtomSlotId(i), atom=atom.id)
        for i, atom in enumerate(facts.atoms)
    )

    bond_slots: list[BondSlot] = []
    seen_tree_bonds: set[BondId] = set()
    for atom in facts.atoms:
        for event in skeleton.events_at[atom.id]:
            if not isinstance(event, ChildEvent):
                raise TypeError(event)
            if event.bond in seen_tree_bonds:
                raise ValueError(f"tree bond has multiple syntax slots: {event.bond!r}")
            seen_tree_bonds.add(event.bond)
            bond_slots.append(
                BondSlot(
                    id=BondSlotId(len(bond_slots)),
                    bond=event.bond,
                    kind=BondSlotKind.TREE,
                    written_from=event.parent,
                    written_to=event.child,
                )
            )

    if seen_tree_bonds != skeleton.tree_bonds:
        missing = skeleton.tree_bonds - seen_tree_bonds
        extra = seen_tree_bonds - skeleton.tree_bonds
        raise ValueError(
            "tree bond slot coverage mismatch: "
            f"missing={missing!r}, extra={extra!r}"
        )

    return SlotBundle(atom_slots=atom_slots, bond_slots=tuple(bond_slots))


def atom_slot_by_atom(slots: SlotBundle) -> dict[AtomId, AtomSlot]:
    return {slot.atom: slot for slot in slots.atom_slots}


def tree_bond_slot_by_bond(slots: SlotBundle) -> dict[BondId, BondSlot]:
    return {
        slot.bond: slot
        for slot in slots.bond_slots
        if slot.kind is BondSlotKind.TREE
    }


__all__ = (
    "AtomSlot",
    "BondSlot",
    "BondSlotKind",
    "CarrierSlot",
    "SlotBundle",
    "allocate_tree_slots",
    "atom_slot_by_atom",
    "tree_bond_slot_by_bond",
)
