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
from .skeleton import RingEvent
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
    syntax_position: int
    ring_endpoint: RingEndpointId | None = None


@dataclass(frozen=True, slots=True)
class RingEndpointSlot:
    id: RingEndpointId
    bond: BondId
    atom: AtomId
    other_atom: AtomId
    bond_slot: BondSlotId
    syntax_position: int


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
    ring_endpoints: tuple[RingEndpointSlot, ...] = ()


def allocate_tree_slots(
    facts: MoleculeFacts,
    skeleton: TraversalSkeleton,
) -> SlotBundle:
    if skeleton.ring_bonds:
        raise NotImplementedError("tree slot allocation does not yet support rings")
    return allocate_traversal_slots(facts, skeleton)


def allocate_traversal_slots(
    facts: MoleculeFacts,
    skeleton: TraversalSkeleton,
) -> SlotBundle:
    facts.validate()

    atom_slots = tuple(
        AtomSlot(id=AtomSlotId(i), atom=atom.id)
        for i, atom in enumerate(facts.atoms)
    )

    bond_slots: list[BondSlot] = []
    seen_tree_bonds: set[BondId] = set()
    ring_endpoint_slots: list[RingEndpointSlot] = []

    for syntax_position, event in enumerate(_iter_syntax_events(skeleton)):
        if isinstance(event, ChildEvent):
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
                    syntax_position=syntax_position,
                )
            )
            continue

        if isinstance(event, RingEvent):
            endpoint_id = RingEndpointId(len(ring_endpoint_slots))
            bond_slot_id = BondSlotId(len(bond_slots))
            bond_slots.append(
                BondSlot(
                    id=bond_slot_id,
                    bond=event.bond,
                    kind=BondSlotKind.RING_ENDPOINT,
                    written_from=event.atom,
                    written_to=event.other_atom,
                    syntax_position=syntax_position,
                    ring_endpoint=endpoint_id,
                )
            )
            ring_endpoint_slots.append(
                RingEndpointSlot(
                    id=endpoint_id,
                    bond=event.bond,
                    atom=event.atom,
                    other_atom=event.other_atom,
                    bond_slot=bond_slot_id,
                    syntax_position=syntax_position,
                )
            )
            continue

        raise TypeError(event)

    if seen_tree_bonds != skeleton.tree_bonds:
        missing = skeleton.tree_bonds - seen_tree_bonds
        extra = seen_tree_bonds - skeleton.tree_bonds
        raise ValueError(
            "tree bond slot coverage mismatch: "
            f"missing={missing!r}, extra={extra!r}"
        )

    ring_endpoint_bonds: dict[BondId, int] = {}
    for endpoint in ring_endpoint_slots:
        ring_endpoint_bonds[endpoint.bond] = (
            ring_endpoint_bonds.get(endpoint.bond, 0) + 1
        )
    bad_ring_bonds = {
        bond
        for bond in skeleton.ring_bonds
        if ring_endpoint_bonds.get(bond) != 2
    }
    if bad_ring_bonds:
        raise ValueError(f"ring bonds must have two endpoint slots: {bad_ring_bonds!r}")

    return SlotBundle(
        atom_slots=atom_slots,
        bond_slots=tuple(bond_slots),
        ring_endpoints=tuple(ring_endpoint_slots),
    )


def atom_slot_by_atom(slots: SlotBundle) -> dict[AtomId, AtomSlot]:
    return {slot.atom: slot for slot in slots.atom_slots}


def tree_bond_slot_by_bond(slots: SlotBundle) -> dict[BondId, BondSlot]:
    return {
        slot.bond: slot
        for slot in slots.bond_slots
        if slot.kind is BondSlotKind.TREE
    }


def ring_bond_slots_by_bond(
    slots: SlotBundle,
) -> dict[BondId, tuple[BondSlot, BondSlot]]:
    by_bond: dict[BondId, list[BondSlot]] = {}
    for slot in slots.bond_slots:
        if slot.kind is BondSlotKind.RING_ENDPOINT:
            by_bond.setdefault(slot.bond, []).append(slot)
    return {
        bond: (endpoints[0], endpoints[1])
        for bond, endpoints in by_bond.items()
        if len(endpoints) == 2
    }


def ring_bond_slot_by_endpoint(
    slots: SlotBundle,
) -> dict[tuple[BondId, AtomId], BondSlot]:
    return {
        (slot.bond, slot.written_from): slot
        for slot in slots.bond_slots
        if slot.kind is BondSlotKind.RING_ENDPOINT
    }


def ring_endpoint_by_id(slots: SlotBundle) -> dict[RingEndpointId, RingEndpointSlot]:
    return {endpoint.id: endpoint for endpoint in slots.ring_endpoints}


def _iter_syntax_events(skeleton: TraversalSkeleton):
    def walk(atom: AtomId):
        for event in skeleton.events_at[atom]:
            yield event
            if isinstance(event, ChildEvent):
                yield from walk(event.child)

    for root in skeleton.roots:
        yield from walk(root)


__all__ = (
    "AtomSlot",
    "BondSlot",
    "BondSlotKind",
    "CarrierSlot",
    "RingEndpointSlot",
    "SlotBundle",
    "allocate_tree_slots",
    "allocate_traversal_slots",
    "atom_slot_by_atom",
    "ring_bond_slot_by_endpoint",
    "ring_bond_slots_by_bond",
    "ring_endpoint_by_id",
    "tree_bond_slot_by_bond",
)
