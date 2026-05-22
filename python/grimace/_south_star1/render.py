"""Pure rendering boundary for satisfying South Star 1 assignments."""

from __future__ import annotations

from .constraints import TraversalAssignment
from .constraints import validate_nonstereo_tree_witness
from .constraints import validate_nonstereo_traversal_witness
from .constraints import validate_stereo_traversal_witness
from .facts import MoleculeFacts
from .ids import AtomId
from .ids import BondSlotId
from .policy import DirectionMark
from .policy import SmilesPolicy
from .semantics import ParserSemantics
from .skeleton import ChildRole
from .skeleton import RingEvent
from .skeleton import TraversalSkeleton
from .slots import BondSlot
from .slots import CarrierSlot
from .slots import SlotBundle
from .slots import carrier_slot_by_bond_slot
from .slots import ring_bond_slot_by_endpoint
from .slots import ring_endpoint_by_id
from .slots import tree_bond_slot_by_bond


def render_nonstereo_tree(
    facts: MoleculeFacts,
    skeleton: TraversalSkeleton,
    slots: SlotBundle,
    assignment: TraversalAssignment,
) -> str:
    validate_nonstereo_tree_witness(facts, skeleton, slots, assignment)
    tree_slot_by_bond = tree_bond_slot_by_bond(slots)

    def render_atom(atom: AtomId) -> str:
        text = assignment.atom_text[atom].render(assignment.tetra_tokens[atom])
        for event in skeleton.events_at[atom]:
            slot = tree_slot_by_bond[event.bond]
            bond_text = assignment.bond_text[slot.id].base_text
            child_text = bond_text + render_atom(event.child)
            if event.role is ChildRole.BRANCH:
                text += f"({child_text})"
            else:
                text += child_text
        return text

    return ".".join(render_atom(root) for root in skeleton.roots)


def render_nonstereo_traversal(
    facts: MoleculeFacts,
    skeleton: TraversalSkeleton,
    slots: SlotBundle,
    assignment: TraversalAssignment,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
) -> str:
    validate_nonstereo_traversal_witness(
        facts,
        skeleton,
        slots,
        assignment,
        policy,
        semantics,
    )
    tree_slot_by_bond = tree_bond_slot_by_bond(slots)
    ring_slot_by_endpoint = ring_bond_slot_by_endpoint(slots)
    endpoint_by_id = ring_endpoint_by_id(slots)

    def render_atom(atom: AtomId) -> str:
        text = assignment.atom_text[atom].render(assignment.tetra_tokens[atom])
        for event in skeleton.events_at[atom]:
            if isinstance(event, RingEvent):
                slot = ring_slot_by_endpoint[(event.bond, event.atom)]
                if slot.ring_endpoint is None:
                    raise ValueError(f"ring slot lacks endpoint id: {slot.id!r}")
                endpoint = endpoint_by_id[slot.ring_endpoint]
                text += (
                    assignment.bond_text[slot.id].base_text
                    + assignment.ring_labels[endpoint.id].text()
                )
                continue

            slot = tree_slot_by_bond[event.bond]
            bond_text = assignment.bond_text[slot.id].base_text
            child_text = bond_text + render_atom(event.child)
            if event.role is ChildRole.BRANCH:
                text += f"({child_text})"
            else:
                text += child_text
        return text

    return ".".join(render_atom(root) for root in skeleton.roots)


def render_stereo_traversal(
    facts: MoleculeFacts,
    skeleton: TraversalSkeleton,
    slots: SlotBundle,
    assignment: TraversalAssignment,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
) -> str:
    validate_stereo_traversal_witness(
        facts,
        skeleton,
        slots,
        assignment,
        policy,
        semantics,
    )
    tree_slot_by_bond = tree_bond_slot_by_bond(slots)
    ring_slot_by_endpoint = ring_bond_slot_by_endpoint(slots)
    endpoint_by_id = ring_endpoint_by_id(slots)
    carrier_by_slot = carrier_slot_by_bond_slot(slots)

    def render_atom(atom: AtomId) -> str:
        text = assignment.atom_text[atom].render(assignment.tetra_tokens[atom])
        for event in skeleton.events_at[atom]:
            if isinstance(event, RingEvent):
                slot = ring_slot_by_endpoint[(event.bond, event.atom)]
                if slot.ring_endpoint is None:
                    raise ValueError(f"ring slot lacks endpoint id: {slot.id!r}")
                endpoint = endpoint_by_id[slot.ring_endpoint]
                text += (
                    render_bond_slot(slot, assignment, carrier_by_slot)
                    + assignment.ring_labels[endpoint.id].text()
                )
                continue

            slot = tree_slot_by_bond[event.bond]
            child_text = (
                render_bond_slot(slot, assignment, carrier_by_slot)
                + render_atom(event.child)
            )
            if event.role is ChildRole.BRANCH:
                text += f"({child_text})"
            else:
                text += child_text
        return text

    return ".".join(render_atom(root) for root in skeleton.roots)


def render_bond_slot(
    slot: BondSlot,
    assignment: TraversalAssignment,
    carrier_by_bond_slot: dict[BondSlotId, CarrierSlot],
) -> str:
    bond_text = assignment.bond_text[slot.id]
    carrier = carrier_by_bond_slot[slot.id]
    mark = assignment.direction_marks[carrier.id]

    if mark is DirectionMark.ABSENT:
        return bond_text.base_text

    if not bond_text.permits_direction:
        raise ValueError(f"direction mark not permitted on bond slot {slot.id!r}")

    if mark is DirectionMark.FWD:
        return "/"

    if mark is DirectionMark.REV:
        return "\\"

    raise ValueError(mark)


__all__ = (
    "render_bond_slot",
    "render_nonstereo_traversal",
    "render_nonstereo_tree",
    "render_stereo_traversal",
)
