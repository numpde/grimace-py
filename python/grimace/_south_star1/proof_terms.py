"""Stable proof-term keys and node constructors for South Star certificates."""

from __future__ import annotations

from collections.abc import Iterable
from hashlib import blake2b

from .constraints import TraversalAssignment
from .enumeration_trace import EnumerationNodeId
from .skeleton import ChildEvent
from .skeleton import RingEvent
from .skeleton import TraversalSkeleton
from .slots import SlotBundle
from .stereo_csp import PresentationPrefix
from .stereo_csp import StereoSolution


def skeleton_key(skeleton: TraversalSkeleton) -> tuple[object, ...]:
    return (
        tuple(int(root) for root in skeleton.roots),
        tuple(
            sorted(
                (
                    int(atom),
                    None if parent is None else int(parent),
                )
                for atom, parent in skeleton.parent.items()
            )
        ),
        tuple(sorted(int(bond) for bond in skeleton.tree_bonds)),
        tuple(sorted(int(bond) for bond in skeleton.ring_bonds)),
        tuple(
            sorted(
                (
                    int(atom),
                    tuple(_event_key(event) for event in events),
                )
                for atom, events in skeleton.events_at.items()
            )
        ),
    )


def slot_key(slots: SlotBundle) -> tuple[object, ...]:
    return (
        tuple(
            (int(slot.id), int(slot.atom))
            for slot in slots.atom_slots
        ),
        tuple(
            (
                int(slot.id),
                int(slot.bond),
                slot.kind.value,
                int(slot.written_from),
                None if slot.written_to is None else int(slot.written_to),
                slot.syntax_position,
                None if slot.ring_endpoint is None else int(slot.ring_endpoint),
            )
            for slot in slots.bond_slots
        ),
        tuple(
            (
                int(endpoint.id),
                int(endpoint.bond),
                int(endpoint.atom),
                int(endpoint.other_atom),
                int(endpoint.bond_slot),
                endpoint.syntax_position,
            )
            for endpoint in slots.ring_endpoints
        ),
        tuple(
            (
                int(carrier.id),
                int(carrier.bond_slot),
                int(carrier.bond),
                int(carrier.written_from),
                None if carrier.written_to is None else int(carrier.written_to),
            )
            for carrier in slots.carrier_slots
        ),
    )


def prefix_key(prefix: PresentationPrefix) -> tuple[object, ...]:
    return (
        tuple(
            sorted(
                (int(atom), choice.name)
                for atom, choice in prefix.atom_text.items()
            )
        ),
        tuple(
            sorted(
                (int(slot), choice.name)
                for slot, choice in prefix.bond_text.items()
            )
        ),
        tuple(
            sorted(
                (int(endpoint), label.value)
                for endpoint, label in prefix.ring_labels.items()
            )
        ),
    )


def csp_key(
    skeleton: TraversalSkeleton,
    prefix: PresentationPrefix,
) -> tuple[object, ...]:
    return (skeleton_key(skeleton), prefix_key(prefix))


def assignment_key(assignment: TraversalAssignment) -> tuple[object, ...]:
    return (
        tuple(
            sorted(
                (int(atom), choice.name)
                for atom, choice in assignment.atom_text.items()
            )
        ),
        tuple(
            sorted(
                (int(atom), token.value)
                for atom, token in assignment.tetra_tokens.items()
            )
        ),
        tuple(
            sorted(
                (int(slot), choice.name)
                for slot, choice in assignment.bond_text.items()
            )
        ),
        tuple(
            sorted(
                (int(endpoint), label.value)
                for endpoint, label in assignment.ring_labels.items()
            )
        ),
        tuple(
            sorted(
                (int(carrier), mark.value)
                for carrier, mark in assignment.direction_marks.items()
            )
        ),
    )


def stereo_solution_key(solution: StereoSolution) -> tuple[object, ...]:
    return (
        -len(solution.marker_support),
        tuple(sorted(int(carrier) for carrier in solution.marker_support)),
        tuple(
            sorted(
                (int(atom), token.value)
                for atom, token in solution.tetra_tokens.items()
            )
        ),
        tuple(
            sorted(
                (int(carrier), mark.value)
                for carrier, mark in solution.direction_marks.items()
            )
        ),
    )


def witness_id(
    *,
    skeleton: TraversalSkeleton,
    slots: SlotBundle,
    assignment: TraversalAssignment,
    rendered: str,
) -> str:
    payload = repr(
        (
            skeleton_key(skeleton),
            slot_key(slots),
            assignment_key(assignment),
            rendered,
        )
    ).encode("utf8")
    return "witness:" + blake2b(payload, digest_size=12).hexdigest()


def witness_node_id(witness_id: str) -> EnumerationNodeId:
    return EnumerationNodeId(kind="witness", key=(witness_id,))


def render_duplicate_node_id(witness_id: str) -> EnumerationNodeId:
    return EnumerationNodeId(kind="witness", key=("render_duplicate", witness_id))


def sequence_hash(values: Iterable[str]) -> str:
    digest = blake2b(digest_size=16)
    for value in values:
        digest.update(value.encode("utf8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _event_key(event: object) -> tuple[object, ...]:
    if isinstance(event, ChildEvent):
        return (
            "child",
            int(event.bond),
            int(event.parent),
            int(event.child),
            event.role.value,
        )

    if isinstance(event, RingEvent):
        return (
            "ring",
            int(event.bond),
            int(event.atom),
            int(event.other_atom),
        )

    raise TypeError(event)


__all__ = (
    "assignment_key",
    "csp_key",
    "prefix_key",
    "render_duplicate_node_id",
    "sequence_hash",
    "skeleton_key",
    "slot_key",
    "stereo_solution_key",
    "witness_id",
    "witness_node_id",
)
