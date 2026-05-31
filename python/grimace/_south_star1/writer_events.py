"""Typed semantic events emitted by writer-shaped transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Literal
from typing import TypeAlias

from .ids import AtomId
from .ids import BondId
from .policy import DirectionMark
from .policy import TetraToken

if TYPE_CHECKING:
    from .writer_state import WriterClosureLabel


@dataclass(frozen=True, slots=True)
class WriterAtomEmitted:
    atom: AtomId
    text: str
    tetra_token: TetraToken
    parent: AtomId | None = None
    incoming_bond: BondId | None = None


@dataclass(frozen=True, slots=True)
class WriterBondEmitted:
    bond: BondId
    parent: AtomId
    child: AtomId
    text: str
    direction_mark: DirectionMark


@dataclass(frozen=True, slots=True)
class WriterBranchOpened:
    parent: AtomId
    child: AtomId
    bond: BondId


@dataclass(frozen=True, slots=True)
class WriterBranchClosed:
    atom: AtomId


@dataclass(frozen=True, slots=True)
class WriterComponentBoundaryEmitted:
    next_root: AtomId


@dataclass(frozen=True, slots=True)
class WriterLocalOrderClosed:
    atom: AtomId


@dataclass(frozen=True, slots=True)
class WriterRingEndpointEmitted:
    bond: BondId
    endpoint_atom: AtomId
    partner_atom: AtomId
    label: "WriterClosureLabel"
    endpoint_text: str
    bond_text: str
    side: Literal["open", "close"] = "open"


@dataclass(frozen=True, slots=True)
class WriterRingEndpointPaired:
    bond: BondId
    endpoint_atom: AtomId
    partner_atom: AtomId
    label: "WriterClosureLabel"
    endpoint_text: str
    bond_text: str
    side: Literal["close"] = "close"


WriterEvent: TypeAlias = (
    WriterAtomEmitted
    | WriterBondEmitted
    | WriterBranchOpened
    | WriterBranchClosed
    | WriterComponentBoundaryEmitted
    | WriterLocalOrderClosed
    | WriterRingEndpointEmitted
    | WriterRingEndpointPaired
)


def writer_event_sort_tuple(event: WriterEvent) -> tuple[object, ...]:
    if isinstance(event, WriterAtomEmitted):
        return (
            "atom",
            int(event.atom),
            event.text,
            event.tetra_token.value,
            None if event.parent is None else int(event.parent),
            None if event.incoming_bond is None else int(event.incoming_bond),
        )
    if isinstance(event, WriterBondEmitted):
        return (
            "bond",
            int(event.bond),
            int(event.parent),
            int(event.child),
            event.text,
            event.direction_mark.value,
        )
    if isinstance(event, WriterBranchOpened):
        return ("branch_open", int(event.parent), int(event.child), int(event.bond))
    if isinstance(event, WriterBranchClosed):
        return ("branch_close", int(event.atom))
    if isinstance(event, WriterComponentBoundaryEmitted):
        return ("component_boundary", int(event.next_root))
    if isinstance(event, WriterLocalOrderClosed):
        return ("local_order_closed", int(event.atom))
    if isinstance(event, WriterRingEndpointEmitted):
        return (
            "ring_endpoint",
            int(event.bond),
            int(event.endpoint_atom),
            int(event.partner_atom),
            event.label.value,
            event.label.text,
            event.endpoint_text,
            event.bond_text,
            event.side,
        )
    if isinstance(event, WriterRingEndpointPaired):
        return (
            "ring_pair",
            int(event.bond),
            int(event.endpoint_atom),
            int(event.partner_atom),
            event.label.value,
            event.label.text,
            event.endpoint_text,
            event.bond_text,
            event.side,
        )
    raise TypeError(f"unknown writer event: {event!r}")


__all__ = (
    "WriterAtomEmitted",
    "WriterBondEmitted",
    "WriterBranchClosed",
    "WriterBranchOpened",
    "WriterComponentBoundaryEmitted",
    "WriterEvent",
    "WriterLocalOrderClosed",
    "WriterRingEndpointEmitted",
    "WriterRingEndpointPaired",
    "writer_event_sort_tuple",
)
