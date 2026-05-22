"""Finite parser-semantics relations for the private proof kernel."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol
from typing import runtime_checkable

from .facts import DirectionalValue
from .facts import MoleculeFacts
from .facts import TetraValue
from .ids import AtomId
from .ids import BondId
from .ids import CarrierSlotId
from .ids import OccurrenceId
from .ids import SiteId
from .policy import AtomTextChoice
from .policy import BondTextChoice
from .policy import DirectionMark
from .policy import TetraToken


class Invalid:
    """Sentinel type for a finite semantic relation that rejects an assignment."""


INVALID = Invalid()


@runtime_checkable
class ParserSemantics(Protocol):
    """Protocol for finite parser relations in the bounded dialect.

    Skeleton and slot objects are intentionally typed as ``object`` in this
    initial boundary slice; later Backlog items will replace them with concrete
    traversal and slot records.
    """

    def atom_decode_ok(
        self,
        facts: MoleculeFacts,
        atom: AtomId,
        atom_text: AtomTextChoice,
        tetra_token: TetraToken,
        incident_bond_texts: tuple[BondTextChoice, ...],
    ) -> bool:
        ...

    def bond_decode_ok(
        self,
        facts: MoleculeFacts,
        bond: BondId,
        bond_text: BondTextChoice,
        direction_mark: DirectionMark,
    ) -> bool:
        ...

    def ring_pair_decode_ok(
        self,
        facts: MoleculeFacts,
        bond: BondId,
        endpoint_1: BondTextChoice,
        mark_1: DirectionMark,
        endpoint_2: BondTextChoice,
        mark_2: DirectionMark,
    ) -> bool:
        ...

    def local_tetra_order(
        self,
        facts: MoleculeFacts,
        skel: object,
        slots: object,
        site: SiteId,
    ) -> tuple[OccurrenceId, ...]:
        ...

    def tetra_value(
        self,
        facts: MoleculeFacts,
        site: SiteId,
        local_order: tuple[OccurrenceId, ...],
        token: TetraToken,
    ) -> TetraValue | Invalid:
        ...

    def directional_value(
        self,
        facts: MoleculeFacts,
        skel: object,
        slots: object,
        site: SiteId,
        marks: Mapping[CarrierSlotId, DirectionMark],
    ) -> DirectionalValue | Invalid:
        ...


__all__ = ("INVALID", "Invalid", "ParserSemantics")
