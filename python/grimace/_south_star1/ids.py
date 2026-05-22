"""Typed identifiers for the private South Star 1 proof kernel."""

from __future__ import annotations

from typing import NewType

AtomId = NewType("AtomId", int)
BondId = NewType("BondId", int)
ComponentId = NewType("ComponentId", int)
SiteId = NewType("SiteId", int)
OccurrenceId = NewType("OccurrenceId", int)

AtomSlotId = NewType("AtomSlotId", int)
BondSlotId = NewType("BondSlotId", int)
CarrierSlotId = NewType("CarrierSlotId", int)
RingEndpointId = NewType("RingEndpointId", int)
EventPos = NewType("EventPos", int)

__all__ = (
    "AtomId",
    "AtomSlotId",
    "BondId",
    "BondSlotId",
    "CarrierSlotId",
    "ComponentId",
    "EventPos",
    "OccurrenceId",
    "RingEndpointId",
    "SiteId",
)
