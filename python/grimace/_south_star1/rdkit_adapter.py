"""RDKit ingestion boundary for the private proof kernel.

This is the only South Star 1 module intended to snapshot RDKit ``Mol`` objects
into immutable molecule facts. It must remain a one-way adapter and must not be
called by core enumeration for candidate validation.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()
