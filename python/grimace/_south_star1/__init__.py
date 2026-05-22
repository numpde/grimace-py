"""Private South Star 1 proof-kernel package.

This package is confined implementation scaffolding for the formal exact-support
model. It is not a public API and must not be re-exported from ``grimace``.

The core modules in this package are intentionally RDKit-free. RDKit belongs
only at the adapter/audit boundary.
"""

from __future__ import annotations

CORE_MODULES: tuple[str, ...] = (
    "annotation",
    "constraints",
    "enumerate",
    "facts",
    "graph_index",
    "ids",
    "policy",
    "render",
    "ring_labels",
    "semantics",
    "skeleton",
    "slots",
)

BOUNDARY_MODULES: tuple[str, ...] = (
    "audit_rdkit",
    "rdkit_adapter",
)

__all__ = ("BOUNDARY_MODULES", "CORE_MODULES")
