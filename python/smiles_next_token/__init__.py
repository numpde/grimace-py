"""Rust-first Python surface for SMILES and next-token functionality."""

from __future__ import annotations

from smiles_next_token.reference.prepared_graph import (
    CONNECTED_NONSTEREO_SURFACE,
    CONNECTED_STEREO_SURFACE,
    PREPARED_SMILES_GRAPH_SCHEMA_VERSION,
    prepare_smiles_graph,
)

try:
    from smiles_next_token._core import (
        PreparedSmilesGraph,
        RootedConnectedNonStereoWalker,
        RootedConnectedNonStereoWalkerState,
        RootedConnectedStereoWalker,
        RootedConnectedStereoWalkerState,
        prepared_smiles_graph_schema_version,
    )
    HAVE_CORE_BINDINGS = True
except ImportError:  # pragma: no cover - exercised only when the extension is absent
    PreparedSmilesGraph = None  # type: ignore[assignment]
    RootedConnectedNonStereoWalker = None  # type: ignore[assignment]
    RootedConnectedNonStereoWalkerState = None  # type: ignore[assignment]
    RootedConnectedStereoWalker = None  # type: ignore[assignment]
    RootedConnectedStereoWalkerState = None  # type: ignore[assignment]
    prepared_smiles_graph_schema_version = None  # type: ignore[assignment]
    HAVE_CORE_BINDINGS = False

__all__ = [
    "CONNECTED_NONSTEREO_SURFACE",
    "CONNECTED_STEREO_SURFACE",
    "PREPARED_SMILES_GRAPH_SCHEMA_VERSION",
    "HAVE_CORE_BINDINGS",
    "PreparedSmilesGraph",
    "RootedConnectedNonStereoWalker",
    "RootedConnectedNonStereoWalkerState",
    "RootedConnectedStereoWalker",
    "RootedConnectedStereoWalkerState",
    "prepare_smiles_graph",
    "prepared_smiles_graph_schema_version",
]
