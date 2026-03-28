"""Rust-first Python runtime surface for SMILES next-token functionality."""

from __future__ import annotations

from typing import Any

from smiles_next_token.reference.policy import ReferencePolicy
from smiles_next_token.reference.prepared_graph import (
    CONNECTED_NONSTEREO_SURFACE,
    CONNECTED_STEREO_SURFACE,
    PREPARED_SMILES_GRAPH_SCHEMA_VERSION,
    PreparedSmilesGraph as ReferencePreparedSmilesGraph,
    prepare_smiles_graph as _prepare_reference_smiles_graph,
)
from smiles_next_token.reference.rooted.connected_nonstereo import (
    RootedConnectedNonStereoWalker as ReferenceRootedConnectedNonStereoWalker,
    RootedConnectedNonStereoWalkerState as ReferenceRootedConnectedNonStereoWalkerState,
    enumerate_rooted_connected_nonstereo_smiles_support as _enumerate_reference_nonstereo_support,
)
from smiles_next_token.reference.rooted.connected_stereo import (
    enumerate_rooted_connected_stereo_smiles_support as _enumerate_reference_stereo_support,
)

try:
    from smiles_next_token._core import (
        PreparedSmilesGraph as CorePreparedSmilesGraph,
        RootedConnectedNonStereoWalker as CoreRootedConnectedNonStereoWalker,
        RootedConnectedNonStereoWalkerState as CoreRootedConnectedNonStereoWalkerState,
        RootedConnectedStereoWalker as CoreRootedConnectedStereoWalker,
        RootedConnectedStereoWalkerState as CoreRootedConnectedStereoWalkerState,
        prepared_smiles_graph_schema_version as _core_prepared_smiles_graph_schema_version,
    )

    HAVE_CORE_BINDINGS = True
except ImportError:  # pragma: no cover - exercised only when the extension is absent
    CorePreparedSmilesGraph = None  # type: ignore[assignment]
    CoreRootedConnectedNonStereoWalker = None  # type: ignore[assignment]
    CoreRootedConnectedNonStereoWalkerState = None  # type: ignore[assignment]
    CoreRootedConnectedStereoWalker = None  # type: ignore[assignment]
    CoreRootedConnectedStereoWalkerState = None  # type: ignore[assignment]
    _core_prepared_smiles_graph_schema_version = None
    HAVE_CORE_BINDINGS = False

PreparedSmilesGraph = (
    CorePreparedSmilesGraph if HAVE_CORE_BINDINGS else ReferencePreparedSmilesGraph
)
RootedConnectedNonStereoWalker = (
    CoreRootedConnectedNonStereoWalker
    if HAVE_CORE_BINDINGS
    else ReferenceRootedConnectedNonStereoWalker
)
RootedConnectedNonStereoWalkerState = (
    CoreRootedConnectedNonStereoWalkerState
    if HAVE_CORE_BINDINGS
    else ReferenceRootedConnectedNonStereoWalkerState
)
RootedConnectedStereoWalker = CoreRootedConnectedStereoWalker
RootedConnectedStereoWalkerState = CoreRootedConnectedStereoWalkerState


def _validate_surface_kind(
    prepared: Any,
    *,
    surface_kind: str,
) -> None:
    if prepared.surface_kind != surface_kind:
        raise ValueError(
            f"PreparedSmilesGraph surface_kind={prepared.surface_kind!r} does not match "
            f"the requested surface_kind={surface_kind!r}"
        )


def _validate_policy(
    prepared: Any,
    policy: ReferencePolicy | None,
) -> None:
    if policy is None:
        return
    if isinstance(prepared, ReferencePreparedSmilesGraph):
        prepared.validate_policy(policy)
        return
    if HAVE_CORE_BINDINGS and CorePreparedSmilesGraph is not None and isinstance(
        prepared, CorePreparedSmilesGraph
    ):
        prepared.validate_policy(policy.policy_name, policy.digest())
        return
    raise TypeError(f"Unsupported prepared graph type: {type(prepared)!r}")


def _coerce_reference_prepared_graph(
    mol_or_prepared: object,
    policy: ReferencePolicy | None,
    *,
    surface_kind: str,
) -> ReferencePreparedSmilesGraph:
    if isinstance(mol_or_prepared, ReferencePreparedSmilesGraph):
        _validate_surface_kind(mol_or_prepared, surface_kind=surface_kind)
        _validate_policy(mol_or_prepared, policy)
        return mol_or_prepared
    if HAVE_CORE_BINDINGS and CorePreparedSmilesGraph is not None and isinstance(
        mol_or_prepared, CorePreparedSmilesGraph
    ):
        _validate_surface_kind(mol_or_prepared, surface_kind=surface_kind)
        _validate_policy(mol_or_prepared, policy)
        return ReferencePreparedSmilesGraph(mol_or_prepared.to_dict())
    if policy is None:
        raise TypeError("policy is required when preparing a graph from an RDKit molecule")
    return _prepare_reference_smiles_graph(
        mol_or_prepared,
        policy,
        surface_kind=surface_kind,
    )


def _coerce_runtime_prepared_graph(
    mol_or_prepared: object,
    policy: ReferencePolicy | None = None,
    *,
    surface_kind: str = CONNECTED_NONSTEREO_SURFACE,
) -> Any:
    if HAVE_CORE_BINDINGS and CorePreparedSmilesGraph is not None and isinstance(
        mol_or_prepared, CorePreparedSmilesGraph
    ):
        _validate_surface_kind(mol_or_prepared, surface_kind=surface_kind)
        _validate_policy(mol_or_prepared, policy)
        return mol_or_prepared

    reference_prepared = _coerce_reference_prepared_graph(
        mol_or_prepared,
        policy,
        surface_kind=surface_kind,
    )
    if HAVE_CORE_BINDINGS and CorePreparedSmilesGraph is not None:
        return CorePreparedSmilesGraph(reference_prepared)
    return reference_prepared


def prepare_smiles_graph(
    mol_or_prepared: object,
    policy: ReferencePolicy | None = None,
    *,
    surface_kind: str = CONNECTED_NONSTEREO_SURFACE,
) -> Any:
    """Build the runtime prepared graph, preferring the Rust core when available."""

    return _coerce_runtime_prepared_graph(
        mol_or_prepared,
        policy,
        surface_kind=surface_kind,
    )


make_prepared_graph = prepare_smiles_graph


def enumerate_rooted_connected_nonstereo_smiles_support(
    mol_or_prepared: object,
    root_idx: int,
    policy: ReferencePolicy | None = None,
) -> set[str]:
    prepared = prepare_smiles_graph(
        mol_or_prepared,
        policy,
        surface_kind=CONNECTED_NONSTEREO_SURFACE,
    )
    if HAVE_CORE_BINDINGS:
        return set(prepared.enumerate_rooted_connected_nonstereo_support(root_idx))
    return _enumerate_reference_nonstereo_support(prepared, root_idx)


def enumerate_rooted_connected_stereo_smiles_support(
    mol_or_prepared: object,
    root_idx: int,
    policy: ReferencePolicy | None = None,
) -> set[str]:
    prepared = prepare_smiles_graph(
        mol_or_prepared,
        policy,
        surface_kind=CONNECTED_STEREO_SURFACE,
    )
    if HAVE_CORE_BINDINGS:
        return set(prepared.enumerate_rooted_connected_stereo_support(root_idx))
    return _enumerate_reference_stereo_support(prepared, root_idx)


enumerate_rooted_nonstereo_smiles_support = (
    enumerate_rooted_connected_nonstereo_smiles_support
)
enumerate_rooted_smiles_support = enumerate_rooted_connected_nonstereo_smiles_support


def make_nonstereo_walker(
    mol_or_prepared: object,
    root_idx: int,
    policy: ReferencePolicy | None = None,
) -> Any:
    prepared = prepare_smiles_graph(
        mol_or_prepared,
        policy,
        surface_kind=CONNECTED_NONSTEREO_SURFACE,
    )
    if HAVE_CORE_BINDINGS and CoreRootedConnectedNonStereoWalker is not None:
        return CoreRootedConnectedNonStereoWalker(prepared, root_idx)
    return ReferenceRootedConnectedNonStereoWalker(prepared, root_idx)


def make_stereo_walker(
    mol_or_prepared: object,
    root_idx: int,
    policy: ReferencePolicy | None = None,
) -> Any:
    prepared = prepare_smiles_graph(
        mol_or_prepared,
        policy,
        surface_kind=CONNECTED_STEREO_SURFACE,
    )
    if not HAVE_CORE_BINDINGS or CoreRootedConnectedStereoWalker is None:
        raise RuntimeError(
            "RootedConnectedStereoWalker requires the Rust core bindings; "
            "use smiles_next_token.rdkit_reference for the reference enumerator"
        )
    return CoreRootedConnectedStereoWalker(prepared, root_idx)


def prepared_smiles_graph_schema_version() -> int:
    if _core_prepared_smiles_graph_schema_version is not None:
        return _core_prepared_smiles_graph_schema_version()
    return PREPARED_SMILES_GRAPH_SCHEMA_VERSION


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
    "enumerate_rooted_connected_nonstereo_smiles_support",
    "enumerate_rooted_connected_stereo_smiles_support",
    "enumerate_rooted_nonstereo_smiles_support",
    "enumerate_rooted_smiles_support",
    "make_nonstereo_walker",
    "make_prepared_graph",
    "make_stereo_walker",
    "prepare_smiles_graph",
    "prepared_smiles_graph_schema_version",
]
