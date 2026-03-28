"""Internal runtime bridge between RDKit input and the Rust core."""

from __future__ import annotations

import importlib
from functools import lru_cache

_core = importlib.import_module("smiles_next_token._core")
from smiles_next_token._reference.policy import (
    DEFAULT_RDKIT_RANDOM_CONNECTED_NONSTEREO_POLICY_PATH,
    DEFAULT_RDKIT_RANDOM_POLICY_PATH,
    ReferencePolicy,
)
from smiles_next_token._reference.prepared_graph import (
    CONNECTED_NONSTEREO_SURFACE,
    CONNECTED_STEREO_SURFACE,
    PREPARED_SMILES_GRAPH_SCHEMA_VERSION,
    PreparedSmilesGraph as ReferencePreparedSmilesGraph,
    prepare_smiles_graph as _prepare_reference_smiles_graph,
)


def _validate_surface_kind(
    prepared: object,
    *,
    surface_kind: str,
) -> None:
    if prepared.surface_kind != surface_kind:
        raise ValueError(
            f"PreparedSmilesGraph surface_kind={prepared.surface_kind!r} does not match "
            f"the requested surface_kind={surface_kind!r}"
        )


def _validate_policy(
    prepared: object,
    policy: ReferencePolicy | None,
) -> None:
    if policy is None:
        return
    if isinstance(prepared, ReferencePreparedSmilesGraph):
        prepared.validate_policy(policy)
        return
    if isinstance(prepared, _core.PreparedSmilesGraph):
        prepared.validate_policy(policy.policy_name, policy.digest())
        return
    raise TypeError(f"Unsupported prepared graph type: {type(prepared)!r}")


@lru_cache(maxsize=2)
def _default_policy(surface_kind: str) -> ReferencePolicy:
    if surface_kind == CONNECTED_NONSTEREO_SURFACE:
        return ReferencePolicy.from_path(DEFAULT_RDKIT_RANDOM_CONNECTED_NONSTEREO_POLICY_PATH)
    if surface_kind == CONNECTED_STEREO_SURFACE:
        return ReferencePolicy.from_path(DEFAULT_RDKIT_RANDOM_POLICY_PATH)
    raise ValueError(f"Unsupported surface kind: {surface_kind!r}")


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
    if isinstance(mol_or_prepared, _core.PreparedSmilesGraph):
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


def prepare_smiles_graph(
    mol_or_prepared: object,
    policy: ReferencePolicy | None = None,
    *,
    surface_kind: str = CONNECTED_NONSTEREO_SURFACE,
) -> _core.PreparedSmilesGraph:
    policy = _default_policy(surface_kind) if policy is None else policy
    if isinstance(mol_or_prepared, _core.PreparedSmilesGraph):
        _validate_surface_kind(mol_or_prepared, surface_kind=surface_kind)
        _validate_policy(mol_or_prepared, policy)
        return mol_or_prepared

    reference_prepared = _coerce_reference_prepared_graph(
        mol_or_prepared,
        policy,
        surface_kind=surface_kind,
    )
    return _core.PreparedSmilesGraph(reference_prepared)


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
    return set(prepared.enumerate_rooted_connected_nonstereo_support(root_idx))


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
    return set(prepared.enumerate_rooted_connected_stereo_support(root_idx))


def make_nonstereo_walker(
    mol_or_prepared: object,
    root_idx: int,
    policy: ReferencePolicy | None = None,
) -> _core.RootedConnectedNonStereoWalker:
    prepared = prepare_smiles_graph(
        mol_or_prepared,
        policy,
        surface_kind=CONNECTED_NONSTEREO_SURFACE,
    )
    return _core.RootedConnectedNonStereoWalker(prepared, root_idx)


def make_stereo_walker(
    mol_or_prepared: object,
    root_idx: int,
    policy: ReferencePolicy | None = None,
) -> _core.RootedConnectedStereoWalker:
    prepared = prepare_smiles_graph(
        mol_or_prepared,
        policy,
        surface_kind=CONNECTED_STEREO_SURFACE,
    )
    return _core.RootedConnectedStereoWalker(prepared, root_idx)


def prepared_smiles_graph_schema_version() -> int:
    core_version = _core.prepared_smiles_graph_schema_version()
    if core_version != PREPARED_SMILES_GRAPH_SCHEMA_VERSION:
        raise RuntimeError(
            "Python RDKit bridge and Rust core disagree on prepared graph schema "
            f"version: python={PREPARED_SMILES_GRAPH_SCHEMA_VERSION}, core={core_version}"
        )
    return core_version


__all__ = [
    "CONNECTED_NONSTEREO_SURFACE",
    "CONNECTED_STEREO_SURFACE",
    "PREPARED_SMILES_GRAPH_SCHEMA_VERSION",
    "enumerate_rooted_connected_nonstereo_smiles_support",
    "enumerate_rooted_connected_stereo_smiles_support",
    "make_nonstereo_walker",
    "make_prepared_graph",
    "make_stereo_walker",
    "prepare_smiles_graph",
    "prepared_smiles_graph_schema_version",
]
