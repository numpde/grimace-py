"""Public runtime API for the Rust-backed SMILES next-token engine."""

from __future__ import annotations

import importlib

from smiles_next_token._reference.policy import ReferencePolicy

_RUNTIME_EXPORTS = (
    "enumerate_rooted_connected_nonstereo_smiles_support",
    "enumerate_rooted_connected_stereo_smiles_support",
    "enumerate_rooted_nonstereo_smiles_support",
    "enumerate_rooted_smiles_support",
)

try:
    _core = importlib.import_module("smiles_next_token._core")
except ImportError as exc:  # pragma: no cover - exercised only in broken installs
    _core = None  # type: ignore[assignment]
    _CORE_IMPORT_ERROR = exc
else:
    _CORE_IMPORT_ERROR = None
    from smiles_next_token._runtime import (
        enumerate_rooted_connected_nonstereo_smiles_support,
        enumerate_rooted_connected_stereo_smiles_support,
    )

    enumerate_rooted_nonstereo_smiles_support = (
        enumerate_rooted_connected_nonstereo_smiles_support
    )
    enumerate_rooted_smiles_support = enumerate_rooted_connected_nonstereo_smiles_support


def __getattr__(name: str) -> object:
    if name in _RUNTIME_EXPORTS and _CORE_IMPORT_ERROR is not None:
        raise ImportError(
            "smiles_next_token requires the compiled Rust extension "
            "'smiles_next_token._core'. Build or install the package with the "
            "extension enabled."
        ) from _CORE_IMPORT_ERROR
    raise AttributeError(name)


__all__ = ["ReferencePolicy", *_RUNTIME_EXPORTS]
