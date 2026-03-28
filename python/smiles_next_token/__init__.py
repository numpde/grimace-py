"""Public runtime API for the Rust-backed SMILES next-token engine."""

from __future__ import annotations

import importlib
from typing import Any

try:
    _RUNTIME = importlib.import_module("smiles_next_token._runtime")
except ImportError as exc:  # pragma: no cover - exercised only in broken installs
    _RUNTIME = None
    _CORE_IMPORT_ERROR = exc
else:  # pragma: no cover - exercised in environments with the extension available
    _CORE_IMPORT_ERROR = None


def _require_runtime() -> Any:
    if _RUNTIME is None:
        raise ImportError(
            "smiles_next_token requires the compiled Rust extension "
            "'smiles_next_token._core'. Build or install the package with the "
            "extension enabled."
        ) from _CORE_IMPORT_ERROR
    return _RUNTIME


def MolToSmilesSupport(
    mol: object,
    *,
    rootedAtAtom: int,
    isomericSmiles: bool = True,
    connectedOnly: bool = True,
) -> set[str]:
    """Return the exact rooted SMILES support for a molecule.

    Parameters mirror the RDKit naming style where possible. The current
    runtime supports only connected molecules, so ``connectedOnly=False`` is
    rejected explicitly.
    """

    if not connectedOnly:
        raise NotImplementedError("connectedOnly=False is not implemented")

    runtime = _require_runtime()
    if isomericSmiles:
        return runtime.enumerate_rooted_connected_stereo_smiles_support(
            mol,
            rootedAtAtom,
        )
    return runtime.enumerate_rooted_connected_nonstereo_smiles_support(
        mol,
        rootedAtAtom,
    )


__all__ = ["MolToSmilesSupport"]
