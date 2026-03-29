"""Public runtime API for the Rust-backed SMILES next-token engine."""

from __future__ import annotations

import importlib
from collections.abc import Iterator
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


def MolToSmilesEnum(
    mol: object,
    *,
    isomericSmiles: bool = True,
    kekuleSmiles: bool = False,
    rootedAtAtom: int = -1,
    canonical: bool = True,
    allBondsExplicit: bool = False,
    allHsExplicit: bool = False,
    doRandom: bool = False,
    ignoreAtomMapNumbers: bool = False,
) -> Iterator[str]:
    """Yield the exact rooted SMILES support for a molecule."""

    runtime = _require_runtime()
    return runtime.mol_to_smiles_enum(
        mol,
        isomeric_smiles=isomericSmiles,
        kekule_smiles=kekuleSmiles,
        rooted_at_atom=rootedAtAtom,
        canonical=canonical,
        all_bonds_explicit=allBondsExplicit,
        all_hs_explicit=allHsExplicit,
        do_random=doRandom,
        ignore_atom_map_numbers=ignoreAtomMapNumbers,
    )


__all__ = ["MolToSmilesEnum"]
