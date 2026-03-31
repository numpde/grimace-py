"""Public runtime API for the Rust-backed SMILES next-token engine."""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from typing import Any

try:
    _RUNTIME = importlib.import_module("grimace._runtime")
except ImportError as exc:  # pragma: no cover - exercised only in broken installs
    _RUNTIME = None
    _CORE_IMPORT_ERROR = exc
else:  # pragma: no cover - exercised in environments with the extension available
    _CORE_IMPORT_ERROR = None


def _require_runtime() -> Any:
    if _RUNTIME is None:
        raise ImportError(
            "grimace requires the compiled Rust extension "
            "'grimace._core'. Build or install the package with the "
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


def MolToSmilesTokenInventory(
    mol: object,
    *,
    isomericSmiles: bool = True,
    kekuleSmiles: bool = False,
    rootedAtAtom: int | None = None,
    canonical: bool = True,
    allBondsExplicit: bool = False,
    allHsExplicit: bool = False,
    doRandom: bool = False,
    ignoreAtomMapNumbers: bool = False,
) -> tuple[str, ...]:
    """Return the token inventory for a molecule under the public runtime flags."""

    runtime = _require_runtime()
    return runtime.mol_to_smiles_token_inventory(
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


class MolToSmilesDecoder:
    __slots__ = ("_impl",)

    def __init__(
        self,
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
    ) -> None:
        runtime = _require_runtime()
        self._impl = runtime.MolToSmilesDecoder(
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

    @classmethod
    def _from_impl(cls, impl: object) -> "MolToSmilesDecoder":
        decoder = cls.__new__(cls)
        decoder._impl = impl
        return decoder

    @property
    def next_tokens(self) -> tuple[str, ...]:
        return self._impl.next_tokens

    def advance(self, token: str) -> "MolToSmilesDecoder":
        self._impl.advance(token)
        return self

    @property
    def prefix(self) -> str:
        return self._impl.prefix

    @property
    def is_terminal(self) -> bool:
        return self._impl.is_terminal

    def copy(self) -> "MolToSmilesDecoder":
        return type(self)._from_impl(self._impl.copy())


__all__ = ["MolToSmilesDecoder", "MolToSmilesEnum", "MolToSmilesTokenInventory"]
