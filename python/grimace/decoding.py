"""Immutable prefix decoding for the pinned RDKit random noncanonical writer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rdkit import Chem

if TYPE_CHECKING:
    from ._core import _SouthStar2TextPrefix


_SUPPORTED_RDKIT_VERSION = "2026.03.1"


@dataclass(frozen=True, slots=True, init=False, eq=False)
class Prefix:
    """An immutable selected token prefix and its owned native writer states."""

    __native_prefix: _SouthStar2TextPrefix = field(repr=False)

    def __new__(cls) -> Prefix:
        raise TypeError("Prefix values are created by decoding.start()")

    @classmethod
    def _from_native(cls, native_prefix: _SouthStar2TextPrefix) -> Prefix:
        prefix = object.__new__(cls)
        object.__setattr__(prefix, "_Prefix__native_prefix", native_prefix)
        return prefix

    @property
    def is_accepted(self) -> bool:
        """Whether this prefix is a complete accepted SMILES string."""

        return self.__native_prefix.is_accepted

    def choices(self) -> tuple[tuple[str, Prefix], ...]:
        """Return unique complete-token choices and their selected successors."""

        return tuple(
            (text, Prefix._from_native(successor))
            for text, successor in self.__native_prefix.choices()
        )


def start(mol: Chem.Mol) -> Prefix:
    """Prepare an RDKit molecule and return its initial immutable prefix.

    This entry point requires the pinned RDKit writer convention. Preparation
    snapshots supported molecular and stereo facts; subsequent choices are
    produced by the native Rust writer state.
    """

    from rdkit import rdBase

    installed_version = rdBase.rdkitVersion
    if installed_version != _SUPPORTED_RDKIT_VERSION:
        raise RuntimeError(
            "grimace.decoding.start requires RDKit "
            f"{_SUPPORTED_RDKIT_VERSION}; installed RDKit is {installed_version}."
        )

    from ._south_star2 import preparation

    prepared = preparation.prepare_molecule(mol)
    native_prefix = prepared.build_writer().initial_prefix()
    return Prefix._from_native(native_prefix)


__all__ = ["Prefix", "start"]
