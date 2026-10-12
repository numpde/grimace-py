"""Immutable prefixes for a bounded OpenSMILES atom-and-writer language.

RDKit supplies molecular facts once; Rust owns spelling and online traversal.
The default is an implemented language subset, not a complete SMILES grammar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from rdkit import Chem

if TYPE_CHECKING:
    from ._core import _SouthStar2TextPrefix


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


def start(
    mol: Chem.Mol,
    *,
    style: Literal["rdkit_random_noncanonical"] | None = None,
) -> Prefix:
    """Prepare an RDKit molecule and return its initial immutable prefix.

    Preparation snapshots supported molecular and stereo facts from the
    installed RDKit. ``style=None`` exposes the implemented OpenSMILES subset;
    ``style="rdkit_random_noncanonical"`` further applies the qualified
    writer-convention restrictions. Neither mode calls RDKit while choices
    are made. The RDKit style label does not certify installed-writer parity.
    """

    from ._south_star2 import preparation

    prepared = preparation.prepare_molecule(mol)
    native_prefix = prepared.build_writer(style=style).initial_prefix()
    return Prefix._from_native(native_prefix)


__all__ = ["Prefix", "start"]
