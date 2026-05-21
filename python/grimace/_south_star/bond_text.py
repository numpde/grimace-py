from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem


SOUTH_STAR_SUPPORTED_BOND_TYPES: frozenset[Chem.BondType] = frozenset(
    {
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
        Chem.BondType.QUADRUPLE,
        Chem.BondType.AROMATIC,
    }
)


@dataclass(frozen=True, slots=True)
class SouthStarBondTextObligation:
    bond_type: Chem.BondType
    emitted_text: str
    token_family: str


def bond_text_obligation_for_supported_bond(
    bond: Chem.Bond,
) -> SouthStarBondTextObligation:
    bond_type = bond.GetBondType()
    if bond_type == Chem.BondType.SINGLE:
        return SouthStarBondTextObligation(
            bond_type=bond_type,
            emitted_text="",
            token_family="elided_single_bond",
        )
    if bond_type == Chem.BondType.DOUBLE:
        return SouthStarBondTextObligation(
            bond_type=bond_type,
            emitted_text="=",
            token_family="explicit_double_bond",
        )
    if bond_type == Chem.BondType.TRIPLE:
        return SouthStarBondTextObligation(
            bond_type=bond_type,
            emitted_text="#",
            token_family="explicit_triple_bond",
        )
    if bond_type == Chem.BondType.QUADRUPLE:
        return SouthStarBondTextObligation(
            bond_type=bond_type,
            emitted_text="$",
            token_family="explicit_quadruple_bond",
        )
    if bond_type == Chem.BondType.AROMATIC:
        return SouthStarBondTextObligation(
            bond_type=bond_type,
            emitted_text="",
            token_family="elided_aromatic_bond",
        )
    raise NotImplementedError(
        f"South Star bond text unsupported for bond type {bond_type}"
    )


def bond_text_for_supported_bond(bond: Chem.Bond) -> str:
    return bond_text_obligation_for_supported_bond(bond).emitted_text
