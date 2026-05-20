from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rdkit import Chem


TETRAHEDRAL_TOKENS: frozenset[str] = frozenset({"@", "@@"})
IMPLICIT_HYDROGEN_LIGAND = "implicit_hydrogen"


@dataclass(frozen=True, slots=True)
class SouthStarTetrahedralCenterFact:
    center_atom_idx: int
    chiral_tag: str
    source_token: str
    explicit_neighbor_atom_indices: tuple[int, ...]
    implicit_hydrogen_count: int
    source_ligand_order: tuple[str, ...]


def extract_tetrahedral_center_facts(
    mol: Chem.Mol,
) -> tuple[SouthStarTetrahedralCenterFact, ...]:
    return tuple(
        _tetrahedral_center_fact(atom)
        for atom in mol.GetAtoms()
        if atom.GetChiralTag()
        in {
            Chem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
        }
    )


def preserving_tetrahedral_token(
    *,
    source_token: str,
    source_ligand_order: Sequence[str],
    emitted_ligand_order: Sequence[str],
) -> str:
    _validate_tetrahedral_token(source_token)
    return (
        source_token
        if _permutation_is_even(source_ligand_order, emitted_ligand_order)
        else _flipped_tetrahedral_token(source_token)
    )


def tetrahedral_token_preserves_orientation(
    *,
    candidate_token: str,
    source_token: str,
    source_ligand_order: Sequence[str],
    emitted_ligand_order: Sequence[str],
) -> bool:
    _validate_tetrahedral_token(candidate_token)
    return candidate_token == preserving_tetrahedral_token(
        source_token=source_token,
        source_ligand_order=source_ligand_order,
        emitted_ligand_order=emitted_ligand_order,
    )


def _tetrahedral_center_fact(atom: Chem.Atom) -> SouthStarTetrahedralCenterFact:
    implicit_hydrogen_count = atom.GetTotalNumHs()
    if implicit_hydrogen_count > 1:
        raise NotImplementedError(
            "South Star tetrahedral fact extraction supports at most one implicit "
            "hydrogen ligand"
        )
    explicit_neighbors = tuple(neighbor.GetIdx() for neighbor in atom.GetNeighbors())
    source_ligand_order = tuple(f"atom:{atom_idx}" for atom_idx in explicit_neighbors)
    if implicit_hydrogen_count:
        source_ligand_order += (IMPLICIT_HYDROGEN_LIGAND,)
    if len(source_ligand_order) != 4:
        raise NotImplementedError(
            "South Star tetrahedral fact extraction requires exactly four ligands"
        )
    return SouthStarTetrahedralCenterFact(
        center_atom_idx=atom.GetIdx(),
        chiral_tag=str(atom.GetChiralTag()),
        source_token=_source_token_for_chiral_tag(atom.GetChiralTag()),
        explicit_neighbor_atom_indices=explicit_neighbors,
        implicit_hydrogen_count=implicit_hydrogen_count,
        source_ligand_order=source_ligand_order,
    )


def _source_token_for_chiral_tag(chiral_tag: Chem.ChiralType) -> str:
    if chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
        return "@"
    if chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
        return "@@"
    raise ValueError(f"unsupported tetrahedral chiral tag {chiral_tag}")


def _permutation_is_even(
    source_ligand_order: Sequence[str],
    emitted_ligand_order: Sequence[str],
) -> bool:
    source = tuple(source_ligand_order)
    emitted = tuple(emitted_ligand_order)
    if len(source) != len(emitted):
        raise ValueError("tetrahedral ligand orders must have equal length")
    if len(set(source)) != len(source) or len(set(emitted)) != len(emitted):
        raise ValueError("tetrahedral ligand orders must contain unique ligands")
    if set(source) != set(emitted):
        raise ValueError("tetrahedral ligand orders must contain the same ligands")

    source_index = {ligand: idx for idx, ligand in enumerate(source)}
    permutation = tuple(source_index[ligand] for ligand in emitted)
    inversion_count = sum(
        1
        for idx, left in enumerate(permutation)
        for right in permutation[idx + 1 :]
        if left > right
    )
    return inversion_count % 2 == 0


def _flipped_tetrahedral_token(token: str) -> str:
    _validate_tetrahedral_token(token)
    return "@@" if token == "@" else "@"


def _validate_tetrahedral_token(token: str) -> None:
    if token not in TETRAHEDRAL_TOKENS:
        raise ValueError(f"tetrahedral token must be one of {TETRAHEDRAL_TOKENS}")
