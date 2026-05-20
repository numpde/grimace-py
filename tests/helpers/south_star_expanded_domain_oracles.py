from __future__ import annotations

"""Independent test oracles for expanded South Star fixture domains.

These helpers intentionally duplicate traversal/rendering concepts instead of
calling `grimace._south_star.enum_s`. They are spec oracles for fixture
completeness, not runtime code and not a second package implementation.
"""

from dataclasses import dataclass
from itertools import permutations
from itertools import product

from rdkit import Chem

from grimace._south_star.annotation_policy import Edge
from grimace._south_star.annotation_policy import normalized_edge
from tests.helpers.south_star_exact_support import SouthStarExpandedSupportCase
from tests.helpers.south_star_semantic_oracle import parse_smiles


@dataclass(frozen=True, slots=True)
class _RingToken:
    text: str
    atom_idx: int | None = None


@dataclass(frozen=True, slots=True)
class _RingFragment:
    tokens: tuple[_RingToken, ...]


def independent_saturated_monocycle_support_for_case(
    case: SouthStarExpandedSupportCase,
) -> tuple[str, ...]:
    """Enumerate saturated-monocycle support without using EnumS traversal."""
    mol = parse_smiles(case.source_smiles)
    _assert_saturated_monocycle_domain(mol)

    outputs = []
    for root_idx in range(mol.GetNumAtoms()):
        for closure_edge in _single_ring_edges(mol):
            for fragment in _tree_fragments_with_blocked_edge(
                mol,
                atom_idx=root_idx,
                parent_idx=None,
                visited=frozenset(),
                blocked_edge=closure_edge,
            ):
                outputs.append(_render_with_ring_digit(fragment, closure_edge))
    return tuple(dict.fromkeys(outputs))


def _tree_fragments_with_blocked_edge(
    mol: Chem.Mol,
    *,
    atom_idx: int,
    parent_idx: int | None,
    visited: frozenset[int],
    blocked_edge: Edge,
) -> tuple[_RingFragment, ...]:
    visited = visited | {atom_idx}
    children = tuple(
        neighbor.GetIdx()
        for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
        if neighbor.GetIdx() != parent_idx
        and neighbor.GetIdx() not in visited
        and normalized_edge((atom_idx, neighbor.GetIdx())) != blocked_edge
    )
    atom_token = _RingToken(_atom_text(mol.GetAtomWithIdx(atom_idx)), atom_idx=atom_idx)
    if not children:
        return (_RingFragment(tokens=(atom_token,)),)

    fragments = []
    for ordered_children in permutations(sorted(children)):
        branch_children = ordered_children[:-1]
        main_child = ordered_children[-1]
        branch_groups = tuple(
            _branch_fragments(
                mol,
                begin_atom_idx=atom_idx,
                end_atom_idx=child_idx,
                visited=visited,
                blocked_edge=blocked_edge,
            )
            for child_idx in branch_children
        )
        main_fragments = _child_fragments(
            mol,
            begin_atom_idx=atom_idx,
            end_atom_idx=main_child,
            visited=visited,
            blocked_edge=blocked_edge,
        )
        for branch_group in product(*branch_groups):
            branch_tokens = tuple(
                token for branch in branch_group for token in branch.tokens
            )
            for main_fragment in main_fragments:
                fragments.append(
                    _RingFragment(
                        tokens=(atom_token,) + branch_tokens + main_fragment.tokens,
                    )
                )
    return tuple(dict.fromkeys(fragments))


def _branch_fragments(
    mol: Chem.Mol,
    *,
    begin_atom_idx: int,
    end_atom_idx: int,
    visited: frozenset[int],
    blocked_edge: Edge,
) -> tuple[_RingFragment, ...]:
    bond_text = _bond_text(mol, begin_atom_idx, end_atom_idx)
    return tuple(
        _RingFragment(
            tokens=(_RingToken("("), _RingToken(bond_text))
            + child_fragment.tokens
            + (_RingToken(")"),)
        )
        for child_fragment in _tree_fragments_with_blocked_edge(
            mol,
            atom_idx=end_atom_idx,
            parent_idx=begin_atom_idx,
            visited=visited,
            blocked_edge=blocked_edge,
        )
    )


def _child_fragments(
    mol: Chem.Mol,
    *,
    begin_atom_idx: int,
    end_atom_idx: int,
    visited: frozenset[int],
    blocked_edge: Edge,
) -> tuple[_RingFragment, ...]:
    bond_text = _bond_text(mol, begin_atom_idx, end_atom_idx)
    return tuple(
        _RingFragment(tokens=(_RingToken(bond_text),) + child_fragment.tokens)
        for child_fragment in _tree_fragments_with_blocked_edge(
            mol,
            atom_idx=end_atom_idx,
            parent_idx=begin_atom_idx,
            visited=visited,
            blocked_edge=blocked_edge,
        )
    )


def _render_with_ring_digit(fragment: _RingFragment, closure_edge: Edge) -> str:
    rendered = []
    endpoint_count = 0
    for token in fragment.tokens:
        rendered.append(token.text)
        if token.atom_idx in closure_edge:
            endpoint_count += 1
            rendered.append("1")
    if endpoint_count != 2:
        raise ValueError(
            f"closure edge {closure_edge!r} endpoints were not both rendered"
        )
    return "".join(rendered)


def _single_ring_edges(mol: Chem.Mol) -> tuple[Edge, ...]:
    bond_rings = mol.GetRingInfo().BondRings()
    if len(bond_rings) != 1:
        raise ValueError("saturated-monocycle oracle expects one ring")
    return tuple(
        normalized_edge(
            (
                mol.GetBondWithIdx(bond_idx).GetBeginAtomIdx(),
                mol.GetBondWithIdx(bond_idx).GetEndAtomIdx(),
            )
        )
        for bond_idx in bond_rings[0]
    )


def _assert_saturated_monocycle_domain(mol: Chem.Mol) -> None:
    if len(Chem.GetMolFrags(mol)) != 1:
        raise NotImplementedError("saturated-monocycle oracle requires one component")
    ring_edges = frozenset(_single_ring_edges(mol))
    if not ring_edges:
        raise NotImplementedError("saturated-monocycle oracle requires a ring")
    for atom in mol.GetAtoms():
        _atom_text(atom)
    for bond in mol.GetBonds():
        edge = normalized_edge((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        if bond.GetBondType() != Chem.BondType.SINGLE:
            raise NotImplementedError(
                "saturated-monocycle oracle supports only single bonds"
            )
        if bond.IsInRing() != (edge in ring_edges):
            raise ValueError(f"inconsistent ring membership for bond {edge!r}")


def _atom_text(atom: Chem.Atom) -> str:
    symbol = atom.GetSymbol()
    if symbol in {"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"}:
        return symbol
    raise NotImplementedError(f"unsupported saturated-monocycle atom {symbol!r}")


def _bond_text(mol: Chem.Mol, begin_atom_idx: int, end_atom_idx: int) -> str:
    bond = mol.GetBondBetweenAtoms(begin_atom_idx, end_atom_idx)
    if bond is None:
        raise ValueError(f"atoms {begin_atom_idx}, {end_atom_idx} are not bonded")
    if bond.GetBondType() == Chem.BondType.SINGLE:
        return ""
    if bond.GetBondType() == Chem.BondType.DOUBLE:
        return "="
    raise NotImplementedError(
        f"unsupported saturated-monocycle bond {bond.GetBondType()}"
    )
