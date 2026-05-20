from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from rdkit import Chem

from tests.helpers.south_star_component_support_state import (
    SouthStarComponentComplexitySnapshot,
    SouthStarComponentMarkerAssignment,
    SouthStarComponentSupportState,
)
from tests.helpers.south_star_annotation_policy import Edge, normalized_edge
from tests.helpers.south_star_semantic_oracle import semantic_oracle_accepts
from tests.helpers.south_star_semantic_oracle import parse_smiles
from tests.helpers.south_star_semantics import SouthStarSemanticCase


@dataclass(frozen=True, slots=True)
class SouthStarEnumSPrototypeResult:
    case_id: str
    outputs: tuple[str, ...]
    complexity_snapshot: SouthStarComponentComplexitySnapshot
    generation_basis: str


def mol_to_smiles_enum_s_prototype_for_case(
    case: SouthStarSemanticCase,
) -> SouthStarEnumSPrototypeResult:
    state = SouthStarComponentSupportState.from_case(case)
    _assert_fixture_carriers_are_supported(case, state)

    outputs = tuple(
        output
        for output in case.positive_semantic_smiles
        if semantic_oracle_accepts(
            source_smiles=case.source_smiles,
            candidate_smiles=output,
        )
    )
    if len(outputs) != len(case.positive_semantic_smiles):
        raise AssertionError(
            f"South Star fixture {case.case_id!r} contains semantic-positive "
            "outputs that the semantic oracle rejects"
        )

    return SouthStarEnumSPrototypeResult(
        case_id=case.case_id,
        outputs=outputs,
        complexity_snapshot=state.complexity_snapshot(),
        generation_basis="south_star_semantic_fixture_witnesses",
    )


def mol_to_smiles_enum_s_graph_native_for_case(
    case: SouthStarSemanticCase,
) -> SouthStarEnumSPrototypeResult:
    mol = parse_smiles(case.source_smiles)
    state = SouthStarComponentSupportState.from_mol(mol)
    outputs = tuple(
        dict.fromkeys(
            _emit_smiles_for_marker_assignment(
                mol,
                marker_by_edge=marker_by_edge,
            )
            for marker_by_edge in _combined_marker_assignments(state)
        )
    )
    return SouthStarEnumSPrototypeResult(
        case_id=case.case_id,
        outputs=outputs,
        complexity_snapshot=state.complexity_snapshot(),
        generation_basis="south_star_graph_native_seed_traversal",
    )


def _assert_fixture_carriers_are_supported(
    case: SouthStarSemanticCase,
    state: SouthStarComponentSupportState,
) -> None:
    for edge in case.eligible_carrier_edges:
        if not state.allowed_directional_markers(edge=edge):
            raise AssertionError(
                f"South Star fixture {case.case_id!r} carrier edge {edge!r} "
                "has no component-supported directional marker"
            )


def _combined_marker_assignments(
    state: SouthStarComponentSupportState,
) -> tuple[dict[Edge, str], ...]:
    per_component = state.component_marker_assignments()
    if not per_component:
        return ({},)

    combined = []
    for assignment_group in product(*per_component):
        marker_by_edge: dict[Edge, str] = {}
        for assignment in assignment_group:
            _merge_assignment_markers(marker_by_edge, assignment)
        combined.append(marker_by_edge)
    return tuple(combined)


def _merge_assignment_markers(
    marker_by_edge: dict[Edge, str],
    assignment: SouthStarComponentMarkerAssignment,
) -> None:
    for edge, marker in assignment.marker_by_edge:
        existing = marker_by_edge.setdefault(edge, marker)
        if existing != marker:
            raise ValueError(
                f"conflicting graph-native marker assignment for edge {edge!r}"
            )


def _emit_smiles_for_marker_assignment(
    mol: Chem.Mol,
    *,
    marker_by_edge: dict[Edge, str],
) -> str:
    _assert_seed_traversal_supported(mol)
    visited: set[int] = set()
    return _emit_atom_subtree(
        mol,
        atom_idx=0,
        parent_idx=None,
        visited=visited,
        marker_by_edge=marker_by_edge,
    )


def _assert_seed_traversal_supported(mol: Chem.Mol) -> None:
    if mol.GetNumAtoms() == 0:
        raise ValueError("South Star graph-native seed traversal requires atoms")
    if mol.GetNumBonds() != mol.GetNumAtoms() - 1:
        raise NotImplementedError(
            "South Star graph-native seed traversal currently requires one "
            "connected acyclic component"
        )


def _emit_atom_subtree(
    mol: Chem.Mol,
    *,
    atom_idx: int,
    parent_idx: int | None,
    visited: set[int],
    marker_by_edge: dict[Edge, str],
) -> str:
    visited.add(atom_idx)
    atom_text = _atom_text(mol.GetAtomWithIdx(atom_idx))
    children = tuple(
        neighbor.GetIdx()
        for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
        if neighbor.GetIdx() != parent_idx and neighbor.GetIdx() not in visited
    )
    ordered_children = tuple(sorted(children))
    if not ordered_children:
        return atom_text

    pieces = [atom_text]
    branch_children = ordered_children[:-1]
    main_child = ordered_children[-1]
    for child_idx in branch_children:
        pieces.append(
            "("
            + _bond_text(mol, atom_idx, child_idx, marker_by_edge=marker_by_edge)
            + _emit_atom_subtree(
                mol,
                atom_idx=child_idx,
                parent_idx=atom_idx,
                visited=visited,
                marker_by_edge=marker_by_edge,
            )
            + ")"
        )
    pieces.append(
        _bond_text(mol, atom_idx, main_child, marker_by_edge=marker_by_edge)
        + _emit_atom_subtree(
            mol,
            atom_idx=main_child,
            parent_idx=atom_idx,
            visited=visited,
            marker_by_edge=marker_by_edge,
        )
    )
    return "".join(pieces)


def _atom_text(atom: Chem.Atom) -> str:
    symbol = atom.GetSymbol()
    if symbol in {"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"}:
        return symbol
    raise NotImplementedError(
        f"South Star graph-native seed traversal does not support atom {symbol!r}"
    )


def _bond_text(
    mol: Chem.Mol,
    begin_atom_idx: int,
    end_atom_idx: int,
    *,
    marker_by_edge: dict[Edge, str],
) -> str:
    edge = normalized_edge((begin_atom_idx, end_atom_idx))
    bond = mol.GetBondBetweenAtoms(begin_atom_idx, end_atom_idx)
    if bond is None:
        raise ValueError(f"edge {edge!r} is not a bond")
    if edge in marker_by_edge:
        return marker_by_edge[edge]
    if bond.GetBondType() == Chem.BondType.DOUBLE:
        return "="
    if bond.GetBondType() == Chem.BondType.SINGLE:
        return ""
    raise NotImplementedError(
        f"South Star graph-native seed traversal does not support bond "
        f"{bond.GetBondType()}"
    )
