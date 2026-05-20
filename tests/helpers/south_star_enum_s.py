from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
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
    candidate_outputs = tuple(
        dict.fromkeys(
            output
            for marker_by_edge in _combined_marker_assignments(state)
            for output in _emit_all_tree_smiles_for_marker_assignment(
                mol,
                marker_by_edge=marker_by_edge,
            )
        )
    )
    outputs = tuple(
        output
        for output in candidate_outputs
        if semantic_oracle_accepts(
            source_smiles=case.source_smiles,
            candidate_smiles=output,
        )
    )
    return SouthStarEnumSPrototypeResult(
        case_id=case.case_id,
        outputs=outputs,
        complexity_snapshot=state.complexity_snapshot(),
        generation_basis="south_star_graph_native_tree_traversal_semantic_filter",
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


def _emit_all_tree_smiles_for_marker_assignment(
    mol: Chem.Mol,
    *,
    marker_by_edge: dict[Edge, str],
) -> tuple[str, ...]:
    _assert_tree_traversal_supported(mol)
    return tuple(
        output
        for root_idx in range(mol.GetNumAtoms())
        for output in _emit_atom_subtree_variants(
            mol,
            atom_idx=root_idx,
            parent_idx=None,
            visited=frozenset(),
            marker_by_edge=marker_by_edge,
        )
    )


def _assert_tree_traversal_supported(mol: Chem.Mol) -> None:
    if mol.GetNumAtoms() == 0:
        raise ValueError("South Star graph-native tree traversal requires atoms")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise NotImplementedError(
            "South Star graph-native tree traversal currently requires one "
            "connected component"
        )
    if mol.GetNumBonds() != mol.GetNumAtoms() - 1:
        raise NotImplementedError(
            "South Star graph-native tree traversal currently requires one "
            "connected acyclic component"
        )


def _emit_atom_subtree_variants(
    mol: Chem.Mol,
    *,
    atom_idx: int,
    parent_idx: int | None,
    visited: frozenset[int],
    marker_by_edge: dict[Edge, str],
) -> tuple[str, ...]:
    visited = visited | {atom_idx}
    atom_text = _atom_text(mol.GetAtomWithIdx(atom_idx))
    children = tuple(
        neighbor.GetIdx()
        for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
        if neighbor.GetIdx() != parent_idx and neighbor.GetIdx() not in visited
    )
    if not children:
        return (atom_text,)

    outputs: list[str] = []
    for ordered_children in permutations(sorted(children)):
        branch_children = ordered_children[:-1]
        main_child = ordered_children[-1]
        branch_variants = tuple(
            _prefixed_branch_variants(
                mol,
                parent_idx=atom_idx,
                child_idx=child_idx,
                visited=visited,
                marker_by_edge=marker_by_edge,
            )
            for child_idx in branch_children
        )
        main_variants = _prefixed_child_variants(
            mol,
            parent_idx=atom_idx,
            child_idx=main_child,
            visited=visited,
            marker_by_edge=marker_by_edge,
        )
        for branch_group in product(*branch_variants):
            for main_text in main_variants:
                outputs.append(atom_text + "".join(branch_group) + main_text)
    return tuple(dict.fromkeys(outputs))


def _prefixed_branch_variants(
    mol: Chem.Mol,
    *,
    parent_idx: int,
    child_idx: int,
    visited: frozenset[int],
    marker_by_edge: dict[Edge, str],
) -> tuple[str, ...]:
    return tuple(
        "("
        + _bond_text(mol, parent_idx, child_idx, marker_by_edge=marker_by_edge)
        + child_text
        + ")"
        for child_text in _emit_atom_subtree_variants(
            mol,
            atom_idx=child_idx,
            parent_idx=parent_idx,
            visited=visited,
            marker_by_edge=marker_by_edge,
        )
    )


def _prefixed_child_variants(
    mol: Chem.Mol,
    *,
    parent_idx: int,
    child_idx: int,
    visited: frozenset[int],
    marker_by_edge: dict[Edge, str],
) -> tuple[str, ...]:
    return tuple(
        _bond_text(mol, parent_idx, child_idx, marker_by_edge=marker_by_edge)
        + child_text
        for child_text in _emit_atom_subtree_variants(
            mol,
            atom_idx=child_idx,
            parent_idx=parent_idx,
            visited=visited,
            marker_by_edge=marker_by_edge,
        )
    )


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
