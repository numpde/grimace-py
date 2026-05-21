from __future__ import annotations

"""Independent test oracles for expanded South Star fixture domains.

These helpers intentionally keep independent traversal searches and renderers
instead of calling `grimace._south_star.enum_s`. They consume the shared South
Star reference record vocabulary where possible so they remain witnesses, not
separate mini-worlds with their own event/slot types.
"""

from dataclasses import dataclass
from itertools import permutations
from itertools import product

from rdkit import Chem

from grimace._south_star.annotation_policy import Edge
from grimace._south_star.annotation_policy import normalized_edge
from grimace._south_star.component_support_state import (
    SouthStarComponentSupportState,
)
from grimace._south_star.enum_s import (
    mol_to_smiles_enum_s_tree_traversals_for_case,
)
from grimace._south_star.fragments import SouthStarDisconnectedCompositionResult
from grimace._south_star.fragments import SouthStarFragmentSupport
from grimace._south_star.fragments import compose_disconnected_fragment_supports
from grimace._south_star.marker_equations import SouthStarMarkerSlotParityEquation
from grimace._south_star.marker_equations import (
    marker_slot_parity_equations_for_traversal,
)
from grimace._south_star.reference_model import SouthStarTraversalEvent
from grimace._south_star.reference_model import SouthStarTraversalFragment
from grimace._south_star.tetrahedral import IMPLICIT_HYDROGEN_LIGAND
from grimace._south_star.tetrahedral import extract_tetrahedral_center_facts
from grimace._south_star.tetrahedral import preserving_tetrahedral_token
from tests.helpers.south_star_exact_support import (
    SouthStarExpandedSupportCase,
    load_south_star_exact_first_domain_cases,
    load_south_star_expanded_support_cases,
)
from tests.helpers.south_star_semantic_oracle import parse_smiles


@dataclass(frozen=True, slots=True)
class SouthStarRingStereoOracleResult:
    outputs: tuple[str, ...]
    equations: tuple[SouthStarMarkerSlotParityEquation, ...]
    closure_edge_count: int
    marker_assignment_count: int


def independent_saturated_monocycle_support_for_case(
    case: SouthStarExpandedSupportCase,
) -> tuple[str, ...]:
    """Enumerate saturated-monocycle support without using EnumS traversal."""
    mol = parse_smiles(case.source_smiles)
    _assert_saturated_monocycle_domain(mol)

    return _independent_nonstereo_monocycle_support_for_mol(mol)


def independent_nonstereo_monocycle_support_for_case(
    case: SouthStarExpandedSupportCase,
) -> tuple[str, ...]:
    """Enumerate nonstereo-monocycle support from shared traversal records."""
    mol = parse_smiles(case.source_smiles)
    _assert_nonstereo_monocycle_domain(mol)

    return _independent_nonstereo_monocycle_support_for_mol(mol)


def _independent_nonstereo_monocycle_support_for_mol(
    mol: Chem.Mol,
) -> tuple[str, ...]:
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
                outputs.append(_render_with_ring_digit(mol, fragment, closure_edge))
    return tuple(dict.fromkeys(outputs))


def independent_disconnected_composition_support_for_case(
    case: SouthStarExpandedSupportCase,
) -> SouthStarDisconnectedCompositionResult:
    fragment_supports = _independent_fragment_supports_for_case(case)
    return compose_disconnected_fragment_supports(
        tuple(
            SouthStarFragmentSupport(
                fragment_id=f"fragment:{fragment_idx}",
                outputs=support,
            )
            for fragment_idx, support in enumerate(fragment_supports)
        )
    )


def shared_ring_stereo_monocycle_support_for_case(
    case: SouthStarExpandedSupportCase,
) -> SouthStarRingStereoOracleResult:
    """Check ring-stereo fixtures through shared traversal/equation records."""
    mol = parse_smiles(case.source_smiles)
    _assert_ring_stereo_monocycle_domain(mol)
    state = SouthStarComponentSupportState.from_mol(mol)
    traversals = mol_to_smiles_enum_s_tree_traversals_for_case(case)

    return SouthStarRingStereoOracleResult(
        outputs=tuple(dict.fromkeys(traversal.render() for traversal in traversals)),
        equations=tuple(
            dict.fromkeys(
                equation
                for traversal in traversals
                for equation in marker_slot_parity_equations_for_traversal(
                    state,
                    traversal,
                )
            )
        ),
        closure_edge_count=len(
            {
                normalized_edge(event.edge)
                for traversal in traversals
                for event in traversal.events
                if event.ring_closure is not None
                and event.ring_closure.role == "open"
                and event.edge is not None
            }
        ),
        marker_assignment_count=state.complexity_snapshot().estimated_product_size,
    )


def independent_tetrahedral_atom_stereo_support_for_case(
    case: SouthStarExpandedSupportCase,
) -> tuple[str, ...]:
    """Enumerate current star-shaped tetrahedral support independently.

    This oracle is intentionally limited to one tetrahedral center whose
    explicit ligands are terminal atoms. It derives `@`/`@@` from emitted ligand
    order and does not call EnumS traversal or rendering.
    """
    mol = parse_smiles(case.source_smiles)
    fact = _single_star_tetrahedral_fact(mol)
    center_idx = fact.center_atom_idx
    neighbor_indices = fact.explicit_neighbor_atom_indices

    outputs: list[str] = []
    for root_idx in range(mol.GetNumAtoms()):
        if root_idx == center_idx:
            for ordered_neighbors in permutations(neighbor_indices):
                outputs.append(
                    _render_tetrahedral_center_root(
                        mol,
                        center_idx=center_idx,
                        ordered_neighbors=ordered_neighbors,
                    )
                )
            continue
        if root_idx not in neighbor_indices:
            raise NotImplementedError("tetrahedral oracle requires a star graph")
        child_indices = tuple(idx for idx in neighbor_indices if idx != root_idx)
        for ordered_children in permutations(child_indices):
            outputs.append(
                _render_tetrahedral_ligand_root(
                    mol,
                    root_idx=root_idx,
                    center_idx=center_idx,
                    ordered_children=ordered_children,
                )
            )
    return tuple(dict.fromkeys(outputs))


def _independent_fragment_supports_for_case(
    case: SouthStarExpandedSupportCase,
) -> tuple[tuple[str, ...], ...]:
    if case.case_id == "markerless_disconnected_ring_and_atom":
        return (
            _expanded_support("simple_saturated_monocycle_cyclohexane"),
            ("O",),
        )
    if case.case_id == "disconnected_stereo_fragment_and_atom":
        return (
            _exact_first_domain_support("isolated_alkene_z"),
            ("O",),
        )
    raise NotImplementedError(
        f"no disconnected-composition oracle fragment supports for {case.case_id!r}"
    )


def _single_star_tetrahedral_fact(mol: Chem.Mol):
    facts = extract_tetrahedral_center_facts(mol)
    if len(facts) != 1:
        raise NotImplementedError("tetrahedral oracle requires exactly one center")
    fact = facts[0]
    center_idx = fact.center_atom_idx
    for neighbor_idx in fact.explicit_neighbor_atom_indices:
        neighbor = mol.GetAtomWithIdx(neighbor_idx)
        if neighbor.GetDegree() != 1:
            raise NotImplementedError(
                "tetrahedral oracle currently requires terminal explicit ligands"
            )
        if mol.GetBondBetweenAtoms(center_idx, neighbor_idx) is None:
            raise ValueError("tetrahedral ligand is not bonded to the center")
    return fact


def _render_tetrahedral_center_root(
    mol: Chem.Mol,
    *,
    center_idx: int,
    ordered_neighbors: tuple[int, ...],
) -> str:
    center_text = _tetrahedral_center_text(
        mol,
        center_idx=center_idx,
        parent_idx=None,
        ordered_children=ordered_neighbors,
    )
    return _render_center_with_ordered_ligands(
        mol,
        center_text=center_text,
        ordered_ligands=ordered_neighbors,
    )


def _render_tetrahedral_ligand_root(
    mol: Chem.Mol,
    *,
    root_idx: int,
    center_idx: int,
    ordered_children: tuple[int, ...],
) -> str:
    center_text = _tetrahedral_center_text(
        mol,
        center_idx=center_idx,
        parent_idx=root_idx,
        ordered_children=ordered_children,
    )
    return _atom_text(mol.GetAtomWithIdx(root_idx)) + _render_center_with_ordered_ligands(
        mol,
        center_text=center_text,
        ordered_ligands=ordered_children,
    )


def _render_center_with_ordered_ligands(
    mol: Chem.Mol,
    *,
    center_text: str,
    ordered_ligands: tuple[int, ...],
) -> str:
    if not ordered_ligands:
        return center_text
    branch_text = "".join(
        f"({_atom_text(mol.GetAtomWithIdx(atom_idx))})"
        for atom_idx in ordered_ligands[:-1]
    )
    return center_text + branch_text + _atom_text(
        mol.GetAtomWithIdx(ordered_ligands[-1])
    )


def _tetrahedral_center_text(
    mol: Chem.Mol,
    *,
    center_idx: int,
    parent_idx: int | None,
    ordered_children: tuple[int, ...],
) -> str:
    fact = _single_star_tetrahedral_fact(mol)
    if fact.center_atom_idx != center_idx:
        raise ValueError("tetrahedral center index mismatch")
    emitted_ligand_order = _emitted_tetrahedral_ligand_order(
        parent_idx=parent_idx,
        ordered_children=ordered_children,
        implicit_hydrogen_count=fact.implicit_hydrogen_count,
    )
    token = preserving_tetrahedral_token(
        source_token=fact.source_token,
        source_ligand_order=fact.source_ligand_order,
        emitted_ligand_order=emitted_ligand_order,
    )
    hydrogen_text = "H" if fact.implicit_hydrogen_count else ""
    return f"[{mol.GetAtomWithIdx(center_idx).GetSymbol()}{token}{hydrogen_text}]"


def _emitted_tetrahedral_ligand_order(
    *,
    parent_idx: int | None,
    ordered_children: tuple[int, ...],
    implicit_hydrogen_count: int,
) -> tuple[str, ...]:
    emitted = []
    if parent_idx is None and implicit_hydrogen_count:
        emitted.append(IMPLICIT_HYDROGEN_LIGAND)
    if parent_idx is not None:
        emitted.append(f"atom:{parent_idx}")
    emitted.extend(f"atom:{child_idx}" for child_idx in ordered_children)
    if parent_idx is not None and implicit_hydrogen_count:
        emitted.append(IMPLICIT_HYDROGEN_LIGAND)
    return tuple(emitted)


def _assert_ring_stereo_monocycle_domain(mol: Chem.Mol) -> None:
    if len(Chem.GetMolFrags(mol)) != 1:
        raise NotImplementedError("ring-stereo oracle requires one component")
    if len(mol.GetRingInfo().BondRings()) != 1:
        raise NotImplementedError("ring-stereo oracle requires one ring")


def _expanded_support(case_id: str) -> tuple[str, ...]:
    return next(
        case.expected_support
        for case in load_south_star_expanded_support_cases()
        if case.case_id == case_id
    )


def _exact_first_domain_support(case_id: str) -> tuple[str, ...]:
    return next(
        case.expected_support
        for case in load_south_star_exact_first_domain_cases()
        if case.case_id == case_id
    )


def _tree_fragments_with_blocked_edge(
    mol: Chem.Mol,
    *,
    atom_idx: int,
    parent_idx: int | None,
    visited: frozenset[int],
    blocked_edge: Edge,
) -> tuple[SouthStarTraversalFragment, ...]:
    visited = visited | {atom_idx}
    children = tuple(
        neighbor.GetIdx()
        for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
        if neighbor.GetIdx() != parent_idx
        and neighbor.GetIdx() not in visited
        and normalized_edge((atom_idx, neighbor.GetIdx())) != blocked_edge
    )
    atom_event = SouthStarTraversalEvent(
        kind="atom",
        text=_atom_text(mol.GetAtomWithIdx(atom_idx)),
        atom_idx=atom_idx,
    )
    if not children:
        return (SouthStarTraversalFragment(events=(atom_event,)),)

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
            branch_events = tuple(
                event for branch in branch_group for event in branch.events
            )
            for main_fragment in main_fragments:
                fragments.append(
                    SouthStarTraversalFragment(
                        events=(atom_event,) + branch_events + main_fragment.events,
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
) -> tuple[SouthStarTraversalFragment, ...]:
    bond_event = _plain_ring_bond_event(
        mol,
        begin_atom_idx=begin_atom_idx,
        end_atom_idx=end_atom_idx,
    )
    return tuple(
        SouthStarTraversalFragment(
            events=(
                SouthStarTraversalEvent(kind="branch_open", text="("),
                bond_event,
            )
            + child_fragment.events
            + (SouthStarTraversalEvent(kind="branch_close", text=")"),)
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
) -> tuple[SouthStarTraversalFragment, ...]:
    bond_event = _plain_ring_bond_event(
        mol,
        begin_atom_idx=begin_atom_idx,
        end_atom_idx=end_atom_idx,
    )
    return tuple(
        SouthStarTraversalFragment(events=(bond_event,) + child_fragment.events)
        for child_fragment in _tree_fragments_with_blocked_edge(
            mol,
            atom_idx=end_atom_idx,
            parent_idx=begin_atom_idx,
            visited=visited,
            blocked_edge=blocked_edge,
        )
    )


def _render_with_ring_digit(
    mol: Chem.Mol,
    fragment: SouthStarTraversalFragment,
    closure_edge: Edge,
) -> str:
    rendered = []
    endpoint_count = 0
    for event in fragment.events:
        rendered.append(event.text)
        if event.atom_idx in closure_edge:
            endpoint_count += 1
            if endpoint_count == 1:
                rendered.append(_bond_text(mol, *closure_edge))
            rendered.append("1")
    if endpoint_count != 2:
        raise ValueError(
            f"closure edge {closure_edge!r} endpoints were not both rendered"
        )
    return "".join(rendered)


def _plain_ring_bond_event(
    mol: Chem.Mol,
    *,
    begin_atom_idx: int,
    end_atom_idx: int,
) -> SouthStarTraversalEvent:
    edge = normalized_edge((begin_atom_idx, end_atom_idx))
    return SouthStarTraversalEvent(
        kind="bond",
        text=_bond_text(mol, begin_atom_idx, end_atom_idx),
        edge=edge,
        begin_atom_idx=begin_atom_idx,
        end_atom_idx=end_atom_idx,
    )


def _single_ring_edges(mol: Chem.Mol) -> tuple[Edge, ...]:
    bond_rings = mol.GetRingInfo().BondRings()
    if len(bond_rings) != 1:
        raise ValueError("nonstereo-monocycle witness expects one ring")
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
    _assert_nonstereo_monocycle_domain(mol)
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            raise NotImplementedError(
                "saturated-monocycle oracle supports only single bonds"
            )


def _assert_nonstereo_monocycle_domain(mol: Chem.Mol) -> None:
    if len(Chem.GetMolFrags(mol)) != 1:
        raise NotImplementedError("nonstereo-monocycle witness requires one component")
    ring_edges = frozenset(_single_ring_edges(mol))
    if not ring_edges:
        raise NotImplementedError("nonstereo-monocycle witness requires a ring")
    for atom in mol.GetAtoms():
        _atom_text(atom)
    for bond in mol.GetBonds():
        edge = normalized_edge((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        if bond.GetBondType() not in {Chem.BondType.SINGLE, Chem.BondType.DOUBLE}:
            raise NotImplementedError(
                "nonstereo-monocycle witness supports only single and double bonds"
            )
        if bond.GetStereo() != Chem.BondStereo.STEREONONE:
            raise NotImplementedError(
                "nonstereo-monocycle witness does not support bond stereo"
            )
        if bond.GetBondDir() != Chem.BondDir.NONE:
            raise NotImplementedError(
                "nonstereo-monocycle witness does not support directional bonds"
            )
        if bond.IsInRing() != (edge in ring_edges):
            raise ValueError(f"inconsistent ring membership for bond {edge!r}")


def _atom_text(atom: Chem.Atom) -> str:
    if atom.GetIsAromatic():
        raise NotImplementedError(
            "nonstereo-monocycle witness does not support aromatic atoms"
        )
    if atom.GetFormalCharge() != 0:
        raise NotImplementedError(
            "nonstereo-monocycle witness does not support charged atoms"
        )
    if atom.GetIsotope() != 0:
        raise NotImplementedError(
            "nonstereo-monocycle witness does not support isotopic atoms"
        )
    if atom.GetNumRadicalElectrons() != 0:
        raise NotImplementedError(
            "nonstereo-monocycle witness does not support radical atoms"
        )
    symbol = atom.GetSymbol()
    if symbol in {"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"}:
        return symbol
    raise NotImplementedError(f"unsupported nonstereo-monocycle atom {symbol!r}")


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


def _flipped_marker(marker: str) -> str:
    if marker == "/":
        return "\\"
    if marker == "\\":
        return "/"
    raise ValueError(f"unsupported directional marker {marker!r}")
