from __future__ import annotations

"""Test evidence helpers for expanded South Star fixture domains.

The remaining `independent_*` helpers are temporary witnesses. Helpers without
that prefix intentionally consume shared EnumS traversal/equation records so
they do not grow into separate support universes.
"""

from dataclasses import dataclass

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
from grimace._south_star.tetrahedral import IMPLICIT_HYDROGEN_LIGAND
from grimace._south_star.tetrahedral import SouthStarTetrahedralCenterFact
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


@dataclass(frozen=True, slots=True)
class SouthStarTetrahedralTraversalObligation:
    center_atom_idx: int
    source_token: str
    expected_token: str
    emitted_token: str
    emitted_ligand_order: tuple[str, ...]
    parent_atom_idx: int | None
    child_atom_indices: tuple[int, ...]

    @property
    def preserves_orientation(self) -> bool:
        return self.emitted_token == self.expected_token


@dataclass(frozen=True, slots=True)
class SouthStarTetrahedralTraversalResult:
    outputs: tuple[str, ...]
    obligations: tuple[SouthStarTetrahedralTraversalObligation, ...]


def shared_saturated_monocycle_support_for_case(
    case: SouthStarExpandedSupportCase,
) -> tuple[str, ...]:
    mol = parse_smiles(case.source_smiles)
    _assert_saturated_monocycle_domain(mol)

    return _shared_traversal_support_for_case(case)


def shared_nonstereo_monocycle_support_for_case(
    case: SouthStarExpandedSupportCase,
) -> tuple[str, ...]:
    mol = parse_smiles(case.source_smiles)
    _assert_nonstereo_monocycle_domain(mol)

    return _shared_traversal_support_for_case(case)


def _shared_traversal_support_for_case(
    case: SouthStarExpandedSupportCase,
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            traversal.render()
            for traversal in mol_to_smiles_enum_s_tree_traversals_for_case(case)
        )
    )


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


def shared_tetrahedral_atom_stereo_support_for_case(
    case: SouthStarExpandedSupportCase,
) -> SouthStarTetrahedralTraversalResult:
    """Check tetrahedral fixtures through shared traversal obligations."""
    mol = parse_smiles(case.source_smiles)
    facts = extract_tetrahedral_center_facts(mol)
    if not facts:
        raise NotImplementedError("tetrahedral traversal check requires centers")
    facts_by_atom = {fact.center_atom_idx: fact for fact in facts}
    traversals = mol_to_smiles_enum_s_tree_traversals_for_case(case)
    return SouthStarTetrahedralTraversalResult(
        outputs=tuple(dict.fromkeys(traversal.render() for traversal in traversals)),
        obligations=tuple(
            _tetrahedral_obligation_for_atom_event(
                traversal.events,
                center_atom_idx=event.atom_idx,
                emitted_token=_tetrahedral_token_from_atom_text(event.text),
                facts_by_atom=facts_by_atom,
            )
            for traversal in traversals
            for event in traversal.events
            if event.kind == "atom"
            and event.atom_idx in facts_by_atom
            and event.atom_idx is not None
        ),
    )


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


def _tetrahedral_obligation_for_atom_event(
    events: tuple[SouthStarTraversalEvent, ...],
    *,
    center_atom_idx: int,
    emitted_token: str,
    facts_by_atom: dict[int, SouthStarTetrahedralCenterFact],
) -> SouthStarTetrahedralTraversalObligation:
    fact = facts_by_atom[center_atom_idx]
    parent_by_atom = _parent_by_atom(events)
    children_by_atom = _ordered_children_by_atom(events)
    parent_idx = parent_by_atom.get(center_atom_idx)
    ordered_children = tuple(children_by_atom.get(center_atom_idx, ()))
    emitted_ligand_order = _emitted_tetrahedral_ligand_order(
        parent_idx=parent_idx,
        ordered_children=ordered_children,
        implicit_hydrogen_count=fact.implicit_hydrogen_count,
    )
    expected_token = preserving_tetrahedral_token(
        source_token=fact.source_token,
        source_ligand_order=fact.source_ligand_order,
        emitted_ligand_order=emitted_ligand_order,
    )
    if emitted_token != expected_token:
        raise ValueError(
            f"tetrahedral traversal emitted {emitted_token!r}, expected "
            f"{expected_token!r} for atom {center_atom_idx}"
        )
    return SouthStarTetrahedralTraversalObligation(
        center_atom_idx=center_atom_idx,
        source_token=fact.source_token,
        expected_token=expected_token,
        emitted_token=emitted_token,
        emitted_ligand_order=emitted_ligand_order,
        parent_atom_idx=parent_idx,
        child_atom_indices=ordered_children,
    )


def _parent_by_atom(events: tuple[SouthStarTraversalEvent, ...]) -> dict[int, int]:
    return {
        event.end_atom_idx: event.begin_atom_idx
        for event in events
        if event.kind == "bond"
        and event.begin_atom_idx is not None
        and event.end_atom_idx is not None
    }


def _ordered_children_by_atom(
    events: tuple[SouthStarTraversalEvent, ...],
) -> dict[int, tuple[int, ...]]:
    children: dict[int, list[int]] = {}
    for event in events:
        if (
            event.kind == "bond"
            and event.begin_atom_idx is not None
            and event.end_atom_idx is not None
        ):
            children.setdefault(event.begin_atom_idx, []).append(event.end_atom_idx)
    return {
        atom_idx: tuple(child_indices)
        for atom_idx, child_indices in children.items()
    }


def _tetrahedral_token_from_atom_text(atom_text: str) -> str:
    if "@@" in atom_text:
        return "@@"
    if "@" in atom_text:
        return "@"
    raise ValueError(f"tetrahedral atom text has no token: {atom_text!r}")


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


def _assert_saturated_monocycle_domain(mol: Chem.Mol) -> None:
    _assert_nonstereo_monocycle_domain(mol)
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            raise NotImplementedError(
                "saturated monocycle check supports only single bonds"
            )


def _assert_nonstereo_monocycle_domain(mol: Chem.Mol) -> None:
    if len(Chem.GetMolFrags(mol)) != 1:
        raise NotImplementedError("nonstereo monocycle check requires one component")
    if len(mol.GetRingInfo().BondRings()) != 1:
        raise NotImplementedError("nonstereo monocycle check requires one ring")
    for atom in mol.GetAtoms():
        _atom_text(atom)
    for bond in mol.GetBonds():
        if bond.GetBondType() not in {Chem.BondType.SINGLE, Chem.BondType.DOUBLE}:
            raise NotImplementedError(
                "nonstereo monocycle check supports only single and double bonds"
            )
        if bond.GetStereo() != Chem.BondStereo.STEREONONE:
            raise NotImplementedError(
                "nonstereo monocycle check does not support bond stereo"
            )
        if bond.GetBondDir() != Chem.BondDir.NONE:
            raise NotImplementedError(
                "nonstereo monocycle check does not support directional bonds"
            )


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
