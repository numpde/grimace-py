from __future__ import annotations

"""Test evidence helpers for expanded South Star fixture domains.

The remaining `independent_*` helpers are temporary witnesses. Helpers without
that prefix intentionally consume shared EnumS traversal/equation records so
they do not grow into separate support universes.

The `TemporarySouthStar*Witness*` records below are fixture evidence envelopes,
not reference-model vocabulary. Shared constraint-family records live under
`grimace._south_star`.
"""

from dataclasses import dataclass

from rdkit import Chem

from grimace._south_star.annotation_policy import Edge
from grimace._south_star.annotation_policy import normalized_edge
from grimace._south_star.component_support_state import (
    SouthStarComponentSupportState,
)
from grimace._south_star.enum_s import (
    SouthStarFragmentGenerationRecord,
    mol_to_smiles_enum_s_graph_native,
    mol_to_smiles_enum_s_tree_traversals_for_case,
    render_south_star_tree_traversal,
)
from grimace._south_star.marker_equations import SouthStarMarkerSlotParityEquation
from grimace._south_star.marker_equations import (
    marker_slot_parity_equations_for_traversal,
)
from grimace._south_star.reference_model import SouthStarConnectedGraphTraversalPlan
from grimace._south_star.tetrahedral import SouthStarTetrahedralCenterFact
from grimace._south_star.tetrahedral import (
    SouthStarTetrahedralTraversalTokenDiagnostic,
)
from grimace._south_star.tetrahedral import extract_tetrahedral_center_facts
from grimace._south_star.tetrahedral import (
    tetrahedral_traversal_observation_from_connected_graph_plan,
)
from grimace._south_star.tetrahedral import tetrahedral_traversal_token_diagnostic
from tests.helpers.south_star_exact_support import (
    SouthStarExpandedSupportCase,
)
from tests.helpers.south_star_semantic_oracle import parse_smiles


@dataclass(frozen=True, slots=True)
class TemporarySouthStarRingStereoWitnessResult:
    outputs: tuple[str, ...]
    equations: tuple[SouthStarMarkerSlotParityEquation, ...]
    closure_edge_count: int
    marker_assignment_count: int


@dataclass(frozen=True, slots=True)
class TemporarySouthStarTetrahedralWitnessResult:
    outputs: tuple[str, ...]
    diagnostics: tuple[SouthStarTetrahedralTraversalTokenDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class TemporarySouthStarDisconnectedCompositionWitnessEvidence:
    outputs: tuple[str, ...]
    fragment_count: int
    fragment_output_counts: tuple[int, ...]
    fragment_generation_records: tuple[SouthStarFragmentGenerationRecord, ...]
    fragment_order_policy: str
    fragment_order_count: int
    estimated_product_size: int


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
            render_south_star_tree_traversal(traversal)
            for traversal in mol_to_smiles_enum_s_tree_traversals_for_case(case)
        )
    )


def shared_disconnected_composition_support_for_case(
    case: SouthStarExpandedSupportCase,
) -> TemporarySouthStarDisconnectedCompositionWitnessEvidence:
    result = mol_to_smiles_enum_s_graph_native(
        case.source_smiles,
        case_id=case.case_id,
    )
    diagnostics = result.generation_diagnostics
    if diagnostics is None:
        raise ValueError("disconnected composition evidence requires diagnostics")
    return TemporarySouthStarDisconnectedCompositionWitnessEvidence(
        outputs=result.outputs,
        fragment_count=diagnostics.fragment_count,
        fragment_output_counts=diagnostics.fragment_output_counts,
        fragment_generation_records=diagnostics.fragment_generation_records,
        fragment_order_policy=result.fragment_order_policy,
        fragment_order_count=diagnostics.fragment_order_count,
        estimated_product_size=diagnostics.estimated_product_size,
    )


def shared_ring_stereo_monocycle_support_for_case(
    case: SouthStarExpandedSupportCase,
) -> TemporarySouthStarRingStereoWitnessResult:
    """Check ring-stereo fixtures through shared traversal/equation records."""
    mol = parse_smiles(case.source_smiles)
    _assert_ring_stereo_monocycle_domain(mol)
    state = SouthStarComponentSupportState.from_mol(mol)
    traversals = mol_to_smiles_enum_s_tree_traversals_for_case(case)

    return TemporarySouthStarRingStereoWitnessResult(
        outputs=tuple(
            dict.fromkeys(
                render_south_star_tree_traversal(traversal)
                for traversal in traversals
            )
        ),
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
) -> TemporarySouthStarTetrahedralWitnessResult:
    """Check tetrahedral fixtures through shared traversal-token diagnostics."""
    mol = parse_smiles(case.source_smiles)
    facts = extract_tetrahedral_center_facts(mol)
    if not facts:
        raise NotImplementedError("tetrahedral traversal check requires centers")
    facts_by_atom = {fact.center_atom_idx: fact for fact in facts}
    traversals = mol_to_smiles_enum_s_tree_traversals_for_case(case)
    return TemporarySouthStarTetrahedralWitnessResult(
        outputs=tuple(
            dict.fromkeys(
                render_south_star_tree_traversal(traversal)
                for traversal in traversals
            )
        ),
        diagnostics=tuple(
            _tetrahedral_token_diagnostic_for_atom_event(
                traversal.connected_graph_plan,
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


def _tetrahedral_token_diagnostic_for_atom_event(
    connected_graph_plan: SouthStarConnectedGraphTraversalPlan | None,
    *,
    center_atom_idx: int,
    emitted_token: str,
    facts_by_atom: dict[int, SouthStarTetrahedralCenterFact],
) -> SouthStarTetrahedralTraversalTokenDiagnostic:
    fact = facts_by_atom[center_atom_idx]
    if connected_graph_plan is None:
        raise ValueError("tetrahedral diagnostic requires connected graph plan")
    observation = tetrahedral_traversal_observation_from_connected_graph_plan(
        connected_graph_plan,
        center_atom_idx=center_atom_idx,
        implicit_hydrogen_count=fact.implicit_hydrogen_count,
    )
    diagnostic = tetrahedral_traversal_token_diagnostic(
        fact,
        observation,
        emitted_token=emitted_token,
    )
    if not diagnostic.preserves_orientation:
        raise ValueError(
            f"tetrahedral traversal emitted {diagnostic.emitted_token!r}, "
            f"expected {diagnostic.expected_token!r} for atom {center_atom_idx}"
        )
    return diagnostic


def _tetrahedral_token_from_atom_text(atom_text: str) -> str:
    if "@@" in atom_text:
        return "@@"
    if "@" in atom_text:
        return "@"
    raise ValueError(f"tetrahedral atom text has no token: {atom_text!r}")


def _assert_ring_stereo_monocycle_domain(mol: Chem.Mol) -> None:
    if len(Chem.GetMolFrags(mol)) != 1:
        raise NotImplementedError("ring-stereo oracle requires one component")
    if len(mol.GetRingInfo().BondRings()) != 1:
        raise NotImplementedError("ring-stereo oracle requires one ring")


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
