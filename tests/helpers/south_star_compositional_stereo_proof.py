from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

from grimace._south_star.components import extract_south_star_components
from grimace._south_star.enum_s import mol_to_smiles_enum_s_tree_traversals_for_case
from grimace._south_star.enum_s import mol_to_smiles_enum_s_graph_native
from grimace._south_star.enum_s import render_south_star_tree_traversal
from grimace._south_star.fragments import (
    SouthStarFragmentSupport,
    compose_disconnected_fragment_supports,
)
from grimace._south_star.support_gates import south_star_support_gate_report
from grimace._south_star.tetrahedral import (
    extract_ring_tetrahedral_interaction_obligations,
    extract_tetrahedral_center_facts,
)
from tests.helpers.south_star_semantic_oracle import south_star_conformance_report


@dataclass(frozen=True, slots=True)
class SouthStarCompositionalStereoObligation:
    obligation_id: str
    family: str
    atom_indices: tuple[int, ...]
    fragment_index: int


@dataclass(frozen=True, slots=True)
class SouthStarCompositionalStereoComponent:
    component_id: str
    obligation_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    coupling_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SouthStarCompositionalStereoProofReport:
    source_smiles: str
    supported: bool
    unsupported_categories: tuple[str, ...]
    classification: str
    obligations: tuple[SouthStarCompositionalStereoObligation, ...]
    components: tuple[SouthStarCompositionalStereoComponent, ...]
    assignment_count_before_rendering: int
    proof_outputs: tuple[str, ...] | None
    proof_output_count: int | None
    runtime_output_count: int | None
    runtime_outputs_match_proof: bool | None
    semantic_parseback_passed: bool | None


@dataclass(frozen=True, slots=True)
class _SourceCase:
    source_smiles: str


def compositional_stereo_proof_report(
    source_smiles: str,
) -> SouthStarCompositionalStereoProofReport:
    mol = Chem.MolFromSmiles(source_smiles)
    if mol is None:
        raise ValueError(f"cannot parse source SMILES: {source_smiles!r}")

    gate = south_star_support_gate_report(mol)
    obligations = _obligations(mol)
    components = _components(mol, obligations)
    proof_outputs = _proof_outputs(source_smiles, mol=mol, supported=gate.supported)
    runtime_outputs = None
    semantic_parseback_passed = None
    if gate.supported:
        runtime_outputs = mol_to_smiles_enum_s_graph_native(source_smiles).outputs
        semantic_parseback_passed = all(
            south_star_conformance_report(
                source_smiles=source_smiles,
                candidate_smiles=output,
            ).accepted
            for output in runtime_outputs
        )

    return SouthStarCompositionalStereoProofReport(
        source_smiles=source_smiles,
        supported=gate.supported,
        unsupported_categories=tuple(sorted(gate.categories)),
        classification=_classification(obligations, components),
        obligations=obligations,
        components=components,
        assignment_count_before_rendering=_assignment_count(components),
        proof_outputs=proof_outputs,
        proof_output_count=None if proof_outputs is None else len(proof_outputs),
        runtime_output_count=None if runtime_outputs is None else len(runtime_outputs),
        runtime_outputs_match_proof=(
            None
            if proof_outputs is None or runtime_outputs is None
            else proof_outputs == runtime_outputs
        ),
        semantic_parseback_passed=semantic_parseback_passed,
    )


def _obligations(
    mol: Chem.Mol,
) -> tuple[SouthStarCompositionalStereoObligation, ...]:
    fragment_by_atom = _fragment_by_atom(mol)
    obligations = [
        SouthStarCompositionalStereoObligation(
            obligation_id=f"tetrahedral:{fact.center_atom_idx}",
            family="tetrahedral",
            atom_indices=(fact.center_atom_idx,),
            fragment_index=fragment_by_atom[fact.center_atom_idx],
        )
        for fact in extract_tetrahedral_center_facts(mol)
    ]
    directional_extraction = extract_south_star_components(mol)
    for component in directional_extraction.components:
        atom_indices = tuple(
            sorted(
                {
                    atom_idx
                    for feature in component.source_features
                    for edge in (
                        feature.central_bond,
                        *feature.eligible_carrier_edges,
                    )
                    for atom_idx in edge
                }
            )
        )
        obligations.append(
            SouthStarCompositionalStereoObligation(
                obligation_id=f"directional:{component.component_id}",
                family="directional",
                atom_indices=atom_indices,
                fragment_index=fragment_by_atom[atom_indices[0]],
            )
        )
    return tuple(obligations)


def _components(
    mol: Chem.Mol,
    obligations: tuple[SouthStarCompositionalStereoObligation, ...],
) -> tuple[SouthStarCompositionalStereoComponent, ...]:
    if not obligations:
        return ()

    parent = list(range(len(obligations)))
    coupling_reasons_by_root: dict[int, set[str]] = {}
    ring_tetrahedral_centers = {
        obligation.center_atom_idx
        for obligation in extract_ring_tetrahedral_interaction_obligations(mol)
    }

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int, reason: str) -> None:
        left_root = find(left)
        right_root = find(right)
        reasons = (
            coupling_reasons_by_root.pop(left_root, set())
            | coupling_reasons_by_root.pop(right_root, set())
            | {reason}
        )
        if left_root != right_root:
            parent[right_root] = left_root
        coupling_reasons_by_root[find(left)] = reasons

    for left_index, left in enumerate(obligations):
        for right_index, right in enumerate(
            obligations[left_index + 1 :],
            left_index + 1,
        ):
            reason = _coupling_reason(
                mol,
                left,
                right,
                ring_tetrahedral_centers=ring_tetrahedral_centers,
            )
            if reason is not None:
                union(left_index, right_index, reason)

    grouped: dict[int, list[SouthStarCompositionalStereoObligation]] = {}
    for index, obligation in enumerate(obligations):
        grouped.setdefault(find(index), []).append(obligation)

    return tuple(
        SouthStarCompositionalStereoComponent(
            component_id=f"component:{component_index}",
            obligation_ids=tuple(obligation.obligation_id for obligation in group),
            family_ids=tuple(sorted({obligation.family for obligation in group})),
            coupling_reasons=tuple(sorted(coupling_reasons_by_root.get(root, ()))),
        )
        for component_index, (root, group) in enumerate(sorted(grouped.items()))
    )


def _coupling_reason(
    mol: Chem.Mol,
    left: SouthStarCompositionalStereoObligation,
    right: SouthStarCompositionalStereoObligation,
    *,
    ring_tetrahedral_centers: set[int],
) -> str | None:
    if left.fragment_index != right.fragment_index:
        return None
    if left.family == "directional" or right.family == "directional":
        if _minimum_atom_distance(mol, left.atom_indices, right.atom_indices) <= 1:
            return "adjacent_directional_obligation"
        return None
    if (
        left.atom_indices[0] in ring_tetrahedral_centers
        and right.atom_indices[0] in ring_tetrahedral_centers
    ):
        return "shared_ring_tetrahedral_system"
    if _minimum_atom_distance(mol, left.atom_indices, right.atom_indices) <= 1:
        return "adjacent_tetrahedral_centers"
    return None


def _minimum_atom_distance(
    mol: Chem.Mol,
    left_atom_indices: tuple[int, ...],
    right_atom_indices: tuple[int, ...],
) -> int:
    if set(left_atom_indices) & set(right_atom_indices):
        return 0
    return min(
        len(Chem.GetShortestPath(mol, left_atom_idx, right_atom_idx)) - 1
        for left_atom_idx in left_atom_indices
        for right_atom_idx in right_atom_indices
    )


def _classification(
    obligations: tuple[SouthStarCompositionalStereoObligation, ...],
    components: tuple[SouthStarCompositionalStereoComponent, ...],
) -> str:
    if len(obligations) <= 1:
        return "single_component"
    if len(components) == len(obligations):
        return "independent_product"
    return "coupled_component"


def _assignment_count(
    components: tuple[SouthStarCompositionalStereoComponent, ...],
) -> int:
    count = 1
    for component in components:
        count *= 2 ** len(component.obligation_ids)
    return count


def _proof_outputs(
    source_smiles: str,
    *,
    mol: Chem.Mol,
    supported: bool,
) -> tuple[str, ...] | None:
    if not supported:
        return None
    fragments = tuple(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True))
    if len(fragments) > 1:
        fragment_supports = tuple(
            SouthStarFragmentSupport(
                fragment_id=f"fragment:{fragment_idx}",
                outputs=_connected_proof_outputs(
                    Chem.MolToSmiles(
                        fragment,
                        canonical=False,
                        isomericSmiles=True,
                    )
                ),
            )
            for fragment_idx, fragment in enumerate(fragments)
        )
        return compose_disconnected_fragment_supports(fragment_supports).outputs
    return _connected_proof_outputs(source_smiles)


def _connected_proof_outputs(source_smiles: str) -> tuple[str, ...]:
    traversals = mol_to_smiles_enum_s_tree_traversals_for_case(
        _SourceCase(source_smiles=source_smiles)
    )
    return tuple(
        dict.fromkeys(
            render_south_star_tree_traversal(traversal)
            for traversal in traversals
        )
    )


def _fragment_by_atom(mol: Chem.Mol) -> dict[int, int]:
    return {
        atom_idx: fragment_index
        for fragment_index, atom_indices in enumerate(Chem.GetMolFrags(mol))
        for atom_idx in atom_indices
    }
