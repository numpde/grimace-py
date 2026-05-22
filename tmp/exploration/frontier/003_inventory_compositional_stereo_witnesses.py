"""Inventory candidate witnesses for South Star compositional stereo scaling."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

from grimace._south_star.components import extract_south_star_components
from grimace._south_star.enum_s import mol_to_smiles_enum_s_graph_native
from grimace._south_star.support_gates import south_star_support_gate_report
from grimace._south_star.tetrahedral import (
    extract_ring_tetrahedral_interaction_obligations,
    extract_tetrahedral_center_facts,
)
from tests.helpers.south_star_compositional_stereo_proof import (
    SouthStarCompositionalStereoComponent,
    SouthStarCompositionalStereoProofReport,
    compositional_stereo_proof_report,
)


@dataclass(frozen=True, slots=True)
class Candidate:
    case_id: str
    source_smiles: str
    intended_classification: str


CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        case_id="acyclic_two_tetra_separated",
        source_smiles="F[C@H](Cl)C[C@H](Br)I",
        intended_classification="independent_product_candidate",
    ),
    Candidate(
        case_id="acyclic_two_tetra_adjacent",
        source_smiles="F[C@H](Cl)[C@H](Br)I",
        intended_classification="coupled_component_candidate",
    ),
    Candidate(
        case_id="disconnected_two_tetra",
        source_smiles="F[C@H](Cl)Br.F[C@H](Cl)I",
        intended_classification="independent_product_candidate",
    ),
    Candidate(
        case_id="directional_tetrahedral_acyclic_existing",
        source_smiles="F/C=C/[C@H](Cl)Br",
        intended_classification="existing_mixed_component_baseline",
    ),
    Candidate(
        case_id="independent_directional_diene_existing",
        source_smiles="F/C=C/C/C=C/Cl",
        intended_classification="existing_independent_directional_baseline",
    ),
    Candidate(
        case_id="monocycle_ring_tetra_directional_existing",
        source_smiles="F[C@H]1CCCC(/C=C/Cl)C1",
        intended_classification="existing_ring_directional_tetra_baseline",
    ),
    Candidate(
        case_id="ring_tetra_plus_branch_tetra",
        source_smiles="F[C@H]1CCCC([C@H](Cl)Br)C1",
        intended_classification="coupled_or_shared_ring_ligand_candidate",
    ),
    Candidate(
        case_id="polycyclic_ring_tetra_plus_branch_tetra",
        source_smiles="F[C@H]1CC2CCC1C2[C@H](Cl)Br",
        intended_classification="coupled_or_shared_ring_ligand_candidate",
    ),
    Candidate(
        case_id="disconnected_polycyclic_ring_tetra_directional",
        source_smiles="F[C@H]1CC2CCC1C2.F/C=C/Cl",
        intended_classification="unsupported_disconnected_product_boundary",
    ),
    Candidate(
        case_id="polycyclic_ring_tetra_directional_branch",
        source_smiles="F[C@H]1CC2CCC1C2/C=C/Cl",
        intended_classification="unsupported_polycyclic_mixed_boundary",
    ),
)


def main() -> None:
    print("# Compositional Stereo Witness Inventory")
    print()
    print(
        "| case | source | classification | supported | categories | "
        "frags | tetra | ring-tetra obligations | directional components | "
        "proof class | components | coupling reasons | proof/runtime | outputs |"
    )
    print(
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | "
        "--- | --- | --- | ---: |"
    )
    for candidate in CANDIDATES:
        mol = Chem.MolFromSmiles(candidate.source_smiles)
        if mol is None:
            print(
                f"| `{candidate.case_id}` | `{candidate.source_smiles}` | "
                f"{candidate.intended_classification} | parse_fail | - | "
                "- | - | - | - | - | - | - | - | - |"
            )
            continue

        gate = south_star_support_gate_report(mol)
        components = extract_south_star_components(mol, support_gate_report=gate)
        proof = compositional_stereo_proof_report(candidate.source_smiles)
        output_count = _output_count(candidate.source_smiles)
        categories = ", ".join(sorted(gate.categories)) or "-"
        proof_runtime = _proof_runtime_status(proof)
        print(
            f"| `{candidate.case_id}` | `{candidate.source_smiles}` | "
            f"{candidate.intended_classification} | {str(gate.supported).lower()} | "
            f"{categories} | {len(Chem.GetMolFrags(mol))} | "
            f"{len(extract_tetrahedral_center_facts(mol))} | "
            f"{len(extract_ring_tetrahedral_interaction_obligations(mol))} | "
            f"{len(components.components)} | {proof.classification} | "
            f"{_component_summary(proof.components)} | "
            f"{_coupling_summary(proof.components)} | {proof_runtime} | "
            f"{output_count} |"
        )


def _output_count(source_smiles: str) -> str:
    try:
        return str(len(mol_to_smiles_enum_s_graph_native(source_smiles).outputs))
    except Exception as exc:
        return f"{type(exc).__name__}"


def _component_summary(
    components: tuple[SouthStarCompositionalStereoComponent, ...],
) -> str:
    return "<br>".join(
        "+".join(component.obligation_ids) for component in components
    ) or "-"


def _coupling_summary(
    components: tuple[SouthStarCompositionalStereoComponent, ...],
) -> str:
    reasons = sorted(
        {
            reason
            for component in components
            for reason in component.coupling_reasons
        }
    )
    return ", ".join(reasons) or "-"


def _proof_runtime_status(proof: SouthStarCompositionalStereoProofReport) -> str:
    if proof.proof_output_count is None or proof.runtime_output_count is None:
        return "-"
    match = "match" if proof.runtime_outputs_match_proof else "mismatch"
    return f"{proof.proof_output_count}/{proof.runtime_output_count} {match}"


if __name__ == "__main__":
    main()
