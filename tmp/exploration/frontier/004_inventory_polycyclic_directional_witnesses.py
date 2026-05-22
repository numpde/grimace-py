"""Inventory polycyclic ring/tetra plus directional South Star witnesses."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

from grimace._south_star.components import SouthStarSemanticStereoComponent
from grimace._south_star.components import SouthStarSourceStereoFeature
from grimace._south_star.components import _componentize_features
from grimace._south_star.components import _source_stereo_features
from grimace._south_star.enum_s import mol_to_smiles_enum_s_graph_native
from grimace._south_star.support_gates import south_star_support_gate_report
from grimace._south_star.tetrahedral import (
    extract_ring_tetrahedral_interaction_obligations,
    extract_tetrahedral_center_facts,
)


@dataclass(frozen=True, slots=True)
class Candidate:
    case_id: str
    source_smiles: str
    role: str


CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        case_id="monocycle_ring_tetra_directional_supported_baseline",
        source_smiles="F[C@H]1CCCC(/C=C/Cl)C1",
        role="supported mixed ring/tetra directional baseline",
    ),
    Candidate(
        case_id="polycyclic_ring_tetra_branch_tetra_supported_baseline",
        source_smiles="F[C@H]1CC2CCC1C2[C@H](Cl)Br",
        role="supported polycyclic ring/tetra scale baseline",
    ),
    Candidate(
        case_id="polycyclic_ring_tetra_directional_branch",
        source_smiles="F[C@H]1CC2CCC1C2/C=C/Cl",
        role="connected mixed polycyclic ring/tetra directional frontier",
    ),
    Candidate(
        case_id="disconnected_polycyclic_ring_tetra_directional",
        source_smiles="F[C@H]1CC2CCC1C2.F/C=C/Cl",
        role="fragment-product mixed polycyclic ring/tetra directional frontier",
    ),
)


def main() -> None:
    print("# Polycyclic Ring/Tetra Directional Witness Inventory")
    print()
    print(
        "| case | source | role | supported | categories | fragments | "
        "tetra facts | ring/tetra obligations | raw directional features | "
        "raw directional components | component couplings | runtime | "
        "proof-model need |"
    )
    print(
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | "
        "--- | --- | --- |"
    )
    for candidate in CANDIDATES:
        mol = Chem.MolFromSmiles(candidate.source_smiles)
        if mol is None:
            raise ValueError(f"cannot parse {candidate.source_smiles!r}")
        gate = south_star_support_gate_report(mol)
        features = _source_stereo_features(mol)
        components = _componentize_features(features)
        categories = ", ".join(sorted(gate.categories)) or "-"
        print(
            f"| `{candidate.case_id}` | `{candidate.source_smiles}` | "
            f"{candidate.role} | {str(gate.supported).lower()} | "
            f"{categories} | {len(Chem.GetMolFrags(mol))} | "
            f"{len(extract_tetrahedral_center_facts(mol))} | "
            f"{len(extract_ring_tetrahedral_interaction_obligations(mol))} | "
            f"{len(features)} | {len(components)} | "
            f"{_coupling_summary(components)} | {_runtime_summary(candidate)} | "
            f"{_proof_model_need(gate.categories, features)} |"
        )


def _coupling_summary(
    components: tuple[SouthStarSemanticStereoComponent, ...],
) -> str:
    couplings = tuple(
        coupling.category
        for component in components
        for coupling in component.coupling_causes
    )
    return ", ".join(couplings) or "-"


def _runtime_summary(candidate: Candidate) -> str:
    try:
        outputs = mol_to_smiles_enum_s_graph_native(candidate.source_smiles).outputs
    except Exception as exc:
        return type(exc).__name__
    return f"{len(outputs)} outputs"


def _proof_model_need(
    categories: set[str],
    features: tuple[SouthStarSourceStereoFeature, ...],
) -> str:
    if not categories:
        return "already supported"
    needs = []
    if "fused_or_polycyclic_ring" in categories:
        needs.append("polycyclic traversal facts")
    if "ring_tetrahedral_interaction" in categories:
        needs.append("polycyclic ring/tetra obligations")
    if features:
        needs.append("directional component facts")
    return ", ".join(needs) or "unknown"


if __name__ == "__main__":
    main()
