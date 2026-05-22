"""Inventory small ring/tetrahedral South Star frontier witnesses."""

from __future__ import annotations

from grimace._south_star.molecule_facts import SouthStarMoleculeFacts
from grimace._south_star.support_gates import south_star_support_gate_report
from grimace._south_star.tetrahedral import (
    extract_ring_tetrahedral_interaction_obligations,
)
from tests.helpers.south_star_semantic_oracle import parse_smiles


CASES: tuple[tuple[str, str], ...] = (
    ("supported_monocycle_center", "F[C@H]1CCCC(C)C1"),
    ("supported_adjacent_monocycle", "F[C@H](Cl)C1CCCCC1"),
    ("supported_monocycle_directional_branch", "F[C@H]1CCCC(/C=C/Cl)C1"),
    ("unsupported_known_fused_witness", "C1CC2CCCC2[C@H]1F"),
    ("unsupported_small_bridged", "F[C@H]1CC2CCC1C2"),
    ("unsupported_small_polycyclic", "F[C@H]1CC2CC1C2"),
    ("supported_no_ring_tetra_obligation_compact", "F[C@H]1C2CC1C2"),
    ("supported_no_ring_tetra_obligation_fused", "F[C@H]1CC2CC2C1"),
    ("unsupported_bridge_variant", "F[C@H]1CCC2CC1C2"),
)


def main() -> None:
    print("| Case | Source | Atoms | Rings | Fused/poly | Supported | Categories | Obligations |")
    print("| --- | --- | ---: | ---: | --- | --- | --- | --- |")
    for case_id, source_smiles in CASES:
        mol = parse_smiles(source_smiles)
        facts = SouthStarMoleculeFacts.from_mol(mol)
        report = south_star_support_gate_report(mol)
        obligations = extract_ring_tetrahedral_interaction_obligations(mol)
        obligation_text = "; ".join(
            (
                f"center={obligation.center_atom_idx}, "
                f"center_in_ring={obligation.center_in_ring}, "
                f"ring_ligands={obligation.ring_ligand_atom_indices}, "
                f"acyclic_ligands={obligation.acyclic_ligand_atom_indices}, "
                f"implicit_h={obligation.implicit_hydrogen_count}"
            )
            for obligation in obligations
        )
        print(
            f"| `{case_id}` | `{source_smiles}` | {mol.GetNumAtoms()} | "
            f"{facts.graph_topology.ring_system.ring_count} | "
            f"{facts.graph_topology.ring_system.fused_or_polycyclic} | "
            f"{report.supported} | `{', '.join(sorted(report.categories))}` | "
            f"{obligation_text or '-'} |"
        )


if __name__ == "__main__":
    main()
