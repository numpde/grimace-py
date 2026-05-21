"""Probe whether the current ring traversal spine can render fused aromatics.

This is not a runtime path. It deliberately bypasses the South Star
`aromatic_ring_surface` support gate and calls private traversal helpers to
answer one question: do the existing closure-edge-set, aromatic atom-text, and
aromatic bond-text mechanisms already form parse-back-correct supports for
small fused aromatic witnesses?
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

from grimace._south_star.annotation_policy import (
    MaximalEligibleCarrierAnnotationPolicy,
)
from grimace._south_star.component_support_state import SouthStarComponentSupportState
from grimace._south_star.enum_s import _ring_system_traversals
from grimace._south_star.enum_s import _supported_polycyclic_closure_edge_sets
from grimace._south_star.enum_s import render_south_star_tree_traversal
from grimace._south_star.molecule_facts import SouthStarMoleculeFacts
from tests.helpers.south_star_semantic_oracle import south_star_conformance_report


@dataclass(frozen=True, slots=True)
class ProbeCase:
    case_id: str
    smiles: str


CASES = (
    ProbeCase("naphthalene", "c1ccc2ccccc2c1"),
    ProbeCase("quinoline", "c1ccc2ncccc2c1"),
    ProbeCase("benzofuran", "c1ccc2occc2c1"),
)


def main() -> None:
    for case in CASES:
        mol = Chem.MolFromSmiles(case.smiles)
        if mol is None:
            raise ValueError(f"RDKit failed to parse {case.smiles!r}")

        facts = SouthStarMoleculeFacts.from_mol(mol)
        state = SouthStarComponentSupportState(
            molecule_facts=facts,
            annotation_policy=MaximalEligibleCarrierAnnotationPolicy(),
        )
        closure_edge_sets = _supported_polycyclic_closure_edge_sets(mol)
        traversals = _ring_system_traversals(
            mol,
            molecule_facts=facts,
            state=state,
            closure_edge_sets=closure_edge_sets,
            marker_by_edge={},
            component_marker_assignments=(),
        )
        raw_outputs = tuple(render_south_star_tree_traversal(t) for t in traversals)
        support = tuple(dict.fromkeys(raw_outputs))
        rejected = tuple(
            output for output in support if Chem.MolFromSmiles(output) is None
        )
        rejected_by_semantic_oracle = tuple(
            output
            for output in support
            if not south_star_conformance_report(
                source_smiles=case.smiles,
                candidate_smiles=output,
            ).accepted
        )

        print(f"{case.case_id}: {case.smiles}")
        print(f"  supported_by_gate={facts.supported}")
        print(f"  unsupported_categories={sorted(facts.unsupported_categories)}")
        print(f"  atoms={mol.GetNumAtoms()} bonds={mol.GetNumBonds()}")
        print(f"  ring_count={facts.graph_topology.ring_count}")
        print(f"  cyclomatic_number={facts.graph_topology.cyclomatic_number}")
        print(f"  closure_edge_sets={len(closure_edge_sets)}")
        print(f"  traversals={len(traversals)}")
        print(f"  raw_outputs={len(raw_outputs)} support={len(support)}")
        print(f"  rdkit_parse_rejections={len(rejected)}")
        print(f"  semantic_oracle_rejections={len(rejected_by_semantic_oracle)}")
        print(f"  first_outputs={support[:8]}")


if __name__ == "__main__":
    main()
