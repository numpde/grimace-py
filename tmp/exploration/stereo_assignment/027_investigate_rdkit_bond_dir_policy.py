from __future__ import annotations

import importlib.util
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem, rdBase


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "stereo_z3",
    ROOT / "tmp/exploration/stereo_assignment/023_investigate_stereo_constraint_system_z3.py",
)
stereo_z3 = importlib.util.module_from_spec(SPEC)
sys.modules["stereo_z3"] = stereo_z3
assert SPEC.loader is not None
SPEC.loader.exec_module(stereo_z3)


CASES = {
    "minimal_nonstereo_double_hazard": (
        "C/N=C1C=C/C(=N/C)[N-]/1"
    ),
    "minimal_hazard_plus_side_chain": (
        "C=C/N=C1C=C/C(=N/C)[N-]/1"
    ),
    "porphyrin_reduced_24": (
        "c1cc2[n-]c1/N=c1/cc/c([n-]1)=N/c1ccc([n-]1)/N=c1/cc/c([n-]1)=N/2"
    ),
    "porphyrin_original": stereo_z3.PINNED_CASES[
        "dataset_regression_02_porphyrin_like_fragment"
    ]["smiles"],
}


@dataclass(frozen=True)
class ShapeReport:
    local_shapes: frozenset[tuple[tuple[int, int], ...]]
    rdkit_like_shapes: frozenset[tuple[tuple[int, int], ...]]
    rdkit_observed_shapes: frozenset[tuple[tuple[int, int], ...]]
    rdkit_outputs: int


def rdkit_outputs(mol: Chem.Mol, *, samples: int) -> tuple[str, ...]:
    outputs = {Chem.MolToSmiles(Chem.Mol(mol), canonical=True, isomericSmiles=True)}
    for root in range(mol.GetNumAtoms()):
        outputs.add(
            Chem.MolToSmiles(
                Chem.Mol(mol),
                canonical=False,
                doRandom=False,
                rootedAtAtom=root,
                isomericSmiles=True,
            )
        )
    for seed in range(samples):
        Chem.rdBase.SeedRandomNumberGenerator(seed)
        outputs.add(
            Chem.MolToSmiles(
                Chem.Mol(mol),
                canonical=False,
                doRandom=True,
                isomericSmiles=True,
            )
        )
    return tuple(sorted(outputs))


def observed_shapes(
    source: Chem.Mol,
    system: object,
    outputs: tuple[str, ...],
) -> frozenset[tuple[tuple[int, int], ...]]:
    shapes: set[tuple[tuple[int, int], ...]] = set()
    for output_smiles in outputs:
        output = Chem.MolFromSmiles(output_smiles)
        if output is None:
            continue
        for match in stereo_z3.preserving_matches(source, output, system.stereo_bonds):
            shapes.add(stereo_z3.mapped_directed_edges(output, match))
    return frozenset(shapes)


def shape_report(smiles: str, *, samples: int) -> ShapeReport:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    system = stereo_z3.stereo_system_from_mol(smiles, mol)

    local_hypothesis = next(
        hypothesis
        for hypothesis in stereo_z3.HYPOTHESES
        if hypothesis.name == "observed-edge-shared-pair"
    )
    local_shapes = stereo_z3.Z3CarrierModel(
        system,
        local_hypothesis,
    ).projected_directed_edge_shapes()

    # This is the first RDKit-policy candidate, not a claimed semantic rule:
    # RDKit removes carrier directions that would put directional bonds on both
    # sides of a non-stereo double bond and thereby risk coercing that double
    # bond into an unintended stereo assignment.
    rdkit_policy = stereo_z3.ConstraintHypothesis(
        name="rdkit-nonstereo-double-hazard",
        directed_edges_observed_by_all_incident_endpoints=True,
        allow_two_candidate_endpoint_only_when_all_edges_shared=True,
        allow_traversal_flip_variables=False,
        enforce_strict_hazards=True,
    )
    rdkit_like_shapes = stereo_z3.Z3CarrierModel(
        system,
        rdkit_policy,
    ).projected_directed_edge_shapes()

    outputs = rdkit_outputs(mol, samples=samples)
    return ShapeReport(
        local_shapes=local_shapes,
        rdkit_like_shapes=rdkit_like_shapes,
        rdkit_observed_shapes=observed_shapes(mol, system, outputs),
        rdkit_outputs=len(outputs),
    )


def print_case(name: str, smiles: str, *, samples: int) -> None:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    system = stereo_z3.stereo_system_from_mol(smiles, mol)
    report = shape_report(smiles, samples=samples)

    print("=" * 100)
    print(name)
    print("smiles:", smiles)
    print(
        "size:",
        f"atoms={mol.GetNumAtoms()}",
        f"bonds={mol.GetNumBonds()}",
        f"stereo_bonds={len(system.stereo_bonds)}",
        f"hazards={len(system.hazards)}",
    )
    for hazard in system.hazards:
        print(
            "  hazard:",
            f"nonstereo_double={hazard.begin}={hazard.end}",
            f"left={hazard.begin_candidate_edges}",
            f"right={hazard.end_candidate_edges}",
        )
    print(
        "shapes:",
        f"semantic_local={len(report.local_shapes)}",
        f"rdkit_hazard_policy={len(report.rdkit_like_shapes)}",
        f"rdkit_observed={len(report.rdkit_observed_shapes)}",
        f"rdkit_outputs={report.rdkit_outputs}",
    )
    print(
        "diffs:",
        f"local_extra_vs_observed={len(report.local_shapes - report.rdkit_observed_shapes)}",
        f"hazard_extra_vs_observed={len(report.rdkit_like_shapes - report.rdkit_observed_shapes)}",
        f"observed_missing_from_hazard={len(report.rdkit_observed_shapes - report.rdkit_like_shapes)}",
    )
    for shape in sorted(report.local_shapes - report.rdkit_like_shapes):
        print("  excluded_by_hazard_policy:", shape)
    for shape in sorted(report.rdkit_like_shapes - report.rdkit_observed_shapes)[:8]:
        print("  still_policy_only:", shape)
    for shape in sorted(report.rdkit_observed_shapes - report.rdkit_like_shapes)[:8]:
        print("  observed_not_policy:", shape)

    two_choice = []
    for endpoint_id, endpoint in enumerate(system.endpoints):
        if len(endpoint.candidate_neighbors) != 2:
            continue
        edges = tuple(
            stereo_z3.canonical_edge(endpoint.endpoint, neighbor)
            for neighbor in endpoint.candidate_neighbors
        )
        two_choice.append((endpoint_id, endpoint.endpoint, edges))
    if two_choice:
        print("two-choice endpoint bits:")
        for bit_idx, (endpoint_id, atom_idx, edges) in enumerate(two_choice):
            print(
                f"  b{bit_idx}: endpoint={endpoint_id} atom={atom_idx} "
                f"0={edges[0]} 1={edges[1]}"
            )

        def bit_pattern(shape: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
            directed = set(shape)
            return tuple(
                1 if edges[1] in directed else 0
                for _endpoint_id, _atom_idx, edges in two_choice
            )

        observed_bits = sorted(
            bit_pattern(shape) for shape in report.rdkit_observed_shapes
        )
        local_extra_bits = sorted(
            bit_pattern(shape)
            for shape in report.local_shapes - report.rdkit_observed_shapes
        )
        print("  observed_bits:", observed_bits)
        print("  local_extra_bits:", local_extra_bits)
        if len(two_choice) == 4:
            excluded_by_pair_disagreement = [
                bits
                for bits in local_extra_bits
                if (bits[0] != bits[1]) and (bits[2] != bits[3])
            ]
            print(
                "  extras_with_(b0!=b1)&&(b2!=b3):",
                len(excluded_by_pair_disagreement),
                "of",
                len(local_extra_bits),
            )


def main() -> None:
    print("rdkit:", rdBase.rdkitVersion)
    print("principle: compare semantic-local constraints separately from RDKit policy constraints.")
    for name, smiles in CASES.items():
        samples = 32768 if "porphyrin" in name else 200000
        print_case(name, smiles, samples=samples)


if __name__ == "__main__":
    main()
