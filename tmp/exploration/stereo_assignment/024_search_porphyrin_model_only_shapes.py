from __future__ import annotations

import importlib.util
import sys
from collections import Counter, defaultdict
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


CASE_ID = "dataset_regression_02_porphyrin_like_fragment"


@dataclass(frozen=True)
class SeenOutput:
    mode: str
    smiles: str


def model_shapes(system: object) -> frozenset[tuple[tuple[int, int], ...]]:
    hypothesis = next(
        hypothesis
        for hypothesis in stereo_z3.HYPOTHESES
        if hypothesis.name == "observed-edge-shared-pair"
    )
    return stereo_z3.Z3CarrierModel(system, hypothesis).projected_directed_edge_shapes()


def two_choice_endpoints(system: object) -> tuple[tuple[int, int, tuple[tuple[int, int], ...]], ...]:
    out = []
    for endpoint_id, endpoint in enumerate(system.endpoints):
        if len(endpoint.candidate_neighbors) != 2:
            continue
        edges = tuple(
            stereo_z3.canonical_edge(endpoint.endpoint, neighbor)
            for neighbor in endpoint.candidate_neighbors
        )
        out.append((endpoint_id, endpoint.endpoint, edges))
    return tuple(out)


def bit_pattern(
    shape: tuple[tuple[int, int], ...],
    two_choice: tuple[tuple[int, int, tuple[tuple[int, int], ...]], ...],
) -> tuple[int, ...]:
    directed = set(shape)
    return tuple(1 if edges[1] in directed else 0 for _endpoint_id, _atom, edges in two_choice)


def shapes_for_smiles(
    source: Chem.Mol,
    system: object,
    smiles: str,
) -> frozenset[tuple[tuple[int, int], ...]]:
    output = Chem.MolFromSmiles(smiles)
    if output is None:
        return frozenset()
    matches = stereo_z3.preserving_matches(source, output, system.stereo_bonds)
    return frozenset(stereo_z3.mapped_directed_edges(output, match) for match in matches)


def collect_rdkit_outputs(source: Chem.Mol, *, random_samples: int) -> dict[str, SeenOutput]:
    outputs: dict[str, SeenOutput] = {}

    def add(mode: str, smiles: str) -> None:
        outputs.setdefault(smiles, SeenOutput(mode=mode, smiles=smiles))

    add(
        "canonical",
        Chem.MolToSmiles(source, canonical=True, isomericSmiles=True),
    )

    for root in range(source.GetNumAtoms()):
        try:
            add(
                f"rooted:{root}",
                Chem.MolToSmiles(
                    Chem.Mol(source),
                    canonical=False,
                    doRandom=False,
                    rootedAtAtom=root,
                    isomericSmiles=True,
                ),
            )
        except Exception as exc:
            add(f"rooted:{root}:error:{type(exc).__name__}", "")

    for seed in range(random_samples):
        Chem.rdBase.SeedRandomNumberGenerator(seed)
        add(
            f"random:{seed}",
            Chem.MolToSmiles(
                Chem.Mol(source),
                canonical=False,
                doRandom=True,
                isomericSmiles=True,
            ),
        )
    return {smiles: seen for smiles, seen in outputs.items() if smiles}


def main() -> None:
    source_smiles = stereo_z3.PINNED_CASES[CASE_ID]["smiles"]
    source = Chem.MolFromSmiles(source_smiles)
    assert source is not None
    system = stereo_z3.stereo_system_from_mol(source_smiles, source)
    two_choice = two_choice_endpoints(system)
    allowed_shapes = model_shapes(system)

    outputs = collect_rdkit_outputs(source, random_samples=32768)
    shape_to_outputs: dict[tuple[tuple[int, int], ...], list[SeenOutput]] = defaultdict(list)
    for seen in outputs.values():
        for shape in shapes_for_smiles(source, system, seen.smiles):
            shape_to_outputs[shape].append(seen)

    observed_shapes = frozenset(shape_to_outputs)
    model_only = sorted(allowed_shapes - observed_shapes)
    observed_only = sorted(observed_shapes - allowed_shapes)

    print("rdkit:", rdBase.rdkitVersion)
    print("legacy:", Chem.GetUseLegacyStereoPerception())
    print("unique rdkit outputs:", len(outputs))
    print("model shapes:", len(allowed_shapes))
    print("observed shapes:", len(observed_shapes))
    print("model-only shapes:", len(model_only))
    print("observed-only shapes:", len(observed_only))
    print()
    print("two-choice bit legend:")
    for idx, (endpoint_id, atom, edges) in enumerate(two_choice):
        print(f"  b{idx}: endpoint {endpoint_id} atom {atom}; 0={edges[0]}, 1={edges[1]}")

    print()
    print("all model shapes:")
    for shape in sorted(allowed_shapes):
        bits = bit_pattern(shape, two_choice)
        examples = shape_to_outputs.get(shape, [])
        status = "OBSERVED" if examples else "MODEL_ONLY"
        print(f"{bits} {status} shape={shape}")
        for seen in examples[:2]:
            print(f"  {seen.mode}: {seen.smiles}")

    if observed_only:
        print()
        print("observed-only shapes outside model:")
        for shape in observed_only:
            print(bit_pattern(shape, two_choice), shape)
            for seen in shape_to_outputs[shape][:2]:
                print(f"  {seen.mode}: {seen.smiles}")

    print()
    print("model-only bit patterns:")
    for shape in model_only:
        print(bit_pattern(shape, two_choice), shape)

    bit_counts = Counter(bit_pattern(shape, two_choice) for shape in observed_shapes)
    print()
    print("observed bit counts:")
    for bits, count in sorted(bit_counts.items()):
        print(bits, count)


if __name__ == "__main__":
    main()
