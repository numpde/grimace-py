from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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


@dataclass(frozen=True)
class Candidate:
    case_id: str
    fixture_path: Path
    smiles: str


@dataclass(frozen=True)
class Discrepancy:
    candidate: Candidate
    atoms: int
    bonds: int
    stereo_bonds: int
    endpoints: int
    incidences: int
    model_shapes: int
    observed_shapes: int
    model_only: tuple[tuple[tuple[int, int], ...], ...]
    observed_only: tuple[tuple[tuple[int, int], ...], ...]
    output_count: int


def iter_json_values(value: object) -> Iterable[dict[str, object]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from iter_json_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_json_values(child)


def load_fixture_smiles(fixture_root: Path) -> tuple[Candidate, ...]:
    candidates: dict[tuple[str, str], Candidate] = {}
    for fixture_path in sorted(fixture_root.rglob("*.json")):
        if "rdkit_upstream_serializer_sources" in fixture_path.parts:
            continue
        try:
            raw = json.loads(fixture_path.read_text())
        except json.JSONDecodeError:
            continue
        for obj in iter_json_values(raw):
            raw_smiles = obj.get("smiles")
            if type(raw_smiles) is not str or not raw_smiles:
                continue
            raw_id = obj.get("id", obj.get("case_id", fixture_path.stem))
            case_id = str(raw_id)
            key = (case_id, raw_smiles)
            candidates.setdefault(
                key,
                Candidate(case_id=case_id, fixture_path=fixture_path, smiles=raw_smiles),
            )
    return tuple(candidates.values())


def collect_rdkit_outputs(mol: Chem.Mol, *, random_samples: int) -> tuple[str, ...]:
    outputs: set[str] = set()
    outputs.add(Chem.MolToSmiles(Chem.Mol(mol), canonical=True, isomericSmiles=True))

    for root in range(mol.GetNumAtoms()):
        try:
            outputs.add(
                Chem.MolToSmiles(
                    Chem.Mol(mol),
                    canonical=False,
                    doRandom=False,
                    rootedAtAtom=root,
                    isomericSmiles=True,
                )
            )
        except Exception:
            pass

    for seed in range(random_samples):
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
    outputs: Iterable[str],
) -> frozenset[tuple[tuple[int, int], ...]]:
    shapes: set[tuple[tuple[int, int], ...]] = set()
    for output_smiles in outputs:
        output = Chem.MolFromSmiles(output_smiles)
        if output is None:
            continue
        matches = stereo_z3.preserving_matches(source, output, system.stereo_bonds)
        for match in matches:
            shapes.add(stereo_z3.mapped_directed_edges(output, match))
    return frozenset(shapes)


def discrepancy_for_candidate(
    candidate: Candidate,
    *,
    random_samples: int,
    max_atoms: int,
) -> Discrepancy | None:
    mol = Chem.MolFromSmiles(candidate.smiles)
    if mol is None or mol.GetNumAtoms() > max_atoms:
        return None
    system = stereo_z3.stereo_system_from_mol(candidate.smiles, mol)
    if len(system.stereo_bonds) < 2:
        return None

    hypothesis = next(
        hypothesis
        for hypothesis in stereo_z3.HYPOTHESES
        if hypothesis.name == "observed-edge-shared-pair"
    )
    model_shapes = stereo_z3.Z3CarrierModel(system, hypothesis).projected_directed_edge_shapes()
    if len(model_shapes) <= 1:
        return None

    outputs = collect_rdkit_outputs(mol, random_samples=random_samples)
    rdkit_shapes = observed_shapes(mol, system, outputs)
    model_only = tuple(sorted(model_shapes - rdkit_shapes))
    observed_only = tuple(sorted(rdkit_shapes - model_shapes))
    if not model_only and not observed_only:
        return None
    return Discrepancy(
        candidate=candidate,
        atoms=mol.GetNumAtoms(),
        bonds=mol.GetNumBonds(),
        stereo_bonds=len(system.stereo_bonds),
        endpoints=len(system.endpoints),
        incidences=len(system.incidences),
        model_shapes=len(model_shapes),
        observed_shapes=len(rdkit_shapes),
        model_only=model_only,
        observed_only=observed_only,
        output_count=len(outputs),
    )


def print_discrepancy(item: Discrepancy) -> None:
    rel = item.candidate.fixture_path.relative_to(ROOT)
    print("=" * 100)
    print(
        item.candidate.case_id,
        f"atoms={item.atoms}",
        f"bonds={item.bonds}",
        f"stereo_bonds={item.stereo_bonds}",
        f"endpoints={item.endpoints}",
        f"incidences={item.incidences}",
    )
    print("fixture:", rel)
    print("smiles:", item.candidate.smiles)
    print(
        "shapes:",
        f"model={item.model_shapes}",
        f"rdkit_observed={item.observed_shapes}",
        f"model_only={len(item.model_only)}",
        f"rdkit_only={len(item.observed_only)}",
        f"rdkit_outputs={item.output_count}",
    )
    for shape in item.model_only[:12]:
        print("  model-only:", shape)
    for shape in item.observed_only[:12]:
        print("  rdkit-only:", shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--max-atoms", type=int, default=45)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    print("rdkit:", rdBase.rdkitVersion)
    print("samples:", args.samples)
    candidates = load_fixture_smiles(ROOT / "tests/fixtures")
    print("fixture smiles candidates:", len(candidates))

    by_atom_count = sorted(
        candidates,
        key=lambda candidate: (
            Chem.MolFromSmiles(candidate.smiles).GetNumAtoms()
            if Chem.MolFromSmiles(candidate.smiles) is not None
            else 10**9,
            candidate.case_id,
        ),
    )
    discrepancies: list[Discrepancy] = []
    skipped_by_atoms = defaultdict(int)
    for idx, candidate in enumerate(by_atom_count, start=1):
        mol = Chem.MolFromSmiles(candidate.smiles)
        if mol is None:
            continue
        skipped_by_atoms[mol.GetNumAtoms()] += 1
        item = discrepancy_for_candidate(
            candidate,
            random_samples=args.samples,
            max_atoms=args.max_atoms,
        )
        if item is None:
            continue
        discrepancies.append(item)
        print_discrepancy(item)
        if len(discrepancies) >= args.limit:
            break

    print("=" * 100)
    print("discrepancies:", len(discrepancies))
    if discrepancies:
        smallest = min(discrepancies, key=lambda item: (item.atoms, item.bonds, item.candidate.case_id))
        print("smallest:")
        print_discrepancy(smallest)


if __name__ == "__main__":
    main()
