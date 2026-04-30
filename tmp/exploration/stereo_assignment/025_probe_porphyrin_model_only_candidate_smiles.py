from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from rdkit import Chem


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "shape_search",
    ROOT / "tmp/exploration/stereo_assignment/024_search_porphyrin_model_only_shapes.py",
)
shape_search = importlib.util.module_from_spec(SPEC)
sys.modules["shape_search"] = shape_search
assert SPEC.loader is not None
SPEC.loader.exec_module(shape_search)

stereo_z3 = shape_search.stereo_z3


def token_assignments_for_shape(
    system: object,
    shape: tuple[tuple[int, int], ...],
    *,
    limit: int = 16,
) -> list[dict[tuple[int, int], bool]]:
    hypothesis = next(
        hypothesis
        for hypothesis in stereo_z3.HYPOTHESES
        if hypothesis.name == "observed-edge-shared-pair"
    )
    model = stereo_z3.Z3CarrierModel(system, hypothesis)
    for edge, var in model.edge_vars.items():
        model.solver.add(var == (edge in shape))

    out = []
    while len(out) < limit and model.solver.check() == stereo_z3.z3.sat:
        z3_model = model.solver.model()
        assignment = {
            edge: bool(z3_model.eval(var, model_completion=True))
            for edge, var in model.token_vars.items()
            if edge in shape
        }
        out.append(assignment)
        model.solver.add(
            stereo_z3.z3.Or(
                *(
                    model.token_vars[edge] != value
                    for edge, value in sorted(assignment.items())
                )
            )
        )
    return out


def set_source_shape_dirs(
    source: Chem.Mol,
    shape: tuple[tuple[int, int], ...],
    token_assignment: dict[tuple[int, int], bool],
) -> Chem.Mol:
    mol = Chem.Mol(source)
    for bond in mol.GetBonds():
        if bond.GetBondDir() in (Chem.BondDir.ENDUPRIGHT, Chem.BondDir.ENDDOWNRIGHT):
            bond.SetBondDir(Chem.BondDir.NONE)
    for edge in shape:
        bond = mol.GetBondBetweenAtoms(*edge)
        if bond is None:
            raise ValueError(edge)
        bond.SetBondDir(
            Chem.BondDir.ENDDOWNRIGHT
            if token_assignment.get(edge, False)
            else Chem.BondDir.ENDUPRIGHT
        )
    Chem.AssignStereochemistry(mol, cleanIt=False, force=True)
    return mol


def realized_shapes(source: Chem.Mol, system: object, candidate: Chem.Mol) -> dict[str, set[tuple[tuple[int, int], ...]]]:
    out: dict[str, set[tuple[tuple[int, int], ...]]] = {}
    for root in range(candidate.GetNumAtoms()):
        smiles = Chem.MolToSmiles(
            Chem.Mol(candidate),
            canonical=False,
            doRandom=False,
            rootedAtAtom=root,
            isomericSmiles=True,
        )
        parsed = Chem.MolFromSmiles(smiles)
        if parsed is None:
            out[smiles] = set()
            continue
        out[smiles] = set()
        for match in stereo_z3.preserving_matches(source, parsed, system.stereo_bonds):
            out[smiles].add(stereo_z3.mapped_directed_edges(parsed, match))
    return out


def main() -> None:
    source_smiles = stereo_z3.PINNED_CASES[shape_search.CASE_ID]["smiles"]
    source = Chem.MolFromSmiles(source_smiles)
    assert source is not None
    system = stereo_z3.stereo_system_from_mol(source_smiles, source)
    two_choice = shape_search.two_choice_endpoints(system)
    allowed = shape_search.model_shapes(system)

    observed = set()
    for output in shape_search.collect_rdkit_outputs(source, random_samples=32768).values():
        observed.update(shape_search.shapes_for_smiles(source, system, output.smiles))
    model_only = sorted(allowed - observed)

    print("target model-only shapes:", len(model_only))
    for target in model_only:
        print("=" * 120)
        print("target bits:", shape_search.bit_pattern(target, two_choice))
        print("target shape:", target)
        assignments = token_assignments_for_shape(system, target)
        print("z3 token assignments:", len(assignments))
        exact_total = []
        nonempty_total = []
        for idx, assignment in enumerate(assignments):
            seeded = set_source_shape_dirs(source, target, assignment)
            realized = realized_shapes(source, system, seeded)
            exact = [(smiles, shapes) for smiles, shapes in realized.items() if target in shapes]
            nonempty = [(smiles, shapes) for smiles, shapes in realized.items() if shapes]
            if exact:
                exact_total.extend((idx, smiles, shapes) for smiles, shapes in exact)
            if nonempty:
                nonempty_total.extend((idx, smiles, shapes) for smiles, shapes in nonempty)
        print("exact target survived RDKit writer roots:", len(exact_total))
        for idx, smiles, shapes in exact_total[:4]:
            print(f"  exact candidate assignment={idx}: {smiles}")
            print("    shapes:", sorted(shapes))
        print("other preserving rooted outputs from mutated source:", len(nonempty_total))
        for idx, smiles, shapes in nonempty_total[:4]:
            print(f"  fallback assignment={idx}: {smiles}")
            print("    bits:", sorted(shape_search.bit_pattern(shape, two_choice) for shape in shapes))
            print("    shapes:", sorted(shapes))


if __name__ == "__main__":
    main()
