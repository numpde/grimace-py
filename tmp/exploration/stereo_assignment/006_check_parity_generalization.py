from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from rdkit import Chem, rdBase

from grimace._reference.prepared_graph import (
    CONNECTED_STEREO_SURFACE,
    prepare_smiles_graph_from_mol_to_smiles_kwargs,
)
from grimace._reference.rooted import connected_stereo as cs
from tests.helpers.rdkit_writer_membership import load_pinned_writer_membership_cases
from tests.rdkit_serialization._support import mol_from_pinned_source

STEREO_KINDS = {"STEREOCIS", "STEREOZ", "STEREOE", "STEREOTRANS"}
DIR_TOKENS = {"ENDUPRIGHT": "/", "ENDDOWNRIGHT": "\\"}
TOKEN_BIT = {"/": 0, "\\": 1}


@dataclass(frozen=True)
class WitnessCase:
    corpus: str
    case_id: str
    source: str
    smiles: str | None
    molblock: str | None
    expected: str
    isomeric_smiles: bool
    kekule_smiles: bool = False
    all_bonds_explicit: bool = False
    all_hs_explicit: bool = False
    ignore_atom_map_numbers: bool = False


def canon(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def is_stereo_double(prep, bond_idx: int) -> bool:
    return (
        prep.bond_kinds[bond_idx] == "DOUBLE"
        and prep.bond_stereo_kinds[bond_idx] in STEREO_KINDS
    )


def nonstereo_copy(mol: Chem.Mol) -> Chem.Mol:
    copy = Chem.Mol(mol)
    Chem.RemoveStereochemistry(copy)
    return copy


def source_mol(case: WitnessCase) -> Chem.Mol | None:
    if case.molblock is not None:
        return Chem.MolFromMolBlock(case.molblock, removeHs=False)
    if case.smiles is not None:
        return Chem.MolFromSmiles(case.smiles)
    return None


def directional_assignments(
    mol: Chem.Mol,
    oriented_nodes: set[tuple[int, int]],
    atom_map: tuple[int, ...],
) -> dict[tuple[int, int], int]:
    out = {}
    for bond in mol.GetBonds():
        token = DIR_TOKENS.get(str(bond.GetBondDir()))
        if not token:
            continue
        begin = atom_map[bond.GetBeginAtomIdx()]
        end = atom_map[bond.GetEndAtomIdx()]
        bit = TOKEN_BIT[token]
        if (begin, end) in oriented_nodes:
            out[(begin, end)] = bit
        if (end, begin) in oriented_nodes:
            out[(end, begin)] = bit ^ 1
    return out


def source_constraints(prep):
    component_ids = cs._stereo_component_ids(prep)
    oriented_nodes: set[tuple[int, int]] = set()
    constraints: defaultdict[
        tuple[int, int], list[tuple[tuple[int, int], int, str]]
    ] = defaultdict(list)
    side_count = 0
    edge_to_sides: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    stereo_bond_to_sides: defaultdict[int, list[int]] = defaultdict(list)

    for bond_idx, component_idx in enumerate(component_ids):
        if component_idx < 0 or not is_stereo_double(prep, bond_idx):
            continue
        begin = prep.bond_begin_atom_indices[bond_idx]
        end = prep.bond_end_atom_indices[bond_idx]
        for endpoint, other in ((begin, end), (end, begin)):
            candidates = tuple(
                neighbor_idx
                for neighbor_idx in prep.neighbors_of(endpoint)
                if neighbor_idx != other
                and prep.bond_kinds[prep.bond_index(endpoint, neighbor_idx)]
                in {"SINGLE", "AROMATIC"}
            )
            if not candidates:
                continue
            side_idx = side_count
            side_count += 1
            stereo_bond_to_sides[bond_idx].append(side_idx)
            nodes = [(endpoint, neighbor_idx) for neighbor_idx in candidates]
            oriented_nodes.update(nodes)
            for neighbor_idx in candidates:
                edge_to_sides[canon(endpoint, neighbor_idx)].append(side_idx)
            if len(nodes) == 2:
                left, right = nodes
                constraints[left].append((right, 1, "same_endpoint_alternatives"))
                constraints[right].append((left, 1, "same_endpoint_alternatives"))

    for begin, end in list(oriented_nodes):
        if (end, begin) in oriented_nodes:
            constraints[(begin, end)].append(((end, begin), 1, "reverse_same_carrier"))

    shared_cross_component_edges = 0
    shared_same_component_edges = 0
    for side_ids in edge_to_sides.values():
        if len(side_ids) <= 1:
            continue
        # We only need a cheap shape metric here, so infer component by side's
        # stereo-bond group from construction order.
        shared_same_component_edges += 1
    return component_ids, oriented_nodes, constraints, side_count, shared_same_component_edges


def propagate(
    seed: dict[tuple[int, int], int],
    constraints: dict[tuple[int, int], list[tuple[tuple[int, int], int, str]]],
):
    assignments = dict(seed)
    queue = deque(assignments)
    while queue:
        node = queue.popleft()
        bit = assignments[node]
        for other, xor, kind in constraints.get(node, ()):
            other_bit = bit ^ xor
            if other in assignments:
                if assignments[other] != other_bit:
                    return False, (node, other, kind, assignments[other], other_bit)
            else:
                assignments[other] = other_bit
                queue.append(other)
    return True, None


def load_known_gap_cases() -> list[WitnessCase]:
    path = ROOT / "tests/fixtures/rdkit_known_stereo_gaps/2026.03.1.json"
    payload = json.loads(path.read_text())
    writer_cases = {
        case.case_id: case for case in load_pinned_writer_membership_cases(rdBase.rdkitVersion)
    }
    out = []
    for raw in payload["cases"]:
        smiles = raw.get("smiles")
        molblock = raw.get("molblock")
        if raw.get("writer_membership_case_id"):
            writer_case = writer_cases[raw["writer_membership_case_id"]]
            smiles = writer_case.smiles
            molblock = writer_case.molblock
        out.append(
            WitnessCase(
                corpus="known_gaps",
                case_id=raw["id"],
                source=raw["source"],
                smiles=smiles,
                molblock=molblock,
                expected=raw["expected"],
                isomeric_smiles=raw.get("isomeric_smiles", True),
            )
        )
    return out


def load_writer_membership_cases() -> list[WitnessCase]:
    out = []
    for case in load_pinned_writer_membership_cases(rdBase.rdkitVersion):
        out.append(
            WitnessCase(
                corpus="writer_membership",
                case_id=case.case_id,
                source=case.source,
                smiles=case.smiles,
                molblock=case.molblock,
                expected=case.expected,
                isomeric_smiles=case.isomeric_smiles,
                kekule_smiles=case.kekule_smiles,
                all_bonds_explicit=case.all_bonds_explicit,
                all_hs_explicit=case.all_hs_explicit,
                ignore_atom_map_numbers=case.ignore_atom_map_numbers,
            )
        )
    return out


def load_serializer_cases() -> list[WitnessCase]:
    out = []
    root = ROOT / "tests/fixtures/rdkit_serializer_regressions/2026.03.1"
    for path in sorted(root.glob("*.json")):
        payload = json.loads(path.read_text())
        for raw in payload["cases"]:
            expected = raw.get("expected")
            if not isinstance(expected, list):
                continue
            for idx, value in enumerate(expected):
                out.append(
                    WitnessCase(
                        corpus=f"serializer:{path.name}",
                        case_id=f"{raw['id']}#{idx}",
                        source=raw["source"],
                        smiles=raw.get("smiles"),
                        molblock=raw.get("molblock"),
                        expected=value,
                        isomeric_smiles=raw.get("isomeric_smiles", True),
                        kekule_smiles=raw.get("kekule_smiles", False),
                        all_bonds_explicit=raw.get("all_bonds_explicit", False),
                        all_hs_explicit=raw.get("all_hs_explicit", False),
                        ignore_atom_map_numbers=raw.get("ignore_atom_map_numbers", False),
                    )
                )
    return out


def any_mapping_satisfies(source: Chem.Mol, witness: Chem.Mol, oriented_nodes, constraints):
    source_ns = nonstereo_copy(source)
    witness_ns = nonstereo_copy(witness)
    checked = 0
    for match in source_ns.GetSubstructMatches(witness_ns, uniquify=False, maxMatches=256):
        checked += 1
        seed = directional_assignments(witness, oriented_nodes, tuple(match))
        ok, conflict = propagate(seed, constraints)
        if ok:
            return True, checked, len(seed), None
    return False, checked, 0, "no satisfying atom map"


def main() -> None:
    cases = load_known_gap_cases() + load_writer_membership_cases() + load_serializer_cases()
    totals = Counter()
    failures = []
    skipped = []
    interesting = Counter()

    for case in cases:
        if not case.isomeric_smiles:
            totals["skip_nonisomeric"] += 1
            continue
        src = source_mol(case)
        witness = Chem.MolFromSmiles(case.expected)
        if src is None or witness is None:
            totals["skip_unparseable"] += 1
            skipped.append((case.corpus, case.case_id, "unparseable"))
            continue
        try:
            prep = prepare_smiles_graph_from_mol_to_smiles_kwargs(
                src,
                surface_kind=CONNECTED_STEREO_SURFACE,
                isomeric_smiles=case.isomeric_smiles,
                kekule_smiles=case.kekule_smiles,
                all_bonds_explicit=case.all_bonds_explicit,
                all_hs_explicit=case.all_hs_explicit,
                ignore_atom_map_numbers=case.ignore_atom_map_numbers,
            )
        except Exception as exc:
            totals["skip_prepare_source"] += 1
            skipped.append((case.corpus, case.case_id, f"source prepare: {type(exc).__name__}: {exc}"))
            continue
        component_ids, oriented_nodes, constraints, side_count, shared_edge_count = source_constraints(prep)
        stereo_bond_count = sum(1 for component_idx in component_ids if component_idx >= 0)
        if stereo_bond_count == 0:
            totals["skip_no_bond_stereo"] += 1
            continue
        ok, checked, seed_count, reason = any_mapping_satisfies(
            src,
            witness,
            oriented_nodes,
            constraints,
        )
        totals["checked"] += 1
        totals[f"corpus:{case.corpus}"] += 1
        if shared_edge_count:
            interesting["has_shared_carrier"] += 1
        if seed_count < len(oriented_nodes):
            interesting["partial_visible_seed"] += 1
        if ok:
            totals["ok"] += 1
        else:
            totals["fail"] += 1
            failures.append((case, checked, reason, stereo_bond_count, side_count, len(oriented_nodes)))

    print("TOTALS")
    for key, value in totals.most_common():
        print(f"  {key}: {value}")
    print("INTERESTING")
    for key, value in interesting.most_common():
        print(f"  {key}: {value}")
    print("FAILURES")
    for case, checked, reason, stereo_bond_count, side_count, oriented_count in failures[:50]:
        print(
            f"  {case.corpus} {case.case_id}: checked_maps={checked} "
            f"stereo_bonds={stereo_bond_count} sides={side_count} "
            f"oriented={oriented_count} reason={reason}"
        )
    if len(failures) > 50:
        print(f"  ... {len(failures) - 50} more")
    print("SKIPPED SAMPLE")
    for item in skipped[:20]:
        print(f"  {item}")


if __name__ == "__main__":
    main()
