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
class SourceCase:
    corpus: str
    case_id: str
    smiles: str | None
    molblock: str | None


def canon(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def nonstereo_copy(mol: Chem.Mol) -> Chem.Mol:
    copy = Chem.Mol(mol)
    Chem.RemoveStereochemistry(copy)
    return copy


def is_stereo_double(prep, bond_idx: int) -> bool:
    return (
        prep.bond_kinds[bond_idx] == "DOUBLE"
        and prep.bond_stereo_kinds[bond_idx] in STEREO_KINDS
    )


def source_mol(case: SourceCase) -> Chem.Mol | None:
    if case.molblock is not None:
        return Chem.MolFromMolBlock(case.molblock, removeHs=False)
    if case.smiles is not None:
        return Chem.MolFromSmiles(case.smiles)
    return None


def constraints(prep):
    component_ids = cs._stereo_component_ids(prep)
    oriented = set()
    cons = defaultdict(list)
    side_count = 0
    shared_edges = 0
    edge_to_sides = defaultdict(list)
    for bond_idx, component_idx in enumerate(component_ids):
        if component_idx < 0 or not is_stereo_double(prep, bond_idx):
            continue
        begin = prep.bond_begin_atom_indices[bond_idx]
        end = prep.bond_end_atom_indices[bond_idx]
        for endpoint, other in ((begin, end), (end, begin)):
            cands = tuple(
                n
                for n in prep.neighbors_of(endpoint)
                if n != other
                and prep.bond_kinds[prep.bond_index(endpoint, n)] in {"SINGLE", "AROMATIC"}
            )
            if not cands:
                continue
            side_idx = side_count
            side_count += 1
            nodes = [(endpoint, n) for n in cands]
            oriented.update(nodes)
            for n in cands:
                edge_to_sides[canon(endpoint, n)].append(side_idx)
            if len(nodes) == 2:
                cons[nodes[0]].append((nodes[1], 1))
                cons[nodes[1]].append((nodes[0], 1))
    for a, b in list(oriented):
        if (b, a) in oriented:
            cons[(a, b)].append(((b, a), 1))
    shared_edges = sum(1 for ids in edge_to_sides.values() if len(ids) > 1)
    return component_ids, oriented, cons, side_count, shared_edges


def seed_from_witness(witness: Chem.Mol, oriented, atom_map):
    out = {}
    for bond in witness.GetBonds():
        token = DIR_TOKENS.get(str(bond.GetBondDir()))
        if not token:
            continue
        begin = atom_map[bond.GetBeginAtomIdx()]
        end = atom_map[bond.GetEndAtomIdx()]
        bit = TOKEN_BIT[token]
        if (begin, end) in oriented:
            out[(begin, end)] = bit
        if (end, begin) in oriented:
            out[(end, begin)] = bit ^ 1
    return out


def sat(seed, cons):
    assign = dict(seed)
    queue = deque(assign)
    while queue:
        node = queue.popleft()
        bit = assign[node]
        for other, xor in cons.get(node, ()):
            val = bit ^ xor
            if other in assign:
                if assign[other] != val:
                    return False
            else:
                assign[other] = val
                queue.append(other)
    return True


def any_map_satisfies(source: Chem.Mol, witness: Chem.Mol, oriented, cons):
    source_ns = nonstereo_copy(source)
    witness_ns = nonstereo_copy(witness)
    for match in source_ns.GetSubstructMatches(witness_ns, uniquify=False, maxMatches=512):
        if sat(seed_from_witness(witness, oriented, tuple(match)), cons):
            return True
    return False


def load_sources() -> list[SourceCase]:
    out = []
    seen = set()

    def add(corpus: str, case_id: str, smiles: str | None, molblock: str | None):
        key = (smiles, molblock)
        if key in seen:
            return
        seen.add(key)
        out.append(SourceCase(corpus, case_id, smiles, molblock))

    known = json.loads((ROOT / "tests/fixtures/rdkit_known_stereo_gaps/2026.03.1.json").read_text())
    writer_cases = {case.case_id: case for case in load_pinned_writer_membership_cases(rdBase.rdkitVersion)}
    for raw in known["cases"]:
        if raw.get("writer_membership_case_id"):
            case = writer_cases[raw["writer_membership_case_id"]]
            add("known_gap_source", raw["id"], case.smiles, case.molblock)
        else:
            add("known_gap_source", raw["id"], raw.get("smiles"), raw.get("molblock"))

    for case in writer_cases.values():
        if case.isomeric_smiles:
            add("writer_membership_source", case.case_id, case.smiles, case.molblock)

    root = ROOT / "tests/fixtures/rdkit_serializer_regressions/2026.03.1"
    for path in sorted(root.glob("*.json")):
        payload = json.loads(path.read_text())
        for raw in payload["cases"]:
            if raw.get("isomeric_smiles", True):
                add(f"serializer_source:{path.name}", raw["id"], raw.get("smiles"), raw.get("molblock"))

    stereo_regression_root = ROOT / "tests/fixtures/rdkit_stereo_regressions"
    rooted_membership = json.loads((stereo_regression_root / "rooted_membership.json").read_text())
    for raw in rooted_membership["cases"]:
        add("stereo_regression_source", raw["id"], raw.get("input_smiles"), None)
    steroid = json.loads((stereo_regression_root / "steroid_ring_coupled_component.json").read_text())
    add("stereo_regression_source", "steroid_ring_coupled_component", steroid["input_smiles"], None)

    add(
        "test_literal",
        "rooted_polyene_bond_stereo_case",
        "CC1=C(C(CCC1)(C)C)/C=C/C(=C/C=C/C(=C/C(=O)O)/C)/C",
        None,
    )
    add(
        "test_literal",
        "rooted_sidechain_steroid",
        "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]\\\\2"
        "[C@@]1(CCC/C2=C\\\\C=C/3\\\\C[C@H](CCC3=C)O)C",
        None,
    )
    return out


def main() -> None:
    totals = Counter()
    failures = []
    for source_case in load_sources():
        mol = source_mol(source_case)
        if mol is None:
            totals["skip_unparseable_source"] += 1
            continue
        try:
            prep = prepare_smiles_graph_from_mol_to_smiles_kwargs(
                mol,
                surface_kind=CONNECTED_STEREO_SURFACE,
                isomeric_smiles=True,
            )
        except Exception:
            totals["skip_prepare"] += 1
            continue
        component_ids, oriented, cons, side_count, shared_edges = constraints(prep)
        stereo_bond_count = sum(1 for component_idx in component_ids if component_idx >= 0)
        if stereo_bond_count == 0:
            totals["skip_no_bond_stereo"] += 1
            continue
        totals["sources_checked"] += 1
        if shared_edges:
            totals["sources_with_shared_carrier"] += 1
        try:
            random_outputs = Chem.MolToRandomSmilesVect(
                Chem.Mol(mol),
                200,
                randomSeed=17,
                isomericSmiles=True,
            )
        except Exception as exc:
            failures.append((source_case, "<sampling failed>", type(exc).__name__, str(exc)))
            totals["sampling_failed"] += 1
            continue
        for output in sorted(set(random_outputs)):
            witness = Chem.MolFromSmiles(output)
            if witness is None:
                totals["skip_unparseable_witness"] += 1
                continue
            totals["outputs_checked"] += 1
            if any_map_satisfies(mol, witness, oriented, cons):
                totals["outputs_ok"] += 1
            else:
                totals["outputs_fail"] += 1
                failures.append((source_case, output, stereo_bond_count, side_count, len(oriented)))
                if len(failures) >= 20:
                    break
        if len(failures) >= 20:
            break

    print("TOTALS")
    for key, value in totals.most_common():
        print(f"  {key}: {value}")
    print("FAILURES")
    for item in failures:
        print(f"  {item}")


if __name__ == "__main__":
    main()
