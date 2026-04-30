from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
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


@dataclass(frozen=True)
class SourceCase:
    corpus: str
    case_id: str
    smiles: str | None
    molblock: str | None


def canon(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def is_stereo_double(prep, bond_idx: int) -> bool:
    return (
        prep.bond_kinds[bond_idx] == "DOUBLE"
        and prep.bond_stereo_kinds[bond_idx] in STEREO_KINDS
    )


class DSU:
    def __init__(self):
        self.parent = {}

    def add(self, item):
        self.parent.setdefault(item, item)

    def find(self, item):
        self.add(item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            nxt = self.parent[item]
            self.parent[item] = root
            item = nxt
        return root

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root

    def groups(self):
        out = defaultdict(list)
        for item in list(self.parent):
            out[self.find(item)].append(item)
        return [tuple(sorted(items)) for items in out.values()]


def source_mol(case: SourceCase):
    if case.molblock is not None:
        return Chem.MolFromMolBlock(case.molblock, removeHs=False)
    if case.smiles is not None:
        return Chem.MolFromSmiles(case.smiles)
    return None


def shape_for(prep):
    component_ids = cs._stereo_component_ids(prep)
    side_to_component = {}
    side_to_bond = {}
    stereo_bond_to_sides = defaultdict(list)
    edge_to_sides = defaultdict(list)
    side_candidate_counts = {}
    side_idx = 0
    for bond_idx, component_idx in enumerate(component_ids):
        if component_idx < 0 or not is_stereo_double(prep, bond_idx):
            continue
        begin = prep.bond_begin_atom_indices[bond_idx]
        end = prep.bond_end_atom_indices[bond_idx]
        for endpoint, other in ((begin, end), (end, begin)):
            candidates = tuple(
                n
                for n in prep.neighbors_of(endpoint)
                if n != other
                and prep.bond_kinds[prep.bond_index(endpoint, n)] in {"SINGLE", "AROMATIC"}
            )
            if not candidates:
                continue
            side_to_component[side_idx] = component_idx
            side_to_bond[side_idx] = bond_idx
            stereo_bond_to_sides[bond_idx].append(side_idx)
            side_candidate_counts[side_idx] = len(candidates)
            for n in candidates:
                edge_to_sides[canon(endpoint, n)].append(side_idx)
            side_idx += 1

    dsu = DSU()
    for side in side_to_component:
        dsu.add(side)
    for ids in stereo_bond_to_sides.values():
        for other in ids[1:]:
            dsu.union(ids[0], other)
    for ids in edge_to_sides.values():
        for other in ids[1:]:
            dsu.union(ids[0], other)

    group_shapes = []
    for group in dsu.groups():
        comps = {side_to_component[side] for side in group}
        bonds = {side_to_bond[side] for side in group}
        candidate_counts = sorted(side_candidate_counts[side] for side in group)
        shared_edges = sum(
            1
            for ids in edge_to_sides.values()
            if len(ids) > 1 and any(side in group for side in ids)
        )
        cross_shared_edges = sum(
            1
            for ids in edge_to_sides.values()
            if len(ids) > 1
            and any(side in group for side in ids)
            and len({side_to_component[side] for side in ids}) > 1
        )
        group_shapes.append(
            (
                len(group),
                len(comps),
                len(bonds),
                tuple(candidate_counts),
                shared_edges,
                cross_shared_edges,
            )
        )
    return tuple(sorted(group_shapes))


def load_sources():
    out = []
    seen = set()

    def add(corpus, case_id, smiles, molblock):
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
            add("known_gap", raw["id"], case.smiles, case.molblock)
        else:
            add("known_gap", raw["id"], raw.get("smiles"), raw.get("molblock"))
    for case in writer_cases.values():
        if case.isomeric_smiles:
            add("writer", case.case_id, case.smiles, case.molblock)
    root = ROOT / "tests/fixtures/rdkit_serializer_regressions/2026.03.1"
    for path in sorted(root.glob("*.json")):
        payload = json.loads(path.read_text())
        for raw in payload["cases"]:
            if raw.get("isomeric_smiles", True):
                add(f"serializer:{path.name}", raw["id"], raw.get("smiles"), raw.get("molblock"))

    stereo_regression_root = ROOT / "tests/fixtures/rdkit_stereo_regressions"
    rooted_membership = json.loads((stereo_regression_root / "rooted_membership.json").read_text())
    for raw in rooted_membership["cases"]:
        add("stereo_regression", raw["id"], raw.get("input_smiles"), None)
    steroid = json.loads((stereo_regression_root / "steroid_ring_coupled_component.json").read_text())
    add("stereo_regression", "steroid_ring_coupled_component", steroid["input_smiles"], None)

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


def main():
    shape_counts = Counter()
    examples = {}
    skipped = Counter()
    for source in load_sources():
        mol = source_mol(source)
        if mol is None:
            skipped["unparseable"] += 1
            continue
        try:
            prep = prepare_smiles_graph_from_mol_to_smiles_kwargs(
                mol,
                surface_kind=CONNECTED_STEREO_SURFACE,
                isomeric_smiles=True,
            )
        except Exception:
            skipped["prepare"] += 1
            continue
        shape = shape_for(prep)
        if not shape:
            skipped["no_bond_stereo"] += 1
            continue
        shape_counts[shape] += 1
        examples.setdefault(shape, source)

    print("SHAPES")
    for shape, count in shape_counts.most_common():
        example = examples[shape]
        print(f"  count={count} shape={shape}")
        print(f"    example={example.corpus}:{example.case_id}")
    print("SKIPPED")
    for key, value in skipped.most_common():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
