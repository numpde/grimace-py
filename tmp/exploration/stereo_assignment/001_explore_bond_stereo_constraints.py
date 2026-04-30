from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable

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

FIXTURE = Path("tests/fixtures/rdkit_known_stereo_gaps/2026.03.1.json")
STEREO_KINDS = {"STEREOCIS", "STEREOZ", "STEREOE", "STEREOTRANS"}


def canon(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def is_stereo_double(prep, bond_idx: int) -> bool:
    return prep.bond_kinds[bond_idx] == "DOUBLE" and prep.bond_stereo_kinds[bond_idx] in STEREO_KINDS


def atom(prep, idx: int) -> str:
    return f"{idx}:{prep.atom_tokens[idx]}"


def edge_label(prep, edge: tuple[int, int]) -> str:
    a, b = edge
    bond_idx = prep.bond_index(a, b)
    return (
        f"{a}-{b} {atom(prep, a)}-{atom(prep, b)} "
        f"b{bond_idx} {prep.bond_kinds[bond_idx]} "
        f"dir={prep.directed_bond_token(a, b)!r}/{prep.directed_bond_token(b, a)!r}"
    )


def case_mol(raw: dict, writer_cases_by_id: dict):
    if raw.get("writer_membership_case_id"):
        return mol_from_pinned_source(writer_cases_by_id[raw["writer_membership_case_id"]])
    if raw.get("molblock"):
        return Chem.MolFromMolBlock(raw["molblock"], removeHs=False)
    return Chem.MolFromSmiles(raw["smiles"])


class DSU:
    def __init__(self):
        self.parent = {}

    def add(self, x):
        self.parent.setdefault(x, x)

    def find(self, x):
        self.add(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != x:
            nxt = self.parent[x]
            self.parent[x] = root
            x = nxt
        return root

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def groups(self):
        out = defaultdict(list)
        for x in list(self.parent):
            out[self.find(x)].append(x)
        return [tuple(sorted(v)) for v in out.values()]


def extract(prep):
    stereo_component_ids = cs._stereo_component_ids(prep)
    stereo_component_sizes = cs._component_sizes(stereo_component_ids)
    sides = []
    edge_to_sides = defaultdict(list)
    stereo_bond_to_sides = defaultdict(list)
    parity_edges = []
    carrier_to_stereo_bonds = defaultdict(set)

    for bond_idx, component_idx in enumerate(stereo_component_ids):
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
            side_idx = len(sides)
            sides.append(
                {
                    "idx": side_idx,
                    "component": component_idx,
                    "stereo_bond": bond_idx,
                    "endpoint": endpoint,
                    "other": other,
                    "candidates": candidates,
                }
            )
            stereo_bond_to_sides[bond_idx].append(side_idx)
            oriented = [(endpoint, n) for n in candidates]
            if len(oriented) == 2:
                parity_edges.append((oriented[0], oriented[1], "same_endpoint_alternatives"))
            for n in candidates:
                edge = canon(endpoint, n)
                edge_to_sides[edge].append(side_idx)
                carrier_to_stereo_bonds[edge].add(bond_idx)

    oriented_nodes = set()
    for side in sides:
        for n in side["candidates"]:
            oriented_nodes.add((side["endpoint"], n))
    for a, b in list(oriented_nodes):
        if (b, a) in oriented_nodes and (b, a) < (a, b):
            parity_edges.append(((a, b), (b, a), "reverse_same_carrier"))

    side_dsu = DSU()
    for side in sides:
        side_dsu.add(side["idx"])
    for ids in stereo_bond_to_sides.values():
        for other in ids[1:]:
            side_dsu.union(ids[0], other)
    bond_side_groups = side_dsu.groups()

    carrier_dsu = DSU()
    for side in sides:
        for n in side["candidates"]:
            carrier_dsu.add((side["endpoint"], n))
    for left, right, _kind in parity_edges:
        carrier_dsu.union(left, right)
    carrier_groups = carrier_dsu.groups()

    side_plus_shared = DSU()
    for side in sides:
        side_plus_shared.add(side["idx"])
    for ids in stereo_bond_to_sides.values():
        for other in ids[1:]:
            side_plus_shared.union(ids[0], other)
    for edge, ids in edge_to_sides.items():
        for other in ids[1:]:
            side_plus_shared.union(ids[0], other)
    shared_side_groups = side_plus_shared.groups()

    return {
        "stereo_component_ids": stereo_component_ids,
        "stereo_component_sizes": stereo_component_sizes,
        "sides": sides,
        "edge_to_sides": dict(edge_to_sides),
        "parity_edges": parity_edges,
        "bond_side_groups": bond_side_groups,
        "carrier_groups": carrier_groups,
        "shared_side_groups": shared_side_groups,
        "carrier_to_stereo_bonds": {k: tuple(sorted(v)) for k, v in carrier_to_stereo_bonds.items()},
    }


def print_case(raw: dict, prep, data):
    print("\n" + "=" * 120)
    print(raw["id"])
    print("source:", raw["source"])
    print("expected:", raw["expected"])
    print("atoms/bonds:", prep.atom_count, prep.bond_count, "rdkit", rdBase.rdkitVersion)
    print("current stereo-bond components:", dict(enumerate(data["stereo_component_sizes"])))

    print("\nstereo bonds")
    for bond_idx, comp in enumerate(data["stereo_component_ids"]):
        if comp < 0:
            continue
        begin = prep.bond_begin_atom_indices[bond_idx]
        end = prep.bond_end_atom_indices[bond_idx]
        print(
            f"  b{bond_idx} comp={comp} {atom(prep, begin)}={atom(prep, end)} "
            f"kind={prep.bond_stereo_kinds[bond_idx]} stereo_atoms={prep.bond_stereo_atoms[bond_idx]}"
        )

    print("\nsides")
    for side in data["sides"]:
        cands = "; ".join(
            f"{atom(prep, n)} via {edge_label(prep, canon(side['endpoint'], n))}"
            for n in side["candidates"]
        )
        print(
            f"  s{side['idx']} comp={side['component']} stereo_bond=b{side['stereo_bond']} "
            f"endpoint={atom(prep, side['endpoint'])} other={atom(prep, side['other'])} "
            f"candidates={len(side['candidates'])}: {cands}"
        )

    print("\nshared carrier edges")
    any_shared = False
    for edge, ids in sorted(data["edge_to_sides"].items()):
        if len(ids) <= 1:
            continue
        any_shared = True
        comps = sorted({data["sides"][i]["component"] for i in ids})
        stereo_bonds = sorted({data["sides"][i]["stereo_bond"] for i in ids})
        print(f"  {edge_label(prep, edge)} sides={tuple(ids)} comps={comps} stereo_bonds={stereo_bonds}")
    if not any_shared:
        print("  none")

    print("\nside groups if grouped by stereo bond only")
    for group in data["bond_side_groups"]:
        comps = sorted({data["sides"][i]["component"] for i in group})
        print(f"  sides={group} comps={comps}")

    print("\nside groups if grouped by stereo bond + shared carrier")
    for group in data["shared_side_groups"]:
        comps = sorted({data["sides"][i]["component"] for i in group})
        bonds = sorted({data["sides"][i]["stereo_bond"] for i in group})
        print(f"  sides={group} comps={comps} stereo_bonds={bonds}")

    print("\ncarrier parity/symmetry groups over oriented carrier edges")
    for group in data["carrier_groups"]:
        print("  group:", ", ".join(f"{a}->{b}" for a, b in group))

    print("\ncurrent token preassignment status")
    try:
        side_infos, edge_to_side_ids = cs._stereo_side_infos(prep, data["stereo_component_ids"])
        isolated = tuple(size == 1 for size in data["stereo_component_sizes"])
        ambiguous = cs._ambiguous_shared_edge_groups(side_infos, edge_to_side_ids, isolated)
        print("  ok")
        print("  current ambiguous shared groups:", ambiguous)
    except Exception as exc:
        print("  FAIL", type(exc).__name__, exc)


def summarize_case(raw: dict, prep, data) -> None:
    cross_shared = []
    same_shared = []
    for edge, ids in sorted(data["edge_to_sides"].items()):
        if len(ids) <= 1:
            continue
        comps = sorted({data["sides"][i]["component"] for i in ids})
        entry = (edge, tuple(ids), tuple(comps))
        if len(comps) > 1:
            cross_shared.append(entry)
        else:
            same_shared.append(entry)

    current_status = "ok"
    current_ambiguous_count = 0
    try:
        side_infos, edge_to_side_ids = cs._stereo_side_infos(prep, data["stereo_component_ids"])
        isolated = tuple(size == 1 for size in data["stereo_component_sizes"])
        current_ambiguous_count = len(
            cs._ambiguous_shared_edge_groups(side_infos, edge_to_side_ids, isolated)
        )
    except Exception as exc:
        current_status = f"FAIL:{type(exc).__name__}:{exc}"

    print(raw["id"])
    print(
        "  stereo_bonds=",
        sum(1 for comp in data["stereo_component_ids"] if comp >= 0),
        "current_components=",
        tuple(data["stereo_component_sizes"]),
        "sides=",
        len(data["sides"]),
        "side_groups_bond+shared=",
        tuple(len(group) for group in data["shared_side_groups"]),
        "carrier_groups=",
        tuple(len(group) for group in data["carrier_groups"]),
    )
    print(
        "  shared:",
        f"same_component={len(same_shared)}",
        f"cross_component={len(cross_shared)}",
        f"current_ambiguous_groups={current_ambiguous_count}",
        f"current_status={current_status}",
    )
    for edge, ids, comps in cross_shared:
        print(
            f"    cross {edge_label(prep, edge)} sides={ids} comps={comps}"
        )
    for edge, ids, comps in same_shared:
        print(
            f"    same  {edge_label(prep, edge)} sides={ids} comps={comps}"
        )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_ids", nargs="*")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args(argv)
    payload = json.loads(FIXTURE.read_text())
    writer_cases_by_id = {
        case.case_id: case for case in load_pinned_writer_membership_cases(rdBase.rdkitVersion)
    }
    cases = payload["cases"] if args.all else [c for c in payload["cases"] if c["id"] in set(args.case_ids)]
    if not cases:
        cases = payload["cases"][:3]
    for raw in cases:
        mol = case_mol(raw, writer_cases_by_id)
        prep = prepare_smiles_graph_from_mol_to_smiles_kwargs(
            mol,
            surface_kind=CONNECTED_STEREO_SURFACE,
            isomeric_smiles=raw.get("isomeric_smiles", True),
        )
        data = extract(prep)
        if args.summary:
            summarize_case(raw, prep, data)
        else:
            print_case(raw, prep, data)

if __name__ == "__main__":
    main()
