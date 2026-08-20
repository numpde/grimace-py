from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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

FIXTURE = Path("tests/fixtures/rdkit_known_stereo_gaps/2026.03.1.json")
STEREO_KINDS = {"STEREOCIS", "STEREOZ", "STEREOE", "STEREOTRANS"}
DIR_TOKENS = {
    "ENDUPRIGHT": "/",
    "ENDDOWNRIGHT": "\\",
}


def canon(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def is_stereo_double(prep, bond_idx: int) -> bool:
    return prep.bond_kinds[bond_idx] == "DOUBLE" and prep.bond_stereo_kinds[bond_idx] in STEREO_KINDS


def atom(prep, idx: int) -> str:
    return f"{idx}:{prep.atom_tokens[idx]}"


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


def extract_side_groups(prep):
    stereo_component_ids = cs._stereo_component_ids(prep)
    sides = []
    edge_to_sides = defaultdict(list)
    stereo_bond_to_sides = defaultdict(list)
    oriented_carrier_edges = set()

    for bond_idx, component_idx in enumerate(stereo_component_ids):
        if component_idx < 0 or not is_stereo_double(prep, bond_idx):
            continue
        begin = prep.bond_begin_atom_indices[bond_idx]
        end = prep.bond_end_atom_indices[bond_idx]
        for endpoint, other in ((begin, end), (end, begin)):
            candidates = tuple(
                n
                for n in prep.neighbors_of(endpoint)
                if n != other and prep.bond_kinds[prep.bond_index(endpoint, n)] in {"SINGLE", "AROMATIC"}
            )
            if not candidates:
                continue
            idx = len(sides)
            sides.append({
                "idx": idx,
                "component": component_idx,
                "stereo_bond": bond_idx,
                "endpoint": endpoint,
                "other": other,
                "candidates": candidates,
            })
            stereo_bond_to_sides[bond_idx].append(idx)
            for n in candidates:
                edge_to_sides[canon(endpoint, n)].append(idx)
                oriented_carrier_edges.add((endpoint, n))

    side_dsu = DSU()
    for side in sides:
        side_dsu.add(side["idx"])
    for ids in stereo_bond_to_sides.values():
        for other in ids[1:]:
            side_dsu.union(ids[0], other)
    for ids in edge_to_sides.values():
        for other in ids[1:]:
            side_dsu.union(ids[0], other)

    oriented_dsu = DSU()
    for node in oriented_carrier_edges:
        oriented_dsu.add(node)
    for side in sides:
        nodes = [(side["endpoint"], n) for n in side["candidates"]]
        for n in nodes[1:]:
            oriented_dsu.union(nodes[0], n)
    for a, b in list(oriented_carrier_edges):
        if (b, a) in oriented_carrier_edges:
            oriented_dsu.union((a, b), (b, a))

    return stereo_component_ids, sides, dict(edge_to_sides), side_dsu.groups(), oriented_dsu.groups()


def directional_edges_from_mol(mol: Chem.Mol):
    out = {}
    for bond in mol.GetBonds():
        token = DIR_TOKENS.get(str(bond.GetBondDir()))
        if not token:
            continue
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        out[canon(begin, end)] = (begin, end, token, str(bond.GetBondDir()))
    return out


def nonstereo_copy(mol: Chem.Mol) -> Chem.Mol:
    copy = Chem.Mol(mol)
    Chem.RemoveStereochemistry(copy)
    return copy


def witness_to_source_atom_map(source_mol: Chem.Mol, witness_mol: Chem.Mol) -> tuple[int, ...]:
    source_ns = nonstereo_copy(source_mol)
    witness_ns = nonstereo_copy(witness_mol)
    match = source_ns.GetSubstructMatch(witness_ns)
    if len(match) != witness_ns.GetNumAtoms():
        raise ValueError("could not map witness atoms back to source atoms")
    return tuple(match)


def remap_directional_edges(edges: dict[tuple[int, int], tuple[int, int, str, str]], atom_map: tuple[int, ...]):
    mapped = {}
    for _edge, (begin, end, token, direction_name) in edges.items():
        mapped_begin = atom_map[begin]
        mapped_end = atom_map[end]
        mapped[canon(mapped_begin, mapped_end)] = (
            mapped_begin,
            mapped_end,
            token,
            direction_name,
        )
    return mapped


def mapped_witness_stereo_bonds(witness_mol: Chem.Mol, atom_map: tuple[int, ...]):
    out = []
    for bond in witness_mol.GetBonds():
        stereo = str(bond.GetStereo())
        if stereo not in STEREO_KINDS:
            continue
        begin = atom_map[bond.GetBeginAtomIdx()]
        end = atom_map[bond.GetEndAtomIdx()]
        stereo_atoms = tuple(atom_map[idx] for idx in bond.GetStereoAtoms())
        out.append(
            {
                "edge": canon(begin, end),
                "begin": begin,
                "end": end,
                "stereo": stereo,
                "stereo_atoms": stereo_atoms,
            }
        )
    return out


def mol_for_expected(expected: str):
    mol = Chem.MolFromSmiles(expected)
    if mol is None:
        raise ValueError(f"RDKit could not parse expected witness {expected!r}")
    return mol


def print_case(raw: dict, writer_cases_by_id: dict):
    source_mol = case_mol(raw, writer_cases_by_id)
    source_prep = prepare_smiles_graph_from_mol_to_smiles_kwargs(
        source_mol,
        surface_kind=CONNECTED_STEREO_SURFACE,
        isomeric_smiles=raw.get("isomeric_smiles", True),
    )
    witness_mol = mol_for_expected(raw["expected"])
    witness_prep = prepare_smiles_graph_from_mol_to_smiles_kwargs(
        witness_mol,
        surface_kind=CONNECTED_STEREO_SURFACE,
        isomeric_smiles=raw.get("isomeric_smiles", True),
    )
    source_dirs = directional_edges_from_mol(source_mol)
    witness_atom_map = witness_to_source_atom_map(source_mol, witness_mol)
    witness_dirs_raw = directional_edges_from_mol(witness_mol)
    witness_dirs = remap_directional_edges(witness_dirs_raw, witness_atom_map)
    witness_stereo_bonds = mapped_witness_stereo_bonds(witness_mol, witness_atom_map)
    comp_ids, sides, edge_to_sides, side_groups, oriented_groups = extract_side_groups(source_prep)

    print("\n" + "=" * 120)
    print(raw["id"])
    print("expected:", raw["expected"])
    print("source canonical:", Chem.MolToSmiles(Chem.Mol(source_mol), canonical=True, isomericSmiles=True))
    print("witness canonical:", Chem.MolToSmiles(Chem.Mol(witness_mol), canonical=True, isomericSmiles=True))
    print("same canonical identity:", Chem.MolToSmiles(Chem.Mol(source_mol), canonical=True, isomericSmiles=True) == Chem.MolToSmiles(Chem.Mol(witness_mol), canonical=True, isomericSmiles=True))
    if witness_atom_map != tuple(range(len(witness_atom_map))):
        print("witness->source atom map:", witness_atom_map)

    print("\nstereo side groups by bond+shared carrier")
    for group in side_groups:
        bonds = sorted({sides[i]["stereo_bond"] for i in group})
        comps = sorted({sides[i]["component"] for i in group})
        carrier_edges = sorted({canon(sides[i]["endpoint"], n) for i in group for n in sides[i]["candidates"]})
        witness_present = [edge for edge in carrier_edges if edge in witness_dirs]
        source_present = [edge for edge in carrier_edges if edge in source_dirs]
        print(f"  sides={group} comps={comps} stereo_bonds={bonds}")
        print(f"    carrier_edges={carrier_edges}")
        print(f"    source directional edges in group={source_present}")
        print(f"    witness directional edges in group={witness_present}")
        for edge in carrier_edges:
            ids = tuple(edge_to_sides.get(edge, ()))
            shared = " shared" if len(ids) > 1 else ""
            sdir = source_dirs.get(edge)
            wdir = witness_dirs.get(edge)
            print(f"      edge {edge}{shared} sides={ids} source_dir={sdir} witness_dir={wdir}")

    print("\noriented carrier symmetry groups")
    for group in oriented_groups:
        undirected = sorted({canon(a,b) for a,b in group})
        witness = [edge for edge in undirected if edge in witness_dirs]
        if len(group) > 1 or witness:
            print(f"  oriented={group} undirected={undirected} witness_dirs={witness}")

    print("\ncurrent prepared source vs parsed witness stereo bonds")
    for bond_idx, comp in enumerate(comp_ids):
        if comp < 0:
            continue
        sb = source_prep.bond_begin_atom_indices[bond_idx]
        se = source_prep.bond_end_atom_indices[bond_idx]
        print(
            f"  source b{bond_idx} comp={comp} {atom(source_prep,sb)}={atom(source_prep,se)} "
            f"kind={source_prep.bond_stereo_kinds[bond_idx]} stereo_atoms={source_prep.bond_stereo_atoms[bond_idx]}"
        )
        matching = [item for item in witness_stereo_bonds if item["edge"] == canon(sb, se)]
        for item in matching:
            selections = []
            if len(item["stereo_atoms"]) == 2:
                begin_sel, end_sel = item["stereo_atoms"]
                selections.append((item["begin"], begin_sel, canon(item["begin"], begin_sel)))
                selections.append((item["end"], end_sel, canon(item["end"], end_sel)))
            print(
                f"    mapped witness same bond begin/end={item['begin']}-{item['end']} "
                f"kind={item['stereo']} stereo_atoms={item['stereo_atoms']} "
                f"selected_edges={[edge for _endpoint, _neighbor, edge in selections]}"
            )
            for endpoint, neighbor, edge in selections:
                matched_sides = [
                    side["idx"]
                    for side in sides
                    if side["endpoint"] == endpoint and neighbor in side["candidates"]
                ]
                print(f"      endpoint {endpoint} selects neighbor {neighbor} edge={edge} sides={matched_sides}")
    witness_comp_ids = cs._stereo_component_ids(witness_prep)
    for bond_idx, comp in enumerate(witness_comp_ids):
        if comp < 0:
            continue
        wb = witness_prep.bond_begin_atom_indices[bond_idx]
        we = witness_prep.bond_end_atom_indices[bond_idx]
        print(
            f"  witness b{bond_idx} comp={comp} {atom(witness_prep,wb)}={atom(witness_prep,we)} "
            f"kind={witness_prep.bond_stereo_kinds[bond_idx]} stereo_atoms={witness_prep.bond_stereo_atoms[bond_idx]}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case_ids", nargs="*")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    payload = json.loads(FIXTURE.read_text())
    writer_cases_by_id = {case.case_id: case for case in load_pinned_writer_membership_cases(rdBase.rdkitVersion)}
    cases = payload["cases"] if args.all else [c for c in payload["cases"] if c["id"] in set(args.case_ids)]
    if not cases:
        cases = payload["cases"][:3]
    for raw in cases:
        print_case(raw, writer_cases_by_id)

if __name__ == "__main__":
    main()
