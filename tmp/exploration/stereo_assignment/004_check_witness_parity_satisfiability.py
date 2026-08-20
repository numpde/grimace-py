from __future__ import annotations

import json
import sys
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from rdkit import Chem, rdBase

from grimace._reference.prepared_graph import CONNECTED_STEREO_SURFACE, prepare_smiles_graph_from_mol_to_smiles_kwargs
from grimace._reference.rooted import connected_stereo as cs
from tests.helpers.rdkit_writer_membership import load_pinned_writer_membership_cases
from tests.rdkit_serialization._support import mol_from_pinned_source

FIXTURE = Path("tests/fixtures/rdkit_known_stereo_gaps/2026.03.1.json")
STEREO_KINDS = {"STEREOCIS", "STEREOZ", "STEREOE", "STEREOTRANS"}
DIR_TOKENS = {"ENDUPRIGHT": "/", "ENDDOWNRIGHT": "\\"}
TOKEN_BIT = {"/": 0, "\\": 1}
BIT_TOKEN = {0: "/", 1: "\\"}


def canon(a,b): return (a,b) if a < b else (b,a)
def is_stereo_double(prep,b): return prep.bond_kinds[b] == "DOUBLE" and prep.bond_stereo_kinds[b] in STEREO_KINDS

def case_mol(raw, writer_cases_by_id):
    if raw.get("writer_membership_case_id"):
        return mol_from_pinned_source(writer_cases_by_id[raw["writer_membership_case_id"]])
    if raw.get("molblock"):
        return Chem.MolFromMolBlock(raw["molblock"], removeHs=False)
    return Chem.MolFromSmiles(raw["smiles"])

def nonstereo_copy(mol):
    c=Chem.Mol(mol); Chem.RemoveStereochemistry(c); return c

def witness_to_source_atom_map(source,witness):
    match=nonstereo_copy(source).GetSubstructMatch(nonstereo_copy(witness))
    if len(match) != witness.GetNumAtoms():
        raise RuntimeError("no atom map")
    return tuple(match)

def witness_emitted_assignments(witness, atom_map):
    out={}
    for b in witness.GetBonds():
        tok=DIR_TOKENS.get(str(b.GetBondDir()))
        if not tok: continue
        begin=atom_map[b.GetBeginAtomIdx()]
        end=atom_map[b.GetEndAtomIdx()]
        bit=TOKEN_BIT[tok]
        # A slash/backslash token reverses when the edge is traversed backward.
        out[(begin,end)] = bit
        out[(end,begin)] = bit ^ 1
    return out

def witness_stereo_selections(witness, atom_map):
    out={}
    for b in witness.GetBonds():
        if str(b.GetStereo()) not in STEREO_KINDS:
            continue
        begin=atom_map[b.GetBeginAtomIdx()]
        end=atom_map[b.GetEndAtomIdx()]
        atoms=tuple(atom_map[i] for i in b.GetStereoAtoms())
        if len(atoms) == 2:
            out[canon(begin,end)] = {begin: atoms[0], end: atoms[1]}
    return out

def sides_and_constraints(prep):
    comp_ids=cs._stereo_component_ids(prep)
    sides=[]
    constraints=defaultdict(list)
    oriented=set()
    for bond_idx, comp in enumerate(comp_ids):
        if comp < 0 or not is_stereo_double(prep,bond_idx): continue
        begin=prep.bond_begin_atom_indices[bond_idx]; end=prep.bond_end_atom_indices[bond_idx]
        for endpoint, other in ((begin,end),(end,begin)):
            cands=tuple(n for n in prep.neighbors_of(endpoint)
                        if n != other and prep.bond_kinds[prep.bond_index(endpoint,n)] in {"SINGLE","AROMATIC"})
            if not cands: continue
            sides.append({"idx":len(sides),"bond":bond_idx,"component":comp,"endpoint":endpoint,"other":other,"candidates":cands})
            nodes=[(endpoint,n) for n in cands]
            for node in nodes:
                oriented.add(node)
            if len(nodes)==2:
                a,b=nodes
                constraints[a].append((b,1,"same_endpoint_alternatives"))
                constraints[b].append((a,1,"same_endpoint_alternatives"))
    for a,b in list(oriented):
        if (b,a) in oriented:
            constraints[(a,b)].append(((b,a),1,"reverse_same_carrier"))
    return comp_ids, sides, constraints, oriented

def propagate(assignments, constraints):
    assignments=dict(assignments)
    q=deque(assignments)
    while q:
        node=q.popleft()
        bit=assignments[node]
        for other, xor, _kind in constraints.get(node,()):
            other_bit=bit ^ xor
            if other in assignments:
                if assignments[other] != other_bit:
                    return False, assignments, (node, other, assignments[other], other_bit)
            else:
                assignments[other]=other_bit
                q.append(other)
    return True, assignments, None

def main():
    payload=json.loads(FIXTURE.read_text())
    writer_cases_by_id={c.case_id:c for c in load_pinned_writer_membership_cases(rdBase.rdkitVersion)}
    failures=0
    for raw in payload["cases"]:
        source=case_mol(raw, writer_cases_by_id)
        witness=Chem.MolFromSmiles(raw["expected"])
        atom_map=witness_to_source_atom_map(source,witness)
        prep=prepare_smiles_graph_from_mol_to_smiles_kwargs(source, surface_kind=CONNECTED_STEREO_SURFACE, isomeric_smiles=raw.get("isomeric_smiles", True))
        comp_ids, sides, constraints, oriented=sides_and_constraints(prep)
        seed={node:bit for node,bit in witness_emitted_assignments(witness, atom_map).items() if node in oriented}
        ok, inferred, conflict=propagate(seed, constraints)
        selections=witness_stereo_selections(witness, atom_map)
        if not ok:
            failures += 1
        print("\n"+raw["id"])
        print("  seeds", len(seed)//2 if seed else 0, "oriented_vars", len(oriented), "satisfiable", ok, "conflict", conflict)
        for side in sides:
            selected=selections.get(canon(side["endpoint"],side["other"]),{}).get(side["endpoint"])
            selected_node=(side["endpoint"], selected) if selected is not None else None
            emitted=[(side["endpoint"],n) for n in side["candidates"] if (side["endpoint"],n) in seed]
            hidden=[] if selected_node is None or selected_node in seed else [selected_node]
            selected_token = BIT_TOKEN[inferred[selected_node]] if selected_node in inferred else None
            emitted_tokens = tuple((node, BIT_TOKEN[inferred[node]]) for node in emitted if node in inferred)
            print(f"    s{side['idx']} b{side['bond']} endpoint={side['endpoint']} cands={side['candidates']} selected={selected_node} selected_token={selected_token} emitted={emitted_tokens} hidden={hidden}")
    print("\nfailures", failures)

if __name__ == "__main__":
    main()
