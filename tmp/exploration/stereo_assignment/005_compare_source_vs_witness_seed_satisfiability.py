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

def atom_map(source,witness):
    match=nonstereo_copy(source).GetSubstructMatch(nonstereo_copy(witness))
    if len(match) != witness.GetNumAtoms(): raise RuntimeError("no map")
    return tuple(match)

def molecule_seed_assignments(mol, oriented, atom_map=None):
    out={}
    for b in mol.GetBonds():
        tok=DIR_TOKENS.get(str(b.GetBondDir()))
        if not tok: continue
        begin=b.GetBeginAtomIdx(); end=b.GetEndAtomIdx()
        if atom_map is not None:
            begin=atom_map[begin]; end=atom_map[end]
        if (begin,end) in oriented:
            out[(begin,end)] = TOKEN_BIT[tok]
        if (end,begin) in oriented:
            out[(end,begin)] = TOKEN_BIT[tok] ^ 1
    return out

def constraints(prep):
    comp_ids=cs._stereo_component_ids(prep)
    oriented=set(); cons=defaultdict(list)
    for b,comp in enumerate(comp_ids):
        if comp < 0 or not is_stereo_double(prep,b): continue
        begin=prep.bond_begin_atom_indices[b]; end=prep.bond_end_atom_indices[b]
        for endpoint, other in ((begin,end),(end,begin)):
            cands=tuple(n for n in prep.neighbors_of(endpoint)
                        if n != other and prep.bond_kinds[prep.bond_index(endpoint,n)] in {"SINGLE","AROMATIC"})
            nodes=[(endpoint,n) for n in cands]
            oriented.update(nodes)
            if len(nodes)==2:
                cons[nodes[0]].append((nodes[1],1)); cons[nodes[1]].append((nodes[0],1))
    for a,b in list(oriented):
        if (b,a) in oriented:
            cons[(a,b)].append(((b,a),1))
    return oriented, cons

def sat(seed, cons):
    assign=dict(seed); q=deque(assign)
    while q:
        node=q.popleft(); bit=assign[node]
        for other,xor in cons.get(node,()):
            val=bit^xor
            if other in assign and assign[other] != val:
                return False, (node, other, assign[other], val)
            if other not in assign:
                assign[other]=val; q.append(other)
    return True, None

def main():
    payload=json.loads(FIXTURE.read_text())
    writer_cases_by_id={c.case_id:c for c in load_pinned_writer_membership_cases(rdBase.rdkitVersion)}
    for raw in payload["cases"]:
        source=case_mol(raw, writer_cases_by_id)
        witness=Chem.MolFromSmiles(raw["expected"])
        prep=prepare_smiles_graph_from_mol_to_smiles_kwargs(source, surface_kind=CONNECTED_STEREO_SURFACE, isomeric_smiles=raw.get("isomeric_smiles", True))
        oriented, cons=constraints(prep)
        src_seed=molecule_seed_assignments(source, oriented)
        wit_seed=molecule_seed_assignments(witness, oriented, atom_map(source,witness))
        src_ok, src_conf=sat(src_seed, cons)
        wit_ok, wit_conf=sat(wit_seed, cons)
        current='ok'
        try:
            cs._stereo_side_infos(prep, cs._stereo_component_ids(prep))
        except Exception as e:
            current=f'FAIL:{type(e).__name__}:{e}'
        print(raw['id'])
        print(f"  source_seed_vars={len(src_seed)} sat={src_ok} conflict={src_conf}")
        print(f"  witness_seed_vars={len(wit_seed)} sat={wit_ok} conflict={wit_conf}")
        print(f"  current_preassign={current}")

if __name__ == '__main__':
    main()
