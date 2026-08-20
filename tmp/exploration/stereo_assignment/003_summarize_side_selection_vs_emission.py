from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
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

def directional_edges(mol, atom_map=None):
    out={}
    for b in mol.GetBonds():
        tok=DIR_TOKENS.get(str(b.GetBondDir()))
        if not tok: continue
        begin=b.GetBeginAtomIdx(); end=b.GetEndAtomIdx()
        if atom_map is not None:
            begin=atom_map[begin]; end=atom_map[end]
        out[canon(begin,end)] = (begin,end,tok)
    return out

def witness_stereo_selections(witness, atom_map):
    out={}
    for b in witness.GetBonds():
        if str(b.GetStereo()) not in STEREO_KINDS:
            continue
        begin=atom_map[b.GetBeginAtomIdx()]
        end=atom_map[b.GetEndAtomIdx()]
        atoms=tuple(atom_map[i] for i in b.GetStereoAtoms())
        if len(atoms) != 2:
            continue
        out[canon(begin,end)] = {begin: atoms[0], end: atoms[1]}
    return out

def sides_for(prep):
    comp_ids=cs._stereo_component_ids(prep)
    sides=[]
    edge_to_sides=defaultdict(list)
    for bond_idx, comp in enumerate(comp_ids):
        if comp < 0 or not is_stereo_double(prep,bond_idx): continue
        begin=prep.bond_begin_atom_indices[bond_idx]; end=prep.bond_end_atom_indices[bond_idx]
        for endpoint, other in ((begin,end),(end,begin)):
            cands=tuple(n for n in prep.neighbors_of(endpoint)
                        if n != other and prep.bond_kinds[prep.bond_index(endpoint,n)] in {"SINGLE","AROMATIC"})
            if not cands: continue
            idx=len(sides)
            side={"idx":idx,"bond":bond_idx,"component":comp,"endpoint":endpoint,"other":other,"candidates":cands}
            sides.append(side)
            for n in cands:
                edge_to_sides[canon(endpoint,n)].append(idx)
    return comp_ids, sides, edge_to_sides

def main():
    payload=json.loads(FIXTURE.read_text())
    writer_cases_by_id={c.case_id:c for c in load_pinned_writer_membership_cases(rdBase.rdkitVersion)}
    totals=Counter()
    for raw in payload["cases"]:
        source=case_mol(raw, writer_cases_by_id)
        witness=Chem.MolFromSmiles(raw["expected"])
        atom_map=witness_to_source_atom_map(source,witness)
        prep=prepare_smiles_graph_from_mol_to_smiles_kwargs(source, surface_kind=CONNECTED_STEREO_SURFACE, isomeric_smiles=raw.get("isomeric_smiles", True))
        _comp_ids, sides, edge_to_sides=sides_for(prep)
        dirs=directional_edges(witness, atom_map)
        selections=witness_stereo_selections(witness, atom_map)
        print("\n"+raw["id"])
        for side in sides:
            selected=selections.get(canon(side["endpoint"], side["other"]),{}).get(side["endpoint"])
            emitted=[n for n in side["candidates"] if canon(side["endpoint"],n) in dirs]
            shared_emitted=[n for n in emitted if len(edge_to_sides[canon(side["endpoint"],n)]) > 1]
            relation = "none"
            if selected is None:
                relation = "no-selected"
            elif not emitted:
                relation = "selected-no-visible-token"
            elif selected in emitted:
                relation = "selected-visible"
            else:
                relation = "selected-hidden-other-visible"
            if len(emitted) > 1:
                relation += "+multi-visible"
            totals[relation] += 1
            marker = " *" if relation != "selected-visible" else ""
            print(
                f"  s{side['idx']} b{side['bond']} endpoint={side['endpoint']} cands={side['candidates']} "
                f"selected={selected} emitted={tuple(emitted)} shared_emitted={tuple(shared_emitted)} relation={relation}{marker}"
            )
    print("\nTOTALS")
    for key,count in totals.most_common():
        print(f"  {key}: {count}")

if __name__ == "__main__":
    main()
