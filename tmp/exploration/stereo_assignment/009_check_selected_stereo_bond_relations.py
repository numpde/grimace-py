from __future__ import annotations

import json, sys
from collections import defaultdict, deque, Counter
from pathlib import Path
ROOT=Path.cwd(); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'python'))
from rdkit import Chem, rdBase
from grimace._reference.prepared_graph import CONNECTED_STEREO_SURFACE, prepare_smiles_graph_from_mol_to_smiles_kwargs
from grimace._reference.rooted import connected_stereo as cs
from tests.helpers.rdkit_writer_membership import load_pinned_writer_membership_cases
from tests.rdkit_serialization._support import mol_from_pinned_source
STEREO={'STEREOCIS','STEREOZ','STEREOE','STEREOTRANS'}; CIS={'STEREOCIS','STEREOZ'}; TRANS={'STEREOE','STEREOTRANS'}
DIR={'ENDUPRIGHT':'/','ENDDOWNRIGHT':'\\'}; BIT={'/':0,'\\':1}
def canon(a,b): return (a,b) if a<b else (b,a)
def ns(m): c=Chem.Mol(m); Chem.RemoveStereochemistry(c); return c
def is_sd(p,b): return p.bond_kinds[b]=='DOUBLE' and p.bond_stereo_kinds[b] in STEREO
def mol_from(smiles,molblock):
    if molblock: return Chem.MolFromMolBlock(molblock, removeHs=False)
    return Chem.MolFromSmiles(smiles)
def source_cases():
    out=[]; seen=set(); writers={c.case_id:c for c in load_pinned_writer_membership_cases(rdBase.rdkitVersion)}
    def add(corpus,id,smiles,molblock,expected=None):
        key=(corpus,id,expected)
        if key not in seen: seen.add(key); out.append((corpus,id,smiles,molblock,expected))
    known=json.loads((ROOT/'tests/fixtures/rdkit_known_stereo_gaps/2026.03.1.json').read_text())
    for r in known['cases']:
        if r.get('writer_membership_case_id'):
            w=writers[r['writer_membership_case_id']]; add('known',r['id'],w.smiles,w.molblock,r['expected'])
        else: add('known',r['id'],r.get('smiles'),r.get('molblock'),r['expected'])
    for w in writers.values():
        if w.isomeric_smiles: add('writer',w.case_id,w.smiles,w.molblock,w.expected)
    for path in sorted((ROOT/'tests/fixtures/rdkit_serializer_regressions/2026.03.1').glob('*.json')):
        payload=json.loads(path.read_text())
        for r in payload['cases']:
            if not r.get('isomeric_smiles', True): continue
            for idx,e in enumerate(r.get('expected',[]) if isinstance(r.get('expected'),list) else []):
                add(path.name,f"{r['id']}#{idx}",r.get('smiles'),r.get('molblock'),e)
    steroid=json.loads((ROOT/'tests/fixtures/rdkit_stereo_regressions/steroid_ring_coupled_component.json').read_text())
    add('negative', 'steroid_expected_member', steroid['input_smiles'], None, steroid['expected_member'])
    add('negative', 'steroid_rejected_member', steroid['input_smiles'], None, steroid['rejected_member'])
    return out
def base_constraints(p):
    ids=cs._stereo_component_ids(p); oriented=set(); cons=defaultdict(list)
    for b,c in enumerate(ids):
        if c<0 or not is_sd(p,b): continue
        begin=p.bond_begin_atom_indices[b]; end=p.bond_end_atom_indices[b]
        for ep,other in ((begin,end),(end,begin)):
            cands=tuple(n for n in p.neighbors_of(ep) if n!=other and p.bond_kinds[p.bond_index(ep,n)] in {'SINGLE','AROMATIC'})
            nodes=[(ep,n) for n in cands]; oriented.update(nodes)
            if len(nodes)==2:
                cons[nodes[0]].append((nodes[1],1,'same_endpoint_alt')); cons[nodes[1]].append((nodes[0],1,'same_endpoint_alt'))
    for a,b in list(oriented):
        if (b,a) in oriented: cons[(a,b)].append(((b,a),1,'reverse_carrier'))
    return ids,oriented,cons
def seed_dirs(wit,oriented,map):
    out={}
    for b in wit.GetBonds():
        tok=DIR.get(str(b.GetBondDir()))
        if not tok: continue
        a=map[b.GetBeginAtomIdx()]; c=map[b.GetEndAtomIdx()]; bit=BIT[tok]
        if (a,c) in oriented: out[(a,c)]=bit
        if (c,a) in oriented: out[(c,a)]=bit^1
    return out
def selected_relation_constraints(wit,map):
    rel=[]
    for b in wit.GetBonds():
        stereo=str(b.GetStereo())
        if stereo not in STEREO: continue
        begin=map[b.GetBeginAtomIdx()]; end=map[b.GetEndAtomIdx()]
        atoms=tuple(map[i] for i in b.GetStereoAtoms())
        if len(atoms)!=2: continue
        left=(begin,atoms[0]); right=(end,atoms[1])
        xor=0 if stereo in CIS else 1
        rel.append((left,right,xor,f'{stereo}_selected'))
    return rel
def sat(seed,cons,extra):
    cons2=defaultdict(list,{k:list(v) for k,v in cons.items()})
    for a,b,x,k in extra:
        cons2[a].append((b,x,k)); cons2[b].append((a,x,k))
    assign=dict(seed); q=deque(assign)
    # extra constraints can connect unseeded selected hidden vars, seed one side of every component lazily only for consistency closure.
    for a,b,x,k in extra:
        if a not in assign and b not in assign:
            assign[a]=0; q.append(a)
    while q:
        u=q.popleft(); bit=assign[u]
        for v,x,k in cons2.get(u,()):
            val=bit^x
            if v in assign:
                if assign[v]!=val: return False,(u,v,k,assign[v],val)
            else:
                assign[v]=val; q.append(v)
    return True,None
def any_map(src,wit,p,oriented,cons):
    for match in ns(src).GetSubstructMatches(ns(wit), uniquify=False, maxMatches=4096):
        extra=selected_relation_constraints(wit,match)
        ok,conf=sat(seed_dirs(wit,oriented,match),cons,extra)
        if ok: return True,conf,match
    return False,'no satisfying map',None
def main():
    counts=Counter(); fails=[]
    for corpus,id,smiles,molblock,expected in source_cases():
        if expected is None: continue
        src=mol_from(smiles,molblock); wit=Chem.MolFromSmiles(expected)
        if src is None or wit is None: continue
        try: p=prepare_smiles_graph_from_mol_to_smiles_kwargs(src,surface_kind=CONNECTED_STEREO_SURFACE,isomeric_smiles=True)
        except Exception: continue
        ids,oriented,cons=base_constraints(p)
        if not any(c>=0 for c in ids): continue
        ok,conf,match=any_map(src,wit,p,oriented,cons)
        same=Chem.MolToSmiles(src,canonical=True,isomericSmiles=True)==Chem.MolToSmiles(wit,canonical=True,isomericSmiles=True)
        counts['checked']+=1; counts['ok' if ok else 'fail']+=1; counts['same' if same else 'different']+=1
        if (same and not ok) or (not same and ok): fails.append((corpus,id,'same' if same else 'different','ok' if ok else 'fail',conf,expected))
    print('COUNTS')
    for k,v in counts.most_common(): print(' ',k,v)
    print('MISMATCHES')
    for row in fails[:50]: print(' ',row)
if __name__=='__main__': main()
