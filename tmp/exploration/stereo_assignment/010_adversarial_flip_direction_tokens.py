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
CIS_KINDS = {"STEREOCIS", "STEREOZ"}
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


def source_constraints(prep):
    component_ids = cs._stereo_component_ids(prep)
    oriented = set()
    constraints = defaultdict(list)
    source_bond_by_edge = {}
    for bond_idx, component_idx in enumerate(component_ids):
        if component_idx < 0 or not is_stereo_double(prep, bond_idx):
            continue
        begin = prep.bond_begin_atom_indices[bond_idx]
        end = prep.bond_end_atom_indices[bond_idx]
        source_bond_by_edge[canon(begin, end)] = bond_idx
        for endpoint, other in ((begin, end), (end, begin)):
            candidates = tuple(
                n
                for n in prep.neighbors_of(endpoint)
                if n != other
                and prep.bond_kinds[prep.bond_index(endpoint, n)] in {"SINGLE", "AROMATIC"}
            )
            nodes = [(endpoint, n) for n in candidates]
            oriented.update(nodes)
            if len(nodes) == 2:
                constraints[nodes[0]].append((nodes[1], 1))
                constraints[nodes[1]].append((nodes[0], 1))
    for begin, end in list(oriented):
        if (end, begin) in oriented:
            constraints[(begin, end)].append(((end, begin), 1))
    return component_ids, oriented, constraints, source_bond_by_edge


def close_assignments(witness: Chem.Mol, atom_map, oriented, constraints):
    assignments = {}
    for bond in witness.GetBonds():
        token = DIR_TOKENS.get(str(bond.GetBondDir()))
        if not token:
            continue
        begin = atom_map[bond.GetBeginAtomIdx()]
        end = atom_map[bond.GetEndAtomIdx()]
        bit = TOKEN_BIT[token]
        if (begin, end) in oriented:
            assignments[(begin, end)] = bit
        if (end, begin) in oriented:
            assignments[(end, begin)] = bit ^ 1

    queue = deque(assignments)
    while queue:
        node = queue.popleft()
        bit = assignments[node]
        for other, xor in constraints.get(node, ()):
            other_bit = bit ^ xor
            if other in assignments:
                if assignments[other] != other_bit:
                    return None
            else:
                assignments[other] = other_bit
                queue.append(other)
    return assignments


def source_stereo_relation_ok(prep, witness: Chem.Mol, atom_map, assignments, source_bond_by_edge):
    seen_source_bonds = set()
    for bond in witness.GetBonds():
        begin = atom_map[bond.GetBeginAtomIdx()]
        end = atom_map[bond.GetEndAtomIdx()]
        source_bond_idx = source_bond_by_edge.get(canon(begin, end))
        if source_bond_idx is None:
            continue
        seen_source_bonds.add(source_bond_idx)
        stereo_atoms = tuple(atom_map[idx] for idx in bond.GetStereoAtoms())
        if len(stereo_atoms) != 2:
            return False
        left = (begin, stereo_atoms[0])
        right = (end, stereo_atoms[1])
        if left not in assignments or right not in assignments:
            return False
        want_xor = 0 if prep.bond_stereo_kinds[source_bond_idx] in CIS_KINDS else 1
        if (assignments[left] ^ assignments[right]) != want_xor:
            return False
    # Avoid accepting outputs that silently lose a normal source stereo bond.
    normal_source_bonds = {
        idx
        for edge, idx in source_bond_by_edge.items()
        if prep.bond_kinds[idx] == "DOUBLE"
        and all(prep.atom_tokens[a] != "[Fe]" for a in edge)
    }
    return normal_source_bonds <= seen_source_bonds


def constraint_accepts(source: Chem.Mol, witness: Chem.Mol, prep, oriented, constraints, source_bond_by_edge):
    source_ns = nonstereo_copy(source)
    witness_ns = nonstereo_copy(witness)
    for match in source_ns.GetSubstructMatches(witness_ns, uniquify=False, maxMatches=2048):
        assignments = close_assignments(witness, match, oriented, constraints)
        if assignments is None:
            continue
        if source_stereo_relation_ok(prep, witness, match, assignments, source_bond_by_edge):
            return True
    return False


def flipped_once(smiles: str):
    for idx, char in enumerate(smiles):
        if char == "/":
            yield idx, smiles[:idx] + "\\" + smiles[idx + 1 :]
        elif char == "\\":
            yield idx, smiles[:idx] + "/" + smiles[idx + 1 :]


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
    mismatches = []
    for case in load_sources():
        source = source_mol(case)
        if source is None:
            continue
        try:
            prep = prepare_smiles_graph_from_mol_to_smiles_kwargs(
                source,
                surface_kind=CONNECTED_STEREO_SURFACE,
                isomeric_smiles=True,
            )
        except Exception:
            continue
        component_ids, oriented, constraints, source_bond_by_edge = source_constraints(prep)
        if not any(component_idx >= 0 for component_idx in component_ids):
            continue
        source_identity = Chem.MolToSmiles(Chem.Mol(source), canonical=True, isomericSmiles=True)
        outputs = sorted(
            set(
                Chem.MolToRandomSmilesVect(
                    Chem.Mol(source),
                    80,
                    randomSeed=23,
                    isomericSmiles=True,
                )
            )
        )
        for output in outputs:
            for offset, flipped in flipped_once(output):
                witness = Chem.MolFromSmiles(flipped)
                if witness is None:
                    totals["invalid_flip"] += 1
                    continue
                same_identity = (
                    Chem.MolToSmiles(Chem.Mol(witness), canonical=True, isomericSmiles=True)
                    == source_identity
                )
                accepts = constraint_accepts(
                    source,
                    witness,
                    prep,
                    oriented,
                    constraints,
                    source_bond_by_edge,
                )
                totals["flips_checked"] += 1
                totals["same_identity" if same_identity else "different_identity"] += 1
                totals["accepted" if accepts else "rejected"] += 1
                if accepts != same_identity:
                    totals["mismatch"] += 1
                    mismatches.append((case, output, offset, flipped, same_identity, accepts))
                    if len(mismatches) >= 30:
                        break
            if len(mismatches) >= 30:
                break
        if len(mismatches) >= 30:
            break

    print("TOTALS")
    for key, value in totals.most_common():
        print(f"  {key}: {value}")
    print("MISMATCHES")
    for case, output, offset, flipped, same_identity, accepts in mismatches:
        print(
            f"  {case.corpus}:{case.case_id} offset={offset} "
            f"same_identity={same_identity} accepts={accepts}"
        )
        print(f"    source output: {output}")
        print(f"    flipped:       {flipped}")


if __name__ == "__main__":
    main()
