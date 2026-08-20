from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from rdkit import Chem, rdBase

from tests.helpers.rdkit_writer_membership import load_pinned_writer_membership_cases
from tests.rdkit_serialization._support import mol_from_pinned_source


E_LIKE = {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS}
Z_LIKE = {Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS}


@dataclass(frozen=True)
class Case:
    case_id: str
    smiles: str | None
    molblock: str | None
    expected: tuple[str, ...]


def load_cases() -> tuple[Case, ...]:
    writer_by_id = {
        case.case_id: case for case in load_pinned_writer_membership_cases(rdBase.rdkitVersion)
    }
    path = ROOT / "tests/fixtures/rdkit_known_stereo_gaps/2026.03.1.json"
    payload = json.loads(path.read_text())
    cases = []
    for raw in payload["cases"]:
        if raw.get("writer_membership_case_id"):
            writer = writer_by_id[raw["writer_membership_case_id"]]
            smiles = writer.smiles
            molblock = writer.molblock
        else:
            smiles = raw.get("smiles")
            molblock = raw.get("molblock")
        cases.append(
            Case(
                case_id=raw["id"],
                smiles=smiles,
                molblock=molblock,
                expected=(raw["expected"],),
            )
        )
    for case_id in (
        "manual_bond_stereo_really_difficult_01",
        "manual_bond_stereo_really_difficult_02",
        "manual_bond_stereo_really_difficult_03",
        "manual_bond_stereo_atoms_passing_surface_01",
        "manual_bond_stereo_atoms_passing_surface_02",
        "dataset_regression_02_porphyrin_like_fragment",
    ):
        writer = writer_by_id[case_id]
        cases.append(
            Case(
                case_id=writer.case_id,
                smiles=writer.smiles,
                molblock=writer.molblock,
                expected=(writer.expected,),
            )
        )
    return tuple(cases)


def mol_from_case(case: Case) -> Chem.Mol:
    if case.molblock is not None:
        mol = Chem.MolFromMolBlock(case.molblock, removeHs=False)
    else:
        mol = Chem.MolFromSmiles(case.smiles)
    if mol is None:
        raise ValueError(case.case_id)
    return mol


def stereo_class(stereo: Chem.BondStereo) -> str:
    if stereo in E_LIKE:
        return "E"
    if stereo in Z_LIKE:
        return "Z"
    if stereo == Chem.BondStereo.STEREONONE:
        return "NONE"
    if stereo == Chem.BondStereo.STEREOANY:
        return "ANY"
    return str(stereo)


def specified_source_bonds(mol: Chem.Mol) -> tuple[tuple[int, int, str], ...]:
    out = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        cls = stereo_class(bond.GetStereo())
        if cls in {"E", "Z"}:
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            out.append((begin, end, cls))
    return tuple(out)


def strip_stereo(mol: Chem.Mol) -> Chem.Mol:
    copy = Chem.Mol(mol)
    Chem.RemoveStereochemistry(copy)
    return copy


def all_matches(source: Chem.Mol, output: Chem.Mol, *, max_matches: int = 4096) -> tuple[tuple[int, ...], ...]:
    query = strip_stereo(source)
    target = strip_stereo(output)
    return tuple(
        target.GetSubstructMatches(
            query,
            uniquify=False,
            maxMatches=max_matches,
            useChirality=False,
        )
    )


def match_state(
    output: Chem.Mol,
    match: tuple[int, ...],
    source_bonds: tuple[tuple[int, int, str], ...],
) -> tuple[str, ...]:
    states = []
    for begin, end, source_cls in source_bonds:
        mapped_begin = match[begin]
        mapped_end = match[end]
        output_bond = output.GetBondBetweenAtoms(mapped_begin, mapped_end)
        if output_bond is None:
            states.append(f"{begin}={end}:{source_cls}->MISSING")
        else:
            states.append(
                f"{begin}={end}:{source_cls}->{stereo_class(output_bond.GetStereo())}"
            )
    return tuple(states)


def preserves_source_stereo(state: tuple[str, ...]) -> bool:
    return all(part.split("->", 1)[0].rsplit(":", 1)[1] == part.split("->", 1)[1] for part in state)


def random_outputs(mol: Chem.Mol, *, samples: int = 256) -> tuple[str, ...]:
    outputs = []
    for seed in range(samples):
        Chem.rdBase.SeedRandomNumberGenerator(seed)
        outputs.append(
            Chem.MolToSmiles(
                Chem.Mol(mol),
                canonical=False,
                doRandom=True,
                isomericSmiles=True,
            )
        )
    return tuple(sorted(set(outputs)))


def audit_case(case: Case) -> None:
    source = mol_from_case(case)
    source_bonds = specified_source_bonds(source)
    if not source_bonds:
        return
    outputs = tuple(dict.fromkeys(case.expected + random_outputs(source)))
    impossible = []
    possible_state_counts: Counter[tuple[str, ...]] = Counter()
    match_count_counter: Counter[int] = Counter()
    for output_smiles in outputs:
        output = Chem.MolFromSmiles(output_smiles)
        if output is None:
            impossible.append((output_smiles, "parse_failed"))
            continue
        matches = all_matches(source, output)
        match_count_counter[len(matches)] += 1
        states = {match_state(output, match, source_bonds) for match in matches}
        for state in states:
            possible_state_counts[state] += 1
        if not any(preserves_source_stereo(state) for state in states):
            impossible.append((output_smiles, tuple(sorted(states))[:8]))

    print("=" * 120)
    print("case:", case.case_id)
    print("source canonical:", Chem.MolToSmiles(source, canonical=True, isomericSmiles=True))
    print("source specified bonds:", source_bonds)
    print("outputs:", len(outputs), "match_counts:", dict(sorted(match_count_counter.items())))
    print("possible mapped state classes:", len(possible_state_counts))
    for state, count in possible_state_counts.most_common(8):
        print(" ", count, state, "preserves=", preserves_source_stereo(state))
    if impossible:
        print("NO preserving match:", len(impossible))
        for output_smiles, states in impossible[:10]:
            print("  ", output_smiles)
            print("    states:", states)


def main() -> None:
    print("rdkit:", rdBase.rdkitVersion)
    print("legacy:", Chem.GetUseLegacyStereoPerception())
    for case in load_cases():
        audit_case(case)


if __name__ == "__main__":
    main()
