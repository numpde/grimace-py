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

from tests.helpers.rdkit_serializer_regressions import load_pinned_serializer_regression_cases
from tests.helpers.rdkit_writer_membership import load_pinned_writer_membership_cases
from tests.rdkit_serialization._support import mol_from_pinned_source


@dataclass(frozen=True)
class SourceCase:
    family: str
    case_id: str
    mol: Chem.Mol
    expected_outputs: tuple[str, ...]


def load_known_gap_cases() -> tuple[SourceCase, ...]:
    path = ROOT / "tests/fixtures/rdkit_known_stereo_gaps/2026.03.1.json"
    payload = json.loads(path.read_text())
    writer_by_id = {
        case.case_id: case for case in load_pinned_writer_membership_cases(rdBase.rdkitVersion)
    }
    cases = []
    for raw in payload["cases"]:
        if raw.get("writer_membership_case_id"):
            source = writer_by_id[raw["writer_membership_case_id"]]
            mol = mol_from_pinned_source(source)
        else:
            if raw.get("molblock") is not None:
                mol = Chem.MolFromMolBlock(raw["molblock"], removeHs=False)
            else:
                mol = Chem.MolFromSmiles(raw["smiles"])
            if mol is None:
                raise ValueError(raw["id"])
        cases.append(
            SourceCase(
                family="known_gap",
                case_id=raw["id"],
                mol=mol,
                expected_outputs=(raw["expected"],),
            )
        )
    return tuple(cases)


def load_writer_cases() -> tuple[SourceCase, ...]:
    return tuple(
        SourceCase(
            family="writer",
            case_id=case.case_id,
            mol=mol_from_pinned_source(case),
            expected_outputs=(case.expected,),
        )
        for case in load_pinned_writer_membership_cases(rdBase.rdkitVersion)
    )


def load_serializer_cases() -> tuple[SourceCase, ...]:
    cases = []
    for case in load_pinned_serializer_regression_cases(rdBase.rdkitVersion):
        cases.append(
            SourceCase(
                family="serializer",
                case_id=case.case_id,
                mol=mol_from_pinned_source(case),
                expected_outputs=case.expected,
            )
        )
    return tuple(cases)


def stereo_name(stereo: Chem.BondStereo) -> str:
    return str(stereo).replace("STEREO", "")


def source_double_bond_key(bond: Chem.Bond) -> tuple[int, int]:
    begin = bond.GetBeginAtomIdx()
    end = bond.GetEndAtomIdx()
    return (begin, end) if begin < end else (end, begin)


def potential_bond_keys(mol: Chem.Mol, *, clean_it: bool) -> set[tuple[int, int]]:
    work = Chem.Mol(mol)
    Chem.FindPotentialStereoBonds(work, cleanIt=clean_it)
    return {
        source_double_bond_key(bond)
        for bond in work.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE
        and bond.GetStereo() == Chem.BondStereo.STEREOANY
    }


def source_to_output_map(source: Chem.Mol, output: Chem.Mol) -> tuple[int, ...] | None:
    match = output.GetSubstructMatch(source, useChirality=False)
    if match:
        return tuple(match)
    source_without_stereo = Chem.Mol(source)
    Chem.RemoveStereochemistry(source_without_stereo)
    match = output.GetSubstructMatch(source_without_stereo, useChirality=False)
    return tuple(match) if match else None


def mapped_output_bond(
    source_key: tuple[int, int],
    output: Chem.Mol,
    atom_map: tuple[int, ...],
) -> Chem.Bond | None:
    return output.GetBondBetweenAtoms(atom_map[source_key[0]], atom_map[source_key[1]])


def output_state_for_source_bond(
    source: Chem.Mol,
    output_smiles: str,
    source_key: tuple[int, int],
) -> str:
    output = Chem.MolFromSmiles(output_smiles)
    if output is None:
        return "parse_failed"
    atom_map = source_to_output_map(source, output)
    if atom_map is None:
        return "connectivity_mismatch"
    bond = mapped_output_bond(source_key, output, atom_map)
    if bond is None:
        return "bond_missing"
    stereo_atoms = tuple(bond.GetStereoAtoms())
    mapped_stereo_atoms = []
    for atom_idx in stereo_atoms:
        try:
            mapped_stereo_atoms.append(atom_map.index(atom_idx))
        except ValueError:
            mapped_stereo_atoms.append(-1)
    return f"{stereo_name(bond.GetStereo())}:{tuple(mapped_stereo_atoms)}"


def random_outputs(mol: Chem.Mol, *, samples: int) -> tuple[str, ...]:
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


def source_bond_summary(mol: Chem.Mol) -> dict[tuple[int, int], str]:
    potential_clean_false = potential_bond_keys(mol, clean_it=False)
    potential_clean_true = potential_bond_keys(mol, clean_it=True)
    summary = {}
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        key = source_double_bond_key(bond)
        labels = []
        if bond.GetStereo() != Chem.BondStereo.STEREONONE:
            labels.append(f"specified:{stereo_name(bond.GetStereo())}:{tuple(bond.GetStereoAtoms())}")
        if key in potential_clean_false:
            labels.append("potential_clean_false")
        if key in potential_clean_true:
            labels.append("potential_clean_true")
        summary[key] = ",".join(labels) if labels else "plain_double"
    return summary


def audit_case(case: SourceCase) -> None:
    source_summary = source_bond_summary(case.mol)
    interesting = {
        key: label
        for key, label in source_summary.items()
        if label != "plain_double"
    }
    if not interesting:
        return

    sampled = random_outputs(case.mol, samples=256)
    outputs = tuple(dict.fromkeys(case.expected_outputs + sampled))
    per_bond_states: dict[tuple[int, int], Counter[str]] = {
        key: Counter(
            output_state_for_source_bond(case.mol, output, key)
            for output in outputs
        )
        for key in interesting
    }
    changed_bonds = {
        key: states
        for key, states in per_bond_states.items()
        if len(states) > 1 or next(iter(states)) == "NONE:()"
    }

    if not changed_bonds and case.family != "known_gap":
        return

    print("=" * 120)
    print(f"{case.family}:{case.case_id}")
    print("canonical:", Chem.MolToSmiles(case.mol, canonical=True, isomericSmiles=True))
    print("source interesting bonds:")
    for key, label in interesting.items():
        print(" ", key, label)
    print("outputs checked:", len(outputs))
    for key, states in per_bond_states.items():
        print(" ", key, dict(states.most_common()))
    for output in case.expected_outputs:
        print(" expected:", output)


def main() -> None:
    print("rdkit:", rdBase.rdkitVersion)
    print("legacy:", Chem.GetUseLegacyStereoPerception())
    all_cases = load_known_gap_cases() + load_writer_cases() + load_serializer_cases()
    for case in all_cases:
        audit_case(case)


if __name__ == "__main__":
    main()
