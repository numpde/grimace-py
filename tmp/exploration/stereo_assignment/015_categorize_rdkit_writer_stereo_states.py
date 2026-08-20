from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from rdkit import Chem, rdBase


@dataclass(frozen=True)
class Case:
    case_id: str
    smiles: str
    tracked_double_bonds: tuple[tuple[int, int], ...]


CASES = (
    Case(
        "unspecified_three_double_bond_dependency",
        "CCC=CC(C=CCC)=C(CO)CC",
        ((2, 3), (5, 6), (4, 9)),
    ),
    Case(
        "outer_EE_central_unspecified",
        "CC/C=C/C(/C=C/CC)=C(CC)CO",
        ((2, 3), (5, 6), (4, 9)),
    ),
    Case(
        "outer_EZ_central_unspecified",
        r"CC/C=C\C(/C=C/CC)=C(CC)CO",
        ((2, 3), (5, 6), (4, 9)),
    ),
    Case(
        "outer_EZ_central_Z",
        r"CC\C=C/C(/C=C/CC)=C(/CC)CO",
        ((2, 3), (5, 6), (4, 9)),
    ),
    Case(
        "outer_EZ_central_E",
        r"CC\C=C/C(/C=C/CC)=C(\CC)CO",
        ((2, 3), (5, 6), (4, 9)),
    ),
)


def parse_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(smiles)
    return mol


def random_outputs(mol: Chem.Mol, seeds: int = 512) -> tuple[str, ...]:
    outputs = []
    for seed in range(seeds):
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


def source_to_output_atom_map(source: Chem.Mol, output: Chem.Mol) -> tuple[int, ...]:
    match = output.GetSubstructMatch(source, useChirality=False)
    if not match:
        raise AssertionError("output did not match source connectivity")
    return tuple(match)


def mapped_double_bond_states(
    source: Chem.Mol,
    output: Chem.Mol,
    tracked_double_bonds: tuple[tuple[int, int], ...],
) -> tuple[str, ...]:
    mapping = source_to_output_atom_map(source, output)
    states = []
    for begin_idx, end_idx in tracked_double_bonds:
        output_begin = mapping[begin_idx]
        output_end = mapping[end_idx]
        bond = output.GetBondBetweenAtoms(output_begin, output_end)
        if bond is None:
            states.append(f"{begin_idx}={end_idx}:MISSING")
            continue
        stereo_atoms = tuple(bond.GetStereoAtoms())
        mapped_stereo_atoms = []
        for atom_idx in stereo_atoms:
            try:
                mapped_stereo_atoms.append(mapping.index(atom_idx))
            except ValueError:
                mapped_stereo_atoms.append(-1)
        states.append(
            f"{begin_idx}={end_idx}:{bond.GetStereo()}:"
            f"{tuple(mapped_stereo_atoms)}"
        )
    return tuple(states)


def visible_direction_count(output: Chem.Mol) -> int:
    return sum(
        1
        for bond in output.GetBonds()
        if bond.GetBondDir()
        in (Chem.BondDir.ENDUPRIGHT, Chem.BondDir.ENDDOWNRIGHT)
    )


def print_case(case: Case, *, legacy: bool) -> None:
    original = Chem.GetUseLegacyStereoPerception()
    Chem.SetUseLegacyStereoPerception(legacy)
    try:
        source = parse_smiles(case.smiles)
        outputs = random_outputs(source)
        state_counts: Counter[tuple[str, ...]] = Counter()
        examples: dict[tuple[str, ...], str] = {}
        direction_counts: Counter[int] = Counter()
        for output_smiles in outputs:
            output = parse_smiles(output_smiles)
            states = mapped_double_bond_states(
                source,
                output,
                case.tracked_double_bonds,
            )
            state_counts[states] += 1
            examples.setdefault(states, output_smiles)
            direction_counts[visible_direction_count(output)] += 1

        print("=" * 100)
        print("case:", case.case_id)
        print("legacy:", legacy)
        print("input:", case.smiles)
        print("canonical:", Chem.MolToSmiles(source, canonical=True, isomericSmiles=True))
        print("unique random outputs:", len(outputs))
        print("visible direction counts:", dict(sorted(direction_counts.items())))
        print("mapped stereo-state classes:", len(state_counts))
        for states, count in state_counts.most_common():
            print(" ", count, states)
            print("    example:", examples[states])
    finally:
        Chem.SetUseLegacyStereoPerception(original)


def main() -> None:
    print("rdkit:", rdBase.rdkitVersion)
    print("default legacy:", Chem.GetUseLegacyStereoPerception())
    for case in CASES:
        print_case(case, legacy=True)
        print_case(case, legacy=False)


if __name__ == "__main__":
    main()
