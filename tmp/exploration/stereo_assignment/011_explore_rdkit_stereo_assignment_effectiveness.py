from __future__ import annotations

from dataclasses import dataclass
from itertools import count

from rdkit import Chem, rdBase


@dataclass(frozen=True)
class Case:
    case_id: str
    smiles: str
    note: str


CASES = (
    Case(
        "simple_trans",
        "F/C=C/F",
        "ordinary parsed directional alkene",
    ),
    Case(
        "difficult_two_bond_cis_trans",
        r"CC/C=C\C(CO)=C(\C)CC",
        "manual difficult case from RDKit rough_test.py",
    ),
    Case(
        "three_double_bond_dependency",
        r"CC\C=C/C(/C=C/CC)=C(/CC)CO",
        "manual stereo-atom missing surface case",
    ),
    Case(
        "github3967_ring_directional",
        r"C1=CC/C=C2C3=C/CC=CC=CC\3C\2C=C1",
        "known ring-closure carrier propagation witness",
    ),
    Case(
        "dative_carbonyl",
        r"C1CCC2=[N]1[Fe](\[O]=C(\C)/C=C/C1CCCC1)[N]1=C(CCC1)C2",
        "dative/carbonyl source annotation quirk",
    ),
)


def parse_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse {smiles!r}")
    return mol


def bond_label(mol: Chem.Mol, bond: Chem.Bond) -> str:
    begin = bond.GetBeginAtom()
    end = bond.GetEndAtom()
    return (
        f"b{bond.GetIdx()}:{begin.GetIdx()}{begin.GetSymbol()}-"
        f"{end.GetIdx()}{end.GetSymbol()}:{bond.GetBondType()}"
    )


def carrier_desc(mol: Chem.Mol, stereo_bond: Chem.Bond) -> str:
    stereo_atoms = tuple(stereo_bond.GetStereoAtoms())
    if len(stereo_atoms) != 2:
        return "via <missing stereo atoms>"
    pieces = []
    for center_idx, carrier_idx in zip(
        (stereo_bond.GetBeginAtomIdx(), stereo_bond.GetEndAtomIdx()),
        stereo_atoms,
    ):
        carrier = mol.GetAtomWithIdx(carrier_idx)
        carrier_bond = mol.GetBondBetweenAtoms(center_idx, carrier_idx)
        if carrier_bond is None:
            pieces.append(f"{carrier_idx}{carrier.GetSymbol()}(MISSING)")
        else:
            pieces.append(
                f"{carrier_idx}{carrier.GetSymbol()}"
                f"({carrier_bond.GetBondType()},{carrier_bond.GetBondDir()})"
            )
    return "via " + ",".join(pieces)


def stereo_bonds(mol: Chem.Mol) -> list[Chem.Bond]:
    return [
        bond
        for bond in mol.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE
        and bond.GetStereo() != Chem.BondStereo.STEREONONE
    ]


def stereo_signature(mol: Chem.Mol) -> tuple[str, ...]:
    return tuple(
        sorted(
            f"{bond_label(mol, bond)} {bond.GetStereo()} {carrier_desc(mol, bond)}"
            for bond in stereo_bonds(mol)
        )
    )


def directional_signature(mol: Chem.Mol) -> tuple[str, ...]:
    return tuple(
        sorted(
            f"{bond_label(mol, bond)} {bond.GetBondDir()}"
            for bond in mol.GetBonds()
            if bond.GetBondDir() != Chem.BondDir.NONE
        )
    )


def canonical(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def random_samples(mol: Chem.Mol, limit: int = 16) -> tuple[str, ...]:
    samples = []
    for seed in range(limit):
        Chem.rdBase.SeedRandomNumberGenerator(seed)
        samples.append(
            Chem.MolToSmiles(
                mol,
                canonical=False,
                doRandom=True,
                isomericSmiles=True,
            )
        )
    return tuple(dict.fromkeys(samples))


def mol_without_one_stereo_bond(mol: Chem.Mol, bond_idx: int) -> Chem.Mol:
    copy = Chem.Mol(mol)
    bond = copy.GetBondWithIdx(bond_idx)
    bond.SetStereo(Chem.BondStereo.STEREONONE)
    return copy


def effective_by_canonical_clear(mol: Chem.Mol) -> tuple[str, ...]:
    base = canonical(mol)
    results = []
    for bond in stereo_bonds(mol):
        cleared = mol_without_one_stereo_bond(mol, bond.GetIdx())
        changed = canonical(cleared) != base
        results.append(f"{bond_label(mol, bond)} clear_changes_canonical={changed}")
    return tuple(results)


def print_case(case: Case) -> None:
    print("=" * 88)
    print(f"{case.case_id}: {case.note}")
    print(f"rdkit={rdBase.rdkitVersion}")
    mol = parse_smiles(case.smiles)
    print("input:", case.smiles)
    print("canonical:", canonical(mol))
    print("source stereos:")
    for line in stereo_signature(mol):
        print("  ", line)
    print("source directed bonds:")
    for line in directional_signature(mol):
        print("  ", line)
    print("clear-one-stereo effect:")
    for line in effective_by_canonical_clear(mol):
        print("  ", line)

    roundtrip = parse_smiles(canonical(mol))
    print("roundtrip stereos:")
    for line in stereo_signature(roundtrip):
        print("  ", line)
    print("roundtrip directed bonds:")
    for line in directional_signature(roundtrip):
        print("  ", line)

    print("random samples:")
    for idx, sample in zip(count(), random_samples(mol, limit=12)):
        sample_mol = parse_smiles(sample)
        print(
            f"  {idx:02d}",
            sample,
            "stereo_count=",
            len(stereo_bonds(sample_mol)),
            "dirs=",
            len(directional_signature(sample_mol)),
        )


def main() -> None:
    for case in CASES:
        print_case(case)


if __name__ == "__main__":
    main()
