from __future__ import annotations

from rdkit import Chem, rdBase


CASES = (
    "CCC=CC(C=CCC)=C(CO)CC",
    "CC/C=C/C(/C=C/CC)=C(CC)CO",
    r"CC/C=C\C(/C=C/CC)=C(CC)CO",
    r"CC/C=C\C(/C=C\CC)=C(CC)CO",
    r"CC\C=C/C(/C=C/CC)=C(/CC)CO",
    r"CC\C=C/C(/C=C/CC)=C(\CC)CO",
)


def stereo_rows(mol: Chem.Mol) -> tuple[str, ...]:
    rows = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        rows.append(
            f"b{bond.GetIdx()} {bond.GetBeginAtomIdx()}={bond.GetEndAtomIdx()} "
            f"{bond.GetStereo()} atoms={tuple(bond.GetStereoAtoms())} "
            f"dir={bond.GetBondDir()}"
        )
    return tuple(rows)


def random_outputs(mol: Chem.Mol, count: int = 48) -> tuple[str, ...]:
    outputs = []
    for seed in range(count):
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


def print_case(smiles: str, *, legacy: bool) -> None:
    original = Chem.GetUseLegacyStereoPerception()
    Chem.SetUseLegacyStereoPerception(legacy)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("parse failed:", smiles)
            return
        print("-" * 100)
        print("legacy:", legacy)
        print("input:", smiles)
        print("canonical:", Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
        print("parsed:")
        for row in stereo_rows(mol):
            print("  ", row)

        potential = Chem.Mol(mol)
        Chem.FindPotentialStereoBonds(potential, cleanIt=False)
        print("FindPotentialStereoBonds(cleanIt=False):")
        for row in stereo_rows(potential):
            print("  ", row)

        print("random outputs:")
        for output in random_outputs(mol):
            print("  ", output)
    finally:
        Chem.SetUseLegacyStereoPerception(original)


def main() -> None:
    print("rdkit:", rdBase.rdkitVersion)
    print("default legacy:", Chem.GetUseLegacyStereoPerception())
    for smiles in CASES:
        print("=" * 100)
        print("case:", smiles)
        print_case(smiles, legacy=True)
        print_case(smiles, legacy=False)


if __name__ == "__main__":
    main()
