from __future__ import annotations

from itertools import product

from rdkit import Chem, rdBase


STEREOS = (
    ("E", Chem.BondStereo.STEREOTRANS),
    ("Z", Chem.BondStereo.STEREOCIS),
)


SCAFFOLDS = (
    "CCC=CC(C=CCC)=C(CO)CC",
    "CCC=CC(C=CCC)=C(C)CO",
    "CCC=CC(C=CCC)=C(O)CC",
    "CCCC=CC(C=CCCC)=C(CO)CC",
    "CC=CC(C=CC)=C(CO)CC",
    "CCC=CC(C=CCC)=C(F)Cl",
)


def parse_smiles(smiles: str) -> Chem.Mol | None:
    return Chem.MolFromSmiles(smiles)


def potential_stereo_bonds(mol: Chem.Mol) -> tuple[int, ...]:
    work = Chem.Mol(mol)
    Chem.FindPotentialStereoBonds(work)
    return tuple(
        bond.GetIdx()
        for bond in work.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE
        and bond.GetStereo() == Chem.BondStereo.STEREOANY
    )


def all_double_bonds(mol: Chem.Mol) -> tuple[int, ...]:
    return tuple(
        bond.GetIdx()
        for bond in mol.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE
    )


def stereo_signature(mol: Chem.Mol) -> tuple[str, ...]:
    pieces = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        if bond.GetStereo() == Chem.BondStereo.STEREONONE:
            continue
        pieces.append(
            f"b{bond.GetIdx()}:{bond.GetBeginAtomIdx()}={bond.GetEndAtomIdx()}:"
            f"{bond.GetStereo()}:{tuple(bond.GetStereoAtoms())}"
        )
    return tuple(pieces)


def rank_pairs_for_double_bonds(mol: Chem.Mol) -> tuple[str, ...]:
    ranks_iso = list(Chem.CanonicalRankAtoms(mol, breakTies=False, includeChirality=True))
    ranks_noiso = list(Chem.CanonicalRankAtoms(mol, breakTies=False, includeChirality=False))
    rows = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        row = [f"b{bond.GetIdx()} {bond.GetBeginAtomIdx()}={bond.GetEndAtomIdx()}"]
        for center, other in (
            (bond.GetBeginAtom(), bond.GetEndAtomIdx()),
            (bond.GetEndAtom(), bond.GetBeginAtomIdx()),
        ):
            nbrs = [nbr.GetIdx() for nbr in center.GetNeighbors() if nbr.GetIdx() != other]
            row.append(
                str(
                    [
                        (
                            idx,
                            mol.GetAtomWithIdx(idx).GetSymbol(),
                            ranks_iso[idx],
                            ranks_noiso[idx],
                        )
                        for idx in nbrs
                    ]
                )
            )
        rows.append(" | ".join(row))
    return tuple(rows)


def assign_potential_config(
    mol: Chem.Mol,
    potential: tuple[int, ...],
    config: tuple[Chem.BondStereo, ...],
) -> Chem.Mol:
    work = Chem.Mol(mol)
    Chem.FindPotentialStereoBonds(work)
    for bond_idx, stereo in zip(potential, config):
        work.GetBondWithIdx(bond_idx).SetStereo(stereo)
    return work


def print_scaffold(smiles: str) -> None:
    mol = parse_smiles(smiles)
    print("=" * 100)
    print("scaffold:", smiles)
    if mol is None:
        print("parse failed")
        return

    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    potential = potential_stereo_bonds(mol)
    print("rdkit:", rdBase.rdkitVersion)
    print("canonical:", canonical)
    print("all double bonds:", all_double_bonds(mol))
    print("initial potential stereo bonds:", potential)
    print("initial rank pairs:")
    for row in rank_pairs_for_double_bonds(mol):
        print("  ", row)

    if len(potential) > 6:
        print("too many potential bonds for exhaustive assignment")
        return

    outputs: dict[str, tuple[str, ...]] = {}
    signature_counts: dict[tuple[str, ...], int] = {}
    for labels_and_stereos in product(STEREOS, repeat=len(potential)):
        labels = tuple(label for label, _stereo in labels_and_stereos)
        config = tuple(stereo for _label, stereo in labels_and_stereos)
        assigned = assign_potential_config(mol, potential, config)
        output = Chem.MolToSmiles(assigned, canonical=True, isomericSmiles=True)
        roundtrip = parse_smiles(output)
        if roundtrip is None:
            raise AssertionError(output)
        signature = stereo_signature(roundtrip)
        outputs[output] = labels
        signature_counts[signature] = signature_counts.get(signature, 0) + 1
        print("  config", labels, "->", output)
        print("    roundtrip stereo:", signature)
        print("    rank pairs:")
        for row in rank_pairs_for_double_bonds(roundtrip):
            print("     ", row)

    print("unique outputs:", len(outputs))
    print("unique roundtrip signatures:", len(signature_counts))


def main() -> None:
    for scaffold in SCAFFOLDS:
        print_scaffold(scaffold)


if __name__ == "__main__":
    main()
