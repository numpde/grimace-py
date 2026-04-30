from __future__ import annotations

from itertools import product

from rdkit import Chem, rdBase


STEREO_CHOICES = (
    Chem.BondStereo.STEREOTRANS,
    Chem.BondStereo.STEREOCIS,
)


def parse_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(smiles)
    return mol


def atom_label(mol: Chem.Mol, idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    return f"{idx}{atom.GetSymbol()}"


def bond_label(mol: Chem.Mol, bond: Chem.Bond) -> str:
    return (
        f"b{bond.GetIdx()}:{atom_label(mol, bond.GetBeginAtomIdx())}="
        f"{atom_label(mol, bond.GetEndAtomIdx())}"
    )


def stereo_bond_signature(mol: Chem.Mol) -> tuple[str, ...]:
    lines = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        if bond.GetStereo() == Chem.BondStereo.STEREONONE:
            continue
        stereo_atoms = ",".join(atom_label(mol, idx) for idx in bond.GetStereoAtoms())
        lines.append(f"{bond_label(mol, bond)} {bond.GetStereo()} [{stereo_atoms}]")
    return tuple(lines)


def directional_signature(mol: Chem.Mol) -> tuple[str, ...]:
    return tuple(
        f"b{bond.GetIdx()}:{atom_label(mol, bond.GetBeginAtomIdx())}-"
        f"{atom_label(mol, bond.GetEndAtomIdx())} {bond.GetBondDir()}"
        for bond in mol.GetBonds()
        if bond.GetBondDir() != Chem.BondDir.NONE
    )


def potential_stereo_bonds(mol: Chem.Mol) -> list[int]:
    work = Chem.Mol(mol)
    Chem.FindPotentialStereoBonds(work)
    return [
        bond.GetIdx()
        for bond in work.GetBonds()
        if bond.GetStereo() == Chem.BondStereo.STEREOANY
    ]


def set_potential_stereo_config(
    mol: Chem.Mol,
    potential_bond_indices: tuple[int, ...],
    config: tuple[Chem.BondStereo, ...],
) -> Chem.Mol:
    work = Chem.Mol(mol)
    Chem.FindPotentialStereoBonds(work)
    for bond_idx, stereo in zip(potential_bond_indices, config):
        work.GetBondWithIdx(bond_idx).SetStereo(stereo)
    return work


def first_neighbor_outside(atom: Chem.Atom, excluded: set[int]) -> int:
    for neighbor in atom.GetNeighbors():
        if neighbor.GetIdx() not in excluded:
            return neighbor.GetIdx()
    raise ValueError(f"no neighbor outside {excluded} for atom {atom.GetIdx()}")


def force_all_double_bonds_stereo_atoms(mol: Chem.Mol) -> tuple[Chem.Mol, tuple[int, ...]]:
    work = Chem.Mol(mol)
    double_bond_indices = []
    for bond in work.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        excluded = {begin.GetIdx(), end.GetIdx()}
        begin_neighbor = first_neighbor_outside(begin, excluded)
        end_neighbor = first_neighbor_outside(end, excluded)
        bond.SetStereoAtoms(begin_neighbor, end_neighbor)
        double_bond_indices.append(bond.GetIdx())
    return work, tuple(double_bond_indices)


def print_assignment_closure(smiles: str) -> None:
    mol = parse_smiles(smiles)
    potential = tuple(potential_stereo_bonds(mol))
    print("=" * 88)
    print(f"input: {smiles}")
    print(f"rdkit={rdBase.rdkitVersion}")
    print("canonical unspec:", Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    print("potential stereo bonds:", potential)
    print()
    print("FindPotentialStereoBonds -> assign listed potential bonds only:")
    seen_outputs: dict[str, tuple[Chem.BondStereo, ...]] = {}
    roundtrip_signatures: dict[tuple[str, ...], list[str]] = {}
    for config in product(STEREO_CHOICES, repeat=len(potential)):
        work = set_potential_stereo_config(mol, potential, config)
        output = Chem.MolToSmiles(work, canonical=True, isomericSmiles=True)
        roundtrip = parse_smiles(output)
        seen_outputs[output] = config
        roundtrip_signatures.setdefault(stereo_bond_signature(roundtrip), []).append(output)
        print(
            " ",
            tuple(str(stereo).replace("STEREO", "") for stereo in config),
            "->",
            output,
            "roundtrip_stereos=",
            stereo_bond_signature(roundtrip),
        )
    print("unique outputs:", len(seen_outputs))
    print("unique roundtrip stereo signatures:", len(roundtrip_signatures))
    print()

    forced, forced_indices = force_all_double_bonds_stereo_atoms(mol)
    print("Force stereo atoms on every double bond, then assign every double bond:")
    print("forced double bond indices:", forced_indices)
    forced_outputs: dict[str, tuple[Chem.BondStereo, ...]] = {}
    forced_roundtrip_outputs: dict[str, list[str]] = {}
    for config in product(STEREO_CHOICES, repeat=len(forced_indices)):
        work = Chem.Mol(forced)
        for bond_idx, stereo in zip(forced_indices, config):
            work.GetBondWithIdx(bond_idx).SetStereo(stereo)
        output = Chem.MolToSmiles(work, canonical=True, isomericSmiles=True)
        roundtrip_output = Chem.MolToSmiles(parse_smiles(output), canonical=True, isomericSmiles=True)
        forced_outputs[output] = config
        forced_roundtrip_outputs.setdefault(roundtrip_output, []).append(output)
        print(
            " ",
            tuple(str(stereo).replace("STEREO", "") for stereo in config),
            "->",
            output,
            "roundtrip=",
            roundtrip_output,
        )
    print("unique forced outputs:", len(forced_outputs))
    print("unique forced roundtrip outputs:", len(forced_roundtrip_outputs))
    for roundtrip_output, pre_outputs in forced_roundtrip_outputs.items():
        if len(pre_outputs) > 1:
            print("  collapse:", roundtrip_output, "<=", pre_outputs)


def main() -> None:
    print_assignment_closure("CCC=CC(CO)=C(C)CC")
    print_assignment_closure("CCC=CC(C=CCC)=C(CO)CC")


if __name__ == "__main__":
    main()
