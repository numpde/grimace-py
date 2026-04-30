from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem, rdBase


E_LIKE = {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS}
Z_LIKE = {Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS}


@dataclass(frozen=True)
class Case:
    label: str
    source: str
    outputs: tuple[str, ...]


CASES = (
    Case(
        label="github3967 ring closure",
        source=r"C1=CC/C=C2C3=C/CC=CC=CC\3C\2C=C1",
        outputs=(
            r"C1=CC/C=C2\C3=C\CC=CC=CC3C2C=C1",
            r"C1/CC=CC=CC2C3C(=C/CC=CC=C3)/C2=1",
        ),
    ),
    Case(
        label="manual difficult cis/cis",
        source=r"CC/C=C\C(CO)=C(/C)CC",
        outputs=(
            r"CC/C=C\C(CO)=C(/C)CC",
            r"C(/C(/C=C\CC)=C(/C)CC)O",
        ),
    ),
    Case(
        label="manual stereo atoms missing surface",
        source=r"CC\C=C/C(/C=C/CC)=C(/CC)CO",
        outputs=(
            r"CC\C=C/C(/C=C/CC)=C(/CC)CO",
            r"C(/C(/C=C/CC)=C(/CC)CO)=C/CC",
        ),
    ),
)


def stereo_class(stereo: Chem.BondStereo) -> str:
    if stereo in E_LIKE:
        return "E"
    if stereo in Z_LIKE:
        return "Z"
    if stereo == Chem.BondStereo.STEREOANY:
        return "ANY"
    if stereo == Chem.BondStereo.STEREONONE:
        return "NONE"
    return str(stereo)


def strip_stereo(mol: Chem.Mol) -> Chem.Mol:
    copy = Chem.Mol(mol)
    Chem.RemoveStereochemistry(copy)
    return copy


def source_stereo_bonds(mol: Chem.Mol) -> tuple[tuple[int, int, str], ...]:
    out = []
    for bond in mol.GetBonds():
        cls = stereo_class(bond.GetStereo())
        if bond.GetBondType() == Chem.BondType.DOUBLE and cls in {"E", "Z"}:
            out.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), cls))
    return tuple(out)


def source_candidate_edges(
    mol: Chem.Mol,
    stereo_bonds: tuple[tuple[int, int, str], ...],
) -> dict[tuple[int, int], tuple[tuple[int, int], ...]]:
    out: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
    for begin, end, _cls in stereo_bonds:
        bond = mol.GetBondBetweenAtoms(begin, end)
        assert bond is not None
        for endpoint, other_endpoint in ((begin, end), (end, begin)):
            candidates = []
            atom = mol.GetAtomWithIdx(endpoint)
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx == other_endpoint:
                    continue
                carrier = mol.GetBondBetweenAtoms(endpoint, neighbor_idx)
                if carrier is None:
                    continue
                if carrier.GetBondType() in (Chem.BondType.SINGLE, Chem.BondType.AROMATIC):
                    candidates.append(tuple(sorted((endpoint, neighbor_idx))))
            out[(begin, end, endpoint)] = tuple(sorted(candidates))
    return out


def preserving_matches(
    source: Chem.Mol,
    output: Chem.Mol,
    stereo_bonds: tuple[tuple[int, int, str], ...],
) -> list[tuple[int, ...]]:
    query = strip_stereo(source)
    target = strip_stereo(output)
    matches = target.GetSubstructMatches(
        query,
        uniquify=False,
        useChirality=False,
        maxMatches=4096,
    )
    preserving = []
    for match in matches:
        ok = True
        for begin, end, expected in stereo_bonds:
            bond = output.GetBondBetweenAtoms(match[begin], match[end])
            if bond is None or stereo_class(bond.GetStereo()) != expected:
                ok = False
                break
        if ok:
            preserving.append(match)
    return preserving


def mapped_directional_edges(output: Chem.Mol, match: tuple[int, ...]) -> tuple[str, ...]:
    inverse = {target: source for source, target in enumerate(match)}
    labels = []
    for bond in output.GetBonds():
        if bond.GetBondDir() not in (
            Chem.BondDir.ENDUPRIGHT,
            Chem.BondDir.ENDDOWNRIGHT,
        ):
            continue
        begin = inverse.get(bond.GetBeginAtomIdx())
        end = inverse.get(bond.GetEndAtomIdx())
        if begin is None or end is None:
            labels.append(
                f"unmapped:{bond.GetBeginAtomIdx()}-{bond.GetEndAtomIdx()}:{bond.GetBondDir()}"
            )
        else:
            labels.append(f"{tuple(sorted((begin, end)))}:{begin}->{end}:{bond.GetBondDir()}")
    return tuple(sorted(labels))


def selected_stereo_edges(
    output: Chem.Mol,
    match: tuple[int, ...],
    stereo_bonds: tuple[tuple[int, int, str], ...],
) -> tuple[str, ...]:
    inverse = {target: source for source, target in enumerate(match)}
    labels = []
    for begin, end, expected in stereo_bonds:
        bond = output.GetBondBetweenAtoms(match[begin], match[end])
        if bond is None:
            labels.append(f"{begin}={end}:{expected}:missing")
            continue
        stereo_atoms = tuple(inverse.get(idx, -1) for idx in bond.GetStereoAtoms())
        selected_edges = []
        for endpoint, stereo_atom in zip((begin, end), stereo_atoms):
            selected_edges.append(tuple(sorted((endpoint, stereo_atom))))
        labels.append(f"{begin}={end}:{expected}:selected={tuple(selected_edges)}")
    return tuple(labels)


def inspect_case(case: Case) -> None:
    source = Chem.MolFromSmiles(case.source)
    assert source is not None
    stereo_bonds = source_stereo_bonds(source)
    print("=" * 120)
    print(case.label)
    print("source:", case.source)
    print("canonical:", Chem.MolToSmiles(source, canonical=True, isomericSmiles=True))
    print("source stereo bonds:", stereo_bonds)
    print("source candidate carrier edges by stereo endpoint:")
    for key, candidates in source_candidate_edges(source, stereo_bonds).items():
        print(" ", key, "->", candidates)

    for smiles in case.outputs:
        output = Chem.MolFromSmiles(smiles)
        print("-" * 120)
        print("output:", smiles)
        if output is None:
            print("parse failed")
            continue
        matches = preserving_matches(source, output, stereo_bonds)
        print("preserving matches:", len(matches))
        shapes = {
            (mapped_directional_edges(output, match), selected_stereo_edges(output, match, stereo_bonds))
            for match in matches
        }
        print("carrier shapes:", len(shapes))
        for directions, selected in sorted(shapes)[:8]:
            print(" directions:", directions)
            print(" selected:  ", selected)


def main() -> None:
    print("rdkit:", rdBase.rdkitVersion)
    print("legacy:", Chem.GetUseLegacyStereoPerception())
    for case in CASES:
        inspect_case(case)


if __name__ == "__main__":
    main()
