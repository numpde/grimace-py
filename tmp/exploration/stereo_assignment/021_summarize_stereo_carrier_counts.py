from __future__ import annotations

from collections import Counter

from rdkit import Chem, rdBase


CASES = {
    "github3967_ring": r"C1=CC/C=C2C3=C/CC=CC=CC\3C\2C=C1",
    "manual_difficult_cis_cis": r"CC/C=C\C(CO)=C(/C)CC",
    "manual_stereo_atoms_missing_surface": r"CC\C=C/C(/C=C/CC)=C(/CC)CO",
    "latent_central_unspecified": r"CCC=CC(C=CCC)=C(CO)CC",
}

DIRS = {Chem.BondDir.ENDUPRIGHT, Chem.BondDir.ENDDOWNRIGHT}
E_OR_Z = {
    Chem.BondStereo.STEREOE,
    Chem.BondStereo.STEREOTRANS,
    Chem.BondStereo.STEREOZ,
    Chem.BondStereo.STEREOCIS,
}


def strip_stereo(mol: Chem.Mol) -> Chem.Mol:
    copy = Chem.Mol(mol)
    Chem.RemoveStereochemistry(copy)
    return copy


def source_stereo_bonds(mol: Chem.Mol) -> tuple[tuple[int, int], ...]:
    return tuple(
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for bond in mol.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE and bond.GetStereo() in E_OR_Z
    )


def candidate_edges(mol: Chem.Mol, begin: int, end: int, endpoint: int) -> tuple[tuple[int, int], ...]:
    other_endpoint = end if endpoint == begin else begin
    out = []
    for neighbor in mol.GetAtomWithIdx(endpoint).GetNeighbors():
        neighbor_idx = neighbor.GetIdx()
        if neighbor_idx == other_endpoint:
            continue
        bond = mol.GetBondBetweenAtoms(endpoint, neighbor_idx)
        if bond is None:
            continue
        if bond.GetBondType() in (Chem.BondType.SINGLE, Chem.BondType.AROMATIC):
            out.append(tuple(sorted((endpoint, neighbor_idx))))
    return tuple(sorted(out))


def sample_outputs(mol: Chem.Mol, *, samples: int = 512) -> tuple[str, ...]:
    outputs = {Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)}
    for seed in range(samples):
        Chem.rdBase.SeedRandomNumberGenerator(seed)
        outputs.add(
            Chem.MolToSmiles(
                Chem.Mol(mol),
                canonical=False,
                doRandom=True,
                isomericSmiles=True,
            )
        )
    return tuple(sorted(outputs))


def preserves_source_stereo(
    source: Chem.Mol,
    output: Chem.Mol,
    match: tuple[int, ...],
    source_stereo: tuple[tuple[int, int], ...],
) -> bool:
    for begin, end in source_stereo:
        source_bond = source.GetBondBetweenAtoms(begin, end)
        output_bond = output.GetBondBetweenAtoms(match[begin], match[end])
        if output_bond is None or output_bond.GetStereo() != source_bond.GetStereo():
            return False
    return True


def first_preserving_match(
    source: Chem.Mol,
    output: Chem.Mol,
    source_stereo: tuple[tuple[int, int], ...],
) -> tuple[int, ...] | None:
    matches = strip_stereo(output).GetSubstructMatches(
        strip_stereo(source),
        uniquify=False,
        useChirality=False,
        maxMatches=4096,
    )
    for match in matches:
        if preserves_source_stereo(source, output, match, source_stereo):
            return match
    return None


def mapped_directed_edges(output: Chem.Mol, match: tuple[int, ...]) -> set[tuple[int, int]]:
    inverse = {target: source for source, target in enumerate(match)}
    out = set()
    for bond in output.GetBonds():
        if bond.GetBondDir() not in DIRS:
            continue
        begin = inverse.get(bond.GetBeginAtomIdx())
        end = inverse.get(bond.GetEndAtomIdx())
        if begin is not None and end is not None:
            out.add(tuple(sorted((begin, end))))
    return out


def summarize_case(label: str, smiles: str) -> None:
    source = Chem.MolFromSmiles(smiles)
    assert source is not None
    source_stereo = source_stereo_bonds(source)
    endpoint_candidates = {
        (begin, end, endpoint): candidate_edges(source, begin, end, endpoint)
        for begin, end in source_stereo
        for endpoint in (begin, end)
    }
    outputs = sample_outputs(source)

    directed_total = Counter()
    endpoint_counts = Counter()
    no_preserving = 0
    for output_smiles in outputs:
        output = Chem.MolFromSmiles(output_smiles)
        if output is None:
            no_preserving += 1
            continue
        match = first_preserving_match(source, output, source_stereo)
        if match is None:
            no_preserving += 1
            continue
        directed_edges = mapped_directed_edges(output, match)
        directed_total[len(directed_edges)] += 1
        for key, candidates in endpoint_candidates.items():
            endpoint_counts[
                (
                    key,
                    sum(1 for edge in candidates if edge in directed_edges),
                    len(candidates),
                )
            ] += 1

    print("=" * 120)
    print(label)
    print("source:", smiles)
    print("outputs:", len(outputs), "no preserving match:", no_preserving)
    print("source stereo bonds:", source_stereo)
    print("directed carrier count distribution:", dict(sorted(directed_total.items())))
    for (key, directed_count, candidate_count), count in sorted(endpoint_counts.items()):
        print(f"  endpoint {key}: directed {directed_count}/{candidate_count}: {count}")


def main() -> None:
    print("rdkit:", rdBase.rdkitVersion)
    print("legacy:", Chem.GetUseLegacyStereoPerception())
    for label, smiles in CASES.items():
        summarize_case(label, smiles)


if __name__ == "__main__":
    main()
