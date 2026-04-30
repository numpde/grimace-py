from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from rdkit import Chem, rdBase


SOURCE = r"C1=CC/C=C2C3=C/CC=CC=CC\3C\2C=C1"
TRACKED_EDGES = {
    (2, 3): "left_outer",
    (4, 5): "shared_internal",
    (4, 13): "left_ring_closure",
    (5, 12): "right_ring_closure",
    (6, 7): "right_outer",
}
SOURCE_STEREO = ((3, 4, "Z"), (5, 6, "E"))


@dataclass(frozen=True)
class OutputClass:
    label: str
    outputs: set[str]


def stereo_class(stereo: Chem.BondStereo) -> str:
    if stereo in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS):
        return "E"
    if stereo in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS):
        return "Z"
    if stereo == Chem.BondStereo.STEREONONE:
        return "NONE"
    return str(stereo)


def strip_stereo(mol: Chem.Mol) -> Chem.Mol:
    copy = Chem.Mol(mol)
    Chem.RemoveStereochemistry(copy)
    return copy


def matches(source: Chem.Mol, output: Chem.Mol) -> tuple[tuple[int, ...], ...]:
    return tuple(
        strip_stereo(output).GetSubstructMatches(
            strip_stereo(source),
            uniquify=False,
            useChirality=False,
            maxMatches=1024,
        )
    )


def preserves_source_stereo(output: Chem.Mol, match: tuple[int, ...]) -> bool:
    for begin, end, expected in SOURCE_STEREO:
        bond = output.GetBondBetweenAtoms(match[begin], match[end])
        if bond is None or stereo_class(bond.GetStereo()) != expected:
            return False
    return True


def mapped_directional_edges(output: Chem.Mol, match: tuple[int, ...]) -> tuple[str, ...]:
    inverse = {target: source for source, target in enumerate(match)}
    labels = []
    for bond in output.GetBonds():
        if bond.GetBondDir() not in (
            Chem.BondDir.ENDUPRIGHT,
            Chem.BondDir.ENDDOWNRIGHT,
        ):
            continue
        source_begin = inverse[bond.GetBeginAtomIdx()]
        source_end = inverse[bond.GetEndAtomIdx()]
        edge = tuple(sorted((source_begin, source_end)))
        label = TRACKED_EDGES.get(edge, f"other:{edge}")
        labels.append(f"{label}:{source_begin}->{source_end}:{bond.GetBondDir()}")
    return tuple(sorted(labels))


def selected_stereo_edges(output: Chem.Mol, match: tuple[int, ...]) -> tuple[str, ...]:
    inverse = {target: source for source, target in enumerate(match)}
    labels = []
    for begin, end, expected in SOURCE_STEREO:
        bond = output.GetBondBetweenAtoms(match[begin], match[end])
        if bond is None:
            labels.append(f"{begin}={end}:missing")
            continue
        stereo_atoms = tuple(inverse[idx] for idx in bond.GetStereoAtoms())
        selected = []
        for center, stereo_atom in zip((begin, end), stereo_atoms):
            edge = tuple(sorted((center, stereo_atom)))
            selected.append(TRACKED_EDGES.get(edge, f"other:{edge}"))
        labels.append(f"{begin}={end}:{expected}:{tuple(selected)}")
    return tuple(labels)


def sample_outputs(*, legacy: bool, samples: int = 2048) -> set[str]:
    original = Chem.GetUseLegacyStereoPerception()
    Chem.SetUseLegacyStereoPerception(legacy)
    try:
        mol = Chem.MolFromSmiles(SOURCE)
        outputs = {
            Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
        }
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
        return outputs
    finally:
        Chem.SetUseLegacyStereoPerception(original)


def classify_outputs(label: str, outputs: set[str]) -> None:
    source = Chem.MolFromSmiles(SOURCE)
    class_counts: Counter[tuple[tuple[str, ...], tuple[str, ...]]] = Counter()
    examples: dict[tuple[tuple[str, ...], tuple[str, ...]], str] = {}
    no_preserving = []
    for smiles in sorted(outputs):
        output = Chem.MolFromSmiles(smiles)
        preserving_matches = [
            match for match in matches(source, output) if preserves_source_stereo(output, match)
        ]
        if not preserving_matches:
            no_preserving.append(smiles)
            continue
        classes_for_output = {
            (
                mapped_directional_edges(output, match),
                selected_stereo_edges(output, match),
            )
            for match in preserving_matches
        }
        # All preserving automorphisms generally describe the same carrier shape
        # modulo symmetry. Count each distinct shape once per output.
        for shape in classes_for_output:
            class_counts[shape] += 1
            examples.setdefault(shape, smiles)

    print("=" * 120)
    print(label, "outputs=", len(outputs), "no_preserving=", len(no_preserving))
    print("classes=", len(class_counts))
    for idx, (shape, count) in enumerate(class_counts.most_common(20), start=1):
        directions, selected = shape
        print(f"  class {idx}: count={count}")
        print("    dirs:", directions)
        print("    selected:", selected)
        print("    example:", examples[shape])
    if no_preserving:
        print("no preserving examples:", no_preserving[:8])


def main() -> None:
    print("rdkit:", rdBase.rdkitVersion)
    print("default legacy:", Chem.GetUseLegacyStereoPerception())
    legacy = sample_outputs(legacy=True)
    new = sample_outputs(legacy=False)
    classify_outputs("legacy_all", legacy)
    classify_outputs("new_all", new)
    classify_outputs("intersection", legacy & new)
    classify_outputs("legacy_only", legacy - new)
    classify_outputs("new_only", new - legacy)


if __name__ == "__main__":
    main()
