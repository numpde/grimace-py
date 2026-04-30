from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem, rdBase


@dataclass(frozen=True)
class Case:
    case_id: str
    smiles: str


CASES = (
    Case("simple_trans", "F/C=C/F"),
    Case("really_difficult_EE", "CC/C=C/C(/C=C/CC)=C(CC)CO"),
    Case("really_difficult_EZ_hidden_central", r"CC/C=C\C(/C=C/CC)=C(CC)CO"),
    Case("really_difficult_ZZ", r"CC/C=C\C(/C=C\CC)=C(CC)CO"),
    Case("stereo_atoms_central_Z", r"CC\C=C/C(/C=C/CC)=C(/CC)CO"),
    Case("stereo_atoms_central_E", r"CC\C=C/C(/C=C/CC)=C(\CC)CO"),
    Case("ring_closure_github3967", r"C1=CC/C=C2C3=C/CC=CC=CC\3C\2C=C1"),
    Case("imine_bug1842174", "F/C=N/Cl"),
    Case("conjugated_diene", "F/C=C/C=C(/Cl)Br"),
)


def sample_outputs(smiles: str, *, legacy: bool, samples: int = 1024) -> set[str]:
    original = Chem.GetUseLegacyStereoPerception()
    Chem.SetUseLegacyStereoPerception(legacy)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(smiles)
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


def parsed_stereo_count(smiles: str, *, legacy: bool) -> int:
    original = Chem.GetUseLegacyStereoPerception()
    Chem.SetUseLegacyStereoPerception(legacy)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(smiles)
        return sum(
            1
            for bond in mol.GetBonds()
            if bond.GetBondType() == Chem.BondType.DOUBLE
            and bond.GetStereo() != Chem.BondStereo.STEREONONE
        )
    finally:
        Chem.SetUseLegacyStereoPerception(original)


def main() -> None:
    print("rdkit:", rdBase.rdkitVersion)
    print("default legacy:", Chem.GetUseLegacyStereoPerception())
    for case in CASES:
        legacy_outputs = sample_outputs(case.smiles, legacy=True)
        new_outputs = sample_outputs(case.smiles, legacy=False)
        only_legacy = sorted(legacy_outputs - new_outputs)
        only_new = sorted(new_outputs - legacy_outputs)
        print("=" * 100)
        print(case.case_id)
        print("input:", case.smiles)
        print(
            "parsed stereo count legacy/new:",
            parsed_stereo_count(case.smiles, legacy=True),
            parsed_stereo_count(case.smiles, legacy=False),
        )
        print(
            "sampled support sizes legacy/new/intersection:",
            len(legacy_outputs),
            len(new_outputs),
            len(legacy_outputs & new_outputs),
        )
        print("only legacy:", len(only_legacy))
        for output in only_legacy[:12]:
            print("  L", output)
        print("only new:", len(only_new))
        for output in only_new[:12]:
            print("  N", output)


if __name__ == "__main__":
    main()
