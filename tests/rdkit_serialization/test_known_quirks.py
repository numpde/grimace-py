from __future__ import annotations

import unittest

from rdkit import Chem, rdBase

from tests.helpers.mols import parse_smiles
from tests.helpers.rdkit_known_quirks import (
    load_pinned_rdkit_known_quirk_cases,
)


def _bond_type_between(mol: Chem.Mol, begin_idx: int, end_idx: int) -> str:
    bond = mol.GetBondBetweenAtoms(begin_idx, end_idx)
    if bond is None:
        return "MISSING"
    return str(bond.GetBondType())


def _stereo_double_bond_descriptions(mol: Chem.Mol) -> tuple[str, ...]:
    descriptions = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        if bond.GetStereo() == Chem.BondStereo.STEREONONE:
            continue

        stereo_atoms = tuple(bond.GetStereoAtoms())
        if len(stereo_atoms) != 2:
            carrier_text = "via <missing stereo atoms>"
        else:
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            begin_carrier = stereo_atoms[0]
            end_carrier = stereo_atoms[1]
            carrier_text = (
                f"via {mol.GetAtomWithIdx(begin_carrier).GetSymbol()}"
                f"({_bond_type_between(mol, begin_idx, begin_carrier)}),"
                f"{mol.GetAtomWithIdx(end_carrier).GetSymbol()}"
                f"({_bond_type_between(mol, end_idx, end_carrier)})"
            )

        descriptions.append(
            f"{bond.GetBeginAtom().GetSymbol()}={bond.GetEndAtom().GetSymbol()} "
            f"{bond.GetStereo()} {carrier_text}"
        )
    return tuple(sorted(descriptions))


class RdkitKnownQuirkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.cases = load_pinned_rdkit_known_quirk_cases(rdBase.rdkitVersion)
        except FileNotFoundError:
            raise unittest.SkipTest(
                f"no pinned RDKit known-quirk corpus for RDKit {rdBase.rdkitVersion}"
            )

    def test_smiles_roundtrip_stereo_annotations_match_pinned_rdkit_behavior(self) -> None:
        for case in self.cases:
            with self.subTest(case_id=case.case_id, category=case.category):
                source_mol = parse_smiles(case.smiles)
                self.assertEqual(
                    case.canonical_smiles,
                    Chem.MolToSmiles(
                        source_mol,
                        canonical=True,
                        isomericSmiles=True,
                    ),
                )
                self.assertEqual(
                    case.source_stereo_double_bonds,
                    _stereo_double_bond_descriptions(source_mol),
                )

                roundtrip_mol = parse_smiles(case.canonical_smiles)
                self.assertEqual(
                    case.canonical_smiles,
                    Chem.MolToSmiles(
                        roundtrip_mol,
                        canonical=True,
                        isomericSmiles=True,
                    ),
                )
                self.assertEqual(
                    case.roundtrip_stereo_double_bonds,
                    _stereo_double_bond_descriptions(roundtrip_mol),
                )


if __name__ == "__main__":
    unittest.main()
