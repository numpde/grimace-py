"""Qualify selected strict leading-dot tetrahedral paths independently of RDKit parsing."""

import json
import unittest

from rdkit import Chem

from grimace.decoding import SmilesLanguage, start
from tests.helpers.fixture_paths import checked_in_fixture_path
from tests.helpers.leading_dot_tetra_witness import tetra_signature


def accepted_path(molecule: Chem.Mol, language: SmilesLanguage, text: str) -> bool:
    prefix = start(molecule, language=language)
    remaining = text
    while remaining:
        matches = [(token, child) for token, child in prefix.choices() if remaining.startswith(token)]
        if not matches:
            return False
        token, prefix = max(matches, key=lambda pair: len(pair[0]))
        remaining = remaining[len(token):]
    return prefix.is_accepted


class LeadingDotTetraSemanticTests(unittest.TestCase):
    def test_fixed_star_event_interpretation(self) -> None:
        source = tetra_signature("F[C@H](Cl)Br")
        self.assertEqual((("Br", "C"), ("C", "Cl"), ("C", "F")), source[0])
        self.assertEqual(1, source[1])
        self.assertEqual(source, tetra_signature("F1(.[C@@H]1(Cl)Br)"))
        self.assertEqual(source, tetra_signature("[C@@H]1(.F1)(Cl)Br"))
        self.assertNotEqual(source, tetra_signature("F1(.[C@H]1(Cl)Br)"))
        self.assertNotEqual(source, tetra_signature("[C@H]1(.F1)(Cl)Br"))
        for invalid in ("F1(.[C@@H]2(Cl)Br)", "F1(.[C@@H]1(Cl)Br", "F1(.[C@@H]1(Cl)Br)O"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                tetra_signature(invalid)

    def test_public_leading_dot_tetra_membership(self) -> None:
        fixture_path = checked_in_fixture_path("opensmiles_examples", "leading_dot_tetra.json")
        cases = json.loads(fixture_path.read_text())["cases"]
        for case in cases:
            source_text = case["input_smiles"]
            source = Chem.MolFromSmiles(source_text)
            assert source is not None
            expected = tetra_signature(source_text)
            for language, field in (
                (SmilesLanguage.MAXIMAL, "maximal_witnesses"),
                (SmilesLanguage.COMPACT, "compact_witnesses"),
            ):
                witnesses = case[field]
                self.assertEqual(len(witnesses), len(set(witnesses)), case["id"])
                for molecule in (source, Chem.RenumberAtoms(source, [3, 2, 1, 0])):
                    for witness in witnesses:
                        with self.subTest(language=language, witness=witness, atom_order=tuple(atom.GetSymbol() for atom in molecule.GetAtoms())):
                            self.assertEqual(expected, tetra_signature(witness))
                            self.assertTrue(accepted_path(molecule, language, witness))
                            opposite = witness.replace("@@H", "@H")
                            self.assertNotEqual(expected, tetra_signature(opposite))
                            self.assertFalse(accepted_path(molecule, language, opposite))


if __name__ == "__main__":
    unittest.main()
