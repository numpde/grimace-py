"""Check semantic graph preservation through real preparation and native walking."""

from __future__ import annotations

import unittest

from rdkit import Chem

from grimace._south_star2.preparation import prepare_molecule


def _accepted_strings(mol: Chem.Mol) -> set[str]:
    writer = prepare_molecule(mol).build_writer()
    frontier = [(writer.initial_state(), "")]
    accepted: set[str] = set()
    while frontier:
        state, text = frontier.pop()
        if state.is_accepted:
            accepted.add(text)
            continue
        choices = state.choices()
        if not choices:
            raise AssertionError(f"Native walker stopped before acceptance at {text!r}")
        frontier.extend((successor, text + token) for token, successor in choices)
    return accepted


class SouthStar2PreparationSemanticTests(unittest.TestCase):
    def test_native_outputs_parse_to_each_source_graph(self) -> None:
        for smiles in ("CCO", "CC(C)O", "C1CC1"):
            with self.subTest(smiles=smiles):
                source = Chem.MolFromSmiles(smiles)
                assert source is not None
                outputs = _accepted_strings(source)
                self.assertTrue(outputs, "the prepared source graph must have an accepted output")

                for output in outputs:
                    parsed = Chem.MolFromSmiles(output)
                    self.assertIsNotNone(parsed, f"native output did not parse: {output!r}")
                    assert parsed is not None
                    self.assertEqual(parsed.GetNumAtoms(), source.GetNumAtoms(), output)
                    self.assertEqual(parsed.GetNumBonds(), source.GetNumBonds(), output)

                    mappings = parsed.GetSubstructMatches(source, useChirality=True)
                    self.assertTrue(mappings, f"output graph differs from source: {output!r}")
                    self.assertTrue(
                        any(
                            all(
                                parsed.GetAtomWithIdx(parsed_index).GetTotalNumHs()
                                == source.GetAtomWithIdx(source_index).GetTotalNumHs()
                                for source_index, parsed_index in enumerate(mapping)
                            )
                            for mapping in mappings
                        ),
                        f"output implicit-hydrogen facts differ from source: {output!r}",
                    )
                    self.assertTrue(
                        all(
                            atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED
                            for atom in parsed.GetAtoms()
                        ),
                        f"output assigned unspecified atom stereo: {output!r}",
                    )


if __name__ == "__main__":
    unittest.main()
