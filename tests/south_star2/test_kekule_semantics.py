"""Bounded aromatic/Kekule witnesses, separate from RDKit writer parity.

These paths exercise representation and sanitized graph semantics. They do not
prove complete Kekule assignment support or exact RDKit string support.
"""
from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from rdkit import Chem

from grimace import decoding
from tests.helpers.fixture_paths import checked_in_fixture_path


FIXTURE_PATH = checked_in_fixture_path("opensmiles_examples", "kekule_representations.json")


def _parse(smiles: str, preserve_graph_hydrogens: bool = False) -> Chem.Mol:
    params = Chem.SmilesParserParams()
    params.removeHs = not preserve_graph_hydrogens
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is None:
        raise AssertionError(f"Could not parse semantic witness {smiles!r}")
    return mol


def _accepts(initial: decoding.Prefix, text: str) -> bool:
    # A lexical prefix can have several token segmentations. Retain every
    # matching successor rather than choosing a first traversal or piece.
    pending = [(initial, 0)]
    while pending:
        prefix, offset = pending.pop()
        if offset == len(text):
            if prefix.is_accepted:
                return True
            continue
        for piece, successor in prefix.choices():
            if not piece:
                raise AssertionError("Public choices must advance the text")
            if text.startswith(piece, offset):
                pending.append((successor, offset + len(piece)))
    return False


def _has_prefix(initial: decoding.Prefix, text: str) -> bool:
    pending = [(initial, 0)]
    while pending:
        prefix, offset = pending.pop()
        if offset == len(text):
            return True
        for piece, successor in prefix.choices():
            if text.startswith(piece, offset):
                pending.append((successor, offset + len(piece)))
    return False


def _atom_facts(atom: Chem.Atom) -> tuple[object, ...]:
    return (
        atom.GetAtomicNum(), atom.GetFormalCharge(), atom.GetIsotope(),
        atom.GetTotalNumHs(), atom.GetAtomMapNum(),
        atom.GetNumRadicalElectrons(), atom.GetIsAromatic(),
        sum(neighbor.GetAtomicNum() == 1 for neighbor in atom.GetNeighbors()),
    )


def _assert_same_graph(test: unittest.TestCase, source: Chem.Mol, parsed: Chem.Mol) -> None:
    test.assertEqual(source.GetNumAtoms(), parsed.GetNumAtoms())
    test.assertEqual(source.GetNumBonds(), parsed.GetNumBonds())
    # Both inputs are sanitized: aromatic integer-bond projections normalize
    # before comparison. Atom class zero is numeric, irrespective of presence.
    for mapping in parsed.GetSubstructMatches(source, useChirality=True, uniquify=False):
        if any(_atom_facts(atom) != _atom_facts(parsed.GetAtomWithIdx(mapping[atom.GetIdx()]))
               for atom in source.GetAtoms()):
            continue
        for bond in source.GetBonds():
            mapped = parsed.GetBondBetweenAtoms(mapping[bond.GetBeginAtomIdx()], mapping[bond.GetEndAtomIdx()])
            if mapped is None or (mapped.GetBondType(), mapped.GetStereo()) != (bond.GetBondType(), bond.GetStereo()):
                break
        else:
            return
    test.fail("No mapping preserves sanitized graph, hydrogens, isotope, class, charge, and stereo")


class KekuleRepresentationSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = json.loads(FIXTURE_PATH.read_text())
        cls.cases = tuple(cls.data["cases"])

    def test_fixture_witnesses_preserve_sanitized_graph_and_stereo(self) -> None:
        self.assertEqual(22, len(self.cases))
        self.assertEqual(len(self.cases), len({case["id"] for case in self.cases}))
        self.assertEqual("https://opensmiles.org/opensmiles.html", self.data["source"]["url"])
        for case in self.cases:
            preserve_h = case.get("preserve_graph_hydrogens", False)
            with self.subTest(case=case["id"]):
                source = _parse(case["source_smiles"], preserve_h)
                for witness in case["kekule_witnesses"]:
                    with self.subTest(witness=witness):
                        _assert_same_graph(self, source, _parse(witness, preserve_h))

    def test_aromatic_and_integer_bond_paths_are_accepted(self) -> None:
        for case in self.cases:
            with self.subTest(case=case["id"]):
                source = _parse(case["source_smiles"], case.get("preserve_graph_hydrogens", False))
                initial = decoding.start(source)
                self.assertTrue(_accepts(initial, case["source_smiles"]), case["source_smiles"])
                for witness in case["kekule_witnesses"]:
                    with self.subTest(witness=witness):
                        self.assertTrue(_accepts(initial, witness), witness)

    def test_aromatic_forest_segments_preserve_matching(self) -> None:
        cases = (
            ("c1ccccc1", ("c1cc2.c2cc1", "C1=CC=2.C2C=C1")),
            ("[nH]1cccc1", ("[nH]1cc2.c2c1", "N1C=C2.C2=C1")),
            ("c1ccccc1.c1ccccc1", ("c1cc2.c2cc1.C1=CC=CC=C1",)),
            ("c1ccccc1-c1ccccc1", ("c1ccccc1-c1ccccc1",)),
        )
        for source_text, witnesses in cases:
            source = _parse(source_text)
            for language in (decoding.SmilesLanguage.GENERAL, decoding.SmilesLanguage.COMPACT):
                initial = decoding.start(source, language=language)
                for witness in witnesses:
                    emitted = witness.translate(str.maketrans("12", "01")) if language == decoding.SmilesLanguage.COMPACT else witness
                    with self.subTest(source=source_text, language=language, witness=witness):
                        _assert_same_graph(self, source, _parse(emitted))
                        self.assertTrue(_accepts(initial, emitted), emitted)

        # The second ring pair would force two double bonds at one aromatic carbon.
        # RDKit sanitizes this text, but it conflicts with the prepared atom demands.
        pyrrole = _parse("[nH]1cccc1")
        for language, prefix, impossible in (
            (decoding.SmilesLanguage.GENERAL, "N1C=C2.C", "N1C=C2.C=2C1"),
            (decoding.SmilesLanguage.COMPACT, "N0C=C1.C", "N0C=C1.C=1C0"),
        ):
            initial = decoding.start(pyrrole, language=language)
            self.assertTrue(_has_prefix(initial, prefix))
            self.assertFalse(_has_prefix(initial, impossible[:-2]))
            self.assertFalse(_accepts(initial, impossible))

    def test_current_whole_component_mode_rejects_conflicting_paths(self) -> None:
        case = next(case for case in self.cases if case["id"] == "benzene")
        initial = decoding.start(_parse(case["source_smiles"]))
        for witness in self.data["current_component_mode_conflict_witnesses"]:
            with self.subTest(witness=witness):
                self.assertFalse(_accepts(initial, witness))

    def test_choices_do_not_call_rdkit_after_start(self) -> None:
        case = next(case for case in self.cases if case["id"] == "benzene")
        initial = decoding.start(_parse(case["source_smiles"]))
        with patch.object(Chem, "MolToSmiles", side_effect=AssertionError("RDKit writer called")), \
             patch.object(Chem, "MolFromSmiles", side_effect=AssertionError("RDKit parser called")):
            for witness in [case["source_smiles"], *case["kekule_witnesses"]]:
                self.assertTrue(_accepts(initial, witness), witness)

    def test_rdkit_language_still_refuses_unqualified_aromatic_sources(self) -> None:
        case = next(case for case in self.cases if case["id"] == "benzene")
        with self.assertRaisesRegex(ValueError, "only qualified|not qualified"):
            decoding.start(_parse(case["source_smiles"]), language=decoding.SmilesLanguage.RDKIT_RANDOM_NONCANONICAL)
