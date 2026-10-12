"""Qualify selected compact spellings as chemistry semantics, not writer parity."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from rdkit import Chem

from grimace import decoding


FIXTURE = Path(__file__).parents[1] / "fixtures" / "opensmiles_examples" / "compact_language.json"
LANGUAGE = decoding.SmilesLanguage.COMPACT


def _parse(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise AssertionError(f"RDKit rejected {smiles!r}")
    return mol


def _choices(prefix: decoding.Prefix) -> dict[str, decoding.Prefix]:
    choices = prefix.choices()
    texts = tuple(text for text, _ in choices)
    if len(texts) != len(set(texts)):
        raise AssertionError(f"duplicate public token choices: {texts!r}")
    return dict(choices)


def _walk(mol: Chem.Mol, text: str) -> decoding.Prefix | None:
    return _walk_language(mol, text, LANGUAGE)


def _walk_language(
    mol: Chem.Mol, text: str, language: decoding.SmilesLanguage, *, complete: bool = True
) -> decoding.Prefix | None:
    pending = [(decoding.start(mol, language=language), 0)]
    while pending:
        prefix, offset = pending.pop()
        if offset == len(text):
            if not complete or prefix.is_accepted:
                return prefix
            continue
        pending.extend(
            (successor, offset + len(piece))
            for piece, successor in prefix.choices()
            if piece and text.startswith(piece, offset)
        )
    return None


def _bounded_outputs(
    mol: Chem.Mol, *, max_nodes: int = 20_000, max_outputs: int = 500, max_depth: int = 128
) -> tuple[frozenset[str], int]:
    pending = [(decoding.start(mol, language=LANGUAGE), "")]
    outputs: set[str] = set()
    nodes = 0
    while pending:
        prefix, text = pending.pop()
        nodes += 1
        if nodes > max_nodes:
            raise AssertionError(f"compact walk exhausted {max_nodes} prefix-node budget")
        if len(text) > max_depth:
            raise AssertionError(f"compact walk exhausted {max_depth} character-depth budget")
        if prefix.is_accepted:
            outputs.add(text)
            if len(outputs) > max_outputs:
                raise AssertionError(f"compact walk exhausted {max_outputs} output budget")
        choices = prefix.choices()
        if not choices and not prefix.is_accepted:
            raise AssertionError(f"reachable nonterminal compact prefix has no choices: {text!r}")
        for piece, successor in choices:
            pending.append((successor, text + piece))
    return frozenset(outputs), nodes


def _canonical_semantics(smiles: str) -> str:
    parsed = _parse(smiles)
    return Chem.MolToSmiles(parsed, canonical=True, isomericSmiles=True)


class CompactLanguageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(FIXTURE.read_text())

    def test_default_is_general_and_compact_is_a_distinct_public_language(self) -> None:
        carbon = _parse("C")
        self.assertEqual(
            tuple(text for text, _ in decoding.start(carbon).choices()),
            tuple(text for text, _ in decoding.start(carbon, language=decoding.SmilesLanguage.GENERAL).choices()),
        )
        self.assertEqual(("C",), tuple(text for text, _ in decoding.start(carbon, language=LANGUAGE).choices()))
        self.assertEqual(
            ("[", "C"),
            tuple(text for text, _ in decoding.start(carbon, language=decoding.SmilesLanguage.GENERAL).choices()),
        )

    def test_complete_small_support_and_graph_semantics(self) -> None:
        self.assertEqual("OpenSMILES compact spelling rules", self.fixture["source"]["name"])
        for case in self.fixture["cases"]:
            with self.subTest(case=case["id"]):
                source = _parse(case["source_smiles"])
                outputs, nodes = _bounded_outputs(source)
                self.assertGreater(nodes, 0)
                self.assertEqual(frozenset(case["expected_outputs"]), outputs)
                expected_graph = _canonical_semantics(case["source_smiles"])
                for output in outputs:
                    self.assertEqual(expected_graph, _canonical_semantics(output), output)
                    self.assertIsNotNone(_walk(source, output), output)

    def test_compact_atoms_preserve_isotope_map_charge_hydrogens_and_stereo(self) -> None:
        examples = (
            ("[13CH4]", "[13CH4]"),
            ("[NH4+]", "[NH4+]"),
            ("[nH]1cccc1", "[nH]0cccc0"),
            ("N", "N"),
            ("[C@H](F)(Cl)Br", "[C@H](F)(Cl)Br"),
        )
        for source_text, required_text in examples:
            with self.subTest(source=source_text):
                source = _parse(source_text)
                self.assertIsNotNone(_walk(source, required_text), required_text)

        mapped = _parse("C")
        mapped.GetAtomWithIdx(0).SetAtomMapNum(17)
        self.assertIsNotNone(_walk(mapped, "[CH4:17]"))
        explicit_zero = _parse("C")
        explicit_zero.GetAtomWithIdx(0).SetProp("molAtomMapNumber", "0")
        self.assertIsNotNone(_walk(explicit_zero, "[CH4:0]"))

    def test_compact_prefixes_omit_verbose_atoms_and_redundant_bond_marks(self) -> None:
        source = _parse("CO")
        after_carbon = _walk_language(source, "C", LANGUAGE, complete=False)
        self.assertIsNotNone(after_carbon)
        assert after_carbon is not None
        choices = _choices(after_carbon)
        self.assertIn("O", choices)
        self.assertNotIn("[", choices)
        self.assertNotIn("-", choices)
        for output in ("CO", "OC"):
            self.assertIsNotNone(_walk(source, output))

    def test_compact_keeps_ring_stereo_and_aromatic_choices_semantically_live(self) -> None:
        examples = (
            ("c1ccccc1", ("c0ccccc0", "C0=CC=CC=C0")),
            ("[nH]1cccc1", ("[nH]0cccc0",)),
            ("c1ccccc1-c1ccccc1", ("c0ccccc0-c0ccccc0",)),
            ("F/C=C/F", ("F/C=C/F",)),
            ("[C@H](F)(Cl)Br", ("[C@H](F)(Cl)Br",)),
            ("C1#CCCC1", ("C0#CCCC0",)),
            ("C1CC1", ("C0CC0",)),
        )
        for source_text, candidates in examples:
            with self.subTest(source=source_text):
                source = _parse(source_text)
                self.assertTrue(
                    any(_walk(source, candidate) is not None for candidate in candidates),
                    f"no compact candidate was reachable: {candidates!r}",
                )

    def test_compact_numeric_fields_keep_internal_zeroes_without_padding(self) -> None:
        isotope = _parse("[101CH4]")
        mapped = _parse("C")
        mapped.GetAtomWithIdx(0).SetAtomMapNum(1002)
        self.assertIsNotNone(_walk(isotope, "[101CH4]"))
        self.assertIsNone(_walk(isotope, "[0101CH4]"))
        self.assertIsNotNone(_walk(mapped, "[CH4:1002]"))
        self.assertIsNone(_walk(mapped, "[CH4:01002]"))

    def test_ring_closure_releases_and_reuses_the_smallest_label(self) -> None:
        source = _parse("C1CC1C2CC2")
        self.assertIsNotNone(_walk(source, "C0CC0C0CC0"))

    def test_top_level_dot_segments_preserve_bonded_graphs(self) -> None:
        # Ring labels can join atoms across top-level segments; these are
        # semantic witnesses, not claims about RDKit writer support.
        examples = (
            ("CC", "C1.C1", decoding.SmilesLanguage.GENERAL),
            ("CCC", "C1.C12.C2", decoding.SmilesLanguage.GENERAL),
            ("CC", "C0.C0", LANGUAGE),
            ("CCCC", "C0.C0C0.C0", LANGUAGE),
        )
        for source_text, candidate, language in examples:
            with self.subTest(source=source_text, candidate=candidate, language=language):
                source = _parse(source_text)
                self.assertIsNotNone(_walk_language(source, candidate, language))
                self.assertEqual(_canonical_semantics(source_text), _canonical_semantics(candidate))

    def test_terminal_branches_preserve_bonded_graphs(self) -> None:
        for source_text, candidate in (
            ("CC", "C(C)"),
            ("CCC", "C(C(C))"),
            ("CCC", "C(C)(C)"),
        ):
            with self.subTest(source=source_text, candidate=candidate):
                source = _parse(source_text)
                for language in (decoding.SmilesLanguage.GENERAL, LANGUAGE):
                    self.assertIsNotNone(_walk_language(source, candidate, language))
                self.assertEqual(_canonical_semantics(source_text), _canonical_semantics(candidate))

    def test_top_level_dot_segments_require_completed_bonds(self) -> None:
        source = _parse("CC")
        for language, incomplete in (
            (decoding.SmilesLanguage.GENERAL, "C1"),
            (LANGUAGE, "C0"),
        ):
            with self.subTest(language=language, incomplete=incomplete):
                prefix = _walk_language(source, incomplete, language, complete=False)
                self.assertIsNotNone(prefix)
                assert prefix is not None
                self.assertFalse(prefix.is_accepted)
                self.assertIsNone(_walk_language(source, incomplete, language))

    def test_nondefault_ring_bond_uses_one_marker_at_either_endpoint(self) -> None:
        source = _parse("C1=CCC1")
        for text in ("C0=CCC0", "C0CCC=0"):
            with self.subTest(text=text):
                self.assertIsNotNone(
                    _walk_language(source, text, decoding.SmilesLanguage.GENERAL)
                )
                self.assertIsNotNone(_walk(source, text))
        for text in ("C0=CCC=0", "C0#CCC=0"):
            with self.subTest(rejected=text):
                self.assertIsNone(
                    _walk_language(source, text, decoding.SmilesLanguage.GENERAL)
                )

        triple = _parse("C1CCCCCCC#1")
        for text in ("C#0CCCCCCC0", "C0CCCCCCC#0"):
            self.assertIsNotNone(_walk(triple, text), text)
        self.assertIsNone(_walk(triple, "C#0CCCCCCC#0"))

    def test_wildcard_aromatic_fixture_witnesses_in_general_and_compact(self) -> None:
        fixture = json.loads(
            (FIXTURE.parents[0] / "kekule_representations.json").read_text()
        )
        case = next(case for case in fixture["cases"] if case["id"] == "wildcard_ring")
        source = _parse(case["source_smiles"])
        witnesses = ("*0ccccc0", "*0=CC=CC=C0")
        for text in witnesses:
            with self.subTest(general=text):
                self.assertIsNotNone(
                    _walk_language(source, text, decoding.SmilesLanguage.GENERAL)
                )
        self.assertIsNone(_walk(source, "*0ccccc0"))
        self.assertIsNotNone(_walk(source, "*0:ccccc:0"))
        self.assertIsNotNone(_walk(source, "*0=CC=CC=C0"))

        bridge = _parse("c1ccccc1*1ccccc1")
        self.assertIsNotNone(_walk(bridge, "c0ccccc0*0:ccccc:0"))

    def test_general_retains_verbose_spellings_and_rdkit_language_is_separate(self) -> None:
        carbon = _parse("C")
        general = decoding.start(carbon, language=decoding.SmilesLanguage.GENERAL)
        rdkit = decoding.start(carbon, language=decoding.SmilesLanguage.RDKIT_RANDOM_NONCANONICAL)
        self.assertIn("[", _choices(general))
        self.assertNotIn("[", _choices(decoding.start(carbon, language=LANGUAGE)))
        self.assertEqual(("C",), tuple(_choices(rdkit)))


if __name__ == "__main__":
    unittest.main()
