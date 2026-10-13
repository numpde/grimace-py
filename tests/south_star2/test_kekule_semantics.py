"""Bounded aromatic/Kekule witnesses, separate from RDKit writer parity.

These paths exercise representation and sanitized graph semantics. They do not
prove complete Kekule assignment support or exact RDKit string support.
"""
from __future__ import annotations

import json
from itertools import product
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


def _accepts(initial: decoding.Prefix, text: str, *, max_nodes: int | None = None) -> bool:
    # A lexical prefix can have several token segmentations. Retain every
    # matching successor rather than choosing a first traversal or piece.
    pending = [(initial, 0)]
    visited = 0
    while pending:
        visited += 1
        if max_nodes is not None and visited > max_nodes:
            raise AssertionError(f"Selected path exceeded {max_nodes} prefix states: {text!r}")
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


def _temporary_witness_mixed_bare_benzene(text: str) -> bool:
    """Interpret selected six-carbon ring spellings without RDKit implicit H perception."""
    parsed = Chem.MolFromSmiles(text, sanitize=False)
    if parsed is None or parsed.GetNumAtoms() != 6 or parsed.GetNumBonds() != 6:
        return False
    if len(Chem.GetMolFrags(parsed)) != 1:
        return False
    atoms = tuple(parsed.GetAtoms())
    if any(atom.GetSymbol() != "C" or not atom.GetIsAromatic() or atom.GetDegree() != 2 or atom.GetFormalCharge()
           or atom.GetIsotope() or atom.GetAtomMapNum() or atom.GetNumExplicitHs() for atom in atoms):
        return False
    bonds = tuple(parsed.GetBonds())
    allowed = {Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.AROMATIC}
    if any(bond.GetBondType() not in allowed for bond in bonds):
        return False
    aromatic = tuple(bond for bond in bonds if bond.GetBondType() == Chem.BondType.AROMATIC)
    if len(aromatic) > 6:
        raise AssertionError("temporary witness exceeded its six-edge assignment cap")
    fixed_orders = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2}
    for assignment in product((1, 2), repeat=len(aromatic)):
        orders = {bond.GetIdx(): order for bond, order in zip(aromatic, assignment)}
        if all(sum(orders.get(bond.GetIdx(), fixed_orders.get(bond.GetBondType(), 0))
                   for bond in atom.GetBonds()) + (3 - atom.GetDegree()) == 4
               for atom in atoms):
            return True
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

    def test_selected_mixed_bracket_paths_preserve_sanitized_facts(self) -> None:
        cases = (
            ("c1ccccc1", "[cH]1:[cH]-[CH]=[CH]-[CH]=[CH]-1"),
            ("[nH]1cccc1", "[nH]1:[cH]=[CH]-[CH]=[cH]:1"),
            ("[nH+]1ccccc1", "[nH+]1:[cH]-[CH]=[CH]-[CH]=[CH]-1"),
            ("c1ccc2ccccc2c1", "[cH]1:[cH]C=C2C=CC=CC2=C1"),
        )
        for source_text, witness in cases:
            with self.subTest(source=source_text):
                source = _parse(source_text)
                _assert_same_graph(self, source, _parse(witness))
                general = decoding.start(source, language=decoding.SmilesLanguage.MAXIMAL)
                self.assertTrue(_accepts(general, witness, max_nodes=20_000))
                self.assertFalse(_accepts(general, witness.replace("[cH]", "[cH2]", 1), max_nodes=20_000))
                compact = decoding.start(source, language=decoding.SmilesLanguage.COMPACT)
                self.assertFalse(_accepts(compact, witness.translate(str.maketrans("12", "01")), max_nodes=20_000))
        benzene = decoding.start(_parse("c1ccccc1"), language=decoding.SmilesLanguage.MAXIMAL)
        self.assertFalse(_accepts(benzene, "[cH]1:[CH]-[CH]=[CH]-[CH]=[CH]-1", max_nodes=20_000))

    def test_mixed_bare_carbon_paths_follow_explicit_h_interpretation(self) -> None:
        source = _parse("c1ccccc1")
        for spelling in ("c1=cc=cc=c1", "c1=c-c=c-c=c-1"):
            with self.subTest(spelling=spelling):
                self.assertTrue(_temporary_witness_mixed_bare_benzene(spelling))
                for language in (decoding.SmilesLanguage.MAXIMAL, decoding.SmilesLanguage.COMPACT):
                    rendered = spelling.replace("1", "0") if language == decoding.SmilesLanguage.COMPACT else spelling
                    self.assertTrue(_accepts(decoding.start(source, language=language), rendered, max_nodes=20_000))
        self.assertFalse(_temporary_witness_mixed_bare_benzene("c1=cc=cc-c1"))
        for language in (decoding.SmilesLanguage.MAXIMAL, decoding.SmilesLanguage.COMPACT):
            rendered = "c0=cc=cc-c0" if language == decoding.SmilesLanguage.COMPACT else "c1=cc=cc-c1"
            self.assertFalse(_accepts(decoding.start(source, language=language), rendered, max_nodes=20_000))

    def test_explicit_graph_hydrogen_can_neighbor_mixed_bare_carbon(self) -> None:
        params = Chem.SmilesParserParams()
        params.removeHs = False
        source = Chem.MolFromSmiles("[H]c1ccccc1", params)
        self.assertIsNotNone(source)
        assert source is not None
        self.assertEqual("H", source.GetAtomWithIdx(0).GetSymbol())
        self.assertEqual(3, source.GetAtomWithIdx(1).GetDegree())
        self.assertEqual(0, source.GetAtomWithIdx(1).GetNumImplicitHs())
        for language, witness in (
            (decoding.SmilesLanguage.MAXIMAL, "[H]c1=cc=cc=c1"),
            (decoding.SmilesLanguage.COMPACT, "[H]c0=cc=cc=c0"),
        ):
            with self.subTest(language=language):
                self.assertTrue(_accepts(decoding.start(source, language=language), witness, max_nodes=20_000))

    def test_aromatic_forest_segments_preserve_matching(self) -> None:
        cases = (
            ("c1ccccc1", ("c1cc2.c2cc1", "C1=CC=2.C2C=C1")),
            ("[nH]1cccc1", ("[nH]1cc2.c2c1", "N1C=C2.C2=C1")),
            ("c1ccccc1.c1ccccc1", ("c1cc2.c2cc1.C1=CC=CC=C1",)),
            ("c1ccccc1-c1ccccc1", ("c1ccccc1-c1ccccc1",)),
        )
        for source_text, witnesses in cases:
            source = _parse(source_text)
            for language in (decoding.SmilesLanguage.MAXIMAL, decoding.SmilesLanguage.COMPACT):
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
            (decoding.SmilesLanguage.MAXIMAL, "N1C=C2.C", "N1C=C2.C=2C1"),
            (decoding.SmilesLanguage.COMPACT, "N0C=C1.C", "N0C=C1.C=1C0"),
        ):
            initial = decoding.start(pyrrole, language=language)
            self.assertTrue(_has_prefix(initial, prefix))
            self.assertFalse(_has_prefix(initial, impossible[:-2]))
            self.assertFalse(_accepts(initial, impossible))

    def test_infeasible_mixed_integer_valence_paths_are_rejected(self) -> None:
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
