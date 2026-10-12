"""Check semantic graph preservation through real preparation and native walking."""

from __future__ import annotations

import unittest

from rdkit import Chem, rdBase

from grimace._south_star2.preparation import prepare_molecule
from tests.helpers.fixture_paths import checked_in_fixture_path
from tests.helpers.pinned_rdkit_fixtures import load_pinned_rdkit_fixture_cases
from tests.helpers.pinned_rdkit_fixtures import normalized_unique_sorted_strings


RDKIT_VERSION = rdBase.rdkitVersion
PINNED_STEREO_FIXTURES = checked_in_fixture_path("rdkit_south_star2_stereo_evidence")


def _accepted_strings(mol: Chem.Mol) -> set[str]:
    """Collect finite RDKit writer support for pinned parity and witness checks."""
    return _collect_writer_strings(
        prepare_molecule(mol).build_writer(style="rdkit_random_noncanonical")
    )


def _generic_witness_prefix(text: str) -> bool:
    """Restrict lexical choices, preserving every matching native hidden state.

    This finite semantic witness uses bare C/O, short carbon tetrahedral
    markers, and canonical lowest-free positive ring labels. Native Generic
    traversal, bond placement, and stereo constraints remain unrestricted;
    this does not enumerate the full Generic SMILES language.
    """
    brackets = ("[C@H]", "[C@@H]", "[C@]", "[C@@]")
    open_labels: set[int] = set()
    index = 0
    while index < len(text):
        char = text[index]
        if char == "[":
            end = text.find("]", index)
            if end == -1:
                return any(marker.startswith(text[index:]) for marker in brackets)
            if text[index:end + 1] not in brackets:
                return False
            index = end + 1
            continue
        if char.isdigit() or char == "%":
            first_free = 1
            while first_free in open_labels:
                first_free += 1
            allowed = open_labels | {first_free}
            spellings = {
                str(label) if label < 10 else f"%{label}": label
                for label in allowed
            }
            length = 3 if char == "%" else 1
            spelling = text[index:index + length]
            if len(spelling) < length:
                return any(candidate.startswith(spelling) for candidate in spellings)
            if spelling not in spellings:
                return False
            label = spellings[spelling]
            if label in open_labels:
                open_labels.remove(label)
            else:
                open_labels.add(label)
            index += length
            continue
        if char not in "CO()-=/#\\.":
            return False
        index += 1
    return True


def _accepted_generic_witness_strings(mol: Chem.Mol) -> set[str]:
    return _collect_writer_strings(
        prepare_molecule(mol).build_writer(style=None),
        finite_generic_witness=True,
    )


def _collect_writer_strings(writer, *, finite_generic_witness: bool = False) -> set[str]:
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
        frontier.extend(
            (successor, text + token) for token, successor in choices
            if not finite_generic_witness or _generic_witness_prefix(text + token)
        )
    return accepted


def _prefix_strings(
    mol: Chem.Mol, *, style: str | None
) -> tuple[list[str], list[str]]:
    """Collect writer parity, or the declared finite Generic lexical witness."""
    prefix = prepare_molecule(mol).build_writer(style=style).initial_prefix()
    pending = [(prefix, "")]
    accepted: list[str] = []
    dead: list[str] = []
    while pending:
        current, text = pending.pop()
        if current.is_accepted:
            accepted.append(text)
        choices = current.choices()
        if not current.is_accepted and not choices:
            dead.append(text)
        pending.extend(
            (successor, text + token) for token, successor in choices
            if style is not None or _generic_witness_prefix(text + token)
        )
    return accepted, dead


def _stereo_cases():
    return load_pinned_rdkit_fixture_cases(
        fixture_root=PINNED_STEREO_FIXTURES,
        rdkit_version=RDKIT_VERSION,
        fixture_label="South Star 2 stereo evidence",
    )


def _triene_source(case) -> Chem.Mol:
    molecule = Chem.MolFromSmiles(case.raw["input_smiles"])
    if molecule is None:
        raise AssertionError(f"Invalid triene fixture input: {case.raw['input_smiles']!r}")
    stereo_by_name = {"STEREOE": Chem.BondStereo.STEREOE}
    for fact in case.raw["configured_bonds"]:
        bond = molecule.GetBondWithIdx(fact["index"])
        if (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) != (fact["begin"], fact["end"]):
            raise AssertionError(f"Triene fixture bond endpoints disagree at bond {fact['index']}")
        if fact["stereo"] == "STEREONONE":
            if bond.GetStereo() != Chem.BondStereo.STEREONONE:
                raise AssertionError(f"Triene middle bond {fact['index']} is unexpectedly configured")
            continue
        bond.SetStereoAtoms(*fact["stereo_atoms"])
        bond.SetStereo(stereo_by_name[fact["stereo"]])
    return molecule


def _tetrahedral_stereo_cases():
    cases = load_pinned_rdkit_fixture_cases(
        fixture_root=PINNED_STEREO_FIXTURES,
        rdkit_version=RDKIT_VERSION,
        fixture_label="South Star 2 stereo evidence",
    )
    by_id = {case.case_id: case for case in cases}
    return (
        by_id["stereo_tetra_ccw_tree"],
        by_id["stereo_tetra_cw_tree"],
    )


class SouthStar2PreparationWriterParityTests(unittest.TestCase):
    """Compare native output sets with pinned RDKit writer support fixtures."""

    def test_exact_tetrahedral_support_and_renumbering_for_both_enantiomers(self) -> None:
        supports: list[set[str]] = []
        for case in _tetrahedral_stereo_cases():
            with self.subTest(case=case.case_id):
                expected = set(
                    normalized_unique_sorted_strings(
                        case.raw["expected_support"],
                        field_name="expected_support",
                        fixture_path=case.fixture_path,
                        case_id=case.case_id,
                    )
                )
                source = Chem.MolFromSmiles(case.raw["input_smiles"])
                self.assertIsNotNone(source)
                assert source is not None
                self.assertEqual(expected, _accepted_strings(source))
                supports.append(expected)

                # Exact RDKit support is invariant to atom and bond ID changes.
                renumbered = Chem.RenumberAtoms(
                    source, list(reversed(range(source.GetNumAtoms())))
                )
                self.assertEqual(expected, _accepted_strings(renumbered))

        self.assertTrue(supports[0].isdisjoint(supports[1]))

    def test_exact_mixed_and_e_butene_support_in_grouped_prefix_path(self) -> None:
        by_id = {case.case_id: case for case in _stereo_cases()}
        mixed = by_id["stereo_mixed_alkene_tetra_tree"]
        mixed_source = Chem.MolFromSmiles(mixed.raw["input_smiles"])
        self.assertIsNotNone(mixed_source)
        assert mixed_source is not None
        mixed_expected = set(
            normalized_unique_sorted_strings(
                mixed.raw["expected_support"], field_name="expected_support",
                fixture_path=mixed.fixture_path, case_id=mixed.case_id,
            )
        )
        mixed_actual, mixed_dead = _prefix_strings(mixed_source, style="rdkit_random_noncanonical")
        self.assertEqual(set(mixed_actual), mixed_expected)
        self.assertEqual(len(mixed_actual), len(mixed_expected))
        self.assertFalse(mixed_dead)

        e_case = by_id["stereo_e_butene_tree"]
        e_source = Chem.MolFromSmiles(e_case.raw["input_smiles"])
        self.assertIsNotNone(e_source)
        assert e_source is not None
        e_expected = {row["rendered_smiles"] for row in e_case.raw["source_schedule_rows"]}
        e_actual, e_dead = _prefix_strings(e_source, style="rdkit_random_noncanonical")
        self.assertEqual(set(e_actual), e_expected)
        self.assertEqual(len(e_actual), len(e_expected))
        self.assertFalse(e_dead)

        z_case = by_id["stereo_z_butene_tree"]
        z_source = Chem.MolFromSmiles(z_case.raw["input_smiles"])
        self.assertIsNotNone(z_source)
        assert z_source is not None
        z_expected = {row["rendered_smiles"] for row in z_case.raw["source_schedule_rows"]}
        z_actual, z_dead = _prefix_strings(z_source, style="rdkit_random_noncanonical")
        self.assertEqual(set(z_actual), z_expected)
        self.assertEqual(len(z_actual), len(z_expected))
        self.assertFalse(z_dead)

        opposite_tetra = Chem.MolFromSmiles("C/C=C/[C@@H](O)C")
        self.assertIsNotNone(opposite_tetra)
        assert opposite_tetra is not None
        self.assertTrue(mixed_expected.isdisjoint(_accepted_strings(opposite_tetra)))
        self.assertTrue(e_expected.isdisjoint(z_expected))


class SouthStar2PreparationSemanticTests(unittest.TestCase):
    """Check the finite Generic lexical witness, independently from writer parity."""
    def test_native_outputs_parse_to_each_source_graph(self) -> None:
        for smiles in ("CCO", "CC(C)O", "C1CC1", "CC(O)CO"):
            with self.subTest(smiles=smiles):
                source = Chem.MolFromSmiles(smiles)
                assert source is not None
                outputs = _accepted_generic_witness_strings(source)
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
                            all(
                                parsed.GetAtomWithIdx(mapping[source_atom.GetIdx()]).GetChiralTag()
                                == Chem.ChiralType.CHI_UNSPECIFIED
                                for source_atom in source.GetAtoms()
                                if source_atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED
                            )
                            for mapping in mappings
                        ),
                        f"output assigned unspecified atom stereo: {output!r}",
                    )

    def test_tetrahedral_outputs_preserve_graph_hydrogens_and_chirality(self) -> None:
        for case in _tetrahedral_stereo_cases():
            source = Chem.MolFromSmiles(case.raw["input_smiles"])
            self.assertIsNotNone(source)
            assert source is not None
            outputs = _accepted_generic_witness_strings(source)
            self.assertTrue(outputs)
            for output in outputs:
                with self.subTest(case=case.case_id, output=output):
                    parsed = Chem.MolFromSmiles(output)
                    self.assertIsNotNone(parsed, output)
                    assert parsed is not None
                    self.assertEqual(source.GetNumAtoms(), parsed.GetNumAtoms(), output)
                    self.assertEqual(source.GetNumBonds(), parsed.GetNumBonds(), output)
                    mappings = parsed.GetSubstructMatches(source, useChirality=True)
                    self.assertTrue(mappings, f"stereo or graph differs: {output!r}")
                    self.assertTrue(
                        any(
                            all(
                                parsed.GetAtomWithIdx(parsed_index).GetSymbol()
                                == source.GetAtomWithIdx(source_index).GetSymbol()
                                and parsed.GetAtomWithIdx(parsed_index).GetTotalNumHs()
                                == source.GetAtomWithIdx(source_index).GetTotalNumHs()
                                for source_index, parsed_index in enumerate(mapping)
                            )
                            for mapping in mappings
                        ),
                        f"element or non-graph hydrogen facts differ: {output!r}",
                    )

    def test_mixed_outputs_preserve_graph_hydrogens_and_configured_stereo(self) -> None:
        case = next(case for case in _stereo_cases() if case.case_id == "stereo_mixed_alkene_tetra_tree")
        source = Chem.MolFromSmiles(case.raw["input_smiles"])
        self.assertIsNotNone(source)
        assert source is not None
        outputs = _accepted_generic_witness_strings(source)
        self.assertTrue(outputs)
        for output in outputs:
            parsed = Chem.MolFromSmiles(output)
            self.assertIsNotNone(parsed, output)
            assert parsed is not None
            self.assertEqual(source.GetNumAtoms(), parsed.GetNumAtoms(), output)
            self.assertEqual(source.GetNumBonds(), parsed.GetNumBonds(), output)
            mappings = parsed.GetSubstructMatches(source, useChirality=True)
            self.assertTrue(mappings, f"stereo or graph differs: {output!r}")
            self.assertTrue(
                any(
                    all(
                        parsed.GetAtomWithIdx(mapping[atom.GetIdx()]).GetAtomicNum()
                        == atom.GetAtomicNum()
                        and parsed.GetAtomWithIdx(mapping[atom.GetIdx()]).GetTotalNumHs()
                        == atom.GetTotalNumHs()
                        for atom in source.GetAtoms()
                    )
                    and all(
                        (target := parsed.GetBondBetweenAtoms(
                            mapping[bond.GetBeginAtomIdx()], mapping[bond.GetEndAtomIdx()]
                        )) is not None
                        and target.GetBondType() == bond.GetBondType()
                        and target.GetStereo() == bond.GetStereo()
                        for bond in source.GetBonds()
                    )
                    for mapping in mappings
                ),
                f"bond stereo or hydrogen facts differ: {output!r}",
            )


    def test_unconfigured_stereo_controls_remain_independent_facts(self) -> None:
        for smiles in ("CC=C[C@H](O)C", "C/C=C/C(O)C"):
            with self.subTest(smiles=smiles):
                source = Chem.MolFromSmiles(smiles)
                self.assertIsNotNone(source)
                assert source is not None
                prepared = prepare_molecule(source)
                configured = [
                    bond for bond in source.GetBonds()
                    if bond.GetBondType() == Chem.BondType.DOUBLE
                    and bond.GetStereo() != Chem.BondStereo.STEREONONE
                ]
                self.assertEqual(len(prepared.directional), len(configured))
                outputs, dead = _prefix_strings(source, style=None)
                self.assertTrue(outputs)
                self.assertFalse(dead)
                for output in outputs:
                    parsed = Chem.MolFromSmiles(output)
                    self.assertIsNotNone(parsed, output)
                    assert parsed is not None
                    matches = parsed.GetSubstructMatches(source, useChirality=True)
                    self.assertTrue(matches, f"specified stereo or graph differs: {output!r}")
                    for mapping in matches:
                        for bond in source.GetBonds():
                            if bond.GetStereo() != Chem.BondStereo.STEREONONE:
                                continue
                            corresponding = parsed.GetBondBetweenAtoms(
                                mapping[bond.GetBeginAtomIdx()], mapping[bond.GetEndAtomIdx()]
                            )
                            self.assertEqual(corresponding.GetStereo(), Chem.BondStereo.STEREONONE)

    def test_tetrahedral_prefix_retains_ambiguous_roots_and_sibling_ownership(self) -> None:
        for case in _tetrahedral_stereo_cases():
            source = Chem.MolFromSmiles(case.raw["input_smiles"])
            self.assertIsNotNone(source)
            assert source is not None
            prepared = prepare_molecule(source).build_writer()
            prefix = prepared.initial_prefix()
            initial_native = prepared.initial_state().choices()
            root_texts = {}
            for text, successor in prefix.choices():
                root_texts[text] = successor
            for text in set(text for text, _ in initial_native):
                native_count = sum(candidate == text for candidate, _ in initial_native)
                self.assertEqual(root_texts[text].frontier_size, native_count)
            self.assertGreater(root_texts["C"].frontier_size, 1)
            sibling = root_texts.get("O")
            if sibling is not None:
                self.assertEqual(sibling.frontier_size, 1)
                self.assertFalse(sibling.is_accepted)
                sibling_choices = tuple(text for text, _ in sibling.choices())
            del prepared, prefix
            self.assertTrue(root_texts["C"].choices())
            if sibling is not None:
                self.assertEqual(
                    tuple(text for text, _ in sibling.choices()), sibling_choices
                )


class SouthStar2PreparationWriterQuirkWitnessTests(unittest.TestCase):
    """Check the pinned positive writer witnesses for the known triene quirk."""

    def test_triene_outputs_include_each_seeded_writer_witness(self) -> None:
        case = next(
            case
            for case in _stereo_cases()
            if case.case_id == "unspecified_middle_alkene_receives_writer_directional_marks"
        )
        source = _triene_source(case)
        configured = [fact for fact in case.raw["configured_bonds"] if fact["stereo"] != "STEREONONE"]
        prepared = prepare_molecule(source)
        self.assertEqual(len(prepared.directional), len(configured))
        actual = _accepted_strings(source)
        self.assertTrue({witness["smiles"] for witness in case.raw["witnesses"]} <= actual)

    def test_triene_witness_parseback_records_unspecified_middle_stereo(self) -> None:
        case = next(
            case
            for case in _stereo_cases()
            if case.case_id == "unspecified_middle_alkene_receives_writer_directional_marks"
        )
        source = _triene_source(case)
        middle_fact = next(
            fact for fact in case.raw["configured_bonds"] if fact["stereo"] == "STEREONONE"
        )
        middle_source_bond = source.GetBondWithIdx(middle_fact["index"])
        self.assertEqual(Chem.BondStereo.STEREONONE, middle_source_bond.GetStereo())
        configured = [
            fact for fact in case.raw["configured_bonds"] if fact["stereo"] != "STEREONONE"
        ]
        for witness in case.raw["witnesses"]:
            parsed = Chem.MolFromSmiles(witness["smiles"])
            self.assertIsNotNone(parsed, witness["smiles"])
            assert parsed is not None
            self.assertEqual(source.GetNumAtoms(), parsed.GetNumAtoms(), witness["smiles"])
            self.assertEqual(source.GetNumBonds(), parsed.GetNumBonds(), witness["smiles"])
            matches = parsed.GetSubstructMatches(source, useChirality=True)
            self.assertTrue(matches, f"triene witness graph or configured stereo differs: {witness['smiles']!r}")
            preserved = False
            for mapping in matches:
                if not all(
                    parsed.GetAtomWithIdx(mapping[atom.GetIdx()]).GetAtomicNum() == atom.GetAtomicNum()
                    and parsed.GetAtomWithIdx(mapping[atom.GetIdx()]).GetTotalNumHs() == atom.GetTotalNumHs()
                    for atom in source.GetAtoms()
                ):
                    continue
                if any(
                    (target := parsed.GetBondBetweenAtoms(
                        mapping[bond.GetBeginAtomIdx()], mapping[bond.GetEndAtomIdx()]
                    )) is None
                    or target.GetBondType() != bond.GetBondType()
                    for bond in source.GetBonds()
                ):
                    continue
                targets = {
                    fact["index"]: parsed.GetBondBetweenAtoms(
                        mapping[fact["begin"]], mapping[fact["end"]]
                    )
                    for fact in case.raw["configured_bonds"]
                }
                if any(target is None or target.GetBondType() != Chem.BondType.DOUBLE for target in targets.values()):
                    continue
                if any(
                    targets[fact["index"]].GetStereo() != Chem.BondStereo.STEREOE
                    for fact in configured
                ):
                    continue
                middle_target = targets[middle_fact["index"]]
                expected_middle = {
                    "STEREOE": Chem.BondStereo.STEREOE,
                    "STEREOZ": Chem.BondStereo.STEREOZ,
                }[witness["parsed_middle_stereo"]]
                self.assertEqual(expected_middle, middle_target.GetStereo())
                preserved = True
            self.assertTrue(preserved, f"no source-to-parsed mapping preserves the triene witness: {witness!r}")



if __name__ == "__main__":
    unittest.main()
