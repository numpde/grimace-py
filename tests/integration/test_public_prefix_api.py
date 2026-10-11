"""Exercise the additive immutable South Star prefix API."""

from __future__ import annotations

import importlib
import inspect
from unittest.mock import patch
import unittest

from rdkit import Chem, rdBase

import grimace
from tests.helpers.fixture_paths import checked_in_fixture_path
from tests.helpers.pinned_rdkit_fixtures import load_pinned_rdkit_fixture_cases
from tests.helpers.pinned_rdkit_fixtures import normalized_unique_sorted_strings


FIXTURE_RDKIT_VERSION = rdBase.rdkitVersion
NONSTEREO_FIXTURES = checked_in_fixture_path("rdkit_south_star2_exact_support")
STEREO_FIXTURES = checked_in_fixture_path("rdkit_south_star2_stereo_evidence")


def _cases(fixture_root: object, label: str):
    return load_pinned_rdkit_fixture_cases(
        fixture_root=fixture_root,
        rdkit_version=FIXTURE_RDKIT_VERSION,
        fixture_label=label,
    )


def _input_molecule(case: object) -> Chem.Mol:
    raw = case.raw
    source = Chem.MolFromSmiles(raw["input_smiles"])
    if source is None:
        raise AssertionError(f"RDKit could not parse {raw['input_smiles']!r}")
    if "atom_order" in raw:
        source = Chem.RenumberAtoms(source, raw["atom_order"])
    return source


def _expected(case: object) -> set[str]:
    if "expected_support" in case.raw:
        values = case.raw["expected_support"]
        field_name = "expected_support"
    else:
        values = tuple(sorted({
            row["rendered_smiles"] for row in case.raw["source_schedule_rows"]
        }))
        field_name = "source_schedule_rows.rendered_smiles"
    return set(
        normalized_unique_sorted_strings(
            values,
            field_name=field_name,
            fixture_path=case.fixture_path,
            case_id=case.case_id,
        )
    )


def _collect_outputs(prefix: object, text: str = "") -> set[str]:
    outputs = {text} if prefix.is_accepted else set()
    for token, successor in prefix.choices():
        outputs.update(_collect_outputs(successor, text + token))
    return outputs


def _immediate_signature(prefix: object) -> tuple[bool, tuple[str, ...]]:
    return (
        prefix.is_accepted,
        tuple(token for token, _successor in prefix.choices()),
    )


def _assert_prefix_liveness_and_union(prefix: object, text: str = "") -> set[str]:
    choices = prefix.choices()
    if type(choices) is not tuple:
        raise AssertionError("Prefix.choices() must return an immutable tuple")
    if any(type(choice) is not tuple or len(choice) != 2 for choice in choices):
        raise AssertionError("each choice must be an immutable (text, successor) pair")
    token_texts = tuple(token for token, _successor in choices)
    if len(token_texts) != len(set(token_texts)):
        raise AssertionError(f"Prefix.choices() repeated a token at {text!r}")

    if not prefix.is_accepted and not choices:
        raise AssertionError(f"nonaccepted prefix has no successor at {text!r}")
    language = {text} if prefix.is_accepted else set()
    for index, (token, successor) in enumerate(choices):
        if type(token) is not str or not token:
            raise AssertionError(f"choice must carry nonempty complete token text: {token!r}")
        siblings_before = tuple(
            _immediate_signature(sibling)
            for sibling_index, (_sibling_text, sibling) in enumerate(choices)
            if sibling_index != index
        )
        language.update(_assert_prefix_liveness_and_union(successor, text + token))
        siblings_after = tuple(
            _immediate_signature(sibling)
            for sibling_index, (_sibling_text, sibling) in enumerate(choices)
            if sibling_index != index
        )
        if siblings_before != siblings_after:
            raise AssertionError(f"selecting {token!r} changed a sibling at {text!r}")
    return language


def _mapped_identity(mol: Chem.Mol) -> str:
    tagged = Chem.Mol(mol)
    for atom in tagged.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    return Chem.MolToSmiles(tagged, canonical=True, isomericSmiles=True)


def _triene_molecule(case: object) -> Chem.Mol:
    source = _input_molecule(case)
    for fact in case.raw["configured_bonds"]:
        bond = source.GetBondWithIdx(fact["index"])
        if (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) != (
            fact["begin"], fact["end"]
        ):
            raise AssertionError(f"triene fixture endpoints disagree at bond {fact['index']}")
        if fact["stereo"] == "STEREONONE":
            if bond.GetStereo() != Chem.BondStereo.STEREONONE:
                raise AssertionError(f"triene middle bond {fact['index']} is configured in the input")
            continue
        if fact["stereo"] != "STEREOE":
            raise AssertionError(f"unsupported configured triene fixture stereo {fact['stereo']!r}")
        bond.SetStereoAtoms(*fact["stereo_atoms"])
        bond.SetStereo(Chem.BondStereo.STEREOE)
    return source


class PublicPrefixPinnedSupportTests(unittest.TestCase):
    """Compare complete public languages with this RDKit version's fixtures."""

    @classmethod
    def setUpClass(cls) -> None:
        from grimace import decoding

        cls.decoding = decoding
        cls.nonstereo_cases = _cases(NONSTEREO_FIXTURES, "South Star 2 exact support")
        cls.stereo_cases = _cases(STEREO_FIXTURES, "South Star 2 stereo evidence")

    def test_public_prefix_language_equals_pinned_writer_support(self) -> None:
        cases = [*self.nonstereo_cases]
        cases.extend(
            case
            for case in self.stereo_cases
            if case.case_id
            in {
                "stereo_tetra_ccw_tree",
                "stereo_tetra_cw_tree",
                "stereo_mixed_alkene_tetra_tree",
                "stereo_e_butene_tree",
                "stereo_z_butene_tree",
            }
        )
        self.assertEqual(
            {
                "ss2_c_o_linear_ethanol",
                "ss2_c_o_isopropanol",
                "ss2_c_cycle_triangle",
                "ss2_disconnected_carbon_oxygen",
                "ss2_disconnected_cc_o_ethane_component_first",
                "ss2_disconnected_cc_o_oxygen_component_first",
                "stereo_tetra_ccw_tree",
                "stereo_tetra_cw_tree",
                "stereo_mixed_alkene_tetra_tree",
                "stereo_e_butene_tree",
                "stereo_z_butene_tree",
            },
            {case.case_id for case in cases},
        )
        for case in cases:
            with self.subTest(case=case.case_id):
                source = _input_molecule(case)
                prefix = self.decoding.start(source)
                self.assertEqual(_expected(case), _collect_outputs(prefix))


class PublicPrefixLivenessTests(unittest.TestCase):
    """Check bounded reachable-prefix laws against the fixed convention."""

    @classmethod
    def setUpClass(cls) -> None:
        from grimace import decoding

        cls.decoding = decoding
        cls.nonstereo_cases = _cases(NONSTEREO_FIXTURES, "South Star 2 exact support")
        cls.stereo_cases = tuple(
            case
            for case in _cases(STEREO_FIXTURES, "South Star 2 stereo evidence")
            if case.case_id in {
                "stereo_tetra_ccw_tree",
                "stereo_tetra_cw_tree",
                "stereo_mixed_alkene_tetra_tree",
                "stereo_e_butene_tree",
                "stereo_z_butene_tree",
            }
        )

    def test_prefix_liveness_eos_union_and_sibling_ownership(self) -> None:
        case = next(case for case in self.nonstereo_cases if case.case_id == "ss2_c_o_linear_ethanol")
        root = self.decoding.start(_input_molecule(case))
        initial = root.choices()
        token_texts = tuple(token for token, _ in initial)
        self.assertEqual(len(token_texts), len(set(token_texts)))
        self.assertGreaterEqual(len(initial), 2)
        first_token, first = initial[0]
        second_token, second = initial[1]
        self.assertNotEqual(first_token, second_token)

        root_before = tuple(token for token, _ in root.choices())
        second_before = tuple(token for token, _ in second.choices())
        _assert_prefix_liveness_and_union(first, first_token)
        self.assertEqual(root_before, tuple(token for token, _ in root.choices()))
        self.assertEqual(second_before, tuple(token for token, _ in second.choices()))

        # Acceptance contributes EOS independently of longer continuations.
        for case in (*self.nonstereo_cases, *self.stereo_cases):
            with self.subTest(case=case.case_id):
                prefix = self.decoding.start(_input_molecule(case))
                self.assertEqual(
                    _collect_outputs(prefix),
                    _assert_prefix_liveness_and_union(prefix),
                )

        triene = next(
            case for case in _cases(STEREO_FIXTURES, "South Star 2 stereo evidence")
            if case.case_id == "unspecified_middle_alkene_receives_writer_directional_marks"
        )
        triene_prefix = self.decoding.start(_triene_molecule(triene))
        self.assertEqual(
            _collect_outputs(triene_prefix),
            _assert_prefix_liveness_and_union(triene_prefix),
        )

        atom = Chem.MolFromSmiles("C")
        self.assertIsNotNone(atom)
        assert atom is not None
        atom_choices = self.decoding.start(atom).choices()
        self.assertEqual(1, len(atom_choices))
        _token, accepted = atom_choices[0]
        self.assertTrue(accepted.is_accepted)
        self.assertEqual((), accepted.choices())

    def test_choices_after_start_use_only_the_owned_native_prefix(self) -> None:
        case = next(case for case in self.stereo_cases if case.case_id == "stereo_mixed_alkene_tetra_tree")
        prefix = self.decoding.start(_input_molecule(case))
        from grimace import _runtime
        from grimace._south_star2 import preparation
        from grimace._south_star1 import public_continuation_asset

        with (
            patch.object(preparation, "prepare_molecule", side_effect=AssertionError),
            patch.object(
                preparation.PreparedMolecule,
                "build_writer",
                side_effect=AssertionError,
            ),
            patch.object(_runtime, "mol_to_smiles_enum", side_effect=AssertionError),
            patch.object(_runtime, "mol_to_smiles_support", side_effect=AssertionError),
            patch.object(Chem, "MolToSmiles", side_effect=AssertionError),
            patch.object(
                public_continuation_asset,
                "build_mol_to_smiles_continuation_asset",
                side_effect=AssertionError,
            ),
            patch.object(
                _runtime,
                "enumerate_rooted_connected_nonstereo_smiles_support",
                side_effect=AssertionError,
            ),
            patch.object(
                _runtime,
                "enumerate_rooted_connected_stereo_smiles_support",
                side_effect=AssertionError,
            ),
        ):
            self.assertTrue(prefix.choices())
            self.assertTrue(_assert_prefix_liveness_and_union(prefix))

class PublicPrefixBoundaryTests(unittest.TestCase):
    def test_additive_api_has_only_the_selected_prefix_surface(self) -> None:
        from grimace import decoding

        self.assertEqual(tuple(inspect.signature(decoding.start).parameters), ("mol",))
        self.assertFalse(hasattr(grimace, "Prefix"))
        self.assertFalse(hasattr(grimace, "start"))
        mol = Chem.MolFromSmiles("C")
        self.assertIsNotNone(mol)
        assert mol is not None
        prefix = decoding.start(mol)
        for name in ("advance", "select_text", "frontier_size", "count", "enumerate"):
            self.assertFalse(hasattr(prefix, name), name)

    def test_import_and_existing_api_remain_available(self) -> None:
        from grimace import decoding

        mol = Chem.MolFromSmiles("CCO")
        self.assertIsNotNone(mol)
        assert mol is not None
        self.assertIs(importlib.import_module("grimace.decoding"), decoding)
        self.assertTrue(callable(grimace.MolToSmilesEnum))
        old_api_outputs = tuple(
            grimace.MolToSmilesEnum(mol, canonical=False, doRandom=True)
        )
        self.assertTrue(old_api_outputs)
        self.assertTrue(all(type(output) is str for output in old_api_outputs))

    def test_rdkit_version_label_does_not_select_a_writer_policy(self) -> None:
        from grimace import decoding

        mol = Chem.MolFromSmiles("C")
        self.assertIsNotNone(mol)
        assert mol is not None
        actual_choices = tuple(token for token, _ in decoding.start(mol).choices())
        with patch.object(rdBase, "rdkitVersion", "2026.03.4"):
            labeled_choices = tuple(token for token, _ in decoding.start(mol).choices())
        self.assertEqual(actual_choices, labeled_choices)

    def test_unsupported_molecular_facts_still_fail_preparation(self) -> None:
        from grimace import decoding

        nitrogen = Chem.MolFromSmiles("N")
        self.assertIsNotNone(nitrogen)
        assert nitrogen is not None
        with self.assertRaisesRegex(ValueError, "atomic number"):
            decoding.start(nitrogen)

    def test_prefix_and_choices_are_immutable_and_source_is_not_mutated(self) -> None:
        from grimace import decoding

        mol = Chem.MolFromSmiles("C[C@H](O)CC")
        self.assertIsNotNone(mol)
        assert mol is not None
        before = _mapped_identity(mol)
        prefix = decoding.start(mol)
        self.assertEqual(before, _mapped_identity(mol))
        outputs_before = _collect_outputs(prefix)
        choices_before = prefix.choices()
        choice_texts_before = tuple(token for token, _ in choices_before)
        self.assertIs(type(choices_before), tuple)
        mol.GetAtomWithIdx(0).SetAtomMapNum(77)
        self.assertEqual(choice_texts_before, tuple(token for token, _ in prefix.choices()))
        self.assertEqual(outputs_before, _collect_outputs(prefix))
        with self.assertRaises((AttributeError, TypeError)):
            prefix.is_accepted = not prefix.is_accepted
        with self.assertRaises((AttributeError, TypeError)):
            choices_before[0][0] = "mutated"
        self.assertFalse(hasattr(prefix, "advance"))

class PublicPrefixSemanticTests(unittest.TestCase):
    """Check graph, hydrogen, and stereo identity independently of parity."""

    def test_outputs_preserve_mapped_graph_hydrogen_and_stereo_identity(self) -> None:
        from grimace import decoding

        cases = list(_cases(NONSTEREO_FIXTURES, "South Star 2 exact support"))
        cases.extend(
            case
            for case in _cases(STEREO_FIXTURES, "South Star 2 stereo evidence")
            if case.case_id
            in {
                "stereo_tetra_ccw_tree",
                "stereo_tetra_cw_tree",
                "stereo_mixed_alkene_tetra_tree",
                "stereo_e_butene_tree",
                "stereo_z_butene_tree",
            }
        )
        for case in cases:
            source = _input_molecule(case)
            expected_identity = _mapped_identity(source)
            for output in _collect_outputs(decoding.start(source)):
                with self.subTest(case=case.case_id, output=output):
                    parsed = Chem.MolFromSmiles(output)
                    self.assertIsNotNone(parsed, output)
                    assert parsed is not None
                    self.assertEqual(source.GetNumAtoms(), parsed.GetNumAtoms(), output)
                    self.assertEqual(source.GetNumBonds(), parsed.GetNumBonds(), output)
                    mapped_matches: set[str] = set()
                    for mapping in parsed.GetSubstructMatches(source, useChirality=True):
                        if not all(
                            parsed.GetAtomWithIdx(mapping[source_index]).GetTotalNumHs()
                            == source_atom.GetTotalNumHs()
                            for source_index, source_atom in enumerate(source.GetAtoms())
                        ):
                            continue
                        if any(
                            parsed.GetAtomWithIdx(mapping[source_atom.GetIdx()]).GetChiralTag()
                            != Chem.ChiralType.CHI_UNSPECIFIED
                            for source_atom in source.GetAtoms()
                            if source_atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED
                        ):
                            continue
                        tagged = Chem.Mol(parsed)
                        for source_index, parsed_index in enumerate(mapping):
                            tagged.GetAtomWithIdx(parsed_index).SetAtomMapNum(source_index + 1)
                        mapped_matches.add(
                            Chem.MolToSmiles(tagged, canonical=True, isomericSmiles=True)
                        )
                    self.assertIn(expected_identity, mapped_matches)

    def test_seeded_unspecified_stereo_controls_remain_unassigned(self) -> None:
        from grimace import decoding

        cases = {
            case.case_id: case
            for case in _cases(STEREO_FIXTURES, "South Star 2 stereo evidence")
        }
        for case_id in (
            "unspecified_alkene_with_tetrahedral_stereo",
            "unspecified_tetrahedron_with_e_alkene",
        ):
            case = cases[case_id]
            source = _input_molecule(case)
            outputs = _collect_outputs(decoding.start(source))
            witnesses = {row["smiles"] for row in case.raw["witnesses"]}
            self.assertTrue(witnesses <= outputs, case_id)
            for output in witnesses:
                parsed = Chem.MolFromSmiles(output)
                self.assertIsNotNone(parsed, output)
                assert parsed is not None
                mappings = parsed.GetSubstructMatches(source, useChirality=True)
                self.assertTrue(mappings, output)
                preserved = False
                for mapping in mappings:
                    if any(
                        parsed.GetAtomWithIdx(mapping[fact["atom_index"]]).GetChiralTag()
                        != Chem.ChiralType.CHI_UNSPECIFIED
                        for fact in case.raw["unspecified_stereo"]
                        if fact["kind"] == "tetrahedral"
                    ):
                        continue
                    if any(
                        parsed.GetBondBetweenAtoms(
                            mapping[source.GetBondWithIdx(fact["bond_index"]).GetBeginAtomIdx()],
                            mapping[source.GetBondWithIdx(fact["bond_index"]).GetEndAtomIdx()],
                        ).GetStereo() != Chem.BondStereo.STEREONONE
                        for fact in case.raw["unspecified_stereo"]
                        if fact["kind"] == "alkene"
                    ):
                        continue
                    preserved = True
                    break
                self.assertTrue(preserved, output)

    def test_pinned_triene_witnesses_record_the_middle_bond_parseback_discrepancy(self) -> None:
        from grimace import decoding

        case = next(
            case
            for case in _cases(STEREO_FIXTURES, "South Star 2 stereo evidence")
            if case.case_id == "unspecified_middle_alkene_receives_writer_directional_marks"
        )
        source = _triene_molecule(case)
        outputs = _collect_outputs(decoding.start(source))
        for witness in case.raw["witnesses"]:
            with self.subTest(seed=witness["seed"]):
                output = witness["smiles"]
                self.assertIn(output, outputs)
                parsed = Chem.MolFromSmiles(output)
                self.assertIsNotNone(parsed, output)
                assert parsed is not None
                self.assertEqual(source.GetNumAtoms(), parsed.GetNumAtoms(), output)
                self.assertEqual(source.GetNumBonds(), parsed.GetNumBonds(), output)
                middle = next(
                    fact
                    for fact in case.raw["configured_bonds"]
                    if fact["stereo"] == "STEREONONE"
                )
                mappings = parsed.GetSubstructMatches(source, useChirality=True)
                self.assertTrue(mappings, output)
                middle_stereo = {
                    "STEREOE": Chem.BondStereo.STEREOE,
                    "STEREOZ": Chem.BondStereo.STEREOZ,
                }[witness["parsed_middle_stereo"]]
                self.assertTrue(
                    any(
                        all(
                            parsed.GetAtomWithIdx(mapping[source_atom.GetIdx()]).GetTotalNumHs()
                            == source_atom.GetTotalNumHs()
                            for source_atom in source.GetAtoms()
                        )
                        and parsed.GetBondBetweenAtoms(
                            mapping[source.GetBondWithIdx(middle["index"]).GetBeginAtomIdx()],
                            mapping[source.GetBondWithIdx(middle["index"]).GetEndAtomIdx()],
                        ).GetStereo() == middle_stereo
                        for mapping in mappings
                    ),
                    output,
                )
