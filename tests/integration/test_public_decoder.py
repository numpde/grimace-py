from __future__ import annotations

from dataclasses import dataclass
import random
import unittest

import grimace
from rdkit import Chem
from grimace import _runtime
from tests.helpers.assertions import assert_prefix_options_match_outputs
from tests.helpers.kernel import CORE_MODULE
from tests.helpers.mols import parse_smiles


@dataclass(frozen=True, slots=True)
class DecoderCase:
    name: str
    smiles: str
    rooted_at_atom: int
    isomeric_smiles: bool = True
    kekule_smiles: bool = False
    all_bonds_explicit: bool = False
    all_hs_explicit: bool = False
    ignore_atom_map_numbers: bool = False


@dataclass(frozen=True, slots=True)
class DecoderAuditCase:
    name: str
    smiles: str
    rooted_at_atom: int | None
    isomeric_smiles: bool = True
    kekule_smiles: bool = False
    all_bonds_explicit: bool = False
    all_hs_explicit: bool = False
    ignore_atom_map_numbers: bool = False


class PublicDecoderTests(unittest.TestCase):
    CASES = (
        DecoderCase(
            name="nonstereo_baseline",
            smiles="CCO",
            rooted_at_atom=0,
            isomeric_smiles=False,
        ),
        DecoderCase(
            name="nonstereo_atom_stereo_input",
            smiles="F[C@H](Cl)Br",
            rooted_at_atom=0,
            isomeric_smiles=False,
        ),
        DecoderCase(
            name="nonstereo_bond_stereo_input",
            smiles="F/C=C\\Cl",
            rooted_at_atom=0,
            isomeric_smiles=False,
        ),
        DecoderCase(
            name="stereo_baseline",
            smiles="F/C=C\\Cl",
            rooted_at_atom=0,
            isomeric_smiles=True,
        ),
        DecoderCase(
            name="kekule_smiles",
            smiles="c1ccncc1",
            rooted_at_atom=0,
            isomeric_smiles=False,
            kekule_smiles=True,
        ),
        DecoderCase(
            name="all_bonds_explicit",
            smiles="CC#N",
            rooted_at_atom=0,
            isomeric_smiles=False,
            all_bonds_explicit=True,
        ),
        DecoderCase(
            name="all_hs_explicit",
            smiles="C",
            rooted_at_atom=0,
            isomeric_smiles=False,
            all_hs_explicit=True,
        ),
        DecoderCase(
            name="ignore_atom_map_numbers",
            smiles="[CH3:7]C",
            rooted_at_atom=0,
            isomeric_smiles=False,
            ignore_atom_map_numbers=True,
        ),
    )

    @classmethod
    def setUpClass(cls) -> None:
        if CORE_MODULE is None:
            raise unittest.SkipTest("private Rust extension is not installed")

    def _make_decoder(
        self,
        case: DecoderCase,
    ) -> grimace.MolToSmilesDecoder:
        return grimace.MolToSmilesDecoder(
            parse_smiles(case.smiles),
            isomericSmiles=case.isomeric_smiles,
            kekuleSmiles=case.kekule_smiles,
            rootedAtAtom=case.rooted_at_atom,
            canonical=False,
            allBondsExplicit=case.all_bonds_explicit,
            allHsExplicit=case.all_hs_explicit,
            doRandom=True,
            ignoreAtomMapNumbers=case.ignore_atom_map_numbers,
        )

    def _make_determinized_decoder(
        self,
        case: DecoderCase,
    ) -> grimace.MolToSmilesDeterminizedDecoder:
        return grimace.MolToSmilesDeterminizedDecoder(
            parse_smiles(case.smiles),
            isomericSmiles=case.isomeric_smiles,
            kekuleSmiles=case.kekule_smiles,
            rootedAtAtom=case.rooted_at_atom,
            canonical=False,
            allBondsExplicit=case.all_bonds_explicit,
            allHsExplicit=case.all_hs_explicit,
            doRandom=True,
            ignoreAtomMapNumbers=case.ignore_atom_map_numbers,
        )

    def _enumerate_outputs(self, case: DecoderCase) -> set[str]:
        return set(
            grimace.MolToSmilesEnum(
                parse_smiles(case.smiles),
                isomericSmiles=case.isomeric_smiles,
                kekuleSmiles=case.kekule_smiles,
                rootedAtAtom=case.rooted_at_atom,
                canonical=False,
                allBondsExplicit=case.all_bonds_explicit,
                allHsExplicit=case.all_hs_explicit,
                doRandom=True,
                ignoreAtomMapNumbers=case.ignore_atom_map_numbers,
            )
        )

    def _atom_tokens(self, case: DecoderCase) -> tuple[str, ...]:
        mol = parse_smiles(case.smiles)
        flags = _runtime.MolToSmilesFlags(
            isomeric_smiles=case.isomeric_smiles,
            kekule_smiles=case.kekule_smiles,
            rooted_at_atom=case.rooted_at_atom,
            canonical=False,
            all_bonds_explicit=case.all_bonds_explicit,
            all_hs_explicit=case.all_hs_explicit,
            do_random=True,
            ignore_atom_map_numbers=case.ignore_atom_map_numbers,
        )
        if len(Chem.GetMolFrags(mol)) == 1:
            prepared = _runtime.prepare_smiles_graph(mol, flags=flags)
            return tuple(prepared.atom_tokens)

        atom_tokens: set[str] = set()
        for fragment_mol in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False):
            prepared = _runtime.prepare_smiles_graph(
                fragment_mol,
                flags=flags.with_rooted_at_atom(0),
            )
            atom_tokens.update(prepared.atom_tokens)
        return tuple(sorted(atom_tokens))

    @staticmethod
    def _choice_texts(decoder: grimace.MolToSmilesDecoder) -> tuple[str, ...]:
        return tuple(choice.text for choice in decoder.next_choices)

    @staticmethod
    def _unique_choice_texts(decoder: grimace.MolToSmilesDecoder) -> tuple[str, ...]:
        return tuple(sorted({choice.text for choice in decoder.next_choices}))

    @staticmethod
    def _reachable_outputs_from_decoder(
        decoder: object,
        *,
        memo: dict[str, frozenset[str]] | None = None,
    ) -> frozenset[str]:
        return _runtime._reachable_terminal_prefixes(decoder._impl._state, memo=memo)

    def test_decoder_sampled_paths_stay_within_public_enum_outputs(self) -> None:
        for case in self.CASES:
            outputs = self._enumerate_outputs(case)
            for seed in range(3):
                with self.subTest(case=case.name, smiles=case.smiles, seed=seed):
                    rng = random.Random(seed)
                    decoder = self._make_decoder(case)
                    atom_tokens = self._atom_tokens(case)
                    chosen_tokens: list[str] = []

                    while not decoder.is_terminal:
                        choices = decoder.next_choices
                        options = self._choice_texts(decoder)
                        branch_outputs = self._reachable_outputs_from_decoder(decoder)
                        self.assertTrue(choices)
                        self.assertTrue(branch_outputs)
                        self.assertTrue(branch_outputs <= outputs)
                        assert_prefix_options_match_outputs(
                            self,
                            decoder.prefix,
                            tuple(sorted(set(options))),
                            branch_outputs,
                            atom_tokens=atom_tokens,
                        )
                        chosen_choice = rng.choice(choices)
                        chosen_tokens.append(chosen_choice.text)
                        decoder = chosen_choice.next_state

                    self.assertEqual((), decoder.next_choices)
                    self.assertEqual(decoder.prefix, "".join(chosen_tokens))
                    self.assertIn(decoder.prefix, outputs)

    def test_decoder_copy_forks_state_without_mutating_original(self) -> None:
        case = DecoderCase(
            name="branching_stereo",
            smiles="F[C@H](Cl)Br",
            rooted_at_atom=0,
            isomeric_smiles=True,
        )
        outputs = self._enumerate_outputs(case)
        decoder = self._make_decoder(case)

        while not decoder.is_terminal:
            options = decoder.next_choices
            distinct_pairs = [
                (left_choice, right_choice)
                for idx, left_choice in enumerate(options)
                for right_choice in options[idx + 1 :]
                if self._reachable_outputs_from_decoder(left_choice.next_state)
                != self._reachable_outputs_from_decoder(right_choice.next_state)
            ]
            if distinct_pairs:
                break
            decoder = options[0].next_state

        self.assertGreater(len(decoder.next_choices), 1)
        original_prefix = decoder.prefix
        left = decoder.copy()
        right = decoder.copy()
        self.assertTrue(distinct_pairs)
        left_choice, right_choice = distinct_pairs[0]

        left = left_choice.next_state
        right = right_choice.next_state
        left_outputs = self._reachable_outputs_from_decoder(left)
        right_outputs = self._reachable_outputs_from_decoder(right)

        self.assertEqual(original_prefix, decoder.prefix)
        self.assertNotEqual(left_outputs, right_outputs)
        self.assertTrue(left_outputs <= outputs)
        self.assertTrue(right_outputs <= outputs)

    def test_decoder_rejects_invalid_token_with_available_choices(self) -> None:
        decoder = grimace.MolToSmilesDecoder(
            parse_smiles("CCO"),
            rootedAtAtom=0,
            isomericSmiles=False,
            canonical=False,
            doRandom=True,
        )

        with self.assertRaisesRegex(KeyError, "choice_count"):
            walker = grimace._runtime.make_nonstereo_walker(parse_smiles("CCO"), 0)
            walker.advance_choice(walker.initial_state(), 99)

    def test_decoder_rejects_unsupported_flag_combinations(self) -> None:
        mol = parse_smiles("CCO")

        with self.assertRaisesRegex(NotImplementedError, "rootedAtAtom == -1 or rootedAtAtom >= 0"):
            grimace.MolToSmilesDecoder(
                mol,
                rootedAtAtom=-2,
                isomericSmiles=False,
                canonical=False,
                doRandom=True,
            )
        with self.assertRaisesRegex(NotImplementedError, "canonical=False"):
            grimace.MolToSmilesDecoder(
                mol,
                rootedAtAtom=0,
                isomericSmiles=False,
                doRandom=True,
            )
        with self.assertRaisesRegex(NotImplementedError, "doRandom=True"):
            grimace.MolToSmilesDecoder(
                mol,
                rootedAtAtom=0,
                isomericSmiles=False,
                canonical=False,
            )

    def test_decoder_without_explicit_root_samples_paths_within_public_enum_outputs(self) -> None:
        for smiles, isomeric_smiles in (
            ("CCO", False),
            ("O.CCO", True),
            ("F[C@H](Cl)Br", True),
        ):
            mol = parse_smiles(smiles)
            outputs = set(
                grimace.MolToSmilesEnum(
                    mol,
                    isomericSmiles=isomeric_smiles,
                    canonical=False,
                    doRandom=True,
                )
            )
            with self.subTest(smiles=smiles, isomeric_smiles=isomeric_smiles):
                for seed in range(3):
                    rng = random.Random(seed)
                    decoder = grimace.MolToSmilesDecoder(
                        mol,
                        isomericSmiles=isomeric_smiles,
                        canonical=False,
                        doRandom=True,
                    )
                    while not decoder.is_terminal:
                        choice = rng.choice(decoder.next_choices)
                        decoder = choice.next_state
                    self.assertIn(decoder.prefix, outputs)

    def test_decoder_supports_disconnected_molecules_and_emits_dot_between_fragments(self) -> None:
        decoder = grimace.MolToSmilesDecoder(
            parse_smiles("[Na+].C#N"),
            rootedAtAtom=0,
            isomericSmiles=False,
            canonical=False,
            doRandom=True,
        )

        self.assertEqual("", decoder.prefix)
        self.assertEqual(("[Na+]",), self._choice_texts(decoder))
        decoder = decoder.next_choices[0].next_state
        self.assertEqual("[Na+]", decoder.prefix)
        self.assertEqual((".",), self._choice_texts(decoder))
        decoder = decoder.next_choices[0].next_state
        self.assertEqual("[Na+].", decoder.prefix)
        self.assertEqual(("C", "N"), self._choice_texts(decoder))

    def test_decoder_preserves_duplicate_same_text_choices(self) -> None:
        decoder = grimace.MolToSmilesDecoder(
            parse_smiles("[Na+].CC"),
            rootedAtAtom=0,
            isomericSmiles=False,
            canonical=False,
            doRandom=True,
        )

        decoder = decoder.next_choices[0].next_state
        self.assertEqual("[Na+]", decoder.prefix)
        decoder = decoder.next_choices[0].next_state
        self.assertEqual("[Na+].", decoder.prefix)
        self.assertEqual(("C", "C"), self._choice_texts(decoder))
        left, right = decoder.next_choices
        left_state = left.next_state
        right_state = right.next_state
        self.assertEqual("C", left.text)
        self.assertEqual("C", right.text)
        self.assertIsNot(left_state, right_state)
        self.assertEqual("[Na+].C", left_state.prefix)
        self.assertEqual("[Na+].C", right_state.prefix)

    def test_decoder_preserves_connected_duplicate_same_text_choices(self) -> None:
        decoder = grimace.MolToSmilesDecoder(
            parse_smiles("C1CCC2=NN=NN2CC1"),
            rootedAtAtom=2,
            isomericSmiles=False,
            canonical=False,
            doRandom=True,
        )

        while decoder.prefix != "C1CCCCn":
            choices = decoder.next_choices
            self.assertTrue(choices)
            decoder = choices[0].next_state

        self.assertEqual(("2", "2"), self._choice_texts(decoder))
        branch_supports = {
            self._reachable_outputs_from_decoder(choice.next_state)
            for choice in decoder.next_choices
        }
        self.assertEqual(
            {
                frozenset({"C1CCCCn2c1nnn2"}),
                frozenset({"C1CCCCn2nnnc12", "C1CCCCn2nnnc21"}),
            },
            branch_supports,
        )

    def test_determinized_decoder_collapses_connected_duplicate_same_text_choices(self) -> None:
        decoder = grimace.MolToSmilesDeterminizedDecoder(
            parse_smiles("C1CCC2=NN=NN2CC1"),
            rootedAtAtom=2,
            isomericSmiles=False,
            canonical=False,
            doRandom=True,
        )

        while decoder.prefix != "C1CCCCn":
            choices = decoder.next_choices
            self.assertTrue(choices)
            decoder = choices[0].next_state

        self.assertEqual(("2",), self._choice_texts(decoder))
        merged_outputs = self._reachable_outputs_from_decoder(decoder.next_choices[0].next_state)
        self.assertEqual(
            frozenset({"C1CCCCn2c1nnn2", "C1CCCCn2nnnc12", "C1CCCCn2nnnc21"}),
            merged_outputs,
        )

    def test_determinized_decoder_exposes_visible_divergence_after_same_text_merge(self) -> None:
        decoder = grimace.MolToSmilesDeterminizedDecoder(
            parse_smiles("CC(=O)Oc1ccccc1C(=O)O"),
            rootedAtAtom=9,
            isomericSmiles=False,
            canonical=False,
            doRandom=True,
        )

        self.assertEqual(("c",), self._choice_texts(decoder))
        decoder = decoder.next_choices[0].next_state
        self.assertEqual("c", decoder.prefix)
        self.assertEqual(("1",), self._choice_texts(decoder))
        decoder = decoder.next_choices[0].next_state
        self.assertEqual("c1", decoder.prefix)
        self.assertEqual(("(",), self._choice_texts(decoder))
        decoder = decoder.next_choices[0].next_state
        self.assertEqual("c1(", decoder.prefix)
        self.assertEqual(("C", "c"), self._choice_texts(decoder))

    def test_determinized_decoder_supports_disconnected_visible_forks(self) -> None:
        decoder = grimace.MolToSmilesDeterminizedDecoder(
            parse_smiles("[Na+].C#N"),
            rootedAtAtom=0,
            isomericSmiles=False,
            canonical=False,
            doRandom=True,
        )

        self.assertEqual(("[Na+]",), self._choice_texts(decoder))
        decoder = decoder.next_choices[0].next_state
        self.assertEqual("[Na+]", decoder.prefix)
        self.assertEqual((".",), self._choice_texts(decoder))
        decoder = decoder.next_choices[0].next_state
        self.assertEqual("[Na+].", decoder.prefix)
        self.assertEqual(("C", "N"), self._choice_texts(decoder))

    def test_determinized_decoder_state_audit_covers_all_reachable_states(self) -> None:
        cases = (
            DecoderAuditCase(
                name="rooted_nonstereo",
                smiles="CCO",
                rooted_at_atom=0,
                isomeric_smiles=False,
            ),
            DecoderAuditCase(
                name="rooted_stereo",
                smiles="F[C@H](Cl)Br",
                rooted_at_atom=0,
                isomeric_smiles=True,
            ),
            DecoderAuditCase(
                name="disconnected_rooted",
                smiles="[Na+].C#N",
                rooted_at_atom=0,
                isomeric_smiles=False,
            ),
            DecoderAuditCase(
                name="duplicate_same_text_connected",
                smiles="C1CCC2=NN=NN2CC1",
                rooted_at_atom=2,
                isomeric_smiles=False,
            ),
            DecoderAuditCase(
                name="merged_then_visible_divergence",
                smiles="CC(=O)Oc1ccccc1C(=O)O",
                rooted_at_atom=9,
                isomeric_smiles=False,
            ),
        )

        for case in cases:
            mol = parse_smiles(case.smiles)
            kwargs = dict(
                isomericSmiles=case.isomeric_smiles,
                kekuleSmiles=case.kekule_smiles,
                canonical=False,
                allBondsExplicit=case.all_bonds_explicit,
                allHsExplicit=case.all_hs_explicit,
                doRandom=True,
                ignoreAtomMapNumbers=case.ignore_atom_map_numbers,
            )
            if case.rooted_at_atom is not None:
                kwargs["rootedAtAtom"] = case.rooted_at_atom
            outputs = frozenset(grimace.MolToSmilesEnum(mol, **kwargs))
            decoder = grimace.MolToSmilesDeterminizedDecoder(mol, **kwargs)
            memo: dict[object, frozenset[str]] = {}
            seen_state_keys: set[object] = set()
            stack = [decoder._impl._state]
            audited_state_count = 0

            with self.subTest(case=case.name, smiles=case.smiles):
                while stack:
                    state = stack.pop()
                    state_key = _runtime._state_cache_key(state)
                    if state_key in seen_state_keys:
                        continue
                    seen_state_keys.add(state_key)
                    audited_state_count += 1

                    reachable = _runtime._reachable_terminal_prefixes(state, memo=memo)
                    prefix = state.prefix()
                    grouped_successors = _runtime._determinized_choice_successors(state)
                    option_texts = tuple(text for text, _ in grouped_successors)

                    self.assertTrue(reachable)
                    self.assertTrue(reachable <= outputs)
                    self.assertTrue(all(output.startswith(prefix) for output in reachable))

                    if state.is_terminal():
                        self.assertEqual((), grouped_successors)
                        self.assertEqual(frozenset({prefix}), reachable)
                        continue

                    self.assertTrue(grouped_successors)
                    assert_prefix_options_match_outputs(
                        self,
                        prefix,
                        option_texts,
                        reachable,
                        atom_tokens=self._atom_tokens(
                            DecoderCase(
                                name=case.name,
                                smiles=case.smiles,
                                rooted_at_atom=0 if case.rooted_at_atom is None else case.rooted_at_atom,
                                isomeric_smiles=case.isomeric_smiles,
                                kekule_smiles=case.kekule_smiles,
                                all_bonds_explicit=case.all_bonds_explicit,
                                all_hs_explicit=case.all_hs_explicit,
                                ignore_atom_map_numbers=case.ignore_atom_map_numbers,
                            )
                        ),
                    )

                    union_of_branch_outputs: set[str] = set()
                    for _, successor in grouped_successors:
                        branch_outputs = _runtime._reachable_terminal_prefixes(
                            successor,
                            memo=memo,
                        )
                        self.assertTrue(branch_outputs)
                        self.assertTrue(branch_outputs <= reachable)
                        self.assertTrue(
                            all(output.startswith(successor.prefix()) for output in branch_outputs)
                        )
                        union_of_branch_outputs.update(branch_outputs)
                        stack.append(successor)

                    self.assertEqual(reachable, frozenset(union_of_branch_outputs))

                self.assertGreater(audited_state_count, 0)

    def test_decoder_state_audit_covers_all_reachable_states(self) -> None:
        cases = (
            DecoderAuditCase(
                name="rooted_nonstereo",
                smiles="CCO",
                rooted_at_atom=0,
                isomeric_smiles=False,
            ),
            DecoderAuditCase(
                name="rooted_stereo",
                smiles="F[C@H](Cl)Br",
                rooted_at_atom=0,
                isomeric_smiles=True,
            ),
            DecoderAuditCase(
                name="nonisomeric_explicit_bond_dirs",
                smiles="F/C=C\\Cl",
                rooted_at_atom=0,
                isomeric_smiles=False,
                all_bonds_explicit=True,
            ),
            DecoderAuditCase(
                name="unrooted_connected",
                smiles="CCO",
                rooted_at_atom=None,
                isomeric_smiles=False,
            ),
            DecoderAuditCase(
                name="disconnected_rooted",
                smiles="[Na+].CC",
                rooted_at_atom=0,
                isomeric_smiles=False,
            ),
            DecoderAuditCase(
                name="disconnected_unrooted",
                smiles="O.CCO",
                rooted_at_atom=None,
                isomeric_smiles=True,
            ),
            DecoderAuditCase(
                name="duplicate_same_text_connected",
                smiles="C1CCC2=NN=NN2CC1",
                rooted_at_atom=2,
                isomeric_smiles=False,
            ),
        )

        for case in cases:
            mol = parse_smiles(case.smiles)
            kwargs = dict(
                isomericSmiles=case.isomeric_smiles,
                kekuleSmiles=case.kekule_smiles,
                canonical=False,
                allBondsExplicit=case.all_bonds_explicit,
                allHsExplicit=case.all_hs_explicit,
                doRandom=True,
                ignoreAtomMapNumbers=case.ignore_atom_map_numbers,
            )
            if case.rooted_at_atom is not None:
                kwargs["rootedAtAtom"] = case.rooted_at_atom
            outputs = frozenset(grimace.MolToSmilesEnum(mol, **kwargs))
            decoder = grimace.MolToSmilesDecoder(mol, **kwargs)
            memo: dict[object, frozenset[str]] = {}
            seen_state_keys: set[object] = set()
            stack = [decoder._impl._state]
            audited_state_count = 0

            with self.subTest(case=case.name, smiles=case.smiles):
                while stack:
                    state = stack.pop()
                    state_key = _runtime._state_cache_key(state)
                    if state_key in seen_state_keys:
                        continue
                    seen_state_keys.add(state_key)
                    audited_state_count += 1

                    reachable = _runtime._reachable_terminal_prefixes(state, memo=memo)
                    prefix = state.prefix()
                    choices = state.choices()

                    self.assertTrue(reachable)
                    self.assertTrue(reachable <= outputs)
                    self.assertTrue(all(output.startswith(prefix) for output in reachable))

                    if state.is_terminal():
                        self.assertEqual((), choices)
                        self.assertEqual(frozenset({prefix}), reachable)
                        continue

                    self.assertTrue(choices)
                    union_of_branch_outputs: set[str] = set()
                    for choice in choices:
                        branch_outputs = _runtime._reachable_terminal_prefixes(
                            choice.next_state,
                            memo=memo,
                        )
                        self.assertTrue(branch_outputs)
                        self.assertTrue(branch_outputs <= reachable)
                        self.assertTrue(
                            all(
                                output.startswith(choice.next_state.prefix())
                                for output in branch_outputs
                            )
                        )
                        union_of_branch_outputs.update(branch_outputs)
                        stack.append(choice.next_state)

                    self.assertEqual(reachable, frozenset(union_of_branch_outputs))

                self.assertGreater(audited_state_count, 0)

    def test_decoder_disconnected_sampled_paths_stay_within_public_enum_outputs(self) -> None:
        mol = parse_smiles("[Na+].C#N")

        for isomeric_smiles in (False, True):
            for root_idx in range(mol.GetNumAtoms()):
                outputs = set(
                    grimace.MolToSmilesEnum(
                        mol,
                        rootedAtAtom=root_idx,
                        isomericSmiles=isomeric_smiles,
                        canonical=False,
                        doRandom=True,
                    )
                )
                for seed in range(3):
                    with self.subTest(
                        isomeric_smiles=isomeric_smiles,
                        root_idx=root_idx,
                        seed=seed,
                    ):
                        rng = random.Random(seed)
                        decoder = grimace.MolToSmilesDecoder(
                            mol,
                            rootedAtAtom=root_idx,
                            isomericSmiles=isomeric_smiles,
                            canonical=False,
                            doRandom=True,
                        )
                        chosen_tokens: list[str] = []

                        while not decoder.is_terminal:
                            choices = decoder.next_choices
                            self.assertTrue(choices)
                            chosen_choice = rng.choice(choices)
                            chosen_tokens.append(chosen_choice.text)
                            decoder = chosen_choice.next_state

                        self.assertEqual(decoder.prefix, "".join(chosen_tokens))
                        self.assertIn(decoder.prefix, outputs)

    def test_decoder_empty_molecule_is_terminal_with_empty_prefix(self) -> None:
        decoder = grimace.MolToSmilesDecoder(
            parse_smiles(""),
            rootedAtAtom=0,
            isomericSmiles=False,
            canonical=False,
            doRandom=True,
        )

        self.assertEqual("", decoder.prefix)
        self.assertTrue(decoder.is_terminal)
        self.assertEqual((), decoder.next_choices)
        self.assertEqual(
            {""},
            set(
                grimace.MolToSmilesEnum(
                    parse_smiles(""),
                    rootedAtAtom=0,
                    isomericSmiles=False,
                    canonical=False,
                    doRandom=True,
                )
            ),
        )

    def test_decoder_reports_branching_choices_for_aspirin_prefix(self) -> None:
        case = DecoderCase(
            name="aspirin",
            smiles="CC(=O)Oc1ccccc1C(=O)O",
            rooted_at_atom=0,
            isomeric_smiles=False,
        )
        atom_tokens = self._atom_tokens(case)
        decoder = self._make_decoder(case)

        target_prefix = "CC(=O)Oc1c"
        while decoder.prefix != target_prefix:
            choices = decoder.next_choices
            self.assertTrue(choices)
            decoder = choices[0].next_state

        assert_prefix_options_match_outputs(
            self,
            decoder.prefix,
            self._unique_choice_texts(decoder),
            self._reachable_outputs_from_decoder(decoder),
            atom_tokens=atom_tokens,
        )

    def test_public_enum_matches_internal_walker_enumeration(self) -> None:
        cases = (
            DecoderCase(
                name="walker_nonstereo",
                smiles="CCO",
                rooted_at_atom=0,
                isomeric_smiles=False,
            ),
            DecoderCase(
                name="walker_stereo",
                smiles="F[C@H](Cl)Br",
                rooted_at_atom=0,
                isomeric_smiles=True,
            ),
        )

        for case in cases:
            with self.subTest(case=case.name, smiles=case.smiles):
                mol = parse_smiles(case.smiles)
                enum_outputs = self._enumerate_outputs(case)
                if case.isomeric_smiles:
                    walker = _runtime.make_stereo_walker(mol, case.rooted_at_atom)
                else:
                    walker = _runtime.make_nonstereo_walker(mol, case.rooted_at_atom)
                self.assertEqual(enum_outputs, set(walker.enumerate_support()))


if __name__ == "__main__":
    unittest.main()
