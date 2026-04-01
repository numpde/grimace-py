from __future__ import annotations

from dataclasses import dataclass
import random
import unittest

import grimace
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
        prepared = _runtime.prepare_smiles_graph(
            parse_smiles(case.smiles),
            flags=_runtime.MolToSmilesFlags(
                isomeric_smiles=case.isomeric_smiles,
                kekule_smiles=case.kekule_smiles,
                rooted_at_atom=case.rooted_at_atom,
                canonical=False,
                all_bonds_explicit=case.all_bonds_explicit,
                all_hs_explicit=case.all_hs_explicit,
                do_random=True,
                ignore_atom_map_numbers=case.ignore_atom_map_numbers,
            ),
        )
        return tuple(prepared.atom_tokens)

    @staticmethod
    def _choice_texts(decoder: grimace.MolToSmilesDecoder) -> tuple[str, ...]:
        return tuple(choice.text for choice in decoder.next_choices)

    @staticmethod
    def _unique_choice_texts(decoder: grimace.MolToSmilesDecoder) -> tuple[str, ...]:
        return tuple(sorted({choice.text for choice in decoder.next_choices}))

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
                        self.assertTrue(choices)
                        assert_prefix_options_match_outputs(
                            self,
                            decoder.prefix,
                            tuple(sorted(set(options))),
                            outputs,
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
            if len(options) > 1:
                break
            decoder = options[0].next_state

        self.assertGreater(len(decoder.next_choices), 1)
        original_prefix = decoder.prefix
        left = decoder.copy()
        right = decoder.copy()
        distinct_pairs = [
            (left_choice, right_choice)
            for idx, left_choice in enumerate(decoder.next_choices)
            for right_choice in decoder.next_choices[idx + 1 :]
            if left_choice.next_state.prefix != right_choice.next_state.prefix
        ]
        self.assertTrue(distinct_pairs)
        left_choice, right_choice = distinct_pairs[0]

        left = left_choice.next_state
        right = right_choice.next_state

        self.assertEqual(original_prefix, decoder.prefix)
        self.assertNotEqual(left.prefix, right.prefix)
        self.assertTrue(any(output.startswith(left.prefix) for output in outputs))
        self.assertTrue(any(output.startswith(right.prefix) for output in outputs))

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

        with self.assertRaisesRegex(NotImplementedError, "rootedAtAtom >= 0"):
            grimace.MolToSmilesDecoder(
                mol,
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
        outputs = self._enumerate_outputs(case)
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
            outputs,
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
