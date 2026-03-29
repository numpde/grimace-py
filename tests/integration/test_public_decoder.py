from __future__ import annotations

from dataclasses import dataclass
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

    def test_decoder_sampled_paths_stay_within_public_enum_outputs(self) -> None:
        for case in self.CASES:
            with self.subTest(case=case.name, smiles=case.smiles):
                outputs = self._enumerate_outputs(case)
                decoder = self._make_decoder(case)

                while not decoder.isTerminal():
                    options = decoder.nextTokens()
                    self.assertTrue(options)
                    assert_prefix_options_match_outputs(
                        self,
                        decoder.prefix(),
                        options,
                        outputs,
                    )
                    decoder.advance(options[0])

                self.assertEqual((), decoder.nextTokens())
                self.assertIn(decoder.prefix(), outputs)

    def test_decoder_copy_forks_state_without_mutating_original(self) -> None:
        case = DecoderCase(
            name="branching_stereo",
            smiles="F[C@H](Cl)Br",
            rooted_at_atom=0,
            isomeric_smiles=True,
        )
        outputs = self._enumerate_outputs(case)
        decoder = self._make_decoder(case)

        while not decoder.isTerminal():
            options = decoder.nextTokens()
            if len(options) > 1:
                break
            decoder.advance(options[0])

        self.assertGreater(len(decoder.nextTokens()), 1)
        original_prefix = decoder.prefix()
        left = decoder.copy()
        right = decoder.copy()
        left_token, right_token = decoder.nextTokens()[:2]

        left.advance(left_token)
        right.advance(right_token)

        self.assertEqual(original_prefix, decoder.prefix())
        self.assertNotEqual(left.prefix(), right.prefix())
        self.assertTrue(any(output.startswith(left.prefix()) for output in outputs))
        self.assertTrue(any(output.startswith(right.prefix()) for output in outputs))

    def test_decoder_rejects_invalid_token_with_available_choices(self) -> None:
        decoder = grimace.MolToSmilesDecoder(
            parse_smiles("CCO"),
            rootedAtAtom=0,
            isomericSmiles=False,
            canonical=False,
            doRandom=True,
        )

        with self.assertRaisesRegex(KeyError, "choices"):
            decoder.advance("(")

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

    def test_decoder_rejects_disconnected_molecules(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "singly-connected"):
            grimace.MolToSmilesDecoder(
                parse_smiles("CC.O"),
                rootedAtAtom=0,
                isomericSmiles=False,
                canonical=False,
                doRandom=True,
            )

    def test_decoder_empty_molecule_is_terminal_with_empty_prefix(self) -> None:
        decoder = grimace.MolToSmilesDecoder(
            parse_smiles(""),
            rootedAtAtom=0,
            isomericSmiles=False,
            canonical=False,
            doRandom=True,
        )

        self.assertEqual("", decoder.prefix())
        self.assertTrue(decoder.isTerminal())
        self.assertEqual((), decoder.nextTokens())
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
