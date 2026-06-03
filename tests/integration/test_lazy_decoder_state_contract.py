from __future__ import annotations

from unittest.mock import patch
import unittest

import grimace
import grimace._core as _core
import grimace._runtime_states as _runtime_states
from tests.helpers.mols import parse_smiles
from tests.helpers.public_runtime import choice_texts, supported_public_kwargs


STEREO_SMILES = "F[C@H](Cl)Br"
DISCONNECTED_STEREO_SMILES = f"{STEREO_SMILES}.O"
STEREO_KWARGS = supported_public_kwargs(isomericSmiles=True, rootedAtAtom=-1)


def _reject_rooted_stereo_decoder_construction(*_args: object, **_kwargs: object) -> None:
    raise AssertionError(
        "unrooted stereo decoder construction must not eagerly instantiate "
        "rooted stereo decoder states"
    )


def _reject_successor_enumeration(
    *_args: object,
    **_kwargs: object,
) -> None:
    raise AssertionError("choices must not eagerly enumerate sibling successor states")


class LazyDecoderStateContractTests(unittest.TestCase):
    """Lazy all-roots decoder-state contract."""

    def _inputs_for_smiles(self, smiles: str) -> tuple[tuple[str, object], ...]:
        mol = parse_smiles(smiles)
        prepared = grimace.PreparedMol.from_bytes(
            grimace.PrepareMol(mol, isomericSmiles=True).to_bytes()
        )
        return (
            ("rdkit_mol", mol),
            ("prepared_mol_bytes_roundtrip", prepared),
        )

    def _connected_stereo_inputs(self) -> tuple[tuple[str, object], ...]:
        return self._inputs_for_smiles(STEREO_SMILES)

    def test_unrooted_stereo_decoder_init_does_not_instantiate_rooted_decoders(
        self,
    ) -> None:
        decoder_classes = (
            grimace.MolToSmilesDecoder,
            grimace.MolToSmilesDeterminizedDecoder,
        )
        input_cases = (
            ("connected", STEREO_SMILES),
            ("disconnected", DISCONNECTED_STEREO_SMILES),
        )

        for decoder_cls in decoder_classes:
            for case_name, smiles in input_cases:
                for input_name, mol_or_prepared in self._inputs_for_smiles(smiles):
                    with self.subTest(
                        decoder_cls=decoder_cls.__name__,
                        case_name=case_name,
                        input_name=input_name,
                    ):
                        with patch.object(
                            _core,
                            "RootedConnectedStereoDecoder",
                            new=_reject_rooted_stereo_decoder_construction,
                        ):
                            decoder = decoder_cls(mol_or_prepared, **STEREO_KWARGS)
                            self.assertFalse(decoder.is_terminal)
                            copied = decoder.copy()
                            self.assertIsInstance(copied, decoder_cls)
                            self.assertFalse(copied.is_terminal)

                        self.assertEqual("", decoder.prefix)
                        self.assertEqual("", copied.prefix)

    def test_choices_advance_selected_branch_without_eager_successors(
        self,
    ) -> None:
        decoder_cases = (
            (
                grimace.MolToSmilesDecoder,
                "choice_successor_states",
            ),
            (
                grimace.MolToSmilesDeterminizedDecoder,
                "grouped_successor_states",
            ),
        )

        for decoder_cls, patched_method in decoder_cases:
            for input_name, mol_or_prepared in self._connected_stereo_inputs():
                with self.subTest(
                    decoder_cls=decoder_cls.__name__,
                    input_name=input_name,
                ):
                    expected = decoder_cls(mol_or_prepared, **STEREO_KWARGS)
                    expected_texts = choice_texts(expected)
                    self.assertGreater(len(expected_texts), 1)
                    selected_idx = len(expected_texts) - 1
                    expected_next = expected.next_choices[selected_idx].next_state

                    decoder = decoder_cls(mol_or_prepared, **STEREO_KWARGS)
                    with patch.object(
                        _runtime_states._LazyAllRootsConnectedStereoState,
                        patched_method,
                        side_effect=_reject_successor_enumeration,
                    ):
                        choices = decoder.next_choices
                        observed_texts = tuple(choice.text for choice in choices)
                        self.assertEqual(expected_texts, observed_texts)
                        advanced = choices[selected_idx].next_state
                        self.assertIs(advanced, choices[selected_idx].next_state)

                    self.assertIsInstance(advanced, decoder_cls)
                    self.assertEqual(expected_next.prefix, advanced.prefix)
                    self.assertEqual(choice_texts(expected_next), choice_texts(advanced))

    def test_sequence_deviation_advances_only_observed_token(self) -> None:
        original_advance_token_state = (
            _runtime_states._LazyAllRootsConnectedStereoState._advance_token_state
        )

        for input_name, mol_or_prepared in self._connected_stereo_inputs():
            with self.subTest(input_name=input_name):
                decoder = grimace.MolToSmilesDeterminizedDecoder(
                    mol_or_prepared,
                    **STEREO_KWARGS,
                )
                initial_tokens = choice_texts(decoder)
                self.assertGreater(len(initial_tokens), 1)
                selected_token = initial_tokens[0]

                def guarded_advance_token_state(
                    decoder: object,
                    chosen_token: str,
                ) -> object:
                    if chosen_token != selected_token:
                        raise AssertionError(
                            "sequence deviation must not advance unobserved tokens"
                        )
                    return original_advance_token_state(decoder, chosen_token)

                with patch.object(
                    _runtime_states._LazyAllRootsConnectedStereoState,
                    "_advance_token_state",
                    side_effect=guarded_advance_token_state,
                ):
                    deviation = grimace.MolToSmilesDeviation(
                        mol_or_prepared,
                        (selected_token,),
                        **STEREO_KWARGS,
                    )

                self.assertIsInstance(deviation, grimace.SmilesDeviation)
                self.assertEqual("incomplete", deviation.reason)
                self.assertEqual(selected_token, deviation.accepted_text)

    def test_merged_determinized_choices_do_not_eagerly_realize_successors(self) -> None:
        decoder = grimace.MolToSmilesDeterminizedDecoder(
            parse_smiles("CCO"),
            **supported_public_kwargs(isomericSmiles=True, rootedAtAtom=-1),
        )
        merged = next(
            choice.next_state
            for choice in decoder.next_choices
            if choice.text == "C"
        )
        self.assertEqual("C", merged.prefix)

        with patch.object(
            _runtime_states._MergedStateAdapter,
            "grouped_successor_states",
            side_effect=_reject_successor_enumeration,
        ):
            choices = merged.next_choices
            self.assertEqual(("C", "("), tuple(choice.text for choice in choices))
            advanced = choices[0].next_state

        self.assertEqual("CC", advanced.prefix)


if __name__ == "__main__":
    unittest.main()
