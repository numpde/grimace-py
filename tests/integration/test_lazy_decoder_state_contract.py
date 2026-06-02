from __future__ import annotations

from unittest.mock import patch
import unittest

import grimace
import grimace._runtime as _runtime
from tests.helpers.mols import parse_smiles
from tests.helpers.public_runtime import choice_texts, supported_public_kwargs


STEREO_SMILES = "F[C@H](Cl)Br"
STEREO_KWARGS = supported_public_kwargs(isomericSmiles=True)


def _reject_rooted_stereo_decoder_construction(*_args: object, **_kwargs: object) -> None:
    raise AssertionError(
        "unrooted stereo decoder construction must not eagerly instantiate "
        "rooted stereo decoder states"
    )


class LazyDecoderStateContractTests(unittest.TestCase):
    """Future lazy all-roots decoder-state contract."""

    def _stereo_inputs(self) -> tuple[tuple[str, object], ...]:
        mol = parse_smiles(STEREO_SMILES)
        prepared = grimace.PreparedMol.from_bytes(
            grimace.PrepareMol(mol, isomericSmiles=True).to_bytes()
        )
        return (
            ("rdkit_mol", mol),
            ("prepared_mol", prepared),
        )

    def test_unrooted_stereo_decoder_init_does_not_instantiate_rooted_decoders(
        self,
    ) -> None:
        decoder_classes = (
            grimace.MolToSmilesDecoder,
            grimace.MolToSmilesDeterminizedDecoder,
        )

        for decoder_cls in decoder_classes:
            for input_name, mol_or_prepared in self._stereo_inputs():
                with self.subTest(
                    decoder_cls=decoder_cls.__name__,
                    input_name=input_name,
                ):
                    with patch.object(
                        _runtime._core,
                        "RootedConnectedStereoDecoder",
                        new=_reject_rooted_stereo_decoder_construction,
                    ):
                        decoder = decoder_cls(mol_or_prepared, **STEREO_KWARGS)

                    self.assertEqual("", decoder.prefix)

    def test_determinized_decoder_can_advance_seen_token_without_public_choice_states(
        self,
    ) -> None:
        mol = parse_smiles(STEREO_SMILES)
        expected = grimace.MolToSmilesDeterminizedDecoder(mol, **STEREO_KWARGS)
        expected_texts = choice_texts(expected)
        self.assertTrue(expected_texts)

        decoder = grimace.MolToSmilesDeterminizedDecoder(mol, **STEREO_KWARGS)
        with patch(
            "grimace._runtime._public_decoder_choices",
            side_effect=AssertionError(
                "lightweight token advance must not materialize public sibling choices"
            ),
        ):
            observed_texts = decoder._choice_texts()
            self.assertEqual(expected_texts, observed_texts)
            selected_token = observed_texts[0]
            advanced = decoder._advance_token(selected_token)

        expected_next = {
            choice.text: choice.next_state for choice in expected.next_choices
        }[selected_token]
        self.assertIsNone(decoder._choices_cache)
        self.assertIsInstance(advanced, grimace.MolToSmilesDeterminizedDecoder)
        self.assertEqual(expected_next.prefix, advanced.prefix)
        self.assertEqual(choice_texts(expected_next), choice_texts(advanced))

    def test_determinized_decoder_advance_rejects_illegal_token(self) -> None:
        decoder = grimace.MolToSmilesDeterminizedDecoder(
            parse_smiles(STEREO_SMILES),
            **STEREO_KWARGS,
        )

        self.assertNotIn("Z", choice_texts(decoder))
        with self.assertRaisesRegex(ValueError, "not a legal next token"):
            decoder._advance_token("Z")


if __name__ == "__main__":
    unittest.main()
