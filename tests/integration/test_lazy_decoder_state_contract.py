from __future__ import annotations

from unittest.mock import patch
import unittest

import grimace
import grimace._core as _core
from tests.helpers.mols import parse_smiles
from tests.helpers.public_runtime import choice_texts, supported_public_kwargs


STEREO_SMILES = "F[C@H](Cl)Br"
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

    def _stereo_inputs(self) -> tuple[tuple[str, object], ...]:
        mol = parse_smiles(STEREO_SMILES)
        prepared = grimace.PreparedMol.from_bytes(
            grimace.PrepareMol(mol, isomericSmiles=True).to_bytes()
        )
        return (
            ("rdkit_mol", mol),
            ("prepared_mol_bytes_roundtrip", prepared),
        )

    def test_unrooted_stereo_decoder_init_does_not_instantiate_rooted_decoders(
        self,
    ) -> None:
        decoder_classes = (
            grimace.MolToSmilesDecoder,
            grimace.MolToSmilesDeterminizedDecoder,
        )
        inputs = self._stereo_inputs()

        for decoder_cls in decoder_classes:
            for input_name, mol_or_prepared in inputs:
                with self.subTest(
                    decoder_cls=decoder_cls.__name__,
                    input_name=input_name,
                ):
                    with patch.object(
                        _core,
                        "RootedConnectedStereoDecoder",
                        new=_reject_rooted_stereo_decoder_construction,
                    ):
                        decoder = decoder_cls(mol_or_prepared, **STEREO_KWARGS)

                    self.assertEqual("", decoder.prefix)

    def test_choices_advance_selected_branch_without_eager_successors(
        self,
    ) -> None:
        decoder_cases = (
            (
                grimace.MolToSmilesDecoder,
                "grimace._runtime_states._choice_successor_states",
            ),
            (
                grimace.MolToSmilesDeterminizedDecoder,
                "grimace._runtime_states._grouped_successor_states",
            ),
        )

        for decoder_cls, patched_name in decoder_cases:
            for input_name, mol_or_prepared in self._stereo_inputs():
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
                    with patch(
                        patched_name,
                        side_effect=_reject_successor_enumeration,
                    ):
                        choices = decoder.next_choices
                        observed_texts = tuple(choice.text for choice in choices)
                        self.assertEqual(expected_texts, observed_texts)
                        advanced = choices[selected_idx].next_state

                    self.assertIsInstance(advanced, decoder_cls)
                    self.assertEqual(expected_next.prefix, advanced.prefix)
                    self.assertEqual(choice_texts(expected_next), choice_texts(advanced))


if __name__ == "__main__":
    unittest.main()
