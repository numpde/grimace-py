from __future__ import annotations

import gc
import unittest
import weakref
from collections.abc import Callable
from unittest.mock import patch

import grimace
import grimace._core as _core
import grimace._runtime_states as _runtime_states
from tests.helpers.mols import parse_smiles
from tests.helpers.public_runtime import (
    choice_texts,
    reachable_terminal_prefixes,
    supported_public_kwargs,
)


STEREO_SMILES = "F[C@H](Cl)Br"
DISCONNECTED_STEREO_SMILES = f"{STEREO_SMILES}.O"
STEREO_KWARGS = supported_public_kwargs(isomericSmiles=True, rootedAtAtom=-1)
_FakeTransitions = tuple[tuple[str, Callable[[], object]], ...]


def _reject_rooted_stereo_decoder_construction(*_args: object, **_kwargs: object) -> None:
    raise AssertionError(
        "unrooted stereo decoder construction must not eagerly instantiate "
        "rooted stereo decoder states"
    )


def _reject_state_realization(
    *_args: object,
    **_kwargs: object,
) -> None:
    raise AssertionError("choices must not eagerly realize sibling successor states")


def _choice_transition_texts(state: object) -> tuple[str, ...]:
    return tuple(
        text
        for text, _ in _runtime_states._realize_state_transitions(
            state._choice_state_transitions()
        )
    )


def _grouped_transition_texts(state: object) -> tuple[str, ...]:
    return tuple(
        text
        for text, _ in _runtime_states._realize_state_transitions(
            state._grouped_state_transitions()
        )
    )


class _FakeState:
    def __init__(
        self,
        prefix: str,
        *,
        terminal: bool,
        transitions: _FakeTransitions = (),
        reject_terminal_transitions: bool = False,
    ) -> None:
        self._prefix = prefix
        self._terminal = terminal
        self._transitions = transitions
        self._reject_terminal_transitions = reject_terminal_transitions

    def prefix(self) -> str:
        return self._prefix

    def is_terminal(self) -> bool:
        return self._terminal

    def copy(self) -> "_FakeState":
        return self

    def cache_key(self) -> tuple[str, str, bool]:
        return ("fake", self._prefix, self._terminal)

    def _choice_state_transitions(self) -> _FakeTransitions:
        if self._terminal and self._reject_terminal_transitions:
            raise AssertionError("terminal child transitions must not be queried")
        return self._transitions

    def _grouped_state_transitions(self) -> _FakeTransitions:
        return self._choice_state_transitions()


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

    def test_state_cache_key_rejects_non_tuple_keys(self) -> None:
        class State:
            def cache_key(self) -> str:
                return "bad"

        with self.assertRaisesRegex(TypeError, "cache_key"):
            _runtime_states._state_cache_key(State())

    def test_state_cache_key_rejects_unhashable_keys(self) -> None:
        class State:
            def cache_key(self) -> tuple[str, list[str]]:
                return ("bad", [])

        with self.assertRaisesRegex(TypeError, "hashable"):
            _runtime_states._state_cache_key(State())

    def test_lazy_all_roots_transitions_do_not_retain_root_decoders(self) -> None:
        created_refs: list[weakref.ReferenceType[object]] = []
        advanced_decoders: list[tuple[str, int]] = []

        def live_decoders() -> tuple[object, ...]:
            gc.collect()
            return tuple(
                decoder
                for ref in created_refs
                if (decoder := ref()) is not None
            )

        class Decoder:
            def __init__(self, _prepared: object, _root_idx: int) -> None:
                self.ordinal = len(created_refs)
                created_refs.append(weakref.ref(self))

            def next_choice_texts(self) -> tuple[str, ...]:
                return ("C",)

            def next_token_support(self) -> tuple[str, ...]:
                return ("C",)

            def copy(self) -> "Decoder":
                return self

            def advance_choice(self, _chosen_idx: int) -> None:
                advanced_decoders.append(("choice", self.ordinal))

            def advance_token(self, _chosen_token: str) -> None:
                advanced_decoders.append(("token", self.ordinal))

        state = _runtime_states._LazyAllRootsConnectedStereoState(
            object(),
            atom_count=2,
        )

        with patch.object(_core, "RootedConnectedStereoDecoder", new=Decoder):
            choice_transitions = state._choice_state_transitions()
            self.assertEqual((), live_decoders())

            created_before_choice_advance = len(created_refs)
            self.assertEqual(2, created_before_choice_advance)

            choice_transitions[0][1]()
            self.assertEqual(
                [("choice", created_before_choice_advance)],
                advanced_decoders,
            )

            created_before_grouped_listing = len(created_refs)
            grouped_transitions = state._grouped_state_transitions()
            self.assertEqual((), live_decoders())

            created_before_grouped_advance = len(created_refs)
            self.assertEqual(
                2,
                created_before_grouped_advance - created_before_grouped_listing,
            )

            grouped_transitions[0][1]()
            self.assertEqual(
                [
                    ("choice", created_before_choice_advance),
                    ("token", created_before_grouped_advance),
                    ("token", created_before_grouped_advance + 1),
                ],
                advanced_decoders,
            )

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
            grimace.MolToSmilesDecoder,
            grimace.MolToSmilesDeterminizedDecoder,
        )

        for decoder_cls in decoder_cases:
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
                        _runtime_states,
                        "_realize_state_transitions",
                        side_effect=_reject_state_realization,
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
        original_advance_token_state = _runtime_states._advance_token_state

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
                    _runtime_states,
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
            _runtime_states,
            "_realize_state_transitions",
            side_effect=_reject_state_realization,
        ):
            choices = merged.next_choices
            self.assertEqual(("C", "("), tuple(choice.text for choice in choices))
            advanced = choices[0].next_state

        self.assertEqual("CC", advanced.prefix)

    def test_core_decoder_choices_do_not_eagerly_realize_successors(self) -> None:
        decoder_cases = (
            (
                grimace.MolToSmilesDecoder,
                "C",
            ),
            (
                grimace.MolToSmilesDeterminizedDecoder,
                "C",
            ),
        )

        for decoder_cls, expected_prefix in decoder_cases:
            with self.subTest(decoder_cls=decoder_cls.__name__):
                decoder = decoder_cls(
                    parse_smiles("CCO"),
                    **supported_public_kwargs(isomericSmiles=False, rootedAtAtom=0),
                )

                with patch.object(
                    _runtime_states,
                    "_realize_state_transitions",
                    side_effect=_reject_state_realization,
                ):
                    choices = decoder.next_choices
                    self.assertEqual(("C",), tuple(choice.text for choice in choices))
                    advanced = choices[0].next_state

                self.assertEqual(expected_prefix, advanced.prefix)

    def test_merged_state_does_not_query_terminal_child_transitions(self) -> None:
        terminal = _FakeState(
            "C",
            terminal=True,
            reject_terminal_transitions=True,
        )
        nonterminal = _FakeState(
            "C",
            terminal=False,
            transitions=(("C", lambda: nonterminal),),
        )
        merged = _runtime_states._MergedStateAdapter(
            (terminal, nonterminal)
        )

        self.assertTrue(merged.is_terminal())
        self.assertEqual(("C",), _choice_transition_texts(merged))
        self.assertEqual(("C",), _grouped_transition_texts(merged))

    def test_merged_state_cache_keys_are_order_insensitive(self) -> None:
        first = _FakeState("C", terminal=False)
        second = _FakeState("N", terminal=False)

        self.assertEqual(
            _runtime_states._state_cache_key(
                _runtime_states._MergedStateAdapter((first, second))
            ),
            _runtime_states._state_cache_key(
                _runtime_states._MergedStateAdapter((second, first))
            ),
        )
        self.assertNotEqual(
            _runtime_states._state_cache_key(
                _runtime_states._MergedStateAdapter((first, second))
            ),
            _runtime_states._state_cache_key(
                _runtime_states._MergedStateAdapter((first, first))
            ),
        )

    def test_reachable_outputs_include_accepting_state_continuations(self) -> None:
        terminal_child = _FakeState("CC", terminal=True)
        accepting_with_continuation = _FakeState(
            "C",
            terminal=True,
            transitions=(("C", lambda: terminal_child),),
        )

        self.assertEqual(
            frozenset({"C", "CC"}),
            reachable_terminal_prefixes(accepting_with_continuation),
        )

    def test_disconnected_accepting_fragment_keeps_continuations_and_separator(
        self,
    ) -> None:
        first_fragment = _FakeState(
            "C",
            terminal=True,
            transitions=(("C", lambda: _FakeState("CC", terminal=True)),),
        )

        disconnected = _runtime_states._DisconnectedStateAdapter(
            (
                first_fragment,
                _FakeState("O", terminal=True),
            )
        )

        self.assertFalse(disconnected.is_terminal())
        self.assertEqual(("C", "."), _choice_transition_texts(disconnected))
        self.assertEqual(("C", "."), _grouped_transition_texts(disconnected))
        last_fragment = _runtime_states._DisconnectedStateAdapter(
            (first_fragment,)
        )

        self.assertTrue(last_fragment.is_terminal())
        self.assertEqual(("C",), _choice_transition_texts(last_fragment))
        self.assertEqual(("C",), _grouped_transition_texts(last_fragment))


if __name__ == "__main__":
    unittest.main()
