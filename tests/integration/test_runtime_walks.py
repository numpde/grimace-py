from __future__ import annotations

import unittest

import grimace
from grimace._runtime_states import _StateTransition
from grimace._runtime_walks import _walk_token_transitions
from tests.helpers.mols import parse_smiles
from tests.helpers.public_runtime import public_enum_support, supported_public_kwargs


class _FakeState:
    def __init__(
        self,
        prefix: str,
        *,
        terminal: bool,
        transitions: tuple[_StateTransition, ...] = (),
    ) -> None:
        self._prefix = prefix
        self._terminal = terminal
        self._transitions = transitions

    def prefix(self) -> str:
        return self._prefix

    def is_terminal(self) -> bool:
        return self._terminal

    def _token_state_transitions(self) -> tuple[_StateTransition, ...]:
        return self._transitions


def _fake_transition(
    text: str,
    next_state: _FakeState,
    *,
    branch_count: int = 1,
) -> _StateTransition:
    return _StateTransition(text, branch_count, lambda: next_state)


class RuntimeWalkTests(unittest.TestCase):
    def test_token_walk_records_flat_choice_payload(self) -> None:
        kwargs = supported_public_kwargs(isomericSmiles=False, rootedAtAtom=-1)
        mol = parse_smiles("CCO")
        decoder = grimace.MolToSmilesDeterminizedDecoder(mol, **kwargs)

        result = _walk_token_transitions(decoder._state, lambda _transitions: 0)

        self.assertIn("".join(result.tokens), public_enum_support(mol, **kwargs))
        self.assertEqual(("C", "(", "C", ")", "O"), result.tokens)
        self.assertEqual((2, 2, 2, 1, 1), result.choice_counts)
        self.assertEqual(
            ("C", "O", "(", "C", "C", "O", ")", "O"),
            result.choice_tokens,
        )
        self.assertEqual((2, 1, 2, 1, 1, 1, 1, 1), result.choice_branch_counts)

    def test_token_walk_records_disconnected_separator(self) -> None:
        kwargs = supported_public_kwargs(isomericSmiles=True, rootedAtAtom=-1)
        mol = parse_smiles("O.CCO")
        decoder = grimace.MolToSmilesDeterminizedDecoder(mol, **kwargs)

        result = _walk_token_transitions(decoder._state, lambda _transitions: 0)

        self.assertIn("".join(result.tokens), public_enum_support(mol, **kwargs))
        self.assertIn(".", result.tokens)
        separator_idx = result.tokens.index(".")
        choice_start = sum(result.choice_counts[:separator_idx])
        self.assertEqual(
            (".",),
            result.choice_tokens[choice_start:choice_start + 1],
        )
        self.assertEqual(
            (1,),
            result.choice_branch_counts[choice_start:choice_start + 1],
        )

    def test_token_walk_stops_at_initial_accepted_state(self) -> None:
        terminal_successor = _FakeState("CC", terminal=True)
        accepted_with_continuation = _FakeState(
            "C",
            terminal=True,
            transitions=(
                _fake_transition("C", terminal_successor),
            ),
        )

        result = _walk_token_transitions(
            accepted_with_continuation,
            lambda _transitions: 0,
        )

        self.assertEqual((), result.tokens)
        self.assertEqual((), result.choice_counts)
        self.assertEqual((), result.choice_tokens)
        self.assertEqual((), result.choice_branch_counts)

    def test_token_walk_rejects_nonterminal_state_without_transitions(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no token transitions"):
            _walk_token_transitions(
                _FakeState("", terminal=False),
                lambda _transitions: 0,
            )

    def test_token_walk_rejects_invalid_selected_index(self) -> None:
        terminal = _FakeState("C", terminal=True)
        initial = _FakeState(
            "",
            terminal=False,
            transitions=(
                _fake_transition("C", terminal),
            ),
        )

        with self.assertRaisesRegex(IndexError, "outside"):
            _walk_token_transitions(initial, lambda _transitions: 1)

    def test_token_walk_rejects_noninteger_selected_index(self) -> None:
        terminal = _FakeState("C", terminal=True)
        initial = _FakeState(
            "",
            terminal=False,
            transitions=(
                _fake_transition("C", terminal),
            ),
        )

        with self.assertRaisesRegex(TypeError, "int"):
            _walk_token_transitions(initial, lambda _transitions: True)


if __name__ == "__main__":
    unittest.main()
