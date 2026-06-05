from __future__ import annotations

import unittest
from unittest import mock

import grimace
import grimace._runtime_walks as _runtime_walks
from grimace._runtime_states import _StateTransition
from grimace._runtime_walks import (
    _TokenWalkResult,
    _branch_multiplicity_chooser,
    _seeded_branch_multiplicity_chooser,
    _seeded_uniform_transition_chooser,
    _uniform_transition_chooser,
    _walk_branch_transitions,
    _walk_token_transitions,
)
from tests.helpers.mols import parse_smiles
from tests.helpers.public_runtime import public_enum_support, supported_public_kwargs


def _assert_token_walk_result_invariants(
    test_case: unittest.TestCase,
    result: _TokenWalkResult,
) -> None:
    step_payloads = tuple(result.step_payloads())
    test_case.assertEqual(len(result.tokens), len(step_payloads))

    for token, selected_idx, choices, branch_counts in step_payloads:
        test_case.assertEqual(len(choices), len(branch_counts))
        test_case.assertGreaterEqual(selected_idx, 0)
        test_case.assertLess(selected_idx, len(choices))
        test_case.assertIn(token, choices)
        test_case.assertEqual(token, choices[selected_idx])
        test_case.assertEqual(len(choices), len(set(choices)))
        test_case.assertTrue(all(count > 0 for count in branch_counts))


class _FakeState:
    def __init__(
        self,
        *,
        terminal: bool,
        token_transitions: tuple[_StateTransition, ...] = (),
        branch_transitions: tuple[_StateTransition, ...] | None = None,
    ) -> None:
        self._terminal = terminal
        self._token_transitions = token_transitions
        self._branch_transitions = (
            token_transitions
            if branch_transitions is None
            else branch_transitions
        )

    def is_terminal(self) -> bool:
        return self._terminal

    def _token_state_transitions(self) -> tuple[_StateTransition, ...]:
        return self._token_transitions

    def _branch_state_transitions(self) -> tuple[_StateTransition, ...]:
        return self._branch_transitions


def _fake_transition(
    text: str,
    next_state: _FakeState,
    *,
    branch_count: int = 1,
) -> _StateTransition:
    return _StateTransition(text, branch_count, lambda: next_state)


def _token_walk_result(
    **overrides: object,
) -> _TokenWalkResult:
    values = {
        "tokens": ("C",),
        "selected_indices": (0,),
        "choice_counts": (1,),
        "choice_tokens": ("C",),
        "choice_branch_counts": (1,),
    }
    values.update(overrides)
    return _TokenWalkResult(**values)


class RuntimeWalkTests(unittest.TestCase):
    def test_token_walk_result_step_payloads_reject_invalid_flat_shape(self) -> None:
        cases = (
            ({"selected_indices": ()}, ValueError, "selected-index count"),
            ({"choice_counts": ()}, ValueError, "choice-count count"),
            ({"choice_branch_counts": ()}, ValueError, "branch-count lengths"),
            ({"choice_counts": (0,)}, ValueError, "choice counts must be positive"),
            ({"choice_counts": (2,)}, ValueError, "choice counts do not span"),
        )

        for overrides, expected_exception, expected_regex in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(expected_exception, expected_regex):
                    tuple(_token_walk_result(**overrides).step_payloads())

    def test_token_walk_records_flat_choice_payload(self) -> None:
        kwargs = supported_public_kwargs(isomericSmiles=False, rootedAtAtom=-1)
        mol = parse_smiles("CCO")
        decoder = grimace.MolToSmilesDeterminizedDecoder(mol, **kwargs)

        result = _walk_token_transitions(decoder._state, lambda _transitions: 0)

        _assert_token_walk_result_invariants(self, result)
        self.assertIn("".join(result.tokens), public_enum_support(mol, **kwargs))
        self.assertEqual(("C", "(", "C", ")", "O"), result.tokens)
        self.assertEqual((0, 0, 0, 0, 0), result.selected_indices)
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

        _assert_token_walk_result_invariants(self, result)
        self.assertIn("".join(result.tokens), public_enum_support(mol, **kwargs))
        self.assertIn(".", result.tokens)
        separator_idx = result.tokens.index(".")
        _, _, separator_choices, separator_branch_counts = tuple(
            result.step_payloads()
        )[separator_idx]
        self.assertEqual(
            (".",),
            separator_choices,
        )
        self.assertEqual(
            (1,),
            separator_branch_counts,
        )

    def test_uniform_transition_chooser_samples_by_exposed_choice_count(self) -> None:
        sampled_counts: list[int] = []

        def sample_index(choice_count: int) -> int:
            sampled_counts.append(choice_count)
            return 1 if choice_count > 1 else 0

        kwargs = supported_public_kwargs(isomericSmiles=False, rootedAtAtom=-1)
        mol = parse_smiles("CCO")
        decoder = grimace.MolToSmilesDeterminizedDecoder(mol, **kwargs)

        result = _walk_token_transitions(
            decoder._state,
            _uniform_transition_chooser(sample_index),
        )

        _assert_token_walk_result_invariants(self, result)
        self.assertIn("".join(result.tokens), public_enum_support(mol, **kwargs))
        self.assertEqual((2, 1), tuple(sampled_counts[:2]))
        self.assertEqual("O", result.tokens[0])

    def test_branch_multiplicity_chooser_samples_by_branch_counts(self) -> None:
        sampled_weights: list[tuple[int, ...]] = []

        def sample_weighted_index(weights: tuple[int, ...]) -> int:
            sampled_weights.append(weights)
            return max(range(len(weights)), key=weights.__getitem__)

        kwargs = supported_public_kwargs(isomericSmiles=False, rootedAtAtom=-1)
        mol = parse_smiles("CCO")
        decoder = grimace.MolToSmilesDeterminizedDecoder(mol, **kwargs)

        result = _walk_token_transitions(
            decoder._state,
            _branch_multiplicity_chooser(sample_weighted_index),
        )

        _assert_token_walk_result_invariants(self, result)
        self.assertIn("".join(result.tokens), public_enum_support(mol, **kwargs))
        self.assertEqual((2, 1), sampled_weights[0])
        self.assertEqual("C", result.tokens[0])

    def test_seeded_uniform_token_walk_is_reproducible(self) -> None:
        kwargs = supported_public_kwargs(isomericSmiles=False, rootedAtAtom=-1)
        mol = parse_smiles("CCO")

        first = _walk_token_transitions(
            grimace.MolToSmilesDeterminizedDecoder(mol, **kwargs)._state,
            _seeded_uniform_transition_chooser(123),
        )
        second = _walk_token_transitions(
            grimace.MolToSmilesDeterminizedDecoder(mol, **kwargs)._state,
            _seeded_uniform_transition_chooser(123),
        )

        self.assertEqual(first, second)
        _assert_token_walk_result_invariants(self, first)
        self.assertIn("".join(first.tokens), public_enum_support(mol, **kwargs))

    def test_seeded_branch_multiplicity_walk_is_reproducible(self) -> None:
        kwargs = supported_public_kwargs(isomericSmiles=True, rootedAtAtom=-1)
        mol = parse_smiles("F[C@H](Cl)Br")

        first = _walk_token_transitions(
            grimace.MolToSmilesDeterminizedDecoder(mol, **kwargs)._state,
            _seeded_branch_multiplicity_chooser(456),
        )
        second = _walk_token_transitions(
            grimace.MolToSmilesDeterminizedDecoder(mol, **kwargs)._state,
            _seeded_branch_multiplicity_chooser(456),
        )

        self.assertEqual(first, second)
        _assert_token_walk_result_invariants(self, first)
        self.assertIn("".join(first.tokens), public_enum_support(mol, **kwargs))

    def test_seeded_walk_rejects_invalid_seed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsigned 64-bit"):
            _seeded_uniform_transition_chooser(-1)
        with self.assertRaisesRegex(ValueError, "unsigned 64-bit"):
            _seeded_branch_multiplicity_chooser(True)

    def test_seeded_walk_validates_invalid_seed_before_core_sampler_lookup(self) -> None:
        with mock.patch.object(_runtime_walks, "_core", object()):
            with self.assertRaisesRegex(ValueError, "unsigned 64-bit"):
                _seeded_uniform_transition_chooser(-1)

    def test_token_walk_stops_at_initial_accepted_state(self) -> None:
        terminal_successor = _FakeState(terminal=True)
        accepted_with_continuation = _FakeState(
            terminal=True,
            token_transitions=(
                _fake_transition("C", terminal_successor),
            ),
        )

        result = _walk_token_transitions(
            accepted_with_continuation,
            lambda _transitions: 0,
        )

        _assert_token_walk_result_invariants(self, result)
        self.assertEqual((), result.tokens)
        self.assertEqual((), result.selected_indices)
        self.assertEqual((), result.choice_counts)
        self.assertEqual((), result.choice_tokens)
        self.assertEqual((), result.choice_branch_counts)

    def test_token_walk_rejects_nonterminal_state_without_transitions(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no token transitions"):
            _walk_token_transitions(
                _FakeState(terminal=False),
                lambda _transitions: 0,
            )

    def test_token_walk_rejects_invalid_selected_index(self) -> None:
        terminal = _FakeState(terminal=True)
        initial = _FakeState(
            terminal=False,
            token_transitions=(
                _fake_transition("C", terminal),
            ),
        )

        with self.assertRaisesRegex(IndexError, "outside"):
            _walk_token_transitions(initial, lambda _transitions: 1)

    def test_token_walk_rejects_noninteger_selected_index(self) -> None:
        terminal = _FakeState(terminal=True)
        initial = _FakeState(
            terminal=False,
            token_transitions=(
                _fake_transition("C", terminal),
            ),
        )

        with self.assertRaisesRegex(TypeError, "int"):
            _walk_token_transitions(initial, lambda _transitions: True)

    def test_branch_preserving_walk_reports_token_buckets_but_advances_branch(
        self,
    ) -> None:
        terminal = _FakeState(terminal=True)
        left = _FakeState(
            terminal=False,
            token_transitions=(_fake_transition("A", terminal),),
        )
        right = _FakeState(
            terminal=False,
            token_transitions=(_fake_transition("B", terminal),),
        )
        merged = _FakeState(
            terminal=False,
            token_transitions=(
                _fake_transition("A", terminal),
                _fake_transition("B", terminal),
            ),
        )
        initial = _FakeState(
            terminal=False,
            token_transitions=(
                _fake_transition("C", merged, branch_count=2),
            ),
            branch_transitions=(
                _fake_transition("C", left),
                _fake_transition("C", right),
            ),
        )

        branch_choices = iter((1, 0))
        branch_result = _walk_branch_transitions(
            initial,
            lambda _transitions: next(branch_choices),
        )
        token_result = _walk_token_transitions(initial, lambda _transitions: 0)

        _assert_token_walk_result_invariants(self, branch_result)
        _assert_token_walk_result_invariants(self, token_result)
        self.assertEqual(("C", "B"), branch_result.tokens)
        self.assertEqual((0, 0), branch_result.selected_indices)
        self.assertEqual(("C", "A"), token_result.tokens)
        self.assertEqual((0, 0), token_result.selected_indices)

    def test_branch_preserving_walk_uses_token_bucket_selected_index(self) -> None:
        terminal = _FakeState(terminal=True)
        initial = _FakeState(
            terminal=False,
            token_transitions=(
                _fake_transition("A", terminal),
                _fake_transition("B", terminal, branch_count=2),
            ),
            branch_transitions=(
                _fake_transition("B", terminal),
                _fake_transition("B", terminal),
            ),
        )

        result = _walk_branch_transitions(initial, lambda _transitions: 1)

        _assert_token_walk_result_invariants(self, result)
        self.assertEqual(("B",), result.tokens)
        self.assertEqual((1,), result.selected_indices)

    def test_seeded_branch_preserving_walk_is_reproducible(self) -> None:
        kwargs = supported_public_kwargs(isomericSmiles=False, rootedAtAtom=-1)
        mol = parse_smiles("CCO")

        first = _walk_branch_transitions(
            grimace.MolToSmilesDecoder(mol, **kwargs)._state,
            _seeded_uniform_transition_chooser(789),
        )
        second = _walk_branch_transitions(
            grimace.MolToSmilesDecoder(mol, **kwargs)._state,
            _seeded_uniform_transition_chooser(789),
        )

        self.assertEqual(first, second)
        _assert_token_walk_result_invariants(self, first)
        self.assertIn("".join(first.tokens), public_enum_support(mol, **kwargs))

    def test_branch_preserving_walk_rejects_invalid_selected_index(self) -> None:
        terminal = _FakeState(terminal=True)
        initial = _FakeState(
            terminal=False,
            token_transitions=(
                _fake_transition("C", terminal),
            ),
        )

        with self.assertRaisesRegex(IndexError, "outside"):
            _walk_branch_transitions(initial, lambda _transitions: 1)

    def test_branch_preserving_walk_rejects_duplicate_token_buckets(self) -> None:
        terminal = _FakeState(terminal=True)
        initial = _FakeState(
            terminal=False,
            token_transitions=(
                _fake_transition("C", terminal),
                _fake_transition("C", terminal),
            ),
        )

        with self.assertRaisesRegex(RuntimeError, "unique text"):
            _walk_branch_transitions(initial, lambda _transitions: 0)

    def test_branch_preserving_walk_rejects_branch_outside_token_view(self) -> None:
        terminal = _FakeState(terminal=True)
        initial = _FakeState(
            terminal=False,
            token_transitions=(
                _fake_transition("C", terminal),
            ),
            branch_transitions=(
                _fake_transition("N", terminal),
            ),
        )

        with self.assertRaisesRegex(RuntimeError, "missing from token transitions"):
            _walk_branch_transitions(initial, lambda _transitions: 0)


if __name__ == "__main__":
    unittest.main()
