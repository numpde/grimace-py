"""Tests for resumable South Star online decoder continuations."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from grimace._south_star1.online_continuation import OnlineDecoderExecutionMode
from grimace._south_star1.online_decoder_api import EOS
from grimace._south_star1.online_decoder_api import make_branch_preserving_online_decoder
from grimace._south_star1.online_decoder_api import make_determinized_online_decoder
from grimace._south_star1.online_stereo_witness import iter_online_stereo_witness_strings
from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.ordinary_semantics import OrdinarySmilesSemantics
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_smiles
from tests.south_star1.helpers import cyclopropane_facts
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import tetrahedral_facts


REPO_ROOT = Path(__file__).resolve().parents[2]
ONLINE_CONTINUATION_PATH = (
    REPO_ROOT / "python" / "grimace" / "_south_star1" / "online_continuation.py"
)


class OnlineContinuationDecoderTest(unittest.TestCase):
    def test_continuation_decoder_matches_replay_tetra(self) -> None:
        self._assert_initial_texts_match_replay(tetrahedral_facts())

    def test_continuation_decoder_matches_replay_directional(self) -> None:
        self._assert_initial_texts_match_replay(directional_facts())

    def test_continuation_decoder_matches_replay_ring_tetra(self) -> None:
        self._assert_initial_texts_match_replay(
            ordinary_molecule_facts_from_smiles("[C@H]1(F)CO1"),
        )

    def test_continuation_decoder_matches_replay_ring(self) -> None:
        self._assert_initial_texts_match_replay(cyclopropane_facts())

    def test_continuation_state_does_not_restart_from_root(self) -> None:
        decoder = _continuation_determinized_decoder(tetrahedral_facts())
        first = decoder.initial_state().choices_with_stats()
        self.assertEqual(first.stats.root_dfs_runs, 1)
        self.assertTrue(first.choices)

        next_state = first.choices[0].next_state
        self.assertIsNotNone(next_state)
        second = next_state.choices_with_stats()

        self.assertEqual(second.stats.root_dfs_runs, 0)
        self.assertGreater(second.stats.resumed_continuations, 0)

    def test_continuation_determinized_choices_merge_same_text(self) -> None:
        choices = _continuation_determinized_decoder(cyclopropane_facts()).initial_state().choices()

        self.assertEqual(
            tuple(choice.text for choice in choices),
            tuple(sorted({choice.text for choice in choices})),
        )
        self.assertTrue(any(choice.multiplicity > 1 for choice in choices))

    def test_continuation_eos_matches_replay(self) -> None:
        facts = tetrahedral_facts()
        replay = _replay_determinized_decoder(facts, include_eos=True)
        continuation = _continuation_determinized_decoder(facts, include_eos=True)
        witness = _witnesses(facts)[0]
        replay_state = _walk_decoder(replay, _tokens_for_witness(replay, witness))
        continuation_state = _walk_decoder(
            continuation,
            _tokens_for_witness(continuation, witness),
        )

        self.assertEqual(
            _choice_texts(replay_state.choices()),
            _choice_texts(continuation_state.choices()),
        )
        self.assertTrue(any(choice.is_eos for choice in continuation_state.choices()))

    def test_continuation_walks_known_witness(self) -> None:
        facts = directional_facts()
        decoder = _continuation_determinized_decoder(facts, include_eos=True)
        witness = _witnesses(facts)[0]

        state = _walk_decoder(decoder, _tokens_for_witness(decoder, witness))

        self.assertEqual(state.prefix, witness)
        self.assertTrue(any(choice.text == EOS and choice.is_eos for choice in state.choices()))

    def test_branch_preserving_continuation_allows_duplicate_text(self) -> None:
        choices = _continuation_branch_decoder(cyclopropane_facts()).initial_state().choices()

        self.assertGreater(
            max(
                sum(1 for choice in choices if choice.text == text)
                for text in {choice.text for choice in choices}
            ),
            1,
        )

    def test_continuation_dead_prefix_matches_replay(self) -> None:
        facts = directional_facts()
        replay = _replay_determinized_decoder(facts)
        continuation = _continuation_determinized_decoder(facts)

        self.assertEqual(
            _choice_texts(_state_for_prefix(replay, "not-smiles").choices()),
            _choice_texts(_state_for_prefix(continuation, "not-smiles").choices()),
        )

    def test_online_continuation_boundary_no_artifact_or_rdkit_imports(self) -> None:
        tree = ast.parse(ONLINE_CONTINUATION_PATH.read_text(encoding="utf-8"))
        banned_modules = {
            "audit_rdkit",
            "finite_space_checker",
            "rdkit_adapter",
            "semantic_relation_checker",
            "support_artifact",
            "support_artifact_checker",
            "support_enumeration",
        }
        banned_calls = {
            "compile_support_artifact",
            "enumerate_stereo_support",
            "render_image_from_witnesses",
        }
        imports: list[str] = []
        calls: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(
                    alias.name
                    for alias in node.names
                    if alias.name.split(".", 1)[0] in banned_modules
                )
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.split(".", 1)[0] in banned_modules:
                    imports.append(module)
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
                if isinstance(node.func, ast.Attribute):
                    calls.append(node.func.attr)

        self.assertEqual(imports, [])
        self.assertEqual(sorted(set(calls) & banned_calls), [])

    def _assert_initial_texts_match_replay(self, facts) -> None:
        replay = _replay_determinized_decoder(facts)
        continuation = _continuation_determinized_decoder(facts)

        self.assertEqual(
            _choice_texts(replay.initial_state().choices()),
            _choice_texts(continuation.initial_state().choices()),
        )


def _replay_determinized_decoder(facts, *, include_eos: bool = False):
    return make_determinized_online_decoder(
        facts=facts,
        policy=ordinary_policy_for_facts(facts),
        semantics=OrdinarySmilesSemantics(),
        include_eos=include_eos,
        execution_mode=OnlineDecoderExecutionMode.PREFIX_REPLAY,
    )


def _continuation_determinized_decoder(facts, *, include_eos: bool = False):
    return make_determinized_online_decoder(
        facts=facts,
        policy=ordinary_policy_for_facts(facts),
        semantics=OrdinarySmilesSemantics(),
        include_eos=include_eos,
        execution_mode=OnlineDecoderExecutionMode.RESUMABLE_CONTINUATIONS,
    )


def _continuation_branch_decoder(facts):
    return make_branch_preserving_online_decoder(
        facts=facts,
        policy=ordinary_policy_for_facts(facts),
        semantics=OrdinarySmilesSemantics(),
        execution_mode=OnlineDecoderExecutionMode.RESUMABLE_CONTINUATIONS,
    )


def _state_for_prefix(decoder, prefix: str):
    state = decoder.initial_state()
    raw_state = type(state.raw_state)(prefix=prefix)
    return type(state)(prefix=prefix, raw_state=raw_state, decoder=decoder)


def _choice_texts(choices) -> tuple[str, ...]:
    return tuple(choice.text for choice in choices)


def _walk_decoder(decoder, token_texts: tuple[str, ...]):
    state = decoder.initial_state()
    for token in token_texts:
        for choice in state.choices():
            if choice.is_eos or choice.text != token or choice.next_state is None:
                continue
            state = choice.next_state
            break
        else:
            raise AssertionError(f"decoder rejected token {token!r} after {state.prefix!r}")
    return state


def _tokens_for_witness(decoder, witness: str) -> tuple[str, ...]:
    def rec(state, out: tuple[str, ...]) -> tuple[str, ...] | None:
        if state.prefix == witness:
            return out
        for choice in state.choices():
            if choice.is_eos or choice.next_state is None:
                continue
            if not witness.startswith(state.prefix + choice.text):
                continue
            result = rec(choice.next_state, out + (choice.text,))
            if result is not None:
                return result
        return None

    result = rec(decoder.initial_state(), ())
    if result is None:
        raise AssertionError(f"cannot tokenize witness {witness!r}")
    return result


def _witnesses(facts) -> tuple[str, ...]:
    return tuple(
        iter_online_stereo_witness_strings(
            facts=facts,
            policy=ordinary_policy_for_facts(facts),
            semantics=OrdinarySmilesSemantics(),
        )
    )


if __name__ == "__main__":
    unittest.main()
