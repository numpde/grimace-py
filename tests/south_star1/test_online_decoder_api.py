"""Tests for the South Star online decoder facade."""

from __future__ import annotations

import ast
import unittest
from collections import Counter
from pathlib import Path

from grimace._south_star1.facts import ComponentFacts
from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import ComponentId
from grimace._south_star1.online_decisions import FrontierCompactionMode
from grimace._south_star1.online_decoder_api import EOS
from grimace._south_star1.online_decoder_api import make_branch_preserving_online_decoder
from grimace._south_star1.online_decoder_api import make_determinized_online_decoder
from grimace._south_star1.online_decoder_api import online_decode_token_texts_for_policy
from grimace._south_star1.online_decoder_state import OnlineDecoderState
from grimace._south_star1.online_decoder_state import online_branch_preserving_choices
from grimace._south_star1.online_decoder_state import online_determinized_choices
from grimace._south_star1.online_stereo_witness import iter_online_stereo_witness_strings
from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.ordinary_semantics import OrdinarySmilesSemantics
from tests.south_star1.helpers import atom
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import single_bond
from tests.south_star1.helpers import tetrahedral_facts


REPO_ROOT = Path(__file__).resolve().parents[2]
ONLINE_DECODER_API_PATH = (
    REPO_ROOT / "python" / "grimace" / "_south_star1" / "online_decoder_api.py"
)


class OnlineDecoderApiTest(unittest.TestCase):
    def test_decoder_exposes_eos_at_complete_witness(self) -> None:
        facts = ethane_facts()
        decoder = _determinized_decoder(facts, include_eos=True)
        state = _walk_decoder(decoder, ("C", "C"))

        choices = state.choices()

        self.assertTrue(any(choice.is_eos for choice in choices))
        eos = next(choice for choice in choices if choice.is_eos)
        self.assertEqual(eos.text, EOS)
        self.assertIsNone(eos.next_state)

    def test_decoder_does_not_expose_eos_for_incomplete_prefix(self) -> None:
        decoder = _determinized_decoder(ethane_facts(), include_eos=True)

        self.assertFalse(any(choice.is_eos for choice in decoder.initial_state().choices()))

    def test_decoder_eos_matches_online_witness_multiset(self) -> None:
        facts = ethane_facts()
        decoder = _determinized_decoder(facts, include_eos=True)
        witness_counts = Counter(_witnesses(facts))

        for witness in witness_counts:
            state = _walk_decoder(decoder, _tokens_for_witness(decoder, witness))
            self.assertTrue(any(choice.is_eos for choice in state.choices()))

    def test_branch_preserving_facade_allows_duplicate_text(self) -> None:
        decoder = _branch_decoder(ethane_facts())
        choices = decoder.initial_state().choices()

        self.assertGreater(sum(1 for choice in choices if choice.text == "C"), 1)

    def test_determinized_facade_merges_duplicate_text(self) -> None:
        decoder = _determinized_decoder(ethane_facts())
        choices = decoder.initial_state().choices()

        self.assertEqual(tuple(choice.text for choice in choices), ("C",))

    def test_determinized_multiplicity_matches_branch_preserving_count(self) -> None:
        facts = ethane_facts()
        branch_choices = _branch_decoder(facts).initial_state().choices()
        det_choices = _determinized_decoder(facts).initial_state().choices()

        self.assertEqual(
            det_choices[0].multiplicity,
            sum(1 for choice in branch_choices if choice.text == det_choices[0].text),
        )

    def test_facade_choices_match_existing_online_decoder_state_functions(self) -> None:
        facts = tetrahedral_facts()
        policy = ordinary_policy_for_facts(facts)
        semantics = OrdinarySmilesSemantics()
        decoder = _determinized_decoder(facts)

        self.assertEqual(
            tuple(choice.text for choice in decoder.initial_state().choices()),
            tuple(
                choice.text
                for choice in online_determinized_choices(
                    facts=facts,
                    policy=policy,
                    semantics=semantics,
                    state=OnlineDecoderState(prefix=""),
                )
            ),
        )

    def test_determinized_decoder_can_walk_known_tetra_witness(self) -> None:
        facts = tetrahedral_facts()
        decoder = _determinized_decoder(facts, include_eos=True)
        witness = _witnesses(facts)[0]

        state = _walk_decoder(decoder, _tokens_for_witness(decoder, witness))

        self.assertEqual(state.prefix, witness)
        self.assertTrue(any(choice.is_eos for choice in state.choices()))

    def test_determinized_decoder_can_walk_known_directional_witness(self) -> None:
        facts = directional_facts()
        decoder = _determinized_decoder(facts, include_eos=True)
        witness = _witnesses(facts)[0]

        state = _walk_decoder(decoder, _tokens_for_witness(decoder, witness))

        self.assertEqual(state.prefix, witness)
        self.assertTrue(any(choice.is_eos for choice in state.choices()))

    def test_branch_preserving_decoder_can_walk_to_completion(self) -> None:
        facts = ethane_facts()
        decoder = _branch_decoder(facts, include_eos=True)
        witness = "CC"

        state = _walk_decoder(decoder, ("C", "C"))

        self.assertEqual(state.prefix, witness)
        self.assertTrue(any(choice.is_eos for choice in state.choices()))

    def test_decoder_rejects_impossible_token_sequence(self) -> None:
        decoder = _determinized_decoder(tetrahedral_facts())

        self.assertIsNone(_try_walk_decoder(decoder, ("not-smiles",)))

    def test_facade_stats_reports_single_dfs_run(self) -> None:
        result = _determinized_decoder(tetrahedral_facts()).initial_state().choices_with_stats()

        self.assertEqual(result.stats.dfs_runs, 1)
        self.assertTrue(result.choices)

    def test_traversal_only_and_full_prefix_facades_same_texts(self) -> None:
        facts = directional_facts()
        traversal = _determinized_decoder(
            facts,
            compaction_mode=FrontierCompactionMode.TRAVERSAL_ONLY,
        )
        full = _determinized_decoder(
            facts,
            compaction_mode=FrontierCompactionMode.FULL_DECISION_PREFIX,
        )

        self.assertEqual(
            tuple(choice.text for choice in traversal.initial_state().choices()),
            tuple(choice.text for choice in full.initial_state().choices()),
        )

    def test_token_text_vocabulary_comes_from_policy(self) -> None:
        facts = tetrahedral_facts()
        policy = ordinary_policy_for_facts(facts)

        token_texts = online_decode_token_texts_for_policy(
            facts=facts,
            policy=policy,
            include_eos=True,
        )

        self.assertIn("[C@H]", token_texts)
        self.assertIn("[C@@H]", token_texts)
        self.assertIn(EOS, token_texts)

    def test_online_decoder_api_boundary_no_artifact_or_rdkit_imports(self) -> None:
        tree = ast.parse(ONLINE_DECODER_API_PATH.read_text(encoding="utf-8"))
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


def _branch_decoder(
    facts: MoleculeFacts,
    *,
    compaction_mode: FrontierCompactionMode = FrontierCompactionMode.TRAVERSAL_ONLY,
    include_eos: bool = False,
):
    return make_branch_preserving_online_decoder(
        facts=facts,
        policy=ordinary_policy_for_facts(facts),
        semantics=OrdinarySmilesSemantics(),
        compaction_mode=compaction_mode,
        include_eos=include_eos,
    )


def _determinized_decoder(
    facts: MoleculeFacts,
    *,
    compaction_mode: FrontierCompactionMode = FrontierCompactionMode.TRAVERSAL_ONLY,
    include_eos: bool = False,
):
    return make_determinized_online_decoder(
        facts=facts,
        policy=ordinary_policy_for_facts(facts),
        semantics=OrdinarySmilesSemantics(),
        compaction_mode=compaction_mode,
        include_eos=include_eos,
    )


def _walk_decoder(decoder, token_texts: tuple[str, ...]):
    state = _try_walk_decoder(decoder, token_texts)
    if state is None:
        raise AssertionError(f"decoder rejected token sequence {token_texts!r}")
    return state


def _try_walk_decoder(decoder, token_texts: tuple[str, ...]):
    def rec(state, index: int):
        if index == len(token_texts):
            return state
        token = token_texts[index]
        choices = tuple(choice for choice in state.choices() if not choice.is_eos)
        for match in choices:
            if match.text != token or match.next_state is None:
                continue
            out = rec(match.next_state, index + 1)
            if out is not None:
                return out
        return None

    return rec(decoder.initial_state(), 0)


def _tokens_for_witness(decoder, witness: str) -> tuple[str, ...]:
    state = decoder.initial_state()
    out: list[str] = []
    while state.prefix != witness:
        choices = tuple(choice for choice in state.choices() if not choice.is_eos)
        match = next(
            (
                choice
                for choice in choices
                if witness.startswith(state.prefix + choice.text)
            ),
            None,
        )
        if match is None or match.next_state is None:
            raise AssertionError(f"cannot tokenize witness {witness!r} at {state.prefix!r}")
        out.append(match.text)
        state = match.next_state
    return tuple(out)


def _witnesses(facts: MoleculeFacts) -> tuple[str, ...]:
    return tuple(
        iter_online_stereo_witness_strings(
            facts=facts,
            policy=ordinary_policy_for_facts(facts),
            semantics=OrdinarySmilesSemantics(),
        )
    )


def ethane_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, "C"), atom(1, "C")),
        bonds=(single_bond(0, 0, 1),),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1)),
                bonds=(BondId(0),),
            ),
        ),
    )


if __name__ == "__main__":
    unittest.main()
