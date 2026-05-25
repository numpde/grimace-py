"""Tests for stateful online decoder choices."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from grimace._south_star1.facts import ComponentFacts
from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import ComponentId
from grimace._south_star1.online_decoder import online_allowed_next_token_texts_one_pass
from grimace._south_star1.online_decisions import DecisionPathFilter
from grimace._south_star1.online_decisions import FrontierCompactionMode
from grimace._south_star1.online_decisions import OnlineDecision
from grimace._south_star1.online_decisions import OnlineDecisionFrontier
from grimace._south_star1.online_decisions import OnlineDecisionPath
from grimace._south_star1.online_decoder_state import OnlineDecoderState
from grimace._south_star1.online_decoder_state import online_branch_preserving_choices
from grimace._south_star1.online_decoder_state import online_determinized_choices
from grimace._south_star1.online_decoder_state import online_determinized_choices_with_stats
from grimace._south_star1.online_stereo_witness import iter_online_stereo_witness_strings
from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.ordinary_semantics import OrdinarySmilesSemantics
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_smiles
from tests.south_star1.helpers import atom
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import single_bond
from tests.south_star1.helpers import tetrahedral_facts


REPO_ROOT = Path(__file__).resolve().parents[2]
ONLINE_DECODER_STATE_PATH = (
    REPO_ROOT / "python" / "grimace" / "_south_star1" / "online_decoder_state.py"
)


class OnlineDecoderStateTest(unittest.TestCase):
    def test_branch_preserving_choices_allow_duplicate_text(self) -> None:
        choices = _branch_choices(ethane_facts(), "")

        self.assertGreater(
            sum(1 for choice in choices if choice.text == "C"),
            1,
        )

    def test_determinized_choices_merge_duplicate_text(self) -> None:
        choices = _determinized_choices(ethane_facts(), "")

        self.assertEqual(tuple(choice.text for choice in choices), ("C",))
        self.assertEqual(choices[0].multiplicity, 4)
        self.assertEqual(choices[0].completion_count, 4)

    def test_determinized_choice_multiplicity_counts_merged_branches(self) -> None:
        state = _determinized_choices(ethane_facts(), "")[0].next_state
        choices = _determinized_choices(ethane_facts(), state.prefix, state)

        self.assertEqual(
            {choice.text: choice.multiplicity for choice in choices},
            {"(": 2, "C": 2},
        )

    def test_choice_next_state_advances_prefix(self) -> None:
        choice = _determinized_choices(ethane_facts(), "")[0]

        self.assertEqual(choice.next_state.prefix, "C")

    def test_branch_preserving_next_state_restricts_to_one_branch(self) -> None:
        choice = _branch_choices(ethane_facts(), "")[0]
        next_choices = _branch_choices(
            ethane_facts(),
            choice.next_state.prefix,
            choice.next_state,
        )

        self.assertEqual(len(next_choices), 1)

    def test_determinized_next_state_restricts_to_merged_branches(self) -> None:
        choice = _determinized_choices(ethane_facts(), "")[0]
        next_choices = _determinized_choices(
            ethane_facts(),
            choice.next_state.prefix,
            choice.next_state,
        )

        self.assertEqual(
            {item.text: item.multiplicity for item in next_choices},
            {"(": 2, "C": 2},
        )

    def test_determinized_frontier_matches_existing_one_pass_token_texts_tetra(self) -> None:
        self._assert_determinized_frontier_matches_one_pass(tetrahedral_facts(), "")

    def test_determinized_frontier_matches_existing_one_pass_token_texts_directional(self) -> None:
        self._assert_determinized_frontier_matches_one_pass(directional_facts(), "")

    def test_determinized_frontier_matches_existing_one_pass_token_texts_ring_tetra(self) -> None:
        self._assert_determinized_frontier_matches_one_pass(
            ordinary_molecule_facts_from_smiles("[C@H]1(F)CO1"),
            "",
        )

    def test_decoder_state_can_walk_to_complete_witness(self) -> None:
        facts = ethane_facts()
        state = OnlineDecoderState(prefix="")
        for _ in range(8):
            choices = _determinized_choices(facts, state.prefix, state)
            if not choices:
                break
            state = choices[0].next_state

        self.assertIn(state.prefix, set(_witnesses(facts)))
        self.assertEqual(_determinized_choices(facts, state.prefix, state), ())

    def test_decoder_state_rejects_dead_prefix(self) -> None:
        self.assertEqual(_determinized_choices(tetrahedral_facts(), "not-smiles"), ())

    def test_frontier_state_stores_boundary_prefix_not_full_completion(self) -> None:
        choice = _choice_with_text(_determinized_choices(directional_facts(), ""), "C")
        frontier = choice.next_state.allowed_frontier
        self.assertIsNotNone(frontier)

        self.assertTrue(frontier.paths)
        self.assertTrue(
            all(
                tuple(decision.kind for decision in path.items) == ("traversal",)
                for path in frontier.paths
            )
        )

    def test_frontier_filter_allows_extensions_beyond_saved_prefix(self) -> None:
        frontier_path = OnlineDecisionPath((OnlineDecision("traversal", ("a",)),))
        longer_path = OnlineDecisionPath(
            (
                OnlineDecision("traversal", ("a",)),
                OnlineDecision("direction_mark", (0, "FWD")),
            )
        )
        unrelated_path = OnlineDecisionPath((OnlineDecision("traversal", ("b",)),))
        path_filter = DecisionPathFilter(OnlineDecisionFrontier(frozenset({frontier_path})))

        self.assertTrue(path_filter.allows_prefix(longer_path))
        self.assertFalse(path_filter.allows_prefix(unrelated_path))

    def test_determinized_choice_multiplicity_counts_frontier_paths_not_completions(self) -> None:
        choice = _choice_with_text(_determinized_choices(directional_facts(), ""), "C")

        self.assertLess(choice.multiplicity, choice.completion_count)
        self.assertEqual(choice.multiplicity, len(choice.next_state.allowed_frontier.paths))

    def test_completion_count_tracks_multiple_completions_for_same_frontier(self) -> None:
        choice = _choice_with_text(_branch_choices(directional_facts(), ""), "C")

        self.assertEqual(choice.multiplicity, 1)
        self.assertGreater(choice.completion_count, 1)

    def test_branch_preserving_choice_next_state_can_continue_after_frontier_prefix(self) -> None:
        choice = _choice_with_text(_branch_choices(directional_facts(), ""), "C")
        next_choices = _branch_choices(
            directional_facts(),
            choice.next_state.prefix,
            choice.next_state,
        )

        self.assertTrue(next_choices)

    def test_determinized_walk_matches_online_witness_prefixes_for_multiple_steps(self) -> None:
        facts = directional_facts()
        witnesses = set(_witnesses(facts))
        state = OnlineDecoderState(prefix="")
        for _ in range(4):
            choices = _determinized_choices(facts, state.prefix, state)
            self.assertEqual(
                tuple(sorted(choice.text for choice in choices)),
                online_allowed_next_token_texts_one_pass(
                    facts=facts,
                    policy=ordinary_policy_for_facts(facts),
                    semantics=OrdinarySmilesSemantics(),
                    prefix=state.prefix,
                ),
            )
            if not choices:
                break
            state = choices[0].next_state
            self.assertTrue(any(witness.startswith(state.prefix) for witness in witnesses))

    def test_traversal_only_frontier_matches_full_prefix_frontier_tetra(self) -> None:
        self._assert_compaction_modes_match(tetrahedral_facts(), ())

    def test_traversal_only_frontier_matches_full_prefix_frontier_directional(self) -> None:
        self._assert_compaction_modes_match(directional_facts(), ())

    def test_traversal_only_frontier_matches_full_prefix_frontier_ring_tetra(self) -> None:
        self._assert_compaction_modes_match(
            ordinary_molecule_facts_from_smiles("[C@H]1(F)CO1"),
            (),
        )

    def test_traversal_only_frontier_matches_full_prefix_frontier_after_two_steps(self) -> None:
        self._assert_compaction_modes_match(directional_facts(), ("C", "("))

    def test_traversal_only_frontier_matches_full_prefix_frontier_after_branch_token(self) -> None:
        self._assert_compaction_modes_match(ethane_facts(), ("C", "("))

    def test_traversal_only_frontier_matches_full_prefix_frontier_after_ring_label_token(self) -> None:
        self._assert_compaction_modes_match(
            ordinary_molecule_facts_from_smiles("[C@H]1(F)CO1"),
            ("C", "1"),
        )

    def test_traversal_only_compaction_prunes_no_more_than_full_prefix(self) -> None:
        facts = directional_facts()
        traversal_state = _state_after_tokens(
            facts,
            ("C",),
            FrontierCompactionMode.TRAVERSAL_ONLY,
        )
        full_state = _state_after_tokens(
            facts,
            ("C",),
            FrontierCompactionMode.FULL_DECISION_PREFIX,
        )

        traversal_choices, traversal_stats = _determinized_choices_with_stats(
            facts,
            traversal_state,
            FrontierCompactionMode.TRAVERSAL_ONLY,
        )
        full_choices, full_stats = _determinized_choices_with_stats(
            facts,
            full_state,
            FrontierCompactionMode.FULL_DECISION_PREFIX,
        )

        self.assertEqual(
            tuple(choice.text for choice in traversal_choices),
            tuple(choice.text for choice in full_choices),
        )
        self.assertLessEqual(
            traversal_stats.decision_prefix_rejections,
            full_stats.decision_prefix_rejections,
        )

    def test_online_decoder_state_boundary_no_artifact_or_rdkit_imports(self) -> None:
        tree = ast.parse(ONLINE_DECODER_STATE_PATH.read_text(encoding="utf-8"))
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

    def _assert_determinized_frontier_matches_one_pass(
        self,
        facts: MoleculeFacts,
        prefix: str,
    ) -> None:
        policy = ordinary_policy_for_facts(facts)
        semantics = OrdinarySmilesSemantics()
        self.assertEqual(
            tuple(
                sorted(
                    choice.text
                    for choice in online_determinized_choices(
                        facts=facts,
                        policy=policy,
                        semantics=semantics,
                        state=OnlineDecoderState(prefix=prefix),
                    )
                )
            ),
            online_allowed_next_token_texts_one_pass(
                facts=facts,
                policy=policy,
                semantics=semantics,
                prefix=prefix,
            ),
        )

    def _assert_compaction_modes_match(
        self,
        facts: MoleculeFacts,
        tokens: tuple[str, ...],
    ) -> None:
        traversal_state = _state_after_tokens(
            facts,
            tokens,
            FrontierCompactionMode.TRAVERSAL_ONLY,
        )
        full_state = _state_after_tokens(
            facts,
            tokens,
            FrontierCompactionMode.FULL_DECISION_PREFIX,
        )

        self.assertEqual(
            _choice_texts(
                _determinized_choices(
                    facts,
                    traversal_state.prefix,
                    traversal_state,
                    FrontierCompactionMode.TRAVERSAL_ONLY,
                )
            ),
            _choice_texts(
                _determinized_choices(
                    facts,
                    full_state.prefix,
                    full_state,
                    FrontierCompactionMode.FULL_DECISION_PREFIX,
                )
            ),
        )


def _branch_choices(
    facts: MoleculeFacts,
    prefix: str,
    state: OnlineDecoderState | None = None,
    compaction_mode: FrontierCompactionMode = FrontierCompactionMode.TRAVERSAL_ONLY,
):
    policy = ordinary_policy_for_facts(facts)
    return online_branch_preserving_choices(
        facts=facts,
        policy=policy,
        semantics=OrdinarySmilesSemantics(),
        state=state or OnlineDecoderState(prefix=prefix),
        compaction_mode=compaction_mode,
    )


def _determinized_choices(
    facts: MoleculeFacts,
    prefix: str,
    state: OnlineDecoderState | None = None,
    compaction_mode: FrontierCompactionMode = FrontierCompactionMode.TRAVERSAL_ONLY,
):
    policy = ordinary_policy_for_facts(facts)
    return online_determinized_choices(
        facts=facts,
        policy=policy,
        semantics=OrdinarySmilesSemantics(),
        state=state or OnlineDecoderState(prefix=prefix),
        compaction_mode=compaction_mode,
    )


def _determinized_choices_with_stats(
    facts: MoleculeFacts,
    state: OnlineDecoderState,
    compaction_mode: FrontierCompactionMode,
):
    policy = ordinary_policy_for_facts(facts)
    return online_determinized_choices_with_stats(
        facts=facts,
        policy=policy,
        semantics=OrdinarySmilesSemantics(),
        state=state,
        compaction_mode=compaction_mode,
    )


def _witnesses(facts: MoleculeFacts) -> tuple[str, ...]:
    return tuple(
        iter_online_stereo_witness_strings(
            facts=facts,
            policy=ordinary_policy_for_facts(facts),
            semantics=OrdinarySmilesSemantics(),
        )
    )


def _choice_with_text(choices, text: str):
    for choice in choices:
        if choice.text == text:
            return choice
    raise AssertionError(f"missing choice text {text!r}: {choices!r}")


def _choice_texts(choices) -> tuple[str, ...]:
    return tuple(sorted(choice.text for choice in choices))


def _state_after_tokens(
    facts: MoleculeFacts,
    tokens: tuple[str, ...],
    compaction_mode: FrontierCompactionMode,
) -> OnlineDecoderState:
    state = OnlineDecoderState(prefix="")
    for token in tokens:
        state = _choice_with_text(
            _determinized_choices(
                facts,
                state.prefix,
                state,
                compaction_mode,
            ),
            token,
        ).next_state
    return state


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
