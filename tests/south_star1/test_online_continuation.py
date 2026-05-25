"""Tests for cached-completion South Star online decoder continuations."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from grimace._south_star1.online_continuation import OnlineDecoderExecutionMode
from grimace._south_star1.online_decoder_api import EOS
from grimace._south_star1.online_decoder_api import make_branch_preserving_online_decoder
from grimace._south_star1.online_decoder_api import make_determinized_online_decoder
from grimace._south_star1.online_decisions import OnlineDecision
from grimace._south_star1.online_decisions import OnlineDecisionPath
from grimace._south_star1.online_residual_continuation import OnlineResidualContinuation
from grimace._south_star1.online_residual_continuation import OnlineResidualContinuationFrontier
from grimace._south_star1.online_residual_continuation import OnlineResidualDecoderState
from grimace._south_star1.online_residual_continuation import ResidualFrontierSink
from grimace._south_star1.online_residual_continuation import ResidualFrontierSinkCheckpoint
from grimace._south_star1.online_residual_continuation import merge_residual_continuations_by_key
from grimace._south_star1.online_residual_continuation import online_search_snapshot_shape
from grimace._south_star1.online_residual_continuation import residual_frontier_shape
from grimace._south_star1.online_residual_continuation import residual_continuation_key
from grimace._south_star1.online_search_vm import OnlineSearchFrame
from grimace._south_star1.online_search_vm import OnlineSearchSnapshot
from grimace._south_star1.online_stereo_witness import iter_online_stereo_witness_strings
from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.ordinary_policy import OrdinaryPolicyOptions
from grimace._south_star1.ordinary_semantics import OrdinarySmilesSemantics
from grimace._south_star1.policy import AnnotationMode
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_smiles
from tests.south_star1.helpers import cyclopropane_facts
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import four_substituent_directional_facts
from tests.south_star1.helpers import tetrahedral_facts


REPO_ROOT = Path(__file__).resolve().parents[2]
ONLINE_CONTINUATION_PATH = (
    REPO_ROOT / "python" / "grimace" / "_south_star1" / "online_continuation.py"
)
SPEC_PATH = REPO_ROOT / "python" / "grimace" / "_south_star1" / "SPEC.md"


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
        facts = four_substituent_directional_facts()
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
        facts = four_substituent_directional_facts()
        replay = _replay_determinized_decoder(facts)
        continuation = _continuation_determinized_decoder(facts)

        self.assertEqual(
            _choice_texts(_state_for_prefix(replay, "not-smiles").choices()),
            _choice_texts(_state_for_prefix(continuation, "not-smiles").choices()),
        )

    def test_cached_completion_mode_stores_completed_tokens(self) -> None:
        decoder = _continuation_determinized_decoder(tetrahedral_facts())
        choice = decoder.initial_state().choices()[0]
        self.assertIsNotNone(choice.next_state)
        frontier = choice.next_state.raw_state.frontier
        self.assertIsNotNone(frontier)

        continuation = frontier.continuations[0]

        self.assertTrue(continuation.rendered)
        self.assertTrue(continuation.tokens)
        self.assertGreater(continuation.token_index, 0)
        self.assertTrue(continuation.rendered.startswith(continuation.prefix))

    def test_cached_completion_mode_does_not_claim_residual_snapshot(self) -> None:
        decoder = _continuation_determinized_decoder(tetrahedral_facts())
        choice = decoder.initial_state().choices()[0]
        self.assertIsNotNone(choice.next_state)
        frontier = choice.next_state.raw_state.frontier
        self.assertIsNotNone(frontier)

        continuation = frontier.continuations[0]

        self.assertEqual(continuation.traversal_cursor, ())
        self.assertEqual(continuation.residual_snapshot, ())
        self.assertEqual(continuation.ring_state, ())

    def test_execution_mode_alias_resumable_continuations_is_deprecated_if_kept(self) -> None:
        self.assertIs(
            OnlineDecoderExecutionMode.RESUMABLE_CONTINUATIONS,
            OnlineDecoderExecutionMode.CACHED_COMPLETIONS,
        )
        self.assertEqual(
            OnlineDecoderExecutionMode("resumable_continuations"),
            OnlineDecoderExecutionMode.CACHED_COMPLETIONS,
        )

    def test_residual_continuations_match_prefix_replay_tetra(self) -> None:
        self._assert_residual_texts_match_replay(tetrahedral_facts())

    def test_residual_continuations_match_prefix_replay_directional(self) -> None:
        self._assert_residual_texts_match_replay(directional_facts())

    def test_residual_continuations_match_prefix_replay_ring_tetra(self) -> None:
        self._assert_residual_texts_match_replay(
            ordinary_molecule_facts_from_smiles("[C@H]1(F)CO1"),
        )

    def test_residual_continuation_next_step_does_not_restart_from_root(self) -> None:
        decoder = _residual_determinized_decoder(tetrahedral_facts())
        first = decoder.initial_state().choices_with_stats()
        self.assertEqual(first.stats.root_dfs_runs, 1)
        self.assertTrue(first.choices)

        next_state = first.choices[0].next_state
        self.assertIsNotNone(next_state)
        second = next_state.choices_with_stats()

        self.assertEqual(second.stats.root_dfs_runs, 0)
        self.assertGreater(second.stats.resumed_snapshots, 0)

    def test_residual_continuation_state_contains_snapshots_not_completed_token_streams(self) -> None:
        decoder = _residual_determinized_decoder(tetrahedral_facts())
        choice = decoder.initial_state().choices()[0]
        self.assertIsNotNone(choice.next_state)
        frontier = choice.next_state.raw_state.frontier
        self.assertIsNotNone(frontier)

        continuation = frontier.continuations[0]

        self.assertTrue(continuation.snapshot.frame_stack)
        self.assertFalse(hasattr(continuation, "rendered"))
        self.assertFalse(hasattr(continuation, "tokens"))

    def test_residual_continuation_snapshot_has_frame_stack(self) -> None:
        continuation = _first_residual_continuation(tetrahedral_facts())

        self.assertTrue(continuation.snapshot.frame_stack)

    def test_residual_continuation_snapshot_has_residual_state(self) -> None:
        continuation = _first_residual_continuation(tetrahedral_facts())

        self.assertIsNotNone(continuation.snapshot.residual_snapshot)

    def test_residual_continuation_snapshot_has_ring_state(self) -> None:
        continuation = _first_residual_continuation(
            ordinary_molecule_facts_from_smiles("[C@H]1(F)CO1"),
        )

        self.assertIsNotNone(continuation.snapshot.ring_state)

    def test_residual_continuation_eos_matches_prefix_replay(self) -> None:
        facts = tetrahedral_facts()
        replay = _replay_determinized_decoder(facts, include_eos=True)
        residual = _residual_determinized_decoder(facts, include_eos=True)
        witness = _witnesses(facts)[0]
        replay_state = _walk_decoder(replay, _tokens_for_witness(replay, witness))
        residual_state = _walk_decoder(residual, _tokens_for_witness(residual, witness))

        self.assertEqual(
            _choice_texts(replay_state.choices()),
            _choice_texts(residual_state.choices()),
        )
        self.assertTrue(any(choice.is_eos for choice in residual_state.choices()))

    def test_residual_continuation_walks_known_tetra_witness(self) -> None:
        facts = tetrahedral_facts()
        decoder = _residual_determinized_decoder(facts, include_eos=True)
        witness = _witnesses(facts)[0]

        state = _walk_decoder(decoder, _tokens_for_witness(decoder, witness))

        self.assertEqual(state.prefix, witness)
        self.assertTrue(any(choice.text == EOS and choice.is_eos for choice in state.choices()))

    def test_residual_continuation_walks_known_directional_witness(self) -> None:
        facts = directional_facts()
        decoder = _residual_determinized_decoder(facts, include_eos=True)
        witness = _witnesses(facts)[0]

        state = _walk_decoder(decoder, _tokens_for_witness(decoder, witness))

        self.assertEqual(state.prefix, witness)
        self.assertTrue(any(choice.text == EOS and choice.is_eos for choice in state.choices()))

    def test_residual_continuation_determinized_merges_same_text(self) -> None:
        choices = _residual_determinized_decoder(cyclopropane_facts()).initial_state().choices()

        self.assertEqual(
            tuple(choice.text for choice in choices),
            tuple(sorted({choice.text for choice in choices})),
        )
        self.assertTrue(any(choice.multiplicity > 1 for choice in choices))

    def test_residual_continuation_branch_preserving_allows_duplicate_text(self) -> None:
        choices = _residual_branch_decoder(cyclopropane_facts()).initial_state().choices()

        self.assertGreater(
            max(
                sum(1 for choice in choices if choice.text == text)
                for text in {choice.text for choice in choices}
            ),
            1,
        )

    def test_residual_frontier_sink_restores_pending_token_after_nested_rollback(self) -> None:
        sink = _sink_with_providers(required_prefix="")
        self.assertTrue(sink.append("C", token_text="C"))
        checkpoint = sink.checkpoint()

        self.assertTrue(sink.append("l", token_text="l"))
        sink.rollback(checkpoint)

        self.assertEqual(sink.pending_token_text, "C")

    def test_residual_frontier_sink_restores_pending_snapshot_after_nested_rollback(self) -> None:
        sink = _sink_with_providers(required_prefix="")
        self.assertTrue(sink.append("C", token_text="C"))
        checkpoint = sink.checkpoint()

        self.assertTrue(sink.append("l", token_text="l"))
        sink.rollback(checkpoint)

        self.assertEqual(sink.pending_snapshot, _snapshot("pending"))

    def test_residual_frontier_sink_restores_pending_frontier_path_after_nested_rollback(self) -> None:
        sink = _sink_with_providers(required_prefix="")
        self.assertTrue(sink.append("C", token_text="C"))
        checkpoint = sink.checkpoint()

        self.assertTrue(sink.append("l", token_text="l"))
        sink.rollback(checkpoint)

        self.assertEqual(sink.pending_frontier_path, _path("pending"))

    def test_residual_frontier_sink_does_not_commit_dead_branch_after_rollback(self) -> None:
        sink = _sink_with_providers(required_prefix="")
        self.assertTrue(sink.append("C", token_text="C"))
        checkpoint = sink.checkpoint()

        self.assertTrue(sink.append("l", token_text="l"))
        sink.rollback(checkpoint)

        self.assertEqual(sink.completed_by_token, {})

    def test_residual_frontier_sink_preserves_committed_frontiers_across_sibling_rollback(self) -> None:
        sink = _sink_with_providers(required_prefix="")
        self.assertTrue(sink.append("C", token_text="C"))
        checkpoint = sink.checkpoint()

        self.assertTrue(sink.append("l", token_text="l"))
        self.assertTrue(sink.complete())
        sink.rollback(checkpoint)

        self.assertIn("C", sink.completed_by_token)

    def test_residual_frontier_sink_rollback_preserves_committed_eos_frontier(self) -> None:
        sink = _sink_with_providers(required_prefix="")
        checkpoint = sink.checkpoint()

        self.assertTrue(sink.complete())
        sink.rollback(checkpoint)

        self.assertTrue(sink.eos_by_frontier)

    def test_residual_determinized_multiplicity_counts_unique_snapshots(self) -> None:
        duplicate = _continuation("C", "C", "same", completion_count=1)
        other = _continuation("C", "C", "other", completion_count=1)

        self.assertNotEqual(
            residual_continuation_key(duplicate),
            residual_continuation_key(other),
        )

    def test_residual_completion_count_counts_multiple_completions(self) -> None:
        one = _continuation("C", "C", "same", completion_count=1)
        three = _continuation("C", "C", "same", completion_count=3)

        self.assertEqual(residual_continuation_key(one), residual_continuation_key(three))
        self.assertNotEqual(one.completion_count, three.completion_count)

    def test_residual_duplicate_continuations_are_merged_by_key(self) -> None:
        one = _continuation("C", "C", "same", completion_count=1)
        three = _continuation("C", "C", "same", completion_count=3)

        merged = merge_residual_continuations_by_key([one, three])

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].completion_count, 4)

    def test_all_execution_modes_walk_same_tetra_witnesses(self) -> None:
        self._assert_all_modes_walk_same_witnesses(tetrahedral_facts())

    def test_all_execution_modes_walk_same_directional_witnesses(self) -> None:
        self._assert_all_modes_walk_same_witnesses(directional_facts())

    def test_all_execution_modes_walk_same_ring_tetra_witnesses(self) -> None:
        self._assert_all_modes_walk_same_witnesses(
            ordinary_molecule_facts_from_smiles("[C@H]1(F)CO1"),
        )

    def test_residual_initial_state_stats_root_run_only(self) -> None:
        result = _residual_determinized_decoder(tetrahedral_facts()).initial_state().choices_with_stats()

        self.assertEqual(result.stats.root_dfs_runs, 1)
        self.assertEqual(result.stats.resumed_snapshots, 0)

    def test_residual_successor_state_stats_resume_only(self) -> None:
        result = _residual_determinized_decoder(tetrahedral_facts()).initial_state().choices_with_stats()
        next_state = result.choices[0].next_state
        self.assertIsNotNone(next_state)

        successor = next_state.choices_with_stats()

        self.assertEqual(successor.stats.root_dfs_runs, 0)
        self.assertGreater(successor.stats.resumed_snapshots, 0)

    def test_residual_state_size_reports_render_payload_for_tetra(self) -> None:
        result = _residual_determinized_decoder(tetrahedral_facts()).initial_state().choices_with_stats()

        self.assertGreater(result.stats.retained_state_size.render_resume_continuation_count, 0)
        self.assertGreater(result.stats.retained_state_size.max_render_piece_count, 0)
        self.assertGreater(result.stats.retained_state_size.max_render_payload_chars, 0)
        self.assertGreater(result.stats.retained_state_size.total_render_payload_chars, 0)

    def test_residual_state_size_reports_render_payload_for_ring_tetra(self) -> None:
        facts = ordinary_molecule_facts_from_smiles("[C@H]1(F)CO1")
        result = _residual_determinized_decoder(facts).initial_state().choices_with_stats()

        self.assertGreater(result.stats.retained_state_size.render_resume_continuation_count, 0)
        self.assertGreater(result.stats.retained_state_size.max_render_piece_count, 0)
        self.assertGreater(result.stats.retained_state_size.max_remaining_render_piece_count, 0)
        self.assertGreater(result.stats.retained_state_size.total_remaining_render_piece_count, 0)

    def test_residual_frontier_shape_counts_unique_continuations(self) -> None:
        result = _residual_determinized_decoder(directional_facts()).initial_state().choices_with_stats()
        choice = next(choice for choice in result.choices if choice.next_state is not None)
        frontier = choice.next_state.raw_state.frontier
        self.assertIsNotNone(frontier)

        shape = residual_frontier_shape(frontier)

        self.assertGreaterEqual(shape.continuation_count, shape.unique_continuation_count)
        self.assertEqual(shape.merged_continuation_count, shape.continuation_count - shape.unique_continuation_count)
        self.assertGreater(shape.max_frame_stack_depth, 0)
        self.assertGreater(shape.max_residual_factor_count, 0)

    def test_online_search_snapshot_shape_reports_residual_and_render_payload(self) -> None:
        continuation = _first_residual_continuation(directional_facts())

        shape = online_search_snapshot_shape(continuation.snapshot)

        self.assertGreater(shape.residual_var_count, 0)
        self.assertGreater(shape.residual_assignment_count, 0)
        self.assertGreater(shape.residual_factor_count, 0)
        self.assertGreater(shape.render_payload.render_resume_continuation_count, 0)

    def test_resuming_two_residual_continuations_is_order_independent(self) -> None:
        facts = tetrahedral_facts()
        decoder = _residual_determinized_decoder(facts)
        choice = next(
            choice
            for choice in decoder.initial_state().choices()
            if choice.next_state is not None
            and choice.next_state.raw_state.frontier is not None
            and len(choice.next_state.raw_state.frontier.continuations) >= 2
        )
        assert choice.next_state is not None
        frontier = choice.next_state.raw_state.frontier
        assert frontier is not None
        left, right = frontier.continuations[:2]

        forward = _residual_state_for_continuations(decoder, choice.text, (left, right))
        backward = _residual_state_for_continuations(decoder, choice.text, (right, left))

        forward_result = forward.choices_with_stats()
        backward_result = backward.choices_with_stats()

        self.assertEqual(
            _choice_signature(forward_result.choices),
            _choice_signature(backward_result.choices),
        )
        self.assertEqual(forward_result.stats.eos_completions_seen, backward_result.stats.eos_completions_seen)
        self.assertEqual(forward_result.stats.eos_frontier_paths, backward_result.stats.eos_frontier_paths)
        self.assertEqual(forward_result.stats.candidate_state_size, backward_result.stats.candidate_state_size)
        self.assertEqual(forward_result.stats.retained_state_size, backward_result.stats.retained_state_size)

    def test_execution_modes_report_distinct_storage_behavior(self) -> None:
        facts = tetrahedral_facts()
        prefix_replay = _decoder_for_mode(
            facts,
            OnlineDecoderExecutionMode.PREFIX_REPLAY,
        ).initial_state().choices_with_stats()
        cached = _decoder_for_mode(
            facts,
            OnlineDecoderExecutionMode.CACHED_COMPLETIONS,
        ).initial_state().choices_with_stats()
        residual = _decoder_for_mode(
            facts,
            OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
        ).initial_state().choices_with_stats()

        self.assertEqual(prefix_replay.stats.dfs_runs, 1)
        self.assertEqual(cached.stats.root_dfs_runs, 1)
        self.assertEqual(residual.stats.root_dfs_runs, 1)
        self.assertFalse(hasattr(prefix_replay.stats, "candidate_state_size"))
        self.assertFalse(hasattr(cached.stats, "candidate_state_size"))
        self.assertGreater(residual.stats.retained_state_size.render_resume_continuation_count, 0)

    def test_candidate_state_size_counts_eos_snapshots_but_retained_state_size_does_not(self) -> None:
        facts = tetrahedral_facts()
        decoder = _residual_determinized_decoder(facts, include_eos=True)
        witness = _witnesses(facts)[0]
        state = _walk_decoder(decoder, _tokens_for_witness(decoder, witness))

        result = state.choices_with_stats()

        self.assertTrue(any(choice.is_eos for choice in result.choices))
        self.assertGreater(result.stats.candidate_state_size.continuation_count, 0)
        self.assertEqual(result.stats.retained_state_size.continuation_count, 0)
        self.assertGreater(result.stats.eos_completions_seen, 0)

    def test_retained_state_size_equals_sum_of_successor_frontiers(self) -> None:
        result = _residual_determinized_decoder(directional_facts()).initial_state().choices_with_stats()
        retained = tuple(
            continuation
            for choice in result.choices
            if choice.next_state is not None
            and choice.next_state.raw_state.frontier is not None
            for continuation in choice.next_state.raw_state.frontier.continuations
        )

        self.assertEqual(
            result.stats.retained_state_size,
            residual_frontier_shape(
                OnlineResidualContinuationFrontier(prefix="", continuations=retained)
            ),
        )

    def test_duplicate_merge_stats_are_reported_per_token(self) -> None:
        duplicate = _continuation("C", "C", "same", completion_count=1)
        same = _continuation("C", "C", "same", completion_count=3)
        other = _continuation("C", "C", "other", completion_count=1)
        frontier = OnlineResidualContinuationFrontier(
            prefix="C",
            continuations=(duplicate, same, other),
        )

        shape = residual_frontier_shape(frontier)

        self.assertEqual(shape.continuation_count, 3)
        self.assertEqual(shape.unique_continuation_count, 2)
        self.assertEqual(shape.max_merge_count_per_key, 2)
        self.assertEqual(shape.max_merge_count_per_token, 3)

    def test_retained_snapshot_output_checkpoint_has_no_nested_pending_snapshot(self) -> None:
        result = _residual_determinized_decoder(tetrahedral_facts()).initial_state().choices_with_stats()
        for choice in result.choices:
            if choice.next_state is None or choice.next_state.raw_state.frontier is None:
                continue
            for continuation in choice.next_state.raw_state.frontier.continuations:
                output_snapshot = continuation.snapshot.output_snapshot
                if isinstance(output_snapshot, ResidualFrontierSinkCheckpoint):
                    self.assertIsNone(output_snapshot.pending_snapshot)

    def test_support_maximal_residual_decoder_ignores_nonmaximal_directional_candidate(self) -> None:
        facts = four_substituent_directional_facts()
        nonmaximal = _hard_only_directional_witness(facts)
        decoder = _residual_determinized_decoder(facts, include_eos=True)
        state = _state_for_prefix(decoder, nonmaximal)

        result = state.choices_with_stats()

        self.assertEqual(result.choices, ())
        self.assertEqual(result.stats.eos_completions_seen, 0)
        self.assertEqual(result.stats.candidate_state_size.continuation_count, 0)
        self.assertEqual(result.stats.retained_state_size.continuation_count, 0)

    def test_support_maximal_modes_agree_on_nonmaximal_directional_prefix(self) -> None:
        facts = four_substituent_directional_facts()
        nonmaximal = _hard_only_directional_witness(facts)
        modes = (
            OnlineDecoderExecutionMode.PREFIX_REPLAY,
            OnlineDecoderExecutionMode.CACHED_COMPLETIONS,
            OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
        )

        frontiers = tuple(
            _choice_signature(
                _state_for_prefix(
                    _decoder_for_mode(facts, mode, include_eos=True),
                    nonmaximal,
                ).choices()
            )
            for mode in modes
        )

        self.assertEqual(frontiers[0], ())
        self.assertEqual(frontiers[0], frontiers[1])
        self.assertEqual(frontiers[0], frontiers[2])

    def test_support_maximal_ring_case_does_not_drop_maximal_candidate(self) -> None:
        facts = ordinary_molecule_facts_from_smiles("C/C=C/1CC1")
        residual = _residual_determinized_decoder(facts, include_eos=True)
        witness = _witnesses(facts)[0]

        state = _walk_decoder(residual, _tokens_for_witness(residual, witness))

        self.assertTrue(any(choice.is_eos for choice in state.choices()))

    def test_spec_mentions_cached_completion_not_true_residual_continuation(self) -> None:
        text = SPEC_PATH.read_text(encoding="utf-8")

        self.assertIn("CACHED_COMPLETIONS", text)
        self.assertIn("not yet a true residual DFS continuation", text)
        self.assertIn("RESIDUAL_CONTINUATIONS", text)
        self.assertIn("stores suspended", text)
        self.assertNotIn("RESUMABLE_CONTINUATIONS` execution mode stores suspended", text)

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

    def _assert_residual_texts_match_replay(self, facts) -> None:
        replay = _replay_determinized_decoder(facts)
        residual = _residual_determinized_decoder(facts)

        self.assertEqual(
            _choice_texts(replay.initial_state().choices()),
            _choice_texts(residual.initial_state().choices()),
        )

    def _assert_all_modes_walk_same_witnesses(self, facts) -> None:
        modes = (
            OnlineDecoderExecutionMode.PREFIX_REPLAY,
            OnlineDecoderExecutionMode.CACHED_COMPLETIONS,
            OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
        )
        for witness in tuple(sorted(set(_witnesses(facts))))[:3]:
            frontiers = []
            for mode in modes:
                decoder = _decoder_for_mode(facts, mode, include_eos=True)
                state = _walk_decoder(decoder, _tokens_for_witness(decoder, witness))
                frontiers.append(_choice_texts(state.choices()))
            self.assertEqual(frontiers[0], frontiers[1])
            self.assertEqual(frontiers[0], frontiers[2])


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
        execution_mode=OnlineDecoderExecutionMode.CACHED_COMPLETIONS,
    )


def _continuation_branch_decoder(facts):
    return make_branch_preserving_online_decoder(
        facts=facts,
        policy=ordinary_policy_for_facts(facts),
        semantics=OrdinarySmilesSemantics(),
        execution_mode=OnlineDecoderExecutionMode.CACHED_COMPLETIONS,
    )


def _residual_determinized_decoder(facts, *, include_eos: bool = False):
    return make_determinized_online_decoder(
        facts=facts,
        policy=ordinary_policy_for_facts(facts),
        semantics=OrdinarySmilesSemantics(),
        include_eos=include_eos,
        execution_mode=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
    )


def _residual_branch_decoder(facts):
    return make_branch_preserving_online_decoder(
        facts=facts,
        policy=ordinary_policy_for_facts(facts),
        semantics=OrdinarySmilesSemantics(),
        execution_mode=OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
    )


def _decoder_for_mode(facts, mode: OnlineDecoderExecutionMode, *, include_eos: bool = False):
    return make_determinized_online_decoder(
        facts=facts,
        policy=ordinary_policy_for_facts(facts),
        semantics=OrdinarySmilesSemantics(),
        include_eos=include_eos,
        execution_mode=mode,
    )


def _first_residual_continuation(facts):
    choice = _residual_determinized_decoder(facts).initial_state().choices()[0]
    if choice.next_state is None or choice.next_state.raw_state.frontier is None:
        raise AssertionError("residual choice lacks continuation frontier")
    return choice.next_state.raw_state.frontier.continuations[0]


def _residual_state_for_continuations(decoder, prefix: str, continuations):
    raw = OnlineResidualDecoderState(
        prefix=prefix,
        frontier=OnlineResidualContinuationFrontier(
            prefix=prefix,
            continuations=tuple(continuations),
        ),
    )
    state_type = type(decoder.initial_state())
    return state_type(prefix=prefix, raw_state=raw, decoder=decoder)


def _sink_with_providers(*, required_prefix: str) -> ResidualFrontierSink:
    sink = ResidualFrontierSink(required_prefix=required_prefix)
    sink.snapshot_provider = lambda: _snapshot("pending")
    sink.decision_path_provider = lambda: _path("pending")
    return sink


def _continuation(
    prefix: str,
    token_text: str,
    tag: str,
    *,
    completion_count: int = 1,
) -> OnlineResidualContinuation:
    return OnlineResidualContinuation(
        prefix=prefix,
        snapshot=_snapshot(tag),
        frontier_path=_path(tag),
        token_text=token_text,
        completion_count=completion_count,
    )


def _snapshot(tag: str) -> OnlineSearchSnapshot:
    return OnlineSearchSnapshot(
        traversal_state=(tag, "traversal"),
        residual_snapshot=(tag, "residual"),
        ring_state=(tag, "ring"),
        output_snapshot=(tag, "output"),
        decision_snapshot=_path(tag),
        frame_stack=(OnlineSearchFrame("render-resume", (tag,)),),
    )


def _path(tag: str) -> OnlineDecisionPath:
    return OnlineDecisionPath((OnlineDecision("test", (tag,)),))


def _state_for_prefix(decoder, prefix: str):
    state = decoder.initial_state()
    raw_state = type(state.raw_state)(prefix=prefix)
    return type(state)(prefix=prefix, raw_state=raw_state, decoder=decoder)


def _choice_texts(choices) -> tuple[str, ...]:
    return tuple(choice.text for choice in choices)


def _choice_signature(choices) -> tuple[tuple[str, int, int, bool], ...]:
    return tuple(
        (choice.text, choice.multiplicity, choice.completion_count, choice.is_eos)
        for choice in choices
    )


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


def _hard_only_directional_witness(facts) -> str:
    hard_policy = ordinary_policy_for_facts(
        facts,
        OrdinaryPolicyOptions(annotation_mode=AnnotationMode.HARD),
    )
    support_policy = ordinary_policy_for_facts(facts)
    hard = set(
        iter_online_stereo_witness_strings(
            facts=facts,
            policy=hard_policy,
            semantics=OrdinarySmilesSemantics(),
        )
    )
    support_maximal = set(
        iter_online_stereo_witness_strings(
            facts=facts,
            policy=support_policy,
            semantics=OrdinarySmilesSemantics(),
        )
    )
    hard_only = tuple(sorted(hard - support_maximal))
    if not hard_only:
        raise AssertionError("directional fixture lacks a HARD-only witness")
    return hard_only[0]


if __name__ == "__main__":
    unittest.main()
