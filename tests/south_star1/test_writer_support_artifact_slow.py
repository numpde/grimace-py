"""Exhaustive rich support-artifact diagnostics.

Directional ring opening/pair semantics are bounded by replay-addressed
branch-transition artifact tests. This module exercises the stronger,
exhaustive rich support-artifact representation and is intentionally diagnostic.
"""

import unittest
import os
from copy import deepcopy
from grimace._south_star1.policy import DirectionMark
from grimace._south_star1.writer_envelope_work import WriterEnvelopeWorkBudget
from grimace._south_star1.writer_support_artifact_checker import verify_writer_support_artifact_consistency
from grimace._south_star1.writer_support_artifact_envelope import verify_writer_support_artifact_envelope
from grimace._south_star1.writer_support_artifact_envelope import writer_support_artifact_envelope_for_snapshot
from grimace._south_star1.writer_support_artifact_fact_verifier import verify_writer_support_artifact_for_facts
from tests.south_star1.writer_test_fixtures import directional_non_single_ring_carrier_facts
from tests.south_star1.writer_test_fixtures import directional_ring_carrier_facts
from tests.south_star1.writer_artifact_test_support import closed_term_field
from tests.south_star1.writer_support_artifact_directional_slow_fixtures import directional_ring_opening_slow_fixture
from tests.south_star1.writer_support_artifact_directional_slow_fixtures import directional_ring_pair_slow_fixture
from tests.south_star1.writer_support_artifact_directional_slow_test_support import forge_ring_bond_occurrence_added, forge_ring_carrier_orientation, forge_ring_compatible_seconds, forge_ring_domain_intersection, forge_ring_factor_discharge, forge_ring_false_change, forge_ring_false_noop, forge_ring_lifecycle_operation, forge_ring_missing_term, forge_ring_pair_carrier, forge_ring_pair_compatible_choices, forge_ring_pair_discharge, forge_ring_pair_first_mark, forge_ring_pair_missing_term, forge_ring_pair_occurrence, forge_ring_pair_orientation, forge_ring_pair_restriction, forge_ring_pair_second_mark, forge_ring_pair_successor, forge_ring_source_snapshot, forge_ring_successor_open_endpoint, forge_ring_term_mark, ring_pair_branch_and_manifest
from tests.south_star1.writer_support_artifact_queries import classify_obligation_replay
from tests.south_star1.writer_support_artifact_queries import first_graph_ring_delta_branch
from tests.south_star1.writer_support_artifact_queries import first_graph_ring_delta_event
from tests.south_star1.writer_support_artifact_queries import require_structurally_valid_support_artifact
from tests.south_star1.writer_test_context import initial_writer_snapshot
from tests.south_star1.writer_test_context import prepare_writer_facts
from tests.south_star1.writer_test_context import writer_runtime_options

class WriterSupportArtifactSlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if os.environ.get("SOUTH_STAR1_RUN_SLOW") != "1":
            raise unittest.SkipTest(
                "set SOUTH_STAR1_RUN_SLOW=1 to run exhaustive rich support-artifact diagnostics"
            )

    def test_reduced_directional_ring_opening_artifact_replays_semantically(self) -> None:
            facts, options, artifact = directional_ring_opening_slow_fixture()

            structural = verify_writer_support_artifact_consistency(artifact)
            live = verify_writer_support_artifact_envelope(
                prepared=prepare_writer_facts(facts),
                envelope=artifact,
            )
            verification = verify_writer_support_artifact_for_facts(
                facts=facts,
                runtime_options=options,
                artifact=artifact,
            )

            self.assertTrue(structural.accepted, structural.reason)
            self.assertTrue(live.accepted, live.reason)
            self.assertTrue(verification.accepted, verification.reason)
            manifests = [
                manifest
                for obj in artifact["objects"]
                if obj["kind"] == "branch_support"
                for manifest in obj["payload"]["obligation_manifests"]["residual_work"]
                if manifest["operation"] == "directional ring endpoint projection"
            ]
            self.assertTrue(manifests)
            snapshots = [
                (
                    closed_term_field(manifest["transition_term"], "source_snapshot"),
                    closed_term_field(manifest["transition_term"], "successor_snapshot"),
                )
                for manifest in manifests
            ]
            self.assertTrue(any(source == successor for source, successor in snapshots))
            self.assertTrue(any(source != successor for source, successor in snapshots))

    def test_reduced_directional_ring_pair_artifacts_replay_semantically(self) -> None:
            for first_mark in (DirectionMark.ABSENT, DirectionMark.FWD):
                with self.subTest(first_mark=first_mark):
                    facts, options, artifact = directional_ring_pair_slow_fixture(first_mark)
                    structural = verify_writer_support_artifact_consistency(artifact)
                    live = verify_writer_support_artifact_envelope(
                        prepared=prepare_writer_facts(facts),
                        envelope=artifact,
                    )
                    verification = verify_writer_support_artifact_for_facts(
                        facts=facts,
                        runtime_options=options,
                        artifact=artifact,
                    )

                    self.assertTrue(structural.accepted, structural.reason)
                    self.assertTrue(live.accepted, live.reason)
                    self.assertTrue(verification.accepted, verification.reason)
                    self.assertTrue(verification.offline_replay_complete)
                    self.assertEqual(verification.offline_unchecked_obligation_families, ())
                    branch, manifest = ring_pair_branch_and_manifest(artifact)
                    self.assertEqual(
                        closed_term_field(manifest["transition_term"], "first_endpoint_direction_mark")["value"],
                        first_mark.value,
                    )
                    self.assertEqual(
                        branch["payload"]["graph_ring_delta"]["kind"],
                        "ring_endpoint_pair",
                    )

    def test_directional_ring_opening_coherent_term_forgeries_are_rejected(self) -> None:
            facts, options, original = directional_ring_opening_slow_fixture()
            cases = (
                ("compatible_seconds", forge_ring_compatible_seconds),
                ("domain_intersection", forge_ring_domain_intersection),
                ("carrier_orientation", forge_ring_carrier_orientation),
                ("event_mark_detached", forge_ring_term_mark),
                ("false_noop", forge_ring_false_noop),
                ("false_change", forge_ring_false_change),
                ("factor_discharge", forge_ring_factor_discharge),
                ("snapshot_detached", forge_ring_source_snapshot),
                ("successor_open_endpoint", forge_ring_successor_open_endpoint),
                ("bond_occurrence_added", forge_ring_bond_occurrence_added),
                ("missing_term", forge_ring_missing_term),
                ("lifecycle_operation", forge_ring_lifecycle_operation),
            )
            for name, mutate in cases:
                with self.subTest(name=name):
                    artifact = deepcopy(original)
                    mutate(artifact)
                    require_structurally_valid_support_artifact(artifact)
                    verification = verify_writer_support_artifact_for_facts(
                        facts=facts,
                        runtime_options=options,
                        artifact=artifact,
                    )
                    self.assertFalse(verification.accepted)

    def test_directional_ring_pair_coherent_term_forgeries_are_rejected(self) -> None:
            cases = (
                ("missing_term", forge_ring_pair_missing_term, "directional_ring_pair_transition_missing"),
                ("compatible_choices", forge_ring_pair_compatible_choices, "directional_ring_pair_compatible_choices_mismatch"),
                ("first_mark", forge_ring_pair_first_mark, "directional_ring_pair_event_first_endpoint_direction_mark_mismatch"),
                ("second_mark", forge_ring_pair_second_mark, "directional_ring_pair_event_direction_mark_mismatch"),
                ("orientation", forge_ring_pair_orientation, "directional_ring_pair_canonical_orientation_mismatch"),
                ("carrier", forge_ring_pair_carrier, "directional_ring_pair_carrier_model_mismatch"),
                ("restriction", forge_ring_pair_restriction, "directional_ring_pair_restriction_mismatch"),
                ("occurrence", forge_ring_pair_occurrence, "directional_ring_pair_bond_occurrence_mismatch"),
                ("discharge", forge_ring_pair_discharge, "directional_ring_pair_discharge_factor_mismatch"),
                ("successor", forge_ring_pair_successor, "directional_ring_pair_successor_state_anchor_mismatch"),
            )
            facts, options, original = directional_ring_pair_slow_fixture(DirectionMark.ABSENT)
            for name, mutate, reason in cases:
                with self.subTest(name=name):
                    artifact = deepcopy(original)
                    mutate(artifact)
                    require_structurally_valid_support_artifact(artifact)
                    verification = verify_writer_support_artifact_for_facts(
                        facts=facts,
                        runtime_options=options,
                        artifact=artifact,
                    )
                    self.assertFalse(verification.accepted)
                    self.assertIn(reason, verification.reason)

    def test_directional_ring_carrier_root_zero_artifact_builds_with_default_budget(
            self,
        ) -> None:
            facts = directional_ring_carrier_facts()
            options = writer_runtime_options(rooted_at_atom=0)
            prepared = prepare_writer_facts(facts)
            budget = WriterEnvelopeWorkBudget()

            self.assertEqual(budget.max_digest_term_bytes, 25_000_000)

            artifact = writer_support_artifact_envelope_for_snapshot(
                prepared=prepared,
                snapshot=initial_writer_snapshot(prepared, options),
                budget=budget,
            )
            structural = verify_writer_support_artifact_consistency(
                artifact,
                budget=budget,
            )
            live = verify_writer_support_artifact_envelope(
                prepared=prepared,
                envelope=artifact,
                budget=budget,
            )
            verification = verify_writer_support_artifact_for_facts(
                facts=facts,
                runtime_options=options,
                artifact=artifact,
                budget=budget,
            )
            classification = classify_obligation_replay(facts=facts, artifact=artifact)

            self.assertTrue(structural.accepted, structural.reason)
            self.assertTrue(live.accepted, live.reason)
            self.assertTrue(verification.accepted, verification.reason)
            self.assertTrue(
                verification.offline_replay_complete,
                verification.offline_unchecked_obligation_families,
            )
            self.assertEqual(verification.offline_unchecked_obligation_families, ())
            self.assertIn(
                "stereo_lifecycle",
                verification.offline_checked_obligation_families,
            )
            self.assertTrue(classification.accepted, classification.reason)
            branch = first_graph_ring_delta_branch(artifact, "ring_endpoint_open")
            event = first_graph_ring_delta_event(branch, "ring_endpoint_emitted")
            self.assertEqual(event["bond"], 3)
            self.assertLessEqual(
                artifact["metrics"]["largest_object_identity_input_bytes"],
                budget.max_digest_term_bytes,
            )

    def test_non_single_directional_ring_root_zero_artifact_replays_completely(
            self,
        ) -> None:
            facts = directional_non_single_ring_carrier_facts()
            options = writer_runtime_options(rooted_at_atom=0)
            prepared = prepare_writer_facts(facts)
            artifact = writer_support_artifact_envelope_for_snapshot(
                prepared=prepared,
                snapshot=initial_writer_snapshot(prepared, options),
            )
            structural = verify_writer_support_artifact_consistency(artifact)
            live = verify_writer_support_artifact_envelope(
                prepared=prepared,
                envelope=artifact,
            )
            verification = verify_writer_support_artifact_for_facts(
                facts=facts,
                runtime_options=options,
                artifact=artifact,
            )

            self.assertTrue(structural.accepted, structural.reason)
            self.assertTrue(live.accepted, live.reason)
            self.assertTrue(verification.accepted, verification.reason)
            self.assertEqual(
                (verification.support_count, verification.witness_count),
                (72, 72),
            )
            self.assertTrue(verification.offline_replay_complete)
            self.assertEqual(verification.offline_unchecked_obligation_families, ())
            self.assertIn(
                "directional_ring_closure_lifecycle",
                verification.offline_checked_obligation_families,
            )
