"""Exhaustive rich support-artifact diagnostics.

Directional ring opening/pair semantics are bounded by replay-addressed
branch-transition artifact tests. This module exercises the stronger,
exhaustive rich support-artifact representation and is intentionally diagnostic.
"""

from __future__ import annotations

import os
import unittest

from tests.south_star1.writer_support_artifact_domain_methods import *

from tests.south_star1.writer_support_artifact_directional_slow_fixtures import (
    directional_ring_opening_slow_fixture,
    directional_ring_pair_slow_fixture,
)


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
                    branch, manifest = _ring_pair_branch_and_manifest(artifact)
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
                ("compatible_seconds", _forge_ring_compatible_seconds),
                ("domain_intersection", _forge_ring_domain_intersection),
                ("carrier_orientation", _forge_ring_carrier_orientation),
                ("event_mark_detached", _forge_ring_term_mark),
                ("false_noop", _forge_ring_false_noop),
                ("false_change", _forge_ring_false_change),
                ("factor_discharge", _forge_ring_factor_discharge),
                ("snapshot_detached", _forge_ring_source_snapshot),
                ("successor_open_endpoint", _forge_ring_successor_open_endpoint),
                ("bond_occurrence_added", _forge_ring_bond_occurrence_added),
                ("missing_term", _forge_ring_missing_term),
                ("lifecycle_operation", _forge_ring_lifecycle_operation),
            )
            for name, mutate in cases:
                with self.subTest(name=name):
                    artifact = deepcopy(original)
                    mutate(artifact)
                    _assert_structural_checker_accepts(self, artifact)
                    verification = verify_writer_support_artifact_for_facts(
                        facts=facts,
                        runtime_options=options,
                        artifact=artifact,
                    )
                    self.assertFalse(verification.accepted)

    def test_directional_ring_pair_coherent_term_forgeries_are_rejected(self) -> None:
            cases = (
                ("missing_term", _forge_ring_pair_missing_term, "directional_ring_pair_transition_missing"),
                ("compatible_choices", _forge_ring_pair_compatible_choices, "directional_ring_pair_compatible_choices_mismatch"),
                ("first_mark", _forge_ring_pair_first_mark, "directional_ring_pair_event_first_endpoint_direction_mark_mismatch"),
                ("second_mark", _forge_ring_pair_second_mark, "directional_ring_pair_event_direction_mark_mismatch"),
                ("orientation", _forge_ring_pair_orientation, "directional_ring_pair_canonical_orientation_mismatch"),
                ("carrier", _forge_ring_pair_carrier, "directional_ring_pair_carrier_model_mismatch"),
                ("restriction", _forge_ring_pair_restriction, "directional_ring_pair_restriction_mismatch"),
                ("occurrence", _forge_ring_pair_occurrence, "directional_ring_pair_bond_occurrence_mismatch"),
                ("discharge", _forge_ring_pair_discharge, "directional_ring_pair_discharge_factor_mismatch"),
                ("successor", _forge_ring_pair_successor, "directional_ring_pair_successor_state_anchor_mismatch"),
            )
            facts, options, original = directional_ring_pair_slow_fixture(DirectionMark.ABSENT)
            for name, mutate, reason in cases:
                with self.subTest(name=name):
                    artifact = deepcopy(original)
                    mutate(artifact)
                    _assert_structural_checker_accepts(self, artifact)
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
            if os.environ.get(RUN_SLOW_ENV) != "1":
                self.skipTest(
                    f"set {RUN_SLOW_ENV}=1 to run the directional ring carrier artifact probe"
                )
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
            classification = _obligation_classification(artifact, facts=facts)

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
            branch = _first_graph_ring_delta_branch(artifact, "ring_endpoint_open")
            event = _first_graph_ring_delta_event(branch, "ring_endpoint_emitted")
            self.assertEqual(event["bond"], 3)
            self.assertLessEqual(
                artifact["metrics"]["largest_object_identity_input_bytes"],
                budget.max_digest_term_bytes,
            )

    def test_non_single_directional_ring_root_zero_artifact_replays_completely(
            self,
        ) -> None:
            if os.environ.get(RUN_SLOW_ENV) != "1":
                self.skipTest(
                    f"set {RUN_SLOW_ENV}=1 to run the non-single directional ring artifact probe"
                )
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
