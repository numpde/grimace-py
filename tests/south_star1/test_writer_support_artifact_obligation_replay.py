"""Physically owned rich support-artifact obligation-replay contracts.Facts-bound writer support artifact verifier tests."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
import os
from types import SimpleNamespace
import unittest

from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.errors import SouthStarErrorKind
from grimace._south_star1.facts import LigandKind
from grimace._south_star1.facts import LigandOccurrence
from grimace._south_star1.facts import SiteStatus
from grimace._south_star1.facts import StereoFacts
from grimace._south_star1.facts import TetraValue
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import OccurrenceId
from grimace._south_star1.ids import SiteId
from grimace._south_star1.ordinary_policy import OrdinaryPolicyOptions
from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.policy import DirectionMark
from grimace._south_star1.rdkit_adapter import RdkitOrdinaryExtractionOptions
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_smiles
from grimace._south_star1.writer_envelope_terms import _digest_terms_bounded
from grimace._south_star1.writer_envelope_terms import _identity_digest
from grimace._south_star1.writer_envelope_work import WriterEnvelopeWorkBudget
from grimace._south_star1.writer_events import WriterRingEndpointEmitted
from grimace._south_star1.writer_events import WriterRingEndpointPaired
from grimace._south_star1.writer_frontier import _checked_writer_frontier_branch_supports
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_snapshot import WriterDecoderBoundary
from grimace._south_star1.writer_snapshot import capture_writer_frontier_snapshot
from grimace._south_star1.writer_snapshot_prefix_envelope import (
    writer_snapshot_prefix_read_envelope_for_emitted_texts,
)
import grimace._south_star1.writer_stereo as writer_stereo_module
import grimace._south_star1.writer_support_artifact_offline_verifier as offline_verifier_module
from grimace._south_star1.writer_support_artifact_checker import (
    verify_writer_support_artifact_consistency,
)
from grimace._south_star1.writer_support_artifact_fact_verifier import (
    OBJECT_KIND_OFFLINE_COVERAGE,
)
from grimace._south_star1.writer_support_artifact_fact_verifier import (
    verify_writer_support_artifact_for_facts,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    classify_residual_stereo_obligations_offline,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    validate_writer_bracket_atom_text_against_facts,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    verify_branch_projection_identities_offline,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    verify_count_dag_arithmetic,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    verify_graph_ring_branch_deltas_offline,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    verify_local_branch_successor_evidence_offline,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    verify_support_image_coverage_offline,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    verify_support_string_replay_paths_offline,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    verify_terminal_support_identities_offline,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    verify_writer_support_artifact_offline_replay,
)
from grimace._south_star1.writer_support_artifact_envelope import (
    verify_writer_support_artifact_envelope,
)
from grimace._south_star1.writer_support_artifact_envelope import (
    writer_support_artifact_envelope_for_prefix_read,
)
from grimace._south_star1.writer_support_artifact_envelope import (
    writer_support_artifact_envelope_for_snapshot,
)
from tests.south_star1.helpers import cco_facts
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import shared_acyclic_directional_facts
from tests.south_star1.writer_artifact_resealing import reseal_support_artifact
from tests.south_star1.writer_artifact_test_support import artifact_object_by_id
from tests.south_star1.writer_artifact_test_support import closed_term_digest
from tests.south_star1.writer_artifact_test_support import closed_term_field
from tests.south_star1.writer_artifact_test_support import set_closed_term_field
from tests.south_star1.writer_artifact_test_support import refresh_cursor_digest
from tests.south_star1.writer_artifact_test_support import refresh_closed_term_digest_field
from tests.south_star1.writer_artifact_test_support import refresh_kind_manifest_digest
from tests.south_star1.writer_test_context import initial_writer_snapshot
from tests.south_star1.writer_test_context import prepare_writer_facts
from tests.south_star1.writer_test_context import writer_runtime_options
from tests.south_star1.helpers import tetrahedral_facts
from tests.south_star1.writer_test_fixtures import (
    directional_non_single_ring_carrier_facts,
)
from tests.south_star1.writer_test_fixtures import (
    directional_ring_carrier_facts,
)
from tests.south_star1.writer_test_fixtures import (
    shared_directional_ring_carrier_facts,
)
from tests.south_star1.helpers import two_atom_facts

RUN_SLOW_ENV = "SOUTH_STAR1_RUN_SLOW"



from tests.south_star1.writer_support_artifact_fixtures import completed_prefix_support_artifact_fixture, rdkit_graph_facts, rdkit_support_artifact_fixture, rdkit_support_artifact_verification, support_artifact_fixture, tetra_support_artifact_fixture
from tests.south_star1.writer_support_artifact_queries import (
    classify_obligation_replay, coverage_object, cursor_state_by_digest, first_branch_support_object,
    first_graph_ring_delta_branch, first_graph_ring_delta_event, first_local_evidence, first_residual_work_branch,
    first_support_string_object, first_terminal_projection_object, first_terminal_support_object, first_text_projection_object,
    require_structurally_valid_support_artifact, single_cursor_state, support_strings, text_projection_for_branch,
    verify_branch_projection_relation, verify_graph_ring_delta_relation, verify_local_branch_evidence_relation,
    verify_support_image_coverage_relation, verify_support_string_replay_relation, verify_terminal_identity_relation,
)
from tests.south_star1.writer_support_artifact_transition_test_support import (linked_tetra_lifecycle_manifest, refresh_linked_raw_lifecycle_residual_digest, text_projection_identity_digest, propagate_text_projection_cursor_change)
from tests.south_star1.writer_support_artifact_graph_test_support import (first_directional_bond_delta_branch, first_closure_evidence_item, tetra_facts_with_implicit_h_only_outside_specified_site)
from tests.south_star1.writer_support_artifact_tetra_test_support import (append_unrelated_raw_lifecycle, different_local_order_digest, refresh_local_order_event_identity_digest)
from tests.south_star1.writer_support_artifact_directional_test_support import (directional_transition_branch_and_manifest, directional_transition_manifest, directional_discharge_key_pairs, bond_occurrence_terms_for_branch, mutate_directional_restriction_sign, mutate_directional_canonical_orientation, mutate_directional_model_field, remove_directional_model, remove_directional_restriction, duplicate_directional_model_site, mutate_directional_successor_snapshot, set_directional_discharges, set_directional_discharges_by_keys, remove_raw_lifecycle_capability, mutate_directional_term_mark, mutate_directional_term_bond, remove_directional_successor_bond_occurrence, duplicate_directional_successor_bond_occurrence, mutate_directional_successor_snapshot_unrelated)

class WriterSupportArtifactObligationReplayTest(unittest.TestCase):
    def test_branch_local_ledger_rejects_transplanted_graph_manifest(self) -> None:
        facts = rdkit_graph_facts("CCO")
        artifact = deepcopy(rdkit_support_artifact_fixture("CCO").artifact)
        branches = [item for item in artifact["objects"] if item["kind"] == "branch_support"]
        source = branches[0]
        target = next(
            branch
            for branch in branches[1:]
            if branch["payload"]["source_state_digest"]
            != source["payload"]["source_state_digest"]
        )
        target_manifest = target["payload"]["obligation_manifests"][
            "graph_obligation_work"
        ][0]
        source["payload"]["obligation_manifests"]["graph_obligation_work"] = [
            deepcopy(target_manifest)
        ]
        for name in ("is_noop", "is_empty", "is_discharged", "terminal_clean"):
            source["payload"]["obligation_manifests"]["graph_obligation_work"][0][
                name
            ] = True
        reseal_support_artifact(artifact)

        structural = verify_writer_support_artifact_consistency(artifact)
        verification = verify_writer_support_artifact_for_facts(
            facts=facts,
            runtime_options=writer_runtime_options(),
            artifact=artifact,
        )

        self.assertTrue(structural.accepted, structural.reason)
        self.assertFalse(verification.accepted)
        self.assertIn("graph_obligation_work_identity_mismatch", verification.reason)

    def test_default_corpus_obligations_are_classified(self) -> None:
        cases = {
            "CCO": (),
            "CC(C)O": (),
            "C1CC1": (),
            "C1CCC1": (),
            "C1=CC1": (),
            "C1#CC1": (),
            "[NH4+]": (),
            "[13CH4]": (),
        }
        for smiles, unchecked_families in cases.items():
            with self.subTest(smiles=smiles):
                facts = rdkit_graph_facts(smiles)
                artifact = rdkit_support_artifact_fixture(smiles).artifact

                classification = classify_obligation_replay(facts=facts, artifact=artifact)
                verification = verify_writer_support_artifact_for_facts(
                    facts=facts,
                    runtime_options=writer_runtime_options(),
                    artifact=artifact,
                )

                self.assertTrue(classification.accepted, classification.reason)
                self.assertEqual(classification.unchecked_families, unchecked_families)
                self.assertTrue(classification.stereo_obligations_present)
                self.assertEqual(
                    classification.graph_obligations_present,
                    True,
                )
                self.assertEqual(
                    classification.residual_obligations_present,
                    smiles.startswith("C1"),
                )
                self.assertIn("stereo_lifecycle", classification.checked_families)
                self.assertIn(
                    "residual_work_checked_empty",
                    classification.checked_empty_families,
                )
                self.assertTrue(verification.accepted, verification.reason)
                self.assertEqual(
                    verification.offline_replay_complete,
                    not unchecked_families,
                )
                self.assertEqual(verification.offline_unchecked_object_kinds, ())
                self.assertEqual(
                    verification.offline_unchecked_obligation_families,
                    unchecked_families,
                )
                self.assertIn(
                    "stereo_lifecycle",
                    verification.offline_checked_obligation_families,
                )
                self.assertIn(
                    "terminal_graph_obligation_work",
                    verification.offline_checked_obligation_families,
                )
                self.assertIn(
                    "terminal_stereo_lifecycle",
                    verification.offline_checked_obligation_families,
                )
                self.assertIn(
                    "residual_work_checked_empty",
                    verification.offline_empty_obligation_families,
                )

    def test_descriptive_flags_cannot_credit_forged_graph_work(self) -> None:
        facts = rdkit_graph_facts("CCO")
        artifact = deepcopy(rdkit_support_artifact_fixture("CCO").artifact)
        branch = first_branch_support_object(artifact)
        manifest = branch["payload"]["obligation_manifests"][
            "graph_obligation_work"
        ][0]
        manifest["operation"] = "forged graph obligation context"
        for name in ("is_noop", "is_empty", "is_discharged", "terminal_clean"):
            manifest[name] = True
        reseal_support_artifact(artifact)

        structural = verify_writer_support_artifact_consistency(artifact)
        verification = verify_writer_support_artifact_for_facts(
            facts=facts,
            runtime_options=writer_runtime_options(),
            artifact=artifact,
        )

        self.assertTrue(structural.accepted, structural.reason)
        self.assertFalse(verification.accepted)
        self.assertIn("graph_obligation_work_operation_mismatch", verification.reason)

    def test_obligation_summary_mutation_is_structurally_rejected(self) -> None:
        artifact = rdkit_support_artifact_fixture("CCO").artifact
        branch = first_branch_support_object(artifact)
        branch["payload"]["obligation_summary"]["stereo_lifecycle_count"] += 1

        verification = verify_writer_support_artifact_for_facts(
            facts=rdkit_graph_facts("CCO"),
            runtime_options=writer_runtime_options(),
            artifact=artifact,
        )

        self.assertFalse(verification.accepted)
        self.assertIn("obligation_manifest_count_mismatch", verification.reason)

    def test_ring_finite_relation_and_graph_obligation_are_checked(self) -> None:
        for smiles in ("C1CC1", "C1CCC1", "C1=CC1", "C1#CC1"):
            with self.subTest(smiles=smiles):
                facts = rdkit_graph_facts(smiles)
                artifact = rdkit_support_artifact_fixture(smiles).artifact
                classification = classify_obligation_replay(facts=facts, artifact=artifact)
                verification = verify_writer_support_artifact_for_facts(
                    facts=facts,
                    runtime_options=writer_runtime_options(),
                    artifact=artifact,
                )

                self.assertTrue(classification.accepted, classification.reason)
                self.assertNotIn(
                    "finite_relation_work",
                    classification.unchecked_families,
                )
                self.assertNotIn(
                    "graph_obligation_work",
                    classification.unchecked_families,
                )
                self.assertIn("finite_relation_work", classification.checked_families)
                self.assertIn("graph_obligation_work", classification.checked_families)
                self.assertTrue(verification.accepted, verification.reason)
                self.assertTrue(verification.offline_replay_complete)

    def test_ring_obligation_cross_link_mutations_are_rejected(self) -> None:
        facts = rdkit_graph_facts("C1=CC1")
        artifact = rdkit_support_artifact_fixture("C1=CC1").artifact
        branch = first_graph_ring_delta_branch(
            artifact,
            "ring_endpoint_pair_non_single",
        )
        graph = branch["payload"]["obligation_manifests"]["graph_obligation_work"][0]
        graph["ring_summary"]["bond"] = "wrong"

        wrong_bond = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(wrong_bond.accepted)
        self.assertIn("ring_obligation_bond_mismatch", wrong_bond.reason)

        artifact = rdkit_support_artifact_fixture("C1=CC1").artifact
        branch = first_graph_ring_delta_branch(
            artifact,
            "ring_endpoint_pair_non_single",
        )
        finite = branch["payload"]["obligation_manifests"]["finite_relation_work"][0]
        finite["operation"] = "unknown closure operation"

        wrong_operation = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(wrong_operation.accepted)
        self.assertIn("ring_obligation_operation_mismatch", wrong_operation.reason)

        artifact = rdkit_support_artifact_fixture("C1=CC1").artifact
        branch = first_graph_ring_delta_branch(
            artifact,
            "ring_endpoint_pair_non_single",
        )
        graph = branch["payload"]["obligation_manifests"]["graph_obligation_work"][0]
        graph["ring_summary"]["marker"] = "#"

        wrong_marker = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(wrong_marker.accepted)
        self.assertIn("ring_obligation_marker_mismatch", wrong_marker.reason)

    def test_ring_obligation_manifest_count_mismatch_is_rejected(self) -> None:
        artifact = rdkit_support_artifact_fixture("C1=CC1").artifact
        branch = first_graph_ring_delta_branch(
            artifact,
            "ring_endpoint_pair_non_single",
        )
        branch["payload"]["obligation_summary"]["finite_relation_work_count"] += 1

        verification = verify_writer_support_artifact_for_facts(
            facts=rdkit_graph_facts("C1=CC1"),
            runtime_options=writer_runtime_options(),
            artifact=artifact,
        )

        self.assertFalse(verification.accepted)
        self.assertIn("obligation_manifest_count_mismatch", verification.reason)

    def test_ring_summary_and_flags_cannot_credit_forged_relation_work(self) -> None:
        facts = rdkit_graph_facts("C1CC1")
        artifact = deepcopy(rdkit_support_artifact_fixture("C1CC1").artifact)
        branch = next(
            item
            for item in artifact["objects"]
            if item["kind"] == "branch_support"
            and item["payload"]["obligation_manifests"]["finite_relation_work"]
        )
        manifest = branch["payload"]["obligation_manifests"][
            "finite_relation_work"
        ][0]
        manifest["operation"] = "forged closure relation"
        for name in ("is_noop", "is_empty", "is_discharged", "terminal_clean"):
            manifest[name] = True
        manifest["ring_summary"] = deepcopy(manifest["ring_summary"])
        reseal_support_artifact(artifact)

        structural = verify_writer_support_artifact_consistency(artifact)
        verification = verify_writer_support_artifact_for_facts(
            facts=facts,
            runtime_options=writer_runtime_options(),
            artifact=artifact,
        )

        self.assertTrue(structural.accepted, structural.reason)
        self.assertFalse(verification.accepted)
        self.assertIn("ring_obligation_operation_mismatch", verification.reason)

    def test_ring_summary_flags_do_not_control_replay_credit(self) -> None:
        facts = rdkit_graph_facts("C1=CC1")
        artifact = rdkit_support_artifact_fixture("C1=CC1").artifact
        branch = first_graph_ring_delta_branch(
            artifact,
            "ring_endpoint_pair_non_single",
        )
        finite = branch["payload"]["obligation_manifests"]["finite_relation_work"][0]
        finite["ring_summary"]["is_exact"] = False

        not_exact = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertTrue(not_exact.accepted, not_exact.reason)
        self.assertNotIn("finite_relation_work", not_exact.unchecked_families)

        artifact = rdkit_support_artifact_fixture("C1=CC1").artifact
        branch = first_graph_ring_delta_branch(
            artifact,
            "ring_endpoint_pair_non_single",
        )
        graph = branch["payload"]["obligation_manifests"]["graph_obligation_work"][0]
        graph["ring_summary"]["is_complete"] = False

        not_complete = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertTrue(not_complete.accepted, not_complete.reason)
        self.assertNotIn("graph_obligation_work", not_complete.unchecked_families)

    def test_stereo_lifecycle_flags_do_not_control_replay_credit(self) -> None:
        facts = rdkit_graph_facts("CCO")
        artifact = rdkit_support_artifact_fixture("CCO").artifact
        branch = first_branch_support_object(artifact)
        manifest = branch["payload"]["obligation_manifests"]["stereo_lifecycle"][0]
        manifest["is_discharged"] = False
        manifest["is_noop"] = False
        manifest["is_empty"] = False

        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertTrue(classification.accepted, classification.reason)
        self.assertTrue(classification.stereo_obligations_present)
        self.assertNotIn("stereo_lifecycle", classification.unchecked_families)

    def test_stereo_lifecycle_requires_exact_replay_credit(self) -> None:
        lifecycle = {
            "family": "stereo_lifecycle",
            "evidence_digest": "lifecycle",
            "linked_residual_work_digests": ["residual"],
            "is_noop": False,
            "is_empty": False,
            "is_discharged": True,
            "terminal_clean": False,
        }

        self.assertFalse(
            offline_verifier_module._obligation_manifest_checked(
                lifecycle,
                replayed_residual_digests=set(),
                replayed_lifecycle_digests=set(),
                replayed_directional_ring_closure_digests=set(),
            )
        )
        self.assertTrue(
            offline_verifier_module._obligation_manifest_checked(
                lifecycle,
                replayed_residual_digests={"residual"},
                replayed_lifecycle_digests={"lifecycle"},
                replayed_directional_ring_closure_digests=set(),
            )
        )
        lifecycle["linked_residual_work_digests"] = []
        self.assertFalse(
            offline_verifier_module._obligation_manifest_checked(
                lifecycle,
                replayed_residual_digests=set(),
                replayed_lifecycle_digests=set(),
                replayed_directional_ring_closure_digests=set(),
            )
        )

    def test_terminal_clean_obligation_manifests_are_checked(self) -> None:
        facts = rdkit_graph_facts("CCO")
        artifact = rdkit_support_artifact_fixture("CCO").artifact
        classification = classify_obligation_replay(facts=facts, artifact=artifact)
        terminal = first_terminal_support_object(artifact)

        self.assertTrue(classification.accepted, classification.reason)
        self.assertNotIn(
            "terminal_graph_obligation_work",
            classification.unchecked_families,
        )
        self.assertNotIn(
            "terminal_stereo_lifecycle",
            classification.unchecked_families,
        )
        self.assertIn(
            "terminal_graph_obligation_work",
            classification.checked_families,
        )
        self.assertIn(
            "terminal_stereo_lifecycle",
            classification.checked_families,
        )
        self.assertTrue(
            terminal["payload"]["obligation_manifests"][
                "terminal_graph_obligation_work"
            ][0]["terminal_clean"]
        )
        self.assertTrue(
            terminal["payload"]["obligation_manifests"][
                "terminal_stereo_lifecycle"
            ][0]["terminal_clean"]
        )

    def test_terminal_manifest_flags_are_reconstructed_not_credited(self) -> None:
        facts = rdkit_graph_facts("CCO")
        artifact = rdkit_support_artifact_fixture("CCO").artifact
        terminal = first_terminal_support_object(artifact)
        for family in (
            "terminal_graph_obligation_work",
            "terminal_stereo_lifecycle",
        ):
            manifest = terminal["payload"]["obligation_manifests"][family][0]
            manifest["terminal_clean"] = False
            manifest["is_noop"] = False
            manifest["is_empty"] = False
            manifest["is_discharged"] = False

        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn("terminal_graph_manifest_mismatch", classification.reason)

    def test_terminal_obligation_manifest_count_mismatch_is_rejected(self) -> None:
        artifact = rdkit_support_artifact_fixture("CCO").artifact
        terminal = first_terminal_support_object(artifact)
        terminal["payload"]["obligation_summary"]["graph_obligation_work_count"] += 1

        verification = verify_writer_support_artifact_for_facts(
            facts=rdkit_graph_facts("CCO"),
            runtime_options=writer_runtime_options(),
            artifact=artifact,
        )

        self.assertFalse(verification.accepted)
        self.assertIn("obligation_manifest_count_mismatch", verification.reason)

    def test_terminal_obligation_manifest_unknown_family_is_rejected(self) -> None:
        artifact = rdkit_support_artifact_fixture("CCO").artifact
        terminal = first_terminal_support_object(artifact)
        terminal["payload"]["obligation_manifests"]["unknown_terminal_family"] = []

        verification = verify_writer_support_artifact_for_facts(
            facts=rdkit_graph_facts("CCO"),
            runtime_options=writer_runtime_options(),
            artifact=artifact,
        )

        self.assertFalse(verification.accepted)
        self.assertIn("object_payload_fields_mismatch", verification.reason)
