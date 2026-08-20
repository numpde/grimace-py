"""Physically owned rich support-artifact obligation-replay contracts."""

from copy import deepcopy
import unittest
import grimace._south_star1.writer_support_artifact_offline_verifier as offline_verifier_module
from grimace._south_star1.writer_support_artifact_checker import verify_writer_support_artifact_consistency
from grimace._south_star1.writer_support_artifact_fact_verifier import verify_writer_support_artifact_for_facts
from tests.south_star1.writer_artifact_resealing import reseal_support_artifact
from tests.south_star1.writer_test_context import writer_runtime_options
from tests.south_star1.writer_support_artifact_fixtures import rdkit_graph_facts, rdkit_support_artifact_fixture
from tests.south_star1.writer_support_artifact_queries import classify_obligation_replay, first_branch_support_object, first_graph_ring_delta_branch, first_terminal_support_object





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
