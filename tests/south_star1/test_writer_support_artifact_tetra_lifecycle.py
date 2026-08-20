"""Physically owned rich support-artifact tetra-lifecycle contracts."""

from copy import deepcopy
import unittest
from grimace._south_star1.writer_support_artifact_fact_verifier import verify_writer_support_artifact_for_facts
from tests.south_star1.writer_test_context import writer_runtime_options
from tests.south_star1.writer_support_artifact_fixtures import tetra_support_artifact_fixture
from tests.south_star1.writer_support_artifact_queries import classify_obligation_replay, first_graph_ring_delta_event, first_residual_work_branch, support_strings
from tests.south_star1.writer_support_artifact_transition_test_support import linked_tetra_lifecycle_manifest
from tests.south_star1.writer_support_artifact_tetra_test_support import append_unrelated_raw_lifecycle





class WriterSupportArtifactTetraLifecycleTest(unittest.TestCase):
    def test_specified_tetra_no_second_authority_from_atom_token_text(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        event = first_graph_ring_delta_event(branch, "atom_emitted")
        self.assertIn(event["tetra_token"]["value"], ("@", "@@"))
        self.assertTrue(
            any(event["tetra_token"]["value"] in text for text in support_strings(artifact))
        )
        manifest["linked_lifecycle_digests"] = []

        classification = classify_obligation_replay(facts=facts, artifact=artifact)
        verification = verify_writer_support_artifact_for_facts(
            facts=facts,
            runtime_options=writer_runtime_options(),
            artifact=artifact,
        )

        self.assertFalse(classification.accepted)
        self.assertIn(
            "residual_lifecycle_forward_link_provenance_mismatch",
            classification.reason,
        )
        self.assertFalse(verification.accepted)
        self.assertIn("object_digest_mismatch", verification.reason)

    def test_specified_tetra_no_second_authority_from_final_support_strings(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        lifecycle = linked_tetra_lifecycle_manifest(
            branch=branch,
            manifest=manifest,
            lifecycle_kind="certificate",
            certificate_kind="tetra_token_restricted",
        )
        self.assertTrue(any("@" in text for text in support_strings(artifact)))
        lifecycle["certificate_capability"] = "tetra_local_order_restriction"

        classification = classify_obligation_replay(facts=facts, artifact=artifact)
        verification = verify_writer_support_artifact_for_facts(
            facts=facts,
            runtime_options=writer_runtime_options(),
            artifact=artifact,
        )

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_residual_certificate_capability_mismatch",
            classification.reason,
        )
        self.assertFalse(verification.accepted)
        self.assertIn("object_digest_mismatch", verification.reason)

    def test_specified_tetra_no_second_authority_from_stale_lifecycle_digest(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        lifecycle = linked_tetra_lifecycle_manifest(
            branch=branch,
            manifest=manifest,
            lifecycle_kind="raw",
            certificate_kind="tetra_token_restricted",
        )
        lifecycle["source_digest"] = "stale_source_digest"

        classification = classify_obligation_replay(facts=facts, artifact=artifact)
        verification = verify_writer_support_artifact_for_facts(
            facts=facts,
            runtime_options=writer_runtime_options(),
            artifact=artifact,
        )

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_residual_lifecycle_source_mismatch",
            classification.reason,
        )
        self.assertFalse(verification.accepted)
        self.assertIn("object_digest_mismatch", verification.reason)

    def test_specified_tetra_residual_lifecycle_provenance_mutations_are_rejected(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        cases = (
            (
                "raw_event_kind",
                "raw",
                "lifecycle_event_kind",
                "local_order_closed",
                "tetra_residual_lifecycle_event_kind_mismatch",
            ),
            (
                "raw_missing_capability",
                "raw",
                "lifecycle_capabilities",
                ["tetra_token_restriction"],
                "tetra_residual_lifecycle_capabilities_mismatch",
            ),
            (
                "raw_extra_capability",
                "raw",
                "lifecycle_capabilities",
                [
                    "residual_factor_discharge",
                    "residual_propagation",
                    "tetra_token_restriction",
                ],
                "tetra_residual_lifecycle_capabilities_mismatch",
            ),
            (
                "raw_outcome",
                "raw",
                "lifecycle_outcome_kind",
                "event_recorded",
                "tetra_residual_lifecycle_outcome_kind_mismatch",
            ),
            (
                "raw_change_flag",
                "raw",
                "residual_snapshot_changed",
                False,
                "tetra_residual_lifecycle_change_flag_mismatch",
            ),
            (
                "raw_work_digests",
                "raw",
                "residual_work_digests",
                [],
                "residual_lifecycle_reverse_link_provenance_mismatch",
            ),
            (
                "raw_work_operations",
                "raw",
                "residual_work_operations",
                ["wrong"],
                "tetra_residual_lifecycle_work_operation_mismatch",
            ),
            (
                "certificate_kind",
                "certificate",
                "certificate_kind",
                "tetra_local_order_restricted",
                "tetra_residual_lifecycle_evidence_missing",
            ),
            (
                "certificate_capability",
                "certificate",
                "certificate_capability",
                "tetra_local_order_restriction",
                "tetra_residual_certificate_capability_mismatch",
            ),
            (
                "certificate_lifecycle_digest",
                "certificate",
                "certificate_lifecycle_digest",
                "wrong",
                "tetra_residual_certificate_lifecycle_digest_mismatch",
            ),
        )
        for name, lifecycle_kind, field, value, reason in cases:
            with self.subTest(name=name):
                mutated = deepcopy(artifact)
                branch = first_residual_work_branch(
                    mutated,
                    operation="tetrahedral atom-token restriction",
                )
                manifest = next(
                    item
                    for item in (
                        branch["payload"]["obligation_manifests"]["residual_work"]
                    )
                    if item["operation"] == "tetrahedral atom-token restriction"
                )
                lifecycle = linked_tetra_lifecycle_manifest(
                    branch=branch,
                    manifest=manifest,
                    lifecycle_kind=lifecycle_kind,
                    certificate_kind="tetra_token_restricted",
                )
                lifecycle[field] = value

                classification = classify_obligation_replay(facts=facts, artifact=mutated)

                self.assertFalse(classification.accepted)
                self.assertIn(reason, classification.reason)

    def test_specified_tetra_residual_reciprocal_extra_link_is_rejected(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        unrelated = append_unrelated_raw_lifecycle(branch, manifest=manifest)
        manifest["linked_lifecycle_digests"].append(unrelated["evidence_digest"])
        unrelated["linked_residual_work_digests"].append(manifest["evidence_digest"])

        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "residual_lifecycle_reverse_link_provenance_mismatch",
            classification.reason,
        )

    def test_specified_tetra_residual_rejects_bogus_reverse_link(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        lifecycle = next(
            item
            for item in branch["payload"]["obligation_manifests"]["stereo_lifecycle"]
            if item["evidence_digest"] in manifest["linked_lifecycle_digests"]
        )
        lifecycle["linked_residual_work_digests"].append("bogus")

        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "residual_lifecycle_reverse_link_provenance_mismatch",
            classification.reason,
        )

    def test_specified_tetra_residual_rejects_coherently_forged_link_projection(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        unrelated = append_unrelated_raw_lifecycle(branch, manifest=manifest)
        unrelated["linked_residual_work_digests"].append(manifest["evidence_digest"])
        unrelated["residual_work_digests"].append(manifest["evidence_digest"])
        manifest["linked_lifecycle_digests"] = [
            item["evidence_digest"]
            for item in branch["payload"]["obligation_manifests"]["stereo_lifecycle"]
            if manifest["evidence_digest"] in item["residual_work_digests"]
        ]

        # Public verification rejects this hand-edited artifact structurally via
        # stale object digests; this isolates the offline classifier precondition.
        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_residual_lifecycle_evidence_missing",
            classification.reason,
        )

    def test_specified_tetra_residual_rejects_extra_lifecycle_link(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        unrelated = append_unrelated_raw_lifecycle(branch, manifest=manifest)
        manifest["linked_lifecycle_digests"].append(unrelated["evidence_digest"])

        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "residual_lifecycle_forward_link_provenance_mismatch",
            classification.reason,
        )

    def test_specified_tetra_residual_rejects_reverse_only_lifecycle_link(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        lifecycle = next(
            item
            for item in branch["payload"]["obligation_manifests"]["stereo_lifecycle"]
            if item["evidence_digest"] in manifest["linked_lifecycle_digests"]
        )
        manifest["linked_lifecycle_digests"].remove(lifecycle["evidence_digest"])

        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "residual_lifecycle_forward_link_provenance_mismatch",
            classification.reason,
        )

    def test_specified_tetra_residual_rejects_unreciprocated_reverse_link(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        target = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        other = deepcopy(target)
        other["evidence_digest"] = "other_residual"
        other["linked_lifecycle_digests"] = []
        branch["payload"]["obligation_manifests"]["residual_work"].append(other)
        lifecycle = next(
            item
            for item in branch["payload"]["obligation_manifests"]["stereo_lifecycle"]
            if item["evidence_digest"] in target["linked_lifecycle_digests"]
        )
        lifecycle["linked_residual_work_digests"].append(other["evidence_digest"])

        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "residual_lifecycle_reverse_link_provenance_mismatch",
            classification.reason,
        )

    def test_specified_tetra_residual_rejects_unrelated_lifecycle_link(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        unrelated = append_unrelated_raw_lifecycle(branch, manifest=manifest)
        manifest["linked_lifecycle_digests"] = [unrelated["evidence_digest"]]

        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "residual_lifecycle_forward_link_provenance_mismatch",
            classification.reason,
        )

    def test_specified_tetra_residual_rejects_wrong_lifecycle_link(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        manifest["linked_lifecycle_digests"] = ["wrong"]

        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "residual_lifecycle_forward_link_provenance_mismatch",
            classification.reason,
        )

    def test_specified_tetra_residual_requires_exact_lifecycle_operations(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        lifecycle = branch["payload"]["obligation_manifests"]["stereo_lifecycle"]
        branch_certificate = next(
            item
            for item in lifecycle
            if item["operation"] == "WriterStereoBranchCertificate"
        )
        branch_certificate["operation"] = "UnexpectedStereoCertificate"

        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_residual_lifecycle_operation_mismatch",
            classification.reason,
        )

    def test_specified_tetra_residual_requires_lifecycle_link(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        manifest["linked_lifecycle_digests"] = []

        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "residual_lifecycle_forward_link_provenance_mismatch",
            classification.reason,
        )

    def test_specified_tetra_residual_requires_reverse_lifecycle_link(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        lifecycle = next(
            item
            for item in branch["payload"]["obligation_manifests"]["stereo_lifecycle"]
            if item["evidence_digest"] in manifest["linked_lifecycle_digests"]
        )
        lifecycle["linked_residual_work_digests"].remove(manifest["evidence_digest"])

        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "residual_lifecycle_reverse_link_provenance_mismatch",
            classification.reason,
        )
