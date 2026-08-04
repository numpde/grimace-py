"""Physically owned rich support-artifact count-coverage contracts."""

from copy import deepcopy
import unittest
from grimace._south_star1.writer_support_artifact_offline_verifier import verify_count_dag_arithmetic
from tests.south_star1.writer_artifact_test_support import artifact_object_by_id
from tests.south_star1.writer_support_artifact_fixtures import completed_prefix_support_artifact_fixture, rdkit_support_artifact_fixture
from tests.south_star1.writer_support_artifact_queries import coverage_object, verify_support_image_coverage_relation





class WriterSupportArtifactCountCoverageTest(unittest.TestCase):
    def test_count_dag_arithmetic_accepts_default_relation_fixtures(self) -> None:
        for smiles in ("CCO", "CC(C)O", "C1CC1", "C1=CC1", "[NH4+]", "[13CH4]"):
            with self.subTest(smiles=smiles):
                artifact = rdkit_support_artifact_fixture(smiles).artifact
                count = artifact_object_by_id(artifact, artifact["roots"]["count_ref"])
                count_dag = artifact_object_by_id(artifact, count["payload"]["count_dag_ref"])

                verification = verify_count_dag_arithmetic(
                    count_dag=count_dag["payload"],
                    count_object=count["payload"],
                )

                self.assertTrue(verification.accepted, verification.reason)
                self.assertEqual(verification.support_count, count["payload"]["support_count"])
                self.assertEqual(
                    verification.completion_count,
                    count["payload"]["completion_count"],
                )

    def test_count_dag_arithmetic_rejects_changed_count_object_totals(self) -> None:
        artifact = rdkit_support_artifact_fixture("CCO").artifact
        count = deepcopy(artifact_object_by_id(artifact, artifact["roots"]["count_ref"]))
        count_dag = artifact_object_by_id(artifact, count["payload"]["count_dag_ref"])
        count["payload"]["support_count"] += 1

        verification = verify_count_dag_arithmetic(
            count_dag=count_dag["payload"],
            count_object=count["payload"],
        )

        self.assertFalse(verification.accepted)
        self.assertIn("count_dag_support_count_mismatch", verification.reason)

    def test_count_dag_arithmetic_rejects_changed_root_node_count(self) -> None:
        artifact = rdkit_support_artifact_fixture("CCO").artifact
        count = artifact_object_by_id(artifact, artifact["roots"]["count_ref"])
        count_dag = deepcopy(artifact_object_by_id(artifact, count["payload"]["count_dag_ref"]))
        root_id = count_dag["payload"]["roots"]["support_count_root"]
        root = next(
            node
            for node in count_dag["payload"]["nodes"]
            if node["node_id"] == root_id
        )
        root["support_count"] += 1

        verification = verify_count_dag_arithmetic(
            count_dag=count_dag["payload"],
            count_object=count["payload"],
        )

        self.assertFalse(verification.accepted)

    def test_count_dag_arithmetic_rejects_missing_child_and_cycle(self) -> None:
        artifact = rdkit_support_artifact_fixture("CCO").artifact
        count = artifact_object_by_id(artifact, artifact["roots"]["count_ref"])
        count_dag = deepcopy(artifact_object_by_id(artifact, count["payload"]["count_dag_ref"]))
        child_id = next(
            node["children"][0]
            for node in count_dag["payload"]["nodes"]
            if node["children"]
        )
        count_dag["payload"]["nodes"] = [
            node for node in count_dag["payload"]["nodes"] if node["node_id"] != child_id
        ]

        missing = verify_count_dag_arithmetic(
            count_dag=count_dag["payload"],
            count_object=count["payload"],
        )

        self.assertFalse(missing.accepted)

        count_dag = deepcopy(artifact_object_by_id(artifact, count["payload"]["count_dag_ref"]))
        count_dag["payload"]["nodes"][0]["children"].append(
            count_dag["payload"]["nodes"][0]["node_id"]
        )
        cycle = verify_count_dag_arithmetic(
            count_dag=count_dag["payload"],
            count_object=count["payload"],
        )

        self.assertFalse(cycle.accepted)

    def test_coverage_missing_or_extra_text_bucket_is_rejected(self) -> None:
        artifact = rdkit_support_artifact_fixture("CCO").artifact
        coverage = coverage_object(artifact)
        coverage["payload"]["text_buckets"] = []

        missing = verify_support_image_coverage_relation(artifact)

        self.assertFalse(missing.accepted)
        self.assertIn("coverage_partition_mismatch", missing.reason)

        artifact = rdkit_support_artifact_fixture("CCO").artifact
        coverage = coverage_object(artifact)
        coverage["payload"]["text_buckets"].append(
            deepcopy(coverage["payload"]["text_buckets"][0])
        )

        extra = verify_support_image_coverage_relation(artifact)

        self.assertFalse(extra.accepted)
        self.assertIn("coverage_duplicate_assignment", extra.reason)

    def test_coverage_support_and_witness_totals_are_rejected(self) -> None:
        artifact = rdkit_support_artifact_fixture("CCO").artifact
        root = artifact_object_by_id(artifact, artifact["roots"]["support_image_root"])
        root["payload"]["distinct_count"] += 1

        distinct = verify_support_image_coverage_relation(artifact)

        self.assertFalse(distinct.accepted)
        self.assertIn("support_image_distinct_count_mismatch", distinct.reason)

        artifact = rdkit_support_artifact_fixture("CCO").artifact
        root = artifact_object_by_id(artifact, artifact["roots"]["support_image_root"])
        root["payload"]["witness_count"] += 1

        witness = verify_support_image_coverage_relation(artifact)

        self.assertFalse(witness.accepted)
        self.assertIn("coverage_count_completion_total_mismatch", witness.reason)

        artifact = rdkit_support_artifact_fixture("CCO").artifact
        count = artifact_object_by_id(artifact, artifact["roots"]["count_ref"])
        count["payload"]["support_count"] += 1

        count_support = verify_support_image_coverage_relation(artifact)

        self.assertFalse(count_support.accepted)
        self.assertIn("coverage_count_support_total_mismatch", count_support.reason)

        artifact = rdkit_support_artifact_fixture("CCO").artifact
        count = artifact_object_by_id(artifact, artifact["roots"]["count_ref"])
        count["payload"]["completion_count"] += 1

        count_completion = verify_support_image_coverage_relation(artifact)

        self.assertFalse(count_completion.accepted)
        self.assertIn(
            "coverage_count_completion_total_mismatch",
            count_completion.reason,
        )

    def test_coverage_terminal_bucket_mutations_are_rejected(self) -> None:
        artifact = completed_prefix_support_artifact_fixture().artifact
        coverage = coverage_object(artifact)
        coverage["payload"]["terminal_bucket"] = None

        missing = verify_support_image_coverage_relation(artifact)

        self.assertFalse(missing.accepted)
        self.assertIn("coverage_terminal_bucket_missing", missing.reason)

        artifact = completed_prefix_support_artifact_fixture().artifact
        coverage = coverage_object(artifact)
        coverage["payload"]["text_buckets"].append(
            {
                "text_projection": {},
                "support_count": 1,
                "string_refs": [
                    artifact_object_by_id(
                        artifact,
                        artifact["roots"]["support_image_root"],
                    )["payload"]["support_string_refs"][0]
                ],
            }
        )

        text_bucket = verify_support_image_coverage_relation(artifact)

        self.assertFalse(text_bucket.accepted)
        self.assertIn("coverage_empty_string_in_text_bucket", text_bucket.reason)

        artifact = completed_prefix_support_artifact_fixture().artifact
        coverage = coverage_object(artifact)
        coverage["payload"]["terminal_bucket"]["terminal_projection"] = {
            **coverage["payload"]["terminal_bucket"]["terminal_projection"],
            "digest": "0" * 64,
        }

        wrong_projection = verify_support_image_coverage_relation(artifact)

        self.assertFalse(wrong_projection.accepted)
        self.assertIn("coverage_terminal_projection_mismatch", wrong_projection.reason)

    def test_coverage_wrong_duplicate_and_unassigned_refs_are_rejected(self) -> None:
        artifact = rdkit_support_artifact_fixture("CCO").artifact
        coverage = coverage_object(artifact)
        coverage["payload"]["text_buckets"][0]["string_refs"] = ["missing"]
        coverage["payload"]["text_buckets"][0]["support_count"] = 1

        wrong = verify_support_image_coverage_relation(artifact)

        self.assertFalse(wrong.accepted)
        self.assertIn("coverage_text_bucket_unknown_ref", wrong.reason)

        artifact = rdkit_support_artifact_fixture("CC(C)O").artifact
        root = artifact_object_by_id(artifact, artifact["roots"]["support_image_root"])
        coverage = coverage_object(artifact)
        first_ref = root["payload"]["support_string_refs"][0]
        coverage["payload"]["text_buckets"][0]["string_refs"] = [first_ref, first_ref]
        coverage["payload"]["text_buckets"][0]["support_count"] = 2

        duplicate = verify_support_image_coverage_relation(artifact)

        self.assertFalse(duplicate.accepted)
        self.assertIn("coverage_duplicate_assignment", duplicate.reason)

    def test_coverage_wrong_text_projection_ref_is_rejected(self) -> None:
        artifact = rdkit_support_artifact_fixture("CCO").artifact
        coverage = coverage_object(artifact)
        coverage["payload"]["text_buckets"][0]["text_projection"] = {
            **coverage["payload"]["text_buckets"][0]["text_projection"],
            "emitted_text": "N",
        }

        verification = verify_support_image_coverage_relation(artifact)

        self.assertFalse(verification.accepted)
        self.assertIn("coverage_text_projection_mismatch", verification.reason)

    def test_support_image_coverage_accepts_default_relation_fixtures(self) -> None:
        for smiles in ("CCO", "C1CC1", "C1=CC1", "[NH4+]", "[13CH4]"):
            with self.subTest(smiles=smiles):
                artifact = rdkit_support_artifact_fixture(smiles).artifact

                verification = verify_support_image_coverage_relation(artifact)

                self.assertTrue(verification.accepted, verification.reason)
                root = artifact_object_by_id(artifact, artifact["roots"]["support_image_root"])
                self.assertEqual(verification.support_count, root["payload"]["distinct_count"])
                self.assertEqual(verification.witness_count, root["payload"]["witness_count"])

    def test_support_image_coverage_accepts_terminal_bucket(self) -> None:
        artifact = completed_prefix_support_artifact_fixture().artifact

        verification = verify_support_image_coverage_relation(artifact)

        self.assertTrue(verification.accepted, verification.reason)
        self.assertEqual(verification.support_count, 1)
        self.assertEqual(verification.witness_count, 2)
