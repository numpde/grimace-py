"""Table-backed writer support artifact envelope tests."""

from __future__ import annotations

from tests.south_star1.writer_test_context import initial_writer_snapshot
from tests.south_star1.writer_test_context import prepare_writer_facts
from tests.south_star1.writer_test_context import writer_runtime_options

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.writer_envelope_terms import _canonical_json
from grimace._south_star1.writer_envelope_terms import _digest_terms_bounded
from grimace._south_star1.writer_envelope_terms import _identity_digest
from grimace._south_star1.writer_envelope_work import WriterEnvelopeWorkBudget
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_snapshot import capture_writer_frontier_snapshot
from grimace._south_star1.writer_snapshot_prefix_envelope import (
    writer_snapshot_prefix_read_envelope_for_emitted_texts,
)
from grimace._south_star1.writer_support_artifact_checker import (
    artifact_manifest,
)
from grimace._south_star1.writer_support_artifact_checker import (
    support_artifact_object_identity_term,
)
from grimace._south_star1.writer_support_artifact_checker import (
    SCHEMA_VERSION,
)
from grimace._south_star1.writer_support_artifact_checker import (
    verify_writer_support_artifact_consistency as check_support_artifact,
)
from grimace._south_star1.writer_support_artifact_checker import (
    WriterSupportArtifactCheckResult,
)
import grimace._south_star1.writer_support_artifact_envelope as artifact_envelope_module
from grimace._south_star1.writer_support_artifact_envelope import (
    verify_writer_support_artifact_consistency,
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
from grimace._south_star1.writer_support_image_envelope import (
    writer_support_image_envelope_for_snapshot,
)
from tests.south_star1.helpers import cyclopropane_facts
from tests.south_star1.helpers import cco_facts
from tests.south_star1.helpers import two_atom_facts


class WriterSupportArtifactEnvelopeTest(unittest.TestCase):
    def test_snapshot_source_artifact_json_round_trips(self) -> None:
        envelope = _json_round_trip(_snapshot_artifact())

        verification = verify_writer_support_artifact_consistency(envelope)
        check = check_support_artifact(envelope)

        self.assertTrue(verification.accepted)
        self.assertTrue(check.accepted)
        self.assertEqual(verification.support_count, 1)
        self.assertEqual(verification.witness_count, 2)
        self.assertEqual(check.object_count, envelope["metrics"]["object_count"])

    def test_new_artifacts_use_schema_v11(self) -> None:
        envelope = _snapshot_artifact()

        self.assertEqual(SCHEMA_VERSION, 11)
        self.assertEqual(envelope["schema_version"], 11)
        self.assertTrue(check_support_artifact(envelope).accepted)

    def test_v10_artifact_is_rejected_without_migration(self) -> None:
        envelope = _snapshot_artifact()
        envelope["schema_version"] = 10
        envelope["digest"] = _digest_terms_bounded(
            artifact_manifest(envelope),
            budget=WriterEnvelopeWorkBudget(),
            operation="test.artifact_manifest.digest",
        )

        check = check_support_artifact(envelope)

        self.assertFalse(check.accepted)
        self.assertIn("unknown_schema_version", check.reason)

    def test_producer_consistency_wrapper_delegates_to_checker(self) -> None:
        envelope = _snapshot_artifact()

        with patch.object(
            artifact_envelope_module,
            "_check_writer_support_artifact_consistency",
            return_value=WriterSupportArtifactCheckResult(
                accepted=False,
                reason="sentinel_checker_rejection",
            ),
        ) as checker:
            verification = verify_writer_support_artifact_consistency(envelope)

        checker.assert_called_once()
        self.assertFalse(verification.accepted)
        self.assertEqual(verification.reason, "sentinel_checker_rejection")

    def test_snapshot_source_artifact_live_verifier_accepts(self) -> None:
        prepared = prepare_writer_facts(two_atom_facts())
        envelope = writer_support_artifact_envelope_for_snapshot(
            prepared=prepared,
            snapshot=initial_writer_snapshot(prepared, writer_runtime_options()),
        )

        verification = verify_writer_support_artifact_envelope(
            prepared=prepared,
            envelope=envelope,
        )

        self.assertTrue(verification.accepted, verification.reason)
        self.assertEqual(verification.support_count, 1)
        self.assertEqual(verification.witness_count, 2)

    def test_prefix_read_source_artifact_verifies(self) -> None:
        prepared = prepare_writer_facts(two_atom_facts())
        prefix = writer_snapshot_prefix_read_envelope_for_emitted_texts(
            prepared=prepared,
            snapshot=initial_writer_snapshot(prepared, writer_runtime_options()),
            emitted_texts=("C", "C"),
        )
        envelope = writer_support_artifact_envelope_for_prefix_read(
            prepared=prepared,
            prefix_read_envelope=prefix,
        )

        verification = verify_writer_support_artifact_consistency(envelope)

        self.assertTrue(verification.accepted)
        self.assertEqual(verification.source_kind, "prefix_read")

    def test_prefix_read_source_artifact_live_verifier_accepts(self) -> None:
        prepared = prepare_writer_facts(two_atom_facts())
        prefix = writer_snapshot_prefix_read_envelope_for_emitted_texts(
            prepared=prepared,
            snapshot=initial_writer_snapshot(prepared, writer_runtime_options()),
            emitted_texts=("C", "C"),
        )
        envelope = writer_support_artifact_envelope_for_prefix_read(
            prepared=prepared,
            prefix_read_envelope=prefix,
        )

        verification = verify_writer_support_artifact_envelope(
            prepared=prepared,
            envelope=envelope,
        )

        self.assertTrue(verification.accepted, verification.reason)
        self.assertEqual(verification.source_kind, "prefix_read")

    def test_branching_artifact_verifies(self) -> None:
        prepared = prepare_writer_facts(cco_facts())
        envelope = writer_support_artifact_envelope_for_snapshot(
            prepared=prepared,
            snapshot=initial_writer_snapshot(prepared, writer_runtime_options()),
        )

        verification = verify_writer_support_artifact_consistency(envelope)

        self.assertTrue(verification.accepted)
        self.assertGreater(envelope["metrics"]["support_string_count"], 1)

    def test_count_and_frontier_objects_are_singletons(self) -> None:
        envelope = _branching_artifact()
        counts = envelope["metrics"]["object_kind_counts"]

        self.assertEqual(counts["count_envelope"], 1)
        self.assertEqual(counts["frontier_product"], 1)
        self.assertTrue(verify_writer_support_artifact_consistency(envelope).accepted)

    def test_artifact_metrics_include_reachability_and_count_dag_summary(self) -> None:
        envelope = _branching_artifact()
        metrics = envelope["metrics"]

        self.assertEqual(metrics["object_count"], metrics["reachable_object_count"])
        self.assertEqual(metrics["unreferenced_object_count"], 0)
        self.assertGreater(metrics["coverage_bucket_count"], 0)
        self.assertGreater(metrics["count_dag_node_count"], 0)
        self.assertGreater(metrics["count_dag_edge_count"], 0)
        self.assertGreater(metrics["unique_branch_support_count"], 0)
        self.assertEqual(
            metrics["total_payload_bytes"],
            metrics["total_artifact_payload_bytes"],
        )
        self.assertEqual(
            metrics["largest_object_digest_bytes"],
            metrics["largest_object_digest_payload_bytes"],
        )
        self.assertEqual(
            metrics["largest_object_digest_bytes"],
            metrics["largest_object_identity_input_bytes"],
        )
        self.assertGreaterEqual(
            metrics["largest_object_payload_bytes"],
            metrics["largest_object_identity_input_bytes"],
        )
        self.assertGreater(metrics["total_object_identity_input_bytes"], 0)

    def test_artifact_and_legacy_nested_support_image_agree_on_core_fixture(self) -> None:
        prepared = prepare_writer_facts(cco_facts())
        snapshot = initial_writer_snapshot(prepared, writer_runtime_options())
        artifact = writer_support_artifact_envelope_for_snapshot(
            prepared=prepared,
            snapshot=snapshot,
        )
        nested = writer_support_image_envelope_for_snapshot(
            prepared=prepared,
            snapshot=snapshot,
        )
        root = _root_payload(artifact)
        count = _object(artifact, root["count_ref"])["payload"]

        self.assertEqual(root["support_strings"], nested["support_strings"])
        self.assertEqual(root["distinct_count"], nested["distinct_count"])
        self.assertEqual(root["witness_count"], nested["witness_count"])
        self.assertEqual(
            count["count_dag_digest"],
            nested["count_envelope"]["count_dag"]["digest"],
        )
        self.assertEqual(
            count["frontier_product_digest"],
            nested["count_envelope"]["frontier_product"]["digest"],
        )

    def test_table_artifact_is_smaller_than_nested_support_image(self) -> None:
        prepared = prepare_writer_facts(cyclopropane_facts())
        snapshot = initial_writer_snapshot(prepared, writer_runtime_options())
        artifact = writer_support_artifact_envelope_for_snapshot(
            prepared=prepared,
            snapshot=snapshot,
        )
        nested = writer_support_image_envelope_for_snapshot(
            prepared=prepared,
            snapshot=snapshot,
        )

        self.assertLess(
            len(_canonical_json(artifact)),
            len(_canonical_json(nested)),
        )

    def test_missing_object_ref_is_rejected(self) -> None:
        envelope = _snapshot_artifact()
        envelope["objects"] = [
            item
            for item in envelope["objects"]
            if item["object_id"] != envelope["roots"]["count_ref"]
        ]

        self.assertFalse(verify_writer_support_artifact_consistency(envelope).accepted)

    def test_duplicate_object_id_is_rejected(self) -> None:
        envelope = _snapshot_artifact()
        envelope["objects"].append(deepcopy(envelope["objects"][0]))

        self.assertFalse(verify_writer_support_artifact_consistency(envelope).accepted)

    def test_wrong_object_digest_is_rejected(self) -> None:
        envelope = _snapshot_artifact()
        envelope["objects"][0]["digest"] = "0" * 64

        self.assertFalse(verify_writer_support_artifact_consistency(envelope).accepted)

    def test_count_dag_object_identity_is_compact_manifest_digest(self) -> None:
        envelope = _snapshot_artifact()
        count = _object(envelope, envelope["roots"]["count_ref"])
        count_dag = _object(envelope, count["payload"]["count_dag_ref"])

        digest = _identity_digest(
            support_artifact_object_identity_term(
                count_dag["kind"],
                count_dag["payload"],
            ),
            budget=WriterEnvelopeWorkBudget(),
            operation="test.count_dag_object.digest",
        )

        self.assertEqual(count_dag["digest"], digest)
        self.assertEqual(count_dag["object_id"], f"obj:{digest}")
        self.assertLess(
            envelope["metrics"]["largest_object_identity_input_bytes"],
            envelope["metrics"]["largest_object_payload_bytes"],
        )
        self.assertTrue(check_support_artifact(envelope).accepted)

    def test_stale_count_dag_internal_node_digest_is_rejected(self) -> None:
        envelope = _snapshot_artifact()
        count = _object(envelope, envelope["roots"]["count_ref"])
        count_dag = _object(envelope, count["payload"]["count_dag_ref"])
        node = next(
            node for node in count_dag["payload"]["nodes"] if "support_count" in node
        )
        node["support_count"] = 999

        check = check_support_artifact(envelope)

        self.assertFalse(check.accepted)
        self.assertIn("count_dag_node_digest_mismatch", check.reason)

    def test_stale_count_dag_object_digest_is_rejected(self) -> None:
        envelope = _snapshot_artifact()
        count = _object(envelope, envelope["roots"]["count_ref"])
        count_dag = _object(envelope, count["payload"]["count_dag_ref"])
        replacement = _branching_artifact()
        replacement_count = _object(replacement, replacement["roots"]["count_ref"])
        replacement_dag = _object(
            replacement,
            replacement_count["payload"]["count_dag_ref"],
        )
        count_dag["payload"] = deepcopy(replacement_dag["payload"])

        check = check_support_artifact(envelope)

        self.assertFalse(check.accepted)
        self.assertIn("object_digest_mismatch", check.reason)

    def test_stale_count_envelope_count_dag_link_digest_is_rejected(self) -> None:
        envelope = _snapshot_artifact()
        count = _object(envelope, envelope["roots"]["count_ref"])
        count_dag = _object(envelope, count["payload"]["count_dag_ref"])
        replacement = _branching_artifact()
        replacement_count = _object(replacement, replacement["roots"]["count_ref"])
        replacement_dag = _object(
            replacement,
            replacement_count["payload"]["count_dag_ref"],
        )
        count_dag["payload"] = deepcopy(replacement_dag["payload"])
        self.assertNotEqual(
            count_dag["payload"]["digest"],
            count["payload"]["count_dag_digest"],
        )
        _refresh_object_digest(envelope, count_dag)

        check = check_support_artifact(envelope)

        self.assertFalse(check.accepted)
        self.assertIn("count_dag_ref_digest_mismatch", check.reason)

    def test_count_envelope_count_dag_link_kind_mismatch_is_rejected(self) -> None:
        envelope = _snapshot_artifact()
        count = _object(envelope, envelope["roots"]["count_ref"])
        original_count_dag_ref = count["payload"]["count_dag_ref"]
        count["payload"]["count_dag_ref"] = envelope["roots"]["source_ref"]
        _refresh_object_digest(envelope, count)
        _drop_object(envelope, original_count_dag_ref)

        check = check_support_artifact(envelope)

        self.assertFalse(check.accepted)
        self.assertIn("count_dag_ref_kind_mismatch", check.reason)

    def test_count_envelope_count_dag_link_node_count_mismatch_is_rejected(
        self,
    ) -> None:
        envelope = _snapshot_artifact()
        count = _object(envelope, envelope["roots"]["count_ref"])
        count["payload"]["count_dag_node_count"] += 1
        _refresh_object_digest(envelope, count)

        check = check_support_artifact(envelope)

        self.assertFalse(check.accepted)
        self.assertIn("count_dag_ref_node_count_mismatch", check.reason)

    def test_count_envelope_count_dag_link_edge_count_mismatch_is_rejected(
        self,
    ) -> None:
        envelope = _snapshot_artifact()
        count = _object(envelope, envelope["roots"]["count_ref"])
        count["payload"]["count_dag_edge_count"] += 1
        _refresh_object_digest(envelope, count)

        check = check_support_artifact(envelope)

        self.assertFalse(check.accepted)
        self.assertIn("count_dag_ref_edge_count_mismatch", check.reason)

    def test_unknown_payload_field_is_rejected(self) -> None:
        envelope = _snapshot_artifact()
        _root_payload(envelope)["unexpected"] = True

        self.assertFalse(check_support_artifact(envelope).accepted)

    def test_obligation_manifest_missing_lifecycle_provenance_field_is_rejected(
        self,
    ) -> None:
        envelope = _branching_artifact()
        manifest = _first_obligation_manifest(envelope)
        del manifest["certificate_capability"]

        self.assertFalse(check_support_artifact(envelope).accepted)

    def test_non_stereo_obligation_lifecycle_provenance_must_be_neutral(
        self,
    ) -> None:
        envelope = _branching_artifact()
        manifest = _first_obligation_manifest(
            envelope,
            excluded_family="stereo_lifecycle",
        )
        manifest["lifecycle_event_kind"] = "atom_emitted"

        self.assertFalse(check_support_artifact(envelope).accepted)

    def test_count_object_missing_count_dag_is_rejected(self) -> None:
        envelope = _snapshot_artifact()
        root = _root_payload(envelope)
        del _object(envelope, root["count_ref"])["payload"]["count_dag_digest"]

        self.assertFalse(check_support_artifact(envelope).accepted)

    def test_changed_support_string_ref_is_rejected(self) -> None:
        envelope = _branching_artifact()
        root = _root_payload(envelope)
        root["support_string_refs"][0] = envelope["roots"]["count_ref"]

        self.assertFalse(verify_writer_support_artifact_consistency(envelope).accepted)

    def test_changed_count_ref_is_rejected(self) -> None:
        envelope = _snapshot_artifact()
        root = _root_payload(envelope)
        root["count_ref"] = envelope["roots"]["frontier_product_ref"]

        self.assertFalse(verify_writer_support_artifact_consistency(envelope).accepted)

    def test_coverage_unknown_string_ref_is_rejected(self) -> None:
        envelope = _branching_artifact()
        coverage = _coverage_payload(envelope)
        coverage["text_buckets"][0]["string_refs"][0] = envelope["roots"]["count_ref"]

        self.assertFalse(verify_writer_support_artifact_consistency(envelope).accepted)

    def test_same_support_string_ref_assigned_to_two_buckets_is_rejected(self) -> None:
        envelope = _branching_artifact()
        coverage = _coverage_payload(envelope)
        if len(coverage["text_buckets"]) < 2:
            self.skipTest("fixture lacks two text buckets")
        coverage["text_buckets"][1]["string_refs"].append(
            coverage["text_buckets"][0]["string_refs"][0]
        )
        coverage["text_buckets"][1]["support_count"] += 1

        self.assertFalse(verify_writer_support_artifact_consistency(envelope).accepted)

    def test_distinct_count_mismatch_is_rejected(self) -> None:
        envelope = _snapshot_artifact()
        _root_payload(envelope)["distinct_count"] += 1

        self.assertFalse(verify_writer_support_artifact_consistency(envelope).accepted)

    def test_witness_count_mismatch_is_rejected(self) -> None:
        envelope = _snapshot_artifact()
        _root_payload(envelope)["witness_count"] += 1

        self.assertFalse(verify_writer_support_artifact_consistency(envelope).accepted)

    def test_replay_path_text_mismatch_is_rejected(self) -> None:
        envelope = _snapshot_artifact()
        support_string = _support_string_payload(envelope)
        replay = _object(envelope, support_string["replay_path_ref"])
        replay["payload"]["emitted_texts"] = ["bad"]

        self.assertFalse(verify_writer_support_artifact_consistency(envelope).accepted)

    def test_terminal_projection_ref_mismatch_is_rejected(self) -> None:
        envelope = _snapshot_artifact()
        _support_string_payload(envelope)["terminal_projection_ref"] = (
            envelope["roots"]["count_ref"]
        )

        self.assertFalse(verify_writer_support_artifact_consistency(envelope).accepted)

    def test_unreferenced_object_is_rejected(self) -> None:
        envelope = _snapshot_artifact()
        extra = deepcopy(_object(envelope, envelope["roots"]["source_ref"]))
        extra["payload"]["digest"] = "1" * 64
        digest = _identity_digest(
            support_artifact_object_identity_term(extra["kind"], extra["payload"]),
            budget=WriterEnvelopeWorkBudget(),
            operation="test.unreferenced_object.digest",
        )
        extra["digest"] = digest
        extra["object_id"] = f"obj:{digest}"
        envelope["objects"].append(extra)

        self.assertFalse(verify_writer_support_artifact_consistency(envelope).accepted)

    def test_live_verifier_rejects_stale_structural_table(self) -> None:
        prepared = prepare_writer_facts(cco_facts())
        envelope = writer_support_artifact_envelope_for_snapshot(
            prepared=prepared,
            snapshot=initial_writer_snapshot(prepared, writer_runtime_options()),
        )
        root = _root_payload(envelope)
        root["support_strings"] = list(reversed(root["support_strings"]))

        verification = verify_writer_support_artifact_envelope(
            prepared=prepared,
            envelope=envelope,
        )

        self.assertFalse(verification.accepted)


def _snapshot_artifact():
    prepared = prepare_writer_facts(two_atom_facts())
    return deepcopy(
        writer_support_artifact_envelope_for_snapshot(
            prepared=prepared,
            snapshot=initial_writer_snapshot(prepared, writer_runtime_options()),
        )
    )


def _branching_artifact():
    prepared = prepare_writer_facts(cco_facts())
    return deepcopy(
        writer_support_artifact_envelope_for_snapshot(
            prepared=prepared,
            snapshot=initial_writer_snapshot(prepared, writer_runtime_options()),
        )
    )


def _root_payload(envelope):
    return _object(envelope, envelope["roots"]["support_image_root"])["payload"]


def _coverage_payload(envelope):
    return _object(envelope, _root_payload(envelope)["coverage_ref"])["payload"]


def _support_string_payload(envelope):
    return _object(envelope, _root_payload(envelope)["support_string_refs"][0])[
        "payload"
    ]


def _first_obligation_manifest(envelope, *, excluded_family: str | None = None):
    for item in envelope["objects"]:
        if item["kind"] != "branch_support":
            continue
        for family, manifests in item["payload"]["obligation_manifests"].items():
            if family == excluded_family or not manifests:
                continue
            return manifests[0]
    raise AssertionError("missing obligation manifest")


def _object(envelope, object_id):
    return next(item for item in envelope["objects"] if item["object_id"] == object_id)


def _refresh_object_digest(envelope, obj):
    old_id = obj["object_id"]
    digest = _identity_digest(
        support_artifact_object_identity_term(obj["kind"], obj["payload"]),
        budget=WriterEnvelopeWorkBudget(),
        operation="test.object.digest",
    )
    obj["digest"] = digest
    obj["object_id"] = f"obj:{digest}"
    _replace_ref(envelope, old_id, obj["object_id"])
    changed = True
    while changed:
        changed = False
        for item in envelope["objects"]:
            digest = _identity_digest(
                support_artifact_object_identity_term(item["kind"], item["payload"]),
                budget=WriterEnvelopeWorkBudget(),
                operation="test.object.digest",
            )
            object_id = f"obj:{digest}"
            if item["digest"] == digest and item["object_id"] == object_id:
                continue
            old_id = item["object_id"]
            item["digest"] = digest
            item["object_id"] = object_id
            _replace_ref(envelope, old_id, object_id)
            changed = True
    envelope["metrics"] = artifact_envelope_module.artifact_metrics(
        envelope["objects"],
        roots=envelope["roots"],
    )
    envelope["digest"] = _digest_terms_bounded(
        artifact_manifest(envelope),
        budget=WriterEnvelopeWorkBudget(),
        operation="test.artifact_manifest.digest",
    )


def _drop_object(envelope, object_id):
    envelope["objects"] = [
        item for item in envelope["objects"] if item["object_id"] != object_id
    ]
    envelope["metrics"] = artifact_envelope_module.artifact_metrics(
        envelope["objects"],
        roots=envelope["roots"],
    )
    envelope["digest"] = _digest_terms_bounded(
        artifact_manifest(envelope),
        budget=WriterEnvelopeWorkBudget(),
        operation="test.artifact_manifest.digest",
    )


def _replace_ref(value, old_id, new_id):
    if isinstance(value, dict):
        for key, item in value.items():
            if item == old_id:
                value[key] = new_id
            else:
                _replace_ref(item, old_id, new_id)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            if item == old_id:
                value[index] = new_id
            else:
                _replace_ref(item, old_id, new_id)


def _json_round_trip(envelope):
    return json.loads(json.dumps(envelope, sort_keys=True))



if __name__ == "__main__":
    unittest.main()
