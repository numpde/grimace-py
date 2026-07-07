"""Table-backed writer support artifact envelope tests."""

from __future__ import annotations

from copy import deepcopy
import json
import unittest

from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.writer_envelope_terms import _canonical_json
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_snapshot import capture_writer_frontier_snapshot
from grimace._south_star1.writer_snapshot_prefix_envelope import (
    writer_snapshot_prefix_read_envelope_for_emitted_texts,
)
from grimace._south_star1.writer_support_artifact_envelope import (
    verify_writer_support_artifact_consistency,
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
from tests.south_star1.test_writer_snapshot import two_atom_facts


class WriterSupportArtifactEnvelopeTest(unittest.TestCase):
    def test_snapshot_source_artifact_json_round_trips(self) -> None:
        envelope = _json_round_trip(_snapshot_artifact())

        verification = verify_writer_support_artifact_consistency(envelope)

        self.assertTrue(verification.accepted)
        self.assertEqual(verification.support_count, 1)
        self.assertEqual(verification.witness_count, 2)

    def test_prefix_read_source_artifact_verifies(self) -> None:
        prepared = _prepare(two_atom_facts())
        prefix = writer_snapshot_prefix_read_envelope_for_emitted_texts(
            prepared=prepared,
            snapshot=_initial_snapshot(prepared),
            emitted_texts=("C", "C"),
        )
        envelope = writer_support_artifact_envelope_for_prefix_read(
            prepared=prepared,
            prefix_read_envelope=prefix,
        )

        verification = verify_writer_support_artifact_consistency(envelope)

        self.assertTrue(verification.accepted)
        self.assertEqual(verification.source_kind, "prefix_read")

    def test_branching_artifact_verifies(self) -> None:
        prepared = _prepare(cco_facts())
        envelope = writer_support_artifact_envelope_for_snapshot(
            prepared=prepared,
            snapshot=_initial_snapshot(prepared),
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

    def test_table_artifact_is_smaller_than_nested_support_image(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        snapshot = _initial_snapshot(prepared)
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


def _snapshot_artifact():
    prepared = _prepare(two_atom_facts())
    return deepcopy(
        writer_support_artifact_envelope_for_snapshot(
            prepared=prepared,
            snapshot=_initial_snapshot(prepared),
        )
    )


def _branching_artifact():
    prepared = _prepare(cco_facts())
    return deepcopy(
        writer_support_artifact_envelope_for_snapshot(
            prepared=prepared,
            snapshot=_initial_snapshot(prepared),
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


def _object(envelope, object_id):
    return next(item for item in envelope["objects"] if item["object_id"] == object_id)


def _json_round_trip(envelope):
    return json.loads(json.dumps(envelope, sort_keys=True))


def _initial_snapshot(prepared):
    options = _writer_options()
    return capture_writer_frontier_snapshot(
        prepared=prepared,
        runtime_options=options,
        cursor=initial_writer_frontier_cursor(prepared, options),
    )


def _writer_options():
    return SouthStarRuntimeOptions(
        rooted_at_atom=-1,
        canonical=False,
        do_random=True,
        serialization_language=SerializationLanguageMode.WRITER_SHAPED,
    )


def _prepare(facts):
    return prepare_south_star_mol_from_facts(
        facts,
        writer_surface=SouthStarWriterSurface(),
    )


if __name__ == "__main__":
    unittest.main()
