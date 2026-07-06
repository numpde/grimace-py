"""Durable writer support-image envelope tests."""

from __future__ import annotations

from copy import deepcopy
import inspect
import json
import unittest

from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_snapshot import capture_writer_frontier_snapshot
from grimace._south_star1.writer_snapshot_prefix_envelope import (
    writer_snapshot_prefix_read_envelope_for_emitted_texts,
)
from grimace._south_star1.writer_support_image_envelope import (
    verify_writer_support_image_envelope,
)
from grimace._south_star1.writer_support_image_envelope import (
    writer_support_image_envelope_for_prefix_read,
)
from grimace._south_star1.writer_support_image_envelope import (
    writer_support_image_envelope_for_snapshot,
)
from tests.south_star1.helpers import cyclopropane_facts
from tests.south_star1.helpers import tetrahedral_facts
from tests.south_star1.test_writer_snapshot import two_atom_facts


_PREPARED_CACHE = None
_SNAPSHOT_ENVELOPE_CACHE = None
_TERMINAL_PREFIX_CACHE = None
_TERMINAL_IMAGE_CACHE = None


class WriterSupportImageEnvelopeTest(unittest.TestCase):
    def test_initial_snapshot_support_image_json_round_trips(self) -> None:
        envelope = _json_round_trip(_snapshot_envelope())
        verification = _verify(envelope)

        self.assertTrue(verification.accepted)
        self.assertEqual(verification.distinct_count, envelope["distinct_count"])
        self.assertEqual(verification.witness_count, envelope["witness_count"])

    def test_prefix_read_support_image_json_round_trips(self) -> None:
        prepared, prefix = _terminal_prefix_read_envelope()
        envelope = _json_round_trip(
            _terminal_image_envelope()
        )

        verification = verify_writer_support_image_envelope(
            prepared=prepared,
            envelope=envelope,
        )

        self.assertTrue(verification.accepted)
        self.assertEqual(verification.source_kind, "prefix_read")

    def test_terminal_only_support_image_verifies(self) -> None:
        prepared, prefix = _terminal_prefix_read_envelope()
        envelope = _terminal_image_envelope()

        self.assertEqual(envelope["support_strings"], [""])
        self.assertIsNotNone(envelope["enumeration_coverage"]["terminal_bucket"])
        self.assertTrue(
            verify_writer_support_image_envelope(
                prepared=prepared,
                envelope=envelope,
            ).accepted
        )

    def test_branching_support_image_verifies(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        envelope = writer_support_image_envelope_for_snapshot(
            prepared=prepared,
            snapshot=_initial_snapshot(prepared),
        )

        self.assertTrue(
            verify_writer_support_image_envelope(
                prepared=prepared,
                envelope=envelope,
            ).accepted
        )
        self.assertGreater(len(envelope["enumeration_coverage"]["text_buckets"]), 0)

    def test_counts_bind_to_count_envelope(self) -> None:
        envelope = _snapshot_envelope()

        self.assertEqual(
            envelope["distinct_count"],
            envelope["count_envelope"]["support_count"],
        )
        self.assertEqual(
            envelope["witness_count"],
            envelope["count_envelope"]["completion_count"],
        )
        self.assertTrue(_verify(envelope).accepted)

    def test_unknown_schema_name_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["schema_name"] = "other"

        self.assertFalse(_verify(envelope).accepted)

    def test_unknown_schema_version_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["schema_version"] = 999

        self.assertFalse(_verify(envelope).accepted)

    def test_extra_top_level_field_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["extra"] = {}

        self.assertFalse(_verify(envelope).accepted)

    def test_missing_required_field_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        del envelope["enumeration_coverage"]

        self.assertFalse(_verify(envelope).accepted)

    def test_wrong_prepared_identity_is_rejected(self) -> None:
        envelope = _snapshot_envelope()

        self.assertFalse(
            verify_writer_support_image_envelope(
                prepared=_prepare(tetrahedral_facts()),
                envelope=envelope,
            ).accepted
        )

    def test_wrong_source_kind_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["source_kind"] = "choice_snapshot"

        self.assertFalse(_verify(envelope).accepted)

    def test_wrong_source_snapshot_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["source_snapshot"]["cursor"]["digest"] = "0" * 64

        self.assertFalse(_verify(envelope).accepted)

    def test_wrong_prefix_read_envelope_is_rejected(self) -> None:
        prepared, prefix = _terminal_prefix_read_envelope()
        envelope = _terminal_image_envelope()
        envelope["prefix_read_envelope"]["schema_name"] = "bad"

        self.assertFalse(
            verify_writer_support_image_envelope(
                prepared=prepared,
                envelope=envelope,
            ).accepted
        )

    def test_wrong_count_envelope_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["count_envelope"]["schema_name"] = "bad"

        self.assertFalse(_verify(envelope).accepted)

    def test_changed_support_strings_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["support_strings"][0] += "C"

        self.assertFalse(_verify(envelope).accepted)

    def test_duplicate_support_string_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["support_strings"].append(envelope["support_strings"][0])
        envelope["support_string_envelopes"].append(
            deepcopy(envelope["support_string_envelopes"][0])
        )

        self.assertFalse(_verify(envelope).accepted)

    def test_removed_support_string_envelope_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["support_string_envelopes"] = []

        self.assertFalse(_verify(envelope).accepted)

    def test_extra_support_string_envelope_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["support_string_envelopes"].append(
            deepcopy(envelope["support_string_envelopes"][0])
        )

        self.assertFalse(_verify(envelope).accepted)

    def test_support_string_from_wrong_source_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        prepared, prefix = _terminal_prefix_read_envelope()
        terminal = _terminal_image_envelope()
        envelope["support_string_envelopes"][0] = (
            terminal["support_string_envelopes"][0]
        )

        self.assertFalse(_verify(envelope).accepted)

    def test_changed_distinct_count_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["distinct_count"] += 1

        self.assertFalse(_verify(envelope).accepted)

    def test_changed_witness_count_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["witness_count"] += 1

        self.assertFalse(_verify(envelope).accepted)

    def test_changed_checked_frontier_identity_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["checked_frontier_certificate"]["digest"] = "1" * 64

        self.assertFalse(_verify(envelope).accepted)

    def test_changed_support_count_certificate_identity_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["support_count_certificate"]["digest"] = "2" * 64

        self.assertFalse(_verify(envelope).accepted)

    def test_changed_witness_count_certificate_identity_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["witness_count_certificate"]["digest"] = "3" * 64

        self.assertFalse(_verify(envelope).accepted)

    def test_removed_text_bucket_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["enumeration_coverage"]["text_buckets"] = []

        self.assertFalse(_verify(envelope).accepted)

    def test_extra_text_bucket_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        bucket = deepcopy(envelope["enumeration_coverage"]["text_buckets"][0])
        envelope["enumeration_coverage"]["text_buckets"].append(bucket)

        self.assertFalse(_verify(envelope).accepted)

    def test_wrong_text_bucket_projection_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        bucket = envelope["enumeration_coverage"]["text_buckets"][0]
        bucket["text_projection"]["digest"] = "4" * 64

        self.assertFalse(_verify(envelope).accepted)

    def test_wrong_bucket_string_index_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        bucket = envelope["enumeration_coverage"]["text_buckets"][0]
        bucket["string_indices"] = []

        self.assertFalse(_verify(envelope).accepted)

    def test_same_string_assigned_to_two_buckets_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        bucket = deepcopy(envelope["enumeration_coverage"]["text_buckets"][0])
        envelope["enumeration_coverage"]["text_buckets"].append(bucket)

        self.assertFalse(_verify(envelope).accepted)

    def test_terminal_bucket_missing_when_empty_string_exists_is_rejected(self) -> None:
        prepared, prefix = _terminal_prefix_read_envelope()
        envelope = _terminal_image_envelope()
        envelope["enumeration_coverage"]["terminal_bucket"] = None

        self.assertFalse(
            verify_writer_support_image_envelope(
                prepared=prepared,
                envelope=envelope,
            ).accepted
        )

    def test_terminal_bucket_present_without_terminal_coverage_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["enumeration_coverage"]["terminal_bucket"] = {
            "terminal_projection": None,
            "terminal_support_term_digest": None,
            "terminal_support_identities": [],
            "support_count": 0,
            "string_index": None,
            "string_digest": None,
            "digest": "0" * 64,
        }

        self.assertFalse(_verify(envelope).accepted)

    def test_terminal_support_identity_changed_is_rejected(self) -> None:
        prepared, prefix = _terminal_prefix_read_envelope()
        envelope = _terminal_image_envelope()
        identity = (
            envelope["enumeration_coverage"]
            ["terminal_bucket"]
            ["terminal_support_identities"][0]
        )
        identity["terminal_ordinal"] += 1

        self.assertFalse(
            verify_writer_support_image_envelope(
                prepared=prepared,
                envelope=envelope,
            ).accepted
        )

    def test_coverage_support_count_changed_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["enumeration_coverage"]["support_count"] += 1

        self.assertFalse(_verify(envelope).accepted)

    def test_producer_does_not_call_per_string_search_helper(self) -> None:
        import grimace._south_star1.writer_support_image_envelope as module

        source = inspect.getsource(module.writer_support_image_envelope_for_snapshot)
        self.assertNotIn("writer_support_string_envelope_for_string", source)

    def test_verifier_does_not_enumerate_support_as_authority(self) -> None:
        import grimace._south_star1.writer_support_image_envelope as module

        source = inspect.getsource(module.verify_writer_support_image_envelope)
        self.assertNotIn("_iter_writer_snapshot_certified_support_strings", source)
        self.assertNotIn("enumerate_prepared_writer_shaped_support", source)
        self.assertNotIn("enumerate_writer_snapshot_writer_shaped_support", source)


def _snapshot_envelope():
    global _SNAPSHOT_ENVELOPE_CACHE
    if _SNAPSHOT_ENVELOPE_CACHE is None:
        prepared = _prepared()
        _SNAPSHOT_ENVELOPE_CACHE = writer_support_image_envelope_for_snapshot(
            prepared=prepared,
            snapshot=_initial_snapshot(prepared),
        )
    return deepcopy(_SNAPSHOT_ENVELOPE_CACHE)


def _terminal_prefix_read_envelope():
    global _TERMINAL_PREFIX_CACHE
    if _TERMINAL_PREFIX_CACHE is None:
        prepared = _prepared()
        snapshot = _initial_snapshot(prepared)
        _TERMINAL_PREFIX_CACHE = (
            prepared,
            writer_snapshot_prefix_read_envelope_for_emitted_texts(
                prepared=prepared,
                snapshot=snapshot,
                emitted_texts=("C", "C"),
            ),
        )
    prepared, prefix = _TERMINAL_PREFIX_CACHE
    return prepared, deepcopy(prefix)


def _terminal_image_envelope():
    global _TERMINAL_IMAGE_CACHE
    if _TERMINAL_IMAGE_CACHE is None:
        prepared, prefix = _terminal_prefix_read_envelope()
        _TERMINAL_IMAGE_CACHE = writer_support_image_envelope_for_prefix_read(
            prepared=prepared,
            prefix_read_envelope=prefix,
        )
    return deepcopy(_TERMINAL_IMAGE_CACHE)


def _verify(envelope):
    return verify_writer_support_image_envelope(
        prepared=_prepared(),
        envelope=envelope,
    )


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


def _prepared():
    global _PREPARED_CACHE
    if _PREPARED_CACHE is None:
        _PREPARED_CACHE = _prepare(two_atom_facts())
    return _PREPARED_CACHE


if __name__ == "__main__":
    unittest.main()
