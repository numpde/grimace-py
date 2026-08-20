"""Structural consistency verifier tests for durable writer envelopes."""

from __future__ import annotations

from tests.south_star1.writer_test_context import initial_writer_snapshot
from tests.south_star1.writer_test_context import prepare_writer_facts
from tests.south_star1.writer_test_context import writer_runtime_options

from copy import deepcopy
import inspect
import json
import unittest
import ast

from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.writer_envelope_consistency import (
    verify_writer_support_image_envelope_consistency,
)
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_snapshot import capture_writer_frontier_snapshot
from grimace._south_star1.writer_snapshot_prefix_envelope import (
    writer_snapshot_prefix_read_envelope_for_emitted_texts,
)
from grimace._south_star1.writer_support_image_envelope import (
    writer_support_image_envelope_for_prefix_read,
)
from grimace._south_star1.writer_support_image_envelope import (
    writer_support_image_envelope_for_snapshot,
)
from tests.south_star1.helpers import cyclopropane_facts
from tests.south_star1.helpers import two_atom_facts


_PREPARED_CACHE = None
_SNAPSHOT_ENVELOPE_CACHE = None
_TERMINAL_ENVELOPE_CACHE = None


class WriterEnvelopeConsistencyTest(unittest.TestCase):
    def test_snapshot_source_support_image_consistency_verifies(self) -> None:
        envelope = _snapshot_envelope()
        verification = verify_writer_support_image_envelope_consistency(envelope)

        self.assertTrue(verification.accepted)
        self.assertEqual(verification.support_count, envelope["distinct_count"])
        self.assertEqual(verification.witness_count, envelope["witness_count"])

    def test_prefix_read_support_image_consistency_verifies(self) -> None:
        envelope = _terminal_envelope()
        verification = verify_writer_support_image_envelope_consistency(envelope)

        self.assertTrue(verification.accepted)
        self.assertEqual(verification.schema_name, "writer_support_image")

    def test_empty_eos_support_image_consistency_verifies(self) -> None:
        envelope = _terminal_envelope()

        self.assertEqual(envelope["support_strings"], [""])
        self.assertTrue(
            verify_writer_support_image_envelope_consistency(envelope).accepted
        )

    def test_branching_support_image_consistency_verifies(self) -> None:
        prepared = prepare_writer_facts(cyclopropane_facts())
        envelope = writer_support_image_envelope_for_snapshot(
            prepared=prepared,
            snapshot=initial_writer_snapshot(prepared, writer_runtime_options()),
        )

        self.assertTrue(
            verify_writer_support_image_envelope_consistency(envelope).accepted
        )

    def test_json_loaded_envelope_consistency_verifies(self) -> None:
        envelope = json.loads(json.dumps(_snapshot_envelope(), sort_keys=True))

        self.assertTrue(
            verify_writer_support_image_envelope_consistency(envelope).accepted
        )

    def test_wrong_nested_prepared_identity_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["count_envelope"]["prepared_identity"]["digest"] = "0" * 64

        self.assertFalse(_verify(envelope).accepted)

    def test_count_envelope_source_mismatch_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["count_envelope"]["frontier_snapshot"]["cursor"]["digest"] = (
            "1" * 64
        )

        self.assertFalse(_verify(envelope).accepted)

    def test_support_string_source_mismatch_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["support_string_envelopes"][0]["source_snapshot"]["cursor"][
            "digest"
        ] = "2" * 64

        self.assertFalse(_verify(envelope).accepted)

    def test_support_string_count_envelope_mismatch_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["support_string_envelopes"][0]["count_envelope"][
            "support_count"
        ] += 1

        self.assertFalse(_verify(envelope).accepted)

    def test_support_strings_order_mismatch_is_rejected(self) -> None:
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

    def test_wrong_distinct_count_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["distinct_count"] += 1

        self.assertFalse(_verify(envelope).accepted)

    def test_wrong_witness_count_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["witness_count"] += 1

        self.assertFalse(_verify(envelope).accepted)

    def test_replay_step_source_mismatch_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        step = envelope["support_string_envelopes"][0]["replay_envelope"][
            "step_advance_envelopes"
        ][0]
        step["source_snapshot"]["cursor"]["digest"] = "3" * 64

        self.assertFalse(_verify(envelope).accepted)

    def test_replay_step_successor_mismatch_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        step = envelope["support_string_envelopes"][0]["replay_envelope"][
            "step_advance_envelopes"
        ][0]
        step["advance_certificate"]["selected_text_projection"][
            "successor_cursor"
        ]["digest"] = "4" * 64

        self.assertFalse(_verify(envelope).accepted)

    def test_decoder_boundary_non_increment_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        step = envelope["support_string_envelopes"][0]["replay_envelope"][
            "step_advance_envelopes"
        ][0]
        _bump_decoder_boundary(
            step["advance_certificate"]["step_certificate"][
                "decoder_boundary_after"
            ]
        )

        self.assertFalse(_verify(envelope).accepted)

    def test_text_projection_chain_order_mismatch_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        chain = envelope["support_string_envelopes"][0]["text_projection_chain"]
        chain[0]["step_index"] += 1

        self.assertFalse(_verify(envelope).accepted)

    def test_text_projection_chain_replay_mismatch_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        chain = envelope["support_string_envelopes"][0]["text_projection_chain"]
        chain[0]["text_projection"]["digest"] = "5" * 64

        self.assertFalse(_verify(envelope).accepted)

    def test_terminal_projection_final_snapshot_mismatch_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        terminal = envelope["support_string_envelopes"][0]["terminal_projection"]
        terminal["source_cursor"]["digest"] = "6" * 64

        self.assertFalse(_verify(envelope).accepted)

    def test_terminal_support_identity_mismatch_is_rejected(self) -> None:
        envelope = _terminal_envelope()
        identity = envelope["support_string_envelopes"][0][
            "terminal_support_identities"
        ][0]
        identity["terminal_ordinal"] += 1

        self.assertFalse(_verify(envelope).accepted)

    def test_missing_text_bucket_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["enumeration_coverage"]["text_buckets"] = []

        self.assertFalse(_verify(envelope).accepted)

    def test_extra_text_bucket_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["enumeration_coverage"]["text_buckets"].append(
            deepcopy(envelope["enumeration_coverage"]["text_buckets"][0])
        )

        self.assertFalse(_verify(envelope).accepted)

    def test_wrong_bucket_string_index_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["enumeration_coverage"]["text_buckets"][0]["string_indices"] = []

        self.assertFalse(_verify(envelope).accepted)

    def test_same_string_index_in_two_buckets_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        bucket = deepcopy(envelope["enumeration_coverage"]["text_buckets"][0])
        envelope["enumeration_coverage"]["text_buckets"].append(bucket)

        self.assertFalse(_verify(envelope).accepted)

    def test_terminal_bucket_missing_for_empty_string_is_rejected(self) -> None:
        envelope = _terminal_envelope()
        envelope["enumeration_coverage"]["terminal_bucket"] = None

        self.assertFalse(_verify(envelope).accepted)

    def test_terminal_bucket_present_without_empty_string_is_rejected(self) -> None:
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

    def test_coverage_support_count_mismatch_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["enumeration_coverage"]["support_count"] += 1

        self.assertFalse(_verify(envelope).accepted)

    def test_frontier_product_identity_mismatch_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["frontier_product"]["digest"] = "7" * 64

        self.assertFalse(_verify(envelope).accepted)

    def test_checked_frontier_identity_mismatch_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["checked_frontier_certificate"]["digest"] = "8" * 64

        self.assertFalse(_verify(envelope).accepted)

    def test_support_count_certificate_mismatch_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["support_count_certificate"]["digest"] = "9" * 64

        self.assertFalse(_verify(envelope).accepted)

    def test_witness_count_certificate_mismatch_is_rejected(self) -> None:
        envelope = _snapshot_envelope()
        envelope["witness_count_certificate"]["digest"] = "a" * 64

        self.assertFalse(_verify(envelope).accepted)

    def test_consistency_verifier_does_not_import_live_runtime_modules(self) -> None:
        import grimace._south_star1.writer_envelope_consistency as module

        source = inspect.getsource(module)
        tree = ast.parse(source)
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        imported_names = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        for name in (
            "writer_frontier",
            "writer_runtime",
            "writer_snapshot",
            "writer_support",
            "writer_support_certificates",
            "rdkit",
            "choice_snapshot",
            "MolToSmiles" + "EnumS",
        ):
            self.assertNotIn(name, imported_modules)
            self.assertNotIn(name, imported_names)
        for forbidden_call in (
            "_checked_writer_frontier_product",
            "_snapshot_advance_writer_frontier_product",
            "_iter_writer_snapshot_certified_support_strings",
            "iter_writer_runtime_certified_support",
            "enumerate_writer_snapshot_writer_shaped_support",
        ):
            self.assertNotIn(forbidden_call, source)


def _snapshot_envelope():
    global _SNAPSHOT_ENVELOPE_CACHE
    if _SNAPSHOT_ENVELOPE_CACHE is None:
        prepared = prepared_two_atom_facts()
        _SNAPSHOT_ENVELOPE_CACHE = writer_support_image_envelope_for_snapshot(
            prepared=prepared,
            snapshot=initial_writer_snapshot(prepared, writer_runtime_options()),
        )
    return deepcopy(_SNAPSHOT_ENVELOPE_CACHE)


def _terminal_envelope():
    global _TERMINAL_ENVELOPE_CACHE
    if _TERMINAL_ENVELOPE_CACHE is None:
        prepared = prepared_two_atom_facts()
        snapshot = initial_writer_snapshot(prepared, writer_runtime_options())
        prefix = writer_snapshot_prefix_read_envelope_for_emitted_texts(
            prepared=prepared,
            snapshot=snapshot,
            emitted_texts=("C", "C"),
        )
        _TERMINAL_ENVELOPE_CACHE = writer_support_image_envelope_for_prefix_read(
            prepared=prepared,
            prefix_read_envelope=prefix,
        )
    return deepcopy(_TERMINAL_ENVELOPE_CACHE)


def _verify(envelope):
    return verify_writer_support_image_envelope_consistency(envelope)



def prepared_two_atom_facts():
    global _PREPARED_CACHE
    if _PREPARED_CACHE is None:
        _PREPARED_CACHE = prepare_writer_facts(two_atom_facts())
    return _PREPARED_CACHE


def _bump_decoder_boundary(boundary):
    if "consumed_token_count" in boundary:
        boundary["consumed_token_count"] += 1
        return
    for field in boundary["fields"]:
        if field[0] == "consumed_token_count":
            field[1] += 1
            return
    raise AssertionError("missing consumed_token_count")


if __name__ == "__main__":
    unittest.main()
