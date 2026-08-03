from __future__ import annotations

from copy import deepcopy
import unittest

from grimace._south_star1.writer_envelope_terms import _digest_terms_bounded
from grimace._south_star1.writer_envelope_terms import _identity_digest
from grimace._south_star1.writer_envelope_work import WriterEnvelopeWorkBudget
from tests.south_star1.writer_artifact_test_support import artifact_object_by_id
from tests.south_star1.writer_artifact_test_support import artifact_objects_by_kind
from tests.south_star1.writer_artifact_test_support import closed_term_digest
from tests.south_star1.writer_artifact_test_support import closed_term_field
from tests.south_star1.writer_artifact_test_support import find_closed_term
from tests.south_star1.writer_artifact_test_support import refresh_closed_term_digest_field
from tests.south_star1.writer_artifact_test_support import refresh_cursor_digest
from tests.south_star1.writer_artifact_test_support import refresh_kind_manifest_digest
from tests.south_star1.writer_artifact_test_support import set_closed_term_field
from tests.south_star1.writer_artifact_test_support import set_nested_closed_term_field
from tests.south_star1.writer_artifact_test_support import unique_artifact_object_by_kind


class WriterArtifactTestSupportTest(unittest.TestCase):
    def test_object_lookup_validates_shape_and_uniqueness(self):
        artifact = {"objects": [{"object_id": "a", "kind": "x"}, {"object_id": "b", "kind": "x"}]}
        self.assertEqual(artifact_object_by_id(artifact, "a")["kind"], "x")
        self.assertEqual(len(artifact_objects_by_kind(artifact, "x")), 2)
        with self.assertRaisesRegex(AssertionError, "exactly one artifact object kind"):
            unique_artifact_object_by_kind(artifact, "x")
        with self.assertRaisesRegex(AssertionError, "exactly one artifact object id"):
            artifact_object_by_id(artifact, "missing")
        with self.assertRaisesRegex(AssertionError, "mutable mapping"):
            artifact_object_by_id({"objects": [()]}, "a")

    def test_closed_term_access_preserves_order_and_rejects_duplicates(self):
        term = {"fields": [["a", 1], ["nested", {"fields": [["b", 2]]}], ["c", 3]]}
        self.assertEqual(closed_term_field(term, "a"), 1)
        set_closed_term_field(term, "a", 4)
        set_nested_closed_term_field(term, "nested", "b", value=5)
        self.assertEqual([field[0] for field in term["fields"]], ["a", "nested", "c"])
        self.assertEqual(find_closed_term({"outer": [term]}, "fields"), term)
        with self.assertRaisesRegex(AssertionError, "exactly one"):
            closed_term_field({"fields": [["a", 1], ["a", 2]]}, "a")
        with self.assertRaises(AssertionError):
            set_nested_closed_term_field(term, value=1)

    def test_find_closed_term_is_depth_first(self):
        left = {"marker": "left"}
        right = {"marker": "right"}
        self.assertIs(find_closed_term({"a": [left], "b": right}, "marker"), left)

    def test_digest_refreshes_match_direct_terms_and_only_declared_fields(self):
        term = {"fields": [["a", 1]]}
        container = {"terms": deepcopy(term), "digest": "old", "other": "unchanged"}
        expected = _identity_digest(term, budget=WriterEnvelopeWorkBudget(), operation="test.direct")
        self.assertEqual(
            refresh_closed_term_digest_field(
                container,
                term_field="terms",
                digest_field="digest",
                operation="test.direct",
            ),
            expected,
        )
        self.assertEqual(container["digest"], expected)
        self.assertEqual(container["other"], "unchanged")
        cursor = {"terms": term, "digest": "old"}
        self.assertEqual(refresh_cursor_digest(cursor, operation="test.cursor"), cursor["digest"])
        manifest = {"kind": "e", "manifest": {"x": 1}, "digest": "old", "other": 2}
        digest = refresh_kind_manifest_digest(manifest, operation="test.manifest")
        self.assertEqual(manifest["digest"], digest)
        self.assertEqual(manifest["other"], 2)
        self.assertEqual(closed_term_digest(term, operation="test.term"), _identity_digest(term, budget=WriterEnvelopeWorkBudget(), operation="test.term"))
