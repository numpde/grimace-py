"""Tests for rich support-artifact query adapters."""

from __future__ import annotations

import copy
import unittest

from tests.south_star1.writer_support_artifact_fixtures import (
    rdkit_support_artifact_fixture,
)
from tests.south_star1.writer_support_artifact_queries import (
    first_support_string_object,
)
from tests.south_star1.writer_support_artifact_queries import (
    require_structurally_valid_support_artifact,
)
from tests.south_star1.writer_support_artifact_queries import support_artifact_object_index
from tests.south_star1.writer_support_artifact_queries import support_image_root_object
from tests.south_star1.writer_support_artifact_queries import support_strings
from tests.south_star1.writer_support_artifact_queries import (
    verify_support_image_coverage_relation,
)


class WriterSupportArtifactQueriesTest(unittest.TestCase):
    def test_index_and_deterministic_selectors(self):
        artifact = rdkit_support_artifact_fixture("CCO").artifact
        index = support_artifact_object_index(artifact)
        self.assertEqual(len(index), len(artifact["objects"]))
        self.assertEqual(support_image_root_object(artifact)["kind"], "support_image")
        self.assertGreater(len(support_strings(artifact)), 0)
        self.assertEqual(first_support_string_object(artifact)["kind"], "support_string")

    def test_duplicate_ids_reject(self):
        artifact = rdkit_support_artifact_fixture("CCO").artifact
        duplicate = copy.deepcopy(artifact)
        duplicate["objects"].append(copy.deepcopy(duplicate["objects"][0]))
        with self.assertRaises(AssertionError):
            support_artifact_object_index(duplicate)

    def test_structural_and_offline_adapters(self):
        artifact = rdkit_support_artifact_fixture("CCO").artifact
        require_structurally_valid_support_artifact(artifact)
        self.assertTrue(verify_support_image_coverage_relation(artifact).accepted)


if __name__ == "__main__":
    unittest.main()
