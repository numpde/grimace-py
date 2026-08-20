"""Tests for copy-on-read rich support-artifact fixtures."""

from __future__ import annotations

import unittest

from grimace._south_star1.writer_support_artifact_envelope import (
    writer_support_artifact_envelope_for_snapshot,
)
from tests.south_star1.writer_support_artifact_fixtures import (
    completed_prefix_support_artifact_fixture,
)
from tests.south_star1.writer_support_artifact_fixtures import (
    rdkit_graph_facts,
)
from tests.south_star1.writer_support_artifact_fixtures import (
    rdkit_support_artifact_fixture,
)
from tests.south_star1.writer_support_artifact_fixtures import (
    support_artifact_fixture,
)
from tests.south_star1.writer_support_artifact_fixtures import (
    tetra_support_artifact_fixture,
)
from tests.south_star1.writer_test_context import initial_writer_snapshot
from tests.south_star1.writer_test_context import prepare_writer_facts


class WriterSupportArtifactFixturesTest(unittest.TestCase):
    def test_rdkit_fixture_is_copy_on_read(self):
        first = rdkit_support_artifact_fixture("CCO")
        second = rdkit_support_artifact_fixture("CCO")
        self.assertEqual(first.artifact, second.artifact)
        self.assertIsNot(first.artifact, second.artifact)
        first.artifact["objects"].clear()
        self.assertTrue(second.artifact["objects"])
        self.assertTrue(rdkit_support_artifact_fixture("CCO").artifact["objects"])

    def test_fixture_matches_direct_production_artifact(self):
        facts = rdkit_graph_facts("CCO")
        prepared = prepare_writer_facts(facts)
        expected = writer_support_artifact_envelope_for_snapshot(
            prepared=prepared,
            snapshot=initial_writer_snapshot(
                prepared,
                support_artifact_fixture(facts).runtime_options,
            ),
        )
        self.assertEqual(support_artifact_fixture(facts).artifact, expected)

    def test_named_fixtures_construct(self):
        self.assertTrue(completed_prefix_support_artifact_fixture().artifact)
        self.assertTrue(tetra_support_artifact_fixture().artifact)

if __name__ == "__main__":
    unittest.main()
