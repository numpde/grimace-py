import unittest

from tests.south_star1.writer_support_artifact_fixtures import (
    tetra_support_artifact_fixture,
)
from tests.south_star1.writer_support_artifact_tetra_test_support import (
    different_local_order_digest,
)
from tests.south_star1.writer_support_artifact_queries import (
    first_residual_work_branch,
)


class WriterSupportArtifactTetraSupportTest(unittest.TestCase):
    def test_tetra_fixture_and_local_order_query_are_available(self):
        fixture = tetra_support_artifact_fixture()
        branch = first_residual_work_branch(
            fixture.artifact,
            operation="tetrahedral atom-token restriction",
        )
        digest = different_local_order_digest(
            fixture.artifact,
            branch=branch,
            cursor_name="successor_cursor",
            atom=fixture.facts.stereo.tetrahedral[0].center,
        )
        self.assertEqual(len(digest), 64)


if __name__ == "__main__":
    unittest.main()
