import unittest

from tests.south_star1.writer_support_artifact_fixtures import (
    rdkit_support_artifact_fixture,
)
from tests.south_star1.writer_support_artifact_graph_test_support import (
    first_closure_evidence_item,
)


class WriterSupportArtifactGraphSupportTest(unittest.TestCase):
    def test_closure_evidence_selector_returns_one_item(self):
        artifact = rdkit_support_artifact_fixture("C1=CC1").artifact
        item = first_closure_evidence_item(artifact)
        self.assertIsInstance(item, dict)


if __name__ == "__main__":
    unittest.main()
