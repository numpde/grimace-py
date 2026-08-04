import copy
import unittest

from tests.south_star1.writer_support_artifact_fixtures import (
    rdkit_support_artifact_fixture,
)
from tests.south_star1.writer_support_artifact_queries import (
    first_branch_support_object,
    first_text_projection_object,
)
from tests.south_star1.writer_support_artifact_transition_test_support import (
    propagate_text_projection_cursor_change,
)


class WriterSupportArtifactTransitionSupportTest(unittest.TestCase):
    def test_projection_propagation_updates_linked_references(self):
        artifact = rdkit_support_artifact_fixture("CCO").artifact
        projection = first_text_projection_object(artifact)
        old_cursor = projection["payload"]["source_cursor"]
        new_cursor = copy.deepcopy(old_cursor)
        new_cursor["digest"] = "f" * 64

        propagate_text_projection_cursor_change(
            artifact,
            old_cursor_digest=old_cursor["digest"],
            new_cursor=new_cursor,
        )

        self.assertEqual(
            first_text_projection_object(artifact)["payload"]["source_cursor"]["digest"],
            new_cursor["digest"],
        )
        self.assertEqual(
            first_branch_support_object(artifact)["payload"]["source_cursor_digest"],
            new_cursor["digest"],
        )


if __name__ == "__main__":
    unittest.main()
