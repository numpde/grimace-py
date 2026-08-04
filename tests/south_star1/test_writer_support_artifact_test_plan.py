"""Architecture checks for bounded rich support-artifact test lanes."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

from tests.south_star1.writer_support_artifact_test_plan import (
    WRITER_SUPPORT_ARTIFACT_TEST_DOMAINS,
)
from tests.south_star1.writer_support_artifact_test_plan import bounded_domains
from tests.south_star1.writer_support_artifact_test_plan import (
    validate_writer_support_artifact_test_plan,
)
from tests.south_star1.writer_support_artifact_test_plan import test_ids_for_domain


class WriterSupportArtifactTestPlanTest(unittest.TestCase):
    def test_plan_validates_and_has_unique_inventory(self):
        validate_writer_support_artifact_test_plan()
        all_ids = [
            test_id
            for domain in WRITER_SUPPORT_ARTIFACT_TEST_DOMAINS
            for test_id in test_ids_for_domain(domain)
        ]
        self.assertEqual(len(all_ids), len(set(all_ids)))

    def test_modules_do_not_import_test_modules(self):
        root = Path(__file__).parent
        for domain in WRITER_SUPPORT_ARTIFACT_TEST_DOMAINS:
            for module_name in domain.modules:
                tree = ast.parse(
                    (root / (module_name.rsplit(".", 1)[1] + ".py")).read_text(
                        encoding="utf-8"
                    )
                )
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        self.assertFalse(node.module.startswith("tests.south_star1.test_"))
                    if isinstance(node, ast.Import):
                        self.assertFalse(
                            any(alias.name.startswith("tests.south_star1.test_") for alias in node.names)
                        )

    def test_only_slow_module_mentions_slow_gate(self):
        root = Path(__file__).parent
        for domain in WRITER_SUPPORT_ARTIFACT_TEST_DOMAINS:
            for module_name in domain.modules:
                path = root / (module_name.rsplit(".", 1)[1] + ".py")
                tree = ast.parse(path.read_text(encoding="utf-8"))
                has_gate = any(
                    isinstance(node, ast.Attribute)
                    and node.attr == "environ"
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "os"
                    for node in ast.walk(tree)
                )
                self.assertEqual(has_gate, domain.kind == "slow-diagnostic", str(path))

    def test_domain_and_fixture_line_bounds(self):
        root = Path(__file__).parent
        for domain in WRITER_SUPPORT_ARTIFACT_TEST_DOMAINS:
            for module_name in domain.modules:
                path = root / (module_name.rsplit(".", 1)[1] + ".py")
                lines = len(path.read_text(encoding="utf-8").splitlines())
                self.assertLessEqual(lines, 1200, str(path))


if __name__ == "__main__":
    unittest.main()
