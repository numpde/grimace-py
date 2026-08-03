from __future__ import annotations

import ast
from pathlib import Path
import unittest

from tests.south_star1.helpers import cco_facts
from tests.south_star1.writer_proof_sources import (
    first_terminal_proof_source,
    shared_ring_branch_source,
    shared_ring_branch_sources,
)
from tests.south_star1.writer_test_context import writer_runtime_options


class WriterProofSourcesTest(unittest.TestCase):
    def test_shared_ring_sources_are_immutable_and_keyed(self) -> None:
        sources = shared_ring_branch_sources()
        self.assertEqual(len(sources), 6)
        keys = [(source.phase, source.direction_mark) for source in sources]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(
            keys,
            sorted(keys, key=lambda item: (item[0], item[1].value)),
        )
        self.assertIs(shared_ring_branch_source(*keys[0]), sources[0])

    def test_terminal_selector_preserves_supplied_context(self) -> None:
        options = writer_runtime_options(rooted_at_atom=0)
        source = first_terminal_proof_source(cco_facts(), options)
        self.assertEqual(source.runtime_options, options)
        self.assertIs(source.facts, source.facts)
        self.assertIsNotNone(source.support)

    def test_shared_support_modules_import_no_test_modules(self) -> None:
        root = Path(__file__).parent
        for name in ("writer_test_context.py", "writer_test_fixtures.py", "writer_proof_sources.py"):
            tree = ast.parse((root / name).read_text(), filename=name)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("tests.south_star1.test_"):
                    self.fail(f"test-module import remains in {name}: {node.module}")


if __name__ == "__main__":
    unittest.main()
