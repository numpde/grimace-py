from __future__ import annotations

import unittest
from unittest.mock import patch

from tests.south_star1.helpers import cco_facts
from tests.south_star1.writer_test_context import (
    initial_writer_snapshot,
    prepare_writer_facts,
    writer_runtime_options,
    writer_test_context,
)


class WriterTestContextTest(unittest.TestCase):
    def test_context_is_the_three_primitive_composition(self) -> None:
        facts = cco_facts()
        with patch("tests.south_star1.writer_test_context.writer_runtime_options", wraps=writer_runtime_options) as options, patch("tests.south_star1.writer_test_context.prepare_writer_facts", wraps=prepare_writer_facts) as prepare, patch("tests.south_star1.writer_test_context.initial_writer_snapshot", wraps=initial_writer_snapshot) as snapshot:
            context = writer_test_context(facts, rooted_at_atom=1)
        self.assertEqual(options.call_args.kwargs, {"rooted_at_atom": 1})
        self.assertEqual(prepare.call_count, 1)
        self.assertEqual(snapshot.call_count, 1)
        self.assertEqual(context.facts, facts)
        self.assertEqual(prepare.call_args.args, (facts,))
        self.assertEqual(snapshot.call_args.args, (context.prepared, context.runtime_options))

    def test_runtime_options_pin_writer_regime_and_root(self) -> None:
        options = writer_runtime_options(rooted_at_atom=3)
        self.assertEqual(options.rooted_at_atom, 3)
        self.assertFalse(options.canonical)
        self.assertTrue(options.do_random)


if __name__ == "__main__":
    unittest.main()
