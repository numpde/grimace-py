from __future__ import annotations

import unittest
from unittest.mock import patch

from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.writer_prepared_identity import writer_prepared_identity
from grimace._south_star1.writer_snapshot import capture_writer_frontier_snapshot
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
        self.assertEqual(context.prepared.facts, facts)
        self.assertEqual(prepare.call_args.args, (facts,))
        self.assertEqual(snapshot.call_args.args, (context.prepared, context.runtime_options))

    def test_runtime_options_pin_writer_regime_and_root(self) -> None:
        options = writer_runtime_options(rooted_at_atom=3)
        self.assertEqual(
            options,
            type(options)(
                rooted_at_atom=3,
                canonical=False,
                do_random=True,
                serialization_language=SerializationLanguageMode.WRITER_SHAPED,
            ),
        )
        self.assertEqual(options.rooted_at_atom, 3)
        self.assertFalse(options.canonical)
        self.assertTrue(options.do_random)

    def test_existing_options_are_preserved_and_root_is_exclusive(self) -> None:
        facts = cco_facts()
        options = writer_runtime_options(rooted_at_atom=2)
        context = writer_test_context(facts, runtime_options=options)
        self.assertIs(context.runtime_options, options)
        with self.assertRaises(ValueError):
            writer_test_context(facts, runtime_options=options, rooted_at_atom=2)

    def test_explicit_policy_and_facts_belong_to_prepared_authority(self) -> None:
        facts = cco_facts()
        from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts

        policy = ordinary_policy_for_facts(facts)
        context = writer_test_context(facts, policy=policy)
        self.assertEqual(context.prepared.facts, facts)
        self.assertIs(context.prepared.policy, policy)

    def test_snapshot_is_direct_capture_with_bound_identity(self) -> None:
        facts = cco_facts()
        context = writer_test_context(facts, rooted_at_atom=1)
        direct = capture_writer_frontier_snapshot(
            prepared=context.prepared,
            runtime_options=context.runtime_options,
            cursor=context.initial_snapshot.cursor,
        )
        self.assertEqual(context.initial_snapshot, direct)
        self.assertIs(context.initial_snapshot.runtime_options, context.runtime_options)
        self.assertEqual(
            context.initial_snapshot.prepared_identity,
            writer_prepared_identity(context.prepared, context.runtime_options),
        )

    def test_context_has_exactly_three_fields(self) -> None:
        from dataclasses import fields

        self.assertEqual(
            tuple(field.name for field in fields(type(writer_test_context(cco_facts())))),
            ("runtime_options", "prepared", "initial_snapshot"),
        )


if __name__ == "__main__":
    unittest.main()
