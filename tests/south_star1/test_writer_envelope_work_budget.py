"""Envelope work-budget fail-closed tests."""

from __future__ import annotations

import unittest

from grimace._south_star1.writer_envelope_work import (
    WriterEnvelopeWorkBudget,
)
from grimace._south_star1.writer_frontier_count_envelope import (
    writer_frontier_count_envelope_for_snapshot,
)
from grimace._south_star1.writer_snapshot_envelope import (
    verify_writer_snapshot_advance_envelope,
)
from grimace._south_star1.writer_snapshot_envelope import (
    writer_snapshot_advance_envelope_for_emitted_text,
)
from grimace._south_star1.writer_snapshot_replay_envelope import (
    verify_writer_snapshot_replay_envelope,
)
from grimace._south_star1.writer_snapshot_replay_envelope import (
    writer_snapshot_replay_envelope_for_emitted_texts,
)
from grimace._south_star1.writer_support_image_envelope import (
    verify_writer_support_image_envelope,
)
from grimace._south_star1.writer_support_image_envelope import (
    writer_support_image_envelope_for_snapshot,
)
from grimace._south_star1.writer_support_string_envelope import (
    writer_support_string_envelope_for_string,
)
from grimace._south_star1.writer_envelope_consistency import (
    verify_writer_support_image_envelope_consistency,
)
from tests.south_star1.helpers import cco_facts
from tests.south_star1.test_writer_frontier_count_envelope import (
    _first_choice_text,
)
from tests.south_star1.test_writer_frontier_count_envelope import (
    _initial_snapshot,
)
from tests.south_star1.test_writer_frontier_count_envelope import _prepare


class WriterEnvelopeWorkBudgetTest(unittest.TestCase):
    def test_count_dag_node_budget_exceeded_is_typed(self) -> None:
        prepared, snapshot = _prepared_snapshot()

        with self.assertRaisesRegex(
            Exception,
            "WRITER_ENVELOPE_WORK_EXCEEDED: .*count_node_count",
        ):
            writer_frontier_count_envelope_for_snapshot(
                prepared=prepared,
                snapshot=snapshot,
                budget=WriterEnvelopeWorkBudget(max_count_nodes=0),
            )

    def test_count_dag_edge_budget_exceeded_is_typed(self) -> None:
        prepared, snapshot = _prepared_snapshot()

        with self.assertRaisesRegex(
            Exception,
            "WRITER_ENVELOPE_WORK_EXCEEDED: .*count_edge_count",
        ):
            writer_frontier_count_envelope_for_snapshot(
                prepared=prepared,
                snapshot=snapshot,
                budget=WriterEnvelopeWorkBudget(max_count_edges=0),
            )

    def test_digest_term_byte_budget_exceeded_is_typed(self) -> None:
        prepared, snapshot = _prepared_snapshot()

        with self.assertRaisesRegex(
            Exception,
            "WRITER_ENVELOPE_WORK_EXCEEDED: .*digest_term_bytes",
        ):
            writer_frontier_count_envelope_for_snapshot(
                prepared=prepared,
                snapshot=snapshot,
                budget=WriterEnvelopeWorkBudget(max_digest_term_bytes=1),
            )

    def test_source_lookup_position_budget_exceeded_is_typed(self) -> None:
        prepared, snapshot = _prepared_snapshot()
        envelope = writer_snapshot_advance_envelope_for_emitted_text(
            prepared=prepared,
            snapshot=snapshot,
            emitted_text=_first_choice_text(prepared, snapshot),
        )

        result = verify_writer_snapshot_advance_envelope(
            prepared=prepared,
            envelope=envelope,
            budget=WriterEnvelopeWorkBudget(max_source_lookup_positions=0),
        )

        self.assertFalse(result.accepted)
        self.assertIn("WRITER_ENVELOPE_WORK_EXCEEDED", result.reason)
        self.assertIn("source_lookup_positions", result.reason)

    def test_replay_step_budget_exceeded_is_typed(self) -> None:
        prepared, snapshot = _prepared_snapshot()
        emitted = (_first_choice_text(prepared, snapshot),)

        with self.assertRaisesRegex(
            Exception,
            "WRITER_ENVELOPE_WORK_EXCEEDED: .*replay_step_count",
        ):
            writer_snapshot_replay_envelope_for_emitted_texts(
                prepared=prepared,
                snapshot=snapshot,
                emitted_texts=emitted,
                budget=WriterEnvelopeWorkBudget(max_replay_steps=0),
            )

    def test_replay_verify_step_budget_exceeded_is_typed(self) -> None:
        prepared, snapshot = _prepared_snapshot()
        emitted = (_first_choice_text(prepared, snapshot),)
        envelope = writer_snapshot_replay_envelope_for_emitted_texts(
            prepared=prepared,
            snapshot=snapshot,
            emitted_texts=emitted,
        )

        result = verify_writer_snapshot_replay_envelope(
            prepared=prepared,
            envelope=envelope,
            budget=WriterEnvelopeWorkBudget(max_replay_steps=0),
        )

        self.assertFalse(result.accepted)
        self.assertIn("WRITER_ENVELOPE_WORK_EXCEEDED", result.reason)
        self.assertIn("replay_step_count", result.reason)

    def test_support_string_search_budget_exceeded_is_typed(self) -> None:
        prepared, snapshot = _prepared_snapshot()

        with self.assertRaisesRegex(
            Exception,
            "WRITER_ENVELOPE_WORK_EXCEEDED: .*visited_support_strings",
        ):
            writer_support_string_envelope_for_string(
                prepared=prepared,
                snapshot=snapshot,
                string="C",
                budget=WriterEnvelopeWorkBudget(max_support_search_strings=0),
            )

    def test_support_image_string_budget_exceeded_is_typed(self) -> None:
        prepared, snapshot = _prepared_snapshot()

        with self.assertRaisesRegex(
            Exception,
            "WRITER_ENVELOPE_WORK_EXCEEDED: .*support_string_count",
        ):
            writer_support_image_envelope_for_snapshot(
                prepared=prepared,
                snapshot=snapshot,
                budget=WriterEnvelopeWorkBudget(max_support_strings=0),
            )

    def test_support_image_verify_assignment_budget_exceeded_is_typed(self) -> None:
        prepared, snapshot = _prepared_snapshot()
        envelope = writer_support_image_envelope_for_snapshot(
            prepared=prepared,
            snapshot=snapshot,
        )

        result = verify_writer_support_image_envelope(
            prepared=prepared,
            envelope=envelope,
            budget=WriterEnvelopeWorkBudget(max_bucket_assignments=0),
        )

        self.assertFalse(result.accepted)
        self.assertIn("WRITER_ENVELOPE_WORK_EXCEEDED", result.reason)
        self.assertIn("bucket_assignment_count", result.reason)

    def test_consistency_nested_envelope_budget_exceeded_is_typed(self) -> None:
        prepared, snapshot = _prepared_snapshot()
        envelope = writer_support_image_envelope_for_snapshot(
            prepared=prepared,
            snapshot=snapshot,
        )

        result = verify_writer_support_image_envelope_consistency(
            envelope,
            budget=WriterEnvelopeWorkBudget(max_nested_envelopes=0),
        )

        self.assertFalse(result.accepted)
        self.assertIn("WRITER_ENVELOPE_WORK_EXCEEDED", result.reason)
        self.assertIn("nested_envelope_count", result.reason)


def _prepared_snapshot():
    prepared = _prepare(cco_facts())
    return prepared, _initial_snapshot(prepared)


if __name__ == "__main__":
    unittest.main()
