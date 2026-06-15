"""Public boundary checks for cyclic writer-shaped support."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import grimace._south_star1.writer_snapshot as writer_snapshot
from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.errors import SouthStarErrorKind
from grimace._south_star1.writer_frontier import (
    initial_writer_frontier_cursor,
    initial_writer_transition_frontier_cursor,
)
from grimace._south_star1.writer_support import (
    enumerate_prepared_writer_shaped_support,
)
from tests.south_star1.helpers import cyclopropane_facts
from tests.south_star1.test_writer_state_kernel import _prepare
from tests.south_star1.test_writer_state_kernel import _writer_options


class WriterCyclicPublicBoundaryTest(unittest.TestCase):
    def test_public_writer_support_rejects_cyclic_prepared_before_count_or_stream(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        options = _writer_options()

        with patch(
            "grimace._south_star1.writer_support.count_writer_frontier_support",
            side_effect=AssertionError("count must not run"),
        ), patch(
            (
                "grimace._south_star1.writer_support"
                ".count_writer_cursor_completions"
            ),
            side_effect=AssertionError("completion count must not run"),
        ), patch(
            "grimace._south_star1.writer_support.iter_writer_frontier_support",
            side_effect=AssertionError("stream must not run"),
        ):
            with self.assertRaises(SouthStarError) as cm:
                enumerate_prepared_writer_shaped_support(
                    prepared=prepared,
                    runtime_options=options,
                )

        text = str(cm.exception).lower()
        self.assertIs(cm.exception.kind, SouthStarErrorKind.UNSUPPORTED_POLICY)
        self.assertTrue(
            "cyclic" in text
            or "cycle" in text
            or "tree" in text
        )

    def test_public_initial_writer_frontier_rejects_cyclic_prepared(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        options = _writer_options()

        with self.assertRaises(SouthStarError) as cm:
            initial_writer_frontier_cursor(prepared, options)

        self.assertIs(cm.exception.kind, SouthStarErrorKind.UNSUPPORTED_POLICY)

    def test_private_cyclic_transition_cursor_reports_ready_but_public_closed(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        options = _writer_options(rooted_at_atom=0)
        cursor = initial_writer_transition_frontier_cursor(prepared, options)

        decision = writer_snapshot._cyclic_writer_admission_decision_from_cursor(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )

        self.assertIs(
            decision.kind,
            (
                writer_snapshot
                ._WriterCyclicAdmissionDecisionKind
                .READY_BUT_PUBLIC_CLOSED
            ),
        )
        self.assertTrue(decision.internally_ready)
        self.assertFalse(decision.public_enabled)
        self.assertFalse(decision.admitted_publicly)

        with self.assertRaises(SouthStarError) as cm:
            writer_snapshot._assert_cyclic_writer_admission_decision(
                decision,
            )

        self.assertIs(cm.exception.kind, SouthStarErrorKind.UNSUPPORTED_POLICY)
        self.assertIn("public support is closed", str(cm.exception))

    def test_private_cyclic_snapshot_admission_matches_cursor_admission(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        options = _writer_options(rooted_at_atom=0)
        cursor = initial_writer_transition_frontier_cursor(prepared, options)

        cursor_decision = (
            writer_snapshot
            ._cyclic_writer_admission_decision_from_cursor(
                prepared=prepared,
                runtime_options=options,
                cursor=cursor,
            )
        )
        snapshot = cursor_decision.readiness_gate.snapshot
        snapshot_decision = (
            writer_snapshot
            ._cyclic_writer_admission_decision_from_snapshot(
                snapshot,
                prepared=prepared,
            )
        )

        self.assertIs(snapshot_decision.kind, cursor_decision.kind)
        self.assertEqual(snapshot.cursor, cursor)
        self.assertEqual(snapshot.runtime_options, options)


if __name__ == "__main__":
    unittest.main()
