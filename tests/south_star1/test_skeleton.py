"""Focused tests for South Star traversal-skeleton branch policy."""

from __future__ import annotations

from pathlib import Path
import unittest

from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.skeleton import ChildRole
from grimace._south_star1.skeleton import RingEvent
from grimace._south_star1.skeleton import _local_child_orders
from grimace._south_star1.skeleton import _local_event_orders


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "python" / "grimace" / "_south_star1" / "SPEC.md"


class SkeletonBranchPolicyTest(unittest.TestCase):
    def test_exhaustive_skeleton_still_enumerates_broad_child_grammar(self) -> None:
        orders = _local_child_orders(
            AtomId(0),
            [(BondId(0), AtomId(1))],
        )

        self.assertEqual(
            tuple(tuple(event.role for event in order) for order in orders),
            ((ChildRole.BRANCH,), (ChildRole.CONTINUATION,)),
        )

    def test_exhaustive_skeleton_keeps_all_branch_order_with_multiple_children(
        self,
    ) -> None:
        orders = _local_child_orders(
            AtomId(0),
            [(BondId(0), AtomId(1)), (BondId(1), AtomId(2))],
        )

        self.assertTrue(
            any(all(event.role is ChildRole.BRANCH for event in order) for order in orders)
        )

    def test_exhaustive_skeleton_keeps_ring_endpoint_branch_orders(self) -> None:
        orders = _local_event_orders(
            AtomId(0),
            [(BondId(0), AtomId(1))],
            [RingEvent(bond=BondId(1), atom=AtomId(0), other_atom=AtomId(2))],
        )

        self.assertTrue(orders)
        self.assertTrue(any(isinstance(order[0], RingEvent) for order in orders))
        self.assertTrue(
            any(
                hasattr(order[-1], "role")
                and order[-1].role is ChildRole.CONTINUATION
                for order in orders
            )
        )
        self.assertTrue(
            any(
                len(order) == 2
                and hasattr(order[0], "role")
                and order[0].role is ChildRole.BRANCH
                for order in orders
            )
        )

    def test_spec_no_longer_documents_writer_shaped_branch_pruning(self) -> None:
        text = SPEC_PATH.read_text(encoding="utf-8")

        self.assertIn(SerializationLanguageMode.WRITER_SHAPED.value, text)
        self.assertNotIn("all-branch local orders", text)


if __name__ == "__main__":
    unittest.main()
