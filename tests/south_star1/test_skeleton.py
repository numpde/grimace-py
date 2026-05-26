"""Focused tests for South Star traversal-skeleton branch policy."""

from __future__ import annotations

from pathlib import Path
import unittest

from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.policy import BranchPresentationMode
from grimace._south_star1.skeleton import ChildRole
from grimace._south_star1.skeleton import RingEvent
from grimace._south_star1.skeleton import _local_child_orders
from grimace._south_star1.skeleton import _local_event_orders


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "python" / "grimace" / "_south_star1" / "SPEC.md"


class SkeletonBranchPolicyTest(unittest.TestCase):
    def test_exhaustive_branch_policy_preserves_all_branch_order(self) -> None:
        orders = _local_child_orders(
            AtomId(0),
            [(BondId(0), AtomId(1))],
            branch_presentation_mode=BranchPresentationMode.EXHAUSTIVE,
        )

        self.assertEqual(
            tuple(tuple(event.role for event in order) for order in orders),
            ((ChildRole.BRANCH,), (ChildRole.CONTINUATION,)),
        )

    def test_writer_shaped_policy_rejects_single_child_all_branch_order(self) -> None:
        orders = _local_child_orders(
            AtomId(0),
            [(BondId(0), AtomId(1))],
            branch_presentation_mode=BranchPresentationMode.WRITER_SHAPED,
        )

        self.assertEqual(len(orders), 1)
        self.assertIs(orders[0][0].role, ChildRole.CONTINUATION)

    def test_writer_shaped_policy_keeps_required_side_branches(self) -> None:
        orders = _local_child_orders(
            AtomId(0),
            [(BondId(0), AtomId(1)), (BondId(1), AtomId(2))],
            branch_presentation_mode=BranchPresentationMode.WRITER_SHAPED,
        )

        self.assertTrue(orders)
        for order in orders:
            self.assertEqual(len(order), 2)
            self.assertIs(order[0].role, ChildRole.BRANCH)
            self.assertIs(order[-1].role, ChildRole.CONTINUATION)

    def test_writer_shaped_policy_keeps_ring_endpoint_decorations(self) -> None:
        orders = _local_event_orders(
            AtomId(0),
            [(BondId(0), AtomId(1))],
            [RingEvent(bond=BondId(1), atom=AtomId(0), other_atom=AtomId(2))],
            branch_presentation_mode=BranchPresentationMode.WRITER_SHAPED,
        )

        self.assertTrue(orders)
        self.assertTrue(any(isinstance(order[0], RingEvent) for order in orders))
        self.assertTrue(
            all(order[-1].role is ChildRole.CONTINUATION for order in orders)
        )
        self.assertFalse(
            any(
                len(order) == 2
                and hasattr(order[0], "role")
                and order[0].role is ChildRole.BRANCH
                for order in orders
            )
        )

    def test_spec_documents_branch_presentation_policy(self) -> None:
        text = SPEC_PATH.read_text(encoding="utf-8")

        self.assertIn("Branch Presentation Policy", text)
        self.assertIn("EXHAUSTIVE", text)
        self.assertIn("WRITER_SHAPED", text)
        self.assertIn("all-branch local orders", text)
        self.assertIn("not a claim of RDKit writer parity", text)


if __name__ == "__main__":
    unittest.main()
