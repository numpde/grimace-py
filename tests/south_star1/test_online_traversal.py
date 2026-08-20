"""Tests for the lazy online traversal event stream."""

from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path

import grimace._south_star1.exhaustive_online_traversal as online_traversal_module
from grimace._south_star1.facts import ComponentFacts
from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.graph_index import build_graph_index
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import ComponentId
from grimace._south_star1.exhaustive_online_traversal import ExhaustiveTraversalDotEvent
from grimace._south_star1.exhaustive_online_traversal import ExhaustiveTraversalRingEndpointEvent
from grimace._south_star1.exhaustive_online_traversal import _local_event_orders_lazy
from grimace._south_star1.exhaustive_online_traversal import _ChildLocalEvent
from grimace._south_star1.exhaustive_online_traversal import iter_exhaustive_online_traversal_traces
from grimace._south_star1.exhaustive_online_traversal import exhaustive_trace_to_skeleton_like_key
from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.proof_terms import skeleton_key
from grimace._south_star1.skeleton import ChildRole
from grimace._south_star1.skeleton import enumerate_traversal_skeletons
from tests.south_star1.helpers import atom
from tests.south_star1.helpers import cco_facts
from tests.south_star1.helpers import cyclopropane_facts
from tests.south_star1.helpers import single_bond


REPO_ROOT = Path(__file__).resolve().parents[2]
ONLINE_TRAVERSAL_PATH = (
    REPO_ROOT / "python" / "grimace" / "_south_star1" / "exhaustive_online_traversal.py"
)


class OnlineTraversalTest(unittest.TestCase):
    def test_online_traversal_single_atom_matches_skeleton_space(self) -> None:
        self.assertEqual(
            _online_keys(single_atom_facts()),
            _offline_keys(single_atom_facts()),
        )

    def test_online_traversal_path_matches_skeleton_space(self) -> None:
        self.assertEqual(_online_keys(cco_facts()), _offline_keys(cco_facts()))

    def test_online_traversal_triangle_ring_matches_skeleton_space(self) -> None:
        self.assertEqual(
            _online_keys(cyclopropane_facts()),
            _offline_keys(cyclopropane_facts()),
        )

    def test_online_traversal_disconnected_components_emit_dot_events(self) -> None:
        trace = next(
            iter_exhaustive_online_traversal_traces(
                facts=disconnected_facts(),
                policy=ordinary_policy_for_facts(disconnected_facts()),
            )
        )

        self.assertTrue(any(isinstance(event, ExhaustiveTraversalDotEvent) for event in trace.events))

    def test_online_traversal_branch_and_continuation_orders_match_skeleton_space(
        self,
    ) -> None:
        self.assertEqual(
            _online_keys(branched_facts()),
            _offline_keys(branched_facts()),
        )

    def test_online_traversal_single_child_keeps_exhaustive_branch_order(
        self,
    ) -> None:
        orders = tuple(
            _local_event_orders_lazy(
                AtomId(0),
                [(BondId(0), AtomId(1))],
                [],
            )
        )

        self.assertEqual(len(orders), 2)
        self.assertIsInstance(orders[0][0], _ChildLocalEvent)
        self.assertEqual(
            {order[0].role for order in orders if isinstance(order[0], _ChildLocalEvent)},
            {ChildRole.BRANCH, ChildRole.CONTINUATION},
        )

    def test_online_traversal_ring_endpoint_events_have_two_endpoints(self) -> None:
        traces = tuple(
            iter_exhaustive_online_traversal_traces(
                facts=cyclopropane_facts(),
                policy=ordinary_policy_for_facts(cyclopropane_facts()),
            )
        )

        for trace in traces:
            counts: dict[int, int] = {}
            for event in trace.events:
                if isinstance(event, ExhaustiveTraversalRingEndpointEvent):
                    counts[int(event.bond)] = counts.get(int(event.bond), 0) + 1
            self.assertTrue(counts)
            self.assertTrue(all(count == 2 for count in counts.values()))

    def test_online_traversal_is_lazy_boundary(self) -> None:
        tree = ast.parse(ONLINE_TRAVERSAL_PATH.read_text(encoding="utf-8"))
        banned_imports = {
            "audit_rdkit",
            "rdkit_adapter",
            "stereo_witness",
            "support_artifact",
            "support_artifact_checker",
            "support_enumeration",
        }
        banned_calls = {
            "compile_support_artifact",
            "enumerate_stereo_support",
            "enumerate_traversal_skeletons",
        }
        found_imports: list[str] = []
        found_calls: list[str] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found_imports.extend(
                    alias.name
                    for alias in node.names
                    if alias.name.split(".", 1)[0] in banned_imports
                )
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.split(".", 1)[0] in banned_imports:
                    found_imports.append(module)
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    found_calls.append(node.func.id)
                if isinstance(node.func, ast.Attribute):
                    found_calls.append(node.func.attr)

        self.assertEqual(found_imports, [])
        self.assertEqual(sorted(set(found_calls) & banned_calls), [])

    def test_generic_online_traversal_names_are_not_exported(self) -> None:
        generic_trace_name = "iter_" + "online_traversal_traces"
        generic_prepared_name = "iter_prepared_" + "online_traversal_traces"
        generic_key_name = "online_" + "trace_key"

        self.assertFalse(hasattr(online_traversal_module, generic_trace_name))
        self.assertNotIn(generic_trace_name, online_traversal_module.__all__)
        self.assertNotIn(
            generic_prepared_name,
            online_traversal_module.__all__,
        )
        self.assertFalse(hasattr(online_traversal_module, generic_prepared_name))
        self.assertFalse(hasattr(online_traversal_module, generic_key_name))
        self.assertNotIn(generic_key_name, online_traversal_module.__all__)

    def test_generic_online_traversal_module_is_absent(self) -> None:
        self.assertIsNone(importlib.util.find_spec("grimace._south_star1.online_traversal"))


def _online_keys(
    facts: MoleculeFacts,
    *,
    policy=None,
) -> set[tuple[object, ...]]:
    if policy is None:
        policy = ordinary_policy_for_facts(facts)
    return {
        exhaustive_trace_to_skeleton_like_key(trace)
        for trace in iter_exhaustive_online_traversal_traces(facts=facts, policy=policy)
    }


def _offline_keys(
    facts: MoleculeFacts,
    *,
    policy=None,
) -> set[tuple[object, ...]]:
    if policy is None:
        policy = ordinary_policy_for_facts(facts)
    return {
        skeleton_key(skeleton)
        for skeleton in enumerate_traversal_skeletons(
            facts,
            build_graph_index(facts),
            policy,
        )
    }


def single_atom_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, "C"),),
        bonds=(),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0),),
                bonds=(),
            ),
        ),
    )


def disconnected_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, "C"), atom(1, "O")),
        bonds=(),
        components=(
            ComponentFacts(id=ComponentId(0), atoms=(AtomId(0),), bonds=()),
            ComponentFacts(id=ComponentId(1), atoms=(AtomId(1),), bonds=()),
        ),
    )


def branched_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, "C"), atom(1, "C"), atom(2, "O"), atom(3, "F")),
        bonds=(
            single_bond(0, 0, 1),
            single_bond(1, 0, 2),
            single_bond(2, 0, 3),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1), AtomId(2), AtomId(3)),
                bonds=(BondId(0), BondId(1), BondId(2)),
            ),
        ),
    )


if __name__ == "__main__":
    unittest.main()
