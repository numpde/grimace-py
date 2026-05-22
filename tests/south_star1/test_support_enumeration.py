"""Tests for South Star 1 molecule-level support enumeration."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

from grimace._south_star1.graph_index import build_graph_index
from grimace._south_star1.ids import AtomId
from grimace._south_star1.skeleton import enumerate_traversal_skeletons
from grimace._south_star1.slots import allocate_traversal_slots
from grimace._south_star1.support_enumeration import enumerate_stereo_support
from grimace._south_star1.support_enumeration import enumerate_stereo_support_with_stats
from grimace._south_star1.support_enumeration import enumerate_stereo_witnesses

from tests.south_star1.helpers import tetrahedral_facts
from tests.south_star1.test_stereo_witness import _TetraOrderSemantics
from tests.south_star1.test_stereo_witness import _policy_for_facts_only
from tests.south_star1.test_stereo_witness import _policy_for_slots


class SupportEnumerationTest(unittest.TestCase):
    def test_stereo_support_enumerates_across_supplied_skeletons(self) -> None:
        facts = tetrahedral_facts()
        base_policy = _policy_for_facts_only(facts)
        skeletons = tuple(
            skeleton
            for skeleton in enumerate_traversal_skeletons(
                facts,
                build_graph_index(facts),
                base_policy,
            )
            if skeleton.roots == (AtomId(0),)
        )[:2]
        policy = _policy_for_slots(
            facts,
            allocate_traversal_slots(facts, skeletons[0]),
            chiral_center=AtomId(0),
        )

        result = enumerate_stereo_support_with_stats(
            facts=facts,
            policy=policy,
            semantics=_TetraOrderSemantics(),
            skeletons=skeletons,
        )

        self.assertEqual(result.stats.skeleton_count, 2)
        self.assertEqual(result.stats.witness_count, 2)
        self.assertEqual(result.image.witness_count, 2)
        self.assertEqual(len(result.image.strings), 2)
        self.assertTrue(
            all("[C@H]" in rendered for rendered in result.image.strings)
        )

    def test_stereo_support_preserves_rendered_witness_multiplicity(self) -> None:
        facts = tetrahedral_facts()
        base_policy = _policy_for_facts_only(facts)
        skeleton = next(
            skeleton
            for skeleton in enumerate_traversal_skeletons(
                facts,
                build_graph_index(facts),
                base_policy,
            )
            if skeleton.roots == (AtomId(0),)
        )
        skeletons = (skeleton, skeleton)
        policy = _policy_for_slots(
            facts,
            allocate_traversal_slots(facts, skeleton),
            chiral_center=AtomId(0),
        )

        image = enumerate_stereo_support(
            facts=facts,
            policy=policy,
            semantics=_TetraOrderSemantics(),
            skeletons=skeletons,
        )

        self.assertEqual(image.witness_count, 2)
        self.assertEqual(image.distinct_count, 1)
        self.assertEqual(image.strings[0], image.strings[1])

    def test_stereo_witnesses_are_streaming(self) -> None:
        facts = tetrahedral_facts()
        base_policy = _policy_for_facts_only(facts)
        skeleton = next(
            skeleton
            for skeleton in enumerate_traversal_skeletons(
                facts,
                build_graph_index(facts),
                base_policy,
            )
            if skeleton.roots == (AtomId(0),)
        )
        policy = _policy_for_slots(
            facts,
            allocate_traversal_slots(facts, skeleton),
            chiral_center=AtomId(0),
        )

        witness_iter = enumerate_stereo_witnesses(
            facts=facts,
            policy=policy,
            semantics=_TetraOrderSemantics(),
            skeletons=(skeleton,),
        )

        self.assertEqual(next(witness_iter).rendered, "[C@H](F)(Cl)(Br)")
        with self.assertRaises(StopIteration):
            next(witness_iter)

    def test_support_enumeration_module_has_no_rdkit_import(self) -> None:
        path = Path("python/grimace/_south_star1/support_enumeration.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))

        self.assertFalse(_imports_rdkit(tree))


def _imports_rdkit(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.startswith("rdkit") for alias in node.names):
                return True
        if isinstance(node, ast.ImportFrom):
            if node.module is not None and node.module.startswith("rdkit"):
                return True
    return False


if __name__ == "__main__":
    unittest.main()
