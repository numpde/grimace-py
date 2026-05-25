"""Tests for the online South Star serialization stream."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from grimace._south_star1.online_continuation import OnlineDecoderExecutionMode
from grimace._south_star1.online_serialization_stream import collect_online_serializations
from grimace._south_star1.online_serialization_stream import iter_online_serializations
from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.ordinary_semantics import OrdinarySmilesSemantics
from grimace._south_star1.policy import AnnotationMode
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_smiles
from grimace._south_star1.support_enumeration import enumerate_stereo_support
from tests.south_star1.helpers import cyclopropane_facts
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import tetrahedral_facts


REPO_ROOT = Path(__file__).resolve().parents[2]
ONLINE_SERIALIZATION_STREAM_PATH = (
    REPO_ROOT / "python" / "grimace" / "_south_star1" / "online_serialization_stream.py"
)


class OnlineSerializationStreamTest(unittest.TestCase):
    def test_online_serialization_stream_matches_offline_support_tetra(self) -> None:
        self._assert_online_matches_offline(tetrahedral_facts())

    def test_online_serialization_stream_matches_offline_support_directional(self) -> None:
        self._assert_online_matches_offline(directional_facts())

    def test_online_serialization_stream_matches_offline_support_ring(self) -> None:
        self._assert_online_matches_offline(cyclopropane_facts())

    def test_online_serialization_stream_matches_offline_support_maximal(self) -> None:
        facts = directional_facts()
        policy = ordinary_policy_for_facts(facts)

        self.assertIs(policy.annotation_mode, AnnotationMode.SUPPORT_MAXIMAL)
        self._assert_online_matches_offline(facts)

    def test_online_serialization_stream_has_no_duplicate_strings(self) -> None:
        result = _online_result(cyclopropane_facts())

        self.assertEqual(len(result.strings), len(set(result.strings)))
        self.assertEqual(result.support_count, len(result.strings))

    def test_online_serialization_stream_support_count_matches_offline(self) -> None:
        facts = ordinary_molecule_facts_from_smiles("[C@H]1(F)CO1")
        result = _online_result(facts)
        support = _offline_support(facts)

        self.assertEqual(result.support_count, support.distinct_count)
        self.assertEqual(result.stats.emitted_support_count, support.distinct_count)

    def test_online_serialization_stream_witness_completion_count_matches_offline_witness_count(
        self,
    ) -> None:
        facts = cyclopropane_facts()
        result = _online_result(facts)
        support = _offline_support(facts)

        self.assertEqual(result.witness_completion_count, support.witness_count)
        self.assertEqual(result.stats.witness_completion_count, support.witness_count)
        self.assertGreater(result.witness_completion_count, result.support_count)

    def test_online_serialization_stream_equivalent_across_execution_modes(self) -> None:
        facts = tetrahedral_facts()
        offline = set(_offline_support(facts).strings)
        modes = (
            OnlineDecoderExecutionMode.PREFIX_REPLAY,
            OnlineDecoderExecutionMode.CACHED_COMPLETIONS,
            OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
        )

        for mode in modes:
            with self.subTest(mode=mode.value):
                result = _online_result(facts, execution_mode=mode)
                self.assertEqual(set(result.strings), offline)
                self.assertEqual(result.support_count, len(offline))

    def test_iter_online_serializations_defaults_to_residual_mode(self) -> None:
        facts = tetrahedral_facts()
        emitted = tuple(
            iter_online_serializations(
                facts=facts,
                policy=ordinary_policy_for_facts(facts),
                semantics=OrdinarySmilesSemantics(),
            )
        )

        self.assertEqual({item.text for item in emitted}, set(_offline_support(facts).strings))
        self.assertTrue(all(item.completion_count >= item.multiplicity for item in emitted))

    def test_online_serialization_stats_reports_residual_retained_continuations(self) -> None:
        result = _online_result(tetrahedral_facts())

        self.assertGreater(result.stats.frontier_queries, 0)
        self.assertGreater(result.stats.max_choice_count, 0)
        self.assertIsNotNone(result.stats.max_retained_continuation_count)
        assert result.stats.max_retained_continuation_count is not None
        self.assertGreater(result.stats.max_retained_continuation_count, 0)

    def test_online_serialization_stream_boundary_no_offline_support_imports(self) -> None:
        tree = ast.parse(ONLINE_SERIALIZATION_STREAM_PATH.read_text(encoding="utf-8"))
        banned_modules = {
            "support_enumeration",
            "support_artifact",
            "support_artifact_checker",
            "finite_space_checker",
            "semantic_relation_checker",
            "rdkit_adapter",
            "audit_rdkit",
        }
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(
                    alias.name
                    for alias in node.names
                    if alias.name.split(".", 1)[0] in banned_modules
                )
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.split(".", 1)[0] in banned_modules:
                    imports.append(module)

        self.assertEqual(imports, [])

    def _assert_online_matches_offline(self, facts) -> None:
        result = _online_result(facts)
        support = _offline_support(facts)

        self.assertEqual(set(result.strings), set(support.strings))
        self.assertEqual(result.support_count, support.distinct_count)
        self.assertEqual(result.witness_completion_count, support.witness_count)


def _online_result(
    facts,
    *,
    execution_mode: OnlineDecoderExecutionMode = OnlineDecoderExecutionMode.RESIDUAL_CONTINUATIONS,
):
    return collect_online_serializations(
        facts=facts,
        policy=ordinary_policy_for_facts(facts),
        semantics=OrdinarySmilesSemantics(),
        execution_mode=execution_mode,
    )


def _offline_support(facts):
    return enumerate_stereo_support(
        facts=facts,
        policy=ordinary_policy_for_facts(facts),
        semantics=OrdinarySmilesSemantics(),
    )


if __name__ == "__main__":
    unittest.main()
