"""Tests for the online South Star serialization stream."""

from __future__ import annotations

import ast
import inspect
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import grimace._south_star1.online_serialization_stream as stream_module
from grimace._south_star1.facts import ComponentFacts
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import ComponentId
from grimace._south_star1.online_continuation import OnlineDecoderExecutionMode
from grimace._south_star1.online_serialization_stream import collect_online_serializations
from grimace._south_star1.online_serialization_stream import count_online_serializations
from grimace._south_star1.online_serialization_stream import iter_online_serializations
from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.ordinary_semantics import OrdinarySmilesSemantics
from grimace._south_star1.policy import AnnotationMode
from grimace._south_star1.prepared_bench_matrix import PreparedRuntimeProbe
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import enumerate_prepared_stereo_support
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_smiles
from grimace._south_star1.support_enumeration import enumerate_stereo_support
from tests.south_star1.helpers import atom
from tests.south_star1.helpers import cyclopropane_facts
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import single_bond
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

    def test_iter_online_serializations_is_determinized_support_stream(self) -> None:
        facts = cyclopropane_facts()
        emitted = tuple(
            iter_online_serializations(
                facts=facts,
                policy=ordinary_policy_for_facts(facts),
                semantics=OrdinarySmilesSemantics(),
            )
        )
        strings = tuple(item.text for item in emitted)

        self.assertEqual(set(strings), set(_offline_support(facts).strings))
        self.assertEqual(len(strings), len(set(strings)))

    def test_iter_online_serializations_default_does_not_keep_emitted_set(self) -> None:
        function = _function_def(iter_online_serializations)
        assigned_names = _assigned_names(function)
        seen_assignment = _single_assignment(function, "seen")

        self.assertNotIn("emitted", assigned_names)
        self.assertNotIn("strings", assigned_names)
        self.assertIsInstance(seen_assignment.value, ast.IfExp)
        assert isinstance(seen_assignment.value, ast.IfExp)
        self.assertEqual(ast.unparse(seen_assignment.value.test), "verify_unique")
        self.assertEqual(ast.unparse(seen_assignment.value.body), "set()")
        self.assertEqual(ast.unparse(seen_assignment.value.orelse), "None")

    def test_iter_online_serializations_verify_unique_keeps_guard_set(self) -> None:
        with patch.object(
            stream_module,
            "make_determinized_online_decoder",
            return_value=_duplicate_support_decoder(),
        ):
            with self.assertRaisesRegex(ValueError, "duplicate"):
                tuple(iter_online_serializations(prepared=object(), verify_unique=True))

    def test_iter_online_serializations_has_no_branch_mode_parameter(self) -> None:
        signature = inspect.signature(iter_online_serializations)

        self.assertNotIn("branch_mode", signature.parameters)
        self.assertIn("verify_unique", signature.parameters)

    def test_iter_online_serializations_does_not_call_branch_preserving_decoder(self) -> None:
        facts = tetrahedral_facts()

        with patch.object(
            stream_module,
            "make_branch_preserving_online_decoder",
            side_effect=AssertionError("support stream called branch-preserving decoder"),
            create=True,
        ):
            emitted = tuple(
                iter_online_serializations(
                    facts=facts,
                    policy=ordinary_policy_for_facts(facts),
                    semantics=OrdinarySmilesSemantics(),
                )
            )

        self.assertEqual({item.text for item in emitted}, set(_offline_support(facts).strings))

    def test_collect_online_serializations_materializes_and_verifies_uniqueness(self) -> None:
        function = _function_def(collect_online_serializations)
        assigned_names = _assigned_names(function)

        self.assertIn("strings", assigned_names)
        self.assertIn("emitted", assigned_names)
        with patch.object(
            stream_module,
            "make_determinized_online_decoder",
            return_value=_duplicate_support_decoder(),
        ):
            with self.assertRaisesRegex(ValueError, "duplicate"):
                collect_online_serializations(prepared=object())

    def test_count_online_serializations_matches_offline_support_count(self) -> None:
        for case in _count_cases():
            with self.subTest(case=case.name):
                prepared = _prepare(case.facts)
                support = enumerate_prepared_stereo_support(
                    prepared=prepared,
                    runtime_options=case.runtime_options,
                )
                count = count_online_serializations(
                    prepared=prepared,
                    runtime_options=case.runtime_options,
                )

                self.assertEqual(count.support_count, support.distinct_count)
                self.assertEqual(count.stats.emitted_support_count, support.distinct_count)

    def test_count_online_serializations_matches_offline_witness_count(self) -> None:
        for case in _count_cases():
            with self.subTest(case=case.name):
                prepared = _prepare(case.facts)
                support = enumerate_prepared_stereo_support(
                    prepared=prepared,
                    runtime_options=case.runtime_options,
                )
                count = count_online_serializations(
                    prepared=prepared,
                    runtime_options=case.runtime_options,
                )

                self.assertEqual(count.witness_completion_count, support.witness_count)
                self.assertEqual(count.stats.witness_completion_count, support.witness_count)

    def test_count_online_serializations_does_not_materialize_strings(self) -> None:
        function = _function_def(count_online_serializations)
        assigned_names = _assigned_names(function)
        called_names = _called_names(function)

        self.assertNotIn("strings", assigned_names)
        self.assertNotIn("emitted", assigned_names)
        self.assertNotIn("seen", assigned_names)
        self.assertNotIn("iter_online_serializations", called_names)
        self.assertNotIn("collect_online_serializations", called_names)

    def test_count_online_serializations_does_not_call_collect(self) -> None:
        facts = tetrahedral_facts()

        with patch.object(
            stream_module,
            "collect_online_serializations",
            side_effect=AssertionError("count path called materializing collector"),
        ):
            count = count_online_serializations(
                facts=facts,
                policy=ordinary_policy_for_facts(facts),
                semantics=OrdinarySmilesSemantics(),
            )

        self.assertEqual(count.support_count, _offline_support(facts).distinct_count)

    def test_count_online_serializations_uses_prepared_path_without_cache_rebuilds(
        self,
    ) -> None:
        prepared = _prepare(tetrahedral_facts())

        with PreparedRuntimeProbe() as probe:
            count = count_online_serializations(prepared=prepared)

        result = probe.result()
        self.assertGreater(count.support_count, 0)
        self.assertEqual(result.graph_index_rebuild_count, 0)
        self.assertEqual(result.online_traversal_graph_from_facts_count, 0)
        self.assertEqual(result.online_traversal_graph_from_index_count, 0)
        self.assertEqual(result.root_domain_recompute_count, 0)
        self.assertEqual(result.root_domain_from_metadata_count, 0)
        self.assertEqual(result.stereo_template_rebuild_count, 0)
        self.assertEqual(result.facts_validate_count, 0)
        self.assertEqual(result.policy_validate_count, 0)
        self.assertEqual(result.online_traversal_graph_view_rebuild_count, 0)
        self.assertEqual(result.online_vm_graph_view_rebuild_count, 0)

    def test_verify_unique_detects_forced_duplicate_support_string(self) -> None:
        with patch.object(
            stream_module,
            "make_determinized_online_decoder",
            return_value=_duplicate_support_decoder(),
        ):
            unchecked = tuple(iter_online_serializations(prepared=object()))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                tuple(iter_online_serializations(prepared=object(), verify_unique=True))

        self.assertEqual(tuple(item.text for item in unchecked), ("C", "C"))

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


@dataclass(frozen=True, slots=True)
class _CountCase:
    name: str
    facts: object
    runtime_options: SouthStarRuntimeOptions = SouthStarRuntimeOptions()


def _count_cases() -> tuple[_CountCase, ...]:
    return (
        _CountCase("tetrahedral", tetrahedral_facts()),
        _CountCase("directional", directional_facts()),
        _CountCase("ring", cyclopropane_facts()),
        _CountCase("support-maximal", directional_facts()),
        _CountCase("duplicate-render", cyclopropane_facts()),
        _CountCase("rooted", tetrahedral_facts(), SouthStarRuntimeOptions(rooted_at_atom=0)),
        _CountCase(
            "disconnected-stereo",
            _disconnected_tetra_and_bond_facts(),
            SouthStarRuntimeOptions(rooted_at_atom=5),
        ),
        _CountCase("sparse-atom-id", _sparse_two_atom_facts()),
    )


def _prepare(facts):
    return prepare_south_star_mol_from_facts(
        facts,
        writer_surface=SouthStarWriterSurface(),
    )


def _function_def(function: object) -> ast.FunctionDef:
    source = inspect.getsource(function)
    tree = ast.parse(source)
    node = tree.body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


def _assigned_names(function: ast.FunctionDef) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(_target_names(target))
        elif isinstance(node, ast.AnnAssign):
            names.update(_target_names(node.target))
    return names


def _target_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        out: set[str] = set()
        for item in target.elts:
            out.update(_target_names(item))
        return out
    return set()


def _single_assignment(function: ast.FunctionDef, name: str) -> ast.Assign | ast.AnnAssign:
    assignments = [
        node
        for node in ast.walk(function)
        if (
            isinstance(node, ast.Assign)
            and any(name in _target_names(target) for target in node.targets)
        )
        or (isinstance(node, ast.AnnAssign) and name in _target_names(node.target))
    ]
    assert len(assignments) == 1
    return assignments[0]


def _called_names(function: ast.FunctionDef) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                out.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                out.add(node.func.attr)
    return out


@dataclass(frozen=True, slots=True)
class _FakeChoice:
    is_eos: bool
    next_state: object | None = None
    completion_count: int = 1
    multiplicity: int = 1


@dataclass(frozen=True, slots=True)
class _FakeChoiceResult:
    choices: tuple[_FakeChoice, ...]
    stats: object


@dataclass(frozen=True, slots=True)
class _FakeState:
    prefix: str

    def choices_with_stats(self) -> _FakeChoiceResult:
        return _FakeChoiceResult(
            choices=(
                _FakeChoice(is_eos=True),
                _FakeChoice(is_eos=True),
            ),
            stats=object(),
        )


@dataclass(frozen=True, slots=True)
class _FakeDecoder:
    def initial_state(self) -> _FakeState:
        return _FakeState(prefix="C")


def _duplicate_support_decoder() -> _FakeDecoder:
    return _FakeDecoder()


def _disconnected_tetra_and_bond_facts():
    tetra = tetrahedral_facts()
    return type(tetra)(
        atoms=(
            atom(0, "C"),
            atom(1, "F"),
            atom(2, "Cl"),
            atom(3, "Br"),
            atom(5, "C"),
            atom(6, "O"),
        ),
        bonds=(
            single_bond(0, 0, 1),
            single_bond(1, 0, 2),
            single_bond(2, 0, 3),
            single_bond(4, 5, 6),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1), AtomId(2), AtomId(3)),
                bonds=(BondId(0), BondId(1), BondId(2)),
            ),
            ComponentFacts(
                id=ComponentId(1),
                atoms=(AtomId(5), AtomId(6)),
                bonds=(BondId(4),),
            ),
        ),
        stereo=tetra.stereo,
        ligand_occurrences=tetra.ligand_occurrences,
    )


def _sparse_two_atom_facts():
    tetra = tetrahedral_facts()
    return type(tetra)(
        atoms=(atom(10, "C"), atom(20, "O")),
        bonds=(single_bond(30, 10, 20),),
        components=(
            ComponentFacts(
                id=ComponentId(7),
                atoms=(AtomId(10), AtomId(20)),
                bonds=(BondId(30),),
            ),
        ),
    )


if __name__ == "__main__":
    unittest.main()
