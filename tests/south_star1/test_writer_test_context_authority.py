"""AST checks for the deliberately narrow direct writer-test boundary."""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import unittest


_CONTEXT_MODULES = {
    "grimace._south_star1.prepared_runtime": {"options", "prepare"},
    "grimace._south_star1.writer_snapshot": {"snapshot"},
}
_OPERATIONS = {"options", "prepare", "snapshot"}


@dataclass(frozen=True, slots=True)
class _Observed:
    module: str
    function: str
    operation: str
    line: int


def _parents(tree: ast.AST) -> dict[int, ast.AST]:
    result: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            result[id(child)] = parent
    return result


def _qualified_function(tree: ast.AST, node: ast.AST, parents=None) -> str:
    parents = _parents(tree) if parents is None else parents
    names: list[str] = []
    current: ast.AST | None = node
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(current.name)
        current = parents.get(id(current))
    return ".".join(reversed(names)) or "<module>"


def _bindings(tree: ast.AST) -> dict[str, tuple[str, str]]:
    bindings: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in _CONTEXT_MODULES:
            for item in node.names:
                local = item.asname or item.name
                operation = {
                    "SouthStarRuntimeOptions": "options",
                    "prepare_south_star_mol_from_facts": "prepare",
                    "capture_writer_frontier_snapshot": "snapshot",
                }.get(item.name)
                if operation:
                    bindings[local] = (node.module, operation)
        elif isinstance(node, ast.Import):
            for item in node.names:
                if item.name in _CONTEXT_MODULES:
                    bindings[item.asname or item.name.rsplit(".", 1)[-1]] = (
                        item.name,
                        "module",
                    )
    return bindings


def _called_operation(call: ast.Call, bindings: dict[str, tuple[str, str]]) -> str | None:
    if isinstance(call.func, ast.Name):
        binding = bindings.get(call.func.id)
        if binding is None:
            return None
        return binding[1] if binding[1] != "module" else None
    if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
        binding = bindings.get(call.func.value.id)
        if binding is None or binding[1] != "module":
            return None
        return {
            ("grimace._south_star1.prepared_runtime", "SouthStarRuntimeOptions"): "options",
            ("grimace._south_star1.prepared_runtime", "prepare_south_star_mol_from_facts"): "prepare",
            ("grimace._south_star1.writer_snapshot", "capture_writer_frontier_snapshot"): "snapshot",
        }.get((binding[0], call.func.attr))
    return None


def _is_writer_surface(node: ast.AST | None) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "SouthStarWriterSurface"
        and not node.args
        and not node.keywords
    )


def _is_initial_cursor(node: ast.AST | None) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
        and (node.func.id if isinstance(node.func, ast.Name) else node.func.attr)
        == "initial_writer_frontier_cursor"
    )


def _is_generic(call: ast.Call, operation: str) -> bool:
    if operation == "options":
        keywords = {item.arg for item in call.keywords if item.arg}
        serial = next((item.value for item in call.keywords if item.arg == "serialization_language"), None)
        canonical = next((item.value for item in call.keywords if item.arg == "canonical"), None)
        random = next((item.value for item in call.keywords if item.arg == "do_random"), None)
        return (
            isinstance(serial, ast.Attribute)
            and serial.attr == "WRITER_SHAPED"
            and keywords <= {"rooted_at_atom", "canonical", "do_random", "serialization_language"}
            and (canonical is None or isinstance(canonical, ast.Constant) and canonical.value is False)
            and (random is None or isinstance(random, ast.Constant) and random.value is True)
        )
    if operation == "prepare":
        return _is_writer_surface(next((item.value for item in call.keywords if item.arg == "writer_surface"), None))
    return _is_initial_cursor(next((item.value for item in call.keywords if item.arg == "cursor"), None))


def _allowance_decorators(function: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[tuple[str, str], ...]:
    allowances = []
    for decorator in function.decorator_list:
        if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Name):
            continue
        if decorator.func.id != "allow_direct_writer_context_construction":
            continue
        reason = next((item.value for item in decorator.keywords if item.arg == "reason"), None)
        reason_value = reason.value if isinstance(reason, ast.Constant) and isinstance(reason.value, str) else ""
        for argument in decorator.args:
            value = argument.value if isinstance(argument, ast.Constant) else None
            allowances.append((value, reason_value))
    return tuple(allowances)


def _scan(path: Path) -> tuple[list[_Observed], dict[str, tuple[tuple[str, str], ...]]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    parents = _parents(tree)
    bindings = _bindings(tree)
    declarations: dict[str, tuple[tuple[str, str], ...]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            declared = _allowance_decorators(node)
            if declared:
                declarations[_qualified_function(tree, node, parents)] = declared
    observed = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        operation = _called_operation(node, bindings)
        if operation and _is_generic(node, operation):
            observed.append(_Observed(path.name, _qualified_function(tree, node, parents), operation, node.lineno))
    return observed, declarations


def _all_scanned():
    observed: list[_Observed] = []
    declarations: dict[tuple[str, str], tuple[tuple[str, str], ...]] = {}
    root = Path(__file__).parent
    for path in sorted(root.glob("test_writer*.py")):
        if path.name in {"test_writer_test_context.py", Path(__file__).name}:
            continue
        current_observed, current_declarations = _scan(path)
        observed.extend(current_observed)
        declarations.update({(path.name, key): value for key, value in current_declarations.items()})
    return observed, declarations


def declared_direct_construction_allowances():
    """Return annotations discovered in test functions, not a registry."""
    _observed, declarations = _all_scanned()
    from tests.south_star1.writer_test_context import DirectWriterContextAllowance
    return tuple(
        DirectWriterContextAllowance(tuple(operation for operation, _ in values), values[0][1])
        for values in declarations.values()
    )


class WriterTestContextAuthorityTest(unittest.TestCase):
    def test_annotations_match_all_observed_generic_constructions(self):
        observed, declarations = _all_scanned()
        observed_by_function: dict[tuple[str, str], set[str]] = {}
        for item in observed:
            observed_by_function.setdefault((item.module, item.function), set()).add(item.operation)
        self.assertEqual(set(observed_by_function), set(declarations))
        for key, operations in observed_by_function.items():
            declared = declarations[key]
            self.assertEqual(len(declared), len(set(declared)))
            declared_operations = {operation for operation, _reason in declared}
            self.assertEqual(declared_operations, operations)
            self.assertTrue(all(operation in _OPERATIONS and reason.strip() for operation, reason in declared))

    def test_exact_allowance_inventory(self):
        _observed, declarations = _all_scanned()
        counts = Counter(operation for values in declarations.values() for operation, _ in values)
        self.assertEqual(counts, {"snapshot": 21, "prepare": 1})
        self.assertNotIn("options", counts)
        self.assertEqual(
            set(declarations),
            {
                ("test_writer_state_kernel.py", "WriterStateKernelTest.test_missing_writer_bond_domain_fails_closed"),
                *(('test_writer_snapshot.py', name) for name in (
                    "WriterSnapshotTest.test_snapshot_advance_emits_step_certificate",
                    "WriterSnapshotTest.test_snapshot_advance_outcome_carries_product_projection_identity",
                    "WriterSnapshotTest.test_snapshot_advance_invalid_text_has_no_projection_match",
                    "WriterSnapshotTest.test_snapshot_replay_sequence_invalid_text_carries_certificate",
                    "WriterSnapshotTest.test_snapshot_advance_invalid_text_certificate_rejects_match",
                    "WriterSnapshotTest.test_snapshot_advance_returns_blocked_product_for_unsupported_capability",
                    "WriterSnapshotTest.test_snapshot_advance_blocked_error_does_not_delegate_to_choice_snapshot_blockers",
                    "WriterSnapshotTest.test_snapshot_blocked_advance_certificate_rejects_cursor_mismatch",
                    "WriterSnapshotTest.test_snapshot_advance_successor_cursor_comes_from_text_projection_certificate",
                    "WriterSnapshotTest.test_snapshot_advance_outcome_rejects_stale_step_projection",
                    "WriterSnapshotTest.test_snapshot_replay_sequence_rejects_projection_chain_mismatch",
                    "WriterSnapshotTest.test_snapshot_replay_certificate_tracks_prefix_steps",
                    "WriterSnapshotTest.test_empty_snapshot_replay_has_empty_certificate",
                    "WriterSnapshotTest.test_prefix_read_exposes_replay_certificate",
                    "WriterSnapshotTest.test_prefix_read_certificate_binds_final_frontier_counts",
                    "WriterSnapshotTest.test_prefix_read_certificate_binds_replay_final_snapshot",
                    "WriterSnapshotTest.test_prefix_read_certificate_rejects_replay_final_snapshot_mismatch",
                    "WriterSnapshotTest.test_invalid_snapshot_advance_has_no_step_certificate",
                    "WriterSnapshotTest.test_snapshot_step_certificate_rejects_malformed_inputs",
                    "WriterSnapshotTest.test_snapshot_replay_certificate_rejects_malformed_inputs",
                )) ,
                ("test_writer_stereo_residual.py", "WriterStereoResidualTest.test_joint_directional_non_single_ring_carrier_support_is_certified"),
            },
        )

    def test_alias_and_qualified_bindings_are_detected(self):
        source = """
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions as OptionsAlias
import grimace._south_star1.writer_snapshot as snapshot_alias
def test_case():
    OptionsAlias(serialization_language=SerializationLanguageMode.WRITER_SHAPED)
    snapshot_alias.capture_writer_frontier_snapshot(cursor=initial_writer_frontier_cursor(prepared, options))
"""
        path = Path("/tmp/writer-context-authority-synthetic.py")
        path.write_text(source, encoding="utf-8")
        observed, _ = _scan(path)
        self.assertEqual({item.operation for item in observed}, {"options", "snapshot"})

    def test_no_top_level_generic_context_helper_clones_remain(self):
        names = {"_prepare", "_prepared", "_prepare_default", "_writer_options", "_options", "_initial_snapshot"}
        for path in Path(__file__).parent.glob("test_writer*.py"):
            if path.name == Path(__file__).name:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.assertNotIn(node.name, names, path.name)
