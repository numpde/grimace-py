"""Architectural boundary tests for the writer-shaped runtime stack."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WRITER_SUPPORT_PATH = (
    REPO_ROOT / "python" / "grimace" / "_south_star1" / "writer_support.py"
)


class WriterRuntimeBoundaryTest(unittest.TestCase):
    def test_writer_support_adapter_routes_through_runtime(self) -> None:
        tree = ast.parse(WRITER_SUPPORT_PATH.read_text(encoding="utf-8"))
        parent_by_child = _parent_map(tree)
        banned_runtime_modules = {
            "rdkit_adapter",
            "support_artifact",
            "support_artifact_checker",
            "support_enumeration",
            "writer_frontier",
            "writer_transitions",
        }
        banned_calls = {
            "count_writer_cursor_completions",
            "count_writer_frontier_support",
            "initial_writer_frontier_cursor",
            "iter_writer_frontier_support",
        }
        bad_imports: list[str] = []
        calls: list[str] = []
        type_checking_snapshot_imports = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                bad_imports.extend(
                    alias.name
                    for alias in node.names
                    if alias.name.split(".", 1)[0] in banned_runtime_modules
                )
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                root = module.split(".", 1)[0]
                if root in banned_runtime_modules:
                    bad_imports.append(module)
                if root == "writer_snapshot":
                    if _inside_type_checking_block(node, parent_by_child):
                        type_checking_snapshot_imports += 1
                    else:
                        bad_imports.append(module)
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
                if isinstance(node.func, ast.Attribute):
                    calls.append(node.func.attr)

        self.assertEqual(bad_imports, [])
        self.assertEqual(sorted(set(calls) & banned_calls), [])
        self.assertEqual(type_checking_snapshot_imports, 1)


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    out: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            out[child] = parent
    return out


def _inside_type_checking_block(
    node: ast.AST,
    parent_by_child: dict[ast.AST, ast.AST],
) -> bool:
    current = node
    while current in parent_by_child:
        parent = parent_by_child[current]
        if isinstance(parent, ast.If):
            test = parent.test
            if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
                return True
        current = parent
    return False


if __name__ == "__main__":
    unittest.main()
