"""Boundary tests for the confined South Star 1 proof kernel."""

from __future__ import annotations

import ast
import importlib
import unittest
from pathlib import Path

import grimace
import grimace._south_star1 as south_star1


REPO_ROOT = Path(__file__).resolve().parents[2]
SOUTH_STAR1_ROOT = REPO_ROOT / "python" / "grimace" / "_south_star1"


class SouthStar1BoundaryTest(unittest.TestCase):
    def test_core_modules_import_without_rdkit_boundary_modules(self) -> None:
        for name in south_star1.CORE_MODULES:
            with self.subTest(module=name):
                importlib.import_module(f"grimace._south_star1.{name}")

    def test_core_modules_do_not_import_rdkit(self) -> None:
        for name in south_star1.CORE_MODULES:
            path = SOUTH_STAR1_ROOT / f"{name}.py"
            with self.subTest(path=path):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                self.assertFalse(
                    _imports_rdkit(tree),
                    f"{path} imports RDKit outside the adapter/audit boundary",
                )

    def test_only_declared_boundary_modules_import_rdkit(self) -> None:
        allowed = set(south_star1.BOUNDARY_MODULES)
        for path in SOUTH_STAR1_ROOT.glob("*.py"):
            if path.name == "__init__.py":
                continue
            with self.subTest(path=path):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                self.assertEqual(
                    _imports_rdkit(tree),
                    path.stem in allowed,
                    f"{path} RDKit import status disagrees with boundary list",
                )

    def test_rdkit_boundary_is_explicit(self) -> None:
        self.assertEqual(
            south_star1.BOUNDARY_MODULES,
            ("audit_rdkit", "rdkit_adapter"),
        )

    def test_private_package_is_not_publicly_exported(self) -> None:
        self.assertNotIn("_south_star1", grimace.__all__)

    def test_deleted_south_star_prototype_stays_deleted(self) -> None:
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("grimace._south_star")

    def test_completeness_checker_does_not_import_private_generator_helpers(
        self,
    ) -> None:
        path = SOUTH_STAR1_ROOT / "completeness_checker.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))

        private_imports: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != "stereo_witness":
                continue
            private_imports.extend(
                alias.name
                for alias in node.names
                if alias.name.startswith("_")
            )

        self.assertEqual(private_imports, [])

    def test_support_artifact_checker_import_boundary_is_producer_free(
        self,
    ) -> None:
        for path in (
            SOUTH_STAR1_ROOT / "finite_space_checker.py",
            SOUTH_STAR1_ROOT / "semantic_relation_checker.py",
            SOUTH_STAR1_ROOT / "support_artifact_checker.py",
            SOUTH_STAR1_ROOT / "support_artifact_schema.py",
        ):
            with self.subTest(path=path):
                self._assert_artifact_checker_boundary(path)

    def _assert_artifact_checker_boundary(self, path: Path) -> None:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        banned_modules = {
            "audit_rdkit",
            "ordinary_semantics",
            "rdkit_adapter",
            "skeleton",
            "stereo_csp",
            "stereo_witness",
            "support_enumeration",
        }
        banned_calls = {
            "OrdinarySmilesSemantics",
            "build_stereo_csp",
            "compile_support_artifact",
            "enumerate_presentation_prefixes",
            "enumerate_exhaustive_traced_certified_stereo_support",
            "enumerate_traversal_skeletons",
            "ordinary_policy_for_facts",
            "render_stereo_traversal",
        }
        banned_imports: list[str] = []
        calls: list[str] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                banned_imports.extend(
                    alias.name
                    for alias in node.names
                    if alias.name.split(".", 1)[0] in banned_modules
                )
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.split(".", 1)[0] in banned_modules:
                    banned_imports.append(module)
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
                if isinstance(node.func, ast.Attribute):
                    calls.append(node.func.attr)

        self.assertFalse(_imports_rdkit(tree))
        self.assertEqual(banned_imports, [])

    def test_online_traversal_boundary_is_lazy_and_producer_free(self) -> None:
        for path in (
            SOUTH_STAR1_ROOT / "exhaustive_online_traversal.py",
            SOUTH_STAR1_ROOT / "online_stereo_witness.py",
            SOUTH_STAR1_ROOT / "online_continuation.py",
            SOUTH_STAR1_ROOT / "online_decoder.py",
            SOUTH_STAR1_ROOT / "online_decoder_api.py",
            SOUTH_STAR1_ROOT / "online_decoder_state.py",
            SOUTH_STAR1_ROOT / "online_decisions.py",
            SOUTH_STAR1_ROOT / "online_render_sink.py",
            SOUTH_STAR1_ROOT / "online_residual_continuation.py",
            SOUTH_STAR1_ROOT / "online_search_vm.py",
        ):
            with self.subTest(path=path):
                self._assert_online_runtime_boundary(path)

    def _assert_online_runtime_boundary(self, path: Path) -> None:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        banned_modules = {
            "audit_rdkit",
            "finite_space_checker",
            "rdkit_adapter",
            "semantic_relation_checker",
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
        banned_imports: list[str] = []
        calls: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                banned_imports.extend(
                    alias.name
                    for alias in node.names
                    if alias.name.split(".", 1)[0] in banned_modules
                )
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.split(".", 1)[0] in banned_modules:
                    banned_imports.append(module)
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
                if isinstance(node.func, ast.Attribute):
                    calls.append(node.func.attr)

        self.assertFalse(_imports_rdkit(tree))
        self.assertEqual(banned_imports, [])
        self.assertEqual(sorted(set(calls) & banned_calls), [])
        self.assertEqual(sorted(set(calls) & banned_calls), [])

    def test_online_residual_kernel_boundary_is_rdkit_and_artifact_free(self) -> None:
        for path in (
            SOUTH_STAR1_ROOT / "residual_constraints.py",
            SOUTH_STAR1_ROOT / "stereo_templates.py",
        ):
            with self.subTest(path=path):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                banned_modules = {
                    "audit_rdkit",
                    "rdkit_adapter",
                    "stereo_witness",
                    "support_artifact",
                    "support_artifact_checker",
                    "support_artifact_schema",
                    "support_enumeration",
                }
                banned_imports: list[str] = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        banned_imports.extend(
                            alias.name
                            for alias in node.names
                            if alias.name.split(".", 1)[0] in banned_modules
                        )
                    if isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        if module.split(".", 1)[0] in banned_modules:
                            banned_imports.append(module)

                self.assertFalse(_imports_rdkit(tree))
                self.assertEqual(banned_imports, [])


def _imports_rdkit(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                alias.name == "rdkit" or alias.name.startswith("rdkit.")
                for alias in node.names
            ):
                return True
        if isinstance(node, ast.ImportFrom):
            if node.module == "rdkit" or (node.module or "").startswith("rdkit."):
                return True
    return False


if __name__ == "__main__":
    unittest.main()
