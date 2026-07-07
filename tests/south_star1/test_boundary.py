"""Boundary tests for the confined South Star 1 proof kernel."""

from __future__ import annotations

import ast
import importlib
import tempfile
import unittest
from pathlib import Path

import grimace
import grimace._south_star1 as south_star1
from tests.helpers.module_boundaries import import_from_observations
from tests.helpers.module_boundaries import scan_module_boundaries


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

    def test_snapshot_advance_envelope_boundary(self) -> None:
        path = SOUTH_STAR1_ROOT / "writer_snapshot_envelope.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        self.assertFalse(_imports_rdkit(tree))
        self.assertNotIn("choice_snapshot", source)
        self.assertNotIn(
            "_maybe_writer_frontier_choice_snapshot_entry_for_emitted_text",
            source,
        )
        self.assertNotIn(
            "_writer_frontier_choice_snapshot_entry_for_emitted_text",
            source,
        )

    def test_snapshot_replay_envelope_boundary(self) -> None:
        path = SOUTH_STAR1_ROOT / "writer_snapshot_replay_envelope.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        self.assertFalse(_imports_rdkit(tree))
        self.assertNotIn("choice_snapshot", source)
        self.assertNotIn(
            "_maybe_writer_frontier_choice_snapshot_entry_for_emitted_text",
            source,
        )
        self.assertNotIn(
            "_writer_frontier_choice_snapshot_entry_for_emitted_text",
            source,
        )
        self.assertIn(
            "writer_snapshot_advance_envelope_for_emitted_text",
            source,
        )
        self.assertIn(
            "_verify_writer_snapshot_advance_envelope_from_known_source",
            source,
        )

    def test_snapshot_prefix_envelope_boundary(self) -> None:
        path = SOUTH_STAR1_ROOT / "writer_snapshot_prefix_envelope.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        self.assertFalse(_imports_rdkit(tree))
        self.assertNotIn("choice_snapshot", source)
        for name in (
            "_count_writer_frontier_choice_snapshot_supports",
            "_count_writer_frontier_choice_snapshot_completions",
            "_iter_writer_frontier_support_suffixes_from_choice_snapshot",
            "_writer_frontier_choice_snapshot_entry_for_emitted_text",
            "_maybe_writer_frontier_choice_snapshot_entry_for_emitted_text",
        ):
            self.assertNotIn(name, source)
        self.assertIn(
            "writer_snapshot_replay_envelope_for_emitted_texts",
            source,
        )
        self.assertIn("verify_writer_snapshot_replay_envelope", source)

    def test_frontier_count_envelope_boundary(self) -> None:
        path = SOUTH_STAR1_ROOT / "writer_frontier_count_envelope.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        self.assertFalse(_imports_rdkit(tree))
        self.assertNotIn("choice_snapshot", source)
        for name in (
            "_count_writer_frontier_choice_snapshot_supports",
            "_count_writer_frontier_choice_snapshot_completions",
            "_iter_writer_frontier_support_suffixes_from_choice_snapshot",
            "iter_writer_frontier_support",
            "writer_runtime_support_image",
            "support_image",
        ):
            self.assertNotIn(name, source)

    def test_count_dag_envelope_boundary(self) -> None:
        path = SOUTH_STAR1_ROOT / "writer_count_dag_envelope.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        self.assertFalse(_imports_rdkit(tree))
        self.assertNotIn("choice_snapshot", source)
        for name in (
            "_iter_writer_snapshot_certified_support_strings",
            "iter_writer_runtime_support",
            "enumerate_writer_snapshot_writer_shaped_support",
            "enumerate_prepared_writer_shaped_support",
            "support_image",
        ):
            self.assertNotIn(name, source)
        calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
        }
        self.assertNotIn("repr", calls)
        self.assertNotIn("id", calls)

    def test_support_string_envelope_boundary(self) -> None:
        path = SOUTH_STAR1_ROOT / "writer_support_string_envelope.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        self.assertFalse(_imports_rdkit(tree))
        self.assertNotIn("choice_snapshot", source)
        for name in (
            "_count_writer_frontier_choice_snapshot_supports",
            "_count_writer_frontier_choice_snapshot_completions",
            "_iter_writer_frontier_support_suffixes_from_choice_snapshot",
            "iter_writer_runtime_support",
            "enumerate_writer_snapshot_writer_shaped_support",
            "enumerate_prepared_writer_shaped_support",
            "support_image",
        ):
            self.assertNotIn(name, source)
        verifier = _function_source(tree, "verify_writer_support_string_envelope")
        self.assertNotIn("_iter_writer_snapshot_certified_support_strings", verifier)

    def test_support_image_envelope_boundary(self) -> None:
        path = SOUTH_STAR1_ROOT / "writer_support_image_envelope.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        self.assertFalse(_imports_rdkit(tree))
        self.assertNotIn("choice_snapshot", source)
        for name in (
            "_count_writer_frontier_choice_snapshot_supports",
            "_count_writer_frontier_choice_snapshot_completions",
            "_iter_writer_frontier_support_suffixes_from_choice_snapshot",
            "_writer_frontier_choice_snapshot_entry_for_emitted_text",
            "_maybe_writer_frontier_choice_snapshot_entry_for_emitted_text",
        ):
            self.assertNotIn(name, source)
        verifier = _function_source(tree, "verify_writer_support_image_envelope")
        for name in (
            "_iter_writer_snapshot_certified_support_strings",
            "iter_writer_runtime_support",
            "enumerate_writer_snapshot_writer_shaped_support",
            "enumerate_prepared_writer_shaped_support",
        ):
            self.assertNotIn(name, verifier)

    def test_support_artifact_envelope_boundary(self) -> None:
        path = SOUTH_STAR1_ROOT / "writer_support_artifact_envelope.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        self.assertFalse(_imports_rdkit(tree))
        self.assertNotIn("choice_snapshot", source)
        for name in (
            "_count_writer_frontier_choice_snapshot_supports",
            "_count_writer_frontier_choice_snapshot_completions",
            "_iter_writer_frontier_support_suffixes_from_choice_snapshot",
            "_writer_frontier_choice_snapshot_entry_for_emitted_text",
            "_maybe_writer_frontier_choice_snapshot_entry_for_emitted_text",
        ):
            self.assertNotIn(name, source)
        structural = _function_source(
            tree,
            "verify_writer_support_artifact_consistency",
        )
        for name in (
            "_checked_writer_frontier_product",
            "_support_image_certificate_for_source",
            "_iter_writer_snapshot_certified_support_strings",
            "writer_support_artifact_envelope_for_snapshot",
            "writer_support_artifact_envelope_for_prefix_read",
        ):
            self.assertNotIn(name, structural)

    def test_writer_support_artifact_checker_boundary(self) -> None:
        path = SOUTH_STAR1_ROOT / "writer_support_artifact_checker.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        self.assertFalse(_imports_rdkit(tree))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        for name in (
            "writer_frontier",
            "writer_runtime",
            "writer_snapshot",
            "writer_support",
            "writer_support_certificates",
            "writer_support_artifact_envelope",
            "rdkit_adapter",
        ):
            self.assertNotIn(name, imported_modules)
        for name in (
            "_checked_writer_frontier_product",
            "_snapshot_advance_writer_frontier_product",
            "_iter_writer_snapshot_certified_support_strings",
            "rdkit_adapter",
            "choice_snapshot",
        ):
            self.assertNotIn(name, source)

    def test_writer_support_artifact_fact_verifier_boundary(self) -> None:
        path = SOUTH_STAR1_ROOT / "writer_support_artifact_fact_verifier.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        self.assertFalse(_imports_rdkit(tree))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        for name in (
            "writer_frontier",
            "writer_runtime",
            "writer_support",
            "writer_support_certificates",
            "writer_support_artifact_envelope",
            "rdkit_adapter",
        ):
            self.assertNotIn(name, imported_modules)
        for name in (
            "_checked_writer_frontier_product",
            "_iter_writer_snapshot_certified_support_strings",
            "writer_support_artifact_envelope_for_snapshot",
            "verify_writer_support_artifact_envelope",
            "choice_snapshot",
        ):
            self.assertNotIn(name, source)

    def test_envelope_consistency_verifier_boundary(self) -> None:
        path = SOUTH_STAR1_ROOT / "writer_envelope_consistency.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        self.assertFalse(_imports_rdkit(tree))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        imported_names = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        for name in (
            "writer_frontier",
            "writer_runtime",
            "writer_snapshot",
            "writer_support",
            "writer_support_certificates",
        ):
            self.assertNotIn(name, imported_modules)
            self.assertNotIn(name, imported_names)
        for name in (
            "_checked_writer_frontier_product",
            "_snapshot_advance_writer_frontier_product",
            "_iter_writer_snapshot_certified_support_strings",
            "iter_writer_runtime_certified_support",
            "enumerate_writer_snapshot_writer_shaped_support",
            "choice_snapshot",
            "MolToSmiles" + "EnumS",
        ):
            self.assertNotIn(name, source)

    def test_durable_envelope_modules_do_not_raw_digest_terms(self) -> None:
        for path in sorted(SOUTH_STAR1_ROOT.glob("*envelope*.py")):
            if path.name == "writer_envelope_terms.py":
                continue
            with self.subTest(path=path.name):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                offenders = [
                    node.lineno
                    for node in ast.walk(tree)
                    if _is_raw_digest_term_call(node)
                ]
                self.assertEqual([], offenders)

    def test_durable_envelope_identity_helpers_are_budgeted(self) -> None:
        for path in sorted(SOUTH_STAR1_ROOT.glob("*envelope*.py")):
            if path.name == "writer_envelope_terms.py":
                continue
            with self.subTest(path=path.name):
                self.assertEqual([], _budgetless_identity_helper_calls(path))

    def test_private_package_is_not_publicly_exported(self) -> None:
        self.assertNotIn("_south_star1", grimace.__all__)

    def test_module_boundary_helper_catches_imported_module_aliases(self) -> None:
        source = "\n".join(
            (
                "from . import writer_support",
                "from grimace._south_star1 import writer_runtime",
                "if TYPE_CHECKING:",
                "    from . import writer_snapshot",
                "",
            )
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.py"
            path.write_text(source, encoding="utf-8")

            scan = scan_module_boundaries(
                path,
                banned_modules={
                    "writer_runtime",
                    "writer_support",
                },
            )
            observations = import_from_observations(
                path,
                module_root="writer_snapshot",
            )

        self.assertIn("writer_support", scan.banned_imports)
        self.assertIn(
            "grimace._south_star1.writer_runtime",
            scan.banned_imports,
        )
        self.assertEqual(len(observations), 1)
        self.assertTrue(observations[0].inside_type_checking)

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


def _is_raw_digest_term_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_digest"
        and bool(node.args)
        and isinstance(node.args[0], ast.Call)
        and isinstance(node.args[0].func, ast.Name)
        and node.args[0].func.id == "_term"
    )


def _budgetless_identity_helper_calls(path: Path) -> list[tuple[int, str]]:
    helpers = {
        "_identity_digest",
        "_identity_envelope",
        "_cursor_envelope",
        "_snapshot_identity_envelope",
        "_digest_terms_bounded",
    }
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        (node.lineno, node.func.id)
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in helpers
            and not any(keyword.arg == "budget" for keyword in node.keywords)
        )
    ]


def _function_source(tree: ast.AST, name: str) -> str:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"function {name} not found")


if __name__ == "__main__":
    unittest.main()
