"""Architecture checks for rich support-artifact dependency surfaces."""

from __future__ import annotations

import ast
from pathlib import Path
import tempfile
import unittest

from tests.helpers.module_boundaries import scan_module_boundaries
from tests.helpers.module_boundaries import scan_module_import_hygiene
from tests.south_star1.writer_support_artifact_test_plan import (
    WRITER_SUPPORT_ARTIFACT_TEST_DOMAINS,
)


RICH_SUPPORT_MODULES = (
    "writer_support_artifact_fixtures.py",
    "writer_support_artifact_queries.py",
    "writer_support_artifact_transition_test_support.py",
    "writer_support_artifact_graph_test_support.py",
    "writer_support_artifact_tetra_test_support.py",
    "writer_support_artifact_directional_test_support.py",
    "writer_support_artifact_directional_slow_test_support.py",
    "writer_support_artifact_directional_slow_fixtures.py",
    "writer_artifact_test_support.py",
    "writer_artifact_resealing.py",
)


class WriterSupportArtifactImportBoundaryTest(unittest.TestCase):
    def test_rich_modules_have_clean_import_hygiene(self) -> None:
        root = Path(__file__).parent
        paths = [
            root / (domain.modules[0].rsplit(".", 1)[1] + ".py")
            for domain in WRITER_SUPPORT_ARTIFACT_TEST_DOMAINS
        ]
        paths.extend(root / name for name in RICH_SUPPORT_MODULES)
        for path in paths:
            with self.subTest(path=path.name):
                scan = scan_module_import_hygiene(path)
                self.assertTrue(scan.clean, scan)
                tree = ast.parse(path.read_text(encoding="utf-8"))
                self.assertTrue(_has_one_module_docstring(tree), path)

    def test_import_hygiene_scanner_detects_all_requested_shapes(self) -> None:
        source = "\n".join(
            (
                '"""doc"""',
                "import alpha.beta as alpha_beta",
                "from gamma import delta as delta_alias",
                "import duplicate",
                "import duplicate",
                "from epsilon import " + "*",
                "def function():",
                "    import nested",
                "    return alpha_beta, delta_alias",
                "import late",
                '"""extra"""',
                "",
            )
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.py"
            path.write_text(source, encoding="utf-8")
            scan = scan_module_import_hygiene(path)

        self.assertIn("duplicate", scan.duplicate_bindings)
        self.assertIn("duplicate", scan.unused_bindings)
        self.assertIn("late", scan.unused_bindings)
        self.assertEqual(scan.star_import_lines, (6,))
        self.assertEqual(scan.nested_import_lines, (8,))
        self.assertEqual(scan.late_import_lines, (10,))
        self.assertEqual(scan.extra_string_expression_lines, (11,))

    def test_domain_modules_have_only_declared_support_dependencies(self) -> None:
        root = Path(__file__).parent
        for domain in WRITER_SUPPORT_ARTIFACT_TEST_DOMAINS:
            path = root / (domain.modules[0].rsplit(".", 1)[1] + ".py")
            if domain.name == "slow":
                continue
            if domain.name == "integration":
                banned = {
                    "writer_support_artifact_graph_test_support",
                    "writer_support_artifact_directional_test_support",
                    "writer_support_artifact_tetra_test_support",
                    "writer_support_artifact_directional_slow_test_support",
                    "writer_support_artifact_directional_slow_fixtures",
                }
            elif domain.name == "graph-relations":
                banned = {
                    "writer_support_artifact_directional_test_support",
                    "writer_support_artifact_tetra_test_support",
                    "writer_support_artifact_directional_slow_test_support",
                    "writer_support_artifact_directional_slow_fixtures",
                }
            elif domain.name in {"count-coverage", "path-identities", "obligation-replay"}:
                banned = {
                    "writer_support_artifact_graph_test_support",
                    "writer_support_artifact_directional_test_support",
                    "writer_support_artifact_tetra_test_support",
                    "writer_support_artifact_directional_slow_test_support",
                    "writer_support_artifact_directional_slow_fixtures",
                }
            elif domain.name.startswith("directional-"):
                banned = {
                    "writer_support_artifact_graph_test_support",
                    "writer_support_artifact_tetra_test_support",
                    "writer_support_artifact_directional_slow_test_support",
                    "writer_support_artifact_directional_slow_fixtures",
                }
            else:
                banned = {
                    "writer_support_artifact_graph_test_support",
                    "writer_support_artifact_directional_test_support",
                    "writer_support_artifact_directional_slow_test_support",
                    "writer_support_artifact_directional_slow_fixtures",
                }
            scan = scan_module_boundaries(path, banned_modules=banned)
            self.assertTrue(scan.clean, (path, scan.violations))

    def test_support_modules_have_one_way_domain_dependencies(self) -> None:
        root = Path(__file__).parent
        groups = {
            "writer_support_artifact_fixtures.py": set(),
            "writer_support_artifact_queries.py": set(),
            "writer_support_artifact_graph_test_support.py": {
                "writer_support_artifact_tetra_test_support",
                "writer_support_artifact_directional_test_support",
                "writer_support_artifact_directional_slow_test_support",
            },
            "writer_support_artifact_tetra_test_support.py": {
                "writer_support_artifact_graph_test_support",
                "writer_support_artifact_directional_test_support",
                "writer_support_artifact_directional_slow_test_support",
            },
            "writer_support_artifact_directional_test_support.py": {
                "writer_support_artifact_graph_test_support",
                "writer_support_artifact_tetra_test_support",
                "writer_support_artifact_directional_slow_test_support",
            },
        }
        for name, banned in groups.items():
            scan = scan_module_boundaries(root / name, banned_modules=banned)
            self.assertTrue(scan.clean, (name, scan.violations))

    def test_nonconstruction_support_has_no_forbidden_producer_imports(self) -> None:
        root = Path(__file__).parent
        construction_exceptions = {
            "writer_support_artifact_fixtures.py",
            "writer_support_artifact_directional_slow_fixtures.py",
        }
        banned = {
            "writer_frontier",
            "writer_runtime",
            "writer_support",
            "writer_support_certificates",
            "writer_support_artifact_envelope",
            "writer_snapshot_prefix_envelope",
            "rdkit_adapter",
        }
        for name in RICH_SUPPORT_MODULES:
            if name in construction_exceptions:
                continue
            scan = scan_module_boundaries(root / name, banned_modules=banned)
            self.assertTrue(scan.clean, (name, scan.violations))

    def test_bounded_modules_do_not_import_rich_artifact_producers(self) -> None:
        root = Path(__file__).parent
        banned_names = {
            "writer_support_artifact_envelope_for_snapshot",
            "writer_support_artifact_envelope_for_prefix_read",
            "writer_snapshot_prefix_read_envelope_for_emitted_texts",
        }
        for domain in WRITER_SUPPORT_ARTIFACT_TEST_DOMAINS:
            if domain.kind != "bounded":
                continue
            path = root / (domain.modules[0].rsplit(".", 1)[1] + ".py")
            scan = scan_module_boundaries(
                path,
                banned_imported_names=banned_names,
            )
            self.assertTrue(scan.clean, (path, scan.violations))

    def test_slow_support_is_owned_only_by_slow_test(self) -> None:
        root = Path(__file__).parent
        for domain in WRITER_SUPPORT_ARTIFACT_TEST_DOMAINS:
            path = root / (domain.modules[0].rsplit(".", 1)[1] + ".py")
            source = path.read_text(encoding="utf-8")
            if "writer_support_artifact_directional_slow" in source:
                self.assertEqual(domain.name, "slow")

    def test_no_duplicate_fixture_or_identity_authorities_remain(self) -> None:
        root = Path(__file__).parent
        source = "\n".join(
            (root / name).read_text(encoding="utf-8") for name in RICH_SUPPORT_MODULES
        )
        self.assertNotIn("rdkit_" + "support_artifact_verification", source)
        self.assertNotIn("text_projection_" + "identity_digest", source)
        for tree in (
            ast.parse((root / name).read_text(encoding="utf-8"))
            for name in RICH_SUPPORT_MODULES
        ):
            self.assertFalse(_has_raw_cursor_field_access(tree))


def _has_one_module_docstring(tree: ast.Module) -> bool:
    if not tree.body:
        return False
    first = tree.body[0]
    if not (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return False
    return not any(
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        for node in tree.body[1:]
    )


def _has_raw_cursor_field_access(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        keys = []
        current: ast.AST = node
        while isinstance(current, ast.Subscript):
            if not isinstance(current.slice, ast.Constant) or not isinstance(
                current.slice.value, str
            ):
                break
            keys.append(current.slice.value)
            current = current.value
        if keys[:2] == ["fields", "terms"]:
            return True
    return False


if __name__ == "__main__":
    unittest.main()
