from __future__ import annotations

import ast
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]


class RuntimeBoundaryTests(unittest.TestCase):
    runtime_modules = (
        REPO_ROOT / "python" / "grimace" / "_runtime.py",
        REPO_ROOT / "python" / "grimace" / "_deviation.py",
    )

    def test_runtime_modules_do_not_directly_import_rdkit(self) -> None:
        for path in self.runtime_modules:
            with self.subTest(path=path.relative_to(REPO_ROOT)):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                imported_roots: set[str] = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imported_roots.update(
                            alias.name.partition(".")[0]
                            for alias in node.names
                        )
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imported_roots.add(node.module.partition(".")[0])

                self.assertNotIn("rdkit", imported_roots)

    def test_runtime_modules_do_not_call_rdkit_methods_directly(self) -> None:
        forbidden_methods = {"GetMolFrags", "GetNumAtoms"}

        for path in self.runtime_modules:
            with self.subTest(path=path.relative_to(REPO_ROOT)):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                method_names = {
                    node.attr
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Attribute)
                }
                self.assertFalse(forbidden_methods & method_names)


if __name__ == "__main__":
    unittest.main()
