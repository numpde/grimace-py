from __future__ import annotations

import ast
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]


class RuntimeBoundaryTests(unittest.TestCase):
    def test_runtime_modules_do_not_directly_import_rdkit(self) -> None:
        runtime_modules = (
            REPO_ROOT / "python" / "grimace" / "_runtime.py",
            REPO_ROOT / "python" / "grimace" / "_deviation.py",
        )

        for path in runtime_modules:
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


if __name__ == "__main__":
    unittest.main()
