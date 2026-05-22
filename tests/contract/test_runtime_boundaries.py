from __future__ import annotations

import ast
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _attribute_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}


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
                self.assertFalse(forbidden_methods & _attribute_names(path))

    def test_runtime_modules_do_not_use_prepared_graph_dict_transport(self) -> None:
        forbidden_methods = {"to_dict", "from_dict"}

        for path in self.runtime_modules:
            with self.subTest(path=path.relative_to(REPO_ROOT)):
                self.assertFalse(forbidden_methods & _attribute_names(path))

    def test_runtime_modules_do_not_inspect_prepared_mol_fragment_storage(self) -> None:
        forbidden_methods = {"fragment_atom_indices", "fragment_prepared_graph"}

        for path in self.runtime_modules:
            with self.subTest(path=path.relative_to(REPO_ROOT)):
                self.assertFalse(forbidden_methods & _attribute_names(path))


if __name__ == "__main__":
    unittest.main()
