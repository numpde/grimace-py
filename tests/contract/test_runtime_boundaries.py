from __future__ import annotations

import ast
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _attribute_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}


def _name_ids(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def _function_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }


def _string_constants(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def _imported_module_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.update(
                f"{node.module}.{alias.name}" for alias in node.names
            )
    return imported


def _caught_exception_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    caught: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler) or node.type is None:
            continue
        if isinstance(node.type, ast.Name):
            caught.add(node.type.id)
        elif isinstance(node.type, ast.Tuple):
            caught.update(
                item.id for item in node.type.elts if isinstance(item, ast.Name)
            )
    return caught


def _import_aliases(path: Path) -> set[tuple[str, str, str | None]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    aliases: set[tuple[str, str, str | None]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            aliases.update(
                (node.module, alias.name, alias.asname)
                for alias in node.names
            )
    return aliases


def _private_package_imports(path: Path) -> set[str]:
    return {
        imported_name
        for module_name, imported_name, _ in _import_aliases(path)
        if module_name == "grimace" and imported_name.startswith("_")
    }


class RuntimeBoundaryTests(unittest.TestCase):
    deviation_module = REPO_ROOT / "python" / "grimace" / "_deviation.py"
    public_package_module = REPO_ROOT / "python" / "grimace" / "__init__.py"
    preparation_module = REPO_ROOT / "python" / "grimace" / "_prepared_mol.py"
    runtime_input_module = REPO_ROOT / "python" / "grimace" / "_runtime_inputs.py"
    runtime_module = REPO_ROOT / "python" / "grimace" / "_runtime.py"
    runtime_state_module = REPO_ROOT / "python" / "grimace" / "_runtime_states.py"
    runtime_modules = (
        REPO_ROOT / "python" / "grimace" / "_runtime.py",
        REPO_ROOT / "python" / "grimace" / "_runtime_graphs.py",
        REPO_ROOT / "python" / "grimace" / "_runtime_inputs.py",
        REPO_ROOT / "python" / "grimace" / "_runtime_states.py",
        REPO_ROOT / "python" / "grimace" / "_deviation.py",
    )
    reference_rooted_modules = (
        REPO_ROOT / "python" / "grimace" / "_reference" / "rooted" / "connected_nonstereo.py",
        REPO_ROOT / "python" / "grimace" / "_reference" / "rooted" / "connected_stereo.py",
    )
    package_modules = tuple(sorted((REPO_ROOT / "python" / "grimace").glob("*.py")))

    def test_runtime_modules_do_not_directly_import_rdkit(self) -> None:
        for path in self.runtime_modules:
            with self.subTest(path=path.relative_to(REPO_ROOT)):
                imported_roots = {
                    module_name.partition(".")[0]
                    for module_name in _imported_module_names(path)
                }
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

    def test_reference_rooted_modules_do_not_use_prepared_graph_dict_transport(self) -> None:
        forbidden_methods = {"to_dict", "from_dict"}

        for path in self.reference_rooted_modules:
            with self.subTest(path=path.relative_to(REPO_ROOT)):
                self.assertFalse(forbidden_methods & _attribute_names(path))

    def test_runtime_modules_do_not_import_private_modules_through_package(self) -> None:
        for path in self.runtime_modules:
            with self.subTest(path=path.relative_to(REPO_ROOT)):
                self.assertFalse(_private_package_imports(path))

    def test_package_modules_do_not_hide_missing_required_imports(self) -> None:
        forbidden = {"ImportError", "ModuleNotFoundError"}

        for path in self.package_modules:
            with self.subTest(path=path.relative_to(REPO_ROOT)):
                self.assertFalse(forbidden & _caught_exception_names(path))

    def test_runtime_modules_import_core_directly(self) -> None:
        core_dependent_modules = (
            self.preparation_module,
            self.runtime_module,
            REPO_ROOT / "python" / "grimace" / "_runtime_graphs.py",
        )

        for path in core_dependent_modules:
            with self.subTest(path=path.relative_to(REPO_ROOT)):
                imported_names = _imported_module_names(path)
                self.assertIn("grimace._core", imported_names)
                self.assertNotIn("importlib", imported_names)

    def test_runtime_modules_do_not_inspect_prepared_mol_fragment_storage(self) -> None:
        forbidden_methods = {"fragment_atom_indices", "fragment_prepared_graph"}

        for path in self.runtime_modules:
            with self.subTest(path=path.relative_to(REPO_ROOT)):
                self.assertFalse(forbidden_methods & _attribute_names(path))

    def test_preparation_module_does_not_import_public_runtime(self) -> None:
        self.assertNotIn(
            "grimace._runtime",
            _imported_module_names(self.preparation_module),
        )
        self.assertNotIn(
            "grimace._runtime",
            _string_constants(self.preparation_module),
        )

    def test_runtime_input_module_does_not_import_reference_graphs(self) -> None:
        imported_names = _imported_module_names(self.runtime_input_module)
        self.assertFalse(
            {
                name
                for name in imported_names
                if name.startswith("grimace._reference.prepared_graph")
            }
        )

    def test_public_runtime_module_does_not_import_prepared_mol_wrapper(self) -> None:
        imported_names = _imported_module_names(self.runtime_module)
        self.assertNotIn("grimace._prepared_mol", imported_names)
        self.assertNotIn("grimace._prepared_mol.PreparedMol", imported_names)
        self.assertNotIn("grimace._prepared_mol", _string_constants(self.runtime_module))

    def test_public_runtime_module_does_not_reexport_owned_helpers(self) -> None:
        owning_module_imports = {
            ("grimace._runtime_graphs", "prepare_smiles_graph"),
            ("grimace._runtime_inputs", "MolToSmilesFlags"),
            ("grimace._reference.prepared_graph", "CONNECTED_NONSTEREO_SURFACE"),
            ("grimace._reference.prepared_graph", "CONNECTED_STEREO_SURFACE"),
            ("grimace._reference.prepared_graph", "PREPARED_SMILES_GRAPH_SCHEMA_VERSION"),
        }

        for module_name, imported_name, alias in _import_aliases(self.runtime_module):
            if (module_name, imported_name) in owning_module_imports:
                with self.subTest(module_name=module_name, imported_name=imported_name):
                    self.assertIsNotNone(alias)
                    assert alias is not None
                    self.assertTrue(alias.startswith("_"), alias)

    def test_public_package_import_requires_runtime_directly(self) -> None:
        imported_names = _imported_module_names(self.public_package_module)
        self.assertIn("grimace._runtime", imported_names)
        self.assertIn("grimace._deviation", imported_names)
        self.assertNotIn("importlib", imported_names)

        forbidden_names = {
            "_CORE_IMPORT_ERROR",
            "_ImportErrorRuntimeBase",
            "_require_runtime",
        }
        self.assertFalse(forbidden_names & _name_ids(self.public_package_module))

    def test_deviation_module_does_not_inspect_decoder_state_storage(self) -> None:
        self.assertNotIn("_state", _attribute_names(self.deviation_module))
        self.assertNotIn("_state_cache_key", _attribute_names(self.deviation_module))
        self.assertNotIn("_state_cache_key", _name_ids(self.deviation_module))

    def test_runtime_states_do_not_use_eager_successor_names(self) -> None:
        forbidden_methods = {
            "choice_successors",
            "grouped_successors",
            "choice_successor_states",
            "grouped_successor_states",
        }

        self.assertFalse(forbidden_methods & _attribute_names(self.runtime_state_module))
        self.assertFalse(forbidden_methods & _function_names(self.runtime_state_module))


if __name__ == "__main__":
    unittest.main()
