from __future__ import annotations

import ast
from pathlib import Path
import unittest


SOUTH_STAR_TEST_ROOT = Path(__file__).resolve().parent
SOUTH_STAR_HELPER_ROOT = SOUTH_STAR_TEST_ROOT.parent / "helpers"
COMPARISON_HELPER_NAMES: frozenset[str] = frozenset(
    {
        "south_star_comparison.py",
    }
)

FORBIDDEN_IMPORT_PREFIXES: tuple[str, ...] = (
    "tests.rdkit_serialization",
    "tests.run_known_stereo_gaps",
    "tests.run_pinned_rdkit_parity",
    "tests.run_stereo_constraint_diagnostics",
    "tests.helpers.pinned_rdkit_fixtures",
    "tests.helpers.rdkit_exact_small_support",
    "tests.helpers.rdkit_known_quirks",
    "tests.helpers.rdkit_rooted_random",
    "tests.helpers.rdkit_serializer_regressions",
    "tests.helpers.rdkit_stereo_regressions",
    "tests.helpers.rdkit_writer_membership",
    "tests.helpers.stereo_constraint_model",
)
FORBIDDEN_CORE_HELPER_IMPORT_PREFIXES: tuple[str, ...] = (
    *FORBIDDEN_IMPORT_PREFIXES,
    "grimace._core",
    "grimace._deviation",
    "grimace._reference",
    "grimace._runtime",
)


def _south_star_python_files() -> tuple[Path, ...]:
    return tuple(sorted(SOUTH_STAR_TEST_ROOT.glob("*.py")))


def _south_star_helper_files() -> tuple[Path, ...]:
    return tuple(sorted(SOUTH_STAR_HELPER_ROOT.glob("south_star_*.py")))


def _core_south_star_helper_files() -> tuple[Path, ...]:
    return tuple(
        path
        for path in _south_star_helper_files()
        if path.name not in COMPARISON_HELPER_NAMES
    )


def _imported_modules(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append(node.module)
    return tuple(modules)


def _imports_public_grimace_runtime(module_name: str) -> bool:
    return module_name == "grimace" or module_name.startswith("grimace.")


class SouthStarDependencyBoundaryTests(unittest.TestCase):
    def test_south_star_tests_do_not_import_rdkit_writer_parity_surfaces(self) -> None:
        for path in _south_star_python_files():
            for module_name in _imported_modules(path):
                with self.subTest(path=path.name, module_name=module_name):
                    self.assertFalse(
                        module_name.startswith(FORBIDDEN_IMPORT_PREFIXES),
                        f"{path} imports North Star writer-parity surface "
                        f"{module_name!r}",
                    )

    def test_core_helpers_do_not_import_runtime_or_writer_parity_surfaces(self) -> None:
        for path in _core_south_star_helper_files():
            for module_name in _imported_modules(path):
                with self.subTest(path=path.name, module_name=module_name):
                    self.assertFalse(
                        module_name.startswith(FORBIDDEN_CORE_HELPER_IMPORT_PREFIXES),
                        f"{path} imports runtime or writer-parity surface "
                        f"{module_name!r}",
                    )

    def test_public_runtime_comparison_helper_is_explicitly_named(self) -> None:
        runtime_importers = tuple(
            path.name
            for path in _south_star_helper_files()
            if any(
                _imports_public_grimace_runtime(module_name)
                and not module_name.startswith("grimace._south_star")
                for module_name in _imported_modules(path)
            )
        )

        self.assertEqual(tuple(sorted(COMPARISON_HELPER_NAMES)), runtime_importers)
