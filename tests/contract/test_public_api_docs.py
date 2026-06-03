from __future__ import annotations

import ast
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _module_all(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise AssertionError("__all__ must be a literal list of strings")
        return tuple(value)
    raise AssertionError("__all__ not found")


def _documented_top_level_exports(path: Path) -> tuple[str, ...]:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = lines.index("Current top-level exports:") + 1
    exports: list[str] = []

    for line in lines[start:]:
        if not line:
            if exports:
                break
            continue
        if not line.startswith("- `") or not line.endswith("`"):
            if exports:
                break
            continue
        exports.append(line.removeprefix("- `").removesuffix("`"))

    if not exports:
        raise AssertionError("documented top-level exports not found")
    return tuple(exports)


class PublicApiDocsTests(unittest.TestCase):
    def test_python_api_page_export_list_matches_package_all(self) -> None:
        self.assertEqual(
            _module_all(REPO_ROOT / "python" / "grimace" / "__init__.py"),
            _documented_top_level_exports(REPO_ROOT / "docs" / "api" / "python.md"),
        )


if __name__ == "__main__":
    unittest.main()
