from __future__ import annotations

import ast
import inspect
from pathlib import Path
import re
import unittest

import grimace
from tests.helpers.sampling import SAMPLING_MODE_PAIRS


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


def _documented_signature(path: Path, label: str) -> str:
    prefix = f"`{label}("
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix) and line.endswith("`"):
            return line.removeprefix("`").removesuffix("`")
    raise AssertionError(f"documented signature for {label!r} not found")


def _documented_sampling_mode_pairs(path: Path) -> tuple[tuple[str, str], ...]:
    pattern = re.compile(r'^- `"([^`]+)"` / `"([^`]+)"`:')
    return tuple(
        match.groups()
        for line in path.read_text(encoding="utf-8").splitlines()
        if (match := pattern.match(line))
    )


def _compact_signature(
    label: str,
    callable_object: object,
    *,
    skip_self: bool = False,
) -> str:
    signature = inspect.signature(callable_object)
    params = tuple(signature.parameters.values())
    if skip_self:
        params = params[1:]

    compact = signature.replace(
        parameters=tuple(
            param.replace(annotation=inspect.Parameter.empty)
            for param in params
        ),
        return_annotation=inspect.Signature.empty,
    )
    return f"{label}{str(compact).replace(chr(39), chr(34))}"


class PublicApiDocsTests(unittest.TestCase):
    def test_python_api_page_export_list_matches_package_all(self) -> None:
        self.assertEqual(
            _module_all(REPO_ROOT / "python" / "grimace" / "__init__.py"),
            _documented_top_level_exports(REPO_ROOT / "docs" / "api" / "python.md"),
        )

    def test_python_api_page_signatures_match_public_callables(self) -> None:
        docs_path = REPO_ROOT / "docs" / "api" / "python.md"
        cases = (
            ("PrepareMol", grimace.PrepareMol, False),
            ("PreparedMol.to_bytes", grimace.PreparedMol.to_bytes, True),
            ("PreparedMol.from_bytes", grimace.PreparedMol.from_bytes, False),
            ("MolToSmilesEnum", grimace.MolToSmilesEnum, False),
            ("MolToSmilesDecoder", grimace.MolToSmilesDecoder.__init__, True),
            (
                "MolToSmilesDeterminizedDecoder",
                grimace.MolToSmilesDeterminizedDecoder.__init__,
                True,
            ),
            ("MolToSmilesSample", grimace.MolToSmilesSample, False),
            ("MolToSmilesDeviation", grimace.MolToSmilesDeviation, False),
        )

        for label, callable_object, skip_self in cases:
            with self.subTest(label=label):
                self.assertEqual(
                    _compact_signature(
                        label,
                        callable_object,
                        skip_self=skip_self,
                    ),
                    _documented_signature(docs_path, label),
                )

    def test_python_api_page_sampling_pairs_match_public_contract(self) -> None:
        documented_pairs = _documented_sampling_mode_pairs(
            REPO_ROOT / "docs" / "api" / "python.md"
        )

        self.assertEqual(
            frozenset(SAMPLING_MODE_PAIRS),
            frozenset(documented_pairs),
        )
        self.assertEqual(len(SAMPLING_MODE_PAIRS), len(documented_pairs))


if __name__ == "__main__":
    unittest.main()
