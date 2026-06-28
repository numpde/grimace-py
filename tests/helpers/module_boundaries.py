"""Small AST helpers for architectural boundary tests."""

from __future__ import annotations

import ast
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ModuleBoundaryScan:
    """Import/call observations for one Python module.

    Boundary tests should stay declarative: each test names the forbidden edges,
    while this helper owns the repetitive AST walking.
    """

    banned_imports: tuple[str, ...]
    banned_imported_names: tuple[str, ...]
    banned_calls: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModuleImportObservation:
    module: str
    inside_type_checking: bool


def scan_module_boundaries(
    path: Path,
    *,
    banned_modules: Collection[str] = frozenset(),
    banned_imported_names: Collection[str] = frozenset(),
    banned_calls: Collection[str] = frozenset(),
) -> ModuleBoundaryScan:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: list[str] = []
    imported_names: list[str] = []
    calls: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(
                alias.name
                for alias in node.names
                if _module_is_banned(alias.name, banned_modules)
            )
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if _module_is_banned(module, banned_modules):
                imports.append(module)
            imported_names.extend(
                alias.name
                for alias in node.names
                if alias.name in banned_imported_names
            )
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name in banned_calls:
                calls.append(name)

    return ModuleBoundaryScan(
        banned_imports=tuple(imports),
        banned_imported_names=tuple(imported_names),
        banned_calls=tuple(calls),
    )


def import_from_observations(
    path: Path,
    *,
    module_root: str,
) -> tuple[ModuleImportObservation, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    parent_by_child = _parent_map(tree)
    observations: list[ModuleImportObservation] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""
        if not _module_is_banned(module, {module_root}):
            continue
        observations.append(
            ModuleImportObservation(
                module=module,
                inside_type_checking=_inside_type_checking_block(
                    node,
                    parent_by_child,
                ),
            )
        )

    return tuple(observations)


def _module_is_banned(module: str, banned_modules: Collection[str]) -> bool:
    if module in banned_modules:
        return True
    return any(part in banned_modules for part in module.split("."))


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    out: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            out[child] = parent
    return out


def _inside_type_checking_block(
    node: ast.AST,
    parent_by_child: dict[ast.AST, ast.AST],
) -> bool:
    current = node
    while current in parent_by_child:
        parent = parent_by_child[current]
        if isinstance(parent, ast.If):
            test = parent.test
            if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
                return True
        current = parent
    return False
