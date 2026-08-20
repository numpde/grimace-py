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

    @property
    def violations(self) -> tuple[str, ...]:
        return (
            *(f"import:{module}" for module in self.banned_imports),
            *(f"imported-name:{name}" for name in self.banned_imported_names),
            *(f"call:{name}" for name in self.banned_calls),
        )

    @property
    def clean(self) -> bool:
        return not self.violations


@dataclass(frozen=True, slots=True)
class ModuleImportObservation:
    module: str
    inside_type_checking: bool


@dataclass(frozen=True, slots=True)
class ModuleImportHygieneScan:
    duplicate_bindings: tuple[str, ...]
    unused_bindings: tuple[str, ...]
    late_import_lines: tuple[int, ...]
    nested_import_lines: tuple[int, ...]
    star_import_lines: tuple[int, ...]
    extra_string_expression_lines: tuple[int, ...]

    @property
    def clean(self) -> bool:
        return not any(
            (
                self.duplicate_bindings,
                self.unused_bindings,
                self.late_import_lines,
                self.nested_import_lines,
                self.star_import_lines,
                self.extra_string_expression_lines,
            )
        )


def scan_module_import_hygiene(path: Path) -> ModuleImportHygieneScan:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    parent_by_child = _parent_map(tree)
    bindings: list[str] = []
    binding_nodes: list[tuple[str, ast.AST]] = []
    late_import_lines: list[int] = []
    nested_import_lines: list[int] = []
    star_import_lines: list[int] = []

    docstring = (
        tree.body[0]
        if tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
        else None
    )
    executable_seen = False
    for node in tree.body:
        if node is docstring:
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if any(isinstance(parent_by_child.get(node), ast.AST) for _ in (0,)):
                nested_import_lines.append(node.lineno)
            if executable_seen:
                late_import_lines.append(node.lineno)
            for alias in node.names:
                if alias.name == "*":
                    star_import_lines.append(node.lineno)
                    continue
                binding = (
                    alias.asname
                    or alias.name.split(".", 1)[0]
                    if isinstance(node, ast.Import)
                    else alias.asname or alias.name
                )
                bindings.append(binding)
                binding_nodes.append((binding, node))
            continue
        executable_seen = True

    # Top-level imports have no parent in the map; imports in any enclosing
    # function, class, conditional, or try block are nested imports.
    nested_import_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and node not in tree.body
    ]

    counts: dict[str, int] = {}
    for binding in bindings:
        counts[binding] = counts.get(binding, 0) + 1
    duplicate_bindings = sorted(
        binding for binding, count in counts.items() if count > 1
    )
    imported_nodes = {id(node) for _, node in binding_nodes}
    used = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and id(node) not in imported_nodes
    }
    unused_bindings = sorted(
        binding for binding in counts if binding not in used
    )
    extra_string_expression_lines = [
        node.lineno
        for node in tree.body[1:]
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ]
    return ModuleImportHygieneScan(
        duplicate_bindings=tuple(duplicate_bindings),
        unused_bindings=tuple(unused_bindings),
        late_import_lines=tuple(sorted(late_import_lines)),
        nested_import_lines=tuple(sorted(nested_import_lines)),
        star_import_lines=tuple(sorted(star_import_lines)),
        extra_string_expression_lines=tuple(extra_string_expression_lines),
    )


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
            imports.extend(
                _joined_import_name(module, alias.name)
                for alias in node.names
                if alias.name in banned_modules
            )
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
        matching_aliases = tuple(
            alias.name
            for alias in node.names
            if alias.name == module_root
        )
        if not _module_is_banned(module, {module_root}) and not matching_aliases:
            continue
        observations.append(
            ModuleImportObservation(
                module=(
                    module
                    if _module_is_banned(module, {module_root})
                    else _joined_import_name(module, matching_aliases[0])
                ),
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


def _joined_import_name(module: str, name: str) -> str:
    if not module:
        return name
    return f"{module}.{name}"


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
