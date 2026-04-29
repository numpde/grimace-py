from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

from tree_sitter import Language, Node, Parser
import tree_sitter_cpp
import tree_sitter_java


EXTRACTOR_VERSION = 1
RDKIT_VERSION = "2026.03.1"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = (
    REPO_ROOT / "tests" / "fixtures" / "rdkit_upstream_serializer_sources" / RDKIT_VERSION
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "tests" / "fixtures" / "rdkit_upstream_serializer_coverage" / f"{RDKIT_VERSION}.json"
)

SERIALIZER_TERMS = (
    "MolToSmiles",
    "MolToRandomSmilesVect",
    "SmilesWriteParams",
    "doRandom",
    "rootedAtAtom",
    "allBondsExplicit",
    "allHsExplicit",
    "isomericSmiles",
    "kekuleSmiles",
    "ignoreAtomMapNumbers",
    "MolToCXSmiles",
    "CXSmiles",
)

DEFAULT_REVIEW = {
    "status": "unreviewed",
    "claim": "needs-triage",
    "grimace_fixtures": [],
    "grimace_cases": [],
    "notes": "",
}
REVIEW_FIELDS = tuple(DEFAULT_REVIEW)


@dataclass(frozen=True)
class ExtractedBlock:
    upstream_file: str
    start_line: int
    end_line: int
    language: str
    kind: str
    name: str
    parent: str | None
    matched_terms: tuple[str, ...]
    snippet_sha256: str


def _node_text(source: bytes, node: Node) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _snippet_hash(source: bytes, start_byte: int, end_byte: int) -> str:
    return hashlib.sha256(source[start_byte:end_byte]).hexdigest()


def _matched_terms(text: str) -> tuple[str, ...]:
    return tuple(term for term in SERIALIZER_TERMS if term in text)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "unnamed"


def _entry_id(block: ExtractedBlock, seen: set[str]) -> str:
    parts = [
        _slug(block.upstream_file.removesuffix(".cpp").removesuffix(".py").removesuffix(".java")),
        _slug(block.kind),
    ]
    if block.parent:
        parts.append(_slug(block.parent))
    parts.append(_slug(block.name))
    candidate = "__".join(parts)
    if candidate not in seen:
        seen.add(candidate)
        return candidate

    candidate_with_line = f"{candidate}__line_{block.start_line}"
    if candidate_with_line not in seen:
        seen.add(candidate_with_line)
        return candidate_with_line

    suffix = 2
    while f"{candidate_with_line}_{suffix}" in seen:
        suffix += 1
    final = f"{candidate_with_line}_{suffix}"
    seen.add(final)
    return final


def _iter_nodes(node: Node) -> list[Node]:
    stack = [node]
    nodes = []
    while stack:
        current = stack.pop()
        nodes.append(current)
        stack.extend(reversed(current.named_children))
    return nodes


def _first_string_literal(source: bytes, node: Node) -> str | None:
    for child in _iter_nodes(node):
        if child.type in {"string_literal", "raw_string_literal"}:
            raw = _node_text(source, child)
            try:
                return ast.literal_eval(raw)
            except Exception:
                return raw.strip('"')
    return None


def _call_name(source: bytes, node: Node) -> str | None:
    if node.type != "call_expression":
        return None
    function = node.child_by_field_name("function")
    if function is None:
        return None
    return _node_text(source, function).strip()


def _call_expression_from_statement(node: Node) -> Node | None:
    for child in _iter_nodes(node):
        if child.type == "call_expression":
            return child
    return None


def _following_compound_statement(parent: Node, child: Node) -> Node | None:
    try:
        index = parent.named_children.index(child)
    except ValueError:
        return None
    for sibling in parent.named_children[index + 1:]:
        if sibling.type == "compound_statement":
            return sibling
        if sibling.start_point[0] > child.end_point[0] + 1:
            return None
    return None


def _nearest_cpp_test_case(source: bytes, node: Node) -> str | None:
    current = node.parent
    while current is not None:
        parent = current.parent
        if parent is not None:
            try:
                index = parent.named_children.index(current)
            except ValueError:
                index = -1
            if index > 0:
                previous = parent.named_children[index - 1]
                if previous.type == "expression_statement":
                    call = _call_expression_from_statement(previous)
                    if call is not None and _call_name(source, call) == "TEST_CASE":
                        return _first_string_literal(source, call)
        current = current.parent
    return None


def _extract_cpp(source_root: Path, rel_path: str) -> list[ExtractedBlock]:
    source_path = source_root / "source" / rel_path
    source = source_path.read_bytes()
    parser = Parser(Language(tree_sitter_cpp.language()))
    tree = parser.parse(source)

    blocks = []
    for node in _iter_nodes(tree.root_node):
        if node.type != "expression_statement" or node.parent is None:
            continue
        call = _call_expression_from_statement(node)
        if call is None:
            continue
        call_name = _call_name(source, call)
        if call_name not in {"TEST_CASE", "SECTION"}:
            continue
        body = _following_compound_statement(node.parent, node)
        if body is None:
            continue

        start_byte = node.start_byte
        end_byte = body.end_byte
        snippet = source[start_byte:end_byte].decode("utf-8", errors="replace")
        matched = _matched_terms(snippet)
        if not matched:
            continue

        name = _first_string_literal(source, call)
        if not name:
            continue

        parent = None
        kind = "cpp_test_case"
        if call_name == "SECTION":
            parent = _nearest_cpp_test_case(source, node)
            kind = "cpp_section"

        blocks.append(
            ExtractedBlock(
                upstream_file=rel_path,
                start_line=node.start_point[0] + 1,
                end_line=body.end_point[0] + 1,
                language="cpp",
                kind=kind,
                name=name,
                parent=parent,
                matched_terms=matched,
                snippet_sha256=_snippet_hash(source, start_byte, end_byte),
            )
        )
    return blocks


def _extract_python(source_root: Path, rel_path: str) -> list[ExtractedBlock]:
    source_path = source_root / "source" / rel_path
    text = source_path.read_text()
    tree = ast.parse(text)
    blocks = []

    class_stack: list[str] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.ClassDef):
            class_stack.append(node.name)
            for child in node.body:
                visit(child)
            class_stack.pop()
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test"):
                start_line = node.lineno
                end_line = node.end_lineno or node.lineno
                lines = text.splitlines(keepends=True)
                snippet = "".join(lines[start_line - 1:end_line])
                matched = _matched_terms(snippet)
                if matched:
                    blocks.append(
                        ExtractedBlock(
                            upstream_file=rel_path,
                            start_line=start_line,
                            end_line=end_line,
                            language="python",
                            kind="python_test",
                            name=node.name,
                            parent=".".join(class_stack) if class_stack else None,
                            matched_terms=matched,
                            snippet_sha256=hashlib.sha256(snippet.encode()).hexdigest(),
                        )
                    )
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return blocks


def _nearest_java_class(source: bytes, node: Node) -> str | None:
    current = node.parent
    while current is not None:
        if current.type == "class_declaration":
            name = current.child_by_field_name("name")
            if name is not None:
                return _node_text(source, name)
        current = current.parent
    return None


def _extract_java(source_root: Path, rel_path: str) -> list[ExtractedBlock]:
    source_path = source_root / "source" / rel_path
    source = source_path.read_bytes()
    parser = Parser(Language(tree_sitter_java.language()))
    tree = parser.parse(source)

    blocks = []
    for node in _iter_nodes(tree.root_node):
        if node.type != "method_declaration":
            continue
        name_node = node.child_by_field_name("name")
        if name_node is None:
            continue
        name = _node_text(source, name_node)
        if not name.startswith("test"):
            continue
        snippet = _node_text(source, node)
        matched = _matched_terms(snippet)
        if not matched:
            continue
        blocks.append(
            ExtractedBlock(
                upstream_file=rel_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                language="java",
                kind="java_test",
                name=name,
                parent=_nearest_java_class(source, node),
                matched_terms=matched,
                snippet_sha256=_snippet_hash(source, node.start_byte, node.end_byte),
            )
        )
    return blocks


def _source_files_from_manifest(source_root: Path) -> list[str]:
    manifest = json.loads((source_root / "manifest.json").read_text())
    return sorted(file["path"] for file in manifest["files"])


def extract_blocks(source_root: Path) -> list[ExtractedBlock]:
    blocks = []
    for rel_path in _source_files_from_manifest(source_root):
        if rel_path.endswith(".cpp"):
            blocks.extend(_extract_cpp(source_root, rel_path))
        elif rel_path.endswith(".py"):
            blocks.extend(_extract_python(source_root, rel_path))
        elif rel_path.endswith(".java"):
            blocks.extend(_extract_java(source_root, rel_path))
    return sorted(
        blocks,
        key=lambda block: (
            block.upstream_file,
            block.start_line,
            block.end_line,
            block.kind,
            block.parent or "",
            block.name,
        ),
    )


def _load_existing_reviews(output_path: Path) -> dict[str, dict[str, Any]]:
    if not output_path.exists():
        return {}
    payload = json.loads(output_path.read_text())
    reviews = {}
    for entry in payload.get("entries", []):
        reviews[entry["id"]] = {field: entry[field] for field in REVIEW_FIELDS if field in entry}
    return reviews


def build_manifest(source_root: Path, output_path: Path) -> dict[str, Any]:
    source_manifest = json.loads((source_root / "manifest.json").read_text())
    reviews = _load_existing_reviews(output_path)
    seen_ids: set[str] = set()
    entries = []
    for block in extract_blocks(source_root):
        entry_id = _entry_id(block, seen_ids)
        entry = {
            "id": entry_id,
            "upstream_file": block.upstream_file,
            "start_line": block.start_line,
            "end_line": block.end_line,
            "language": block.language,
            "kind": block.kind,
            "name": block.name,
            "parent": block.parent,
            "matched_terms": list(block.matched_terms),
            "snippet_sha256": block.snippet_sha256,
        }
        review = {**DEFAULT_REVIEW, **reviews.get(entry_id, {})}
        entry.update(review)
        entries.append(entry)

    return {
        "rdkit_version": source_manifest["rdkit_version"],
        "extractor_version": EXTRACTOR_VERSION,
        "source_manifest": str(source_root.relative_to(REPO_ROOT) / "manifest.json"),
        "source_commit": source_manifest["source_commit"],
        "entries": entries,
    }


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract RDKit serializer test inventory from vendored sources.",
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="write the generated manifest")
    mode.add_argument("--check", action="store_true", help="fail if the manifest is stale")
    args = parser.parse_args(argv)

    source_root = args.source_root.resolve()
    output_path = args.output.resolve()
    manifest = build_manifest(source_root, output_path)
    generated = _canonical_json(manifest)

    if args.write:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(generated)
        print(f"wrote {output_path.relative_to(REPO_ROOT)} ({len(manifest['entries'])} entries)")
        return 0

    if not output_path.exists():
        print(f"missing {output_path.relative_to(REPO_ROOT)}; run with --write", file=sys.stderr)
        return 1
    current = output_path.read_text()
    if current != generated:
        print(f"stale {output_path.relative_to(REPO_ROOT)}; rerun with --write", file=sys.stderr)
        return 1
    print(f"ok {output_path.relative_to(REPO_ROOT)} ({len(manifest['entries'])} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
