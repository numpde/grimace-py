"""Authoritative bounded test-domain plan for rich support artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import ast
import importlib
import inspect
from pathlib import Path
from typing import Literal
import unittest


WriterSupportArtifactDomainKind = Literal["bounded", "slow-diagnostic"]


@dataclass(frozen=True, slots=True)
class WriterSupportArtifactTestDomain:
    name: str
    modules: tuple[str, ...]
    kind: WriterSupportArtifactDomainKind
    role: str


_PREFIX = "tests.south_star1."
_EXPECTED_DOMAIN_TEST_COUNTS = {
    "integration": 15,
    "graph-relations": 15,
    "count-coverage": 11,
    "path-identities": 13,
    "obligation-replay": 15,
    "directional-acyclic": 3,
    "directional-acyclic-forgeries": 2,
    "tetra-transitions": 21,
    "tetra-lifecycle": 15,
    "slow": 6,
}

WRITER_SUPPORT_ARTIFACT_TEST_DOMAINS = (
    WriterSupportArtifactTestDomain(
        "integration",
        ("tests.south_star1.test_writer_support_artifact_fact_verifier",),
        "bounded",
        "top-level facts-bound composition",
    ),
    WriterSupportArtifactTestDomain(
        "graph-relations",
        ("tests.south_star1.test_writer_support_artifact_graph_relations",),
        "bounded",
        "graph/ring and local evidence relations",
    ),
    WriterSupportArtifactTestDomain(
        "count-coverage",
        ("tests.south_star1.test_writer_support_artifact_count_coverage",),
        "bounded",
        "count-DAG and support-image coverage",
    ),
    WriterSupportArtifactTestDomain(
        "path-identities",
        ("tests.south_star1.test_writer_support_artifact_path_identities",),
        "bounded",
        "support-string and terminal identities",
    ),
    WriterSupportArtifactTestDomain(
        "obligation-replay",
        ("tests.south_star1.test_writer_support_artifact_obligation_replay",),
        "bounded",
        "obligation classification and replay credit",
    ),
    WriterSupportArtifactTestDomain(
        "directional-acyclic",
        ("tests.south_star1.test_writer_support_artifact_directional_acyclic",),
        "bounded",
        "bounded directional replay",
    ),
    WriterSupportArtifactTestDomain(
        "directional-acyclic-forgeries",
        ("tests.south_star1.test_writer_support_artifact_directional_forgeries",),
        "bounded",
        "bounded directional forgery replay",
    ),
    WriterSupportArtifactTestDomain(
        "tetra-transitions",
        ("tests.south_star1.test_writer_support_artifact_tetra_transitions",),
        "bounded",
        "specified tetrahedral transitions",
    ),
    WriterSupportArtifactTestDomain(
        "tetra-lifecycle",
        ("tests.south_star1.test_writer_support_artifact_tetra_lifecycle",),
        "bounded",
        "specified tetrahedral lifecycle",
    ),
    WriterSupportArtifactTestDomain(
        "slow",
        ("tests.south_star1.test_writer_support_artifact_slow",),
        "slow-diagnostic",
        "exhaustive rich support-artifact materialization and replay diagnostics",
    ),
)


def domain_by_name(name: str) -> WriterSupportArtifactTestDomain:
    for domain in WRITER_SUPPORT_ARTIFACT_TEST_DOMAINS:
        if domain.name == name:
            return domain
    raise ValueError(f"unknown support-artifact test domain: {name}")


def test_ids_for_domain(domain: WriterSupportArtifactTestDomain) -> tuple[str, ...]:
    names: list[str] = []
    for module_name in domain.modules:
        module = importlib.import_module(module_name)
        for _, value in inspect.getmembers(module, inspect.isclass):
            if issubclass(value, unittest.TestCase):
                names.extend(
                    f"{module_name}.{value.__name__}.{test_name}"
                    for test_name in unittest.defaultTestLoader.getTestCaseNames(value)
                )
    return tuple(sorted(names))


def validate_writer_support_artifact_test_plan() -> None:
    domains = WRITER_SUPPORT_ARTIFACT_TEST_DOMAINS
    names = [domain.name for domain in domains]
    if len(names) != len(set(names)):
        raise AssertionError("duplicate support-artifact test domain")
    if not domains or domains[-1].kind != "slow-diagnostic":
        raise AssertionError("slow support-artifact domain must be last")
    if sum(domain.kind == "slow-diagnostic" for domain in domains) != 1:
        raise AssertionError("expected exactly one slow support-artifact domain")
    slow_domain = domains[-1]
    if len(slow_domain.modules) != 1:
        raise AssertionError("slow diagnostic must own one module")
    directional_domains = [domain for domain in domains if domain.name.startswith("directional-")]
    if [domain.name for domain in directional_domains] != [
        "directional-acyclic",
        "directional-acyclic-forgeries",
    ]:
        raise AssertionError("directional domains must be consecutive before tetrahedral domains")

    module_owners: dict[str, str] = {}
    all_ids: list[str] = []
    for domain in domains:
        if not domain.modules:
            raise AssertionError(f"domain has no modules: {domain.name}")
        if not domain.role.strip():
            raise AssertionError(f"domain has no role: {domain.name}")
        for module_name in domain.modules:
            if not module_name.startswith(_PREFIX) or module_name.endswith("."):
                raise AssertionError(f"module outside South Star tests: {module_name}")
            if module_name in module_owners:
                raise AssertionError(f"module has two owners: {module_name}")
            module_owners[module_name] = domain.name
            importlib.import_module(module_name)
            path_text = getattr(importlib.import_module(module_name), "__file__", "")
            source = Path(path_text).read_text(encoding="utf-8")
            tree = ast.parse(source)
            if any(
                isinstance(node, ast.ImportFrom)
                and any(alias.name == "*" for alias in node.names)
                for node in ast.walk(tree)
            ):
                raise AssertionError(f"wildcard import in domain module: {module_name}")
        domain_ids = test_ids_for_domain(domain)
        if not domain_ids:
            raise AssertionError(f"domain has no tests: {domain.name}")
        if len(domain_ids) != _EXPECTED_DOMAIN_TEST_COUNTS[domain.name]:
            raise AssertionError(f"unexpected test count: {domain.name}")
        direct_methods = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            for node in node.body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        }
        expected_methods = {test_id.rsplit(".", 1)[1] for test_id in domain_ids}
        if direct_methods != expected_methods:
            raise AssertionError(f"test method is not physically owned: {module_name}")
        all_ids.extend(domain_ids)
    if len(all_ids) != len(set(all_ids)):
        raise AssertionError("duplicate support-artifact test ID")
    if len(all_ids) != 116:
        raise AssertionError("expected 116 rich support-artifact test IDs")

    support_root = Path(__file__).parent
    for path in support_root.glob("writer_support_artifact_*.py"):
        if path.name.startswith("test_"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if any(
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("tests.south_star1.test_")
            for node in ast.walk(tree)
        ):
            raise AssertionError(f"test-to-test import in support module: {path}")
        if any(
            isinstance(node, ast.ImportFrom)
            and any(alias.name == "*" for alias in node.names)
            for node in ast.walk(tree)
        ):
            raise AssertionError(f"wildcard import in support module: {path}")
        if any(
            isinstance(node, ast.ClassDef)
            and any(
                isinstance(base, ast.Attribute) and base.attr == "TestCase"
                for base in node.bases
            )
            for node in tree.body
        ):
            raise AssertionError(f"TestCase in support module: {path}")
        if len(path.read_text(encoding="utf-8").splitlines()) > 700:
            raise AssertionError(f"support module exceeds 700 lines: {path}")


def bounded_domains() -> tuple[WriterSupportArtifactTestDomain, ...]:
    return tuple(domain for domain in WRITER_SUPPORT_ARTIFACT_TEST_DOMAINS if domain.kind == "bounded")
