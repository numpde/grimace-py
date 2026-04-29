from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.helpers.fixture_paths import checked_in_fixture_path


DEFAULT_RDKIT_SERIALIZER_VERSION = "2026.03.1"
RDKIT_SERIALIZER_COVERAGE_ROOT = checked_in_fixture_path(
    "rdkit_upstream_serializer_coverage"
)
RDKIT_SERIALIZER_SOURCE_ROOT = checked_in_fixture_path(
    "rdkit_upstream_serializer_sources"
)

COVERAGE_STATUS_COVERED = "covered"
COVERAGE_STATUS_NEEDS_FIXTURE = "needs-fixture"
COVERAGE_STATUS_OUT_OF_SCOPE = "out-of-scope"
COVERAGE_STATUS_UNREVIEWED = "unreviewed"
COVERAGE_STATUSES = {
    COVERAGE_STATUS_COVERED,
    COVERAGE_STATUS_NEEDS_FIXTURE,
    COVERAGE_STATUS_OUT_OF_SCOPE,
    COVERAGE_STATUS_UNREVIEWED,
}
UNTRIAGED_COVERAGE_STATUSES = {
    COVERAGE_STATUS_NEEDS_FIXTURE,
    COVERAGE_STATUS_UNREVIEWED,
}

DEFAULT_COVERAGE_REVIEW: dict[str, Any] = {
    "status": COVERAGE_STATUS_UNREVIEWED,
    "claim": "needs-triage",
    "grimace_links": [],
    "notes": "",
}
REVIEW_ENTRY_FIELDS = set(DEFAULT_COVERAGE_REVIEW)

GENERATED_ENTRY_FIELDS = {
    "id",
    "upstream_file",
    "start_line",
    "end_line",
    "language",
    "kind",
    "name",
    "parent",
    "matched_terms",
    "snippet_sha256",
}
ENTRY_FIELDS = GENERATED_ENTRY_FIELDS | REVIEW_ENTRY_FIELDS
COVERAGE_FIELDS = {
    "entries",
    "extractor_version",
    "rdkit_version",
    "source_commit",
    "source_manifest",
}

GRIMACE_LINK_FIELDS = {
    "fixture",
    "cases",
    "note",
}

VALID_ENTRY_KINDS = {
    "cpp_test_case",
    "cpp_section",
    "java_test",
    "python_test",
}
VALID_ENTRY_LANGUAGES = {
    "cpp",
    "java",
    "python",
}
KIND_LANGUAGES = {
    "cpp_section": "cpp",
    "cpp_test_case": "cpp",
    "java_test": "java",
    "python_test": "python",
}


def default_serializer_coverage_path(rdkit_version: str) -> Path:
    return RDKIT_SERIALIZER_COVERAGE_ROOT / f"{rdkit_version}.json"


def default_serializer_source_root(rdkit_version: str) -> Path:
    return RDKIT_SERIALIZER_SOURCE_ROOT / rdkit_version
