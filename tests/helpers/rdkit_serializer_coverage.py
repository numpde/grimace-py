from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from tests.helpers.fixture_paths import checked_in_fixture_path


DEFAULT_RDKIT_SERIALIZER_VERSION = "2026.03.1"
RDKIT_SERIALIZER_COVERAGE_ROOT = checked_in_fixture_path(
    "rdkit_upstream_serializer_coverage"
)
RDKIT_SERIALIZER_SOURCE_ROOT = checked_in_fixture_path(
    "rdkit_upstream_serializer_sources"
)

COVERAGE_STATUS_COVERED = "covered"
COVERAGE_STATUS_KNOWN_GAP = "known-gap"
COVERAGE_STATUS_NEEDS_FIXTURE = "needs-fixture"
COVERAGE_STATUS_OUT_OF_SCOPE = "out-of-scope"
COVERAGE_STATUS_UNREVIEWED = "unreviewed"
COVERAGE_STATUSES = {
    COVERAGE_STATUS_COVERED,
    COVERAGE_STATUS_KNOWN_GAP,
    COVERAGE_STATUS_NEEDS_FIXTURE,
    COVERAGE_STATUS_OUT_OF_SCOPE,
    COVERAGE_STATUS_UNREVIEWED,
}
UNTRIAGED_COVERAGE_STATUSES = {
    COVERAGE_STATUS_NEEDS_FIXTURE,
    COVERAGE_STATUS_UNREVIEWED,
}
STATUSES_REQUIRING_EXECUTABLE_LINKS = {
    COVERAGE_STATUS_COVERED,
    COVERAGE_STATUS_KNOWN_GAP,
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


def load_serializer_coverage(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"{path} must define entries as a list")
    if not entries:
        raise ValueError(f"{path} must define at least one entry")
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"{path} contains a non-object ledger entry")
        status = entry.get("status")
        if not isinstance(status, str) or not status:
            raise ValueError(f"{path} contains a ledger entry without status")
        if status not in COVERAGE_STATUSES:
            raise ValueError(f"{path} contains unknown ledger status: {status!r}")
    return payload


def checked_in_fixture_case_ids_by_path(
    *,
    fixture_root: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, frozenset[str]]:
    if fixture_root is None:
        fixture_root = checked_in_fixture_path()
    if repo_root is None:
        repo_root = fixture_root.parents[1]

    case_ids_by_fixture: dict[str, frozenset[str]] = {}
    for fixture_path in sorted(fixture_root.rglob("*.json")):
        relative_path = fixture_path.relative_to(repo_root).as_posix()
        try:
            payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        raw_cases = payload.get("cases")
        if not isinstance(raw_cases, list):
            continue
        case_ids = frozenset(
            raw_case["id"]
            for raw_case in raw_cases
            if isinstance(raw_case, dict) and type(raw_case.get("id")) is str
        )
        if case_ids:
            case_ids_by_fixture[relative_path] = case_ids
    return case_ids_by_fixture


def validate_serializer_coverage_links(
    coverage: dict[str, Any],
    *,
    coverage_path: Path,
    fixture_case_ids: dict[str, frozenset[str]],
) -> None:
    entries = cast(list[dict[str, Any]], coverage["entries"])
    for entry in entries:
        validate_serializer_entry_links(
            entry,
            coverage_path=coverage_path,
            fixture_case_ids=fixture_case_ids,
        )


def validate_serializer_entry_links(
    entry: dict[str, Any],
    *,
    coverage_path: Path,
    fixture_case_ids: dict[str, frozenset[str]],
) -> None:
    entry_id = entry.get("id")
    status = cast(str, entry["status"])
    links = entry.get("grimace_links")
    if not isinstance(links, list):
        raise ValueError(
            f"{coverage_path}: entry {entry_id!r} must define grimace_links"
        )
    if status in STATUSES_REQUIRING_EXECUTABLE_LINKS and not links:
        raise ValueError(
            f"{coverage_path}: entry {entry_id!r} with status {status!r} "
            "must link executable fixture cases"
        )
    linkable_statuses = {
        COVERAGE_STATUS_NEEDS_FIXTURE,
        *STATUSES_REQUIRING_EXECUTABLE_LINKS,
    }
    if status not in linkable_statuses:
        if links:
            raise ValueError(
                f"{coverage_path}: entry {entry_id!r} with status {status!r} "
                "must not link executable fixture cases"
            )

    seen_fixtures: set[str] = set()
    for link in links:
        if not isinstance(link, dict):
            raise ValueError(f"{coverage_path}: entry {entry_id!r} has invalid link")
        if set(link) != GRIMACE_LINK_FIELDS:
            raise ValueError(
                f"{coverage_path}: entry {entry_id!r} has invalid link fields"
            )

        fixture = link.get("fixture")
        cases = link.get("cases")
        note = link.get("note")
        if not isinstance(fixture, str) or not fixture:
            raise ValueError(
                f"{coverage_path}: entry {entry_id!r} has link without fixture"
            )
        if fixture in seen_fixtures:
            raise ValueError(
                f"{coverage_path}: entry {entry_id!r} links fixture twice: {fixture}"
            )
        seen_fixtures.add(fixture)
        if fixture not in fixture_case_ids:
            raise ValueError(
                f"{coverage_path}: entry {entry_id!r} references missing fixture: "
                f"{fixture}"
            )
        if not isinstance(cases, list) or not cases or not all(
            isinstance(case_id, str) and case_id for case_id in cases
        ):
            raise ValueError(
                f"{coverage_path}: entry {entry_id!r} has link without cases"
            )
        if len(set(cases)) != len(cases):
            raise ValueError(
                f"{coverage_path}: entry {entry_id!r} has duplicate linked cases"
            )
        if not (note is None or type(note) is str):
            raise ValueError(
                f"{coverage_path}: entry {entry_id!r} has invalid link note"
            )

        missing = sorted(set(cases) - fixture_case_ids[fixture])
        if missing:
            raise ValueError(
                f"{coverage_path}: entry {entry_id!r} references missing "
                f"cases in {fixture}: {missing}"
            )
