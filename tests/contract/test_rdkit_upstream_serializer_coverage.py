from __future__ import annotations

import json
from pathlib import Path
import re
import unittest

from tests.helpers.fixture_paths import CHECKED_IN_FIXTURE_ROOT, checked_in_fixture_path
from tests.helpers.rdkit_serializer_coverage import (
    COVERAGE_FIELDS,
    COVERAGE_STATUS_COVERED,
    COVERAGE_STATUS_KNOWN_GAP,
    COVERAGE_STATUS_NEEDS_FIXTURE,
    COVERAGE_STATUSES,
    ENTRY_FIELDS,
    GRIMACE_LINK_FIELDS,
    KIND_LANGUAGES,
    RDKIT_SERIALIZER_COVERAGE_ROOT,
    RDKIT_SERIALIZER_SOURCE_ROOT,
    VALID_ENTRY_KINDS,
    VALID_ENTRY_LANGUAGES,
)


REPO_ROOT = CHECKED_IN_FIXTURE_ROOT.parents[1]
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _line_span(text: str, start_line: int, end_line: int) -> str:
    lines = text.splitlines(keepends=True)
    return "".join(lines[start_line - 1:end_line])


class RdkitUpstreamSerializerCoverageFixtureTest(unittest.TestCase):
    def test_coverage_inventory_matches_upstream_source_snippets(self) -> None:
        fixture_paths = tuple(sorted(RDKIT_SERIALIZER_COVERAGE_ROOT.glob("*.json")))
        self.assertTrue(fixture_paths)

        for fixture_path in fixture_paths:
            with self.subTest(fixture=fixture_path.name):
                self._assert_coverage_fixture(fixture_path)

    def _assert_coverage_fixture(self, fixture_path: Path) -> None:
        coverage = json.loads(fixture_path.read_text())
        self.assertEqual(COVERAGE_FIELDS, set(coverage))

        rdkit_version = fixture_path.stem
        self.assertEqual(rdkit_version, coverage["rdkit_version"])
        self.assertIs(type(coverage["extractor_version"]), int)
        self.assertGreater(coverage["extractor_version"], 0)

        source_manifest_path = RDKIT_SERIALIZER_SOURCE_ROOT / rdkit_version / "manifest.json"
        self.assertTrue(source_manifest_path.is_file())
        self.assertEqual(
            source_manifest_path.relative_to(REPO_ROOT).as_posix(),
            coverage["source_manifest"],
        )
        source_manifest = json.loads(source_manifest_path.read_text())
        self.assertEqual(source_manifest["source_commit"], coverage["source_commit"])
        manifest_paths = {entry["path"] for entry in source_manifest["files"]}
        fixture_case_ids = self._load_fixture_case_ids()

        entries = coverage["entries"]
        self.assertIs(type(entries), list)
        self.assertTrue(entries)

        seen_ids: set[str] = set()
        previous_sort_key: tuple[object, ...] | None = None
        for entry in entries:
            with self.subTest(entry_id=entry.get("id")):
                self._assert_entry(
                    entry,
                    seen_ids,
                    manifest_paths,
                    fixture_case_ids,
                    rdkit_version,
                )
                sort_key = self._entry_sort_key(entry)
                if previous_sort_key is not None:
                    self.assertLessEqual(previous_sort_key, sort_key)
                previous_sort_key = sort_key

    def _assert_entry(
        self,
        entry: dict[str, object],
        seen_ids: set[str],
        manifest_paths: set[str],
        fixture_case_ids: dict[str, set[str]],
        rdkit_version: str,
    ) -> None:
        self.assertEqual(ENTRY_FIELDS, set(entry))

        entry_id = self._required_string(entry, "id")
        self.assertNotIn(entry_id, seen_ids)
        seen_ids.add(entry_id)

        upstream_file = self._required_string(entry, "upstream_file")
        self.assertIn(upstream_file, manifest_paths)
        self.assertIn(entry["language"], VALID_ENTRY_LANGUAGES)
        self.assertIn(entry["kind"], VALID_ENTRY_KINDS)
        self.assertEqual(KIND_LANGUAGES[entry["kind"]], entry["language"])
        self.assert_required_optional_string(entry, "parent")
        self._required_string(entry, "name")
        matched_terms = self.assert_required_string_list(entry, "matched_terms")
        self.assert_grimace_links(entry, fixture_case_ids)
        self.assert_required_optional_string(entry, "notes", allow_empty=True)
        self.assertIn(entry["status"], COVERAGE_STATUSES)
        self._required_string(entry, "claim")
        self.assertRegex(self._required_string(entry, "snippet_sha256"), SHA256_RE)

        start_line = self._required_int(entry, "start_line")
        end_line = self._required_int(entry, "end_line")
        self.assertLessEqual(start_line, end_line)

        source_path = RDKIT_SERIALIZER_SOURCE_ROOT / rdkit_version / "source" / upstream_file
        source_text = source_path.read_text()
        self.assertLessEqual(end_line, len(source_text.splitlines()))
        line_span = _line_span(source_text, start_line, end_line)
        self.assertTrue(line_span)
        for term in matched_terms:
            self.assertIn(term, line_span)

    def _load_fixture_case_ids(self) -> dict[str, set[str]]:
        fixture_case_ids: dict[str, set[str]] = {}
        for fixture_path in sorted(checked_in_fixture_path().rglob("*.json")):
            relative_path = fixture_path.relative_to(REPO_ROOT).as_posix()
            try:
                payload = json.loads(fixture_path.read_text())
            except json.JSONDecodeError:
                continue
            raw_cases = payload.get("cases")
            if not isinstance(raw_cases, list):
                continue
            case_ids = {
                raw_case["id"]
                for raw_case in raw_cases
                if isinstance(raw_case, dict) and type(raw_case.get("id")) is str
            }
            if case_ids:
                fixture_case_ids[relative_path] = case_ids
        return fixture_case_ids

    def assert_grimace_links(
        self,
        entry: dict[str, object],
        fixture_case_ids: dict[str, set[str]],
    ) -> None:
        links = entry["grimace_links"]
        self.assertIs(type(links), list, "grimace_links")
        if entry["status"] in {COVERAGE_STATUS_COVERED, COVERAGE_STATUS_KNOWN_GAP}:
            self.assertTrue(links, entry["id"])
        elif entry["status"] != COVERAGE_STATUS_NEEDS_FIXTURE:
            self.assertFalse(links, entry["id"])

        seen_fixtures: set[str] = set()
        for link in links:
            self.assertIs(type(link), dict, entry["id"])
            self.assertEqual(GRIMACE_LINK_FIELDS, set(link), entry["id"])
            fixture = self._required_string(link, "fixture")
            self.assertNotIn(fixture, seen_fixtures)
            seen_fixtures.add(fixture)
            self.assertIn(fixture, fixture_case_ids)
            self.assert_required_string_list(link, "cases")
            self.assert_required_optional_string(link, "note", allow_empty=True)
            for case_id in link["cases"]:
                self.assertIn(case_id, fixture_case_ids[fixture], fixture)

    def _entry_sort_key(self, entry: dict[str, object]) -> tuple[object, ...]:
        return (
            entry["upstream_file"],
            entry["start_line"],
            entry["end_line"],
            entry["kind"],
            entry["parent"] or "",
            entry["name"],
        )

    def _required_string(self, data: dict[str, object], field_name: str) -> str:
        value = data[field_name]
        self.assertIs(type(value), str, field_name)
        self.assertTrue(value, field_name)
        return value

    def _required_int(self, data: dict[str, object], field_name: str) -> int:
        value = data[field_name]
        self.assertIs(type(value), int, field_name)
        return value

    def assert_required_optional_string(
        self,
        data: dict[str, object],
        field_name: str,
        *,
        allow_empty: bool = False,
    ) -> None:
        value = data[field_name]
        self.assertTrue(value is None or type(value) is str, field_name)
        if value is not None and not allow_empty:
            self.assertTrue(value, field_name)

    def assert_required_string_list(
        self,
        data: dict[str, object],
        field_name: str,
        *,
        allow_empty: bool = False,
    ) -> list[str]:
        value = data[field_name]
        self.assertIs(type(value), list, field_name)
        if not allow_empty:
            self.assertTrue(value, field_name)
        self.assertTrue(all(type(item) is str and item for item in value), field_name)
        self.assertEqual(len(set(value)), len(value), field_name)
        return value


if __name__ == "__main__":
    unittest.main()
