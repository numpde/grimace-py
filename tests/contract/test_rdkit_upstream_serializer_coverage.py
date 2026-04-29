from __future__ import annotations

import json
from pathlib import Path
import re
import unittest

from tests.helpers.fixture_paths import CHECKED_IN_FIXTURE_ROOT, checked_in_fixture_path


COVERAGE_ROOT = checked_in_fixture_path("rdkit_upstream_serializer_coverage")
SOURCE_ROOT = checked_in_fixture_path("rdkit_upstream_serializer_sources")
REPO_ROOT = CHECKED_IN_FIXTURE_ROOT.parents[1]
COVERAGE_FIELDS = {
    "entries",
    "extractor_version",
    "rdkit_version",
    "source_commit",
    "source_manifest",
}
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
REVIEW_ENTRY_FIELDS = {
    "status",
    "claim",
    "grimace_fixtures",
    "grimace_cases",
    "notes",
}
ENTRY_FIELDS = GENERATED_ENTRY_FIELDS | REVIEW_ENTRY_FIELDS
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
VALID_REVIEW_STATUSES = {
    "covered",
    "needs-fixture",
    "out-of-scope",
    "unreviewed",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _line_span(text: str, start_line: int, end_line: int) -> str:
    lines = text.splitlines(keepends=True)
    return "".join(lines[start_line - 1:end_line])


class RdkitUpstreamSerializerCoverageFixtureTest(unittest.TestCase):
    def test_coverage_inventory_matches_upstream_source_snippets(self) -> None:
        fixture_paths = tuple(sorted(COVERAGE_ROOT.glob("*.json")))
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

        source_manifest_path = SOURCE_ROOT / rdkit_version / "manifest.json"
        self.assertTrue(source_manifest_path.is_file())
        self.assertEqual(
            source_manifest_path.relative_to(REPO_ROOT).as_posix(),
            coverage["source_manifest"],
        )
        source_manifest = json.loads(source_manifest_path.read_text())
        self.assertEqual(source_manifest["source_commit"], coverage["source_commit"])
        manifest_paths = {entry["path"] for entry in source_manifest["files"]}

        entries = coverage["entries"]
        self.assertIs(type(entries), list)
        self.assertTrue(entries)

        seen_ids: set[str] = set()
        previous_sort_key: tuple[object, ...] | None = None
        for entry in entries:
            with self.subTest(entry_id=entry.get("id")):
                self._assert_entry(entry, seen_ids, manifest_paths, rdkit_version)
                sort_key = self._entry_sort_key(entry)
                if previous_sort_key is not None:
                    self.assertLessEqual(previous_sort_key, sort_key)
                previous_sort_key = sort_key

    def _assert_entry(
        self,
        entry: dict[str, object],
        seen_ids: set[str],
        manifest_paths: set[str],
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
        self.assert_required_string_list(entry, "grimace_fixtures", allow_empty=True)
        self.assert_required_string_list(entry, "grimace_cases", allow_empty=True)
        self.assert_required_optional_string(entry, "notes", allow_empty=True)
        self.assertIn(entry["status"], VALID_REVIEW_STATUSES)
        self._required_string(entry, "claim")
        self.assertRegex(self._required_string(entry, "snippet_sha256"), SHA256_RE)

        start_line = self._required_int(entry, "start_line")
        end_line = self._required_int(entry, "end_line")
        self.assertLessEqual(start_line, end_line)

        source_path = SOURCE_ROOT / rdkit_version / "source" / upstream_file
        source_text = source_path.read_text()
        self.assertLessEqual(end_line, len(source_text.splitlines()))
        line_span = _line_span(source_text, start_line, end_line)
        self.assertTrue(line_span)
        for term in matched_terms:
            self.assertIn(term, line_span)

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
