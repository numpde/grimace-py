from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from tests.helpers.fixture_paths import checked_in_fixture_path


FIXTURE_FAMILY_ROOT = checked_in_fixture_path("rdkit_upstream_serializer_sources")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RdkitUpstreamSerializerSourceFixtureTest(unittest.TestCase):
    def test_manifest_matches_copied_source_files(self) -> None:
        fixture_roots = tuple(
            path
            for path in sorted(FIXTURE_FAMILY_ROOT.iterdir())
            if path.is_dir()
        )
        self.assertTrue(fixture_roots)

        for fixture_root in fixture_roots:
            with self.subTest(rdkit_version=fixture_root.name):
                self._assert_manifest_matches_copied_source_files(fixture_root)

    def _assert_manifest_matches_copied_source_files(self, fixture_root: Path) -> None:
        manifest_path = fixture_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())

        self.assertEqual(fixture_root.name, manifest["rdkit_version"])
        self.assert_nonempty_string(manifest, "source_commit", manifest_path)
        self.assertEqual("BSD-3-Clause", manifest["license"]["spdx"])

        license_path = fixture_root / manifest["license"]["path"]
        self.assertTrue(license_path.is_file())
        self.assertEqual(manifest["license"]["sha256"], _sha256(license_path))

        seen_paths: set[str] = set()
        source_root = fixture_root / "source"
        for entry in manifest["files"]:
            with self.subTest(path=entry["path"]):
                rel_path = Path(entry["path"])
                self.assertFalse(rel_path.is_absolute())
                self.assertNotIn("..", rel_path.parts)
                self.assertNotIn(entry["path"], seen_paths)
                seen_paths.add(entry["path"])

                source_path = source_root / rel_path
                data = source_path.read_bytes()

                self.assertEqual(entry["bytes"], len(data))
                self.assertEqual(entry["lines"], data.count(b"\n"))
                self.assertEqual(entry["sha256"], hashlib.sha256(data).hexdigest())

        actual_paths = {
            path.relative_to(source_root).as_posix()
            for path in source_root.rglob("*")
            if path.is_file()
        }
        self.assertEqual(actual_paths, seen_paths)

    def assert_nonempty_string(
        self,
        data: dict[str, object],
        field_name: str,
        fixture_path: Path,
    ) -> None:
        value = data.get(field_name)
        self.assertIs(type(value), str, f"{fixture_path} {field_name}")
        self.assertTrue(value, f"{fixture_path} {field_name}")


if __name__ == "__main__":
    unittest.main()
