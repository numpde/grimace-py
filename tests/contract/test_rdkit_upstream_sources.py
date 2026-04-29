from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "rdkit_upstream_serializer_sources"
    / "2026.03.1"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RdkitUpstreamSerializerSourceFixtureTest(unittest.TestCase):
    def test_manifest_matches_copied_source_files(self) -> None:
        manifest_path = FIXTURE_ROOT / "manifest.json"
        manifest = json.loads(manifest_path.read_text())

        self.assertEqual("2026.03.1", manifest["rdkit_version"])
        self.assertEqual(
            "cb251343b9448601963cbcec5057a76ff02b27c2",
            manifest["source_commit"],
        )
        self.assertEqual("BSD-3-Clause", manifest["license"]["spdx"])

        license_path = FIXTURE_ROOT / manifest["license"]["path"]
        self.assertTrue(license_path.is_file())
        self.assertEqual(manifest["license"]["sha256"], _sha256(license_path))

        seen_paths: set[str] = set()
        for entry in manifest["files"]:
            with self.subTest(path=entry["path"]):
                rel_path = Path(entry["path"])
                self.assertFalse(rel_path.is_absolute())
                self.assertNotIn("..", rel_path.parts)
                self.assertNotIn(entry["path"], seen_paths)
                seen_paths.add(entry["path"])

                source_path = FIXTURE_ROOT / "source" / rel_path
                data = source_path.read_bytes()

                self.assertEqual(entry["bytes"], len(data))
                self.assertEqual(entry["lines"], data.count(b"\n"))
                self.assertEqual(entry["sha256"], hashlib.sha256(data).hexdigest())

        self.assertEqual(
            {
                "Code/GraphMol/SmilesParse/SmilesWrite.cpp",
                "Code/GraphMol/SmilesParse/SmilesWrite.h",
                "Code/GraphMol/SmilesParse/catch_tests.cpp",
                "Code/GraphMol/SmilesParse/cxsmiles_test.cpp",
                "Code/GraphMol/Wrap/rough_test.py",
                "Code/JavaWrappers/gmwrapper/src-test/org/RDKit/SmilesDetailsTests.java",
            },
            seen_paths,
        )


if __name__ == "__main__":
    unittest.main()
