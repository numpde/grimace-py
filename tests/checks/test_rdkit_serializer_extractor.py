from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "extract_rdkit_serializer_cases.py"


def load_extractor_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "extract_rdkit_serializer_cases",
        SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load extractor module spec for {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXTRACTOR = load_extractor_module()


class RdkitSerializerExtractorTests(unittest.TestCase):
    def test_source_manifest_rejects_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_root = Path(tmpdir)
            source_root.joinpath("manifest.json").write_text("{", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "source manifest"):
                EXTRACTOR._load_source_manifest(source_root)

    def test_source_manifest_rejects_unsafe_file_paths(self) -> None:
        cases = (
            "/absolute.cpp",
            "../escape.cpp",
            "dir/./alias.cpp",
            r"dir\..\x.cpp",
            "C:/escape.cpp",
        )
        for rel_path in cases:
            with self.subTest(rel_path=rel_path):
                with self.assertRaisesRegex(ValueError, "unsafe file path"):
                    EXTRACTOR._source_files_from_manifest(
                        {"files": [{"path": rel_path}]},
                    )

    def test_existing_reviews_reject_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "coverage.json"
            output_path.write_text(
                json.dumps(
                    {
                        "entries": [
                            {"id": "same", "status": "covered"},
                            {"id": "same", "status": "known-gap"},
                        ],
                    },
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate ledger id"):
                EXTRACTOR._load_existing_reviews(output_path)

    def test_existing_reviews_preserve_review_fields_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "coverage.json"
            output_path.write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "id": "entry",
                                "status": "covered",
                                "claim": "covered",
                                "generated_field": "not preserved",
                            },
                        ],
                    },
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                {"entry": {"status": "covered", "claim": "covered"}},
                EXTRACTOR._load_existing_reviews(output_path),
            )


if __name__ == "__main__":
    unittest.main()
