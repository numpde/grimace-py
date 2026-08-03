"""Focused tests for the slow qualification cache registry and identity."""

from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from tests.south_star1.default_writer_capability_ledger import ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES
from tests.south_star1.qualification_cache import (
    QUALIFICATION_CACHE_ENTRY_DEFINITIONS,
    QualificationCacheKind,
    QualificationCachePayloadKind,
    qualification_cache_common_metadata,
    qualification_cache_context,
    qualification_cache_metadata,
    qualification_cache_paths,
    validate_qualification_cache_registry,
)


class QualificationCacheRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.case = ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES[0]

    def test_registry_has_exactly_four_entries(self) -> None:
        validate_qualification_cache_registry()
        self.assertEqual(len(QUALIFICATION_CACHE_ENTRY_DEFINITIONS), 4)
        self.assertEqual(
            {item.kind for item in QUALIFICATION_CACHE_ENTRY_DEFINITIONS},
            set(QualificationCacheKind),
        )

    def test_registry_paths_and_payload_kinds_are_authoritative(self) -> None:
        expected = {
            QualificationCacheKind.CONTINUATION_CANDIDATE: ("candidate", "candidate-metadata.json", QualificationCachePayloadKind.DIRECTORY),
            QualificationCacheKind.CONTINUATION_ASSET: ("asset", "metadata.json", QualificationCachePayloadKind.DIRECTORY),
            QualificationCacheKind.COUNT_ENVELOPE: ("count-envelope.json", "count-envelope-metadata.json", QualificationCachePayloadKind.JSON),
            QualificationCacheKind.SUPPORT_ARTIFACT: ("support-artifact.json", "support-artifact-metadata.json", QualificationCachePayloadKind.JSON),
        }
        for definition in QUALIFICATION_CACHE_ENTRY_DEFINITIONS:
            self.assertEqual(
                (definition.payload_name, definition.metadata_name, definition.payload_kind),
                expected[definition.kind],
            )

    def test_context_reads_head_and_root_once(self) -> None:
        with patch.dict("os.environ", {"SOUTH_STAR1_SLOW_ASSET_ROOT": "/tmp/qualification"}), patch(
            "tests.south_star1.qualification_cache._git_head", return_value="head"
        ) as read_head:
            context = qualification_cache_context(self.case)
        self.assertEqual(context.root, Path("/tmp/qualification"))
        self.assertEqual(context.case_dir, Path("/tmp/qualification") / self.case.name)
        self.assertEqual(context.git_head, "head")
        read_head.assert_called_once_with()

    def test_paths_come_from_context_and_kind(self) -> None:
        with patch.dict("os.environ", {"SOUTH_STAR1_SLOW_ASSET_ROOT": "/tmp/qualification"}), patch(
            "tests.south_star1.qualification_cache._git_head", return_value="head"
        ):
            context = qualification_cache_context(self.case)
        paths = qualification_cache_paths(context, QualificationCacheKind.COUNT_ENVELOPE)
        self.assertEqual(paths.payload_path, context.case_dir / "count-envelope.json")
        self.assertEqual(paths.metadata_path, context.case_dir / "count-envelope-metadata.json")

    def test_common_and_entry_metadata_have_one_identity(self) -> None:
        with patch.dict("os.environ", {"SOUTH_STAR1_SLOW_ASSET_ROOT": "/tmp/qualification"}), patch(
            "tests.south_star1.qualification_cache._git_head", return_value="head"
        ):
            context = qualification_cache_context(self.case)
        common = qualification_cache_common_metadata(context)
        metadata = qualification_cache_metadata(
            context,
            QualificationCacheKind.COUNT_ENVELOPE,
            details={"count_dag_digest": "digest"},
        )
        self.assertEqual({key: metadata[key] for key in common}, common)
        self.assertEqual(metadata["schema"], "south_star1_slow_count_envelope")
        self.assertEqual(metadata["count_dag_digest"], "digest")


if __name__ == "__main__":
    unittest.main()
