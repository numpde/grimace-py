"""Focused tests for the slow qualification cache registry and identity."""

from __future__ import annotations

from pathlib import Path
import ast
import json
from tempfile import TemporaryDirectory
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
    atomic_write_text,
    inspect_qualification_cache,
    publish_json_qualification_cache,
    promote_directory_qualification_cache,
    QualificationCacheState,
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

    def test_pair_states_are_shared(self) -> None:
        with patch.dict("os.environ", {"SOUTH_STAR1_SLOW_ASSET_ROOT": "/tmp/qualification"}), patch(
            "tests.south_star1.qualification_cache._git_head", return_value="head"
        ):
            context = qualification_cache_context(self.case)
        paths = qualification_cache_paths(context, QualificationCacheKind.COUNT_ENVELOPE)
        self.assertEqual(inspect_qualification_cache(paths), QualificationCacheState.ABSENT)

    def test_json_publication_failure_leaves_no_pair(self) -> None:
        with TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"SOUTH_STAR1_SLOW_ASSET_ROOT": directory}
        ), patch("tests.south_star1.qualification_cache._git_head", return_value="head"):
            context = qualification_cache_context(self.case)
            paths = qualification_cache_paths(context, QualificationCacheKind.COUNT_ENVELOPE)
            real_replace = __import__("os").replace
            calls = 0

            def fail_metadata(source, destination):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("metadata publication failed")
                return real_replace(source, destination)

            with patch("tests.south_star1.qualification_cache.os.replace", side_effect=fail_metadata):
                with self.assertRaisesRegex(OSError, "metadata publication failed"):
                    publish_json_qualification_cache(
                        paths,
                        payload={"value": 1},
                        metadata_details={"digest": "x"},
                    )
            self.assertFalse(paths.payload_path.exists())
            self.assertFalse(paths.metadata_path.exists())
            self.assertEqual(tuple(context.case_dir.glob(".*.tmp")), ())

    def test_promotion_failure_restores_candidate_pair(self) -> None:
        with TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"SOUTH_STAR1_SLOW_ASSET_ROOT": directory}
        ), patch("tests.south_star1.qualification_cache._git_head", return_value="head"):
            context = qualification_cache_context(self.case)
            source = qualification_cache_paths(context, QualificationCacheKind.CONTINUATION_CANDIDATE)
            destination = qualification_cache_paths(context, QualificationCacheKind.CONTINUATION_ASSET)
            context.case_dir.mkdir(parents=True)
            source.payload_path.mkdir()
            (source.payload_path / "manifest.json").write_text("{}")
            source.metadata_path.write_text("{}")
            real_replace = __import__("os").replace
            calls = 0

            def fail_destination_metadata(src, dst):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise OSError("promotion metadata failed")
                return real_replace(src, dst)

            with patch("tests.south_star1.qualification_cache.os.replace", side_effect=fail_destination_metadata):
                with self.assertRaisesRegex(OSError, "promotion metadata failed"):
                    promote_directory_qualification_cache(
                        source_paths=source,
                        destination_paths=destination,
                        destination_metadata_details={"digest": "x"},
                    )
            self.assertTrue(source.payload_path.is_dir())
            self.assertTrue(source.metadata_path.is_file())
            self.assertFalse(destination.payload_path.exists())
            self.assertFalse(destination.metadata_path.exists())

    def test_slow_cache_module_has_no_storage_implementation(self) -> None:
        tree = ast.parse(Path(__file__).with_name("slow_qualification_assets.py").read_text())
        forbidden_calls = []
        forbidden_names = {"write_text", "replace", "dumps"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in forbidden_names:
                    forbidden_calls.append((node.func.attr, node.lineno))
            if isinstance(node, ast.ExceptHandler) and node.type is not None:
                if isinstance(node.type, ast.Name) and node.type.id == "Exception":
                    forbidden_calls.append(("except Exception", node.lineno))
        self.assertEqual(forbidden_calls, [])


if __name__ == "__main__":
    unittest.main()
