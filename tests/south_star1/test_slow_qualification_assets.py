from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tests.south_star1 import slow_qualification_assets as cache
from tests.south_star1.default_writer_capability_ledger import ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES
from tests.south_star1.qualification_cache import (
    QualificationCacheKind,
    QualificationCacheState,
    inspect_qualification_cache,
    qualification_cache_context,
    qualification_cache_paths,
)
from tests.south_star1.default_writer_capability_ledger import default_writer_capability_case


class SlowQualificationAssetsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.case = default_writer_capability_case("ethanol")

    def test_absent_and_mismatched_metadata_fail_before_replay(self) -> None:
        with TemporaryDirectory() as directory, patch.dict(
            os.environ, {"SOUTH_STAR1_SLOW_ASSET_ROOT": directory}
        ):
            with self.assertRaisesRegex(AssertionError, "asset"):
                cache.require_slow_qualification_asset(self.case)

    def test_all_entry_states_are_shared(self) -> None:
        with TemporaryDirectory() as directory, patch.dict(
            os.environ, {"SOUTH_STAR1_SLOW_ASSET_ROOT": directory}
        ):
            context = qualification_cache_context(self.case)
            for kind in QualificationCacheKind:
                paths = qualification_cache_paths(context, kind)
                self.assertEqual(inspect_qualification_cache(paths), QualificationCacheState.ABSENT)
                paths.context.case_dir.mkdir(parents=True, exist_ok=True)
                if paths.definition.payload_kind.value == "directory":
                    paths.payload_path.mkdir()
                else:
                    paths.payload_path.write_text("{}")
                self.assertEqual(inspect_qualification_cache(paths), QualificationCacheState.PAYLOAD_ONLY)
                paths.metadata_path.write_text("{}")
                self.assertEqual(inspect_qualification_cache(paths), QualificationCacheState.COMPLETE)
                if paths.payload_path.is_dir():
                    paths.payload_path.rmdir()
                else:
                    paths.payload_path.unlink()
                paths.metadata_path.unlink()

    def test_new_count_cache_does_not_require_after_publish(self) -> None:
        envelope = {
            "prepared_identity": {"digest": "prepared"},
            "support_count": 1,
            "completion_count": 1,
            "count_dag": {"digest": "dag", "metrics": {"node_count": 1}},
        }
        with TemporaryDirectory() as directory, patch.dict(
            os.environ, {"SOUTH_STAR1_SLOW_ASSET_ROOT": directory}
        ):
            with (
                patch.object(cache, "_prepared_and_snapshot", return_value=(None, type("S", (), {"cursor": object()})())),
                patch.object(cache, "_checked_writer_frontier_product", return_value=object()),
                patch.object(cache, "_envelope_from_product", return_value=envelope),
                patch.object(cache, "require_slow_count_envelope", side_effect=AssertionError("require called after publish")),
            ):
                cached = cache.build_slow_count_envelope(self.case)
            self.assertTrue(cached.entry.paths.payload_path.is_file())
            self.assertTrue(cached.entry.paths.metadata_path.is_file())
            self.assertEqual(cached.envelope, envelope)

    def test_candidate_materialization_failure_leaves_no_public_pair(self) -> None:
        with TemporaryDirectory() as directory, patch.dict(
            os.environ, {"SOUTH_STAR1_SLOW_ASSET_ROOT": directory}
        ), patch.object(
            cache, "_prepared_and_snapshot", side_effect=AssertionError("preparation failed")
        ):
            with self.assertRaisesRegex(AssertionError, "preparation failed"):
                cache.build_slow_qualification_candidate(self.case)
            case_dir = Path(directory) / self.case.name
            self.assertFalse((case_dir / "candidate").exists())
            self.assertFalse((case_dir / "candidate-metadata.json").exists())

    def test_complete_invalid_candidate_is_not_rebuilt(self) -> None:
        with TemporaryDirectory() as directory, patch.dict(
            os.environ, {"SOUTH_STAR1_SLOW_ASSET_ROOT": directory}
        ), patch.object(
            cache, "_prepared_and_snapshot", side_effect=AssertionError("producer called")
        ):
            case_dir = Path(directory) / self.case.name
            (case_dir / "candidate").mkdir(parents=True)
            (case_dir / "candidate-metadata.json").write_text("{}")
            with self.assertRaisesRegex(AssertionError, "metadata mismatch"):
                cache.build_slow_qualification_candidate(self.case)

    def test_complete_invalid_asset_is_not_rebuilt(self) -> None:
        with TemporaryDirectory() as directory, patch.dict(
            os.environ, {"SOUTH_STAR1_SLOW_ASSET_ROOT": directory}
        ), patch.object(
            cache, "verify_writer_continuation_asset_consistency", side_effect=AssertionError("producer called")
        ):
            case_dir = Path(directory) / self.case.name
            (case_dir / "asset").mkdir(parents=True)
            (case_dir / "metadata.json").write_text("{}")
            with self.assertRaisesRegex(AssertionError, "metadata mismatch"):
                cache.require_slow_qualification_asset(self.case)


if __name__ == "__main__":
    unittest.main()
