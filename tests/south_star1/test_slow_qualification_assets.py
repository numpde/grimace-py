from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from tests.south_star1 import slow_qualification_assets as cache
from tests.south_star1.default_writer_capability_ledger import (
    ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES,
)
from tests.south_star1.qualification_plan import (
    bind_slow_qualification_shard,
    reset_slow_qualification_shard,
)


class SlowQualificationAssetsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.case = next(
            case
            for case in ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES
            if case.name == "ethanol"
        )

    def test_absent_and_mismatched_metadata_fail_before_replay(self) -> None:
        with TemporaryDirectory() as directory:
            os.environ["SOUTH_STAR1_SLOW_ASSET_ROOT"] = directory
            try:
                with self.assertRaisesRegex(AssertionError, "asset is absent"):
                    cache.require_slow_qualification_asset(self.case)
                case_dir = Path(directory) / self.case.name
                (case_dir / "asset").mkdir(parents=True)
                (case_dir / "metadata.json").write_text(json.dumps({}))
                with self.assertRaisesRegex(
                    AssertionError, "metadata mismatch|missing asset manifest digest"
                ):
                    cache.require_slow_qualification_asset(self.case)
            finally:
                os.environ.pop("SOUTH_STAR1_SLOW_ASSET_ROOT", None)

    def test_complete_count_cache_reuses_without_builders(self) -> None:
        with TemporaryDirectory() as directory:
            os.environ["SOUTH_STAR1_SLOW_ASSET_ROOT"] = directory
            try:
                case_dir = Path(directory) / self.case.name
                case_dir.mkdir()
                (case_dir / "count-envelope.json").write_text("{}")
                (case_dir / "count-envelope-metadata.json").write_text("{}")
                cached = cache.CachedQualificationCountEnvelope(
                    self.case,
                    case_dir / "count-envelope.json",
                    case_dir / "count-envelope-metadata.json",
                    "digest",
                )
                with (
                    patch.object(cache, "require_slow_count_envelope", return_value=cached) as require,
                    patch.object(cache, "_checked_writer_frontier_product", side_effect=AssertionError("builder called")),
                    patch.object(cache, "_envelope_from_product", side_effect=AssertionError("DAG builder called")),
                ):
                    self.assertIs(cache.build_slow_count_envelope(self.case), cached)
                require.assert_called_once_with(self.case)
            finally:
                os.environ.pop("SOUTH_STAR1_SLOW_ASSET_ROOT", None)

    def test_new_count_cache_does_not_require_after_publish(self) -> None:
        envelope = {
            "prepared_identity": {"digest": "prepared"},
            "support_count": 1,
            "completion_count": 1,
            "count_dag": {"digest": "dag", "metrics": {"node_count": 1}},
        }
        with TemporaryDirectory() as directory:
            os.environ["SOUTH_STAR1_SLOW_ASSET_ROOT"] = directory
            try:
                with (
                    patch.object(cache, "_prepared_and_snapshot", return_value=(None, SimpleNamespace(cursor=object()))),
                    patch.object(cache, "_checked_writer_frontier_product", return_value=object()),
                    patch.object(cache, "_envelope_from_product", return_value=envelope),
                    patch.object(cache, "require_slow_count_envelope", side_effect=AssertionError("require called after publish")),
                ):
                    cached = cache.build_slow_count_envelope(self.case)
                self.assertTrue(cached.envelope_path.is_file())
                self.assertTrue(cached.metadata_path.is_file())
            finally:
                os.environ.pop("SOUTH_STAR1_SLOW_ASSET_ROOT", None)

    def test_phase_failures_publish_no_metadata(self) -> None:
        from grimace._south_star1 import public_continuation_asset as public_asset

        phases = (
            ("preparation", "prepare_public_continuation_molecule", AssertionError("preparation failed")),
            ("snapshot", "capture_initial_writer_frontier_snapshot", AssertionError("snapshot failed")),
            ("write", "write_writer_continuation_asset", AssertionError("write failed")),
        )
        for name, attribute, error in phases:
            with self.subTest(phase=name), TemporaryDirectory() as directory:
                os.environ["SOUTH_STAR1_SLOW_ASSET_ROOT"] = directory
                try:
                    with ExitStack() as stack:
                        if name in {"snapshot", "write"}:
                            stack.enter_context(
                                patch.object(
                                    public_asset,
                                    "prepare_public_continuation_molecule",
                                    return_value=None,
                                )
                            )
                        if name == "write":
                            stack.enter_context(
                                patch.object(
                                    public_asset,
                                    "capture_initial_writer_frontier_snapshot",
                                    return_value=None,
                                )
                            )
                        stack.enter_context(
                            patch.object(public_asset, attribute, side_effect=error)
                        )
                        with self.assertRaises(AssertionError):
                            cache.build_slow_qualification_asset(self.case)
                    self.assertFalse(
                        (Path(directory) / self.case.name / "metadata.json").exists()
                    )
                finally:
                    os.environ.pop("SOUTH_STAR1_SLOW_ASSET_ROOT", None)

        with TemporaryDirectory() as directory:
            os.environ["SOUTH_STAR1_SLOW_ASSET_ROOT"] = directory
            try:
                with patch.object(
                    cache,
                    "grimace",
                ) as mocked_grimace, patch.object(
                    cache,
                    "verify_writer_continuation_asset_consistency",
                    return_value=SimpleNamespace(accepted=False, reason="write validation failed"),
                ):
                    mocked_grimace.BuildMolToSmilesContinuationAsset.return_value = "digest"
                    with self.assertRaisesRegex(AssertionError, "write validation failed"):
                        cache.build_slow_qualification_asset(self.case)
                self.assertFalse(
                    (Path(directory) / self.case.name / "metadata.json").exists()
                )
            finally:
                os.environ.pop("SOUTH_STAR1_SLOW_ASSET_ROOT", None)


if __name__ == "__main__":
    unittest.main()
