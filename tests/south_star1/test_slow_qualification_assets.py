from __future__ import annotations

import inspect
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
from tests.south_star1.default_writer_qualification_shards import (
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

    def test_slow_consumers_require_cache_before_replay(self) -> None:
        from tests.south_star1.test_public_continuation_proofs import PublicContinuationProofTest
        from tests.south_star1.test_writer_default_continuation_corpus import WriterDefaultContinuationCorpusTest
        self.assertIn(
            "require_slow_qualification_asset",
            inspect.getsource(PublicContinuationProofTest.test_slow_coupled_public_proof_shard_0),
        )
        self.assertIn(
            "require_slow_qualification_asset",
            inspect.getsource(WriterDefaultContinuationCorpusTest._cross_cached_continuation_tiers),
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

    def test_fast_fixture_class_does_not_use_slow_cache(self) -> None:
        from tests.south_star1.test_writer_default_stereo_audit_fixture import WriterDefaultStereoAuditSlowTest
        from tests.south_star1.test_writer_default_continuation_corpus import WriterDefaultContinuationCorpusTest
        self.assertFalse(WriterDefaultStereoAuditSlowTest.__mro__[1].USE_CACHED_SLOW_ASSETS)
        self.assertFalse(
            "require_slow_qualification_asset"
            in inspect.getsource(WriterDefaultContinuationCorpusTest._cross_all_continuation_tiers)
        )

    def test_slow_stereo_class_filters_then_uses_cache(self) -> None:
        from tests.south_star1.test_writer_default_stereo_audit_fixture import WriterDefaultStereoAuditSlowTest
        self.assertTrue(WriterDefaultStereoAuditSlowTest.USE_CACHED_SLOW_ASSETS)
        source = inspect.getsource(WriterDefaultStereoAuditSlowTest.setUpClass)
        self.assertIn("selected_slow_qualification_cases", source)
        self.assertIn("super().setUpClass()", source)

    def test_all_continuation_slow_layers_require_cache(self) -> None:
        from tests.south_star1.test_public_continuation_proofs import PublicContinuationProofTest
        from tests.south_star1.test_writer_default_continuation_corpus import WriterDefaultContinuationCorpusTest
        from tests.south_star1.test_public_continuation_asset import PublicContinuationAssetTest
        from tests.south_star1.test_public_continuation_asset_verification import (
            PublicContinuationAssetVerificationTest,
        )

        self.assertIn(
            "require_slow_qualification_asset",
            inspect.getsource(PublicContinuationAssetTest.test_slow_coupled_cases_run_public_runtime),
        )
        self.assertIn(
            "require_slow_qualification_asset",
            inspect.getsource(PublicContinuationAssetVerificationTest.test_slow_coupled_cases_recertify_copied_assets),
        )
        self.assertIn(
            "require_slow_qualification_asset",
            inspect.getsource(PublicContinuationProofTest.test_slow_coupled_public_proof_shard_0),
        )
        self.assertIn(
            "require_slow_qualification_asset",
            inspect.getsource(WriterDefaultContinuationCorpusTest._cross_cached_continuation_tiers),
        )

    def test_only_public_build_calls_the_build_cache_operation(self) -> None:
        from tests.south_star1.test_public_continuation_proofs import PublicContinuationProofTest
        from tests.south_star1.test_public_continuation_asset import (
            PublicContinuationAssetTest,
        )

        self.assertIn(
            "build_slow_qualification_candidate",
            inspect.getsource(PublicContinuationAssetTest.test_slow_coupled_cases_build_through_public_api),
        )
        self.assertIn(
            "require_slow_qualification_asset",
            inspect.getsource(PublicContinuationProofTest.test_slow_coupled_public_proof_shard_0),
        )

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
