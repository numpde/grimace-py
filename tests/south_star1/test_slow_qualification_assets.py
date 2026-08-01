from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tests.south_star1 import slow_qualification_assets as cache
from tests.south_star1.default_writer_capability_ledger import (
    ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES,
)
from tests.south_star1.default_writer_qualification_shards import (
    bind_slow_qualification_shard,
    reset_slow_qualification_shard,
)
from tests.south_star1.test_public_continuation_proofs import PublicContinuationProofTest
from tests.south_star1.test_writer_default_continuation_corpus import (
    WriterDefaultContinuationCorpusTest,
)
from tests.south_star1.test_writer_default_stereo_audit_fixture import (
    WriterDefaultStereoAuditSlowTest,
)


class SlowQualificationAssetsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.case = next(
            case
            for case in ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES
            if case.name == "ethanol"
        )

    def test_slow_consumers_require_cache_before_replay(self) -> None:
        token = bind_slow_qualification_shard("remote-a")
        old = os.environ.get("SOUTH_STAR1_RUN_SLOW")
        os.environ["SOUTH_STAR1_RUN_SLOW"] = "1"
        try:
            for consumer in (
                PublicContinuationProofTest().test_slow_coupled_cases_expose_and_verify_every_local_proof,
                WriterDefaultContinuationCorpusTest().test_slow_coupled_cases_cross_all_continuation_tiers,
            ):
                with self.subTest(consumer=consumer):
                    with patch.object(
                        cache,
                        "require_slow_qualification_asset",
                        side_effect=AssertionError("cache required"),
                    ), patch(
                        "tests.south_star1.test_public_continuation_proofs.require_slow_qualification_asset",
                        side_effect=AssertionError("cache required"),
                    ), patch(
                        "tests.south_star1.test_writer_default_continuation_corpus.require_slow_qualification_asset",
                        side_effect=AssertionError("cache required"),
                    ):
                        with self.assertRaisesRegex(AssertionError, "cache required"):
                            consumer()
        finally:
            if old is None:
                os.environ.pop("SOUTH_STAR1_RUN_SLOW", None)
            else:
                os.environ["SOUTH_STAR1_RUN_SLOW"] = old
            reset_slow_qualification_shard(token)

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

    def test_fast_fixture_class_does_not_use_slow_cache(self) -> None:
        self.assertFalse(WriterDefaultStereoAuditSlowTest.__mro__[1].USE_CACHED_SLOW_ASSETS)
        self.assertFalse(
            "require_slow_qualification_asset"
            in inspect.getsource(WriterDefaultContinuationCorpusTest._cross_all_continuation_tiers)
        )

    def test_slow_stereo_class_filters_then_uses_cache(self) -> None:
        self.assertTrue(WriterDefaultStereoAuditSlowTest.USE_CACHED_SLOW_ASSETS)
        source = inspect.getsource(WriterDefaultStereoAuditSlowTest.setUpClass)
        self.assertIn("selected_slow_qualification_cases", source)
        self.assertIn("super().setUpClass()", source)

    def test_only_public_build_calls_the_build_cache_operation(self) -> None:
        from tests.south_star1.test_public_continuation_asset import (
            PublicContinuationAssetTest,
        )

        self.assertIn(
            "build_slow_qualification_asset",
            inspect.getsource(PublicContinuationAssetTest.test_slow_coupled_cases_build_through_public_api),
        )
        self.assertIn(
            "require_slow_qualification_asset",
            inspect.getsource(PublicContinuationProofTest.test_slow_coupled_cases_expose_and_verify_every_local_proof),
        )


if __name__ == "__main__":
    unittest.main()
