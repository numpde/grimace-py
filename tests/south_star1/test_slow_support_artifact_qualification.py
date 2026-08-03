"""Case-sharded slow qualification for cached rich support artifacts."""

from __future__ import annotations

import json
import os
from hashlib import sha256
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from grimace._south_star1 import writer_frontier_count_envelope
from grimace._south_star1 import writer_support_artifact_envelope
from grimace._south_star1.writer_envelope_work import WriterEnvelopeWorkBudget
from grimace._south_star1.writer_support_artifact_checker import (
    verify_writer_support_artifact_consistency,
)
from grimace._south_star1.writer_support_artifact_envelope import (
    writer_support_artifact_envelope_for_snapshot,
    verify_writer_support_artifact_envelope,
)
from grimace._south_star1.writer_support_artifact_fact_verifier import (
    verify_writer_support_artifact_for_facts,
)
from tests.south_star1.qualification_plan import (
    case_by_name,
    selected_slow_qualification_cases,
)
from tests.south_star1.slow_qualification_assets import (
    build_slow_count_envelope,
    build_slow_support_artifact,
    require_slow_qualification_asset,
    require_slow_count_envelope,
    require_slow_support_artifact,
    _prepared_and_snapshot,
)
from tests.south_star1.qualification_support import facts_for_case
from tests.south_star1.qualification_support import prepare_default_case
from tests.south_star1.qualification_support import runtime_options_for_root
from tests.south_star1.qualification_guards import forbid_qualification_profile
from tests.south_star1.qualification_assertions import assert_offline_case_matches_ledger


class SlowSupportArtifactQualificationTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("SOUTH_STAR1_RUN_SLOW") == "1", "slow lane")
    def test_slow_count_dag_cache_is_present(self):
        for case in selected_slow_qualification_cases():
            with self.subTest(case=case.name):
                cached = require_slow_count_envelope(case)
                self.assertTrue(cached.envelope_path.is_file())

    @unittest.skipUnless(os.environ.get("SOUTH_STAR1_RUN_SLOW") == "1", "slow lane")
    def test_slow_support_artifact_build(self):
        for case in selected_slow_qualification_cases():
            with self.subTest(case=case.name):
                with forbid_qualification_profile("slow-rich-artifact-build") as guard_report:
                    cached = build_slow_support_artifact(case)
                guard_report.assert_unused(self)
                self.assertEqual(cached.support_count, case.expected_support_count)
                self.assertEqual(cached.completion_count, case.expected_completion_count)
                self.assertTrue(cached.artifact_path.is_file())
                self.assertTrue(cached.metadata_path.is_file())

    @unittest.skipUnless(os.environ.get("SOUTH_STAR1_RUN_SLOW") == "1", "slow lane")
    def test_slow_support_artifact_live(self):
        for case in selected_slow_qualification_cases():
            with self.subTest(case=case.name):
                cached = require_slow_support_artifact(case)
                artifact = json.loads(cached.artifact_path.read_text())
                prepared, snapshot = _prepared_and_snapshot(case)
                started = time.monotonic()
                with forbid_qualification_profile("slow-rich-artifact-live") as guard_report:
                    result = verify_writer_support_artifact_envelope(
                        prepared=prepared, envelope=artifact, budget=WriterEnvelopeWorkBudget()
                    )
                guard_report.assert_unused(self)
                print(f"artifact_cache_validation_seconds=0.000", flush=True)
                print(f"live_verification_seconds={time.monotonic() - started:.3f}", flush=True)
                self.assertTrue(result.accepted, result.reason)
                self.assertEqual(result.support_count, case.expected_support_count)
                self.assertEqual(result.witness_count, case.expected_completion_count)

    @unittest.skipUnless(os.environ.get("SOUTH_STAR1_RUN_SLOW") == "1", "slow lane")
    def test_slow_support_artifact_offline_complete(self):
        for case in selected_slow_qualification_cases():
            with self.subTest(case=case.name):
                cached = require_slow_support_artifact(case)
                artifact = json.loads(cached.artifact_path.read_text())
                facts = facts_for_case(case)
                started = time.monotonic()
                with forbid_qualification_profile("slow-rich-artifact-live") as guard_report:
                    result = verify_writer_support_artifact_for_facts(
                        facts=facts,
                        runtime_options=runtime_options_for_root(case.rooted_at_atom),
                        artifact=artifact,
                    )
                guard_report.assert_unused(self)
                print("artifact_cache_validation_seconds=0.000", flush=True)
                print(f"facts_bound_replay_seconds={time.monotonic() - started:.3f}", flush=True)
                self.assertTrue(result.accepted, result.reason)
                assert_offline_case_matches_ledger(self, case=case, verification=result)

    def test_cached_count_composition_is_producer_free_for_small_case(self):
        case = case_by_name("ethanol")
        with TemporaryDirectory() as directory:
            previous = os.environ.get("SOUTH_STAR1_SLOW_ASSET_ROOT")
            os.environ["SOUTH_STAR1_SLOW_ASSET_ROOT"] = directory
            try:
                build_slow_count_envelope(case)
                with forbid_qualification_profile("slow-rich-artifact-build") as guard_report:
                    cached = build_slow_support_artifact(case)
                    artifact = json.loads(cached.artifact_path.read_text())
                    structural = verify_writer_support_artifact_consistency(artifact)
                    prepared, _snapshot = _prepared_and_snapshot(case)
                    live = verify_writer_support_artifact_envelope(
                        prepared=prepared, envelope=artifact
                    )
                    facts = facts_for_case(case)
                    offline = verify_writer_support_artifact_for_facts(
                        facts=facts,
                        runtime_options=runtime_options_for_root(case.rooted_at_atom),
                        artifact=artifact,
                    )
                guard_report.assert_unused(self)
                self.assertTrue(structural.accepted, structural.reason)
                self.assertTrue(live.accepted, live.reason)
                self.assertTrue(offline.accepted, offline.reason)
                self.assertEqual(guard_report.call_counts()["count_envelope_build"], 0)
                self.assertEqual(guard_report.call_counts()["count_dag_build"], 0)
            finally:
                if previous is None:
                    os.environ.pop("SOUTH_STAR1_SLOW_ASSET_ROOT", None)
                else:
                    os.environ["SOUTH_STAR1_SLOW_ASSET_ROOT"] = previous

    def test_fresh_support_artifact_build_is_single_pass_for_small_case(self):
        case = case_by_name("ethanol")
        import tests.south_star1.slow_qualification_assets as cache

        with TemporaryDirectory() as directory:
            previous = os.environ.get("SOUTH_STAR1_SLOW_ASSET_ROOT")
            os.environ["SOUTH_STAR1_SLOW_ASSET_ROOT"] = directory
            try:
                build_slow_count_envelope(case)
                with forbid_qualification_profile("slow-rich-artifact-build") as guard_report:
                    with (
                    patch.object(cache, "require_slow_count_envelope", side_effect=AssertionError("count cache reread")),
                    patch.object(cache, "require_slow_support_artifact", side_effect=AssertionError("artifact reread")),
                    patch.object(cache, "_prepared_and_snapshot", wraps=cache._prepared_and_snapshot) as prepared,
                    patch.object(cache, "_checked_writer_frontier_product", wraps=cache._checked_writer_frontier_product) as product,
                    patch.object(cache, "_verify_writer_frontier_count_envelope_against_product", wraps=cache._verify_writer_frontier_count_envelope_against_product) as binding,
                    patch.object(cache, "_support_image_certificate_for_source", wraps=cache._support_image_certificate_for_source) as image,
                    patch.object(cache, "_artifact_from_image", wraps=cache._artifact_from_image) as assembly,
                    ):
                        cached = cache.build_slow_support_artifact(case)
                guard_report.assert_unused(self)
                self.assertFalse(cached.cache_reused)
                self.assertEqual(prepared.call_count, 1)
                self.assertEqual(product.call_count, 1)
                self.assertEqual(binding.call_count, 1)
                self.assertEqual(image.call_count, 1)
                self.assertEqual(assembly.call_count, 1)
            finally:
                if previous is None:
                    os.environ.pop("SOUTH_STAR1_SLOW_ASSET_ROOT", None)
                else:
                    os.environ["SOUTH_STAR1_SLOW_ASSET_ROOT"] = previous

    def test_single_pass_artifact_matches_public_builder_for_small_case(self):
        case = case_by_name("ethanol")

        with TemporaryDirectory() as directory:
            previous = os.environ.get("SOUTH_STAR1_SLOW_ASSET_ROOT")
            os.environ["SOUTH_STAR1_SLOW_ASSET_ROOT"] = directory
            try:
                build_slow_count_envelope(case)
                cached = build_slow_support_artifact(case)
                artifact = json.loads(cached.artifact_path.read_text())
                prepared, snapshot = _prepared_and_snapshot(case)
                expected = writer_support_artifact_envelope_for_snapshot(
                    prepared=prepared,
                    snapshot=snapshot,
                    budget=WriterEnvelopeWorkBudget(),
                )
                self.assertEqual(
                    json.dumps(artifact, sort_keys=True, separators=(",", ":")),
                    json.dumps(expected, sort_keys=True, separators=(",", ":")),
                )
            finally:
                if previous is None:
                    os.environ.pop("SOUTH_STAR1_SLOW_ASSET_ROOT", None)
                else:
                    os.environ["SOUTH_STAR1_SLOW_ASSET_ROOT"] = previous
