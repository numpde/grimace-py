"""Explicit, case-sharded South Star slow qualification runner."""

from __future__ import annotations

import os
import sys
import time
import unittest

from tests.south_star1.default_writer_qualification_shards import (
    SLOW_QUALIFICATION_SHARDS,
    bind_slow_qualification_shard,
    reset_slow_qualification_shard,
    slow_cases_for_shard,
)


SLOW_QUALIFICATION_LAYERS = {
    "public-build": (
        "tests.south_star1.test_public_continuation_asset.PublicContinuationAssetTest.test_slow_coupled_cases_build_through_public_api",
    ),
    "public-certify": (
        "tests.south_star1.test_public_continuation_asset.PublicContinuationAssetTest.test_slow_coupled_cases_certify_public_candidates",
    ),
    "public-runtime": (
        "tests.south_star1.test_public_continuation_asset.PublicContinuationAssetTest.test_slow_coupled_cases_run_public_runtime",
    ),
    "public-recertification": (
        "tests.south_star1.test_public_continuation_asset_verification.PublicContinuationAssetVerificationTest.test_slow_coupled_cases_recertify_copied_assets",
    ),
    "public-proofs": (
        "tests.south_star1.test_public_continuation_proofs.PublicContinuationProofTest.test_slow_coupled_cases_expose_and_verify_every_local_proof",
    ),
    "count-dag-build": (
        "tests.south_star1.test_writer_count_dag_envelope.WriterCountDagEnvelopeTest.test_slow_coupled_count_dag_build",
    ),
    "count-dag-validate": (
        "tests.south_star1.test_writer_count_dag_envelope.WriterCountDagEnvelopeTest.test_slow_coupled_count_dag_validate",
    ),
    "offline-zero-h": (
        "tests.south_star1.test_writer_default_offline_complete.WriterDefaultOfflineCompleteTest.test_zero_h_tetrahedral_is_offline_complete",
    ),
    "offline-adjacent": (
        "tests.south_star1.test_writer_default_offline_complete.WriterDefaultOfflineCompleteTest.test_adjacent_specified_tetrahedral_is_offline_complete",
    ),
    "support-artifact-build": (
        "tests.south_star1.test_slow_support_artifact_qualification.SlowSupportArtifactQualificationTest.test_slow_support_artifact_build",
    ),
    "support-artifact-live": (
        "tests.south_star1.test_slow_support_artifact_qualification.SlowSupportArtifactQualificationTest.test_slow_support_artifact_live",
    ),
    "offline-complete": (
        "tests.south_star1.test_slow_support_artifact_qualification.SlowSupportArtifactQualificationTest.test_slow_support_artifact_offline_complete",
    ),
    "support-zero-h": (
        "tests.south_star1.test_writer_default_parity_corpus.WriterDefaultParityCorpusTest.test_zero_h_tetrahedral_support_artifact",
    ),
    "support-adjacent": (
        "tests.south_star1.test_writer_default_parity_corpus.WriterDefaultParityCorpusTest.test_adjacent_specified_tetrahedral_support_artifact",
    ),
    "support-reparse": (
        "tests.south_star1.test_writer_default_parity_corpus.WriterDefaultParityCorpusTest.test_slow_coupled_corpus_reparses_to_isomorphic_facts",
    ),
    "continuation": (
        "tests.south_star1.test_writer_default_continuation_corpus.WriterDefaultContinuationCorpusTest.test_slow_coupled_cases_cross_all_continuation_tiers",
    ),
    "stereo-audit": (
        "tests.south_star1.test_writer_default_stereo_audit_fixture.WriterDefaultStereoAuditSlowTest",
    ),
}


def validate_selection(shard: str | None, layer: str | None) -> None:
    if not shard or shard not in SLOW_QUALIFICATION_SHARDS:
        raise ValueError(f"unknown slow qualification shard: {shard!r}")
    if not layer or layer not in SLOW_QUALIFICATION_LAYERS:
        raise ValueError(f"unknown slow qualification layer: {layer!r}")


def load_selected_layer(
    loader: unittest.TestLoader, shard: str, layer: str
) -> tuple[unittest.TestSuite, object]:
    validate_selection(shard, layer)
    token = bind_slow_qualification_shard(shard)
    suite = unittest.TestSuite()
    for test_id in SLOW_QUALIFICATION_LAYERS[layer]:
        suite.addTests(loader.loadTestsFromName(test_id))
    return suite, token


def main() -> int:
    shard = os.environ.get("SOUTH_STAR1_SLOW_SHARD")
    layer = os.environ.get("SOUTH_STAR1_SLOW_LAYER")
    try:
        validate_selection(shard, layer)
        cases = slow_cases_for_shard(shard)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    print("south-star1 slow qualification:")
    print(f"  shard={shard}")
    print(f"  cases={','.join(case.name for case in cases)}")
    print(f"  layer={layer}")
    started = time.monotonic()
    suite, token = load_selected_layer(unittest.defaultTestLoader, shard, layer)
    try:
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    finally:
        reset_slow_qualification_shard(token)
    elapsed = time.monotonic() - started
    print(f"elapsed_seconds={elapsed:.3f}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
