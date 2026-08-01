"""Explicit slow-only South Star qualification runner."""

from __future__ import annotations

import unittest


_SLOW_TESTS = (
    "tests.south_star1.test_public_continuation_asset.PublicContinuationAssetTest.test_slow_coupled_cases_build_through_public_api",
    "tests.south_star1.test_public_continuation_asset_verification.PublicContinuationAssetVerificationTest.test_slow_coupled_cases_recertify_copied_assets",
    "tests.south_star1.test_public_continuation_proofs.PublicContinuationProofTest.test_slow_coupled_cases_expose_and_verify_every_local_proof",
    "tests.south_star1.test_writer_default_offline_complete.WriterDefaultOfflineCompleteTest.test_slow_coupled_cases_are_offline_complete",
    "tests.south_star1.test_writer_default_parity_corpus.WriterDefaultParityCorpusTest.test_slow_coupled_corpus_verifies_support_artifacts",
    "tests.south_star1.test_writer_default_parity_corpus.WriterDefaultParityCorpusTest.test_slow_coupled_corpus_reparses_to_isomorphic_facts",
    "tests.south_star1.test_writer_default_continuation_corpus.WriterDefaultContinuationCorpusTest.test_slow_coupled_cases_cross_all_continuation_tiers",
    "tests.south_star1.test_writer_default_stereo_audit_fixture.WriterDefaultStereoAuditSlowTest",
)


def load_tests(
    loader: unittest.TestLoader,
    standard_tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del standard_tests, pattern
    suite = unittest.TestSuite()
    for name in _SLOW_TESTS:
        suite.addTests(loader.loadTestsFromName(name))
    return suite


if __name__ == "__main__":
    unittest.main()
