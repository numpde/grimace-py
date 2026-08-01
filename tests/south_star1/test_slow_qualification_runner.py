from __future__ import annotations

import unittest

from tests import run_south_star1_slow as runner
from tests.south_star1.default_writer_qualification_shards import FAST_ACCEPTED_CASES
from tests.south_star1.default_writer_qualification_shards import SLOW_COUPLED_CASES
from tests.south_star1.default_writer_qualification_shards import (
    bind_slow_qualification_shard,
    reset_slow_qualification_shard,
    selected_slow_qualification_cases,
)


def _test_ids(suite: unittest.TestSuite) -> tuple[str, ...]:
    ids = []
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            ids.extend(_test_ids(test))
        else:
            ids.append(test.id())
    return tuple(ids)


class SlowQualificationRunnerTest(unittest.TestCase):
    def test_layers_are_nonempty_and_disjoint(self) -> None:
        layers = runner.SLOW_QUALIFICATION_LAYERS
        self.assertTrue(all(layers.values()))
        ids = [test_id for layer in layers.values() for test_id in layer]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 15)

    def test_all_declared_slow_tests_are_in_one_layer(self) -> None:
        expected = {
            "tests.south_star1.test_public_continuation_asset.PublicContinuationAssetTest.test_slow_coupled_cases_build_through_public_api",
            "tests.south_star1.test_public_continuation_asset.PublicContinuationAssetTest.test_slow_coupled_cases_certify_public_candidates",
            "tests.south_star1.test_public_continuation_asset.PublicContinuationAssetTest.test_slow_coupled_cases_run_public_runtime",
            "tests.south_star1.test_public_continuation_asset_verification.PublicContinuationAssetVerificationTest.test_slow_coupled_cases_recertify_copied_assets",
            "tests.south_star1.test_public_continuation_proofs.PublicContinuationProofTest.test_slow_coupled_cases_expose_and_verify_every_local_proof",
            "tests.south_star1.test_writer_count_dag_envelope.WriterCountDagEnvelopeTest.test_slow_coupled_count_dag_envelope_diagnostics",
            "tests.south_star1.test_writer_default_offline_complete.WriterDefaultOfflineCompleteTest.test_slow_coupled_cases_are_offline_complete",
            "tests.south_star1.test_writer_default_offline_complete.WriterDefaultOfflineCompleteTest.test_zero_h_tetrahedral_is_offline_complete",
            "tests.south_star1.test_writer_default_offline_complete.WriterDefaultOfflineCompleteTest.test_adjacent_specified_tetrahedral_is_offline_complete",
            "tests.south_star1.test_writer_default_parity_corpus.WriterDefaultParityCorpusTest.test_zero_h_tetrahedral_support_artifact",
            "tests.south_star1.test_writer_default_parity_corpus.WriterDefaultParityCorpusTest.test_adjacent_specified_tetrahedral_support_artifact",
            "tests.south_star1.test_writer_default_parity_corpus.WriterDefaultParityCorpusTest.test_slow_coupled_corpus_verifies_support_artifacts",
            "tests.south_star1.test_writer_default_parity_corpus.WriterDefaultParityCorpusTest.test_slow_coupled_corpus_reparses_to_isomorphic_facts",
            "tests.south_star1.test_writer_default_continuation_corpus.WriterDefaultContinuationCorpusTest.test_slow_coupled_cases_cross_all_continuation_tiers",
            "tests.south_star1.test_writer_default_stereo_audit_fixture.WriterDefaultStereoAuditSlowTest",
        }
        actual = {test_id for layer in runner.SLOW_QUALIFICATION_LAYERS.values() for test_id in layer}
        self.assertEqual(actual, expected)

    def test_selected_public_layers_are_case_sharded(self) -> None:
        for shard, layer, test_id, expected_case in (
            (
                "remote-a",
                "public-build",
                runner.SLOW_QUALIFICATION_LAYERS["public-build"][0],
                "remote_coupled_tetrahedral_a",
            ),
            (
                "remote-b",
                "public-proofs",
                runner.SLOW_QUALIFICATION_LAYERS["public-proofs"][0],
                "remote_coupled_tetrahedral_b",
            ),
        ):
            suite, token = runner.load_selected_layer(unittest.defaultTestLoader, shard, layer)
            try:
                self.assertEqual(_test_ids(suite), (test_id,))
                self.assertEqual(
                    tuple(case.name for case in selected_slow_qualification_cases()),
                    (expected_case,),
                )
            finally:
                reset_slow_qualification_shard(token)

    def test_invalid_selection_fails_before_loading_tests(self) -> None:
        for shard, layer in ((None, "public-build"), ("unknown", "public-build"), ("remote-a", None), ("remote-a", "unknown")):
            with self.subTest(shard=shard, layer=layer):
                with self.assertRaises(ValueError):
                    runner.validate_selection(shard, layer)

    def test_fast_cases_never_select_a_slow_case(self) -> None:
        self.assertTrue({case.name for case in FAST_ACCEPTED_CASES}.isdisjoint(
            case.name for case in SLOW_COUPLED_CASES
        ))

    def test_stereo_slow_class_defers_case_selection_until_setup(self) -> None:
        from tests.south_star1.test_writer_default_stereo_audit_fixture import (
            WriterDefaultStereoAuditSlowTest,
        )

        self.assertIsNone(WriterDefaultStereoAuditSlowTest.QUALIFICATION_CASES)
        token = bind_slow_qualification_shard("remote-a")
        try:
            self.assertEqual(
                tuple(case.name for case in selected_slow_qualification_cases()),
                ("remote_coupled_tetrahedral_a",),
            )
        finally:
            reset_slow_qualification_shard(token)


if __name__ == "__main__":
    unittest.main()
