"""Continuation-proof authority qualification for coupled writer assets."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import grimace
from rdkit import Chem

from grimace._south_star1 import writer_continuation_asset
from grimace._south_star1 import writer_continuation_rust
from grimace._south_star1 import writer_count_dag_envelope
from grimace._south_star1 import writer_frontier_count_envelope
from grimace._south_star1 import writer_snapshot
from grimace._south_star1 import writer_support
from grimace._south_star1 import writer_support_artifact_envelope
from tests.south_star1.default_writer_qualification_shards import (
    CONTINUATION_PROOF_QUALIFIED_CASES,
    selected_slow_qualification_cases,
)
from tests.south_star1.slow_qualification_assets import require_slow_qualification_asset
from tests.south_star1.test_public_continuation_asset import (
    _decoder_support,
    _support_digest,
)
from tests.south_star1.test_public_continuation_proofs import (
    _verify_all_public_proofs_timed,
)


class ContinuationProofQualificationTest(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("SOUTH_STAR1_RUN_SLOW") == "1",
        "slow lane",
    )
    def test_slow_continuation_proof_complete(self):
        selected = selected_slow_qualification_cases()
        for case in selected:
            self.assertIn(case, CONTINUATION_PROOF_QUALIFIED_CASES)
            with self.subTest(case=case.name):
                cached = require_slow_qualification_asset(case)
                mol = Chem.MolFromSmiles(case.smiles)
                guards = (
                    patch.object(
                        writer_support_artifact_envelope,
                        "writer_support_artifact_envelope_for_snapshot",
                        side_effect=AssertionError("rich support artifact invoked"),
                    ),
                    patch.object(
                        writer_support_artifact_envelope,
                        "_writer_support_artifact_envelope_for_snapshot_with_count_envelope",
                        side_effect=AssertionError("cached rich support artifact invoked"),
                    ),
                    patch.object(
                        writer_frontier_count_envelope,
                        "writer_frontier_count_envelope_for_snapshot",
                        side_effect=AssertionError("count envelope invoked"),
                    ),
                    patch.object(
                        writer_count_dag_envelope,
                        "writer_count_certificate_dag_envelope_for_product",
                        side_effect=AssertionError("count DAG invoked"),
                    ),
                    patch.object(
                        writer_snapshot,
                        "_iter_writer_snapshot_certified_support_strings",
                        side_effect=AssertionError("support strings materialized"),
                    ),
                    patch.object(
                        writer_support,
                        "enumerate_prepared_writer_shaped_support",
                        side_effect=AssertionError("legacy support enumeration invoked"),
                    ),
                )
                with guards[0], guards[1], guards[2], guards[3], guards[4], guards[5]:
                    report = grimace.VerifyMolToSmilesContinuationAsset(
                        mol,
                        cached.asset_path,
                        expected_manifest_digest=cached.manifest_digest,
                    )
                    self.assertTrue(report.accepted)
                    self.assertTrue(report.live_replay_complete)
                    self.assertEqual(report.branch_locator_count, report.branch_proof_count)
                    self.assertEqual(report.terminal_locator_count, report.terminal_proof_count)
                    self.assertEqual(report.unchecked_obligation_families, ())
                    self.assertEqual(
                        report.unchecked_obligation_families,
                        case.expected_continuation_unchecked_obligation_families,
                    )
                    self.assertEqual(report.raw_cursor_count, case.expected_continuation_raw_cursor_count)
                    self.assertEqual(report.edge_locator_count, case.expected_continuation_edge_locator_count)
                    self.assertEqual(report.branch_locator_count, case.expected_continuation_branch_locator_count)
                    self.assertEqual(report.terminal_record_count, case.expected_continuation_terminal_record_count)
                    self.assertEqual(report.terminal_locator_count, case.expected_continuation_terminal_locator_count)
                    self.assertEqual(
                        report.semantically_replayed_operations,
                        case.expected_continuation_replayed_operations,
                    )
                    self.assertEqual(
                        report.checked_relation_families,
                        case.expected_continuation_checked_relation_families,
                    )
                    self.assertEqual(
                        report.checked_obligation_families,
                        case.expected_continuation_checked_obligation_families,
                    )

                    decoder = grimace.MolToSmilesContinuationDecoder.from_asset(
                        cached.asset_path,
                        expected_manifest_digest=cached.manifest_digest,
                        proof_capable=True,
                        mol=mol,
                    )
                    self.assertEqual(decoder.support_count, case.expected_support_count)
                    self.assertEqual(decoder.completion_count, case.expected_completion_count)
                    self.assertEqual(_support_digest(_decoder_support(decoder)), case.expected_support_digest)
                    self.assertEqual(
                        sum(item.numerator for item in decoder.exact_probabilities()),
                        decoder.completion_count,
                    )
                    resumed = grimace.MolToSmilesContinuationDecoder.from_snapshot(
                        cached.asset_path,
                        decoder.next_choices[0].next_state.snapshot(),
                        proof_capable=True,
                        mol=mol,
                    )
                    self.assertEqual(
                        resumed.cache_key(), decoder.next_choices[0].next_state.cache_key()
                    )
                    branch_count, terminal_count, branches, terminals, _timings = (
                        _verify_all_public_proofs_timed(decoder)
                    )
                    self.assertEqual(branch_count, report.branch_proof_count)
                    self.assertEqual(terminal_count, report.terminal_proof_count)
                    self.assertEqual(len(branches), report.branch_locator_count)
                    self.assertEqual(len(terminals), report.terminal_locator_count)

    def test_materialized_and_continuation_authorities_are_disjoint(self):
        from tests.south_star1.default_writer_qualification_shards import (
            MATERIALIZED_ARTIFACT_QUALIFIED_CASES,
        )

        materialized = {case.name for case in MATERIALIZED_ARTIFACT_QUALIFIED_CASES}
        continuation = {case.name for case in CONTINUATION_PROOF_QUALIFIED_CASES}
        self.assertTrue(materialized.isdisjoint(continuation))


if __name__ == "__main__":
    unittest.main()
