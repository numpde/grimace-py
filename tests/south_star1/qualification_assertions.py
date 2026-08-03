"""Shared assertions for South Star qualification contracts."""

from __future__ import annotations


def assert_continuation_recertification_matches_case(test, *, case, report) -> None:
    test.assertTrue(report.accepted)
    test.assertTrue(report.live_replay_complete)
    test.assertEqual(report.branch_locator_count, report.branch_proof_count)
    test.assertEqual(report.terminal_locator_count, report.terminal_proof_count)
    test.assertEqual(report.unchecked_obligation_families, ())
    if case.qualification_authority != "continuation_proof_complete":
        return
    test.assertEqual(report.raw_cursor_count, case.expected_continuation_raw_cursor_count)
    test.assertEqual(report.edge_locator_count, case.expected_continuation_edge_locator_count)
    test.assertEqual(report.branch_locator_count, case.expected_continuation_branch_locator_count)
    test.assertEqual(report.branch_proof_count, case.expected_continuation_branch_locator_count)
    test.assertEqual(report.terminal_record_count, case.expected_continuation_terminal_record_count)
    test.assertEqual(report.terminal_locator_count, case.expected_continuation_terminal_locator_count)
    test.assertEqual(report.terminal_proof_count, case.expected_continuation_terminal_locator_count)
    test.assertEqual(report.semantically_replayed_operations, case.expected_continuation_replayed_operations)
    test.assertEqual(report.checked_relation_families, case.expected_continuation_checked_relation_families)
    test.assertEqual(report.checked_obligation_families, case.expected_continuation_checked_obligation_families)
    test.assertEqual(report.unchecked_obligation_families, case.expected_continuation_unchecked_obligation_families)


def assert_materialized_case_matches_ledger(test, *, case, result) -> None:
    test.assertEqual(result["support_count"], case.expected_support_count)
    test.assertEqual(result["completion_count"], case.expected_completion_count)
    test.assertEqual(result["support_count"], result["artifact_support_count"])
    test.assertEqual(result["completion_count"], result["artifact_witness_count"])


def assert_offline_case_matches_ledger(test, *, case, verification) -> None:
    test.assertTrue(verification.accepted, verification.reason)
    test.assertTrue(verification.offline_replay_complete)
    test.assertEqual(verification.offline_unchecked_obligation_families, ())
