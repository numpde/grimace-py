"""Fast contract tests for continuation qualification topology and coverage."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from tests.south_star1.default_writer_qualification_shards import (
    CONTINUATION_PROOF_QUALIFIED_CASES,
    MATERIALIZED_ARTIFACT_QUALIFIED_CASES,
)
from tests.run_south_star1_slow import (
    CONTINUATION_AUTHORITY_DIAGNOSTIC_LAYERS,
    CONTINUATION_AUTHORITY_PRODUCT_LAYERS,
)
from tests.south_star1.test_public_continuation_proofs import (
    PublicProofCursorTargets,
    partition_public_proof_targets,
)


def assert_continuation_recertification_matches_case(
    test: unittest.TestCase,
    *,
    case,
    report,
) -> None:
    test.assertTrue(report.accepted)
    test.assertTrue(report.live_replay_complete)
    test.assertEqual(report.raw_cursor_count, case.expected_continuation_raw_cursor_count)
    test.assertEqual(report.edge_locator_count, case.expected_continuation_edge_locator_count)
    test.assertEqual(report.branch_locator_count, case.expected_continuation_branch_locator_count)
    test.assertEqual(report.branch_proof_count, case.expected_continuation_branch_locator_count)
    test.assertEqual(report.terminal_record_count, case.expected_continuation_terminal_record_count)
    test.assertEqual(report.terminal_locator_count, case.expected_continuation_terminal_locator_count)
    test.assertEqual(report.terminal_proof_count, case.expected_continuation_terminal_locator_count)
    test.assertEqual(report.branch_locator_count, report.branch_proof_count)
    test.assertEqual(report.terminal_locator_count, report.terminal_proof_count)
    test.assertEqual(report.semantically_replayed_operations, case.expected_continuation_replayed_operations)
    test.assertEqual(report.checked_relation_families, case.expected_continuation_checked_relation_families)
    test.assertEqual(report.checked_obligation_families, case.expected_continuation_checked_obligation_families)
    test.assertEqual(report.unchecked_obligation_families, case.expected_continuation_unchecked_obligation_families)
    test.assertEqual(report.unchecked_obligation_families, ())


class ContinuationQualificationContractTest(unittest.TestCase):
    def test_four_way_cursor_partition_is_deterministic_and_disjoint(self):
        groups = tuple(
            PublicProofCursorTargets(
                source_raw_cursor_digest=f"cursor-{index}",
                state=None,
                branch_locators=tuple(
                    SimpleNamespace(
                        source_raw_cursor_digest=f"cursor-{index}",
                        emitted_text=f"text-{offset}",
                        branch_certificate_digest=f"branch-{index}-{offset}",
                    )
                    for offset in range(index + 1)
                ),
                terminal_locators=(
                    SimpleNamespace(
                        source_raw_cursor_digest=f"cursor-{index}",
                        terminal_support_identity_digest=f"terminal-{index}",
                    ),
                ) if index % 2 == 0 else (),
            )
            for index in range(9)
        )
        first = partition_public_proof_targets(groups)
        second = partition_public_proof_targets(tuple(reversed(groups)))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        cursors = [
            group.source_raw_cursor_digest
            for shard in first
            for group in shard
        ]
        self.assertEqual(len(cursors), len(set(cursors)))
        self.assertEqual(
            set(cursors),
            {group.source_raw_cursor_digest for group in groups},
        )
        self.assertEqual(
            sum(len(group.branch_locators) for shard in first for group in shard),
            sum(len(group.branch_locators) for group in groups),
        )
        self.assertEqual(
            sum(len(group.terminal_locators) for shard in first for group in shard),
            sum(len(group.terminal_locators) for group in groups),
        )

    def test_duplicate_partition_locator_rejects(self):
        group = PublicProofCursorTargets(
            source_raw_cursor_digest="cursor",
            state=None,
            branch_locators=(
                SimpleNamespace(
                    source_raw_cursor_digest="cursor",
                    emitted_text="C",
                    branch_certificate_digest="branch",
                ),
                SimpleNamespace(
                    source_raw_cursor_digest="cursor",
                    emitted_text="C",
                    branch_certificate_digest="branch",
                ),
            ),
            terminal_locators=(),
        )
        with self.assertRaisesRegex(AssertionError, "duplicate"):
            partition_public_proof_targets((group,))

    def test_product_and_diagnostic_layers_are_disjoint_and_exact(self):
        self.assertEqual(
            CONTINUATION_AUTHORITY_PRODUCT_LAYERS,
            (
                "public-build",
                "public-certify",
                "public-runtime",
                "public-recertification",
                "public-proofs-0",
                "public-proofs-1",
                "public-proofs-2",
                "public-proofs-3",
                "support-reparse",
                "continuation",
                "stereo-audit",
            ),
        )
        self.assertEqual(
            CONTINUATION_AUTHORITY_DIAGNOSTIC_LAYERS,
            (
                "count-dag-build",
                "count-dag-validate",
                "support-artifact-build",
                "support-artifact-live",
                "offline-complete",
            ),
        )
        self.assertTrue(
            set(CONTINUATION_AUTHORITY_PRODUCT_LAYERS).isdisjoint(
                CONTINUATION_AUTHORITY_DIAGNOSTIC_LAYERS
            )
        )

    def test_monolithic_layer_is_absent(self):
        from tests import run_south_star1_slow

        self.assertNotIn(
            "continuation-proof-complete",
            run_south_star1_slow.SLOW_QUALIFICATION_LAYERS,
        )

    def test_authority_cases_have_exactly_one_product_authority(self):
        accepted = {
            case.name
            for case in (*MATERIALIZED_ARTIFACT_QUALIFIED_CASES, *CONTINUATION_PROOF_QUALIFIED_CASES)
        }
        self.assertEqual(
            accepted,
            {
                case.name
                for case in __import__(
                    "tests.south_star1.default_writer_capability_ledger",
                    fromlist=["ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES"],
                ).ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES
            },
        )
        self.assertTrue(
            {case.name for case in CONTINUATION_PROOF_QUALIFIED_CASES}.isdisjoint(
                case.name for case in MATERIALIZED_ARTIFACT_QUALIFIED_CASES
            )
        )
        self.assertEqual(
            {case.name for case in CONTINUATION_PROOF_QUALIFIED_CASES},
            {"remote_coupled_tetrahedral_a", "remote_coupled_tetrahedral_b"},
        )


if __name__ == "__main__":
    unittest.main()
