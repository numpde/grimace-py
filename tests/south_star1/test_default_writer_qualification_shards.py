"""Partition invariants for South Star qualification lanes."""

from __future__ import annotations

import unittest
import importlib

from tests.south_star1.default_writer_capability_ledger import (
    ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES,
)
from tests.south_star1.default_writer_qualification_shards import (
    CONTINUATION_PROOF_QUALIFIED_CASES,
    FAST_ACCEPTED_CASES,
    MATERIALIZED_ARTIFACT_QUALIFIED_CASES,
    SLOW_COUPLED_CASES,
    SLOW_COUPLED_CASE_NAMES,
    SLOW_QUALIFICATION_SHARDS,
    slow_cases_for_shard,
)


class DefaultWriterQualificationShardsTest(unittest.TestCase):
    def test_qualification_authorities_partition_accepted_cases(self) -> None:
        accepted = {case.name for case in ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES}
        materialized = {case.name for case in MATERIALIZED_ARTIFACT_QUALIFIED_CASES}
        continuation = {case.name for case in CONTINUATION_PROOF_QUALIFIED_CASES}
        self.assertTrue(materialized.isdisjoint(continuation))
        self.assertEqual(materialized | continuation, accepted)
        self.assertEqual(
            continuation,
            {"remote_coupled_tetrahedral_a", "remote_coupled_tetrahedral_b"},
        )
        for case in CONTINUATION_PROOF_QUALIFIED_CASES:
            with self.subTest(case=case.name):
                self.assertEqual(case.expected_continuation_raw_cursor_count, 3075)
                self.assertEqual(case.expected_continuation_edge_locator_count, 3074)
                self.assertEqual(case.expected_continuation_branch_locator_count, 3848)
                self.assertEqual(case.expected_continuation_terminal_record_count, 216)
                self.assertEqual(case.expected_continuation_terminal_locator_count, 216)

    def test_partition_is_disjoint_complete_and_ledger_ordered(self) -> None:
        accepted = tuple(ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES)
        fast_names = tuple(case.name for case in FAST_ACCEPTED_CASES)
        slow_names = tuple(case.name for case in SLOW_COUPLED_CASES)
        accepted_names = tuple(case.name for case in accepted)
        self.assertTrue(set(fast_names).isdisjoint(slow_names))
        self.assertEqual(set(fast_names) | set(slow_names), set(accepted_names))
        self.assertEqual(len(fast_names) + len(slow_names), len(accepted_names))
        self.assertEqual(slow_names, SLOW_COUPLED_CASE_NAMES)
        self.assertEqual(
            tuple(
                case.name
                for case in accepted
                if case.name in fast_names or case.name in slow_names
            ),
            accepted_names,
        )

    def test_expensive_consumers_bind_to_the_shared_partition(self) -> None:
        module_names = (
            "tests.south_star1.test_public_continuation_asset",
            "tests.south_star1.test_public_continuation_asset_verification",
            "tests.south_star1.test_public_continuation_proofs",
            "tests.south_star1.test_writer_default_offline_complete",
            "tests.south_star1.test_writer_default_parity_corpus",
            "tests.south_star1.test_writer_default_continuation_corpus",
        )
        for module_name in module_names:
            module = importlib.import_module(module_name)
            with self.subTest(module=module_name):
                self.assertIs(module.FAST_ACCEPTED_CASES, FAST_ACCEPTED_CASES)
                self.assertIs(module.SLOW_COUPLED_CASES, SLOW_COUPLED_CASES)
        audit = importlib.import_module(
            "tests.south_star1.test_writer_default_stereo_audit_fixture"
        )
        self.assertIs(audit.WriterDefaultStereoAuditFixtureTest.QUALIFICATION_CASES, FAST_ACCEPTED_CASES)
        self.assertIsNone(audit.WriterDefaultStereoAuditSlowTest.QUALIFICATION_CASES)

    def test_slow_case_shards_are_disjoint_complete_and_ordered(self) -> None:
        shard_names = tuple(SLOW_QUALIFICATION_SHARDS)
        self.assertEqual(
            set(shard_names), {"zero-h-adjacent", "remote-a", "remote-b"}
        )
        shard_sets = [set(SLOW_QUALIFICATION_SHARDS[name]) for name in shard_names]
        for index, left in enumerate(shard_sets):
            for right in shard_sets[index + 1 :]:
                self.assertTrue(left.isdisjoint(right))
        self.assertEqual(
            set().union(*shard_sets), {case.name for case in SLOW_COUPLED_CASES}
        )
        for name in shard_names:
            self.assertEqual(
                tuple(case.name for case in slow_cases_for_shard(name)),
                tuple(
                    case.name
                    for case in SLOW_COUPLED_CASES
                    if case.name in SLOW_QUALIFICATION_SHARDS[name]
                ),
            )

    def test_unknown_or_empty_slow_case_shards_are_rejected(self) -> None:
        for name in ("", "unknown"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    slow_cases_for_shard(name)


if __name__ == "__main__":
    unittest.main()
