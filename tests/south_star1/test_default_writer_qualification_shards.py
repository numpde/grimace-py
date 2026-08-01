"""Partition invariants for South Star qualification lanes."""

from __future__ import annotations

import unittest
import importlib

from tests.south_star1.default_writer_capability_ledger import (
    ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES,
)
from tests.south_star1.default_writer_qualification_shards import (
    FAST_ACCEPTED_CASES,
    SLOW_COUPLED_CASES,
    SLOW_COUPLED_CASE_NAMES,
)


class DefaultWriterQualificationShardsTest(unittest.TestCase):
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
        self.assertIs(audit.WriterDefaultStereoAuditSlowTest.QUALIFICATION_CASES, SLOW_COUPLED_CASES)


if __name__ == "__main__":
    unittest.main()
