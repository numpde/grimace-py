from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from tests.south_star1 import qualification_guards as guards


class QualificationGuardsTest(unittest.TestCase):
    def test_profiles_are_named_and_nonempty(self) -> None:
        self.assertTrue(guards.QUALIFICATION_GUARD_PROFILES)
        self.assertTrue(all(profile for profile in guards.QUALIFICATION_GUARD_PROFILES.values()))

    def test_guard_reports_zero_calls_when_paths_are_unused(self) -> None:
        with guards.forbid_qualification_paths(*guards.guard_profile("public-runtime")) as report:
            pass
        report.assert_unused(self)
        self.assertTrue(all(count == 0 for count in report.call_counts().values()))

    def test_guard_blocks_real_owner_lookup(self) -> None:
        with guards.forbid_qualification_paths(guards.QualificationPath.COUNT_DAG_BUILD):
            with self.assertRaisesRegex(AssertionError, "count_dag_build"):
                guards.writer_count_dag_envelope.writer_count_certificate_dag_envelope_for_product(
                    prepared=None, product=None
                )

    def test_guard_targets_are_real_attributes(self) -> None:
        for path, targets in guards._TARGETS.items():
            with self.subTest(path=path):
                for owner, name in targets:
                    self.assertTrue(hasattr(owner, name), (path, owner, name))


if __name__ == "__main__":
    unittest.main()
