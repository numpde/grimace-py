from __future__ import annotations

import unittest

from rdkit import rdBase

from tests.helpers.pinned_rdkit_fixtures import (
    has_pinned_rdkit_fixture,
    pinned_rdkit_parity_fixture_roots,
)


PINNED_RDKIT_PARITY_MODULES: tuple[str, ...] = (
    "tests.rdkit_serialization.test_exact_small_support",
    "tests.rdkit_serialization.test_rooted_random.RDKITRootedRandomWriterTests."
    "test_rdkit_rooted_random_generation_cases_are_in_grimace_support",
    "tests.rdkit_serialization.test_serializer_regressions",
    "tests.rdkit_serialization.test_writer_membership",
)


class PinnedRdkitParityFixtureAvailabilityTest(unittest.TestCase):
    def test_installed_rdkit_version_has_checked_in_pinned_fixtures(self) -> None:
        missing = [
            str(fixture_root)
            for fixture_root in pinned_rdkit_parity_fixture_roots()
            if not has_pinned_rdkit_fixture(fixture_root, rdBase.rdkitVersion)
        ]

        self.assertEqual(
            [],
            missing,
            "pinned RDKit parity runner requires checked-in fixtures for "
            f"installed RDKit {rdBase.rdkitVersion}",
        )


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del tests, pattern
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(PinnedRdkitParityFixtureAvailabilityTest))
    suite.addTests(loader.loadTestsFromNames(PINNED_RDKIT_PARITY_MODULES))
    return suite


if __name__ == "__main__":
    unittest.main()
