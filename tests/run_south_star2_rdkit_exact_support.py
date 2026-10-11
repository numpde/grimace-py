from __future__ import annotations

import unittest

from rdkit import rdBase

from tests.helpers.pinned_rdkit_fixtures import load_pinned_rdkit_fixture_cases
from tests.helpers.fixture_paths import checked_in_fixture_path
from tests.south_star2.test_rdkit_exact_support import SouthStar2RdkitExactSupportTests


class SouthStar2ExactSupportFixtureAvailabilityTest(unittest.TestCase):
    def test_installed_rdkit_version_has_exact_support_fixture(self) -> None:
        load_pinned_rdkit_fixture_cases(
            fixture_root=checked_in_fixture_path("rdkit_south_star2_exact_support"),
            rdkit_version=rdBase.rdkitVersion,
            fixture_label="South Star 2 exact support",
        )


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del tests, pattern
    suite = unittest.TestSuite()
    suite.addTests(
        loader.loadTestsFromTestCase(SouthStar2ExactSupportFixtureAvailabilityTest)
    )
    suite.addTests(loader.loadTestsFromTestCase(SouthStar2RdkitExactSupportTests))
    return suite


if __name__ == "__main__":
    unittest.main()
