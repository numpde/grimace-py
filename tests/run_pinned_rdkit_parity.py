from __future__ import annotations

import unittest

from rdkit import rdBase

from tests.helpers.pinned_rdkit_fixtures import (
    PINNED_RDKIT_PARITY_MODULES,
    missing_pinned_rdkit_fixture_roots,
    pinned_rdkit_parity_fixture_roots,
)


class PinnedRdkitParityFixtureAvailabilityTest(unittest.TestCase):
    def test_installed_rdkit_version_has_checked_in_pinned_fixtures(self) -> None:
        missing = missing_pinned_rdkit_fixture_roots(
            pinned_rdkit_parity_fixture_roots(),
            rdBase.rdkitVersion,
        )

        self.assertEqual(
            (),
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
