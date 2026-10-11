from __future__ import annotations

import unittest

from rdkit import rdBase

from tests.helpers.fixture_paths import checked_in_fixture_path
from tests.helpers.pinned_rdkit_fixtures import load_pinned_rdkit_fixture_cases
from tests.south_star2.test_preparation_semantics import (
    SouthStar2PreparationWriterParityTests,
)
from tests.south_star2.test_rdkit_exact_support import SouthStar2RdkitExactSupportTests


class SouthStar2ExactSupportFixtureAvailabilityTest(unittest.TestCase):
    def test_installed_rdkit_version_has_required_support_fixtures(self) -> None:
        for family, label in (
            ("rdkit_south_star2_exact_support", "South Star 2 exact support"),
            ("rdkit_south_star2_stereo_evidence", "South Star 2 stereo evidence"),
        ):
            load_pinned_rdkit_fixture_cases(
                fixture_root=checked_in_fixture_path(family),
                rdkit_version=rdBase.rdkitVersion,
                fixture_label=label,
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
    suite.addTests(loader.loadTestsFromTestCase(SouthStar2PreparationWriterParityTests))
    return suite


if __name__ == "__main__":
    unittest.main()
