from __future__ import annotations

from pathlib import Path
import unittest

from rdkit import rdBase


PINNED_RDKIT_PARITY_MODULES: tuple[str, ...] = (
    "tests.rdkit_serialization.test_exact_small_support",
    "tests.rdkit_serialization.test_serializer_regressions",
)
PINNED_RDKIT_FIXTURE_ROOTS: tuple[Path, ...] = (
    Path(__file__).resolve().parent / "fixtures" / "rdkit_exact_small_support",
    Path(__file__).resolve().parent / "fixtures" / "rdkit_serializer_regressions",
)


def _has_pinned_fixture(fixture_root: Path, rdkit_version: str) -> bool:
    fixture_path = fixture_root / f"{rdkit_version}.json"
    fixture_dir = fixture_root / rdkit_version
    return fixture_path.is_file() or (
        fixture_dir.is_dir() and any(fixture_dir.glob("*.json"))
    )


class PinnedRdkitParityFixtureAvailabilityTest(unittest.TestCase):
    def test_installed_rdkit_version_has_checked_in_pinned_fixtures(self) -> None:
        missing = [
            str(fixture_root)
            for fixture_root in PINNED_RDKIT_FIXTURE_ROOTS
            if not _has_pinned_fixture(fixture_root, rdBase.rdkitVersion)
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
