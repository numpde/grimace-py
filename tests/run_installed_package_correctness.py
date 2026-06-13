from __future__ import annotations

import unittest

from rdkit import rdBase

from tests.helpers.pinned_rdkit_fixtures import (
    PINNED_RDKIT_WRITER_SUPPORT_COUNTS,
    missing_pinned_rdkit_fixture_roots,
    pinned_rdkit_fixture_root,
)


INSTALLED_PACKAGE_CORRECTNESS_MODULES: tuple[str, ...] = (
    "tests.integration.test_python_api_smoke",
    "tests.integration.test_prepared_mol_zstd_contract",
    "tests.rdkit_serialization.test_writer_support_counts",
    "tests.run_exact_public_invariants",
    "tests.run_pinned_rdkit_parity",
)


class InstalledPackageFixtureAvailabilityTest(unittest.TestCase):
    def test_installed_rdkit_version_has_writer_support_count_fixtures(self) -> None:
        fixture_root = pinned_rdkit_fixture_root(PINNED_RDKIT_WRITER_SUPPORT_COUNTS)
        missing = missing_pinned_rdkit_fixture_roots(
            (fixture_root,),
            rdBase.rdkitVersion,
        )

        self.assertEqual(
            (),
            missing,
            "installed-package correctness requires checked-in writer "
            f"support-count fixtures for installed RDKit {rdBase.rdkitVersion}",
        )


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del tests, pattern
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(InstalledPackageFixtureAvailabilityTest))
    suite.addTests(loader.loadTestsFromNames(INSTALLED_PACKAGE_CORRECTNESS_MODULES))
    return suite


if __name__ == "__main__":
    unittest.main()
