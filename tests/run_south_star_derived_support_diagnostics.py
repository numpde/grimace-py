from __future__ import annotations

import os
import unittest


SOUTH_STAR_DERIVED_SUPPORT_DIAGNOSTIC_ENV = (
    "RUN_SOUTH_STAR_DERIVED_SUPPORT_DIAGNOSTICS"
)
SOUTH_STAR_DERIVED_SUPPORT_DIAGNOSTIC_MODULES: tuple[str, ...] = (
    "tests.south_star.test_derived_support_fixtures."
    "SouthStarDerivedSupportFixtureTests."
    "test_derived_support_all_outputs_preserve_semantics",
)


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del tests, pattern
    os.environ[SOUTH_STAR_DERIVED_SUPPORT_DIAGNOSTIC_ENV] = "1"
    return loader.loadTestsFromNames(SOUTH_STAR_DERIVED_SUPPORT_DIAGNOSTIC_MODULES)


if __name__ == "__main__":
    unittest.main()
