from __future__ import annotations

import unittest


SOUTH_STAR_PACKAGE_READINESS_MODULES: tuple[str, ...] = (
    "tests.south_star.test_domain_manifest",
    "tests.south_star.test_expanded_support_fixtures",
    "tests.south_star.test_first_domain_completeness",
    "tests.south_star.test_package_readiness",
    "tests.south_star.test_support_gates",
    "tests.south_star.test_enum_s_prototype",
)


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del tests, pattern
    return loader.loadTestsFromNames(SOUTH_STAR_PACKAGE_READINESS_MODULES)


if __name__ == "__main__":
    unittest.main()
