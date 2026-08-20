from __future__ import annotations

import unittest


INSTALLED_PACKAGE_CORRECTNESS_MODULES: tuple[str, ...] = (
    "tests.integration.test_python_api_smoke",
    "tests.run_exact_public_invariants",
    "tests.run_pinned_rdkit_parity",
)


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del tests, pattern
    return loader.loadTestsFromNames(INSTALLED_PACKAGE_CORRECTNESS_MODULES)


if __name__ == "__main__":
    unittest.main()
