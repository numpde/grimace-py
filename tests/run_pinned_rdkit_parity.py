from __future__ import annotations

import unittest


PINNED_RDKIT_PARITY_MODULES: tuple[str, ...] = (
    "tests.rdkit_serialization.test_exact_small_support",
    "tests.rdkit_serialization.test_serializer_regressions",
)


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del tests, pattern
    return loader.loadTestsFromNames(PINNED_RDKIT_PARITY_MODULES)


if __name__ == "__main__":
    unittest.main()
