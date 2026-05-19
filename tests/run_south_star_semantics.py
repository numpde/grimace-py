from __future__ import annotations

import unittest


SOUTH_STAR_SEMANTIC_MODULES: tuple[str, ...] = (
    "tests.south_star.test_annotation_policy_boundary",
    "tests.south_star.test_comparison_labels",
    "tests.south_star.test_dependency_boundaries",
    "tests.south_star.test_harness",
    "tests.south_star.test_semantic_diagnostics",
    "tests.south_star.test_semantic_witnesses",
)


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del tests, pattern
    return loader.loadTestsFromNames(SOUTH_STAR_SEMANTIC_MODULES)


if __name__ == "__main__":
    unittest.main()
