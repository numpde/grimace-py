"""Canonical unittest runner for the South Star 1 proof-kernel suite."""

from __future__ import annotations

import unittest


def load_tests(
    loader: unittest.TestLoader,
    standard_tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del standard_tests, pattern
    return loader.discover(
        start_dir="tests/south_star1",
        pattern="test*.py",
        top_level_dir=".",
    )


if __name__ == "__main__":
    unittest.main()
