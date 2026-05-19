from __future__ import annotations

import unittest


KNOWN_STEREO_GAP_MODULES: tuple[str, ...] = (
    "tests.rdkit_serialization.known_stereo_gaps",
)


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del tests, pattern
    return loader.loadTestsFromNames(KNOWN_STEREO_GAP_MODULES)


if __name__ == "__main__":
    unittest.main()
