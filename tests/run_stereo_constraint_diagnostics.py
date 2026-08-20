from __future__ import annotations

import os
import unittest

from tests.integration.test_stereo_constraint_model import (
    STEREO_CONSTRAINT_DIAGNOSTIC_ENV,
)


STEREO_CONSTRAINT_DIAGNOSTIC_MODULES: tuple[str, ...] = (
    "tests.integration.test_stereo_constraint_model."
    "StereoConstraintModelFixtureTests."
    "test_current_runtime_support_count_matches_pinned_witnesses",
    "tests.integration.test_stereo_constraint_model."
    "StereoConstraintModelFixtureTests."
    "test_reduced_porphyrin_terminal_rows_keep_marker_boundary_survivors",
)


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del tests, pattern
    os.environ[STEREO_CONSTRAINT_DIAGNOSTIC_ENV] = "1"
    return loader.loadTestsFromNames(STEREO_CONSTRAINT_DIAGNOSTIC_MODULES)


if __name__ == "__main__":
    unittest.main()
