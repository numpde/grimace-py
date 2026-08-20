from __future__ import annotations

import unittest


EXACT_PUBLIC_INVARIANT_MODULES: tuple[str, ...] = (
    "tests.integration.test_public_decoder",
    "tests.integration.test_token_inventory",
    "tests.integration.test_public_runtime_writer_flags",
    "tests.integration.test_public_all_roots_identities",
    "tests.integration.test_public_prepared_equivalence",
)


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del tests, pattern
    return loader.loadTestsFromNames(EXACT_PUBLIC_INVARIANT_MODULES)


if __name__ == "__main__":
    unittest.main()
