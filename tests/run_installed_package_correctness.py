from __future__ import annotations

import unittest


INSTALLED_PACKAGE_CORRECTNESS_MODULES: tuple[str, ...] = (
    "tests.integration.test_python_api_smoke",
    "tests.integration.test_public_decoder",
    "tests.integration.test_token_inventory",
    "tests.integration.test_public_runtime_writer_flags",
    "tests.integration.test_public_all_roots_identities",
    "tests.integration.test_public_prepared_equivalence",
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
