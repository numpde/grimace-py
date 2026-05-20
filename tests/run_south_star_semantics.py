from __future__ import annotations

import unittest


SOUTH_STAR_SEMANTIC_MODULES: tuple[str, ...] = (
    "tests.south_star.test_annotation_conformance",
    "tests.south_star.test_annotation_policy_boundary",
    "tests.south_star.test_component_extraction",
    "tests.south_star.test_component_support_state",
    "tests.south_star.test_comparison_labels",
    "tests.south_star.test_conformance_oracle",
    "tests.south_star.test_dependency_boundaries",
    "tests.south_star.test_enum_s_prototype",
    "tests.south_star.test_first_domain_completeness",
    "tests.south_star.test_fragment_composition",
    "tests.south_star.test_harness",
    "tests.south_star.test_marker_slot_equations",
    "tests.south_star.test_output_correctness_harness",
    "tests.south_star.test_parity_solver",
    "tests.south_star.test_policy_modularity",
    "tests.south_star.test_semantic_diagnostics",
    "tests.south_star.test_semantic_witnesses",
    "tests.south_star.test_support_boundary",
    "tests.south_star.test_support_gates",
    "tests.south_star.test_z3_equation_oracle",
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
