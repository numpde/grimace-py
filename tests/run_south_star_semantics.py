from __future__ import annotations

import unittest


SOUTH_STAR_SEMANTIC_MODULES: tuple[str, ...] = (
    "tests.south_star.test_adversarial_corpus",
    "tests.south_star.test_annotation_policy_boundary",
    "tests.south_star.test_component_extraction",
    "tests.south_star.test_component_support_state",
    "tests.south_star.test_comparison_labels",
    "tests.south_star.test_compositional_stereo_proof",
    "tests.south_star.test_complexity_guardrails",
    "tests.south_star.test_conformance_oracle",
    "tests.south_star.test_constraint_vocabulary",
    "tests.south_star.test_dependency_boundaries",
    "tests.south_star.test_domain_manifest",
    "tests.south_star.test_enum_s_prototype",
    "tests.south_star.test_expanded_support_fixtures",
    "tests.south_star.test_first_domain_completeness",
    "tests.south_star.test_fragment_composition",
    "tests.south_star.test_grammar_conformance",
    "tests.south_star.test_harness",
    "tests.south_star.test_marker_slot_equations",
    "tests.south_star.test_output_correctness_harness",
    "tests.south_star.test_package_readiness",
    "tests.south_star.test_parity_solver",
    "tests.south_star.test_policy_modularity",
    "tests.south_star.test_private_api_boundary",
    "tests.south_star.test_semantic_diagnostics",
    "tests.south_star.test_semantic_identity",
    "tests.south_star.test_semantic_witnesses",
    "tests.south_star.test_support_boundary",
    "tests.south_star.test_support_gates",
    "tests.south_star.test_tetrahedral_facts",
    "tests.south_star.test_unified_reference_promotion",
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
