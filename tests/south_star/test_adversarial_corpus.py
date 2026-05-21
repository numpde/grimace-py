from __future__ import annotations

import unittest

from tests.helpers.south_star_adversarial_corpus import (
    SOUTH_STAR_ADVERSARIAL_AXES,
    SOUTH_STAR_ADVERSARIAL_BOUNDARY_TARGETS,
    SOUTH_STAR_ADVERSARIAL_REQUIRED_SUPPORTED_FEATURE_TARGETS,
    SOUTH_STAR_ADVERSARIAL_REQUIRED_UNSUPPORTED_CATEGORY_TARGETS,
    generate_south_star_adversarial_candidates,
    south_star_adversarial_triage,
)
from tests.helpers.south_star_domain_manifest import SOUTH_STAR_PRIVATE_DOMAIN
from tests.helpers.south_star_semantic_oracle import parse_smiles


class SouthStarAdversarialCorpusTests(unittest.TestCase):
    def test_generator_is_deterministic_and_shrink_sorted(self) -> None:
        first = generate_south_star_adversarial_candidates()
        second = generate_south_star_adversarial_candidates()

        self.assertEqual(first, second)
        self.assertEqual(tuple(sorted(first, key=lambda case: case.shrink_key)), first)
        self.assertEqual(len({case.candidate_id for case in first}), len(first))

    def test_generator_covers_declared_axes(self) -> None:
        observed_axes = frozenset(
            axis
            for candidate in generate_south_star_adversarial_candidates()
            for axis in candidate.axes
        )

        self.assertEqual(SOUTH_STAR_ADVERSARIAL_AXES, observed_axes)

    def test_generator_covers_declared_boundary_targets(self) -> None:
        candidates = generate_south_star_adversarial_candidates()
        observed_targets = frozenset(
            target for candidate in candidates for target in candidate.boundary_targets
        )

        self.assertTrue(observed_targets <= SOUTH_STAR_ADVERSARIAL_BOUNDARY_TARGETS)
        self.assertTrue(
            SOUTH_STAR_ADVERSARIAL_REQUIRED_SUPPORTED_FEATURE_TARGETS
            <= observed_targets
        )
        self.assertTrue(
            SOUTH_STAR_ADVERSARIAL_REQUIRED_UNSUPPORTED_CATEGORY_TARGETS
            <= observed_targets
        )

    def test_generated_candidates_are_triage_inputs_not_support_fixtures(self) -> None:
        for candidate in generate_south_star_adversarial_candidates():
            with self.subTest(candidate_id=candidate.candidate_id):
                self.assertFalse(hasattr(candidate, "expected_support"))
                self.assertTrue(candidate.mutation_path)
                self.assertTrue(candidate.boundary_targets)
                parse_smiles(candidate.source_smiles)

    def test_triage_exposes_supported_and_unsupported_boundary_cases(self) -> None:
        triage_reports = tuple(
            south_star_adversarial_triage(candidate)
            for candidate in generate_south_star_adversarial_candidates()
        )

        self.assertTrue(any(report.supported_by_gate for report in triage_reports))
        self.assertTrue(any(not report.supported_by_gate for report in triage_reports))
        for report in triage_reports:
            with self.subTest(candidate_id=report.candidate.candidate_id):
                if report.supported_by_gate:
                    self.assertEqual((), report.unsupported_categories)
                    self.assertIsNotNone(report.generation_diagnostics)
                    self.assertIsNotNone(report.generated_output_count)
                    if report.generated_output_count is not None:
                        self.assertGreater(report.generated_output_count, 0)
                else:
                    self.assertNotEqual((), report.unsupported_categories)
                    self.assertIsNone(report.generation_diagnostics)
                    self.assertIsNone(report.generated_output_count)

    def test_unsupported_trigger_candidates_expose_named_gate_categories(self) -> None:
        unsupported_reports = tuple(
            report
            for report in (
                south_star_adversarial_triage(candidate)
                for candidate in generate_south_star_adversarial_candidates()
            )
            if "unsupported_feature_trigger" in report.candidate.axes
        )

        observed_categories = {
            category
            for report in unsupported_reports
            for category in report.unsupported_categories
        }
        targeted_categories = {
            target
            for report in unsupported_reports
            for target in report.candidate.boundary_targets
            if target in SOUTH_STAR_PRIVATE_DOMAIN.unsupported_feature_categories
        }

        self.assertIn("unsupported_bond_type", observed_categories)
        self.assertIn("dative_bond", observed_categories)
        self.assertIn("aromatic_ring_surface", observed_categories)
        self.assertTrue(targeted_categories <= observed_categories)


if __name__ == "__main__":
    unittest.main()
