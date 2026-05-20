from __future__ import annotations

import unittest

from grimace._south_star.enum_s import mol_to_smiles_enum_s_graph_native
from tests.helpers.south_star_complexity_diagnostics import (
    south_star_complexity_diagnostic_for_case,
)
from tests.helpers.south_star_exact_support import (
    load_south_star_expanded_support_cases,
)


class SouthStarComplexityGuardrailTests(unittest.TestCase):
    def test_generation_diagnostics_expose_product_and_dedup_guardrails(
        self,
    ) -> None:
        for case in load_south_star_expanded_support_cases():
            result = mol_to_smiles_enum_s_graph_native(
                case.source_smiles,
                case_id=case.case_id,
            )

            with self.subTest(case_id=case.case_id):
                diagnostics = result.generation_diagnostics
                self.assertIsNotNone(diagnostics)
                if diagnostics is None:
                    continue
                self.assertEqual(len(result.outputs), diagnostics.output_count)
                self.assertEqual(
                    diagnostics.estimated_product_size,
                    diagnostics.raw_output_count,
                )
                self.assertGreaterEqual(
                    diagnostics.raw_output_count,
                    diagnostics.output_count,
                )
                self.assertEqual(
                    diagnostics.raw_output_count - diagnostics.output_count,
                    diagnostics.deduplication_drop_count,
                )
                self.assertGreater(diagnostics.deduplicated_output_ratio, 0.0)
                self.assertLessEqual(diagnostics.deduplicated_output_ratio, 1.0)
                self.assertGreaterEqual(diagnostics.fragment_order_count, 1)
                if diagnostics.fragment_count == 1:
                    self.assertEqual(1, diagnostics.fragment_order_count)
                else:
                    self.assertGreater(diagnostics.fragment_order_count, 1)

    def test_named_complexity_diagnostic_tracks_per_layer_timing(self) -> None:
        case = next(
            case
            for case in load_south_star_expanded_support_cases()
            if case.case_id == "ring_stereo_monocycle_cyclooctene"
        )

        diagnostic = south_star_complexity_diagnostic_for_case(case)

        self.assertEqual(case.case_id, diagnostic.case_id)
        self.assertGreater(diagnostic.generation_diagnostics.raw_output_count, 0)
        self.assertGreaterEqual(diagnostic.timing.fact_extraction_seconds, 0.0)
        self.assertGreaterEqual(diagnostic.timing.generation_seconds, 0.0)
        self.assertGreaterEqual(diagnostic.timing.conformance_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()
