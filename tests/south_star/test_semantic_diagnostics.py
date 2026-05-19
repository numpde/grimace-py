from __future__ import annotations

import unittest

from tests.helpers.south_star_diagnostics import south_star_semantic_diagnostic
from tests.helpers.south_star_semantics import load_south_star_semantic_cases


class SouthStarSemanticDiagnosticTests(unittest.TestCase):
    def test_diagnostic_separates_semantic_facts_from_policy_decisions(self) -> None:
        for case in load_south_star_semantic_cases():
            diagnostic = south_star_semantic_diagnostic(case)

            with self.subTest(case_id=case.case_id):
                self.assertEqual(case.case_id, diagnostic.semantic_facts.case_id)
                self.assertEqual(case.source_smiles, diagnostic.semantic_facts.source_smiles)
                self.assertEqual(
                    case.eligible_carrier_edges,
                    tuple(
                        opportunity.edge
                        for opportunity in diagnostic.semantic_facts.carrier_opportunities
                    ),
                )
                self.assertEqual(
                    case.maximal_eligible_carrier.required_marker_edge_count,
                    sum(
                        int(decision.marker_required)
                        for decision in diagnostic.annotation_policy_decisions
                    ),
                )

    def test_diagnostic_does_not_expose_rdkit_writer_membership(self) -> None:
        for case in load_south_star_semantic_cases():
            diagnostic = south_star_semantic_diagnostic(case)

            with self.subTest(case_id=case.case_id):
                self.assertFalse(hasattr(diagnostic, "rdkit_writer_membership_status"))
                self.assertFalse(
                    hasattr(diagnostic.semantic_facts, "rdkit_writer_membership_status")
                )
