from __future__ import annotations

import unittest

from tests.helpers.south_star_semantic_identity import (
    SOUTH_STAR_GRAPH_IDENTITY_BASIS,
    SOUTH_STAR_PARSER_DEPENDENCY_BASIS,
    SOUTH_STAR_STEREO_IDENTITY_BASIS,
    south_star_semantic_identity_report,
)


class SouthStarSemanticIdentityTests(unittest.TestCase):
    def test_graph_and_stereo_identity_pass_for_equivalent_spelling(self) -> None:
        report = south_star_semantic_identity_report(
            source_smiles="F/C=C\\Cl",
            candidate_smiles="F\\C=C/Cl",
        )

        self.assertTrue(report.accepted)
        self.assertTrue(report.parser_dependency.passed)
        self.assertTrue(report.graph_identity.passed)
        self.assertTrue(report.stereo_identity.passed)
        self.assertEqual(
            SOUTH_STAR_PARSER_DEPENDENCY_BASIS,
            report.parser_dependency.basis,
        )
        self.assertEqual(SOUTH_STAR_GRAPH_IDENTITY_BASIS, report.graph_identity.basis)
        self.assertEqual(
            SOUTH_STAR_STEREO_IDENTITY_BASIS,
            report.stereo_identity.basis,
        )

    def test_stereo_identity_can_fail_after_graph_identity_passes(self) -> None:
        report = south_star_semantic_identity_report(
            source_smiles="F/C=C\\Cl",
            candidate_smiles="F/C=C/Cl",
        )

        self.assertTrue(report.parser_dependency.passed)
        self.assertTrue(report.graph_identity.passed)
        self.assertFalse(report.stereo_identity.passed)

    def test_ring_adjacent_tetrahedral_ligand_order_affects_stereo_identity(
        self,
    ) -> None:
        report = south_star_semantic_identity_report(
            source_smiles="F[C@H](Cl)C1CCCCC1",
            candidate_smiles="F[C@@H](Cl)C1CCCCC1",
        )

        self.assertTrue(report.parser_dependency.passed)
        self.assertTrue(report.graph_identity.passed)
        self.assertFalse(report.stereo_identity.passed)

    def test_parser_failure_is_classified_before_identity_checks(self) -> None:
        report = south_star_semantic_identity_report(
            source_smiles="CC",
            candidate_smiles="C(C)(C)(C)(C)C",
        )

        self.assertFalse(report.accepted)
        self.assertFalse(report.parser_dependency.passed)
        self.assertFalse(report.graph_identity.passed)
        self.assertFalse(report.stereo_identity.passed)
        self.assertIn("unavailable", report.graph_identity.detail)


if __name__ == "__main__":
    unittest.main()
