from __future__ import annotations

import unittest

from tests.helpers.south_star_semantic_oracle import (
    semantic_oracle_accepts,
    south_star_conformance_report,
)
from tests.helpers.south_star_grammar_conformance import (
    SOUTH_STAR_GRAMMAR_CONFORMANCE_BASIS,
)


class SouthStarConformanceOracleTests(unittest.TestCase):
    def test_positive_output_passes_all_conformance_checks(self) -> None:
        report = south_star_conformance_report(
            source_smiles="F/C=C\\Cl",
            candidate_smiles="F\\C=C/Cl",
        )

        self.assertTrue(report.accepted)
        self.assertEqual((), report.rejection_reasons)
        self.assertTrue(report.rdkit_parseability.passed)
        self.assertTrue(report.graph_equivalence.passed)
        self.assertTrue(report.stereo_equivalence.passed)
        self.assertTrue(report.grammar_conformance.passed)
        self.assertEqual(
            SOUTH_STAR_GRAMMAR_CONFORMANCE_BASIS,
            report.grammar_conformance.basis,
        )

    def test_inverted_stereo_keeps_graph_but_fails_stereo_equivalence(self) -> None:
        report = south_star_conformance_report(
            source_smiles="F/C=C\\Cl",
            candidate_smiles="F/C=C/Cl",
        )

        self.assertFalse(report.accepted)
        self.assertEqual(("stereo_equivalence",), report.rejection_reasons)
        self.assertTrue(report.rdkit_parseability.passed)
        self.assertTrue(report.graph_equivalence.passed)
        self.assertFalse(report.stereo_equivalence.passed)
        self.assertTrue(report.grammar_conformance.passed)

    def test_invalid_smiles_fails_each_dependent_check_without_raising(self) -> None:
        report = south_star_conformance_report(
            source_smiles="F/C=C\\Cl",
            candidate_smiles="F/C=C\\",
        )

        self.assertFalse(report.accepted)
        self.assertEqual(
            (
                "rdkit_parseability",
                "graph_equivalence",
                "stereo_equivalence",
                "grammar_conformance",
            ),
            report.rejection_reasons,
        )

    def test_grammar_conformance_is_distinct_from_semantic_identity(self) -> None:
        report = south_star_conformance_report(
            source_smiles="F/C=C\\Cl",
            candidate_smiles="F/C=C/Cl",
        )

        self.assertTrue(report.grammar_conformance.passed)
        self.assertTrue(report.rdkit_parseability.passed)
        self.assertTrue(report.graph_equivalence.passed)
        self.assertFalse(report.stereo_equivalence.passed)

    def test_grammar_conformance_is_distinct_from_rdkit_parseability(self) -> None:
        report = south_star_conformance_report(
            source_smiles="CC",
            candidate_smiles="C(C)(C)(C)(C)C",
        )

        self.assertTrue(report.grammar_conformance.passed)
        self.assertFalse(report.rdkit_parseability.passed)

    def test_legacy_boolean_oracle_is_backed_by_structured_report(self) -> None:
        self.assertTrue(
            semantic_oracle_accepts(
                source_smiles="C/C=N/O",
                candidate_smiles="C\\C=N\\O",
            )
        )
        self.assertFalse(
            semantic_oracle_accepts(
                source_smiles="C/C=N/O",
                candidate_smiles="C/C=N\\O",
            )
        )
