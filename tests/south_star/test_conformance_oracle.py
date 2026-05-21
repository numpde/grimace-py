from __future__ import annotations

import unittest

from grimace._south_star.enum_s import mol_to_smiles_enum_s_graph_native
from tests.helpers.south_star_semantic_oracle import (
    semantic_oracle_accepts,
    south_star_conformance_report,
)
from tests.helpers.south_star_spec_oracle import (
    SOUTH_STAR_SPEC_ORACLE_BASIS,
    SOUTH_STAR_SPEC_ORACLE_GENERATION_AUTHORITY,
    SOUTH_STAR_SMALL_SUPPORT_ORACLE_BASIS,
    south_star_spec_oracle_report,
    south_star_small_support_completeness_report,
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

    def test_spec_oracle_is_evidence_not_generation_authority(self) -> None:
        report = south_star_spec_oracle_report(
            source_smiles="F/C=C\\Cl",
            candidate_smiles=("F\\C=C/Cl", "F/C=C/Cl"),
        )

        self.assertEqual(SOUTH_STAR_SPEC_ORACLE_BASIS, report.basis)
        self.assertEqual(
            SOUTH_STAR_SPEC_ORACLE_GENERATION_AUTHORITY,
            report.generation_authority,
        )
        self.assertEqual(2, report.candidate_count)
        self.assertEqual(1, report.accepted_count)
        self.assertFalse(report.all_accepted)
        self.assertEqual(1, len(report.rejected_candidates))
        self.assertEqual(
            ("stereo_equivalence",),
            report.rejected_candidates[0].rejection_reasons,
        )

    def test_small_support_oracle_checks_completeness_without_runtime_renderer(
        self,
    ) -> None:
        cases = (
            "C#N",
            "[2H][H]",
            "[CH3:1]C",
            "[H+]",
            "[Cl-]",
            "[NH4+]",
            "[CH3]",
            "[O]",
        )

        for smiles in cases:
            observed = mol_to_smiles_enum_s_graph_native(smiles).outputs
            report = south_star_small_support_completeness_report(
                source_smiles=smiles,
                observed_support=observed,
            )

            with self.subTest(smiles=smiles):
                self.assertEqual(SOUTH_STAR_SMALL_SUPPORT_ORACLE_BASIS, report.basis)
                self.assertEqual(
                    SOUTH_STAR_SPEC_ORACLE_GENERATION_AUTHORITY,
                    report.generation_authority,
                )
                self.assertTrue(report.complete)
                self.assertEqual((), report.missing_candidates)
                self.assertEqual((), report.extra_candidates)
