from __future__ import annotations

import unittest

from tests.helpers.south_star_annotation_conformance import (
    ANNOTATION_CONFORMANCE_BASIS,
    south_star_annotation_conformance,
)
from tests.helpers.south_star_semantic_oracle import south_star_conformance_report
from tests.helpers.south_star_grammar_conformance import (
    SOUTH_STAR_GRAMMAR_CONFORMANCE_BASIS,
    south_star_grammar_conformance,
)


class SouthStarAnnotationConformanceTests(unittest.TestCase):
    def test_current_subset_accepts_directional_markers_on_atom_bonds(self) -> None:
        for smiles in (
            "F/C=C\\Cl",
            "Cl/C=C(\\F)/Br",
            "C/C=N/O",
        ):
            with self.subTest(smiles=smiles):
                self.assertTrue(south_star_annotation_conformance(smiles).passed)

    def test_current_subset_rejects_misplaced_directional_markers(self) -> None:
        for smiles in (
            "/FC=CCl",
            "FC=CCl/",
            "F/=CCl",
            "F//C=CCl",
        ):
            with self.subTest(smiles=smiles):
                self.assertFalse(south_star_annotation_conformance(smiles).passed)

    def test_current_subset_accepts_ring_labels_brackets_and_fragments(self) -> None:
        for smiles in (
            "C1=CCCCC1",
            "C(/C=C\\1)CCCCC1",
            "C[C@H](F)Cl",
            "[C@@H](C)(F)Cl",
            "[H][H]",
            "O.F/C=C\\Cl",
        ):
            with self.subTest(smiles=smiles):
                self.assertTrue(south_star_grammar_conformance(smiles).passed)

    def test_current_subset_reports_structured_grammar_rejections(self) -> None:
        cases = (
            ("C1CC", "unpaired_ring_label"),
            ("C..O", "dot_context"),
            ("[NH4+]", "unsupported_token"),
            ("C//C", "consecutive_bonds"),
        )

        for smiles, rejection_code in cases:
            with self.subTest(smiles=smiles):
                report = south_star_grammar_conformance(smiles)
                self.assertFalse(report.passed)
                self.assertEqual(rejection_code, report.rejection_code)

    def test_conformance_report_uses_rdkit_independent_annotation_basis(self) -> None:
        report = south_star_conformance_report(
            source_smiles="F/C=C\\Cl",
            candidate_smiles="F\\C=C/Cl",
        )

        self.assertTrue(report.grammar_conformance.passed)
        self.assertEqual(
            SOUTH_STAR_GRAMMAR_CONFORMANCE_BASIS,
            report.grammar_conformance.basis,
        )
        self.assertEqual(ANNOTATION_CONFORMANCE_BASIS, report.grammar_conformance.basis)
