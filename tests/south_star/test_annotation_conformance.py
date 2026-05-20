from __future__ import annotations

import unittest

from tests.helpers.south_star_annotation_conformance import (
    ANNOTATION_CONFORMANCE_BASIS,
    south_star_annotation_conformance,
)
from tests.helpers.south_star_semantic_oracle import south_star_conformance_report


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

    def test_conformance_report_uses_rdkit_independent_annotation_basis(self) -> None:
        report = south_star_conformance_report(
            source_smiles="F/C=C\\Cl",
            candidate_smiles="F\\C=C/Cl",
        )

        self.assertTrue(report.annotation_conformance.passed)
        self.assertEqual(
            ANNOTATION_CONFORMANCE_BASIS,
            report.annotation_conformance.basis,
        )
