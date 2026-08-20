"""Tests for static South Star stereo templates."""

from __future__ import annotations

import unittest

from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import OccurrenceId
from grimace._south_star1.ids import SiteId
from grimace._south_star1.stereo_templates import build_stereo_templates
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import tetrahedral_facts


class StereoTemplateTest(unittest.TestCase):
    def test_template_builder_extracts_tetra_templates(self) -> None:
        bundle = build_stereo_templates(tetrahedral_facts())

        self.assertEqual(len(bundle.tetrahedral), 1)
        self.assertEqual(len(bundle.directional), 0)
        template = bundle.tetrahedral[0]
        self.assertEqual(template.site, SiteId(0))
        self.assertEqual(template.center, AtomId(0))
        self.assertEqual(
            template.reference_order,
            tuple(OccurrenceId(index) for index in range(4)),
        )
        self.assertEqual(template.ligand_occurrences, template.reference_order)

    def test_template_builder_extracts_directional_templates(self) -> None:
        bundle = build_stereo_templates(directional_facts())

        self.assertEqual(len(bundle.tetrahedral), 0)
        self.assertEqual(len(bundle.directional), 1)
        template = bundle.directional[0]
        self.assertEqual(template.site, SiteId(0))
        self.assertEqual(template.center_bond, BondId(0))
        self.assertEqual(template.left_endpoint, AtomId(0))
        self.assertEqual(template.right_endpoint, AtomId(1))
        self.assertEqual(template.left_ligands, (OccurrenceId(0),))
        self.assertEqual(template.right_ligands, (OccurrenceId(1),))
        self.assertEqual(template.reference_pair, (OccurrenceId(0), OccurrenceId(1)))


if __name__ == "__main__":
    unittest.main()
