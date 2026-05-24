"""Tests for ordinary specified-stereo closure construction."""

from __future__ import annotations

from dataclasses import replace
import unittest

from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.facts import TetraValue
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ordinary_stereo_closure import RawSpecifiedStereo
from grimace._south_star1.ordinary_stereo_closure import RawTetraStereoRecord
from grimace._south_star1.ordinary_stereo_closure import (
    build_ordinary_stereo_specified_closure,
)
from grimace._south_star1.ordinary_stereo_sites import OrdinaryStereoSiteOptions

from tests.south_star1.helpers import deep_tetra_ligand_facts


class OrdinaryStereoClosureTest(unittest.TestCase):
    def test_specified_closure_accepts_graph_exact_raw_tetra_site(self) -> None:
        facts = _deep_tetra_with_implicit_h(right_terminal="Cl")
        raw = _raw_tetra_for_center_zero()

        result = build_ordinary_stereo_specified_closure(
            facts,
            raw_specified=raw,
            site_options=OrdinaryStereoSiteOptions(
                ligand_equivalence="exact_graph_automorphism"
            ),
        )

        self.assertEqual(result.promoted_tetrahedral, 1)
        self.assertEqual(len(result.facts.stereo.tetrahedral), 1)
        self.assertEqual(result.facts.stereo.tetrahedral[0].center, AtomId(0))

    def test_specified_closure_rejects_self_only_distinctness(self) -> None:
        facts = _deep_tetra_with_implicit_h(right_terminal="Br")
        raw = _raw_tetra_for_center_zero()

        with self.assertRaisesRegex(SouthStarError, "specified closure"):
            build_ordinary_stereo_specified_closure(
                facts,
                raw_specified=raw,
                site_options=OrdinaryStereoSiteOptions(
                    ligand_equivalence="exact_stereochemical_graph_automorphism"
                ),
            )


def _deep_tetra_with_implicit_h(*, right_terminal: str) -> MoleculeFacts:
    facts = deep_tetra_ligand_facts(right_terminal=right_terminal)
    return replace(
        facts,
        atoms=(replace(facts.atoms[0], implicit_h_count=1),) + facts.atoms[1:],
    )


def _raw_tetra_for_center_zero() -> RawSpecifiedStereo:
    return RawSpecifiedStereo(
        tetrahedral=(
            RawTetraStereoRecord(
                center=AtomId(0),
                target=TetraValue.PLUS,
                reference_atoms=(AtomId(1), AtomId(2), AtomId(5), None),
            ),
        )
    )


if __name__ == "__main__":
    unittest.main()
