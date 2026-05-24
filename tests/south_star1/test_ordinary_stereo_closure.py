"""Tests for ordinary specified-stereo closure construction."""

from __future__ import annotations

from dataclasses import replace
import unittest

from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.errors import SouthStarErrorKind
from grimace._south_star1.facts import ComponentFacts
from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.facts import TetraValue
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import ComponentId
from grimace._south_star1.ordinary_stereo_closure import RawSpecifiedStereo
from grimace._south_star1.ordinary_stereo_closure import RawTetraStereoRecord
from grimace._south_star1.ordinary_stereo_closure import (
    build_ordinary_stereo_specified_closure,
)
from grimace._south_star1.ordinary_stereo_closure import (
    certify_ordinary_stereo_specified_closure,
)
from grimace._south_star1.ordinary_stereo_closure import raw_record_id
from grimace._south_star1.ordinary_stereo_sites import OrdinaryStereoSiteOptions

from tests.south_star1.helpers import atom
from tests.south_star1.helpers import deep_tetra_ligand_facts
from tests.south_star1.helpers import single_bond


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

    def test_specified_closure_accepts_dependency_chain(self) -> None:
        facts = _deep_tetra_with_implicit_h(right_terminal="Cl")

        result = build_ordinary_stereo_specified_closure(
            facts,
            raw_specified=_raw_tetra_for_center_zero(),
            site_options=OrdinaryStereoSiteOptions(
                ligand_equivalence="exact_graph_automorphism"
            ),
        )

        self.assertEqual(result.certificates[0].status, "promoted")

    def test_specified_closure_accepts_mutual_dependency(self) -> None:
        facts = _two_independent_tetra_facts()
        raw = _two_independent_tetra_raw()

        result = build_ordinary_stereo_specified_closure(
            facts,
            raw_specified=raw,
            site_options=OrdinaryStereoSiteOptions(
                ligand_equivalence="exact_graph_automorphism"
            ),
        )

        self.assertEqual(result.promoted_tetrahedral, 2)

    def test_specified_closure_rejects_one_sided_mutual_dependency(self) -> None:
        facts = _deep_tetra_with_implicit_h(right_terminal="Br")

        certificates = certify_ordinary_stereo_specified_closure(
            facts,
            raw_specified=_raw_tetra_for_center_zero(),
            site_options=OrdinaryStereoSiteOptions(
                ligand_equivalence="exact_stereochemical_graph_automorphism"
            ),
        )

        self.assertEqual(certificates[0].status, "rejected")

    def test_specified_closure_rejects_invalid_raw_record_as_whole(self) -> None:
        facts = _deep_tetra_with_implicit_h(right_terminal="Cl")
        raw = RawSpecifiedStereo(
            tetrahedral=(
                RawTetraStereoRecord(
                    center=AtomId(0),
                    target=TetraValue.NONE,
                    reference_atoms=(AtomId(1), AtomId(2), AtomId(5), None),
                ),
            )
        )

        with self.assertRaises(SouthStarError) as raised:
            build_ordinary_stereo_specified_closure(
                facts,
                raw_specified=raw,
                site_options=OrdinaryStereoSiteOptions(
                    ligand_equivalence="exact_graph_automorphism"
                ),
            )
        self.assertIs(raised.exception.kind, SouthStarErrorKind.INVALID_RAW_STEREO)

    def test_specified_closure_adds_unspecified_potential_sites_under_accepted_context(
        self,
    ) -> None:
        facts = _two_independent_tetra_facts()
        raw = RawSpecifiedStereo(
            tetrahedral=(_two_independent_tetra_raw().tetrahedral[0],)
        )

        result = build_ordinary_stereo_specified_closure(
            facts,
            raw_specified=raw,
            site_options=OrdinaryStereoSiteOptions(
                ligand_equivalence="exact_graph_automorphism"
            ),
        )

        self.assertEqual(result.promoted_tetrahedral, 1)
        self.assertGreaterEqual(result.potential_tetrahedral, 1)

    def test_specified_closure_is_idempotent(self) -> None:
        facts = _two_independent_tetra_facts()
        raw = _two_independent_tetra_raw()
        options = OrdinaryStereoSiteOptions(
            ligand_equivalence="exact_graph_automorphism"
        )

        first = build_ordinary_stereo_specified_closure(
            facts,
            raw_specified=raw,
            site_options=options,
        )
        second = build_ordinary_stereo_specified_closure(
            facts,
            raw_specified=raw,
            site_options=options,
        )

        self.assertEqual(
            tuple(site.center for site in first.facts.stereo.tetrahedral),
            tuple(site.center for site in second.facts.stereo.tetrahedral),
        )

    def test_specified_closure_is_raw_record_order_invariant(self) -> None:
        facts = _two_independent_tetra_facts()
        raw = _two_independent_tetra_raw()
        reversed_raw = RawSpecifiedStereo(tetrahedral=tuple(reversed(raw.tetrahedral)))
        options = OrdinaryStereoSiteOptions(
            ligand_equivalence="exact_graph_automorphism"
        )

        first = build_ordinary_stereo_specified_closure(
            facts,
            raw_specified=raw,
            site_options=options,
        )
        second = build_ordinary_stereo_specified_closure(
            facts,
            raw_specified=reversed_raw,
            site_options=options,
        )

        self.assertEqual(
            {site.center for site in first.facts.stereo.tetrahedral},
            {site.center for site in second.facts.stereo.tetrahedral},
        )

    def test_specified_closure_certificates_explain_promotions(self) -> None:
        facts = _deep_tetra_with_implicit_h(right_terminal="Cl")
        raw = _raw_tetra_for_center_zero()

        result = build_ordinary_stereo_specified_closure(
            facts,
            raw_specified=raw,
            site_options=OrdinaryStereoSiteOptions(
                ligand_equivalence="exact_graph_automorphism"
            ),
        )

        self.assertEqual(
            result.certificates[0].raw_record,
            raw_record_id(raw.tetrahedral[0]),
        )
        self.assertEqual(result.certificates[0].status, "promoted")
        self.assertIsNotNone(result.certificates[0].matched_site)

    def test_specified_closure_certificates_explain_rejections(self) -> None:
        facts = _deep_tetra_with_implicit_h(right_terminal="Br")

        certificates = certify_ordinary_stereo_specified_closure(
            facts,
            raw_specified=_raw_tetra_for_center_zero(),
            site_options=OrdinaryStereoSiteOptions(
                ligand_equivalence="exact_stereochemical_graph_automorphism"
            ),
        )

        self.assertEqual(certificates[0].status, "rejected")
        self.assertIn("specified closure", certificates[0].reason or "")


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


def _two_independent_tetra_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(
            replace(atom(0, "C"), implicit_h_count=1),
            atom(1, "F"),
            atom(2, "Cl"),
            atom(3, "Br"),
            replace(atom(4, "C"), implicit_h_count=1),
            atom(5, "F"),
            atom(6, "Cl"),
            atom(7, "Br"),
        ),
        bonds=(
            single_bond(0, 0, 1),
            single_bond(1, 0, 2),
            single_bond(2, 0, 3),
            single_bond(3, 4, 5),
            single_bond(4, 4, 6),
            single_bond(5, 4, 7),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1), AtomId(2), AtomId(3)),
                bonds=(BondId(0), BondId(1), BondId(2)),
            ),
            ComponentFacts(
                id=ComponentId(1),
                atoms=(AtomId(4), AtomId(5), AtomId(6), AtomId(7)),
                bonds=(BondId(3), BondId(4), BondId(5)),
            ),
        ),
    )


def _two_independent_tetra_raw() -> RawSpecifiedStereo:
    return RawSpecifiedStereo(
        tetrahedral=(
            RawTetraStereoRecord(
                center=AtomId(0),
                target=TetraValue.PLUS,
                reference_atoms=(AtomId(1), AtomId(2), AtomId(3), None),
            ),
            RawTetraStereoRecord(
                center=AtomId(4),
                target=TetraValue.MINUS,
                reference_atoms=(AtomId(5), AtomId(6), AtomId(7), None),
            ),
        )
    )


if __name__ == "__main__":
    unittest.main()
