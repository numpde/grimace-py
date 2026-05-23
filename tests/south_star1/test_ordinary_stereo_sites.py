"""Tests for RDKit-free ordinary stereo-site construction."""

from __future__ import annotations

from dataclasses import replace
import unittest

from grimace._south_star1.facts import BondOrder
from grimace._south_star1.facts import ComponentFacts
from grimace._south_star1.facts import DirectionalValue
from grimace._south_star1.facts import LigandKind
from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.facts import SiteStatus
from grimace._south_star1.facts import StereoFacts
from grimace._south_star1.facts import TetraValue
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import ComponentId
from grimace._south_star1.ordinary_stereo_sites import add_ordinary_potential_sites
from grimace._south_star1.ordinary_stereo_sites import (
    ordinary_directional_candidates,
)
from grimace._south_star1.ordinary_stereo_sites import (
    ordinary_tetrahedral_candidates,
)

from tests.south_star1.helpers import atom
from tests.south_star1.helpers import bond
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import single_bond
from tests.south_star1.helpers import tetrahedral_facts


class OrdinaryStereoSitesTest(unittest.TestCase):
    def test_tetrahedral_candidates_are_unspecified_carbon_sites(self) -> None:
        facts = _tetrahedral_carbon_without_stereo()

        sites, occurrences = ordinary_tetrahedral_candidates(facts)

        self.assertEqual(len(sites), 1)
        self.assertEqual(sites[0].center, AtomId(0))
        self.assertEqual(sites[0].status, SiteStatus.UNSPECIFIED)
        self.assertEqual(sites[0].target, TetraValue.NONE)
        self.assertEqual(sites[0].ligand_occurrences, sites[0].reference_order)
        self.assertEqual(len(occurrences), 4)
        self.assertEqual(
            sum(
                occurrence.kind is LigandKind.IMPLICIT_H
                for occurrence in occurrences
            ),
            1,
        )

    def test_directional_candidates_are_unspecified_double_bond_sites(self) -> None:
        facts = _directional_without_stereo()

        sites, occurrences = ordinary_directional_candidates(facts)

        self.assertEqual(len(sites), 1)
        self.assertEqual(sites[0].center_bond, BondId(0))
        self.assertEqual(sites[0].status, SiteStatus.UNSPECIFIED)
        self.assertEqual(sites[0].target, DirectionalValue.NONE)
        self.assertIsNone(sites[0].reference_pair)
        self.assertEqual(
            sites[0].left_ligands + sites[0].right_ligands,
            tuple(occurrence.id for occurrence in occurrences),
        )

    def test_add_potential_sites_preserves_existing_specified_sites(self) -> None:
        facts = directional_facts()

        augmented = add_ordinary_potential_sites(facts)

        self.assertEqual(augmented.stereo.directional, facts.stereo.directional)
        self.assertEqual(augmented.ligand_occurrences, facts.ligand_occurrences)

    def test_add_potential_sites_adds_no_accidental_stereo_universe(self) -> None:
        facts = _combined_nonstereo_facts()

        augmented = add_ordinary_potential_sites(facts)

        self.assertEqual(len(augmented.stereo.tetrahedral), 1)
        self.assertEqual(len(augmented.stereo.directional), 1)
        self.assertEqual(augmented.stereo.tetrahedral[0].target, TetraValue.NONE)
        self.assertEqual(
            augmented.stereo.directional[0].target,
            DirectionalValue.NONE,
        )
        augmented.validate()

    def test_implicit_h_ligands_are_included_for_directional_sites(self) -> None:
        facts = _implicit_h_alkene_without_stereo()

        sites, occurrences = ordinary_directional_candidates(facts)
        occurrence_by_id = {occurrence.id: occurrence for occurrence in occurrences}

        self.assertEqual(len(sites), 1)
        self.assertTrue(
            any(
                occurrence_by_id[occurrence_id].kind is LigandKind.IMPLICIT_H
                for occurrence_id in sites[0].left_ligands
            )
        )

    def test_nonstereogenic_tetrahedral_shapes_are_not_candidates(self) -> None:
        for facts in (
            _ch3cl_facts(),
            _ch2cl2_facts(),
            _duplicate_f_tetrahedral_facts(),
        ):
            sites, occurrences = ordinary_tetrahedral_candidates(facts)

            self.assertEqual(sites, ())
            self.assertEqual(occurrences, ())

    def test_nonstereogenic_directional_shapes_are_not_candidates(self) -> None:
        for facts in (
            _ethene_facts(),
            _duplicate_f_alkene_facts(),
        ):
            sites, occurrences = ordinary_directional_candidates(facts)

            self.assertEqual(sites, ())
            self.assertEqual(occurrences, ())


def _tetrahedral_carbon_without_stereo() -> MoleculeFacts:
    base = tetrahedral_facts()
    return replace(
        base,
        atoms=(replace(base.atoms[0], implicit_h_count=1),) + base.atoms[1:],
        stereo=StereoFacts(),
        ligand_occurrences=(),
    )


def _directional_without_stereo() -> MoleculeFacts:
    return replace(
        directional_facts(),
        stereo=StereoFacts(),
        ligand_occurrences=(),
    )


def _combined_nonstereo_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(
            atom(0, "C"),
            atom(1, "F"),
            atom(2, "Cl"),
            atom(3, "Br"),
            atom(4, "O"),
            atom(5, "C"),
            atom(6, "C"),
            atom(7, "F"),
            atom(8, "Cl"),
        ),
        bonds=(
            single_bond(0, 0, 1),
            single_bond(1, 0, 2),
            single_bond(2, 0, 3),
            single_bond(3, 0, 4),
            bond(4, 5, 6, BondOrder.DOUBLE),
            single_bond(5, 5, 7),
            single_bond(6, 6, 8),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1), AtomId(2), AtomId(3), AtomId(4)),
                bonds=(BondId(0), BondId(1), BondId(2), BondId(3)),
            ),
            ComponentFacts(
                id=ComponentId(1),
                atoms=(AtomId(5), AtomId(6), AtomId(7), AtomId(8)),
                bonds=(BondId(4), BondId(5), BondId(6)),
            ),
        ),
    )


def _implicit_h_alkene_without_stereo() -> MoleculeFacts:
    left = replace(atom(0, "C"), implicit_h_count=1)
    return MoleculeFacts(
        atoms=(left, atom(1, "C"), atom(2, "F"), atom(3, "Cl")),
        bonds=(
            bond(0, 0, 1, BondOrder.DOUBLE),
            single_bond(1, 0, 2),
            single_bond(2, 1, 3),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1), AtomId(2), AtomId(3)),
                bonds=(BondId(0), BondId(1), BondId(2)),
            ),
        ),
    )


def _ch3cl_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(replace(atom(0, "C"), implicit_h_count=3), atom(1, "Cl")),
        bonds=(single_bond(0, 0, 1),),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1)),
                bonds=(BondId(0),),
            ),
        ),
    )


def _ch2cl2_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(
            replace(atom(0, "C"), implicit_h_count=2),
            atom(1, "Cl"),
            atom(2, "Cl"),
        ),
        bonds=(single_bond(0, 0, 1), single_bond(1, 0, 2)),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1), AtomId(2)),
                bonds=(BondId(0), BondId(1)),
            ),
        ),
    )


def _duplicate_f_tetrahedral_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(
            atom(0, "C"),
            atom(1, "F"),
            atom(2, "F"),
            atom(3, "Cl"),
            atom(4, "Br"),
        ),
        bonds=(
            single_bond(0, 0, 1),
            single_bond(1, 0, 2),
            single_bond(2, 0, 3),
            single_bond(3, 0, 4),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1), AtomId(2), AtomId(3), AtomId(4)),
                bonds=(BondId(0), BondId(1), BondId(2), BondId(3)),
            ),
        ),
    )


def _ethene_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(
            replace(atom(0, "C"), implicit_h_count=2),
            replace(atom(1, "C"), implicit_h_count=2),
        ),
        bonds=(bond(0, 0, 1, BondOrder.DOUBLE),),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1)),
                bonds=(BondId(0),),
            ),
        ),
    )


def _duplicate_f_alkene_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(
            atom(0, "C"),
            atom(1, "C"),
            atom(2, "F"),
            atom(3, "F"),
            atom(4, "Cl"),
            atom(5, "Br"),
        ),
        bonds=(
            bond(0, 0, 1, BondOrder.DOUBLE),
            single_bond(1, 0, 2),
            single_bond(2, 0, 3),
            single_bond(3, 1, 4),
            single_bond(4, 1, 5),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(
                    AtomId(0),
                    AtomId(1),
                    AtomId(2),
                    AtomId(3),
                    AtomId(4),
                    AtomId(5),
                ),
                bonds=(
                    BondId(0),
                    BondId(1),
                    BondId(2),
                    BondId(3),
                    BondId(4),
                ),
            ),
        ),
    )


if __name__ == "__main__":
    unittest.main()
