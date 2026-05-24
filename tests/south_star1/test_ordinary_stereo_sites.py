"""Tests for RDKit-free ordinary stereo-site construction."""

from __future__ import annotations

from dataclasses import replace
import unittest

from grimace._south_star1.facts import BondOrder
from grimace._south_star1.facts import ComponentFacts
from grimace._south_star1.facts import DirectionalValue
from grimace._south_star1.facts import LigandKind
from grimace._south_star1.facts import LigandOccurrence
from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.facts import SiteStatus
from grimace._south_star1.facts import StereoFacts
from grimace._south_star1.facts import TetraValue
from grimace._south_star1.facts import TetrahedralSiteFacts
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import ComponentId
from grimace._south_star1.ids import OccurrenceId
from grimace._south_star1.ids import SiteId
from grimace._south_star1.ordinary_stereo_sites import add_ordinary_potential_sites
from grimace._south_star1.ordinary_stereo_sites import OrdinaryStereoSiteOptions
from grimace._south_star1.ordinary_stereo_sites import (
    ordinary_directional_candidates,
)
from grimace._south_star1.ordinary_stereo_sites import (
    ordinary_tetrahedral_candidates,
)

from tests.south_star1.helpers import atom
from tests.south_star1.helpers import bond
from tests.south_star1.helpers import deep_directional_endpoint_facts
from tests.south_star1.helpers import deep_tetra_ligand_facts
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

    def test_ordinary_potential_tetra_sites_unique_by_center(self) -> None:
        facts = _combined_nonstereo_facts()

        sites, _ = ordinary_tetrahedral_candidates(facts)

        centers = tuple(site.center for site in sites)
        self.assertEqual(len(set(centers)), len(centers))

    def test_ordinary_potential_directional_sites_unique_by_center_bond(self) -> None:
        facts = _combined_nonstereo_facts()

        sites, _ = ordinary_directional_candidates(facts)

        center_bonds = tuple(site.center_bond for site in sites)
        self.assertEqual(len(set(center_bonds)), len(center_bonds))

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

    def test_exact_ligand_equivalence_admits_deep_distinct_tetra_site(self) -> None:
        facts = _deep_tetra_candidate_facts(right_terminal="Cl")
        exact = OrdinaryStereoSiteOptions(
            ligand_equivalence="exact_graph_automorphism"
        )

        immediate_sites, _ = ordinary_tetrahedral_candidates(facts)
        exact_sites, exact_occurrences = ordinary_tetrahedral_candidates(
            facts,
            exact,
        )

        self.assertEqual(immediate_sites, ())
        self.assertEqual(len(exact_sites), 1)
        self.assertEqual(exact_sites[0].center, AtomId(0))
        self.assertEqual(len(exact_occurrences), 4)

    def test_exact_ligand_equivalence_rejects_deep_duplicate_tetra_site(self) -> None:
        facts = _deep_tetra_candidate_facts(right_terminal="Br")
        exact = OrdinaryStereoSiteOptions(
            ligand_equivalence="exact_graph_automorphism"
        )

        sites, occurrences = ordinary_tetrahedral_candidates(facts, exact)

        self.assertEqual(sites, ())
        self.assertEqual(occurrences, ())

    def test_exact_ligand_equivalence_admits_deep_distinct_directional_site(
        self,
    ) -> None:
        facts = deep_directional_endpoint_facts(right_terminal="Cl")
        exact = OrdinaryStereoSiteOptions(
            ligand_equivalence="exact_graph_automorphism"
        )

        immediate_sites, _ = ordinary_directional_candidates(facts)
        exact_sites, exact_occurrences = ordinary_directional_candidates(
            facts,
            exact,
        )

        self.assertEqual(immediate_sites, ())
        self.assertEqual(len(exact_sites), 1)
        self.assertEqual(exact_sites[0].center_bond, BondId(0))
        self.assertEqual(len(exact_occurrences), 3)

    def test_exact_ligand_equivalence_rejects_symmetric_ring_ligands(self) -> None:
        facts = _symmetric_ring_tetra_candidate_facts()
        exact = OrdinaryStereoSiteOptions(
            ligand_equivalence="exact_graph_automorphism"
        )

        sites, occurrences = ordinary_tetrahedral_candidates(facts, exact)

        self.assertEqual(sites, ())
        self.assertEqual(occurrences, ())

    def test_stereochemical_equivalence_admits_remote_opposite_tetra_ligands(
        self,
    ) -> None:
        facts = _remote_tetra_ligand_facts(right_target=TetraValue.MINUS)
        graph_exact = OrdinaryStereoSiteOptions(
            ligand_equivalence="exact_graph_automorphism"
        )
        stereo_exact = OrdinaryStereoSiteOptions(
            ligand_equivalence="exact_stereochemical_graph_automorphism"
        )

        graph_augmented = add_ordinary_potential_sites(facts, options=graph_exact)
        stereo_augmented = add_ordinary_potential_sites(facts, options=stereo_exact)

        self.assertEqual(len(graph_augmented.stereo.tetrahedral), 2)
        self.assertEqual(len(stereo_augmented.stereo.tetrahedral), 3)
        self.assertIn(
            AtomId(0),
            {site.center for site in stereo_augmented.stereo.tetrahedral},
        )

    def test_stereochemical_equivalence_rejects_remote_identical_tetra_ligands(
        self,
    ) -> None:
        facts = _remote_tetra_ligand_facts(right_target=TetraValue.PLUS)
        stereo_exact = OrdinaryStereoSiteOptions(
            ligand_equivalence="exact_stereochemical_graph_automorphism"
        )

        augmented = add_ordinary_potential_sites(facts, options=stereo_exact)

        self.assertEqual(len(augmented.stereo.tetrahedral), 2)

    def test_stereochemical_equivalence_ignores_candidate_self_stereo(self) -> None:
        facts = _self_stereo_with_equivalent_ligands()
        stereo_exact = OrdinaryStereoSiteOptions(
            ligand_equivalence="exact_stereochemical_graph_automorphism"
        )

        augmented = add_ordinary_potential_sites(
            facts,
            preserve_specified=False,
            options=stereo_exact,
        )

        self.assertEqual(augmented.stereo.tetrahedral, facts.stereo.tetrahedral)


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


def _deep_tetra_candidate_facts(*, right_terminal: str) -> MoleculeFacts:
    base = deep_tetra_ligand_facts(right_terminal=right_terminal)
    return replace(
        base,
        atoms=(replace(base.atoms[0], implicit_h_count=1),) + base.atoms[1:],
    )


def _symmetric_ring_tetra_candidate_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(
            replace(atom(0, "C"), implicit_h_count=1),
            atom(1, "C"),
            atom(2, "C"),
            atom(3, "O"),
            atom(4, "F"),
        ),
        bonds=(
            single_bond(0, 0, 1),
            single_bond(1, 0, 2),
            single_bond(2, 1, 3),
            single_bond(3, 2, 3),
            single_bond(4, 0, 4),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=tuple(AtomId(index) for index in range(5)),
                bonds=tuple(BondId(index) for index in range(5)),
            ),
        ),
    )


def _remote_tetra_ligand_facts(*, right_target: TetraValue) -> MoleculeFacts:
    left_site = SiteId(0)
    right_site = SiteId(1)
    return MoleculeFacts(
        atoms=(
            replace(atom(0, "C"), implicit_h_count=1),
            atom(1, "F"),
            atom(2, "C"),
            atom(3, "F"),
            atom(4, "Cl"),
            atom(5, "Br"),
            atom(6, "C"),
            atom(7, "F"),
            atom(8, "Cl"),
            atom(9, "Br"),
        ),
        bonds=(
            single_bond(0, 0, 1),
            single_bond(1, 0, 2),
            single_bond(2, 2, 3),
            single_bond(3, 2, 4),
            single_bond(4, 2, 5),
            single_bond(5, 0, 6),
            single_bond(6, 6, 7),
            single_bond(7, 6, 8),
            single_bond(8, 6, 9),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=tuple(AtomId(index) for index in range(10)),
                bonds=tuple(BondId(index) for index in range(9)),
            ),
        ),
        stereo=StereoFacts(
            tetrahedral=(
                _remote_tetra_site(
                    site=left_site,
                    center=2,
                    target=TetraValue.PLUS,
                    occurrence_offset=0,
                    ligands=((0, 1), (3, 2), (4, 3), (5, 4)),
                ),
                _remote_tetra_site(
                    site=right_site,
                    center=6,
                    target=right_target,
                    occurrence_offset=4,
                    ligands=((0, 5), (7, 6), (8, 7), (9, 8)),
                ),
            ),
        ),
        ligand_occurrences=(
            _neighbor(left_site, 0, atom=0, bond=1),
            _neighbor(left_site, 1, atom=3, bond=2),
            _neighbor(left_site, 2, atom=4, bond=3),
            _neighbor(left_site, 3, atom=5, bond=4),
            _neighbor(right_site, 4, atom=0, bond=5),
            _neighbor(right_site, 5, atom=7, bond=6),
            _neighbor(right_site, 6, atom=8, bond=7),
            _neighbor(right_site, 7, atom=9, bond=8),
        ),
    )


def _remote_tetra_site(
    *,
    site: SiteId,
    center: int,
    target: TetraValue,
    occurrence_offset: int,
    ligands: tuple[tuple[int, int], ...],
):
    occurrence_ids = tuple(
        OccurrenceId(occurrence_offset + index)
        for index, _ligand in enumerate(ligands)
    )
    return TetrahedralSiteFacts(
        id=site,
        center=AtomId(center),
        status=SiteStatus.SPECIFIED,
        target=target,
        ligand_occurrences=occurrence_ids,
        reference_order=occurrence_ids,
    )


def _self_stereo_with_equivalent_ligands() -> MoleculeFacts:
    site = SiteId(0)
    occurrence_ids = tuple(OccurrenceId(index) for index in range(4))
    return MoleculeFacts(
        atoms=(
            replace(atom(0, "C"), implicit_h_count=1),
            atom(1, "C"),
            atom(2, "C"),
            atom(3, "F"),
        ),
        bonds=(
            single_bond(0, 0, 1),
            single_bond(1, 0, 2),
            single_bond(2, 0, 3),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1), AtomId(2), AtomId(3)),
                bonds=(BondId(0), BondId(1), BondId(2)),
            ),
        ),
        stereo=StereoFacts(
            tetrahedral=(
                TetrahedralSiteFacts(
                    id=site,
                    center=AtomId(0),
                    status=SiteStatus.SPECIFIED,
                    target=TetraValue.PLUS,
                    ligand_occurrences=occurrence_ids,
                    reference_order=occurrence_ids,
                ),
            ),
        ),
        ligand_occurrences=(
            _neighbor(site, 0, atom=1, bond=0),
            _neighbor(site, 1, atom=2, bond=1),
            _neighbor(site, 2, atom=3, bond=2),
            LigandOccurrence(
                id=OccurrenceId(3),
                site=site,
                kind=LigandKind.IMPLICIT_H,
                atom=AtomId(0),
                bond=None,
            ),
        ),
    )


def _neighbor(site: SiteId, idx: int, *, atom: int, bond: int) -> LigandOccurrence:
    return LigandOccurrence(
        id=OccurrenceId(idx),
        site=site,
        kind=LigandKind.NEIGHBOR_ATOM,
        atom=AtomId(atom),
        bond=BondId(bond),
    )


if __name__ == "__main__":
    unittest.main()
