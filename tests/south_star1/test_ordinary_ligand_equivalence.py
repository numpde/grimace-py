"""Tests for exact ordinary ligand equivalence."""

from __future__ import annotations

from dataclasses import replace
import unittest

from grimace._south_star1.facts import BondOrder
from grimace._south_star1.facts import ComponentFacts
from grimace._south_star1.facts import DirectionalSiteFacts
from grimace._south_star1.facts import DirectionalValue
from grimace._south_star1.facts import LigandKind
from grimace._south_star1.facts import LigandOccurrence
from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.facts import SiteStatus
from grimace._south_star1.facts import StereoFacts
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import ComponentId
from grimace._south_star1.ids import OccurrenceId
from grimace._south_star1.ids import SiteId
from grimace._south_star1.ordinary_ligand_equivalence import AutomorphismAnchor
from grimace._south_star1.ordinary_ligand_equivalence import LigandEquivalenceCache
from grimace._south_star1.ordinary_ligand_equivalence import LigandEquivalenceStats
from grimace._south_star1.ordinary_ligand_equivalence import (
    ligand_occurrences_equivalent,
)

from tests.south_star1.helpers import deep_directional_endpoint_facts
from tests.south_star1.helpers import deep_tetra_ligand_facts
from tests.south_star1.helpers import symmetric_ring_center_facts
from tests.south_star1.helpers import atom
from tests.south_star1.helpers import bond
from tests.south_star1.helpers import single_bond


class OrdinaryLigandEquivalenceTest(unittest.TestCase):
    def test_deep_distinct_tetra_ligands_are_not_equivalent(self) -> None:
        facts = deep_tetra_ligand_facts(right_terminal="Cl")

        self.assertFalse(
            ligand_occurrences_equivalent(
                facts,
                anchor=AutomorphismAnchor(fixed_atoms=frozenset({AtomId(0)})),
                left=_neighbor_occurrence(atom=2, bond_id=1),
                right=_neighbor_occurrence(atom=5, bond_id=3),
            )
        )

    def test_deep_duplicate_tetra_ligands_are_equivalent(self) -> None:
        facts = deep_tetra_ligand_facts(right_terminal="Br")

        self.assertTrue(
            ligand_occurrences_equivalent(
                facts,
                anchor=AutomorphismAnchor(fixed_atoms=frozenset({AtomId(0)})),
                left=_neighbor_occurrence(atom=2, bond_id=1),
                right=_neighbor_occurrence(atom=5, bond_id=3),
            )
        )

    def test_directional_endpoint_anchor_fixes_center_bond_and_endpoints(self) -> None:
        facts = deep_directional_endpoint_facts(right_terminal="Cl")

        self.assertFalse(
            ligand_occurrences_equivalent(
                facts,
                anchor=AutomorphismAnchor(
                    fixed_atoms=frozenset({AtomId(0), AtomId(1)}),
                    fixed_bonds=frozenset({BondId(0)}),
                ),
                left=_neighbor_occurrence(atom=2, bond_id=1),
                right=_neighbor_occurrence(atom=5, bond_id=3),
            )
        )

    def test_symmetric_ring_ligands_are_equivalent(self) -> None:
        facts = symmetric_ring_center_facts()

        self.assertTrue(
            ligand_occurrences_equivalent(
                facts,
                anchor=AutomorphismAnchor(fixed_atoms=frozenset({AtomId(0)})),
                left=_neighbor_occurrence(atom=1, bond_id=0),
                right=_neighbor_occurrence(atom=2, bond_id=1),
            )
        )

    def test_exact_equivalence_queries_can_share_cache(self) -> None:
        facts = deep_tetra_ligand_facts(right_terminal="Br")
        cache = LigandEquivalenceCache()
        kwargs = {
            "facts": facts,
            "anchor": AutomorphismAnchor(fixed_atoms=frozenset({AtomId(0)})),
            "left": _neighbor_occurrence(atom=2, bond_id=1),
            "right": _neighbor_occurrence(atom=5, bond_id=3),
            "cache": cache,
        }

        self.assertTrue(ligand_occurrences_equivalent(**kwargs))
        size_after_first = len(cache.by_key)
        self.assertTrue(ligand_occurrences_equivalent(**kwargs))

        self.assertGreater(size_after_first, 0)
        self.assertEqual(len(cache.by_key), size_after_first)

    def test_exact_equivalence_stats_observe_cache_and_search_work(self) -> None:
        facts = deep_tetra_ligand_facts(right_terminal="Br")
        cache = LigandEquivalenceCache()
        first = LigandEquivalenceStats()
        kwargs = {
            "facts": facts,
            "anchor": AutomorphismAnchor(fixed_atoms=frozenset({AtomId(0)})),
            "left": _neighbor_occurrence(atom=2, bond_id=1),
            "right": _neighbor_occurrence(atom=5, bond_id=3),
            "cache": cache,
        }

        self.assertTrue(ligand_occurrences_equivalent(**kwargs, stats=first))
        second = LigandEquivalenceStats()
        self.assertTrue(ligand_occurrences_equivalent(**kwargs, stats=second))

        self.assertEqual(first.cache_hits, 0)
        self.assertEqual(first.cache_misses, 1)
        self.assertGreater(first.atom_maps_considered, 0)
        self.assertGreater(first.complete_automorphisms_considered, 0)
        self.assertEqual(second.cache_hits, 1)
        self.assertEqual(second.cache_misses, 0)
        self.assertEqual(second.atom_maps_considered, 0)
        self.assertEqual(second.complete_automorphisms_considered, 0)

    def test_stereochemical_mode_distinguishes_remote_directional_stereo(self) -> None:
        facts = _remote_directional_ligand_facts(
            right_target=DirectionalValue.TOGETHER
        )
        kwargs = {
            "facts": facts,
            "anchor": AutomorphismAnchor(fixed_atoms=frozenset({AtomId(0)})),
            "left": _neighbor_occurrence(atom=1, bond_id=0),
            "right": _neighbor_occurrence(atom=5, bond_id=4),
        }

        self.assertTrue(ligand_occurrences_equivalent(**kwargs))
        self.assertFalse(
            ligand_occurrences_equivalent(
                **kwargs,
                stereo_mode="stereochemical",
            )
        )

    def test_cache_separates_graph_and_stereochemical_modes(self) -> None:
        facts = _remote_directional_ligand_facts(
            right_target=DirectionalValue.TOGETHER
        )
        cache = LigandEquivalenceCache()
        kwargs = {
            "facts": facts,
            "anchor": AutomorphismAnchor(fixed_atoms=frozenset({AtomId(0)})),
            "left": _neighbor_occurrence(atom=1, bond_id=0),
            "right": _neighbor_occurrence(atom=5, bond_id=4),
            "cache": cache,
        }

        self.assertTrue(ligand_occurrences_equivalent(**kwargs))
        self.assertFalse(
            ligand_occurrences_equivalent(
                **kwargs,
                stereo_mode="stereochemical",
            )
        )

        self.assertEqual(len(cache.by_key), 2)

    def test_cache_separates_stereo_ignore_site_sets(self) -> None:
        facts = _remote_directional_ligand_facts(
            right_target=DirectionalValue.TOGETHER
        )
        cache = LigandEquivalenceCache()
        kwargs = {
            "facts": facts,
            "anchor": AutomorphismAnchor(fixed_atoms=frozenset({AtomId(0)})),
            "left": _neighbor_occurrence(atom=1, bond_id=0),
            "right": _neighbor_occurrence(atom=5, bond_id=4),
            "stereo_mode": "stereochemical",
            "cache": cache,
        }

        self.assertFalse(ligand_occurrences_equivalent(**kwargs))
        self.assertTrue(
            ligand_occurrences_equivalent(
                **kwargs,
                ignore_site_ids=frozenset({SiteId(0), SiteId(1)}),
            )
        )

        self.assertEqual(len(cache.by_key), 2)

    def test_potential_unspecified_sites_do_not_distinguish_ligands(self) -> None:
        facts = _with_unspecified_directional_sites(
            _remote_directional_ligand_facts(
                right_target=DirectionalValue.TOGETHER
            )
        )

        self.assertTrue(
            ligand_occurrences_equivalent(
                facts=facts,
                anchor=AutomorphismAnchor(fixed_atoms=frozenset({AtomId(0)})),
                left=_neighbor_occurrence(atom=1, bond_id=0),
                right=_neighbor_occurrence(atom=5, bond_id=4),
                stereo_mode="stereochemical",
            )
        )

    def test_stereochemical_mode_preserves_identical_remote_directional_stereo(
        self,
    ) -> None:
        facts = _remote_directional_ligand_facts(
            right_target=DirectionalValue.OPPOSITE
        )

        self.assertTrue(
            ligand_occurrences_equivalent(
                facts=facts,
                anchor=AutomorphismAnchor(fixed_atoms=frozenset({AtomId(0)})),
                left=_neighbor_occurrence(atom=1, bond_id=0),
                right=_neighbor_occurrence(atom=5, bond_id=4),
                stereo_mode="stereochemical",
            )
        )


def _neighbor_occurrence(*, atom: int, bond_id: int) -> LigandOccurrence:
    return LigandOccurrence(
        id=OccurrenceId(-1),
        site=SiteId(-1),
        kind=LigandKind.NEIGHBOR_ATOM,
        atom=AtomId(atom),
        bond=BondId(bond_id),
    )


def _remote_directional_ligand_facts(
    *,
    right_target: DirectionalValue,
) -> MoleculeFacts:
    left_site = SiteId(0)
    right_site = SiteId(1)
    return MoleculeFacts(
        atoms=(
            atom(0, "C"),
            atom(1, "C"),
            atom(2, "C"),
            atom(3, "F"),
            atom(4, "Cl"),
            atom(5, "C"),
            atom(6, "C"),
            atom(7, "F"),
            atom(8, "Cl"),
        ),
        bonds=(
            single_bond(0, 0, 1),
            bond(1, 1, 2, BondOrder.DOUBLE),
            single_bond(2, 2, 3),
            single_bond(3, 1, 4),
            single_bond(4, 0, 5),
            bond(5, 5, 6, BondOrder.DOUBLE),
            single_bond(6, 6, 7),
            single_bond(7, 5, 8),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=tuple(AtomId(index) for index in range(9)),
                bonds=tuple(BondId(index) for index in range(8)),
            ),
        ),
        stereo=StereoFacts(
            directional=(
                DirectionalSiteFacts(
                    id=left_site,
                    center_bond=BondId(1),
                    left_endpoint=AtomId(1),
                    right_endpoint=AtomId(2),
                    status=SiteStatus.SPECIFIED,
                    target=DirectionalValue.OPPOSITE,
                    left_ligands=(OccurrenceId(0),),
                    right_ligands=(OccurrenceId(1),),
                    reference_pair=(OccurrenceId(0), OccurrenceId(1)),
                ),
                DirectionalSiteFacts(
                    id=right_site,
                    center_bond=BondId(5),
                    left_endpoint=AtomId(5),
                    right_endpoint=AtomId(6),
                    status=SiteStatus.SPECIFIED,
                    target=right_target,
                    left_ligands=(OccurrenceId(2),),
                    right_ligands=(OccurrenceId(3),),
                    reference_pair=(OccurrenceId(2), OccurrenceId(3)),
                ),
            ),
        ),
        ligand_occurrences=(
            LigandOccurrence(
                id=OccurrenceId(0),
                site=left_site,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(0),
                bond=BondId(0),
            ),
            LigandOccurrence(
                id=OccurrenceId(1),
                site=left_site,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(3),
                bond=BondId(2),
            ),
            LigandOccurrence(
                id=OccurrenceId(2),
                site=right_site,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(0),
                bond=BondId(4),
            ),
            LigandOccurrence(
                id=OccurrenceId(3),
                site=right_site,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(7),
                bond=BondId(6),
            ),
        ),
    )


def _with_unspecified_directional_sites(facts: MoleculeFacts) -> MoleculeFacts:
    return replace(
        facts,
        stereo=replace(
            facts.stereo,
            directional=tuple(
                replace(
                    site,
                    status=SiteStatus.UNSPECIFIED,
                    target=DirectionalValue.NONE,
                    reference_pair=None,
                )
                for site in facts.stereo.directional
            ),
        ),
    )


if __name__ == "__main__":
    unittest.main()
