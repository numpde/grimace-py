"""Focused tests for bounded directional non-neighbor writer support."""

from __future__ import annotations

import unittest
from dataclasses import replace

from grimace._south_star1.facts import BondOrder, ComponentFacts, DirectionalSiteFacts, DirectionalValue, LigandKind, LigandOccurrence, MoleculeFacts, SiteStatus, StereoFacts
from grimace._south_star1.ids import AtomId, BondId, ComponentId, OccurrenceId, SiteId
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions, SouthStarWriterSurface, prepare_south_star_mol_from_facts
from grimace._south_star1.writer_frontier import count_writer_cursor_completions, count_writer_frontier_support, initial_writer_frontier_cursor, iter_writer_frontier_support
from tests.south_star1.helpers import atom, bond, single_bond


def _options() -> SouthStarRuntimeOptions:
    return SouthStarRuntimeOptions(serialization_language=SerializationLanguageMode.WRITER_SHAPED)


def _prepared():
    site = SiteId(0)
    facts = MoleculeFacts(
        atoms=(replace(atom(0, "C"), implicit_h_count=1), atom(1, "C"), atom(2, "F"), atom(3, "Cl")),
        bonds=(bond(0, 0, 1, BondOrder.DOUBLE), single_bond(1, 0, 2), single_bond(2, 1, 3)),
        components=(ComponentFacts(id=ComponentId(0), atoms=(AtomId(0), AtomId(1), AtomId(2), AtomId(3)), bonds=(BondId(0), BondId(1), BondId(2))),),
        stereo=StereoFacts(directional=(DirectionalSiteFacts(id=site, center_bond=BondId(0), left_endpoint=AtomId(0), right_endpoint=AtomId(1), status=SiteStatus.SPECIFIED, target=DirectionalValue.OPPOSITE, left_ligands=(OccurrenceId(0), OccurrenceId(1)), right_ligands=(OccurrenceId(2),), reference_pair=(OccurrenceId(0), OccurrenceId(2))),)),
        ligand_occurrences=(
            LigandOccurrence(id=OccurrenceId(0), site=site, kind=LigandKind.NEIGHBOR_ATOM, atom=AtomId(2), bond=BondId(1)),
            LigandOccurrence(id=OccurrenceId(1), site=site, kind=LigandKind.IMPLICIT_H, atom=AtomId(0), bond=None),
            LigandOccurrence(id=OccurrenceId(2), site=site, kind=LigandKind.NEIGHBOR_ATOM, atom=AtomId(3), bond=BondId(2)),
        ),
    )
    return prepare_south_star_mol_from_facts(facts, writer_surface=SouthStarWriterSurface())


class BoundedDirectionalNonNeighborWriterTest(unittest.TestCase):
    def test_bounded_non_neighbor_directional_site_has_writer_support(self) -> None:
        prepared = _prepared()
        cursor = initial_writer_frontier_cursor(prepared, _options())
        strings = tuple(iter_writer_frontier_support(prepared, cursor))
        self.assertTrue(strings)
        self.assertEqual(len(strings), count_writer_frontier_support(prepared, cursor.support_state))
        self.assertGreater(count_writer_cursor_completions(prepared, cursor), 0)


if __name__ == "__main__":
    unittest.main()
