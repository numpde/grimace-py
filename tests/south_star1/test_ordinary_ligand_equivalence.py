"""Tests for exact ordinary ligand equivalence."""

from __future__ import annotations

import unittest

from grimace._south_star1.facts import LigandKind
from grimace._south_star1.facts import LigandOccurrence
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import OccurrenceId
from grimace._south_star1.ids import SiteId
from grimace._south_star1.ordinary_ligand_equivalence import AutomorphismAnchor
from grimace._south_star1.ordinary_ligand_equivalence import (
    ligand_occurrences_equivalent,
)

from tests.south_star1.helpers import deep_directional_endpoint_facts
from tests.south_star1.helpers import deep_tetra_ligand_facts
from tests.south_star1.helpers import symmetric_ring_center_facts


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


def _neighbor_occurrence(*, atom: int, bond_id: int) -> LigandOccurrence:
    return LigandOccurrence(
        id=OccurrenceId(-1),
        site=SiteId(-1),
        kind=LigandKind.NEIGHBOR_ATOM,
        atom=AtomId(atom),
        bond=BondId(bond_id),
    )


if __name__ == "__main__":
    unittest.main()
