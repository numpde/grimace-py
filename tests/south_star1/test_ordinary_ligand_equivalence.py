"""Tests for exact ordinary ligand equivalence."""

from __future__ import annotations

import unittest

from grimace._south_star1.facts import BondOrder
from grimace._south_star1.facts import ComponentFacts
from grimace._south_star1.facts import LigandKind
from grimace._south_star1.facts import LigandOccurrence
from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import ComponentId
from grimace._south_star1.ids import OccurrenceId
from grimace._south_star1.ids import SiteId
from grimace._south_star1.ordinary_ligand_equivalence import AutomorphismAnchor
from grimace._south_star1.ordinary_ligand_equivalence import (
    ligand_occurrences_equivalent,
)

from tests.south_star1.helpers import atom
from tests.south_star1.helpers import bond
from tests.south_star1.helpers import single_bond


class OrdinaryLigandEquivalenceTest(unittest.TestCase):
    def test_deep_distinct_tetra_ligands_are_not_equivalent(self) -> None:
        facts = _branched_center_facts(right_terminal="Cl")

        self.assertFalse(
            ligand_occurrences_equivalent(
                facts,
                anchor=AutomorphismAnchor(fixed_atoms=frozenset({AtomId(0)})),
                left=_neighbor_occurrence(atom=2, bond_id=1),
                right=_neighbor_occurrence(atom=5, bond_id=3),
            )
        )

    def test_deep_duplicate_tetra_ligands_are_equivalent(self) -> None:
        facts = _branched_center_facts(right_terminal="Br")

        self.assertTrue(
            ligand_occurrences_equivalent(
                facts,
                anchor=AutomorphismAnchor(fixed_atoms=frozenset({AtomId(0)})),
                left=_neighbor_occurrence(atom=2, bond_id=1),
                right=_neighbor_occurrence(atom=5, bond_id=3),
            )
        )

    def test_directional_endpoint_anchor_fixes_center_bond_and_endpoints(self) -> None:
        facts = _directional_deep_endpoint_facts(right_terminal="Cl")

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
        facts = _symmetric_ring_center_facts()

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


def _branched_center_facts(*, right_terminal: str) -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(
            atom(0, "C"),
            atom(1, "F"),
            atom(2, "C"),
            atom(3, "C"),
            atom(4, "Br"),
            atom(5, "C"),
            atom(6, "C"),
            atom(7, right_terminal),
        ),
        bonds=(
            single_bond(0, 0, 1),
            single_bond(1, 0, 2),
            single_bond(2, 2, 3),
            single_bond(3, 0, 5),
            single_bond(4, 5, 6),
            single_bond(5, 3, 4),
            single_bond(6, 6, 7),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=tuple(AtomId(index) for index in range(8)),
                bonds=tuple(BondId(index) for index in range(7)),
            ),
        ),
    )


def _directional_deep_endpoint_facts(*, right_terminal: str) -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(
            atom(0, "C"),
            atom(1, "C"),
            atom(2, "C"),
            atom(3, "C"),
            atom(4, "Br"),
            atom(5, "C"),
            atom(6, "C"),
            atom(7, right_terminal),
            atom(8, "F"),
        ),
        bonds=(
            bond(0, 0, 1, BondOrder.DOUBLE),
            single_bond(1, 0, 2),
            single_bond(2, 2, 3),
            single_bond(3, 0, 5),
            single_bond(4, 5, 6),
            single_bond(5, 3, 4),
            single_bond(6, 6, 7),
            single_bond(7, 1, 8),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=tuple(AtomId(index) for index in range(9)),
                bonds=tuple(BondId(index) for index in range(8)),
            ),
        ),
    )


def _symmetric_ring_center_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, "C"), atom(1, "C"), atom(2, "C"), atom(3, "O")),
        bonds=(
            single_bond(0, 0, 1),
            single_bond(1, 0, 2),
            single_bond(2, 1, 3),
            single_bond(3, 2, 3),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1), AtomId(2), AtomId(3)),
                bonds=(BondId(0), BondId(1), BondId(2), BondId(3)),
            ),
        ),
    )


if __name__ == "__main__":
    unittest.main()
