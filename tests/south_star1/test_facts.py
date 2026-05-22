"""Tests for South Star 1 molecule facts and graph indexes."""

from __future__ import annotations

import dataclasses
import unittest

from grimace._south_star1.facts import AtomFacts
from grimace._south_star1.facts import BondFacts
from grimace._south_star1.facts import BondOrder
from grimace._south_star1.facts import ComponentFacts
from grimace._south_star1.facts import DirectionalSiteFacts
from grimace._south_star1.facts import DirectionalValue
from grimace._south_star1.facts import LigandKind
from grimace._south_star1.facts import LigandOccurrence
from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.facts import SiteStatus
from grimace._south_star1.facts import StereoFacts
from grimace._south_star1.facts import TetraValue
from grimace._south_star1.facts import TetrahedralSiteFacts
from grimace._south_star1.graph_index import build_graph_index
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import ComponentId
from grimace._south_star1.ids import OccurrenceId
from grimace._south_star1.ids import SiteId


class MoleculeFactsTest(unittest.TestCase):
    def test_valid_path_molecule_builds_graph_index(self) -> None:
        facts = _cco_facts()

        facts.validate()
        index = build_graph_index(facts)

        self.assertEqual(index.neighbors[AtomId(1)], (AtomId(0), AtomId(2)))
        self.assertEqual(index.incident_bonds[AtomId(1)], (BondId(0), BondId(1)))
        self.assertEqual(index.bond_between[(AtomId(0), AtomId(1))], BondId(0))
        self.assertEqual(index.bond_between[(AtomId(1), AtomId(2))], BondId(1))

    def test_fact_records_are_frozen(self) -> None:
        atom = _atom(0, "C")

        with self.assertRaises(dataclasses.FrozenInstanceError):
            atom.symbol = "N"  # type: ignore[misc]

    def test_duplicate_atom_ids_fail(self) -> None:
        facts = MoleculeFacts(
            atoms=(_atom(0, "C"), _atom(0, "O")),
            bonds=(),
            components=(ComponentFacts(ComponentId(0), (AtomId(0),), ()),),
        )

        with self.assertRaisesRegex(ValueError, "duplicate atom id"):
            facts.validate()

    def test_unknown_bond_endpoint_fails(self) -> None:
        facts = MoleculeFacts(
            atoms=(_atom(0, "C"),),
            bonds=(_single_bond(0, 0, 1),),
            components=(ComponentFacts(ComponentId(0), (AtomId(0),), (BondId(0),)),),
        )

        with self.assertRaisesRegex(ValueError, "unknown atom endpoint"):
            facts.validate()

    def test_component_atom_partition_mismatch_fails(self) -> None:
        facts = MoleculeFacts(
            atoms=(_atom(0, "C"), _atom(1, "O")),
            bonds=(),
            components=(ComponentFacts(ComponentId(0), (AtomId(0),), ()),),
        )

        with self.assertRaisesRegex(ValueError, "component atom partition mismatch"):
            facts.validate()

    def test_tetrahedral_reference_order_must_match_ligands(self) -> None:
        facts = _tetrahedral_facts(reference_order=(0, 1, 2, 99))

        with self.assertRaisesRegex(ValueError, "reference order is inconsistent"):
            facts.validate()

    def test_valid_tetrahedral_site_validates(self) -> None:
        _tetrahedral_facts().validate()

    def test_valid_directional_site_validates(self) -> None:
        _directional_facts().validate()

    def test_unused_ligand_occurrence_fails(self) -> None:
        site_id = SiteId(0)
        base = _tetrahedral_facts()
        facts = dataclasses.replace(
            base,
            ligand_occurrences=base.ligand_occurrences
            + (
                LigandOccurrence(
                    id=OccurrenceId(99),
                    site=site_id,
                    kind=LigandKind.PSEUDO,
                    atom=None,
                    bond=None,
                ),
            ),
        )

        with self.assertRaisesRegex(ValueError, "ligand occurrence coverage mismatch"):
            facts.validate()


def _cco_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(_atom(0, "C"), _atom(1, "C"), _atom(2, "O")),
        bonds=(
            _single_bond(0, 0, 1),
            _single_bond(1, 1, 2),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1), AtomId(2)),
                bonds=(BondId(0), BondId(1)),
            ),
        ),
    )


def _tetrahedral_facts(
    *,
    reference_order: tuple[int, int, int, int] = (0, 1, 2, 3),
) -> MoleculeFacts:
    site_id = SiteId(0)
    occurrence_ids = tuple(OccurrenceId(i) for i in range(4))
    return MoleculeFacts(
        atoms=(_atom(0, "C"), _atom(1, "F"), _atom(2, "Cl"), _atom(3, "Br")),
        bonds=(
            _single_bond(0, 0, 1),
            _single_bond(1, 0, 2),
            _single_bond(2, 0, 3),
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
                    id=site_id,
                    center=AtomId(0),
                    status=SiteStatus.SPECIFIED,
                    target=TetraValue.PLUS,
                    ligand_occurrences=occurrence_ids,
                    reference_order=tuple(OccurrenceId(i) for i in reference_order),
                ),
            ),
        ),
        ligand_occurrences=_tetrahedral_occurrences(site_id),
    )


def _directional_facts() -> MoleculeFacts:
    site_id = SiteId(0)
    return MoleculeFacts(
        atoms=(_atom(0, "C"), _atom(1, "C"), _atom(2, "F"), _atom(3, "Cl")),
        bonds=(
            _bond(0, 0, 1, BondOrder.DOUBLE),
            _single_bond(1, 0, 2),
            _single_bond(2, 1, 3),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1), AtomId(2), AtomId(3)),
                bonds=(BondId(0), BondId(1), BondId(2)),
            ),
        ),
        stereo=StereoFacts(
            directional=(
                DirectionalSiteFacts(
                    id=site_id,
                    center_bond=BondId(0),
                    left_endpoint=AtomId(0),
                    right_endpoint=AtomId(1),
                    status=SiteStatus.SPECIFIED,
                    target=DirectionalValue.OPPOSITE,
                    left_ligands=(OccurrenceId(0),),
                    right_ligands=(OccurrenceId(1),),
                    reference_pair=(OccurrenceId(0), OccurrenceId(1)),
                ),
            ),
        ),
        ligand_occurrences=(
            LigandOccurrence(
                id=OccurrenceId(0),
                site=site_id,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(2),
                bond=BondId(1),
            ),
            LigandOccurrence(
                id=OccurrenceId(1),
                site=site_id,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(3),
                bond=BondId(2),
            ),
        ),
    )


def _tetrahedral_occurrences(site_id: SiteId) -> tuple[LigandOccurrence, ...]:
    return (
        LigandOccurrence(
            id=OccurrenceId(0),
            site=site_id,
            kind=LigandKind.NEIGHBOR_ATOM,
            atom=AtomId(1),
            bond=BondId(0),
        ),
        LigandOccurrence(
            id=OccurrenceId(1),
            site=site_id,
            kind=LigandKind.NEIGHBOR_ATOM,
            atom=AtomId(2),
            bond=BondId(1),
        ),
        LigandOccurrence(
            id=OccurrenceId(2),
            site=site_id,
            kind=LigandKind.NEIGHBOR_ATOM,
            atom=AtomId(3),
            bond=BondId(2),
        ),
        LigandOccurrence(
            id=OccurrenceId(3),
            site=site_id,
            kind=LigandKind.IMPLICIT_H,
            atom=AtomId(0),
            bond=None,
        ),
    )


def _atom(idx: int, symbol: str) -> AtomFacts:
    return AtomFacts(
        id=AtomId(idx),
        atomic_num={"C": 6, "O": 8, "F": 9, "Cl": 17, "Br": 35}[symbol],
        symbol=symbol,
        isotope=None,
        formal_charge=0,
        is_aromatic=False,
        explicit_h_count=0,
        implicit_h_count=0,
        no_implicit=False,
    )


def _single_bond(idx: int, a: int, b: int) -> BondFacts:
    return _bond(idx, a, b, BondOrder.SINGLE)


def _bond(idx: int, a: int, b: int, order: BondOrder) -> BondFacts:
    return BondFacts(
        id=BondId(idx),
        a=AtomId(a),
        b=AtomId(b),
        order=order,
        is_aromatic=False,
        is_conjugated=False,
    )


if __name__ == "__main__":
    unittest.main()
