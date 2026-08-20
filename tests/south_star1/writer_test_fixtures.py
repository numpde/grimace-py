"""Reusable writer facts and policy fixtures."""

from __future__ import annotations

from dataclasses import replace

from grimace._south_star1.facts import (BondOrder, ComponentFacts, DirectionalSiteFacts, DirectionalValue, LigandKind, LigandOccurrence, MoleculeFacts, SiteStatus, StereoFacts, TetraValue, TetrahedralSiteFacts)
from grimace._south_star1.ids import AtomId, BondId, ComponentId, OccurrenceId, SiteId
from grimace._south_star1.policy import AnnotationMode, AtomTextChoice, AtomTextDomain, BondTextChoice, BondTextDomain, RingLabel, SmilesPolicy, TetraToken
from tests.south_star1.helpers import atom, bond, single_bond

def directional_ring_carrier_facts() -> MoleculeFacts:
    site_id = SiteId(0)
    return MoleculeFacts(
        atoms=(
            atom(0, "C"),
            atom(1, "C"),
            atom(2, "F"),
            atom(3, "Cl"),
            atom(4, "Br"),
            atom(5, "O"),
        ),
        bonds=(
            bond(0, 0, 1, BondOrder.DOUBLE),
            single_bond(1, 1, 4),
            single_bond(2, 4, 2),
            single_bond(3, 2, 0),
            single_bond(4, 0, 3),
            single_bond(5, 1, 5),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=tuple(AtomId(index) for index in range(6)),
                bonds=tuple(BondId(index) for index in range(6)),
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
                    left_ligands=(OccurrenceId(0), OccurrenceId(1)),
                    right_ligands=(OccurrenceId(2), OccurrenceId(3)),
                    reference_pair=(OccurrenceId(0), OccurrenceId(2)),
                ),
            ),
        ),
        ligand_occurrences=(
            LigandOccurrence(
                id=OccurrenceId(0),
                site=site_id,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(2),
                bond=BondId(3),
            ),
            LigandOccurrence(
                id=OccurrenceId(1),
                site=site_id,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(3),
                bond=BondId(4),
            ),
            LigandOccurrence(
                id=OccurrenceId(2),
                site=site_id,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(4),
                bond=BondId(1),
            ),
            LigandOccurrence(
                id=OccurrenceId(3),
                site=site_id,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(5),
                bond=BondId(5),
            ),
        ),
    )

def directional_non_single_ring_carrier_facts() -> MoleculeFacts:
    facts = directional_ring_carrier_facts()
    bonds = list(facts.bonds)
    bonds[3] = replace(bonds[3], order=BondOrder.DOUBLE)
    return replace(facts, bonds=tuple(bonds))

def shared_directional_ring_carrier_facts() -> MoleculeFacts:
    left_site = SiteId(0)
    right_site = SiteId(1)
    return MoleculeFacts(
        atoms=tuple(atom(index, symbol) for index, symbol in (
            (0, "C"),
            (1, "C"),
            (2, "C"),
            (3, "C"),
            (4, "F"),
            (5, "Cl"),
            (6, "Br"),
            (7, "O"),
            (8, "F"),
            (9, "Cl"),
        )),
        bonds=(
            bond(0, 0, 1, BondOrder.DOUBLE),
            single_bond(1, 1, 2),
            bond(2, 2, 3, BondOrder.DOUBLE),
            single_bond(3, 3, 4),
            single_bond(4, 4, 5),
            single_bond(5, 5, 0),
            single_bond(6, 0, 6),
            single_bond(7, 1, 7),
            single_bond(8, 2, 8),
            single_bond(9, 3, 9),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=tuple(AtomId(index) for index in range(10)),
                bonds=tuple(BondId(index) for index in range(10)),
            ),
        ),
        stereo=StereoFacts(
            directional=(
                DirectionalSiteFacts(
                    id=left_site,
                    center_bond=BondId(0),
                    left_endpoint=AtomId(0),
                    right_endpoint=AtomId(1),
                    status=SiteStatus.SPECIFIED,
                    target=DirectionalValue.OPPOSITE,
                    left_ligands=(OccurrenceId(0), OccurrenceId(1)),
                    right_ligands=(OccurrenceId(2), OccurrenceId(3)),
                    reference_pair=(OccurrenceId(0), OccurrenceId(2)),
                ),
                DirectionalSiteFacts(
                    id=right_site,
                    center_bond=BondId(2),
                    left_endpoint=AtomId(2),
                    right_endpoint=AtomId(3),
                    status=SiteStatus.SPECIFIED,
                    target=DirectionalValue.OPPOSITE,
                    left_ligands=(OccurrenceId(4), OccurrenceId(5)),
                    right_ligands=(OccurrenceId(6), OccurrenceId(7)),
                    reference_pair=(OccurrenceId(4), OccurrenceId(6)),
                ),
            ),
        ),
        ligand_occurrences=(
            LigandOccurrence(
                id=OccurrenceId(0),
                site=left_site,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(5),
                bond=BondId(5),
            ),
            LigandOccurrence(
                id=OccurrenceId(1),
                site=left_site,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(6),
                bond=BondId(6),
            ),
            LigandOccurrence(
                id=OccurrenceId(2),
                site=left_site,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(2),
                bond=BondId(1),
            ),
            LigandOccurrence(
                id=OccurrenceId(3),
                site=left_site,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(7),
                bond=BondId(7),
            ),
            LigandOccurrence(
                id=OccurrenceId(4),
                site=right_site,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(1),
                bond=BondId(1),
            ),
            LigandOccurrence(
                id=OccurrenceId(5),
                site=right_site,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(8),
                bond=BondId(8),
            ),
            LigandOccurrence(
                id=OccurrenceId(6),
                site=right_site,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(4),
                bond=BondId(3),
            ),
            LigandOccurrence(
                id=OccurrenceId(7),
                site=right_site,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(9),
                bond=BondId(9),
            ),
        ),
    )

def terminal_tetra_center_facts() -> MoleculeFacts:
    site = SiteId(0)
    return MoleculeFacts(
        atoms=(
            atom(0, "F"),
            replace(atom(1, "C"), implicit_h_count=3),
        ),
        bonds=(single_bond(0, 0, 1),),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1)),
                bonds=(BondId(0),),
            ),
        ),
        stereo=StereoFacts(
            tetrahedral=(
                TetrahedralSiteFacts(
                    id=site,
                    center=AtomId(1),
                    status=SiteStatus.SPECIFIED,
                    target=TetraValue.PLUS,
                    ligand_occurrences=tuple(OccurrenceId(index) for index in range(4)),
                    reference_order=tuple(OccurrenceId(index) for index in range(4)),
                ),
            ),
        ),
        ligand_occurrences=(
            LigandOccurrence(
                id=OccurrenceId(0),
                site=site,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(0),
                bond=BondId(0),
            ),
            LigandOccurrence(
                id=OccurrenceId(1),
                site=site,
                kind=LigandKind.IMPLICIT_H,
                atom=AtomId(1),
                bond=None,
            ),
            LigandOccurrence(
                id=OccurrenceId(2),
                site=site,
                kind=LigandKind.IMPLICIT_H,
                atom=AtomId(1),
                bond=None,
            ),
            LigandOccurrence(
                id=OccurrenceId(3),
                site=site,
                kind=LigandKind.IMPLICIT_H,
                atom=AtomId(1),
                bond=None,
            ),
        ),
    )

def terminal_tetra_center_policy() -> SmilesPolicy:
    return SmilesPolicy(
        ring_labels=(RingLabel(1),),
        annotation_mode=AnnotationMode.HARD,
        atom_text_domains=(
            AtomTextDomain(
                atom=AtomId(0),
                choices=(AtomTextChoice("fluorine", ((TetraToken.NONE, "F"),)),),
            ),
            AtomTextDomain(
                atom=AtomId(1),
                choices=(
                    AtomTextChoice(
                        "terminal_tetra_carbon",
                        (
                            (TetraToken.AT, "[C@H3]"),
                            (TetraToken.ATAT, "[C@@H3]"),
                        ),
                    ),
                ),
            ),
        ),
        bond_text_domains=(
            BondTextDomain(
                bond=BondId(0),
                slot_kind="tree",
                choices=(BondTextChoice("single_elided", "", False),),
            ),
        ),
    )

def chain_facts(symbols: tuple[str, ...]) -> MoleculeFacts:
    return MoleculeFacts(
        atoms=tuple(atom(index, symbol) for index, symbol in enumerate(symbols)),
        bonds=tuple(
            single_bond(index, index, index + 1)
            for index in range(len(symbols) - 1)
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=tuple(AtomId(index) for index in range(len(symbols))),
                bonds=tuple(BondId(index) for index in range(len(symbols) - 1)),
            ),
        ),
    )

def duplicate_single_atom_policy() -> SmilesPolicy:
    return SmilesPolicy(
        ring_labels=(RingLabel(1),),
        annotation_mode=AnnotationMode.HARD,
        atom_text_domains=(
            AtomTextDomain(
                atom=AtomId(0),
                choices=(
                    AtomTextChoice(
                        name="carbon_a",
                        text_by_tetra=((TetraToken.NONE, "C"),),
                    ),
                    AtomTextChoice(
                        name="carbon_b",
                        text_by_tetra=((TetraToken.NONE, "C"),),
                    ),
                ),
            ),
        ),
        bond_text_domains=(),
    )
