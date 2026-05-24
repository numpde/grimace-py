"""Shared hand-built fixtures for South Star 1 proof-kernel tests."""

from __future__ import annotations

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
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import ComponentId
from grimace._south_star1.ids import OccurrenceId
from grimace._south_star1.ids import SiteId
from grimace._south_star1.policy import AnnotationMode
from grimace._south_star1.policy import AtomTextChoice
from grimace._south_star1.policy import AtomTextDomain
from grimace._south_star1.policy import BondTextChoice
from grimace._south_star1.policy import RingLabel
from grimace._south_star1.policy import SmilesPolicy
from grimace._south_star1.policy import TetraToken


def cco_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, "C"), atom(1, "C"), atom(2, "O")),
        bonds=(
            single_bond(0, 0, 1),
            single_bond(1, 1, 2),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1), AtomId(2)),
                bonds=(BondId(0), BondId(1)),
            ),
        ),
    )


def cyclopropane_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, "C"), atom(1, "C"), atom(2, "C")),
        bonds=(
            single_bond(0, 0, 1),
            single_bond(1, 1, 2),
            single_bond(2, 2, 0),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1), AtomId(2)),
                bonds=(BondId(0), BondId(1), BondId(2)),
            ),
        ),
    )


def tetrahedral_facts() -> MoleculeFacts:
    site_id = SiteId(0)
    occurrence_ids = tuple(OccurrenceId(i) for i in range(4))
    return MoleculeFacts(
        atoms=(atom(0, "C"), atom(1, "F"), atom(2, "Cl"), atom(3, "Br")),
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
                    id=site_id,
                    center=AtomId(0),
                    status=SiteStatus.SPECIFIED,
                    target=TetraValue.PLUS,
                    ligand_occurrences=occurrence_ids,
                    reference_order=occurrence_ids,
                ),
            ),
        ),
        ligand_occurrences=(
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
        ),
    )


def directional_facts() -> MoleculeFacts:
    site_id = SiteId(0)
    return MoleculeFacts(
        atoms=(atom(0, "C"), atom(1, "C"), atom(2, "F"), atom(3, "Cl")),
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


def deep_tetra_ligand_facts(*, right_terminal: str) -> MoleculeFacts:
    """Tetra-candidate graph whose carbon ligands differ only distally."""

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


def deep_directional_endpoint_facts(*, right_terminal: str) -> MoleculeFacts:
    """Directional-candidate graph with same-endpoint deep ligand contrast."""

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


def symmetric_ring_center_facts() -> MoleculeFacts:
    """Central atom with two symmetry-equivalent ring ligand occurrences."""

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


def atom(idx: int, symbol: str) -> AtomFacts:
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


def single_bond(idx: int, a: int, b: int) -> BondFacts:
    return bond(idx, a, b, BondOrder.SINGLE)


def bond(idx: int, a: int, b: int, order: BondOrder) -> BondFacts:
    return BondFacts(
        id=BondId(idx),
        a=AtomId(a),
        b=AtomId(b),
        order=order,
        is_aromatic=False,
        is_conjugated=False,
    )


def organic_subset_policy(facts: MoleculeFacts) -> SmilesPolicy:
    return SmilesPolicy(
        ring_labels=(RingLabel(1), RingLabel(2)),
        annotation_mode=AnnotationMode.HARD,
        atom_text_domains=tuple(
            AtomTextDomain(
                atom=atom_facts.id,
                choices=(organic_atom_choice(atom_facts.symbol),),
            )
            for atom_facts in facts.atoms
        ),
        bond_text_domains=(),
    )


def organic_atom_choice(symbol: str) -> AtomTextChoice:
    return AtomTextChoice(
        name=f"organic_{symbol}",
        text_by_tetra=((TetraToken.NONE, symbol),),
    )


def empty_bond_choice() -> BondTextChoice:
    return BondTextChoice(name="elided_single", base_text="", permits_direction=False)
