"""Shared hand-built fixtures for South Star 1 proof-kernel tests."""

from __future__ import annotations

from dataclasses import replace

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


def simple_monocycle_with_pendant_forest_facts(
    *,
    ring_size: int,
    pendant_paths: tuple[int, ...] = (),
    ring_bond_orders: tuple[BondOrder, ...] | None = None,
    pendant_path_bond_orders:
        tuple[tuple[BondOrder, ...], ...] | None = None,
) -> MoleculeFacts:
    ring_atoms = tuple(range(ring_size))
    atoms = tuple(atom(atom_id, "C") for atom_id in ring_atoms)
    bonds: tuple[BondFacts, ...] = ()

    next_atom_id = len(atoms)
    next_bond_id = 0

    if ring_bond_orders is None:
        ring_bond_orders = (BondOrder.SINGLE,) * ring_size

    if len(ring_bond_orders) != ring_size:
        raise AssertionError("ring_bond_orders must match ring_size")

    if pendant_path_bond_orders is None:
        pendant_path_bond_orders = tuple(
            () for _ in range(len(pendant_paths))
        )

    if len(pendant_path_bond_orders) != len(pendant_paths):
        raise AssertionError(
            "pendant_path_bond_orders must match pendant_paths length",
        )

    def next_bond(left: int, right: int, *, order: BondOrder) -> None:
        nonlocal next_bond_id, bonds
        bonds = (
            *bonds,
            bond(next_bond_id, left, right, order),
        )
        next_bond_id += 1

    for left in range(ring_size):
        right = (left + 1) % ring_size
        next_bond(left, right, order=ring_bond_orders[left])

    for pendant_index, length in enumerate(pendant_paths):
        current_atom = ring_atoms[pendant_index % ring_size]
        pendant_orders = (
            pendant_path_bond_orders[pendant_index]
            if pendant_path_bond_orders
            else ()
        )
        if not pendant_orders:
            pendant_orders = (BondOrder.SINGLE,) * length

        if len(pendant_orders) != length:
            raise AssertionError(
                "pendant_path_bond_orders must match pendant path lengths",
            )

        for _step in range(length):
            new_atom_id = next_atom_id
            next_atom_id += 1
            atoms = (*atoms, atom(new_atom_id, "C"))
            next_bond(
                current_atom,
                new_atom_id,
                order=pendant_orders[_step],
            )
            current_atom = new_atom_id

    component_atom_ids = tuple(AtomId(atom_id) for atom_id in range(next_atom_id))
    component_bond_ids = tuple(BondId(index) for index in range(next_bond_id))

    return MoleculeFacts(
        atoms=atoms,
        bonds=tuple(bonds),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=component_atom_ids,
                bonds=component_bond_ids,
            ),
        ),
    )


def cyclopropane_facts() -> MoleculeFacts:
    return simple_monocycle_with_pendant_forest_facts(
        ring_size=3,
    )


def methylcyclopropane_facts() -> MoleculeFacts:
    return simple_monocycle_with_pendant_forest_facts(
        ring_size=3,
        pendant_paths=(1,),
    )


def ethylcyclopropane_facts() -> MoleculeFacts:
    return simple_monocycle_with_pendant_forest_facts(
        ring_size=3,
        pendant_paths=(2,),
    )


def dimethylcyclopropane_facts() -> MoleculeFacts:
    return simple_monocycle_with_pendant_forest_facts(
        ring_size=3,
        pendant_paths=(1, 1),
    )


def tetrahedral_facts() -> MoleculeFacts:
    site_id = SiteId(0)
    occurrence_ids = tuple(OccurrenceId(i) for i in range(4))
    return MoleculeFacts(
        atoms=(
            replace(atom(0, "C"), implicit_h_count=1),
            atom(1, "F"),
            atom(2, "Cl"),
            atom(3, "Br"),
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
    """Return the synthetic zero-H directional proof-kernel fixture.

    This intentionally differs from ordinary RDKit F/C=C/Cl ingestion, whose
    alkene endpoints each carry an implicit-H ligand occurrence.
    """
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


def shared_acyclic_directional_facts() -> MoleculeFacts:
    left_site = SiteId(0)
    right_site = SiteId(1)
    return MoleculeFacts(
        atoms=(
            atom(0, "F"),
            atom(1, "C"),
            atom(2, "C"),
            atom(3, "C"),
            atom(4, "C"),
            atom(5, "Cl"),
        ),
        bonds=(
            single_bond(0, 0, 1),
            bond(1, 1, 2, BondOrder.DOUBLE),
            single_bond(2, 2, 3),
            bond(3, 3, 4, BondOrder.DOUBLE),
            single_bond(4, 4, 5),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=tuple(AtomId(index) for index in range(6)),
                bonds=tuple(BondId(index) for index in range(5)),
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
                    center_bond=BondId(3),
                    left_endpoint=AtomId(3),
                    right_endpoint=AtomId(4),
                    status=SiteStatus.SPECIFIED,
                    target=DirectionalValue.OPPOSITE,
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
                atom=AtomId(2),
                bond=BondId(2),
            ),
            LigandOccurrence(
                id=OccurrenceId(3),
                site=right_site,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(5),
                bond=BondId(4),
            ),
        ),
    )


def four_substituent_directional_facts() -> MoleculeFacts:
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
            single_bond(1, 0, 2),
            single_bond(2, 0, 3),
            single_bond(3, 1, 4),
            single_bond(4, 1, 5),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=tuple(AtomId(index) for index in range(6)),
                bonds=tuple(BondId(index) for index in range(5)),
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
                    target=DirectionalValue.TOGETHER,
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
                bond=BondId(1),
            ),
            LigandOccurrence(
                id=OccurrenceId(1),
                site=site_id,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(3),
                bond=BondId(2),
            ),
            LigandOccurrence(
                id=OccurrenceId(2),
                site=site_id,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(4),
                bond=BondId(3),
            ),
            LigandOccurrence(
                id=OccurrenceId(3),
                site=site_id,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(5),
                bond=BondId(4),
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
