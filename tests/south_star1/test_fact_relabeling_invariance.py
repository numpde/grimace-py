"""RDKit-free support invariance under fact-id relabeling."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import unittest

from grimace._south_star1.facts import AtomFacts
from grimace._south_star1.facts import BondFacts
from grimace._south_star1.facts import BondOrder
from grimace._south_star1.facts import ComponentFacts
from grimace._south_star1.facts import LigandKind
from grimace._south_star1.facts import LigandOccurrence
from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.facts import StereoFacts
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import ComponentId
from grimace._south_star1.ids import OccurrenceId
from grimace._south_star1.ids import SiteId
from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.ordinary_semantics import OrdinarySmilesSemantics
from grimace._south_star1.ordinary_stereo_sites import add_ordinary_potential_sites
from grimace._south_star1.ordinary_stereo_sites import OrdinaryStereoSiteOptions
from grimace._south_star1.support_enumeration import enumerate_exhaustive_stereo_support

from tests.south_star1.helpers import deep_directional_endpoint_facts
from tests.south_star1.helpers import deep_tetra_ligand_facts
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import tetrahedral_facts


class FactRelabelingInvarianceTest(unittest.TestCase):
    def test_connected_support_is_invariant_under_fact_relabeling(self) -> None:
        cases = (
            tetrahedral_facts(),
            directional_facts(),
            _ring_tetrahedral_facts(),
            _mixed_tetra_directional_facts(),
        )

        for facts in cases:
            with self.subTest(atom_count=len(facts.atoms), bond_count=len(facts.bonds)):
                relabeled = relabel_molecule_facts(facts, _reverse_relabeling(facts))

                self.assertEqual(_support_set(facts), _support_set(relabeled))

    def test_exact_equivalence_support_is_invariant_under_fact_relabeling(self) -> None:
        exact = OrdinaryStereoSiteOptions(
            ligand_equivalence="exact_graph_automorphism"
        )
        cases = (
            _exact_tetra_candidate_facts(),
            deep_directional_endpoint_facts(right_terminal="Cl"),
        )

        for facts in cases:
            with self.subTest(atom_count=len(facts.atoms), bond_count=len(facts.bonds)):
                original = add_ordinary_potential_sites(facts, options=exact)
                relabeled = add_ordinary_potential_sites(
                    relabel_molecule_facts(facts, _reverse_relabeling(facts)),
                    options=exact,
                )

                self.assertEqual(_support_set(original), _support_set(relabeled))


@dataclass(frozen=True, slots=True)
class FactRelabeling:
    atom_map: dict[AtomId, AtomId]
    bond_map: dict[BondId, BondId]
    component_map: dict[ComponentId, ComponentId]
    site_map: dict[SiteId, SiteId]
    occurrence_map: dict[OccurrenceId, OccurrenceId]


def relabel_molecule_facts(
    facts: MoleculeFacts,
    relabeling: FactRelabeling,
) -> MoleculeFacts:
    out = MoleculeFacts(
        atoms=tuple(
            sorted(
                (
                    replace(atom, id=relabeling.atom_map[atom.id])
                    for atom in facts.atoms
                ),
                key=lambda atom: int(atom.id),
            )
        ),
        bonds=tuple(
            sorted(
                (
                    _relabel_bond(bond, relabeling)
                    for bond in facts.bonds
                ),
                key=lambda bond: int(bond.id),
            )
        ),
        components=tuple(
            sorted(
                (
                    _relabel_component(component, relabeling)
                    for component in facts.components
                ),
                key=lambda component: int(component.id),
            )
        ),
        stereo=StereoFacts(
            tetrahedral=tuple(
                sorted(
                    (
                        _relabel_tetrahedral_site(site, relabeling)
                        for site in facts.stereo.tetrahedral
                    ),
                    key=lambda site: int(site.id),
                )
            ),
            directional=tuple(
                sorted(
                    (
                        _relabel_directional_site(site, relabeling)
                        for site in facts.stereo.directional
                    ),
                    key=lambda site: int(site.id),
                )
            ),
        ),
        ligand_occurrences=tuple(
            sorted(
                (
                    _relabel_occurrence(occurrence, relabeling)
                    for occurrence in facts.ligand_occurrences
                ),
                key=lambda occurrence: int(occurrence.id),
            )
        ),
    )
    out.validate()
    return out


def _support_set(facts: MoleculeFacts) -> frozenset[str]:
    image = enumerate_exhaustive_stereo_support(
        facts=facts,
        policy=ordinary_policy_for_facts(facts),
        semantics=OrdinarySmilesSemantics(),
    )
    return frozenset(image.strings)


def _reverse_relabeling(facts: MoleculeFacts) -> FactRelabeling:
    return FactRelabeling(
        atom_map=_reverse_map(tuple(atom.id for atom in facts.atoms), AtomId),
        bond_map=_reverse_map(tuple(bond.id for bond in facts.bonds), BondId),
        component_map=_reverse_map(
            tuple(component.id for component in facts.components),
            ComponentId,
        ),
        site_map=_reverse_map(
            tuple(
                site.id
                for site in facts.stereo.tetrahedral + facts.stereo.directional
            ),
            SiteId,
        ),
        occurrence_map=_reverse_map(
            tuple(occurrence.id for occurrence in facts.ligand_occurrences),
            OccurrenceId,
        ),
    )


def _reverse_map(values, constructor):
    sorted_values = tuple(sorted(values, key=int))
    reversed_values = tuple(reversed(sorted_values))
    return {
        value: constructor(int(reversed_value))
        for value, reversed_value in zip(sorted_values, reversed_values)
    }


def _exact_tetra_candidate_facts() -> MoleculeFacts:
    base = deep_tetra_ligand_facts(right_terminal="Cl")
    return replace(
        base,
        atoms=(replace(base.atoms[0], implicit_h_count=1),) + base.atoms[1:],
    )


def _relabel_bond(bond: BondFacts, relabeling: FactRelabeling) -> BondFacts:
    return BondFacts(
        id=relabeling.bond_map[bond.id],
        a=relabeling.atom_map[bond.a],
        b=relabeling.atom_map[bond.b],
        order=bond.order,
        is_aromatic=bond.is_aromatic,
        is_conjugated=bond.is_conjugated,
    )


def _relabel_component(
    component: ComponentFacts,
    relabeling: FactRelabeling,
) -> ComponentFacts:
    return ComponentFacts(
        id=relabeling.component_map[component.id],
        atoms=tuple(
            sorted(
                (relabeling.atom_map[atom] for atom in component.atoms),
                key=int,
            )
        ),
        bonds=tuple(
            sorted(
                (relabeling.bond_map[bond] for bond in component.bonds),
                key=int,
            )
        ),
    )


def _relabel_tetrahedral_site(site, relabeling: FactRelabeling):
    return replace(
        site,
        id=relabeling.site_map[site.id],
        center=relabeling.atom_map[site.center],
        ligand_occurrences=tuple(
            relabeling.occurrence_map[occurrence]
            for occurrence in site.ligand_occurrences
        ),
        reference_order=tuple(
            relabeling.occurrence_map[occurrence]
            for occurrence in site.reference_order
        ),
    )


def _relabel_directional_site(site, relabeling: FactRelabeling):
    return replace(
        site,
        id=relabeling.site_map[site.id],
        center_bond=relabeling.bond_map[site.center_bond],
        left_endpoint=relabeling.atom_map[site.left_endpoint],
        right_endpoint=relabeling.atom_map[site.right_endpoint],
        left_ligands=tuple(
            relabeling.occurrence_map[occurrence]
            for occurrence in site.left_ligands
        ),
        right_ligands=tuple(
            relabeling.occurrence_map[occurrence]
            for occurrence in site.right_ligands
        ),
        reference_pair=None
        if site.reference_pair is None
        else (
            relabeling.occurrence_map[site.reference_pair[0]],
            relabeling.occurrence_map[site.reference_pair[1]],
        ),
    )


def _relabel_occurrence(
    occurrence: LigandOccurrence,
    relabeling: FactRelabeling,
) -> LigandOccurrence:
    return LigandOccurrence(
        id=relabeling.occurrence_map[occurrence.id],
        site=relabeling.site_map[occurrence.site],
        kind=occurrence.kind,
        atom=None
        if occurrence.atom is None
        else relabeling.atom_map[occurrence.atom],
        bond=None
        if occurrence.bond is None
        else relabeling.bond_map[occurrence.bond],
        ordinal=occurrence.ordinal,
    )


def _ring_tetrahedral_facts() -> MoleculeFacts:
    site = SiteId(0)
    return MoleculeFacts(
        atoms=(
            _atom(0, "C"),
            _atom(1, "C"),
            _atom(2, "O"),
            _atom(3, "F"),
        ),
        bonds=(
            _bond(0, 0, 1, BondOrder.SINGLE),
            _bond(1, 1, 2, BondOrder.SINGLE),
            _bond(2, 2, 0, BondOrder.SINGLE),
            _bond(3, 0, 3, BondOrder.SINGLE),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1), AtomId(2), AtomId(3)),
                bonds=(BondId(0), BondId(1), BondId(2), BondId(3)),
            ),
        ),
        stereo=StereoFacts(
            tetrahedral=(
                replace(
                    tetrahedral_facts().stereo.tetrahedral[0],
                    id=site,
                    center=AtomId(0),
                    ligand_occurrences=(
                        OccurrenceId(0),
                        OccurrenceId(1),
                        OccurrenceId(2),
                        OccurrenceId(3),
                    ),
                    reference_order=(
                        OccurrenceId(0),
                        OccurrenceId(1),
                        OccurrenceId(2),
                        OccurrenceId(3),
                    ),
                ),
            ),
        ),
        ligand_occurrences=(
            _neighbor(site, 0, atom=1, bond=0),
            _neighbor(site, 1, atom=2, bond=2),
            _neighbor(site, 2, atom=3, bond=3),
            _implicit_h(site, 3, atom=0),
        ),
    )


def _mixed_tetra_directional_facts() -> MoleculeFacts:
    tetra_site = SiteId(0)
    directional_site = SiteId(1)
    return MoleculeFacts(
        atoms=(
            _atom(0, "C"),
            _atom(1, "F"),
            _atom(2, "Cl"),
            _atom(3, "C"),
            _atom(4, "C"),
            _atom(5, "F"),
            _atom(6, "Cl"),
        ),
        bonds=(
            _bond(0, 0, 1, BondOrder.SINGLE),
            _bond(1, 0, 2, BondOrder.SINGLE),
            _bond(2, 0, 3, BondOrder.SINGLE),
            _bond(3, 3, 4, BondOrder.DOUBLE),
            _bond(4, 3, 5, BondOrder.SINGLE),
            _bond(5, 4, 6, BondOrder.SINGLE),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=tuple(AtomId(idx) for idx in range(7)),
                bonds=tuple(BondId(idx) for idx in range(6)),
            ),
        ),
        stereo=StereoFacts(
            tetrahedral=(
                replace(
                    tetrahedral_facts().stereo.tetrahedral[0],
                    id=tetra_site,
                    center=AtomId(0),
                    ligand_occurrences=(
                        OccurrenceId(0),
                        OccurrenceId(1),
                        OccurrenceId(2),
                        OccurrenceId(3),
                    ),
                    reference_order=(
                        OccurrenceId(0),
                        OccurrenceId(1),
                        OccurrenceId(2),
                        OccurrenceId(3),
                    ),
                ),
            ),
            directional=(
                replace(
                    directional_facts().stereo.directional[0],
                    id=directional_site,
                    center_bond=BondId(3),
                    left_endpoint=AtomId(3),
                    right_endpoint=AtomId(4),
                    left_ligands=(OccurrenceId(4), OccurrenceId(5)),
                    right_ligands=(OccurrenceId(6), OccurrenceId(7)),
                    reference_pair=(OccurrenceId(5), OccurrenceId(6)),
                ),
            ),
        ),
        ligand_occurrences=(
            _neighbor(tetra_site, 0, atom=1, bond=0),
            _neighbor(tetra_site, 1, atom=2, bond=1),
            _neighbor(tetra_site, 2, atom=3, bond=2),
            _implicit_h(tetra_site, 3, atom=0),
            _neighbor(directional_site, 4, atom=0, bond=2),
            _neighbor(directional_site, 5, atom=5, bond=4),
            _neighbor(directional_site, 6, atom=6, bond=5),
            _implicit_h(directional_site, 7, atom=4),
        ),
    )


def _atom(idx: int, symbol: str) -> AtomFacts:
    atomic_nums = {"C": 6, "O": 8, "F": 9, "Cl": 17}
    return AtomFacts(
        id=AtomId(idx),
        atomic_num=atomic_nums[symbol],
        symbol=symbol,
        isotope=None,
        formal_charge=0,
        is_aromatic=False,
        explicit_h_count=0,
        implicit_h_count=0,
        no_implicit=False,
    )


def _bond(idx: int, a: int, b: int, order: BondOrder) -> BondFacts:
    return BondFacts(
        id=BondId(idx),
        a=AtomId(a),
        b=AtomId(b),
        order=order,
        is_aromatic=False,
        is_conjugated=False,
    )


def _neighbor(site: SiteId, idx: int, *, atom: int, bond: int) -> LigandOccurrence:
    return LigandOccurrence(
        id=OccurrenceId(idx),
        site=site,
        kind=LigandKind.NEIGHBOR_ATOM,
        atom=AtomId(atom),
        bond=BondId(bond),
    )


def _implicit_h(site: SiteId, idx: int, *, atom: int) -> LigandOccurrence:
    return LigandOccurrence(
        id=OccurrenceId(idx),
        site=site,
        kind=LigandKind.IMPLICIT_H,
        atom=AtomId(atom),
        bond=None,
    )


if __name__ == "__main__":
    unittest.main()
