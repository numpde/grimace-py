"""Tests for RDKit-free South Star fact isomorphism."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
import unittest

from grimace._south_star1.fact_isomorphism import facts_are_isomorphic
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
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import ComponentId
from grimace._south_star1.ids import OccurrenceId
from grimace._south_star1.ids import SiteId

from tests.south_star1.helpers import cco_facts
from tests.south_star1.helpers import atom
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import single_bond
from tests.south_star1.helpers import tetrahedral_facts


class FactIsomorphismTest(unittest.TestCase):
    def test_accepts_nonstereo_atom_and_bond_renumbering(self) -> None:
        facts = cco_facts()
        renumbered = _renumber_facts(
            facts,
            atom_map={
                AtomId(0): AtomId(10),
                AtomId(1): AtomId(12),
                AtomId(2): AtomId(11),
            },
            bond_map={BondId(0): BondId(4), BondId(1): BondId(3)},
        )

        self.assertTrue(facts_are_isomorphic(facts, renumbered))

    def test_rejects_different_bond_attributes(self) -> None:
        facts = cco_facts()
        changed = replace(
            facts,
            bonds=(
                facts.bonds[0],
                replace(facts.bonds[1], order=BondOrder.DOUBLE),
            ),
        )

        self.assertFalse(facts_are_isomorphic(facts, changed))

    def test_accepts_component_reordering_without_raw_component_id_match(self) -> None:
        facts = MoleculeFacts(
            atoms=(atom(0, "C"), atom(1, "C"), atom(2, "C"), atom(3, "C")),
            bonds=(
                single_bond(0, 0, 1),
                single_bond(1, 2, 3),
            ),
            components=(
                ComponentFacts(
                    id=ComponentId(0),
                    atoms=(AtomId(0), AtomId(1)),
                    bonds=(BondId(0),),
                ),
                ComponentFacts(
                    id=ComponentId(1),
                    atoms=(AtomId(2), AtomId(3)),
                    bonds=(BondId(1),),
                ),
            ),
        )
        swapped = replace(
            facts,
            components=(
                replace(facts.components[1], id=ComponentId(10)),
                replace(facts.components[0], id=ComponentId(11)),
            ),
        )

        self.assertTrue(facts_are_isomorphic(facts, swapped))

    def test_accepts_stereo_site_renumbering(self) -> None:
        facts = tetrahedral_facts()
        renumbered = _renumber_facts(
            facts,
            atom_map={
                AtomId(0): AtomId(20),
                AtomId(1): AtomId(23),
                AtomId(2): AtomId(21),
                AtomId(3): AtomId(22),
            },
            bond_map={
                BondId(0): BondId(32),
                BondId(1): BondId(30),
                BondId(2): BondId(31),
            },
            site_map={SiteId(0): SiteId(7)},
            occurrence_map={
                OccurrenceId(0): OccurrenceId(42),
                OccurrenceId(1): OccurrenceId(40),
                OccurrenceId(2): OccurrenceId(41),
                OccurrenceId(3): OccurrenceId(43),
            },
        )

        self.assertTrue(facts_are_isomorphic(facts, renumbered))

    def test_rejects_tetrahedral_target_mismatch_when_comparing_stereo(self) -> None:
        facts = tetrahedral_facts()
        site = replace(facts.stereo.tetrahedral[0], target=TetraValue.MINUS)
        changed = replace(facts, stereo=StereoFacts(tetrahedral=(site,)))

        self.assertFalse(facts_are_isomorphic(facts, changed))
        self.assertTrue(facts_are_isomorphic(facts, changed, compare_stereo=False))

    def test_accepts_tetrahedral_reference_order_parity_flip(self) -> None:
        facts = tetrahedral_facts()
        site = facts.stereo.tetrahedral[0]
        changed_site = replace(
            site,
            target=TetraValue.MINUS,
            reference_order=(
                site.reference_order[1],
                site.reference_order[0],
                site.reference_order[2],
                site.reference_order[3],
            ),
        )
        changed = replace(facts, stereo=StereoFacts(tetrahedral=(changed_site,)))

        result = facts_are_isomorphic(facts, changed)

        self.assertTrue(result)
        self.assertTrue(result.isomorphic)
        self.assertIsNotNone(result.isomorphism)

    def test_rejects_tetrahedral_reference_order_parity_mismatch(self) -> None:
        facts = tetrahedral_facts()
        site = facts.stereo.tetrahedral[0]
        changed_site = replace(
            site,
            reference_order=(
                site.reference_order[1],
                site.reference_order[0],
                site.reference_order[2],
                site.reference_order[3],
            ),
        )
        changed = replace(facts, stereo=StereoFacts(tetrahedral=(changed_site,)))

        result = facts_are_isomorphic(facts, changed)

        self.assertFalse(result)
        self.assertEqual(result.reason, "tetrahedral stereo mismatch")

    def test_rejects_directional_target_mismatch_when_comparing_stereo(self) -> None:
        facts = directional_facts()
        site = replace(
            facts.stereo.directional[0],
            target=DirectionalValue.TOGETHER,
        )
        changed = replace(facts, stereo=StereoFacts(directional=(site,)))

        self.assertFalse(facts_are_isomorphic(facts, changed))
        self.assertTrue(facts_are_isomorphic(facts, changed, compare_stereo=False))

    def test_accepts_directional_site_side_swap(self) -> None:
        facts = directional_facts()
        site = facts.stereo.directional[0]
        swapped_site = replace(
            site,
            left_endpoint=site.right_endpoint,
            right_endpoint=site.left_endpoint,
            left_ligands=site.right_ligands,
            right_ligands=site.left_ligands,
            reference_pair=None
            if site.reference_pair is None
            else (site.reference_pair[1], site.reference_pair[0]),
        )
        swapped = replace(facts, stereo=StereoFacts(directional=(swapped_site,)))

        self.assertTrue(facts_are_isomorphic(facts, swapped))

    def test_accepts_directional_reference_pair_single_side_flip(self) -> None:
        facts = _two_substituent_directional_facts()
        site = facts.stereo.directional[0]
        changed_site = replace(
            site,
            target=DirectionalValue.TOGETHER,
            reference_pair=(OccurrenceId(1), OccurrenceId(2)),
        )
        changed = replace(facts, stereo=StereoFacts(directional=(changed_site,)))

        self.assertTrue(facts_are_isomorphic(facts, changed))

    def test_rejects_directional_reference_pair_single_side_mismatch(self) -> None:
        facts = _two_substituent_directional_facts()
        site = facts.stereo.directional[0]
        changed_site = replace(
            site,
            reference_pair=(OccurrenceId(1), OccurrenceId(2)),
        )
        changed = replace(facts, stereo=StereoFacts(directional=(changed_site,)))

        self.assertFalse(facts_are_isomorphic(facts, changed))

    def test_accepts_directional_reference_pair_two_side_flip(self) -> None:
        facts = _two_substituent_directional_facts()
        site = facts.stereo.directional[0]
        changed_site = replace(
            site,
            reference_pair=(OccurrenceId(1), OccurrenceId(3)),
        )
        changed = replace(facts, stereo=StereoFacts(directional=(changed_site,)))

        self.assertTrue(facts_are_isomorphic(facts, changed))

    def test_rejects_missing_unspecified_stereo_site(self) -> None:
        facts = tetrahedral_facts()
        site = replace(
            facts.stereo.tetrahedral[0],
            target=TetraValue.NONE,
            status=SiteStatus.UNSPECIFIED,
        )
        with_unspecified = replace(facts, stereo=StereoFacts(tetrahedral=(site,)))
        without_stereo = replace(
            facts,
            stereo=StereoFacts(),
            ligand_occurrences=(),
        )

        self.assertFalse(facts_are_isomorphic(with_unspecified, without_stereo))
        self.assertTrue(
            facts_are_isomorphic(
                with_unspecified,
                without_stereo,
                compare_potential_sites=False,
            )
        )

    def test_module_has_no_rdkit_import(self) -> None:
        path = Path("python/grimace/_south_star1/fact_isomorphism.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))

        self.assertFalse(_imports_rdkit(tree))


def _renumber_facts(
    facts: MoleculeFacts,
    *,
    atom_map: dict[AtomId, AtomId],
    bond_map: dict[BondId, BondId],
    site_map: dict[SiteId, SiteId] | None = None,
    occurrence_map: dict[OccurrenceId, OccurrenceId] | None = None,
) -> MoleculeFacts:
    site_map = site_map or {}
    occurrence_map = occurrence_map or {}
    identity_site_map = {site.id: site.id for site in facts.stereo.tetrahedral}
    identity_site_map.update({site.id: site.id for site in facts.stereo.directional})
    identity_occurrence_map = {
        occurrence.id: occurrence.id
        for occurrence in facts.ligand_occurrences
    }
    site_map = identity_site_map | site_map
    occurrence_map = identity_occurrence_map | occurrence_map

    return MoleculeFacts(
        atoms=tuple(replace(atom, id=atom_map[atom.id]) for atom in facts.atoms),
        bonds=tuple(
            BondFacts(
                id=bond_map[bond.id],
                a=atom_map[bond.a],
                b=atom_map[bond.b],
                order=bond.order,
                is_aromatic=bond.is_aromatic,
                is_conjugated=bond.is_conjugated,
            )
            for bond in facts.bonds
        ),
        components=tuple(
            ComponentFacts(
                id=component.id,
                atoms=tuple(atom_map[atom] for atom in component.atoms),
                bonds=tuple(bond_map[bond] for bond in component.bonds),
            )
            for component in facts.components
        ),
        stereo=StereoFacts(
            tetrahedral=tuple(
                replace(
                    site,
                    id=site_map[site.id],
                    center=atom_map[site.center],
                    ligand_occurrences=tuple(
                        occurrence_map[occurrence]
                        for occurrence in site.ligand_occurrences
                    ),
                    reference_order=tuple(
                        occurrence_map[occurrence]
                        for occurrence in site.reference_order
                    ),
                )
                for site in facts.stereo.tetrahedral
            ),
            directional=tuple(
                replace(
                    site,
                    id=site_map[site.id],
                    center_bond=bond_map[site.center_bond],
                    left_endpoint=atom_map[site.left_endpoint],
                    right_endpoint=atom_map[site.right_endpoint],
                    left_ligands=tuple(
                        occurrence_map[occurrence] for occurrence in site.left_ligands
                    ),
                    right_ligands=tuple(
                        occurrence_map[occurrence] for occurrence in site.right_ligands
                    ),
                    reference_pair=None
                    if site.reference_pair is None
                    else tuple(
                        occurrence_map[occurrence]
                        for occurrence in site.reference_pair
                    ),
                )
                for site in facts.stereo.directional
            ),
        ),
        ligand_occurrences=tuple(
            LigandOccurrence(
                id=occurrence_map[occurrence.id],
                site=site_map[occurrence.site],
                kind=occurrence.kind,
                atom=None if occurrence.atom is None else atom_map[occurrence.atom],
                bond=None if occurrence.bond is None else bond_map[occurrence.bond],
                ordinal=occurrence.ordinal,
            )
            for occurrence in facts.ligand_occurrences
        ),
    )


def _two_substituent_directional_facts() -> MoleculeFacts:
    site = SiteId(0)
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
            BondFacts(
                id=BondId(0),
                a=AtomId(0),
                b=AtomId(1),
                order=BondOrder.DOUBLE,
                is_aromatic=False,
                is_conjugated=False,
            ),
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
        stereo=StereoFacts(
            directional=(
                DirectionalSiteFacts(
                    id=site,
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
                site=site,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(2),
                bond=BondId(1),
            ),
            LigandOccurrence(
                id=OccurrenceId(1),
                site=site,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(3),
                bond=BondId(2),
            ),
            LigandOccurrence(
                id=OccurrenceId(2),
                site=site,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(4),
                bond=BondId(3),
            ),
            LigandOccurrence(
                id=OccurrenceId(3),
                site=site,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(5),
                bond=BondId(4),
            ),
        ),
    )


def _imports_rdkit(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.startswith("rdkit") for alias in node.names):
                return True
        if isinstance(node, ast.ImportFrom):
            if node.module is not None and node.module.startswith("rdkit"):
                return True
    return False


if __name__ == "__main__":
    unittest.main()
