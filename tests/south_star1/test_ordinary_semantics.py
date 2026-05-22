"""Tests for the bounded ordinary South Star SMILES semantics."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

from grimace._south_star1.facts import BondFacts
from grimace._south_star1.facts import BondOrder
from grimace._south_star1.graph_index import build_graph_index
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ordinary_semantics import OrdinarySmilesSemantics
from grimace._south_star1.policy import AnnotationMode
from grimace._south_star1.policy import AtomTextChoice
from grimace._south_star1.policy import AtomTextDomain
from grimace._south_star1.policy import BondTextChoice
from grimace._south_star1.policy import BondTextDomain
from grimace._south_star1.policy import DirectionMark
from grimace._south_star1.policy import RingLabel
from grimace._south_star1.policy import SmilesPolicy
from grimace._south_star1.policy import TetraToken
from grimace._south_star1.skeleton import enumerate_traversal_skeletons
from grimace._south_star1.slots import BondSlotKind
from grimace._south_star1.slots import allocate_traversal_slots
from grimace._south_star1.slots import carrier_slot_by_bond_slot
from grimace._south_star1.stereo_csp import build_stereo_csp
from grimace._south_star1.stereo_witness import enumerate_presentation_prefixes
from grimace._south_star1.support_enumeration import enumerate_stereo_support

from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import organic_atom_choice
from tests.south_star1.helpers import tetrahedral_facts


class OrdinarySemanticsTest(unittest.TestCase):
    def test_tetrahedral_parity_splits_root_zero_support(self) -> None:
        facts = tetrahedral_facts()
        policy = _ordinary_policy(facts, chiral_center=AtomId(0))
        skeletons = tuple(
            skeleton
            for skeleton in enumerate_traversal_skeletons(
                facts,
                build_graph_index(facts),
                policy,
            )
            if skeleton.roots == (AtomId(0),)
        )

        image = enumerate_stereo_support(
            facts=facts,
            policy=policy,
            semantics=OrdinarySmilesSemantics(),
            skeletons=skeletons,
        )

        self.assertEqual(image.witness_count, 12)
        self.assertEqual(sum("[C@H]" in text for text in image.strings), 6)
        self.assertEqual(sum("[C@@H]" in text for text in image.strings), 6)

    def test_tetrahedral_local_order_includes_incoming_parent_ligand(self) -> None:
        facts = tetrahedral_facts()
        policy = _ordinary_policy(facts, chiral_center=AtomId(0))
        skeleton = next(
            skeleton
            for skeleton in enumerate_traversal_skeletons(
                facts,
                build_graph_index(facts),
                policy,
            )
            if skeleton.roots == (AtomId(1),)
        )

        local_order = OrdinarySmilesSemantics().local_tetra_order(
            facts,
            skeleton,
            allocate_traversal_slots(facts, skeleton),
            facts.stereo.tetrahedral[0].id,
        )

        self.assertEqual(local_order[0], facts.ligand_occurrences[0].id)
        self.assertEqual(
            set(local_order),
            set(facts.stereo.tetrahedral[0].reference_order),
        )

    def test_ordinary_directional_scope_excludes_center_bond(self) -> None:
        facts = directional_facts()
        policy = _ordinary_policy(facts)
        skeleton = _first_skeleton(facts, policy)
        slots = allocate_traversal_slots(facts, skeleton)
        site = facts.stereo.directional[0]
        semantics = OrdinarySmilesSemantics()
        carrier_by_id = {carrier.id: carrier for carrier in slots.carrier_slots}

        scope = semantics.directional_scope(facts, skeleton, slots, site.id)

        self.assertTrue(scope)
        self.assertTrue(
            all(carrier_by_id[carrier].bond != site.center_bond for carrier in scope)
        )
        self.assertEqual(
            {carrier_by_id[carrier].bond for carrier in scope},
            {facts.bonds[1].id, facts.bonds[2].id},
        )

    def test_ordinary_directional_center_carrier_is_forced_absent(self) -> None:
        facts = directional_facts()
        policy = _ordinary_policy(facts)
        skeleton = _first_skeleton(facts, policy)
        slots = allocate_traversal_slots(facts, skeleton)
        prefix = next(
            enumerate_presentation_prefixes(
                facts=facts,
                slots=slots,
                policy=policy,
            )
        )
        csp = build_stereo_csp(
            facts=facts,
            skeleton=skeleton,
            slots=slots,
            prefix=prefix,
            policy=policy,
            semantics=OrdinarySmilesSemantics(),
        )
        site = facts.stereo.directional[0]
        carrier_by_bond_slot = carrier_slot_by_bond_slot(slots)

        for slot in slots.bond_slots:
            if slot.bond != site.center_bond:
                continue
            carrier = carrier_by_bond_slot[slot.id]
            self.assertEqual(prefix.bond_text[slot.id].base_text, "=")
            self.assertEqual(csp.direction_domains[carrier.id], (DirectionMark.ABSENT,))

    def test_ordinary_directional_support_renders_center_double_bond(self) -> None:
        facts = directional_facts()
        policy = _ordinary_policy(facts)

        image = enumerate_stereo_support(
            facts=facts,
            policy=policy,
            semantics=OrdinarySmilesSemantics(),
        )

        self.assertGreater(image.distinct_count, 0)
        self.assertTrue(all("=" in text for text in image.strings))
        self.assertIn("C(/F)=C(\\Cl)", image.strings)

    def test_module_has_no_rdkit_import(self) -> None:
        path = Path("python/grimace/_south_star1/ordinary_semantics.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))

        self.assertFalse(_imports_rdkit(tree))


def _first_skeleton(facts, policy: SmilesPolicy):
    return enumerate_traversal_skeletons(
        facts,
        build_graph_index(facts),
        policy,
    )[0]


def _ordinary_policy(
    facts,
    *,
    chiral_center: AtomId | None = None,
) -> SmilesPolicy:
    return SmilesPolicy(
        ring_labels=(RingLabel(1), RingLabel(2)),
        annotation_mode=AnnotationMode.HARD,
        atom_text_domains=tuple(
            AtomTextDomain(
                atom=atom.id,
                choices=(
                    _chiral_carbon_choice()
                    if atom.id == chiral_center
                    else organic_atom_choice(atom.symbol),
                ),
            )
            for atom in facts.atoms
        ),
        bond_text_domains=tuple(
            BondTextDomain(
                bond=bond.id,
                slot_kind=BondSlotKind.TREE.value,
                choices=(_ordinary_bond_choice(bond),),
            )
            for bond in facts.bonds
        ),
    )


def _ordinary_bond_choice(bond: BondFacts) -> BondTextChoice:
    if bond.order is BondOrder.SINGLE:
        return BondTextChoice(
            name="single_or_directional",
            base_text="",
            permits_direction=True,
        )
    if bond.order is BondOrder.DOUBLE:
        return BondTextChoice(
            name="double",
            base_text="=",
            permits_direction=False,
        )
    if bond.order is BondOrder.TRIPLE:
        return BondTextChoice(
            name="triple",
            base_text="#",
            permits_direction=False,
        )
    if bond.order is BondOrder.AROMATIC:
        return BondTextChoice(
            name="aromatic",
            base_text="",
            permits_direction=False,
        )
    raise ValueError(bond.order)


def _chiral_carbon_choice() -> AtomTextChoice:
    return AtomTextChoice(
        name="chiral_c",
        text_by_tetra=(
            (TetraToken.NONE, "C"),
            (TetraToken.AT, "[C@H]"),
            (TetraToken.ATAT, "[C@@H]"),
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
