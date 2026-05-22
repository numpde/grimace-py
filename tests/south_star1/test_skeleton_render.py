"""Tests for South Star 1 traversal skeletons, slots, and rendering."""

from __future__ import annotations

import unittest

from grimace._south_star1.constraints import NonStereoTreeAssignment
from grimace._south_star1.constraints import validate_nonstereo_tree_witness
from grimace._south_star1.facts import ComponentFacts
from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.graph_index import build_graph_index
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import BondSlotId
from grimace._south_star1.ids import ComponentId
from grimace._south_star1.policy import TetraToken
from grimace._south_star1.render import render_nonstereo_tree
from grimace._south_star1.skeleton import enumerate_tree_skeletons
from grimace._south_star1.slots import allocate_tree_slots
from grimace._south_star1.slots import atom_slot_by_atom
from grimace._south_star1.slots import tree_bond_slot_by_bond

from tests.south_star1.helpers import cco_facts
from tests.south_star1.helpers import empty_bond_choice
from tests.south_star1.helpers import atom
from tests.south_star1.helpers import organic_atom_choice
from tests.south_star1.helpers import organic_subset_policy
from tests.south_star1.helpers import single_bond


class SkeletonRenderTest(unittest.TestCase):
    def test_tree_skeletons_enumerate_roots_and_local_roles(self) -> None:
        facts = cco_facts()
        skeletons = enumerate_tree_skeletons(
            facts,
            build_graph_index(facts),
            organic_subset_policy(facts),
        )

        self.assertEqual({skeleton.roots for skeleton in skeletons}, {
            (AtomId(0),),
            (AtomId(1),),
            (AtomId(2),),
        })
        self.assertTrue(
            any(_renders_as(facts, skeleton, "CCO") for skeleton in skeletons)
        )
        self.assertTrue(
            any(_renders_as(facts, skeleton, "C(C)O") for skeleton in skeletons)
        )

    def test_slots_are_owned_by_atoms_and_tree_bonds(self) -> None:
        facts = cco_facts()
        skeleton = _skeleton_rendering(facts, "CCO")
        slots = allocate_tree_slots(facts, skeleton)

        self.assertEqual(
            set(atom_slot_by_atom(slots)),
            {AtomId(0), AtomId(1), AtomId(2)},
        )
        self.assertEqual(set(tree_bond_slot_by_bond(slots)), skeleton.tree_bonds)
        self.assertEqual(len(slots.bond_slots), len(skeleton.tree_bonds))

    def test_tree_skeletons_reject_disconnected_component_facts(self) -> None:
        facts = MoleculeFacts(
            atoms=(atom(0, "C"), atom(1, "C"), atom(2, "C"), atom(3, "O")),
            bonds=(
                single_bond(0, 0, 1),
                single_bond(1, 1, 2),
                single_bond(2, 2, 0),
            ),
            components=(
                ComponentFacts(
                    id=ComponentId(0),
                    atoms=(AtomId(0), AtomId(1), AtomId(2), AtomId(3)),
                    bonds=(BondId(0), BondId(1), BondId(2)),
                ),
            ),
        )

        with self.assertRaisesRegex(NotImplementedError, "connected components"):
            enumerate_tree_skeletons(
                facts,
                build_graph_index(facts),
                organic_subset_policy(facts),
            )

    def test_named_constraints_back_rendered_witness(self) -> None:
        facts = cco_facts()
        skeleton = _skeleton_rendering(facts, "CCO")
        slots = allocate_tree_slots(facts, skeleton)
        assignment = _assignment_for(facts, slots)

        constraints = validate_nonstereo_tree_witness(
            facts,
            skeleton,
            slots,
            assignment,
        )

        self.assertEqual(
            render_nonstereo_tree(facts, skeleton, slots, assignment),
            "CCO",
        )
        self.assertIn(
            "tree_bond_slot_coverage",
            {constraint.name for constraint in constraints},
        )

    def test_render_fails_when_atom_text_is_unowned(self) -> None:
        facts = cco_facts()
        skeleton = _skeleton_rendering(facts, "CCO")
        slots = allocate_tree_slots(facts, skeleton)
        assignment = _assignment_for(facts, slots)
        del assignment.atom_text[AtomId(2)]

        with self.assertRaisesRegex(ValueError, "atom text coverage mismatch"):
            render_nonstereo_tree(facts, skeleton, slots, assignment)

    def test_render_fails_when_tree_bond_text_is_unowned(self) -> None:
        facts = cco_facts()
        skeleton = _skeleton_rendering(facts, "CCO")
        slots = allocate_tree_slots(facts, skeleton)
        assignment = _assignment_for(facts, slots)
        del assignment.bond_text[BondSlotId(1)]

        with self.assertRaisesRegex(ValueError, "tree bond text coverage mismatch"):
            render_nonstereo_tree(facts, skeleton, slots, assignment)


def _skeleton_rendering(facts, expected: str):
    for skeleton in enumerate_tree_skeletons(
        facts,
        build_graph_index(facts),
        organic_subset_policy(facts),
    ):
        if _renders_as(facts, skeleton, expected):
            return skeleton
    raise AssertionError(f"no skeleton rendered {expected!r}")


def _renders_as(facts, skeleton, expected: str) -> bool:
    slots = allocate_tree_slots(facts, skeleton)
    rendered = render_nonstereo_tree(
        facts,
        skeleton,
        slots,
        _assignment_for(facts, slots),
    )
    return rendered == expected


def _assignment_for(facts, slots) -> NonStereoTreeAssignment:
    return NonStereoTreeAssignment(
        atom_text={
            atom.id: organic_atom_choice(atom.symbol)
            for atom in facts.atoms
        },
        tetra_tokens={atom.id: TetraToken.NONE for atom in facts.atoms},
        bond_text={
            slot.id: empty_bond_choice()
            for slot in slots.bond_slots
        },
        ring_labels={},
    )


if __name__ == "__main__":
    unittest.main()
