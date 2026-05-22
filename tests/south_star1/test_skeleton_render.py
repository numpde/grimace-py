"""Tests for South Star 1 traversal skeletons, slots, and rendering."""

from __future__ import annotations

import unittest

from grimace._south_star1.constraints import TraversalAssignment
from grimace._south_star1.constraints import validate_nonstereo_tree_witness
from grimace._south_star1.facts import DirectionalValue
from grimace._south_star1.facts import ComponentFacts
from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.facts import TetraValue
from grimace._south_star1.graph_index import build_graph_index
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import BondSlotId
from grimace._south_star1.ids import ComponentId
from grimace._south_star1.ids import OccurrenceId
from grimace._south_star1.ids import RingEndpointId
from grimace._south_star1.ids import SiteId
from grimace._south_star1.policy import DirectionMark
from grimace._south_star1.policy import RingLabel
from grimace._south_star1.policy import TetraToken
from grimace._south_star1.ring_labels import validate_bounded_ring_labels
from grimace._south_star1.render import render_nonstereo_tree
from grimace._south_star1.render import render_nonstereo_traversal
from grimace._south_star1.semantics import INVALID
from grimace._south_star1.semantics import Invalid
from grimace._south_star1.skeleton import enumerate_traversal_skeletons
from grimace._south_star1.skeleton import enumerate_tree_skeletons
from grimace._south_star1.slots import allocate_tree_slots
from grimace._south_star1.slots import allocate_traversal_slots
from grimace._south_star1.slots import atom_slot_by_atom
from grimace._south_star1.slots import RingEndpointSlot
from grimace._south_star1.slots import SlotBundle
from grimace._south_star1.slots import tree_bond_slot_by_bond

from tests.south_star1.helpers import cco_facts
from tests.south_star1.helpers import cyclopropane_facts
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

    def test_traversal_skeletons_include_ring_endpoints(self) -> None:
        facts = cyclopropane_facts()
        skeletons = enumerate_traversal_skeletons(
            facts,
            build_graph_index(facts),
            organic_subset_policy(facts),
        )

        self.assertTrue(any(len(skeleton.ring_bonds) == 1 for skeleton in skeletons))
        self.assertTrue(
            any(
                _renders_traversal_as(facts, skeleton, "C1CC1")
                for skeleton in skeletons
            )
        )

    def test_ring_labels_reject_overlapping_reuse(self) -> None:
        slots = SlotBundle(
            atom_slots=(),
            bond_slots=(),
            ring_endpoints=(
                RingEndpointSlot(
                    id=RingEndpointId(0),
                    bond=BondId(0),
                    atom=AtomId(0),
                    other_atom=AtomId(1),
                    bond_slot=BondSlotId(0),
                    syntax_position=0,
                ),
                RingEndpointSlot(
                    id=RingEndpointId(1),
                    bond=BondId(0),
                    atom=AtomId(1),
                    other_atom=AtomId(0),
                    bond_slot=BondSlotId(1),
                    syntax_position=5,
                ),
                RingEndpointSlot(
                    id=RingEndpointId(2),
                    bond=BondId(1),
                    atom=AtomId(2),
                    other_atom=AtomId(3),
                    bond_slot=BondSlotId(2),
                    syntax_position=2,
                ),
                RingEndpointSlot(
                    id=RingEndpointId(3),
                    bond=BondId(1),
                    atom=AtomId(3),
                    other_atom=AtomId(2),
                    bond_slot=BondSlotId(3),
                    syntax_position=4,
                ),
            ),
        )
        labels = {endpoint.id: RingLabel(1) for endpoint in slots.ring_endpoints}

        with self.assertRaisesRegex(ValueError, "overlapping intervals"):
            validate_bounded_ring_labels(
                organic_subset_policy(cco_facts()),
                slots,
                labels,
            )

    def test_ring_pair_decode_relation_is_not_a_posthoc_parse_filter(self) -> None:
        facts = cyclopropane_facts()
        skeleton = _traversal_rendering(facts, "C1CC1")
        slots = allocate_traversal_slots(facts, skeleton)
        assignment = _assignment_for(facts, slots, ring_label=RingLabel(1))

        with self.assertRaisesRegex(ValueError, "ring-pair decode relation"):
            render_nonstereo_traversal(
                facts,
                skeleton,
                slots,
                assignment,
                organic_subset_policy(facts),
                _MinimalSemantics(reject_ring_pairs=True),
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


def _traversal_rendering(facts, expected: str):
    for skeleton in enumerate_traversal_skeletons(
        facts,
        build_graph_index(facts),
        organic_subset_policy(facts),
    ):
        if _renders_traversal_as(facts, skeleton, expected):
            return skeleton
    raise AssertionError(f"no traversal skeleton rendered {expected!r}")


def _renders_as(facts, skeleton, expected: str) -> bool:
    slots = allocate_tree_slots(facts, skeleton)
    rendered = render_nonstereo_tree(
        facts,
        skeleton,
        slots,
        _assignment_for(facts, slots),
    )
    return rendered == expected


def _renders_traversal_as(facts, skeleton, expected: str) -> bool:
    slots = allocate_traversal_slots(facts, skeleton)
    assignment = _assignment_for(facts, slots, ring_label=RingLabel(1))
    rendered = render_nonstereo_traversal(
        facts,
        skeleton,
        slots,
        assignment,
        organic_subset_policy(facts),
        _MinimalSemantics(),
    )
    return rendered == expected


def _assignment_for(
    facts,
    slots,
    *,
    ring_label: RingLabel | None = None,
) -> TraversalAssignment:
    return TraversalAssignment(
        atom_text={
            atom.id: organic_atom_choice(atom.symbol)
            for atom in facts.atoms
        },
        tetra_tokens={atom.id: TetraToken.NONE for atom in facts.atoms},
        bond_text={
            slot.id: empty_bond_choice()
            for slot in slots.bond_slots
        },
        ring_labels={
            endpoint.id: ring_label
            for endpoint in slots.ring_endpoints
            if ring_label is not None
        },
        direction_marks={
            slot.id: DirectionMark.ABSENT
            for slot in slots.carrier_slots
        },
    )


class _MinimalSemantics:
    def __init__(self, *, reject_ring_pairs: bool = False) -> None:
        self.reject_ring_pairs = reject_ring_pairs

    def atom_decode_ok(
        self,
        facts,
        atom_id,
        atom_text,
        tetra_token,
        incident_bond_texts,
    ) -> bool:
        return atom_id in {atom.id for atom in facts.atoms} and atom_text.permits(
            tetra_token
        )

    def bond_decode_ok(self, facts, bond, bond_text, direction_mark) -> bool:
        return direction_mark is DirectionMark.ABSENT or bond_text.permits_direction

    def ring_pair_decode_ok(
        self,
        facts,
        bond,
        endpoint_1,
        mark_1,
        endpoint_2,
        mark_2,
    ) -> bool:
        return not self.reject_ring_pairs and all(
            self.bond_decode_ok(facts, bond, endpoint, mark)
            for endpoint, mark in ((endpoint_1, mark_1), (endpoint_2, mark_2))
        )

    def local_tetra_order(self, facts, skel, slots, site: SiteId):
        return ()

    def tetra_value(
        self,
        facts,
        site: SiteId,
        local_order: tuple[OccurrenceId, ...],
        token: TetraToken,
    ) -> TetraValue | Invalid:
        return TetraValue.NONE if token is TetraToken.NONE else INVALID

    def directional_value(
        self,
        facts,
        skel,
        slots,
        site: SiteId,
        marks,
    ) -> DirectionalValue | Invalid:
        return DirectionalValue.NONE if not marks else INVALID


if __name__ == "__main__":
    unittest.main()
