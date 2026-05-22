"""Tests for South Star 1 stereo relations over typed traversal assignments."""

from __future__ import annotations

from dataclasses import replace
import unittest

from grimace._south_star1.constraints import TraversalAssignment
from grimace._south_star1.constraints import validate_stereo_traversal_witness
from grimace._south_star1.facts import DirectionalValue
from grimace._south_star1.facts import SiteStatus
from grimace._south_star1.facts import StereoFacts
from grimace._south_star1.facts import TetraValue
from grimace._south_star1.graph_index import build_graph_index
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import OccurrenceId
from grimace._south_star1.ids import SiteId
from grimace._south_star1.policy import AtomTextChoice
from grimace._south_star1.policy import BondTextChoice
from grimace._south_star1.policy import DirectionMark
from grimace._south_star1.policy import TetraToken
from grimace._south_star1.semantics import INVALID
from grimace._south_star1.semantics import Invalid
from grimace._south_star1.skeleton import enumerate_traversal_skeletons
from grimace._south_star1.slots import SlotBundle
from grimace._south_star1.slots import allocate_traversal_slots

from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import organic_atom_choice
from tests.south_star1.helpers import organic_subset_policy
from tests.south_star1.helpers import tetrahedral_facts


class StereoRelationsTest(unittest.TestCase):
    def test_tetrahedral_relation_accepts_semantics_defined_token(self) -> None:
        facts = tetrahedral_facts()
        skeleton = _first_skeleton(facts)
        slots = allocate_traversal_slots(facts, skeleton)
        assignment = _assignment(facts, slots, tetra_center_token=TetraToken.AT)

        constraints = validate_stereo_traversal_witness(
            facts,
            skeleton,
            slots,
            assignment,
            organic_subset_policy(facts),
            _StereoSemantics(),
        )

        self.assertIn(
            "tetrahedral_relations",
            {constraint.name for constraint in constraints},
        )

    def test_tetrahedral_relation_rejects_wrong_token(self) -> None:
        facts = tetrahedral_facts()
        skeleton = _first_skeleton(facts)
        slots = allocate_traversal_slots(facts, skeleton)
        assignment = _assignment(facts, slots, tetra_center_token=TetraToken.NONE)

        with self.assertRaisesRegex(ValueError, "tetrahedral relation"):
            validate_stereo_traversal_witness(
                facts,
                skeleton,
                slots,
                assignment,
                organic_subset_policy(facts),
                _StereoSemantics(),
            )

    def test_unspecified_tetrahedral_site_forces_no_accidental_token(self) -> None:
        facts = _unspecified_tetrahedral_facts()
        skeleton = _first_skeleton(facts)
        slots = allocate_traversal_slots(facts, skeleton)

        validate_stereo_traversal_witness(
            facts,
            skeleton,
            slots,
            _assignment(facts, slots, tetra_center_token=TetraToken.NONE),
            organic_subset_policy(facts),
            _StereoSemantics(),
        )

        with self.assertRaisesRegex(ValueError, "tetrahedral relation"):
            validate_stereo_traversal_witness(
                facts,
                skeleton,
                slots,
                _assignment(facts, slots, tetra_center_token=TetraToken.AT),
                organic_subset_policy(facts),
                _StereoSemantics(),
            )

    def test_directional_relation_accepts_carrier_marks(self) -> None:
        facts = directional_facts()
        skeleton = _first_skeleton(facts)
        slots = allocate_traversal_slots(facts, skeleton)
        assignment = _assignment(facts, slots, mark_first_carrier=True)

        constraints = validate_stereo_traversal_witness(
            facts,
            skeleton,
            slots,
            assignment,
            organic_subset_policy(facts),
            _StereoSemantics(),
        )

        self.assertIn(
            "directional_relations",
            {constraint.name for constraint in constraints},
        )

    def test_directional_relation_rejects_missing_carrier_marks(self) -> None:
        facts = directional_facts()
        skeleton = _first_skeleton(facts)
        slots = allocate_traversal_slots(facts, skeleton)
        assignment = _assignment(facts, slots, mark_first_carrier=False)

        with self.assertRaisesRegex(ValueError, "directional relation"):
            validate_stereo_traversal_witness(
                facts,
                skeleton,
                slots,
                assignment,
                organic_subset_policy(facts),
                _StereoSemantics(),
            )

    def test_directional_relation_uses_declared_carrier_scope(self) -> None:
        facts = _unspecified_directional_facts()
        skeleton = _first_skeleton(facts)
        slots = allocate_traversal_slots(facts, skeleton)
        assignment = _assignment(facts, slots, mark_first_carrier=False)
        assignment.direction_marks[slots.carrier_slots[-1].id] = DirectionMark.FWD

        constraints = validate_stereo_traversal_witness(
            facts,
            skeleton,
            slots,
            assignment,
            organic_subset_policy(facts),
            _StereoSemantics(scope=(slots.carrier_slots[0].id,)),
        )

        self.assertIn(
            "directional_relations",
            {constraint.name for constraint in constraints},
        )

    def test_unspecified_directional_site_forces_no_accidental_marks(self) -> None:
        facts = _unspecified_directional_facts()
        skeleton = _first_skeleton(facts)
        slots = allocate_traversal_slots(facts, skeleton)

        validate_stereo_traversal_witness(
            facts,
            skeleton,
            slots,
            _assignment(facts, slots, mark_first_carrier=False),
            organic_subset_policy(facts),
            _StereoSemantics(),
        )

        with self.assertRaisesRegex(ValueError, "directional relation"):
            validate_stereo_traversal_witness(
                facts,
                skeleton,
                slots,
                _assignment(facts, slots, mark_first_carrier=True),
                organic_subset_policy(facts),
                _StereoSemantics(),
            )


def _first_skeleton(facts):
    return enumerate_traversal_skeletons(
        facts,
        build_graph_index(facts),
        organic_subset_policy(facts),
    )[0]


def _unspecified_tetrahedral_facts():
    facts = tetrahedral_facts()
    site = replace(
        facts.stereo.tetrahedral[0],
        status=SiteStatus.UNSPECIFIED,
        target=TetraValue.NONE,
    )
    return replace(facts, stereo=StereoFacts(tetrahedral=(site,)))


def _unspecified_directional_facts():
    facts = directional_facts()
    site = replace(
        facts.stereo.directional[0],
        status=SiteStatus.UNSPECIFIED,
        target=DirectionalValue.NONE,
    )
    return replace(facts, stereo=StereoFacts(directional=(site,)))


def _assignment(
    facts,
    slots: SlotBundle,
    *,
    tetra_center_token: TetraToken = TetraToken.NONE,
    mark_first_carrier: bool = False,
) -> TraversalAssignment:
    first_carrier_id = slots.carrier_slots[0].id if mark_first_carrier else None
    return TraversalAssignment(
        atom_text={
            atom.id: _chiral_carbon_choice()
            if atom.id == AtomId(0)
            else organic_atom_choice(atom.symbol)
            for atom in facts.atoms
        },
        tetra_tokens={
            atom.id: tetra_center_token if atom.id == AtomId(0) else TetraToken.NONE
            for atom in facts.atoms
        },
        bond_text={
            slot.id: BondTextChoice(
                name="test_bond",
                base_text="",
                permits_direction=True,
            )
            for slot in slots.bond_slots
        },
        ring_labels={},
        direction_marks={
            carrier.id: DirectionMark.FWD
            if carrier.id == first_carrier_id
            else DirectionMark.ABSENT
            for carrier in slots.carrier_slots
        },
    )


def _chiral_carbon_choice() -> AtomTextChoice:
    return AtomTextChoice(
        name="chiral_c",
        text_by_tetra=(
            (TetraToken.NONE, "C"),
            (TetraToken.AT, "[C@H]"),
            (TetraToken.ATAT, "[C@@H]"),
        ),
    )


class _StereoSemantics:
    def __init__(self, *, scope=None) -> None:
        self.scope = scope

    def atom_decode_ok(
        self,
        facts,
        atom,
        atom_text,
        tetra_token,
        incident_bond_texts,
    ) -> bool:
        return atom_text.permits(tetra_token)

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
        return all(
            self.bond_decode_ok(facts, bond, endpoint, mark)
            for endpoint, mark in ((endpoint_1, mark_1), (endpoint_2, mark_2))
        )

    def local_tetra_order(
        self,
        facts,
        skel,
        slots,
        site: SiteId,
    ) -> tuple[OccurrenceId, ...]:
        return facts.stereo.tetrahedral[0].reference_order

    def tetra_value(
        self,
        facts,
        site: SiteId,
        local_order: tuple[OccurrenceId, ...],
        token: TetraToken,
    ) -> TetraValue | Invalid:
        if token is TetraToken.AT:
            return TetraValue.PLUS
        if token is TetraToken.NONE:
            return TetraValue.NONE
        return INVALID

    def directional_scope(self, facts, skel, slots, site: SiteId):
        if self.scope is not None:
            return self.scope
        return tuple(carrier.id for carrier in slots.carrier_slots)

    def directional_value(
        self,
        facts,
        skel,
        slots,
        site: SiteId,
        marks,
    ) -> DirectionalValue:
        if any(mark is not DirectionMark.ABSENT for mark in marks.values()):
            return DirectionalValue.OPPOSITE
        return DirectionalValue.NONE


if __name__ == "__main__":
    unittest.main()
