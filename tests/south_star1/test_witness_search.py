"""Tests for South Star 1 finite non-stereo witness search."""

from __future__ import annotations

import unittest

from grimace._south_star1.facts import DirectionalValue
from grimace._south_star1.facts import TetraValue
from grimace._south_star1.ids import OccurrenceId
from grimace._south_star1.ids import SiteId
from grimace._south_star1.policy import AnnotationMode
from grimace._south_star1.policy import AtomTextDomain
from grimace._south_star1.policy import BondTextChoice
from grimace._south_star1.policy import BondTextDomain
from grimace._south_star1.policy import DirectionMark
from grimace._south_star1.policy import RingLabel
from grimace._south_star1.policy import SmilesPolicy
from grimace._south_star1.policy import TetraToken
from grimace._south_star1.semantics import INVALID
from grimace._south_star1.semantics import Invalid
from grimace._south_star1.slots import BondSlotKind
from grimace._south_star1.witness_search import enumerate_nonstereo_support
from grimace._south_star1.witness_search import enumerate_nonstereo_witnesses

from tests.south_star1.helpers import cco_facts
from tests.south_star1.helpers import cyclopropane_facts
from tests.south_star1.helpers import organic_atom_choice


class WitnessSearchTest(unittest.TestCase):
    def test_nonstereo_tree_support_is_image_of_valid_witnesses(self) -> None:
        facts = cco_facts()
        policy = _policy_with_bond_domains(facts, duplicate_first_atom_text=True)

        image = enumerate_nonstereo_support(facts, policy, _MinimalSemantics())

        self.assertGreater(image.witness_count, image.distinct_count)
        self.assertIn("CCO", image.strings)
        self.assertIn("C(C)O", image.strings)

    def test_nonstereo_ring_support_uses_bounded_ring_labels(self) -> None:
        facts = cyclopropane_facts()
        policy = _policy_with_bond_domains(facts)

        image = enumerate_nonstereo_support(facts, policy, _MinimalSemantics())

        self.assertIn("C1CC1", image.strings)
        self.assertTrue(all("2" not in rendered for rendered in image.strings))

    def test_assignment_generation_is_nonstereo_only(self) -> None:
        facts = cco_facts()
        policy = _policy_with_bond_domains(facts)
        witness = next(enumerate_nonstereo_witnesses(facts, policy, _MinimalSemantics()))

        self.assertEqual(witness.annotation_count, 0)
        self.assertIn("nonstereo_only", {constraint.name for constraint in witness.constraints})


def _policy_with_bond_domains(
    facts,
    *,
    duplicate_first_atom_text: bool = False,
) -> SmilesPolicy:
    bond_choice = BondTextChoice(
        name="elided_bond",
        base_text="",
        permits_direction=False,
    )
    atom_domains = []
    for atom in facts.atoms:
        choices = (organic_atom_choice(atom.symbol),)
        if duplicate_first_atom_text and atom == facts.atoms[0]:
            duplicate = organic_atom_choice(atom.symbol)
            choices = (
                choices[0],
                type(duplicate)(
                    name=f"{duplicate.name}_duplicate",
                    text_by_tetra=duplicate.text_by_tetra,
                ),
            )
        atom_domains.append(AtomTextDomain(atom=atom.id, choices=choices))

    return SmilesPolicy(
        ring_labels=(RingLabel(1), RingLabel(2)),
        annotation_mode=AnnotationMode.HARD,
        atom_text_domains=tuple(atom_domains),
        bond_text_domains=tuple(
            BondTextDomain(
                bond=bond.id,
                slot_kind=slot_kind.value,
                choices=(bond_choice,),
            )
            for bond in facts.bonds
            for slot_kind in (BondSlotKind.TREE, BondSlotKind.RING_ENDPOINT)
        ),
    )


class _MinimalSemantics:
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
        return all(
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
