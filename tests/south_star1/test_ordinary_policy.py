"""Tests for the bounded ordinary South Star policy factory."""

from __future__ import annotations

from dataclasses import replace
import unittest

from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.errors import SouthStarErrorKind
from grimace._south_star1.facts import BondOrder
from grimace._south_star1.facts import ComponentFacts
from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import ComponentId
from grimace._south_star1.ordinary_policy import OrdinaryPolicyOptions
from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.policy import AnnotationMode
from grimace._south_star1.policy import TetraToken
from grimace._south_star1.slots import BondSlotKind

from tests.south_star1.helpers import atom
from tests.south_star1.helpers import bond
from tests.south_star1.helpers import cyclopropane_facts
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import single_bond
from tests.south_star1.helpers import tetrahedral_facts


class OrdinaryPolicyTest(unittest.TestCase):
    def test_builds_tree_and_ring_domains_for_supported_bonds(self) -> None:
        facts = cyclopropane_facts()

        policy = ordinary_policy_for_facts(
            facts,
            OrdinaryPolicyOptions(ring_label_values=(1, 2)),
        )

        for bond_facts in facts.bonds:
            self.assertEqual(
                policy.bond_text_domain(
                    facts,
                    bond_facts.id,
                    slot_kind=BondSlotKind.TREE.value,
                )[0].base_text,
                "",
            )
            self.assertEqual(
                policy.bond_text_domain(
                    facts,
                    bond_facts.id,
                    slot_kind=BondSlotKind.RING_ENDPOINT.value,
                )[0].base_text,
                "",
            )

    def test_directional_single_bond_choices_permit_marker_overlay(self) -> None:
        facts = directional_facts()

        policy = ordinary_policy_for_facts(facts)

        single_choice = policy.bond_text_domain(
            facts,
            BondId(1),
            slot_kind=BondSlotKind.TREE.value,
        )[0]
        double_choice = policy.bond_text_domain(
            facts,
            BondId(0),
            slot_kind=BondSlotKind.TREE.value,
        )[0]

        self.assertEqual(single_choice.base_text, "")
        self.assertTrue(single_choice.permits_direction)
        self.assertEqual(double_choice.base_text, "=")
        self.assertFalse(double_choice.permits_direction)

    def test_tetrahedral_center_gets_bracketed_token_text(self) -> None:
        facts = tetrahedral_facts()

        policy = ordinary_policy_for_facts(
            facts,
            OrdinaryPolicyOptions(annotation_mode=AnnotationMode.HARD),
        )
        choice = policy.atom_text_domain(facts, AtomId(0))[0]

        self.assertEqual(choice.render(TetraToken.NONE), "C")
        self.assertEqual(choice.render(TetraToken.AT), "[C@H]")
        self.assertEqual(choice.render(TetraToken.ATAT), "[C@@H]")

    def test_non_single_ring_closure_is_explicitly_unsupported(self) -> None:
        facts = _cyclopropene_facts()

        with self.assertRaisesRegex(
            SouthStarError,
            "non-single ring closures",
        ) as raised:
            ordinary_policy_for_facts(facts)
        self.assertIs(raised.exception.kind, SouthStarErrorKind.UNSUPPORTED_POLICY)

    def test_unsupported_atom_facts_fail_fast(self) -> None:
        base = tetrahedral_facts()
        facts = replace(
            base,
            atoms=(replace(base.atoms[0], formal_charge=1),) + base.atoms[1:],
        )

        with self.assertRaisesRegex(SouthStarError, "charged atoms") as raised:
            ordinary_policy_for_facts(facts)
        self.assertIs(raised.exception.kind, SouthStarErrorKind.UNSUPPORTED_ATOM)


def _cyclopropene_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, "C"), atom(1, "C"), atom(2, "C")),
        bonds=(
            bond(0, 0, 1, BondOrder.DOUBLE),
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


if __name__ == "__main__":
    unittest.main()
