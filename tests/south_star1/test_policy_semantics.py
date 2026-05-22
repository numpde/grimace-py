"""Tests for South Star 1 finite policy and parser-semantics contracts."""

from __future__ import annotations

from collections.abc import Mapping
import unittest

from grimace._south_star1.facts import AtomFacts
from grimace._south_star1.facts import ComponentFacts
from grimace._south_star1.facts import DirectionalValue
from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.facts import TetraValue
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import CarrierSlotId
from grimace._south_star1.ids import ComponentId
from grimace._south_star1.ids import OccurrenceId
from grimace._south_star1.ids import SiteId
from grimace._south_star1.policy import AnnotationMode
from grimace._south_star1.policy import AtomTextChoice
from grimace._south_star1.policy import AtomTextDomain
from grimace._south_star1.policy import BondTextChoice
from grimace._south_star1.policy import DirectionMark
from grimace._south_star1.policy import RingLabel
from grimace._south_star1.policy import SmilesPolicy
from grimace._south_star1.policy import TetraToken
from grimace._south_star1.semantics import INVALID
from grimace._south_star1.semantics import Invalid
from grimace._south_star1.semantics import ParserSemantics


class PolicySemanticsTest(unittest.TestCase):
    def test_ring_label_text_is_bounded_policy_data(self) -> None:
        self.assertEqual(RingLabel(1).text(), "1")
        self.assertEqual(RingLabel(10).text(), "%10")

        with self.assertRaisesRegex(ValueError, "positive"):
            RingLabel(0)

    def test_direction_absent_is_first_class_domain_value(self) -> None:
        self.assertIn(DirectionMark.ABSENT, tuple(DirectionMark))
        self.assertEqual(DirectionMark.ABSENT.value, 0)

    def test_atom_text_choice_renders_only_declared_tetra_tokens(self) -> None:
        choice = AtomTextChoice(
            name="organic_c",
            text_by_tetra=((TetraToken.NONE, "C"), (TetraToken.AT, "[C@H]")),
        )

        self.assertTrue(choice.permits(TetraToken.NONE))
        self.assertTrue(choice.permits(TetraToken.AT))
        self.assertFalse(choice.permits(TetraToken.ATAT))
        self.assertEqual(choice.render(TetraToken.NONE), "C")

        with self.assertRaises(KeyError):
            choice.render(TetraToken.ATAT)

    def test_policy_validates_finite_atom_domain_coverage(self) -> None:
        facts = _single_atom_facts()
        policy = SmilesPolicy(
            ring_labels=(RingLabel(1),),
            annotation_mode=AnnotationMode.HARD,
            atom_text_domains=(
                AtomTextDomain(
                    atom=AtomId(0),
                    choices=(_organic_c_choice(),),
                ),
            ),
            bond_text_domains=(),
        )

        policy.validate_for_facts(facts)
        self.assertEqual(
            policy.atom_text_domain(facts, AtomId(0)),
            (_organic_c_choice(),),
        )

    def test_policy_rejects_duplicate_ring_labels(self) -> None:
        facts = _single_atom_facts()
        policy = SmilesPolicy(
            ring_labels=(RingLabel(1), RingLabel(1)),
            annotation_mode=AnnotationMode.HARD,
            atom_text_domains=(
                AtomTextDomain(
                    atom=AtomId(0),
                    choices=(_organic_c_choice(),),
                ),
            ),
            bond_text_domains=(),
        )

        with self.assertRaisesRegex(
            ValueError,
            "ring label domain contains duplicates",
        ):
            policy.validate_for_facts(facts)

    def test_policy_rejects_missing_atom_text_domain(self) -> None:
        facts = _single_atom_facts()
        policy = SmilesPolicy(
            ring_labels=(RingLabel(1),),
            annotation_mode=AnnotationMode.HARD,
            atom_text_domains=(),
            bond_text_domains=(),
        )

        with self.assertRaisesRegex(ValueError, "atom text domain coverage mismatch"):
            policy.validate_for_facts(facts)

    def test_parser_semantics_protocol_is_explicit(self) -> None:
        self.assertIsInstance(_MinimalSemantics(), ParserSemantics)
        self.assertIs(INVALID, INVALID)
        self.assertIsInstance(INVALID, Invalid)


class _MinimalSemantics:
    def atom_decode_ok(
        self,
        facts: MoleculeFacts,
        atom: AtomId,
        atom_text: AtomTextChoice,
        tetra_token: TetraToken,
        incident_bond_texts: tuple[BondTextChoice, ...],
    ) -> bool:
        return atom in {fact.id for fact in facts.atoms} and atom_text.permits(
            tetra_token
        )

    def bond_decode_ok(
        self,
        facts: MoleculeFacts,
        bond: BondId,
        bond_text: BondTextChoice,
        direction_mark: DirectionMark,
    ) -> bool:
        return direction_mark is DirectionMark.ABSENT or bond_text.permits_direction

    def ring_pair_decode_ok(
        self,
        facts: MoleculeFacts,
        bond: BondId,
        endpoint_1: BondTextChoice,
        mark_1: DirectionMark,
        endpoint_2: BondTextChoice,
        mark_2: DirectionMark,
    ) -> bool:
        return self.bond_decode_ok(facts, bond, endpoint_1, mark_1) and (
            self.bond_decode_ok(facts, bond, endpoint_2, mark_2)
        )

    def local_tetra_order(
        self,
        facts: MoleculeFacts,
        skel: object,
        slots: object,
        site: SiteId,
    ) -> tuple[OccurrenceId, ...]:
        return ()

    def tetra_value(
        self,
        facts: MoleculeFacts,
        site: SiteId,
        local_order: tuple[OccurrenceId, ...],
        token: TetraToken,
    ) -> TetraValue | Invalid:
        return TetraValue.NONE if token is TetraToken.NONE else INVALID

    def directional_value(
        self,
        facts: MoleculeFacts,
        skel: object,
        slots: object,
        site: SiteId,
        marks: Mapping[CarrierSlotId, DirectionMark],
    ) -> DirectionalValue | Invalid:
        return DirectionalValue.NONE if not marks else INVALID


def _single_atom_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(
            AtomFacts(
                id=AtomId(0),
                atomic_num=6,
                symbol="C",
                isotope=None,
                formal_charge=0,
                is_aromatic=False,
                explicit_h_count=0,
                implicit_h_count=4,
                no_implicit=False,
            ),
        ),
        bonds=(),
        components=(ComponentFacts(ComponentId(0), (AtomId(0),), ()),),
    )


def _organic_c_choice() -> AtomTextChoice:
    return AtomTextChoice(
        name="organic_c",
        text_by_tetra=((TetraToken.NONE, "C"),),
    )


if __name__ == "__main__":
    unittest.main()
