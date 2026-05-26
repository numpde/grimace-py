"""Tests for South Star 1 finite stereo-CSP assignment generation."""

from __future__ import annotations

from collections.abc import Mapping
import unittest

from grimace._south_star1.facts import DirectionalValue
from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.facts import TetraValue
from grimace._south_star1.graph_index import build_graph_index
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import CarrierSlotId
from grimace._south_star1.ids import OccurrenceId
from grimace._south_star1.ids import SiteId
from grimace._south_star1.policy import AnnotationMode
from grimace._south_star1.policy import AtomTextChoice
from grimace._south_star1.policy import AtomTextDomain
from grimace._south_star1.policy import BondTextChoice
from grimace._south_star1.policy import BondTextDomain
from grimace._south_star1.policy import BranchPresentationMode
from grimace._south_star1.policy import DirectionMark
from grimace._south_star1.policy import RingLabel
from grimace._south_star1.policy import SmilesPolicy
from grimace._south_star1.policy import TetraToken
from grimace._south_star1.policy import with_branch_presentation_mode
from grimace._south_star1.render import render_stereo_traversal
from grimace._south_star1.semantics import INVALID
from grimace._south_star1.semantics import Invalid
from grimace._south_star1.skeleton import enumerate_traversal_skeletons
from grimace._south_star1.slots import SlotBundle
from grimace._south_star1.slots import allocate_traversal_slots
from grimace._south_star1.stereo_csp import PresentationPrefix
from grimace._south_star1.stereo_csp import build_stereo_csp
from grimace._south_star1.stereo_csp import enumerate_stereo_assignments_for_prefix
from grimace._south_star1.stereo_csp import select_stereo_solutions
from grimace._south_star1.stereo_csp import solve_stereo_csp

from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import organic_atom_choice
from tests.south_star1.helpers import tetrahedral_facts


class StereoCSPTest(unittest.TestCase):
    def test_csp_assignments_feed_tetra_tokens_to_stereo_renderer(self) -> None:
        facts = tetrahedral_facts()
        skeleton = _first_skeleton(facts)
        slots = allocate_traversal_slots(facts, skeleton)
        atom_text = _atom_text_for_facts(facts, chiral_center=AtomId(0))
        bond_choice = _plain_bond_choice()
        semantics = _ScopedDirectionalSemantics(scope=())
        policy = _policy_for_slots(
            facts,
            slots,
            mode=AnnotationMode.HARD,
            atom_text=atom_text,
            bond_choice=bond_choice,
        )
        prefix = _prefix_for_slots(
            facts,
            slots,
            atom_text=atom_text,
            bond_choice=bond_choice,
        )

        assignments = tuple(
            enumerate_stereo_assignments_for_prefix(
                facts=facts,
                skeleton=skeleton,
                slots=slots,
                prefix=prefix,
                policy=policy,
                semantics=semantics,
            )
        )

        self.assertEqual(len(assignments), 1)
        self.assertIs(assignments[0].tetra_tokens[AtomId(0)], TetraToken.AT)
        self.assertTrue(
            render_stereo_traversal(
                facts,
                skeleton,
                slots,
                assignments[0],
                policy,
                semantics,
            ).startswith("[C@H]")
        )

    def test_csp_assignments_feed_stereo_renderer_through_declared_scope(
        self,
    ) -> None:
        facts = directional_facts()
        skeleton = _first_skeleton(facts)
        slots = allocate_traversal_slots(facts, skeleton)
        semantics = _ScopedDirectionalSemantics(
            scope=(slots.carrier_slots[0].id,),
        )
        policy = _policy_for_slots(facts, slots, mode=AnnotationMode.HARD)
        prefix = _prefix_for_slots(facts, slots)

        assignments = tuple(
            enumerate_stereo_assignments_for_prefix(
                facts=facts,
                skeleton=skeleton,
                slots=slots,
                prefix=prefix,
                policy=policy,
                semantics=semantics,
            )
        )

        self.assertEqual(
            tuple(
                assignment.direction_marks[slots.carrier_slots[0].id]
                for assignment in assignments
            ),
            (DirectionMark.FWD, DirectionMark.REV),
        )
        self.assertEqual(
            tuple(
                render_stereo_traversal(
                    facts,
                    skeleton,
                    slots,
                    assignment,
                    policy,
                    semantics,
                )
                for assignment in assignments
            ),
            ("C(/C(Cl))(F)", "C(\\C(Cl))(F)"),
        )

    def test_support_maximal_policy_keeps_inclusion_maximal_marker_sets(
        self,
    ) -> None:
        facts = directional_facts()
        skeleton = _first_skeleton(facts)
        slots = allocate_traversal_slots(facts, skeleton)
        semantics = _ScopedDirectionalSemantics(
            scope=tuple(carrier.id for carrier in slots.carrier_slots),
        )
        policy = _policy_for_slots(
            facts,
            slots,
            mode=AnnotationMode.SUPPORT_MAXIMAL,
        )
        prefix = _prefix_for_slots(facts, slots)

        assignments = tuple(
            enumerate_stereo_assignments_for_prefix(
                facts=facts,
                skeleton=skeleton,
                slots=slots,
                prefix=prefix,
                policy=policy,
                semantics=semantics,
            )
        )

        self.assertEqual(len(assignments), 8)
        self.assertTrue(
            all(
                all(
                    mark is not DirectionMark.ABSENT
                    for mark in assignment.direction_marks.values()
                )
                for assignment in assignments
            )
        )

    def test_writer_shaped_policy_does_not_change_csp_feasible_selected_counts_for_fixed_prefix(
        self,
    ) -> None:
        facts = directional_facts()
        skeleton = _first_skeleton(facts)
        slots = allocate_traversal_slots(facts, skeleton)
        semantics = _ScopedDirectionalSemantics(
            scope=tuple(carrier.id for carrier in slots.carrier_slots),
        )
        policy = _policy_for_slots(
            facts,
            slots,
            mode=AnnotationMode.SUPPORT_MAXIMAL,
        )
        writer_policy = with_branch_presentation_mode(
            policy,
            BranchPresentationMode.WRITER_SHAPED,
        )
        prefix = _prefix_for_slots(facts, slots)

        self.assertEqual(
            _csp_solution_counts(
                facts=facts,
                skeleton=skeleton,
                slots=slots,
                prefix=prefix,
                policy=writer_policy,
                semantics=semantics,
            ),
            _csp_solution_counts(
                facts=facts,
                skeleton=skeleton,
                slots=slots,
                prefix=prefix,
                policy=policy,
                semantics=semantics,
            ),
        )


def _first_skeleton(facts: MoleculeFacts):
    return enumerate_traversal_skeletons(
        facts,
        build_graph_index(facts),
        _policy_for_facts_only(facts),
    )[0]


def _csp_solution_counts(
    *,
    facts: MoleculeFacts,
    skeleton,
    slots: SlotBundle,
    prefix: PresentationPrefix,
    policy: SmilesPolicy,
    semantics,
) -> tuple[int, int]:
    csp = build_stereo_csp(
        facts=facts,
        skeleton=skeleton,
        slots=slots,
        prefix=prefix,
        policy=policy,
        semantics=semantics,
    )
    feasible = tuple(solve_stereo_csp(csp))
    selected = tuple(
        select_stereo_solutions(
            csp=csp,
            solutions=feasible,
            mode=policy.annotation_mode,
        )
    )
    return (len(feasible), len(selected))


def _policy_for_slots(
    facts: MoleculeFacts,
    slots: SlotBundle,
    *,
    mode: AnnotationMode,
    atom_text: Mapping[AtomId, AtomTextChoice] | None = None,
    bond_choice: BondTextChoice | None = None,
) -> SmilesPolicy:
    if atom_text is None:
        atom_text = _atom_text_for_facts(facts)
    if bond_choice is None:
        bond_choice = _directional_bond_choice()
    bond_domain_keys = sorted(
        {(slot.bond, slot.kind.value) for slot in slots.bond_slots},
        key=lambda key: (int(key[0]), key[1]),
    )
    return SmilesPolicy(
        ring_labels=(RingLabel(1), RingLabel(2)),
        annotation_mode=mode,
        atom_text_domains=tuple(
            AtomTextDomain(
                atom=atom.id,
                choices=(atom_text[atom.id],),
            )
            for atom in facts.atoms
        ),
        bond_text_domains=tuple(
            BondTextDomain(
                bond=bond,
                slot_kind=kind,
                choices=(bond_choice,),
            )
            for bond, kind in bond_domain_keys
        ),
    )


def _policy_for_facts_only(facts: MoleculeFacts) -> SmilesPolicy:
    return SmilesPolicy(
        ring_labels=(RingLabel(1), RingLabel(2)),
        annotation_mode=AnnotationMode.HARD,
        atom_text_domains=tuple(
            AtomTextDomain(
                atom=atom.id,
                choices=(organic_atom_choice(atom.symbol),),
            )
            for atom in facts.atoms
        ),
        bond_text_domains=(),
    )


def _prefix_for_slots(
    facts: MoleculeFacts,
    slots: SlotBundle,
    *,
    atom_text: Mapping[AtomId, AtomTextChoice] | None = None,
    bond_choice: BondTextChoice | None = None,
) -> PresentationPrefix:
    if atom_text is None:
        atom_text = _atom_text_for_facts(facts)
    if bond_choice is None:
        bond_choice = _directional_bond_choice()
    return PresentationPrefix(
        atom_text=dict(atom_text),
        bond_text={slot.id: bond_choice for slot in slots.bond_slots},
        ring_labels={},
    )


def _atom_text_for_facts(
    facts: MoleculeFacts,
    *,
    chiral_center: AtomId | None = None,
) -> dict[AtomId, AtomTextChoice]:
    return {
        atom.id: _chiral_carbon_choice()
        if atom.id == chiral_center
        else organic_atom_choice(atom.symbol)
        for atom in facts.atoms
    }


def _chiral_carbon_choice() -> AtomTextChoice:
    return AtomTextChoice(
        name="chiral_c",
        text_by_tetra=(
            (TetraToken.NONE, "C"),
            (TetraToken.AT, "[C@H]"),
            (TetraToken.ATAT, "[C@@H]"),
        ),
    )


def _plain_bond_choice() -> BondTextChoice:
    return BondTextChoice(
        name="plain_single",
        base_text="",
        permits_direction=False,
    )


def _directional_bond_choice() -> BondTextChoice:
    return BondTextChoice(
        name="single_or_directional",
        base_text="",
        permits_direction=True,
    )


class _ScopedDirectionalSemantics:
    def __init__(self, *, scope: tuple[CarrierSlotId, ...]) -> None:
        self.scope = scope

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
        if token is TetraToken.NONE:
            return TetraValue.NONE
        if token is TetraToken.AT:
            return TetraValue.PLUS
        if token is TetraToken.ATAT:
            return TetraValue.MINUS
        return INVALID

    def directional_scope(
        self,
        facts: MoleculeFacts,
        skel: object,
        slots: object,
        site: SiteId,
    ) -> tuple[CarrierSlotId, ...]:
        return self.scope

    def directional_value(
        self,
        facts: MoleculeFacts,
        skel: object,
        slots: object,
        site: SiteId,
        marks: Mapping[CarrierSlotId, DirectionMark],
    ) -> DirectionalValue | Invalid:
        if tuple(marks) != self.scope:
            return INVALID
        if any(mark is not DirectionMark.ABSENT for mark in marks.values()):
            return DirectionalValue.OPPOSITE
        return DirectionalValue.NONE


if __name__ == "__main__":
    unittest.main()
