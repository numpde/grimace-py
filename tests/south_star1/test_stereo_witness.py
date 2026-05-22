"""Tests for South Star 1 stereo witness enumeration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from itertools import permutations
import unittest

from grimace._south_star1.annotation import ValidWitness
from grimace._south_star1.constraints import NamedConstraint
from grimace._south_star1.enumerate import render_image_from_witnesses
from grimace._south_star1.facts import ComponentFacts
from grimace._south_star1.facts import DirectionalSiteFacts
from grimace._south_star1.facts import DirectionalValue
from grimace._south_star1.facts import LigandKind
from grimace._south_star1.facts import LigandOccurrence
from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.facts import SiteStatus
from grimace._south_star1.facts import StereoFacts
from grimace._south_star1.facts import TetraValue
from grimace._south_star1.graph_index import build_graph_index
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import BondSlotId
from grimace._south_star1.ids import CarrierSlotId
from grimace._south_star1.ids import ComponentId
from grimace._south_star1.ids import OccurrenceId
from grimace._south_star1.ids import RingEndpointId
from grimace._south_star1.ids import SiteId
from grimace._south_star1.policy import AnnotationMode
from grimace._south_star1.policy import AtomTextChoice
from grimace._south_star1.policy import AtomTextDomain
from grimace._south_star1.policy import BondTextChoice
from grimace._south_star1.policy import BondTextDomain
from grimace._south_star1.policy import DirectionMark
from grimace._south_star1.policy import RingLabel
from grimace._south_star1.policy import SmilesPolicy
from grimace._south_star1.policy import TetraToken
from grimace._south_star1.ring_labels import enumerate_ring_label_assignments
from grimace._south_star1.ring_labels import validate_bounded_ring_labels
from grimace._south_star1.semantics import INVALID
from grimace._south_star1.semantics import Invalid
from grimace._south_star1.skeleton import ChildEvent
from grimace._south_star1.skeleton import RingEvent
from grimace._south_star1.skeleton import TraversalSkeleton
from grimace._south_star1.skeleton import enumerate_traversal_skeletons
from grimace._south_star1.slots import RingEndpointSlot
from grimace._south_star1.slots import SlotBundle
from grimace._south_star1.slots import allocate_traversal_slots
from grimace._south_star1.slots import carrier_slot_by_bond_slot
from grimace._south_star1.stereo_csp import build_stereo_csp
from grimace._south_star1.stereo_witness import collect_stereo_witnesses_for_skeleton
from grimace._south_star1.stereo_witness import enumerate_presentation_prefixes

from tests.south_star1.helpers import atom
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import organic_atom_choice
from tests.south_star1.helpers import single_bond
from tests.south_star1.helpers import tetrahedral_facts


class StereoWitnessTest(unittest.TestCase):
    def test_tetrahedral_required_tokens_follow_ligand_order_parity(self) -> None:
        facts = tetrahedral_facts()
        semantics = _TetraOrderSemantics()
        reference = facts.stereo.tetrahedral[0].reference_order
        neighbor_ligands = reference[:3]
        implicit_h = reference[3]
        required_tokens: dict[tuple[OccurrenceId, ...], TetraToken] = {}

        for local_order in permutations(reference):
            token = _required_tetra_token(facts, semantics, local_order)
            required_tokens[local_order] = token

            for left in range(len(local_order)):
                for right in range(left + 1, len(local_order)):
                    swapped = list(local_order)
                    swapped[left], swapped[right] = swapped[right], swapped[left]
                    self.assertIsNot(
                        token,
                        _required_tetra_token(facts, semantics, tuple(swapped)),
                    )

        self.assertEqual(
            sum(token is TetraToken.AT for token in required_tokens.values()),
            12,
        )
        self.assertEqual(
            sum(token is TetraToken.ATAT for token in required_tokens.values()),
            12,
        )

        required_tokens = {}
        for order_3 in permutations(neighbor_ligands):
            local_order = order_3 + (implicit_h,)
            token = _required_tetra_token(facts, semantics, local_order)
            required_tokens[local_order] = token

            swapped = (order_3[1], order_3[0], order_3[2], implicit_h)
            self.assertIsNot(token, _required_tetra_token(facts, semantics, swapped))

        self.assertEqual(
            sum(token is TetraToken.AT for token in required_tokens.values()),
            3,
        )
        self.assertEqual(
            sum(token is TetraToken.ATAT for token in required_tokens.values()),
            3,
        )

    def test_tetrahedral_skeleton_orders_can_require_opposite_tokens(self) -> None:
        facts = tetrahedral_facts()
        semantics = _TetraOrderSemantics()
        rendered = _rendered_for_matching_skeletons(facts, semantics)

        self.assertEqual(sum("[C@H]" in value for value in rendered), 6)
        self.assertEqual(sum("[C@@H]" in value for value in rendered), 6)

    def test_unspecified_tetrahedral_site_rejects_accidental_tokens(self) -> None:
        facts = _unspecified_tetrahedral_facts()
        skeleton = _first_skeleton(facts)
        slots = allocate_traversal_slots(facts, skeleton)
        witnesses, stats = collect_stereo_witnesses_for_skeleton(
            facts=facts,
            skeleton=skeleton,
            slots=slots,
            policy=_policy_for_slots(facts, slots, chiral_center=AtomId(0)),
            semantics=_TetraOrderSemantics(),
        )

        self.assertGreater(stats.witness_count, 0)
        self.assertTrue(
            all("[C@H]" not in witness.rendered for witness in witnesses)
        )
        self.assertTrue(
            all("[C@@H]" not in witness.rendered for witness in witnesses)
        )

    def test_toy_directional_site_can_have_exactly_one_valid_carrier_pair(self) -> None:
        """Toy fixture: exercises carrier relations, not alkene SMILES spelling."""
        facts = directional_facts()
        skeleton = _first_skeleton(facts)
        slots = allocate_traversal_slots(facts, skeleton)
        scope = (slots.carrier_slots[0].id, slots.carrier_slots[1].id)
        semantics = _DirectionalPairSemantics(
            scope_by_site={SiteId(0): scope},
            required_by_site={
                SiteId(0): (DirectionMark.FWD, DirectionMark.REV),
            },
        )
        witnesses, stats = collect_stereo_witnesses_for_skeleton(
            facts=facts,
            skeleton=skeleton,
            slots=slots,
            policy=_policy_for_slots(facts, slots),
            semantics=semantics,
        )

        self.assertEqual(stats.witness_count, 1)
        self.assertEqual(len(witnesses), 1)
        self.assertIn("/", witnesses[0].rendered)
        self.assertIn("\\", witnesses[0].rendered)

    def test_directional_carriers_are_adjacent_to_center_bond(self) -> None:
        facts = directional_facts()
        skeleton = _first_skeleton(facts)
        slots = allocate_traversal_slots(facts, skeleton)
        carrier_by_bond = _carrier_by_bond(slots)
        center_bond = facts.stereo.directional[0].center_bond
        policy = _alkene_policy_for_slots(facts, slots)
        semantics = _alkene_directional_semantics(slots)
        witnesses, stats = collect_stereo_witnesses_for_skeleton(
            facts=facts,
            skeleton=skeleton,
            slots=slots,
            policy=policy,
            semantics=semantics,
        )
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
            semantics=semantics,
        )

        scoped_carriers = (carrier_by_bond[BondId(1)], carrier_by_bond[BondId(2)])
        self.assertTrue(all(carrier.bond != center_bond for carrier in scoped_carriers))
        self.assertEqual(
            csp.direction_domains[carrier_by_bond[center_bond].id],
            (DirectionMark.ABSENT,),
        )
        self.assertEqual(stats.witness_count, 1)
        self.assertIn("=", witnesses[0].rendered)
        self.assertIn("/", witnesses[0].rendered)
        self.assertIn("\\", witnesses[0].rendered)

    def test_alkene_style_directional_examples_preserve_center_double_bond(self) -> None:
        facts = directional_facts()
        rendered: list[str] = []

        for skeleton in enumerate_traversal_skeletons(
            facts,
            build_graph_index(facts),
            _policy_for_facts_only(facts),
        ):
            if skeleton.roots != (AtomId(0),):
                continue

            slots = allocate_traversal_slots(facts, skeleton)
            witnesses, _ = collect_stereo_witnesses_for_skeleton(
                facts=facts,
                skeleton=skeleton,
                slots=slots,
                policy=_alkene_policy_for_slots(facts, slots),
                semantics=_alkene_directional_semantics(slots),
            )
            rendered.extend(witness.rendered for witness in witnesses)
            if len(rendered) >= 8:
                break

        self.assertEqual(len(rendered), 8)
        self.assertTrue(all("=" in value for value in rendered))
        self.assertTrue(all("/" in value and "\\" in value for value in rendered))
        self.assertIn("C(/F)=C(\\Cl)", rendered)

    def test_shared_carrier_directional_constraints_are_global(self) -> None:
        facts = _two_site_directional_facts()
        skeleton = _first_skeleton(facts)
        slots = allocate_traversal_slots(facts, skeleton)
        shared_scope = (slots.carrier_slots[0].id,)
        semantics = _DirectionalPairSemantics(
            scope_by_site={
                SiteId(0): shared_scope,
                SiteId(1): shared_scope,
            },
            required_by_site={
                SiteId(0): (DirectionMark.FWD,),
                SiteId(1): (DirectionMark.REV,),
            },
        )
        witnesses, stats = collect_stereo_witnesses_for_skeleton(
            facts=facts,
            skeleton=skeleton,
            slots=slots,
            policy=_policy_for_slots(facts, slots),
            semantics=semantics,
        )

        self.assertEqual(stats.witness_count, 0)
        self.assertEqual(witnesses, ())
        self.assertEqual(
            _independent_directional_rows(facts, slots, semantics),
            {
                SiteId(0): ((slots.carrier_slots[0].id, DirectionMark.FWD),),
                SiteId(1): ((slots.carrier_slots[0].id, DirectionMark.REV),),
            },
        )

    def test_unspecified_directional_site_rejects_accidental_stereo(self) -> None:
        facts = _specified_and_unspecified_directional_facts()
        skeleton = _first_skeleton(facts)
        slots = allocate_traversal_slots(facts, skeleton)
        scope = (slots.carrier_slots[0].id, slots.carrier_slots[1].id)
        semantics = _DirectionalPairSemantics(
            scope_by_site={
                SiteId(0): scope,
                SiteId(1): scope,
            },
            required_by_site={
                SiteId(0): (DirectionMark.FWD, DirectionMark.REV),
                SiteId(1): (DirectionMark.FWD, DirectionMark.REV),
            },
        )
        witnesses, stats = collect_stereo_witnesses_for_skeleton(
            facts=facts,
            skeleton=skeleton,
            slots=slots,
            policy=_policy_for_slots(facts, slots),
            semantics=semantics,
        )

        self.assertEqual(stats.witness_count, 0)
        self.assertEqual(witnesses, ())

    def test_ring_endpoint_order_can_change_required_tetra_token(self) -> None:
        facts = _ring_tetrahedral_facts()
        semantics = _TetraOrderSemantics()
        rendered = _rendered_for_matching_skeletons(facts, semantics)

        self.assertTrue(any("[C@H]" in value for value in rendered))
        self.assertTrue(any("[C@@H]" in value for value in rendered))
        self.assertTrue(any("1" in value for value in rendered))

    def test_ring_tetra_local_order_uses_lexical_ring_event_position(self) -> None:
        facts = _ring_tetrahedral_facts()
        semantics = _TetraOrderSemantics()
        by_order: dict[tuple[OccurrenceId, ...], TetraToken] = {}

        for skeleton in enumerate_traversal_skeletons(
            facts,
            build_graph_index(facts),
            _policy_for_facts_only(facts),
        ):
            if skeleton.roots != (AtomId(0),):
                continue
            local_order = semantics.local_tetra_order(
                facts,
                skeleton,
                allocate_traversal_slots(facts, skeleton),
                SiteId(0),
            )
            by_order[local_order] = _required_tetra_token(
                facts,
                semantics,
                local_order,
            )

        for left, left_token in by_order.items():
            for right, right_token in by_order.items():
                if left[:1] == right[:1]:
                    continue
                if left[2:] != right[2:]:
                    continue
                if set(left[:2]) == set(right[:2]):
                    self.assertIsNot(left_token, right_token)
                    return

        raise AssertionError("no ring/local-order swap witness found")

    def test_least_free_ring_labels_are_generator_and_validator_policy(self) -> None:
        slots = _two_nonoverlapping_ring_slot_bundle()
        policy = _ring_only_policy(least_free=True)

        generated = tuple(
            enumerate_ring_label_assignments(slots=slots, policy=policy)
        )

        self.assertEqual(len(generated), 1)
        self.assertEqual(set(generated[0].values()), {RingLabel(1)})

        non_normalized = {
            endpoint.id: RingLabel(2)
            for endpoint in slots.ring_endpoints
        }
        with self.assertRaisesRegex(ValueError, "least-free label policy"):
            validate_bounded_ring_labels(policy, slots, non_normalized)

        validate_bounded_ring_labels(
            _ring_only_policy(least_free=False),
            slots,
            non_normalized,
        )

    def test_rendered_image_preserves_witness_multiplicity(self) -> None:
        constraints = (NamedConstraint("semantic_validity", "assignment"),)
        witnesses = (
            ValidWitness("witness:a", "CC", 0, constraints),
            ValidWitness("witness:b", "CC", 0, constraints),
        )

        image = render_image_from_witnesses(witnesses)

        self.assertEqual(image.witness_count, 2)
        self.assertEqual(image.distinct_count, 1)
        self.assertEqual(image.strings, ("CC", "CC"))


def _rendered_for_matching_skeletons(
    facts: MoleculeFacts,
    semantics,
) -> tuple[str, ...]:
    rendered: list[str] = []
    for skeleton in enumerate_traversal_skeletons(
        facts,
        build_graph_index(facts),
        _policy_for_facts_only(facts),
    ):
        if skeleton.roots != (AtomId(0),):
            continue

        slots = allocate_traversal_slots(facts, skeleton)
        witnesses, _ = collect_stereo_witnesses_for_skeleton(
            facts=facts,
            skeleton=skeleton,
            slots=slots,
            policy=_policy_for_slots(facts, slots, chiral_center=AtomId(0)),
            semantics=semantics,
        )
        rendered.extend(witness.rendered for witness in witnesses)

    return tuple(rendered)


def _required_tetra_token(
    facts: MoleculeFacts,
    semantics,
    local_order: tuple[OccurrenceId, ...],
) -> TetraToken:
    target = facts.stereo.tetrahedral[0].target
    matching = tuple(
        token
        for token in (TetraToken.AT, TetraToken.ATAT)
        if semantics.tetra_value(
            facts,
            SiteId(0),
            local_order,
            token,
        ) == target
    )
    if len(matching) != 1:
        raise AssertionError(
            f"expected exactly one tetra token for {local_order!r}, got {matching!r}"
        )
    return matching[0]


def _carrier_by_bond(slots: SlotBundle):
    by_slot = carrier_slot_by_bond_slot(slots)
    return {
        carrier.bond: carrier
        for carrier in by_slot.values()
    }


def _alkene_directional_semantics(slots: SlotBundle) -> "_DirectionalPairSemantics":
    carrier_by_bond = _carrier_by_bond(slots)
    scope = (
        carrier_by_bond[BondId(1)].id,
        carrier_by_bond[BondId(2)].id,
    )
    return _DirectionalPairSemantics(
        scope_by_site={SiteId(0): scope},
        required_by_site={
            SiteId(0): (DirectionMark.FWD, DirectionMark.REV),
        },
    )


def _independent_directional_rows(
    facts: MoleculeFacts,
    slots: SlotBundle,
    semantics,
) -> dict[SiteId, tuple[tuple[CarrierSlotId, DirectionMark], ...]]:
    rows = {}
    for site in facts.stereo.directional:
        scope = semantics.directional_scope(facts, object(), slots, site.id)
        row = semantics.required_by_site[site.id]
        marks = dict(zip(scope, row, strict=True))
        if semantics.directional_value(facts, object(), slots, site.id, marks) != (
            site.target
        ):
            continue
        rows[site.id] = tuple(zip(scope, row, strict=True))
    return rows


def _first_skeleton(facts: MoleculeFacts) -> TraversalSkeleton:
    return enumerate_traversal_skeletons(
        facts,
        build_graph_index(facts),
        _policy_for_facts_only(facts),
    )[0]


def _policy_for_facts_only(facts: MoleculeFacts) -> SmilesPolicy:
    return SmilesPolicy(
        ring_labels=(RingLabel(1), RingLabel(2)),
        annotation_mode=AnnotationMode.HARD,
        atom_text_domains=tuple(
            AtomTextDomain(
                atom=atom_facts.id,
                choices=(organic_atom_choice(atom_facts.symbol),),
            )
            for atom_facts in facts.atoms
        ),
        bond_text_domains=(),
    )


def _policy_for_slots(
    facts: MoleculeFacts,
    slots: SlotBundle,
    *,
    chiral_center: AtomId | None = None,
    mode: AnnotationMode = AnnotationMode.HARD,
) -> SmilesPolicy:
    atom_text = _atom_text_for_facts(facts, chiral_center=chiral_center)
    bond_choice = _directional_bond_choice()
    bond_domain_keys = sorted(
        {(slot.bond, slot.kind.value) for slot in slots.bond_slots},
        key=lambda key: (int(key[0]), key[1]),
    )
    return SmilesPolicy(
        ring_labels=(RingLabel(1), RingLabel(2)),
        annotation_mode=mode,
        atom_text_domains=tuple(
            AtomTextDomain(atom=atom.id, choices=(atom_text[atom.id],))
            for atom in facts.atoms
        ),
        bond_text_domains=tuple(
            BondTextDomain(
                bond=bond_id,
                slot_kind=kind,
                choices=(bond_choice,),
            )
            for bond_id, kind in bond_domain_keys
        ),
    )


def _alkene_policy_for_slots(
    facts: MoleculeFacts,
    slots: SlotBundle,
) -> SmilesPolicy:
    atom_text = _atom_text_for_facts(facts)
    center_bond = facts.stereo.directional[0].center_bond
    bond_domain_keys = sorted(
        {(slot.bond, slot.kind.value) for slot in slots.bond_slots},
        key=lambda key: (int(key[0]), key[1]),
    )
    return SmilesPolicy(
        ring_labels=(RingLabel(1), RingLabel(2)),
        annotation_mode=AnnotationMode.HARD,
        atom_text_domains=tuple(
            AtomTextDomain(atom=atom.id, choices=(atom_text[atom.id],))
            for atom in facts.atoms
        ),
        bond_text_domains=tuple(
            BondTextDomain(
                bond=bond_id,
                slot_kind=kind,
                choices=(_alkene_bond_choice(bond_id == center_bond),),
            )
            for bond_id, kind in bond_domain_keys
        ),
    )


def _alkene_bond_choice(is_center_bond: bool) -> BondTextChoice:
    if is_center_bond:
        return BondTextChoice(
            name="double",
            base_text="=",
            permits_direction=False,
        )
    return _directional_bond_choice()


def _atom_text_for_facts(
    facts: MoleculeFacts,
    *,
    chiral_center: AtomId | None = None,
) -> dict[AtomId, AtomTextChoice]:
    return {
        atom_facts.id: _chiral_carbon_choice()
        if atom_facts.id == chiral_center
        else organic_atom_choice(atom_facts.symbol)
        for atom_facts in facts.atoms
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


def _directional_bond_choice() -> BondTextChoice:
    # Toy witness policy: useful for isolating carrier constraints before
    # SMILES-like bond spelling is introduced.  Alkene-style tests use
    # _alkene_policy_for_slots so the center double bond renders as "=".
    return BondTextChoice(
        name="single_or_directional",
        base_text="",
        permits_direction=True,
    )


def _ring_only_policy(*, least_free: bool) -> SmilesPolicy:
    return SmilesPolicy(
        ring_labels=(RingLabel(1), RingLabel(2)),
        annotation_mode=AnnotationMode.HARD,
        atom_text_domains=(),
        bond_text_domains=(),
        least_free_ring_labels=least_free,
    )


def _unspecified_tetrahedral_facts() -> MoleculeFacts:
    facts = tetrahedral_facts()
    site = replace(
        facts.stereo.tetrahedral[0],
        status=SiteStatus.UNSPECIFIED,
        target=TetraValue.NONE,
    )
    return replace(facts, stereo=StereoFacts(tetrahedral=(site,)))


def _two_site_directional_facts() -> MoleculeFacts:
    facts = directional_facts()
    site_0 = facts.stereo.directional[0]
    site_1 = DirectionalSiteFacts(
        id=SiteId(1),
        center_bond=site_0.center_bond,
        left_endpoint=site_0.left_endpoint,
        right_endpoint=site_0.right_endpoint,
        status=SiteStatus.SPECIFIED,
        target=DirectionalValue.OPPOSITE,
        left_ligands=(OccurrenceId(2),),
        right_ligands=(OccurrenceId(3),),
        reference_pair=(OccurrenceId(2), OccurrenceId(3)),
    )
    occurrence_2 = replace(
        facts.ligand_occurrences[0],
        id=OccurrenceId(2),
        site=SiteId(1),
    )
    occurrence_3 = replace(
        facts.ligand_occurrences[1],
        id=OccurrenceId(3),
        site=SiteId(1),
    )
    return replace(
        facts,
        stereo=StereoFacts(directional=(site_0, site_1)),
        ligand_occurrences=facts.ligand_occurrences + (occurrence_2, occurrence_3),
    )


def _specified_and_unspecified_directional_facts() -> MoleculeFacts:
    facts = _two_site_directional_facts()
    site_0, site_1 = facts.stereo.directional
    return replace(
        facts,
        stereo=StereoFacts(
            directional=(
                site_0,
                replace(
                    site_1,
                    status=SiteStatus.UNSPECIFIED,
                    target=DirectionalValue.NONE,
                ),
            )
        ),
    )


def _ring_tetrahedral_facts() -> MoleculeFacts:
    site = SiteId(0)
    return MoleculeFacts(
        atoms=(atom(0, "C"), atom(1, "C"), atom(2, "C"), atom(3, "F")),
        bonds=(
            single_bond(0, 0, 1),
            single_bond(1, 1, 2),
            single_bond(2, 2, 0),
            single_bond(3, 0, 3),
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
            LigandOccurrence(
                id=OccurrenceId(0),
                site=site,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(1),
                bond=BondId(0),
            ),
            LigandOccurrence(
                id=OccurrenceId(1),
                site=site,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(2),
                bond=BondId(2),
            ),
            LigandOccurrence(
                id=OccurrenceId(2),
                site=site,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(3),
                bond=BondId(3),
            ),
            LigandOccurrence(
                id=OccurrenceId(3),
                site=site,
                kind=LigandKind.IMPLICIT_H,
                atom=AtomId(0),
                bond=None,
            ),
        ),
    )


def _two_nonoverlapping_ring_slot_bundle() -> SlotBundle:
    return SlotBundle(
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
                syntax_position=1,
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
                syntax_position=3,
            ),
        ),
    )


class _TetraOrderSemantics:
    def atom_decode_ok(
        self,
        facts: MoleculeFacts,
        atom_id: AtomId,
        atom_text: AtomTextChoice,
        tetra_token: TetraToken,
        incident_bond_texts: tuple[BondTextChoice, ...],
    ) -> bool:
        return atom_id in {atom.id for atom in facts.atoms} and atom_text.permits(
            tetra_token
        )

    def bond_decode_ok(
        self,
        facts: MoleculeFacts,
        bond_id: BondId,
        bond_text: BondTextChoice,
        direction_mark: DirectionMark,
    ) -> bool:
        return direction_mark is DirectionMark.ABSENT or bond_text.permits_direction

    def ring_pair_decode_ok(
        self,
        facts: MoleculeFacts,
        bond_id: BondId,
        endpoint_1: BondTextChoice,
        mark_1: DirectionMark,
        endpoint_2: BondTextChoice,
        mark_2: DirectionMark,
    ) -> bool:
        return self.bond_decode_ok(facts, bond_id, endpoint_1, mark_1) and (
            self.bond_decode_ok(facts, bond_id, endpoint_2, mark_2)
        )

    def local_tetra_order(
        self,
        facts: MoleculeFacts,
        skel: TraversalSkeleton,
        slots: object,
        site: SiteId,
    ) -> tuple[OccurrenceId, ...]:
        center = facts.stereo.tetrahedral[0].center
        occurrence_by_atom = {
            occurrence.atom: occurrence.id
            for occurrence in facts.ligand_occurrences
            if occurrence.kind is LigandKind.NEIGHBOR_ATOM
        }
        order: list[OccurrenceId] = []
        for event in skel.events_at[center]:
            if isinstance(event, ChildEvent):
                order.append(occurrence_by_atom[event.child])
            elif isinstance(event, RingEvent):
                order.append(occurrence_by_atom[event.other_atom])
        order.extend(
            occurrence.id
            for occurrence in facts.ligand_occurrences
            if occurrence.kind is LigandKind.IMPLICIT_H
        )
        return tuple(order)

    def tetra_value(
        self,
        facts: MoleculeFacts,
        site: SiteId,
        local_order: tuple[OccurrenceId, ...],
        token: TetraToken,
    ) -> TetraValue | Invalid:
        if token is TetraToken.NONE:
            return TetraValue.NONE
        reference = facts.stereo.tetrahedral[0].reference_order
        if set(local_order) != set(reference) or len(local_order) != len(reference):
            return INVALID
        is_even = _is_even_permutation(
            tuple(reference.index(occurrence) for occurrence in local_order)
        )
        if token is TetraToken.AT:
            return TetraValue.PLUS if is_even else TetraValue.MINUS
        if token is TetraToken.ATAT:
            return TetraValue.MINUS if is_even else TetraValue.PLUS
        return INVALID

    def directional_scope(
        self,
        facts: MoleculeFacts,
        skel: object,
        slots: object,
        site: SiteId,
    ) -> tuple[CarrierSlotId, ...]:
        return ()

    def directional_value(
        self,
        facts: MoleculeFacts,
        skel: object,
        slots: object,
        site: SiteId,
        marks: Mapping[CarrierSlotId, DirectionMark],
    ) -> DirectionalValue | Invalid:
        return DirectionalValue.NONE if not marks else INVALID


class _DirectionalPairSemantics(_TetraOrderSemantics):
    def __init__(
        self,
        *,
        scope_by_site: Mapping[SiteId, tuple[CarrierSlotId, ...]],
        required_by_site: Mapping[SiteId, tuple[DirectionMark, ...]],
    ) -> None:
        self.scope_by_site = dict(scope_by_site)
        self.required_by_site = dict(required_by_site)

    def directional_scope(
        self,
        facts: MoleculeFacts,
        skel: object,
        slots: object,
        site: SiteId,
    ) -> tuple[CarrierSlotId, ...]:
        return self.scope_by_site[site]

    def directional_value(
        self,
        facts: MoleculeFacts,
        skel: object,
        slots: object,
        site: SiteId,
        marks: Mapping[CarrierSlotId, DirectionMark],
    ) -> DirectionalValue | Invalid:
        scope = self.scope_by_site[site]
        if tuple(marks) != scope:
            return INVALID
        row = tuple(marks[carrier] for carrier in scope)
        if row == self.required_by_site[site]:
            return DirectionalValue.OPPOSITE
        return DirectionalValue.NONE


def _is_even_permutation(indices: tuple[int, ...]) -> bool:
    inversions = 0
    for left, value in enumerate(indices):
        for other in indices[left + 1 :]:
            if value > other:
                inversions += 1
    return inversions % 2 == 0


if __name__ == "__main__":
    unittest.main()
