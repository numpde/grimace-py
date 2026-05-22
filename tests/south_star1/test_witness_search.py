"""Tests for South Star 1 finite non-stereo witness search."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from grimace._south_star1.facts import ComponentFacts
from grimace._south_star1.facts import DirectionalValue
from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.facts import TetraValue
from grimace._south_star1.graph_index import build_graph_index
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import ComponentId
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
from grimace._south_star1.skeleton import enumerate_traversal_skeletons
from grimace._south_star1.slots import BondSlotKind
from grimace._south_star1.slots import allocate_traversal_slots
from grimace._south_star1.witness_search import enumerate_nonstereo_assignments
from grimace._south_star1.witness_search import enumerate_nonstereo_support
from grimace._south_star1.witness_search import enumerate_nonstereo_witnesses
from grimace._south_star1.witness_search import enumerate_ring_label_assignments

from tests.south_star1.helpers import cco_facts
from tests.south_star1.helpers import cyclopropane_facts
from tests.south_star1.helpers import atom
from tests.south_star1.helpers import organic_atom_choice
from tests.south_star1.helpers import single_bond


class WitnessSearchTest(unittest.TestCase):
    def test_one_atom_support_has_one_string(self) -> None:
        facts = _single_atom_facts()
        policy = _policy_with_bond_domains(facts)

        image = enumerate_nonstereo_support(facts, policy, _MinimalSemantics())

        self.assertEqual(image.witness_count, 1)
        self.assertEqual(image.distinct_count, 1)
        self.assertEqual(image.strings, ("C",))

    def test_two_atom_tree_roots_can_collapse_duplicate_rendered_strings(self) -> None:
        facts = _two_carbon_facts()
        policy = _policy_with_bond_domains(facts)

        image = enumerate_nonstereo_support(facts, policy, _MinimalSemantics())

        self.assertGreater(image.witness_count, image.distinct_count)
        self.assertIn("CC", image.strings)

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

    def test_triangle_skeletons_force_ring_endpoint_slots_and_labels(self) -> None:
        facts = cyclopropane_facts()
        policy = _policy_with_bond_domains(facts)
        skeletons = enumerate_traversal_skeletons(
            facts,
            build_graph_index(facts),
            policy,
        )

        self.assertTrue(skeletons)
        for skeleton in skeletons:
            slots = allocate_traversal_slots(facts, skeleton)
            label_assignments = tuple(enumerate_ring_label_assignments(policy, slots))

            self.assertEqual(len(skeleton.ring_bonds), 1)
            self.assertEqual(len(slots.ring_endpoints), 2)
            self.assertTrue(label_assignments)
            self.assertTrue(
                all(
                    len(set(labels.values())) == 1
                    for labels in label_assignments
                )
            )

    def test_assignment_generation_is_nonstereo_only(self) -> None:
        facts = cco_facts()
        policy = _policy_with_bond_domains(facts)
        witness = next(enumerate_nonstereo_witnesses(facts, policy, _MinimalSemantics()))

        self.assertEqual(witness.annotation_count, 0)
        self.assertIn("nonstereo_only", {constraint.name for constraint in witness.constraints})

    def test_generated_nonstereo_assignments_have_no_stereo_annotations(self) -> None:
        facts = cyclopropane_facts()
        policy = _policy_with_bond_domains(facts)
        skeleton = enumerate_traversal_skeletons(
            facts,
            build_graph_index(facts),
            policy,
        )[0]
        slots = allocate_traversal_slots(facts, skeleton)

        assignments = tuple(
            enumerate_nonstereo_assignments(
                facts=facts,
                skeleton=skeleton,
                slots=slots,
                policy=policy,
            )
        )

        self.assertTrue(assignments)
        for assignment in assignments:
            self.assertTrue(
                all(
                    token is TetraToken.NONE
                    for token in assignment.tetra_tokens.values()
                )
            )
            self.assertTrue(
                all(
                    mark is DirectionMark.ABSENT
                    for mark in assignment.direction_marks.values()
                )
            )

    def test_witness_search_module_has_no_rdkit_import(self) -> None:
        path = Path("python/grimace/_south_star1/witness_search.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))

        self.assertFalse(_imports_rdkit(tree))


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


def _single_atom_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, "C"),),
        bonds=(),
        components=(ComponentFacts(ComponentId(0), (AtomId(0),), ()),),
    )


def _two_carbon_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, "C"), atom(1, "C")),
        bonds=(single_bond(0, 0, 1),),
        components=(
            ComponentFacts(
                ComponentId(0),
                (AtomId(0), AtomId(1)),
                (BondId(0),),
            ),
        ),
    )


def _imports_rdkit(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                alias.name == "rdkit" or alias.name.startswith("rdkit.")
                for alias in node.names
            ):
                return True
        if isinstance(node, ast.ImportFrom):
            if node.module == "rdkit" or (node.module or "").startswith("rdkit."):
                return True
    return False


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

    def directional_scope(self, facts, skel, slots, site: SiteId):
        return ()

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
