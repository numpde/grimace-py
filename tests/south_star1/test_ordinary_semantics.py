"""Tests for the bounded ordinary South Star SMILES semantics."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
import unittest

from grimace._south_star1.facts import BondOrder
from grimace._south_star1.facts import ComponentFacts
from grimace._south_star1.facts import DirectionalSiteFacts
from grimace._south_star1.facts import DirectionalValue
from grimace._south_star1.facts import LigandKind
from grimace._south_star1.facts import LigandOccurrence
from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.facts import SiteStatus
from grimace._south_star1.facts import StereoFacts
from grimace._south_star1.graph_index import build_graph_index
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import CarrierSlotId
from grimace._south_star1.ids import ComponentId
from grimace._south_star1.ids import OccurrenceId
from grimace._south_star1.ids import SiteId
from grimace._south_star1.ordinary_policy import OrdinaryPolicyOptions
from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.ordinary_semantics import OrdinarySmilesSemantics
from grimace._south_star1.policy import AnnotationMode
from grimace._south_star1.policy import DirectionMark
from grimace._south_star1.policy import SmilesPolicy
from grimace._south_star1.semantics import INVALID
from grimace._south_star1.skeleton import enumerate_traversal_skeletons
from grimace._south_star1.slots import CarrierSlot
from grimace._south_star1.slots import allocate_traversal_slots
from grimace._south_star1.slots import carrier_slot_by_bond_slot
from grimace._south_star1.stereo_csp import build_stereo_csp
from grimace._south_star1.stereo_csp import select_stereo_solutions
from grimace._south_star1.stereo_csp import solve_stereo_csp
from grimace._south_star1.stereo_witness import enumerate_presentation_prefixes
from grimace._south_star1.support_enumeration import enumerate_stereo_support

from tests.south_star1.helpers import atom
from tests.south_star1.helpers import bond
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import single_bond
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

    def test_directional_reference_pair_flip_changes_value(self) -> None:
        facts = four_substituent_directional_facts()
        policy = _ordinary_policy(facts)
        skeleton = _first_skeleton(facts, policy)
        slots = allocate_traversal_slots(facts, skeleton)
        marks = _marks_for_normalized_values(
            slots,
            left={BondId(1): 1, BondId(2): 1},
            right={BondId(3): 1, BondId(4): 1},
        )
        semantics = OrdinarySmilesSemantics()

        original = semantics.directional_value(
            facts,
            skeleton,
            slots,
            SiteId(0),
            marks,
        )
        left_flipped = semantics.directional_value(
            _with_directional_reference(facts, (OccurrenceId(1), OccurrenceId(2))),
            skeleton,
            slots,
            SiteId(0),
            marks,
        )
        both_flipped = semantics.directional_value(
            _with_directional_reference(facts, (OccurrenceId(1), OccurrenceId(3))),
            skeleton,
            slots,
            SiteId(0),
            marks,
        )

        self.assertEqual(original, DirectionalValue.TOGETHER)
        self.assertEqual(left_flipped, DirectionalValue.OPPOSITE)
        self.assertEqual(both_flipped, DirectionalValue.TOGETHER)

    def test_four_substituent_directional_support_can_mark_all_carriers(self) -> None:
        facts = four_substituent_directional_facts(target=DirectionalValue.OPPOSITE)
        policy = _ordinary_policy(
            facts,
            annotation_mode=AnnotationMode.SUPPORT_MAXIMAL,
        )
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
        selected = select_stereo_solutions(
            csp=csp,
            solutions=tuple(solve_stereo_csp(csp)),
            mode=AnnotationMode.SUPPORT_MAXIMAL,
        )

        self.assertTrue(selected)
        self.assertTrue(
            all(len(solution.marker_support) == 4 for solution in selected)
        )

    def test_same_endpoint_directional_disagreement_is_invalid(self) -> None:
        facts = four_substituent_directional_facts()
        policy = _ordinary_policy(facts)
        skeleton = _first_skeleton(facts, policy)
        slots = allocate_traversal_slots(facts, skeleton)
        marks = _marks_for_raw_values(
            slots,
            left={BondId(1): 1, BondId(2): 1},
            right={BondId(3): 1},
        )

        value = OrdinarySmilesSemantics().directional_value(
            facts,
            skeleton,
            slots,
            SiteId(0),
            marks,
        )

        self.assertIs(value, INVALID)

    def test_implicit_h_directional_reference_flips_printed_ligand(self) -> None:
        facts = implicit_h_directional_facts(
            reference_pair=(OccurrenceId(0), OccurrenceId(2))
        )
        policy = _ordinary_policy(facts)
        skeleton = _first_skeleton(facts, policy)
        slots = allocate_traversal_slots(facts, skeleton)
        marks = _marks_for_raw_values(
            slots,
            left={BondId(1): 1},
            right={BondId(2): 1},
        )
        semantics = OrdinarySmilesSemantics()

        implicit_h_reference = semantics.directional_value(
            facts,
            skeleton,
            slots,
            SiteId(0),
            marks,
        )
        printed_reference = semantics.directional_value(
            _with_directional_reference(facts, (OccurrenceId(1), OccurrenceId(2))),
            skeleton,
            slots,
            SiteId(0),
            marks,
        )

        self.assertEqual(implicit_h_reference, DirectionalValue.OPPOSITE)
        self.assertEqual(printed_reference, DirectionalValue.TOGETHER)

    def test_unspecified_directional_site_uses_reference_only_for_non_none(self) -> None:
        facts = four_substituent_directional_facts(
            status=SiteStatus.UNSPECIFIED,
            target=DirectionalValue.NONE,
            reference_pair=None,
        )
        policy = _ordinary_policy(facts)
        skeleton = _first_skeleton(facts, policy)
        slots = allocate_traversal_slots(facts, skeleton)
        carrier_by_bond = _carrier_by_bond(slots)
        one_sided_marks = {
            carrier_by_bond[BondId(1)].id: DirectionMark.FWD,
            carrier_by_bond[BondId(2)].id: DirectionMark.REV,
        }
        two_sided_marks = _marks_for_normalized_values(
            slots,
            left={BondId(1): 1},
            right={BondId(3): 1},
        )
        semantics = OrdinarySmilesSemantics()

        self.assertEqual(
            semantics.directional_value(facts, skeleton, slots, SiteId(0), {}),
            DirectionalValue.NONE,
        )
        self.assertEqual(
            semantics.directional_value(
                facts,
                skeleton,
                slots,
                SiteId(0),
                one_sided_marks,
            ),
            DirectionalValue.NONE,
        )
        self.assertNotEqual(
            semantics.directional_value(
                facts,
                skeleton,
                slots,
                SiteId(0),
                two_sided_marks,
            ),
            DirectionalValue.NONE,
        )

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
    annotation_mode: AnnotationMode = AnnotationMode.HARD,
) -> SmilesPolicy:
    if chiral_center is not None:
        self_declared_tetra_centers = {
            site.center for site in facts.stereo.tetrahedral
        }
        if chiral_center not in self_declared_tetra_centers:
            raise ValueError(f"not a declared tetrahedral center: {chiral_center!r}")
    return ordinary_policy_for_facts(
        facts,
        OrdinaryPolicyOptions(
            ring_label_values=(1, 2),
            annotation_mode=annotation_mode,
        ),
    )


def four_substituent_directional_facts(
    *,
    status: SiteStatus = SiteStatus.SPECIFIED,
    target: DirectionalValue = DirectionalValue.TOGETHER,
    reference_pair: tuple[OccurrenceId, OccurrenceId] | None = (
        OccurrenceId(0),
        OccurrenceId(2),
    ),
) -> MoleculeFacts:
    site_id = SiteId(0)
    return MoleculeFacts(
        atoms=(
            atom(0, "C"),
            atom(1, "C"),
            atom(2, "F"),
            atom(3, "Cl"),
            atom(4, "Br"),
            atom(5, "O"),
        ),
        bonds=(
            bond(0, 0, 1, BondOrder.DOUBLE),
            single_bond(1, 0, 2),
            single_bond(2, 0, 3),
            single_bond(3, 1, 4),
            single_bond(4, 1, 5),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(
                    AtomId(0),
                    AtomId(1),
                    AtomId(2),
                    AtomId(3),
                    AtomId(4),
                    AtomId(5),
                ),
                bonds=(
                    BondId(0),
                    BondId(1),
                    BondId(2),
                    BondId(3),
                    BondId(4),
                ),
            ),
        ),
        stereo=StereoFacts(
            directional=(
                DirectionalSiteFacts(
                    id=site_id,
                    center_bond=BondId(0),
                    left_endpoint=AtomId(0),
                    right_endpoint=AtomId(1),
                    status=status,
                    target=target,
                    left_ligands=(OccurrenceId(0), OccurrenceId(1)),
                    right_ligands=(OccurrenceId(2), OccurrenceId(3)),
                    reference_pair=reference_pair,
                ),
            ),
        ),
        ligand_occurrences=(
            LigandOccurrence(
                id=OccurrenceId(0),
                site=site_id,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(2),
                bond=BondId(1),
            ),
            LigandOccurrence(
                id=OccurrenceId(1),
                site=site_id,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(3),
                bond=BondId(2),
            ),
            LigandOccurrence(
                id=OccurrenceId(2),
                site=site_id,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(4),
                bond=BondId(3),
            ),
            LigandOccurrence(
                id=OccurrenceId(3),
                site=site_id,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(5),
                bond=BondId(4),
            ),
        ),
    )


def implicit_h_directional_facts(
    *,
    reference_pair: tuple[OccurrenceId, OccurrenceId],
) -> MoleculeFacts:
    site_id = SiteId(0)
    return MoleculeFacts(
        atoms=(atom(0, "C"), atom(1, "C"), atom(2, "F"), atom(3, "Cl")),
        bonds=(
            bond(0, 0, 1, BondOrder.DOUBLE),
            single_bond(1, 0, 2),
            single_bond(2, 1, 3),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1), AtomId(2), AtomId(3)),
                bonds=(BondId(0), BondId(1), BondId(2)),
            ),
        ),
        stereo=StereoFacts(
            directional=(
                DirectionalSiteFacts(
                    id=site_id,
                    center_bond=BondId(0),
                    left_endpoint=AtomId(0),
                    right_endpoint=AtomId(1),
                    status=SiteStatus.SPECIFIED,
                    target=DirectionalValue.OPPOSITE,
                    left_ligands=(OccurrenceId(0), OccurrenceId(1)),
                    right_ligands=(OccurrenceId(2),),
                    reference_pair=reference_pair,
                ),
            ),
        ),
        ligand_occurrences=(
            LigandOccurrence(
                id=OccurrenceId(0),
                site=site_id,
                kind=LigandKind.IMPLICIT_H,
                atom=AtomId(0),
                bond=None,
            ),
            LigandOccurrence(
                id=OccurrenceId(1),
                site=site_id,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(2),
                bond=BondId(1),
            ),
            LigandOccurrence(
                id=OccurrenceId(2),
                site=site_id,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(3),
                bond=BondId(2),
            ),
        ),
    )


def _with_directional_reference(
    facts: MoleculeFacts,
    reference_pair: tuple[OccurrenceId, OccurrenceId] | None,
) -> MoleculeFacts:
    site = facts.stereo.directional[0]
    return replace(
        facts,
        stereo=replace(
            facts.stereo,
            directional=(replace(site, reference_pair=reference_pair),),
        ),
    )


def _carrier_by_bond(slots) -> dict[BondId, CarrierSlot]:
    return {carrier.bond: carrier for carrier in slots.carrier_slots}


def _marks_for_normalized_values(
    slots,
    *,
    left: dict[BondId, int],
    right: dict[BondId, int],
) -> dict[CarrierSlotId, DirectionMark]:
    # In four_substituent_directional_facts, bonds 1/3 are references and 2/4
    # are the alternate ligands on the same endpoints.
    ligand_factor = {
        BondId(1): 1,
        BondId(2): -1,
        BondId(3): 1,
        BondId(4): -1,
    }
    raw_left = {
        bond_id: value * ligand_factor[bond_id]
        for bond_id, value in left.items()
    }
    raw_right = {
        bond_id: value * ligand_factor[bond_id]
        for bond_id, value in right.items()
    }
    return _marks_for_raw_values(slots, left=raw_left, right=raw_right)


def _marks_for_raw_values(
    slots,
    *,
    left: dict[BondId, int],
    right: dict[BondId, int],
) -> dict[CarrierSlotId, DirectionMark]:
    carrier_by_bond = _carrier_by_bond(slots)
    marks: dict[CarrierSlotId, DirectionMark] = {}
    for bond_id, raw in left.items():
        carrier = carrier_by_bond[bond_id]
        marks[carrier.id] = _mark_for_raw_value(carrier, AtomId(0), raw)
    for bond_id, raw in right.items():
        carrier = carrier_by_bond[bond_id]
        marks[carrier.id] = _mark_for_raw_value(carrier, AtomId(1), raw)
    return marks


def _mark_for_raw_value(
    carrier: CarrierSlot,
    endpoint: AtomId,
    raw: int,
) -> DirectionMark:
    if raw not in {-1, 1}:
        raise ValueError(f"raw direction value must be +/-1, got {raw!r}")
    if carrier.written_from == endpoint:
        orientation = 1
    elif carrier.written_to == endpoint:
        orientation = -1
    else:
        raise ValueError(
            f"carrier {carrier.id!r} is not incident to endpoint {endpoint!r}"
        )
    mark_sign = raw * orientation
    return DirectionMark.FWD if mark_sign == 1 else DirectionMark.REV

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
