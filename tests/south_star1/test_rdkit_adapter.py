"""Tests for the isolated South Star 1 RDKit ingestion boundary."""

from __future__ import annotations

from dataclasses import dataclass
import unittest

from rdkit import Chem

from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.errors import SouthStarErrorKind
from grimace._south_star1.facts import BondOrder
from grimace._south_star1.facts import DirectionalValue
from grimace._south_star1.facts import LigandKind
from grimace._south_star1.facts import SiteStatus
from grimace._south_star1.facts import TetraValue
from grimace._south_star1.fact_isomorphism import facts_are_isomorphic
from grimace._south_star1.graph_index import build_graph_index
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import OccurrenceId
from grimace._south_star1.ids import SiteId
from grimace._south_star1.ordinary_semantics import OrdinarySmilesSemantics
from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.policy import DirectionMark
from grimace._south_star1.policy import TetraToken
from grimace._south_star1.rdkit_adapter import RdkitOrdinaryExtractionOptions
from grimace._south_star1.rdkit_adapter import molecule_facts_from_rdkit
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_rdkit
from grimace._south_star1.render import render_stereo_traversal
from grimace._south_star1.skeleton import enumerate_traversal_skeletons
from grimace._south_star1.slots import allocate_traversal_slots
from grimace._south_star1.stereo_csp import enumerate_stereo_assignments_for_prefix
from grimace._south_star1.stereo_witness import enumerate_presentation_prefixes
from grimace._south_star1.support_enumeration import enumerate_stereo_support


class RdkitAdapterTest(unittest.TestCase):
    def test_snapshots_simple_nonstereo_molecule_facts(self) -> None:
        mol = Chem.MolFromSmiles("CCO")

        facts = molecule_facts_from_rdkit(mol)

        self.assertEqual(tuple(atom.symbol for atom in facts.atoms), ("C", "C", "O"))
        self.assertEqual(tuple(bond.order for bond in facts.bonds), (
            BondOrder.SINGLE,
            BondOrder.SINGLE,
        ))
        self.assertEqual(facts.components[0].atoms, (AtomId(0), AtomId(1), AtomId(2)))
        self.assertEqual(facts.components[0].bonds, (BondId(0), BondId(1)))

    def test_snapshots_disconnected_components_without_reordering_atoms(self) -> None:
        mol = Chem.MolFromSmiles("CO.CC")

        facts = molecule_facts_from_rdkit(mol)

        self.assertEqual(
            tuple(component.atoms for component in facts.components),
            ((AtomId(0), AtomId(1)), (AtomId(2), AtomId(3))),
        )

    def test_rejects_rdkit_atom_stereo_until_stereo_adapter_is_explicit(self) -> None:
        mol = Chem.MolFromSmiles("F[C@H](Cl)Br")

        with self.assertRaisesRegex(SouthStarError, "atom stereo") as raised:
            molecule_facts_from_rdkit(mol)
        self.assertIs(raised.exception.kind, SouthStarErrorKind.UNSUPPORTED_STEREO)

    def test_rejects_rdkit_bond_stereo_until_stereo_adapter_is_explicit(self) -> None:
        mol = Chem.MolFromSmiles("F/C=C/Cl")

        with self.assertRaisesRegex(SouthStarError, "bond stereo") as raised:
            molecule_facts_from_rdkit(mol)
        self.assertIs(raised.exception.kind, SouthStarErrorKind.UNSUPPORTED_STEREO)

    def test_ordinary_adapter_normalizes_non_graph_hydrogens(self) -> None:
        mol = Chem.MolFromSmiles("[C@H](F)(Cl)Br")

        facts = ordinary_molecule_facts_from_rdkit(
            mol,
            RdkitOrdinaryExtractionOptions(
                extract_specified_tetrahedral=False,
                reject_unsupported_stereo=False,
            ),
        )

        center = facts.atoms[0]
        self.assertEqual(center.explicit_h_count, 0)
        self.assertEqual(center.implicit_h_count, 1)
        self.assertFalse(center.no_implicit)
        self.assertEqual(len(facts.stereo.tetrahedral), 1)
        occurrence_by_id = {
            occurrence.id: occurrence
            for occurrence in facts.ligand_occurrences
        }
        tetra = facts.stereo.tetrahedral[0]
        self.assertEqual(
            sum(
                occurrence_by_id[occurrence_id].kind is LigandKind.IMPLICIT_H
                for occurrence_id in tetra.ligand_occurrences
            ),
            1,
        )
        ordinary_policy_for_facts(facts)

    def test_ordinary_adapter_promotes_rdkit_tetrahedral_stereo(self) -> None:
        mol = Chem.MolFromSmiles("[C@H](F)(Cl)Br")

        facts = ordinary_molecule_facts_from_rdkit(mol)

        self.assertEqual(len(facts.stereo.tetrahedral), 1)
        site = facts.stereo.tetrahedral[0]
        self.assertEqual(site.status, SiteStatus.SPECIFIED)
        self.assertEqual(site.target, TetraValue.PLUS)
        self.assertEqual(set(site.reference_order), set(site.ligand_occurrences))
        ordinary_policy_for_facts(facts)

    def test_ordinary_adapter_declares_smiles_parse_order_viewpoint_mode(self) -> None:
        options = RdkitOrdinaryExtractionOptions()

        self.assertEqual(options.tetra_viewpoint_mode, "smiles_parse_order")

    def test_ordinary_adapter_rejects_unknown_tetra_viewpoint_mode(self) -> None:
        mol = Chem.MolFromSmiles("[C@H](F)(Cl)Br")
        options = RdkitOrdinaryExtractionOptions(tetra_viewpoint_mode="renumbered")

        with self.assertRaisesRegex(SouthStarError, "tetra viewpoint mode") as raised:
            ordinary_molecule_facts_from_rdkit(mol, options)
        self.assertIs(raised.exception.kind, SouthStarErrorKind.UNSUPPORTED_POLICY)

    def test_ordinary_adapter_tetra_viewpoint_is_not_renumbering_invariant(
        self,
    ) -> None:
        mol = Chem.MolFromSmiles("C[C@H](F)Cl")
        renumbered = Chem.RenumberAtoms(mol, [1, 0, 2, 3])

        original = ordinary_molecule_facts_from_rdkit(mol)
        changed = ordinary_molecule_facts_from_rdkit(renumbered)

        self.assertFalse(facts_are_isomorphic(original, changed).isomorphic)

    def test_rdkit_tetra_adapter_tag_matches_semantics_for_root_and_nonroot(self) -> None:
        semantics = OrdinarySmilesSemantics()
        cases = {
            "[C@H](F)(Cl)Br": TetraToken.AT,
            "[C@@H](F)(Cl)Br": TetraToken.ATAT,
            "F[C@H](Cl)Br": TetraToken.AT,
            "F[C@@H](Cl)Br": TetraToken.ATAT,
            "C[C@H](F)Cl": TetraToken.AT,
            "C[C@@H](F)Cl": TetraToken.ATAT,
        }

        for text, token in cases.items():
            with self.subTest(text=text):
                facts = ordinary_molecule_facts_from_rdkit(Chem.MolFromSmiles(text))
                site = _only_tetra_site(facts)

                self.assertEqual(
                    semantics.tetra_value(
                        facts,
                        site.id,
                        site.reference_order,
                        token,
                    ),
                    site.target,
                )

    def test_ordinary_adapter_distinguishes_tetrahedral_enantiomer_tags(self) -> None:
        clockwise = ordinary_molecule_facts_from_rdkit(
            Chem.MolFromSmiles("[C@H](F)(Cl)Br")
        )
        counterclockwise = ordinary_molecule_facts_from_rdkit(
            Chem.MolFromSmiles("[C@@H](F)(Cl)Br")
        )

        self.assertEqual(
            clockwise.stereo.tetrahedral[0].ligand_occurrences,
            counterclockwise.stereo.tetrahedral[0].ligand_occurrences,
        )
        self.assertNotEqual(
            clockwise.stereo.tetrahedral[0].target,
            counterclockwise.stereo.tetrahedral[0].target,
        )

    def test_tetrahedral_all_root_support_roundtrips_to_isomorphic_facts(self) -> None:
        facts = ordinary_molecule_facts_from_rdkit(
            Chem.MolFromSmiles("[C@H](F)(Cl)Br")
        )
        policy = ordinary_policy_for_facts(facts)
        image = enumerate_stereo_support(
            facts=facts,
            policy=policy,
            semantics=OrdinarySmilesSemantics(),
        )

        self.assertGreater(image.distinct_count, 0)
        for text in image.strings:
            parsed = Chem.MolFromSmiles(text)
            self.assertIsNotNone(parsed, text)
            reparsed = ordinary_molecule_facts_from_rdkit(parsed)
            compare = facts_are_isomorphic(facts, reparsed)
            self.assertTrue(compare.isomorphic, (text, compare.reason))

    def test_tetrahedral_witness_trace_matches_reparsed_rdkit_target(self) -> None:
        facts = ordinary_molecule_facts_from_rdkit(
            Chem.MolFromSmiles("[C@H](F)(Cl)Br")
        )
        traces = _tetra_round_trip_traces(facts)

        self.assertTrue(traces)
        for trace in traces:
            with self.subTest(text=trace.text):
                self.assertEqual(
                    trace.expected_reparsed_target,
                    trace.actual_reparsed_target,
                    trace,
                )

    def test_ordinary_adapter_promotes_rdkit_directional_stereo(self) -> None:
        facts = ordinary_molecule_facts_from_rdkit(Chem.MolFromSmiles("F/C=C/Cl"))

        self.assertEqual(len(facts.stereo.directional), 1)
        site = facts.stereo.directional[0]
        self.assertEqual(site.status, SiteStatus.SPECIFIED)
        self.assertEqual(site.target, DirectionalValue.OPPOSITE)
        self.assertIsNotNone(site.reference_pair)
        self.assertNotEqual(site.reference_pair[0], site.reference_pair[1])
        ordinary_policy_for_facts(facts)

    def test_ordinary_adapter_distinguishes_directional_e_z_tags(self) -> None:
        opposite = ordinary_molecule_facts_from_rdkit(Chem.MolFromSmiles("F/C=C/Cl"))
        together = ordinary_molecule_facts_from_rdkit(Chem.MolFromSmiles("F/C=C\\Cl"))

        self.assertEqual(
            opposite.stereo.directional[0].reference_pair,
            together.stereo.directional[0].reference_pair,
        )
        self.assertEqual(
            opposite.stereo.directional[0].target,
            DirectionalValue.OPPOSITE,
        )
        self.assertEqual(
            together.stereo.directional[0].target,
            DirectionalValue.TOGETHER,
        )

    def test_rdkit_directional_adapter_contract_for_literal_slashes(self) -> None:
        semantics = OrdinarySmilesSemantics()
        cases = {
            "C(/F)=C(\\Cl)": DirectionalValue.OPPOSITE,
            "C(/F)=C(/Cl)": DirectionalValue.TOGETHER,
            "F/C=C/Cl": DirectionalValue.OPPOSITE,
            "F/C=C\\Cl": DirectionalValue.TOGETHER,
        }

        for text, expected in cases.items():
            with self.subTest(text=text):
                facts = ordinary_molecule_facts_from_rdkit(Chem.MolFromSmiles(text))
                site = _only_directional_site(facts)

                self.assertEqual(site.status, SiteStatus.SPECIFIED)
                self.assertEqual(site.target, expected)
                self.assertEqual(
                    _bond_by_id(facts)[site.center_bond].order,
                    BondOrder.DOUBLE,
                )
                self.assertIsNotNone(site.reference_pair)
                left_reference, right_reference = site.reference_pair
                self.assertIn(left_reference, site.left_ligands)
                self.assertIn(right_reference, site.right_ligands)
                self.assertEqual(
                    _literal_directional_value(facts, Chem.MolFromSmiles(text), site),
                    expected,
                )

                policy = ordinary_policy_for_facts(facts)
                for skeleton in enumerate_traversal_skeletons(
                    facts,
                    build_graph_index(facts),
                    policy,
                ):
                    slots = allocate_traversal_slots(facts, skeleton)
                    carrier_by_id = {
                        carrier.id: carrier
                        for carrier in slots.carrier_slots
                    }
                    scoped = semantics.directional_scope(
                        facts,
                        skeleton,
                        slots,
                        site.id,
                    )
                    self.assertTrue(scoped)
                    self.assertTrue(
                        all(
                            carrier_by_id[carrier].bond != site.center_bond
                            for carrier in scoped
                        )
                    )

    def test_ordinary_adapter_rejects_enhanced_stereo_groups(self) -> None:
        mol = Chem.MolFromSmiles("F[C@H](Cl)Br |&1:1|")

        with self.assertRaisesRegex(SouthStarError, "enhanced stereo") as raised:
            ordinary_molecule_facts_from_rdkit(mol)
        self.assertIs(raised.exception.kind, SouthStarErrorKind.UNSUPPORTED_STEREO)

    def test_ordinary_adapter_rejects_non_tetrahedral_atom_stereo(self) -> None:
        mol = Chem.MolFromSmiles("C(F)(Cl)(Br)I")
        mol.GetAtomWithIdx(0).SetChiralTag(Chem.ChiralType.CHI_SQUAREPLANAR)

        with self.assertRaisesRegex(SouthStarError, "unsupported RDKit atom") as raised:
            ordinary_molecule_facts_from_rdkit(mol)
        self.assertIs(raised.exception.kind, SouthStarErrorKind.UNSUPPORTED_STEREO)

    def test_ordinary_adapter_rejects_unknown_bond_stereo(self) -> None:
        mol = Chem.MolFromSmiles("F/C=C/Cl")
        mol.GetBondWithIdx(1).SetStereo(Chem.BondStereo.STEREOANY)

        with self.assertRaisesRegex(SouthStarError, "unsupported RDKit bond") as raised:
            ordinary_molecule_facts_from_rdkit(mol)
        self.assertIs(raised.exception.kind, SouthStarErrorKind.UNSUPPORTED_STEREO)

    def test_ordinary_adapter_rejects_atropisomeric_bond_stereo(self) -> None:
        mol = Chem.MolFromSmiles("F/C=C/Cl")
        mol.GetBondWithIdx(1).SetStereo(Chem.BondStereo.STEREOATROPCW)

        with self.assertRaisesRegex(SouthStarError, "unsupported RDKit bond") as raised:
            ordinary_molecule_facts_from_rdkit(mol)
        self.assertIs(raised.exception.kind, SouthStarErrorKind.UNSUPPORTED_STEREO)

@dataclass(frozen=True, slots=True)
class TetraRoundTripTrace:
    text: str
    original_center: AtomId
    original_site: SiteId
    rendered_token: TetraToken
    rendered_local_order: tuple[OccurrenceId, ...]
    original_reference_order: tuple[OccurrenceId, ...]
    original_target: TetraValue
    reparsed_reference_order: tuple[OccurrenceId, ...]
    reparsed_target: TetraValue
    mapped_rendered_order_in_reparsed_occurrences: tuple[OccurrenceId, ...]
    expected_reparsed_target: TetraValue
    actual_reparsed_target: TetraValue


def _tetra_round_trip_traces(facts) -> tuple[TetraRoundTripTrace, ...]:
    semantics = OrdinarySmilesSemantics()
    policy = ordinary_policy_for_facts(facts)
    site = _only_tetra_site(facts)
    traces: list[TetraRoundTripTrace] = []

    for skeleton in enumerate_traversal_skeletons(
        facts,
        build_graph_index(facts),
        policy,
    ):
        slots = allocate_traversal_slots(facts, skeleton)
        local_order = semantics.local_tetra_order(facts, skeleton, slots, site.id)
        for prefix in enumerate_presentation_prefixes(
            facts=facts,
            slots=slots,
            policy=policy,
        ):
            for assignment in enumerate_stereo_assignments_for_prefix(
                facts=facts,
                skeleton=skeleton,
                slots=slots,
                prefix=prefix,
                policy=policy,
                semantics=semantics,
            ):
                token = assignment.tetra_tokens[site.center]
                if token is TetraToken.NONE:
                    continue
                text = render_stereo_traversal(
                    facts=facts,
                    skeleton=skeleton,
                    slots=slots,
                    assignment=assignment,
                    policy=policy,
                    semantics=semantics,
                    validate=True,
                )
                reparsed = ordinary_molecule_facts_from_rdkit(Chem.MolFromSmiles(text))
                reparsed_site = _only_tetra_site(reparsed)
                mapped_order = _map_occurrences_by_signature(
                    facts,
                    reparsed,
                    local_order,
                )
                expected = semantics.tetra_value(
                    reparsed,
                    reparsed_site.id,
                    mapped_order,
                    token,
                )
                traces.append(
                    TetraRoundTripTrace(
                        text=text,
                        original_center=site.center,
                        original_site=site.id,
                        rendered_token=token,
                        rendered_local_order=local_order,
                        original_reference_order=site.reference_order,
                        original_target=site.target,
                        reparsed_reference_order=reparsed_site.reference_order,
                        reparsed_target=reparsed_site.target,
                        mapped_rendered_order_in_reparsed_occurrences=mapped_order,
                        expected_reparsed_target=expected,
                        actual_reparsed_target=reparsed_site.target,
                    )
                )

    return tuple(traces)


def _only_tetra_site(facts):
    self_declared = facts.stereo.tetrahedral
    if len(self_declared) != 1:
        raise AssertionError(f"expected one tetra site, got {self_declared!r}")
    return self_declared[0]


def _only_directional_site(facts):
    self_declared = facts.stereo.directional
    if len(self_declared) != 1:
        raise AssertionError(f"expected one directional site, got {self_declared!r}")
    return self_declared[0]


def _bond_by_id(facts):
    return {bond.id: bond for bond in facts.bonds}


def _literal_directional_value(facts, mol, site):
    semantics = OrdinarySmilesSemantics()
    policy = ordinary_policy_for_facts(facts)
    marked_bonds = _literal_direction_marks_by_bond(mol)

    for skeleton in enumerate_traversal_skeletons(
        facts,
        build_graph_index(facts),
        policy,
    ):
        slots = allocate_traversal_slots(facts, skeleton)
        carrier_by_bond = {
            carrier.bond: carrier
            for carrier in slots.carrier_slots
        }
        if not _carrier_orientations_match_rdkit(
            mol,
            carrier_by_bond,
            marked_bonds,
        ):
            continue

        marks = {
            carrier.id: DirectionMark.ABSENT
            for carrier in slots.carrier_slots
        }
        for bond, mark in marked_bonds.items():
            marks[carrier_by_bond[bond].id] = mark

        return semantics.directional_value(
            facts,
            skeleton,
            slots,
            site.id,
            marks,
        )

    raise AssertionError("no skeleton matched RDKit literal bond orientations")


def _literal_direction_marks_by_bond(mol) -> dict[BondId, DirectionMark]:
    out: dict[BondId, DirectionMark] = {}
    for bond in mol.GetBonds():
        direction = bond.GetBondDir()
        if direction == Chem.BondDir.ENDUPRIGHT:
            out[BondId(bond.GetIdx())] = DirectionMark.FWD
        elif direction == Chem.BondDir.ENDDOWNRIGHT:
            out[BondId(bond.GetIdx())] = DirectionMark.REV
    return out


def _carrier_orientations_match_rdkit(mol, carrier_by_bond, marked_bonds) -> bool:
    for bond_id in marked_bonds:
        carrier = carrier_by_bond.get(bond_id)
        if carrier is None:
            return False
        rdkit_bond = mol.GetBondWithIdx(int(bond_id))
        if carrier.written_from != AtomId(rdkit_bond.GetBeginAtomIdx()):
            return False
        if carrier.written_to != AtomId(rdkit_bond.GetEndAtomIdx()):
            return False
    return True


def _map_occurrences_by_signature(left, right, occurrence_order):
    left_by_id = {occurrence.id: occurrence for occurrence in left.ligand_occurrences}
    right_by_signature = {
        _occurrence_signature(right, occurrence): occurrence.id
        for occurrence in right.ligand_occurrences
    }
    return tuple(
        right_by_signature[_occurrence_signature(left, left_by_id[occurrence_id])]
        for occurrence_id in occurrence_order
    )


def _occurrence_signature(facts, occurrence):
    atom_by_id = {atom.id: atom for atom in facts.atoms}
    if occurrence.kind is LigandKind.IMPLICIT_H:
        atom = atom_by_id[occurrence.atom]
        return ("implicit_h", atom.atomic_num, atom.symbol)
    if occurrence.atom is None:
        raise AssertionError(f"neighbor occurrence lacks atom: {occurrence!r}")
    if occurrence.kind is LigandKind.NEIGHBOR_ATOM:
        atom = atom_by_id[occurrence.atom]
        return (
            "neighbor",
            atom.atomic_num,
            atom.symbol,
            atom.isotope,
            atom.formal_charge,
            atom.is_aromatic,
            atom.explicit_h_count,
            atom.implicit_h_count,
            atom.no_implicit,
        )
    raise AssertionError(f"unsupported occurrence kind: {occurrence.kind!r}")


if __name__ == "__main__":
    unittest.main()
