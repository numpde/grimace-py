from __future__ import annotations

import unittest

from grimace._south_star.enum_s import mol_to_smiles_enum_s_graph_native
from grimace._south_star.molecule_facts import SouthStarMoleculeFacts
from tests.helpers.south_star_domain_manifest import (
    SOUTH_STAR_MARKERLESS_ACYCLIC_TREE_UNIFIED_REFERENCE_AUTHORITY,
    SOUTH_STAR_NONSTEREO_MONOCYCLE_UNIFIED_REFERENCE_AUTHORITY,
    SOUTH_STAR_SINGLE_ATOM_ATOM_TEXT_UNIFIED_REFERENCE_AUTHORITY,
    SOUTH_STAR_TWO_ATOM_MARKERLESS_ATOM_TEXT_UNIFIED_REFERENCE_AUTHORITY,
)
from tests.helpers.south_star_exact_support import load_south_star_expanded_support_cases
from tests.helpers.south_star_semantic_oracle import parse_smiles
from tests.helpers.south_star_semantics import load_south_star_semantic_cases
from tests.helpers.south_star_unified_reference import (
    is_markerless_acyclic_tree_domain,
    is_nonstereo_monocycle_ring_traversal_domain,
    is_single_atom_atom_text_domain,
    is_two_atom_markerless_atom_text_domain,
    markerless_acyclic_tree_support_from_shared_spine,
    nonstereo_monocycle_support_from_shared_spine,
    single_atom_atom_text_support_from_facts,
    two_atom_markerless_atom_text_support_from_facts,
    two_atom_markerless_atom_text_support_proof_from_facts,
)


SINGLE_ATOM_ATOM_TEXT_CASE_IDS = frozenset(
    {
        "radical_atom_text_hydrogen",
        "radical_atom_text_methyl",
        "radical_atom_text_oxygen",
        "charged_atom_text_chloride",
        "charged_atom_text_ammonium",
        "isotope_atom_text_methane",
    }
)

MARKERLESS_ACYCLIC_TREE_CASE_IDS = frozenset(
    {
        "markerless_acyclic_ethanol",
        "markerless_acyclic_isopropanol",
        "markerless_acyclic_acetone",
        "markerless_acyclic_acetonitrile",
    }
)

NONSTEREO_MONOCYCLE_CASE_IDS = frozenset(
    {
        "simple_saturated_monocycle_cyclohexane",
        "branched_saturated_monocycle_methylcyclohexane",
        "unsaturated_nonstereo_monocycle_cyclohexene",
        "branched_unsaturated_nonstereo_monocycle_methylcyclohexene",
        "unsaturated_nonstereo_monocycle_cyclohexadiene",
    }
)


class SouthStarUnifiedReferencePromotionTests(unittest.TestCase):
    def test_single_atom_atom_text_cases_have_facts_derived_singleton_support(
        self,
    ) -> None:
        cases = {
            case.case_id: case
            for case in load_south_star_expanded_support_cases()
            if case.case_id in SINGLE_ATOM_ATOM_TEXT_CASE_IDS
        }
        self.assertEqual(SINGLE_ATOM_ATOM_TEXT_CASE_IDS, frozenset(cases))

        for case in cases.values():
            with self.subTest(case_id=case.case_id):
                facts = SouthStarMoleculeFacts.from_mol(parse_smiles(case.source_smiles))
                self.assertTrue(is_single_atom_atom_text_domain(facts))

                support = single_atom_atom_text_support_from_facts(facts)
                self.assertEqual(case.expected_support, support.support)
                self.assertEqual(1, len(support.atom_text_obligations))
                self.assertEqual(
                    support.emitted_text,
                    support.atom_text_obligations[0].emitted_text,
                )
                self.assertEqual(
                    support.bracket_obligation_count,
                    len(support.atom_text_obligations[0].bracket_obligations),
                )
                if case.feature_area in {
                    "charged_atom_text",
                    "isotope_atom_text",
                    "radical_atom_text",
                }:
                    self.assertGreaterEqual(support.modifier_obligation_count, 1)

                result = mol_to_smiles_enum_s_graph_native(
                    case.source_smiles,
                    case_id=case.case_id,
                )
                self.assertEqual(support.support, result.outputs)
                diagnostics = result.generation_diagnostics
                self.assertIsNotNone(diagnostics)
                if diagnostics is None:
                    continue
                self.assertEqual(1, diagnostics.fragment_count)
                self.assertEqual(1, diagnostics.traversal_skeleton_count)
                self.assertEqual(0, diagnostics.marker_slot_count)
                self.assertEqual(1, diagnostics.local_assignment_count)
                self.assertEqual(1, diagnostics.solved_assignment_count)
                self.assertEqual(1, diagnostics.raw_output_count)
                self.assertEqual(1, diagnostics.output_count)
                self.assertEqual(
                    SOUTH_STAR_SINGLE_ATOM_ATOM_TEXT_UNIFIED_REFERENCE_AUTHORITY,
                    case.support_authority,
                )

    def test_two_atom_markerless_atom_text_cases_have_facts_derived_support(
        self,
    ) -> None:
        cases = {
            case.case_id: case for case in load_south_star_expanded_support_cases()
        }
        for case_id in (
            "explicit_bracket_hydrogen_h2",
            "charged_atom_text_methylammonium",
            "atom_map_text_ethane",
            "triple_bond_text_hydrogen_cyanide",
            "double_bond_text_formaldimine",
            "combined_atom_text_isotope_map_ethane",
            "combined_atom_text_isotope_charge_methylammonium",
        ):
            case = cases[case_id]
            with self.subTest(case_id=case_id):
                facts = SouthStarMoleculeFacts.from_mol(parse_smiles(case.source_smiles))
                self.assertTrue(is_two_atom_markerless_atom_text_domain(facts))
                support = two_atom_markerless_atom_text_support_proof_from_facts(facts)
                self.assertEqual(case.expected_support, support.support)
                self.assertEqual(2, len(support.atom_text_obligations))
                expected_bond_token_family = (
                    "explicit_triple_bond"
                    if case_id == "triple_bond_text_hydrogen_cyanide"
                    else "explicit_double_bond"
                    if case_id == "double_bond_text_formaldimine"
                    else "elided_single_bond"
                )
                self.assertEqual(
                    expected_bond_token_family,
                    support.bond_text_obligation.token_family,
                )
                if case_id not in {
                    "triple_bond_text_hydrogen_cyanide",
                    "double_bond_text_formaldimine",
                }:
                    self.assertGreaterEqual(support.bracket_obligation_count, 1)
                self.assertEqual(
                    support.support,
                    two_atom_markerless_atom_text_support_from_facts(facts),
                )
                if case_id in {
                    "atom_map_text_ethane",
                    "charged_atom_text_methylammonium",
                    "combined_atom_text_isotope_map_ethane",
                    "combined_atom_text_isotope_charge_methylammonium",
                }:
                    self.assertGreaterEqual(support.modifier_obligation_count, 1)

                result = mol_to_smiles_enum_s_graph_native(
                    case.source_smiles,
                    case_id=case.case_id,
                )
                self.assertEqual(support.support, result.outputs)
                self.assertEqual(
                    SOUTH_STAR_TWO_ATOM_MARKERLESS_ATOM_TEXT_UNIFIED_REFERENCE_AUTHORITY,
                    case.support_authority,
                )

    def test_markerless_acyclic_tree_cases_have_shared_spine_support(
        self,
    ) -> None:
        cases = {
            case.case_id: case
            for case in load_south_star_expanded_support_cases()
            if case.case_id in MARKERLESS_ACYCLIC_TREE_CASE_IDS
        }
        self.assertEqual(MARKERLESS_ACYCLIC_TREE_CASE_IDS, frozenset(cases))

        for case in cases.values():
            with self.subTest(case_id=case.case_id):
                facts = SouthStarMoleculeFacts.from_mol(parse_smiles(case.source_smiles))
                self.assertTrue(is_markerless_acyclic_tree_domain(facts))
                proof = markerless_acyclic_tree_support_from_shared_spine(case)

                self.assertFalse(proof.expected_support_strings_used)
                self.assertEqual(case.expected_support, proof.support)
                self.assertEqual(proof.output_count, len(case.expected_support))
                self.assertGreaterEqual(proof.raw_output_count, proof.output_count)
                self.assertEqual(facts.graph_topology.atom_count, proof.atom_count)
                self.assertEqual(facts.graph_topology.bond_count, proof.bond_count)
                self.assertGreater(proof.traversal_count, 0)
                self.assertEqual(proof.atom_count, proof.atom_text_obligation_count)
                self.assertEqual(proof.bond_count, proof.bond_text_obligation_count)
                self.assertEqual(
                    proof.traversal_count * proof.atom_count,
                    proof.atom_event_count,
                )
                self.assertGreater(proof.bond_event_count, 0)
                self.assertTrue(proof.bond_token_families)
                if proof.atom_count > 3:
                    self.assertGreater(proof.branch_event_count, 0)

                result = mol_to_smiles_enum_s_graph_native(
                    case.source_smiles,
                    case_id=case.case_id,
                )
                self.assertEqual(proof.support, result.outputs)
                self.assertEqual(
                    SOUTH_STAR_MARKERLESS_ACYCLIC_TREE_UNIFIED_REFERENCE_AUTHORITY,
                    case.support_authority,
                )

    def test_nonstereo_monocycle_cases_have_shared_spine_support(self) -> None:
        cases = {
            case.case_id: case
            for case in load_south_star_expanded_support_cases()
            if case.case_id in NONSTEREO_MONOCYCLE_CASE_IDS
        }
        self.assertEqual(NONSTEREO_MONOCYCLE_CASE_IDS, frozenset(cases))

        for case in cases.values():
            with self.subTest(case_id=case.case_id):
                facts = SouthStarMoleculeFacts.from_mol(parse_smiles(case.source_smiles))
                self.assertTrue(is_nonstereo_monocycle_ring_traversal_domain(facts))
                proof = nonstereo_monocycle_support_from_shared_spine(case)

                self.assertFalse(proof.expected_support_strings_used)
                self.assertEqual(case.expected_support, proof.support)
                self.assertEqual(1, proof.ring_count)
                self.assertEqual(proof.output_count, len(case.expected_support))
                self.assertGreaterEqual(proof.raw_output_count, proof.output_count)
                self.assertEqual(proof.atom_count, proof.atom_text_obligation_count)
                self.assertEqual(proof.bond_count, proof.bond_text_obligation_count)
                self.assertEqual(
                    proof.traversal_count * proof.atom_count,
                    proof.atom_event_count,
                )
                self.assertTrue(proof.bond_token_families)
                self.assertEqual(2 * proof.traversal_count, proof.closure_event_count)
                self.assertGreater(proof.closure_event_count, 0)
                self.assertEqual(0, proof.marker_slot_count)
                self.assertEqual(0, proof.renderer_input_count)
                self.assertEqual(
                    case.feature_area == "unsaturated_nonstereo_monocycle",
                    any(text == "=" for text in proof.closure_open_bond_texts),
                )

                result = mol_to_smiles_enum_s_graph_native(
                    case.source_smiles,
                    case_id=case.case_id,
                )
                self.assertEqual(proof.support, result.outputs)
                self.assertEqual(
                    SOUTH_STAR_NONSTEREO_MONOCYCLE_UNIFIED_REFERENCE_AUTHORITY,
                    case.support_authority,
                )

    def test_single_atom_atom_text_domain_rejects_wider_atom_text_cases(self) -> None:
        cases = {
            case.case_id: case for case in load_south_star_expanded_support_cases()
        }
        for case_id in (
            "explicit_bracket_hydrogen_h2",
            "charged_atom_text_methylammonium",
        ):
            case = cases[case_id]
            with self.subTest(case_id=case_id):
                facts = SouthStarMoleculeFacts.from_mol(parse_smiles(case.source_smiles))
                self.assertFalse(is_single_atom_atom_text_domain(facts))
                with self.assertRaisesRegex(NotImplementedError, "single-atom"):
                    single_atom_atom_text_support_from_facts(facts)
                self.assertNotEqual(
                    SOUTH_STAR_SINGLE_ATOM_ATOM_TEXT_UNIFIED_REFERENCE_AUTHORITY,
                    case.support_authority,
                )

    def test_two_atom_markerless_atom_text_domain_rejects_stereo_case(self) -> None:
        case = next(
            case
            for case in load_south_star_expanded_support_cases()
            if case.case_id == "implicit_h_tetrahedral_center"
        )
        facts = SouthStarMoleculeFacts.from_mol(parse_smiles(case.source_smiles))

        self.assertFalse(is_two_atom_markerless_atom_text_domain(facts))
        with self.assertRaisesRegex(NotImplementedError, "two-atom"):
            two_atom_markerless_atom_text_support_from_facts(facts)

    def test_markerless_acyclic_tree_domain_rejects_stereo_and_ring_cases(
        self,
    ) -> None:
        cases_by_id = {
            case.case_id: case for case in load_south_star_expanded_support_cases()
        }
        cases_by_id.update(
            {case.case_id: case for case in load_south_star_semantic_cases()}
        )
        for case_id in (
            "isolated_alkene_z",
            "simple_saturated_monocycle_cyclohexane",
        ):
            case = cases_by_id[case_id]
            with self.subTest(case_id=case_id):
                facts = SouthStarMoleculeFacts.from_mol(parse_smiles(case.source_smiles))
                self.assertFalse(is_markerless_acyclic_tree_domain(facts))
                with self.assertRaisesRegex(NotImplementedError, "acyclic-tree"):
                    markerless_acyclic_tree_support_from_shared_spine(case)

    def test_nonstereo_monocycle_domain_rejects_stereo_ring_case(self) -> None:
        cases_by_id = {
            case.case_id: case for case in load_south_star_expanded_support_cases()
        }
        case = cases_by_id["ring_stereo_monocycle_cyclooctene"]

        facts = SouthStarMoleculeFacts.from_mol(parse_smiles(case.source_smiles))
        self.assertFalse(is_nonstereo_monocycle_ring_traversal_domain(facts))
        with self.assertRaisesRegex(NotImplementedError, "nonstereo-monocycle"):
            nonstereo_monocycle_support_from_shared_spine(case)


if __name__ == "__main__":
    unittest.main()
