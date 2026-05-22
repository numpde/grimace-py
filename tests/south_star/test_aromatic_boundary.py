from __future__ import annotations

import unittest

from rdkit import Chem

from grimace._south_star.aromatic_policy import (
    DEFAULT_SOUTH_STAR_AROMATIC_POLICY_CONTRACT,
    SOUTH_STAR_AROMATIC_TEXT_POLICY_CONTRACT,
    SOUTH_STAR_AROMATIC_POLICY_FAMILY_CONTRACTS,
)
from grimace._south_star.enum_s import mol_to_smiles_enum_s_graph_native
from grimace._south_star.enum_s import mol_to_smiles_enum_s_tree_traversals_for_case
from grimace._south_star.enum_s import render_south_star_tree_traversal
from grimace._south_star.molecule_facts import SouthStarMoleculeFacts
from grimace._south_star.support_gates import south_star_support_gate_report
from tests.helpers.south_star_semantic_oracle import parse_smiles
from tests.helpers.south_star_exact_support import SouthStarExpandedSupportCase
from tests.helpers.south_star_exact_support import load_south_star_expanded_support_cases
from tests.helpers.south_star_grammar_conformance import south_star_grammar_conformance


class SouthStarAromaticBoundaryTests(unittest.TestCase):
    def test_active_contract_names_non_aromatic_fact_boundary(self) -> None:
        contract = DEFAULT_SOUTH_STAR_AROMATIC_POLICY_CONTRACT

        self.assertEqual("non_aromatic_molecule_facts", contract.name)
        self.assertEqual("active", contract.status)
        self.assertEqual("non_aromatic_molecule_facts", contract.molecule_fact_contract)
        self.assertEqual(
            "non_aromatic_organic_and_bracket_atom_text",
            contract.atom_text_policy,
        )
        self.assertEqual(
            "non_aromatic_single_double_bond_text",
            contract.bond_text_policy,
        )
        self.assertEqual(
            "non_aromatic_parse_back_graph_stereo_identity",
            contract.semantic_equivalence_relation,
        )
        self.assertEqual(
            "unsupported_aromatic_directional_overlay",
            contract.directional_surface_policy,
        )
        self.assertEqual(
            ("aromatic_ring_surface", "aromatic_directional_surface"),
            contract.support_gate_categories,
        )
        self.assertIn("semantic_parseback", contract.required_fixture_fields)
        self.assertIn(
            "non_aromatic_parseback_identity",
            contract.required_proof_obligations,
        )
        self.assertFalse(contract.supports_aromatic_facts)

    def test_policy_family_names_future_aromatic_boundaries(self) -> None:
        contracts_by_name = {
            contract.name: contract
            for contract in SOUTH_STAR_AROMATIC_POLICY_FAMILY_CONTRACTS
        }

        self.assertEqual(
            {
                "non_aromatic_molecule_facts",
                "non_aromatic_kekule_facts",
                "aromatic_text_policy",
            },
            set(contracts_by_name),
        )
        self.assertEqual(
            "active",
            contracts_by_name["non_aromatic_molecule_facts"].status,
        )
        self.assertEqual(
            "caller_prepared_non_aromatic_kekule_facts",
            contracts_by_name["non_aromatic_kekule_facts"].molecule_fact_contract,
        )
        self.assertFalse(
            contracts_by_name["non_aromatic_kekule_facts"].supports_aromatic_facts
        )
        self.assertIn(
            "preparation_contract",
            contracts_by_name["non_aromatic_kekule_facts"].required_fixture_fields,
        )
        self.assertIn(
            "caller_prepared_kekule_fact_boundary",
            contracts_by_name[
                "non_aromatic_kekule_facts"
            ].required_proof_obligations,
        )
        self.assertEqual(
            "sanitized_aromatic_molecule_facts",
            contracts_by_name["aromatic_text_policy"].molecule_fact_contract,
        )
        self.assertTrue(
            contracts_by_name["aromatic_text_policy"].supports_aromatic_facts
        )
        self.assertEqual("active", contracts_by_name["aromatic_text_policy"].status)
        self.assertEqual(
            ("aromatic_ring_surface", "aromatic_directional_surface"),
            contracts_by_name["aromatic_text_policy"].support_gate_categories,
        )
        self.assertIn(
            "aromatic_fact_signature",
            contracts_by_name["aromatic_text_policy"].required_fixture_fields,
        )
        self.assertIn(
            "lowercase_aromatic_atom_text_obligations",
            contracts_by_name["aromatic_text_policy"].required_proof_obligations,
        )
        self.assertIn(
            "aromatic_or_kekule_parseback_equivalence",
            contracts_by_name["aromatic_text_policy"].required_proof_obligations,
        )

    def test_policy_family_contracts_are_implementation_ready(self) -> None:
        for contract in SOUTH_STAR_AROMATIC_POLICY_FAMILY_CONTRACTS:
            with self.subTest(contract=contract.name):
                self.assertIn(contract.status, {"active", "candidate"})
                self.assertTrue(contract.molecule_fact_contract)
                self.assertTrue(contract.atom_text_policy)
                self.assertTrue(contract.bond_text_policy)
                self.assertTrue(contract.semantic_equivalence_relation)
                self.assertTrue(contract.directional_surface_policy)
                self.assertTrue(contract.required_fixture_fields)
                self.assertTrue(contract.required_proof_obligations)
                self.assertEqual(
                    len(set(contract.required_fixture_fields)),
                    len(contract.required_fixture_fields),
                )
                self.assertEqual(
                    len(set(contract.required_proof_obligations)),
                    len(contract.required_proof_obligations),
                )

    def test_markerless_aromatic_text_cases_are_supported(
        self,
    ) -> None:
        case_ids = (
            "aromatic_text_monocycle_benzene",
            "aromatic_text_monocycle_pyridine",
            "aromatic_text_monocycle_furan",
            "aromatic_text_branch_toluene",
            "aromatic_text_branch_methyl_pyridine",
            "aromatic_text_branch_methyl_furan",
        )

        for case_id in case_ids:
            case = _expanded_support_case(case_id)
            result = mol_to_smiles_enum_s_graph_native(
                case.source_smiles,
                case_id=case.case_id,
            )
            report = south_star_support_gate_report(parse_smiles(case.source_smiles))

            with self.subTest(case_id=case_id):
                self.assertTrue(report.supported, report.unsupported_features)
                self.assertEqual(case.expected_support, result.outputs)
            for output in result.outputs:
                with self.subTest(case_id=case_id, output=output):
                    self.assertTrue(south_star_grammar_conformance(output).passed)

    def test_modified_aromatic_atom_text_cases_are_supported(
        self,
    ) -> None:
        case_ids = (
            "modified_aromatic_atom_text_pyrrole_nh",
            "modified_aromatic_atom_text_isotope_pyrrole",
            "modified_aromatic_atom_text_mapped_pyrrole",
            "modified_aromatic_atom_text_mapped_pyridine",
            "modified_aromatic_atom_text_pyridinium_h",
            "modified_aromatic_atom_text_pyridine_n_oxide",
        )

        for case_id in case_ids:
            case = _expanded_support_case(case_id)
            result = mol_to_smiles_enum_s_graph_native(
                case.source_smiles,
                case_id=case.case_id,
            )
            report = south_star_support_gate_report(parse_smiles(case.source_smiles))

            with self.subTest(case_id=case_id):
                self.assertTrue(report.supported, report.unsupported_features)
                self.assertEqual(case.expected_support, result.outputs)
            for output in result.outputs:
                with self.subTest(case_id=case_id, output=output):
                    self.assertTrue(south_star_grammar_conformance(output).passed)

    def test_bracket_only_aromatic_element_text_cases_are_supported(
        self,
    ) -> None:
        case_ids = (
            "aromatic_selenium_text_selenophene",
            "aromatic_selenium_text_mapped_selenophene",
            "aromatic_tellurium_text_tellurophene",
        )

        for case_id in case_ids:
            case = _expanded_support_case(case_id)
            result = mol_to_smiles_enum_s_graph_native(
                case.source_smiles,
                case_id=case.case_id,
            )
            report = south_star_support_gate_report(parse_smiles(case.source_smiles))

            with self.subTest(case_id=case_id):
                self.assertTrue(report.supported, report.unsupported_features)
                self.assertEqual(case.expected_support, result.outputs)
            for output in result.outputs:
                with self.subTest(case_id=case_id, output=output):
                    self.assertTrue(south_star_grammar_conformance(output).passed)

    def test_aromatic_monocycle_fixture_uses_aromatic_renderer_obligations(
        self,
    ) -> None:
        case = _expanded_support_case("aromatic_text_monocycle_benzene")
        traversals = mol_to_smiles_enum_s_tree_traversals_for_case(case)
        rendered = tuple(
            dict.fromkeys(render_south_star_tree_traversal(t) for t in traversals)
        )

        self.assertEqual(case.expected_support, rendered)
        self.assertTrue(
            all(
                event.kind != "bond" or event.text == ""
                for traversal in traversals
                for event in traversal.events
            )
        )
        atom_texts = tuple(
            event.text
            for traversal in traversals
            for event in traversal.events
            if event.kind == "atom"
        )

        self.assertEqual({"c"}, set(atom_texts))

    def test_hetero_aromatic_monocycle_fixtures_use_lowercase_atom_text(
        self,
    ) -> None:
        expected_atom_texts = {
            "aromatic_text_monocycle_pyridine": {"c", "n"},
            "aromatic_text_monocycle_furan": {"c", "o"},
            "aromatic_selenium_text_selenophene": {"c", "[se]"},
            "aromatic_selenium_text_mapped_selenophene": {"c", "[se:7]"},
            "aromatic_tellurium_text_tellurophene": {"c", "[te]"},
            "aromatic_tellurium_text_mapped_tellurophene": {"c", "[te:7]"},
        }

        for case_id, expected_texts in expected_atom_texts.items():
            case = _expanded_support_case(case_id)
            traversals = mol_to_smiles_enum_s_tree_traversals_for_case(case)
            rendered = tuple(
                dict.fromkeys(render_south_star_tree_traversal(t) for t in traversals)
            )
            atom_texts = {
                event.text
                for traversal in traversals
                for event in traversal.events
                if event.kind == "atom"
            }

            with self.subTest(case_id=case_id):
                self.assertEqual(case.expected_support, rendered)
                self.assertEqual(expected_texts, atom_texts)
                self.assertTrue(
                    all(
                        event.kind != "bond" or event.text == ""
                        for traversal in traversals
                        for event in traversal.events
                    )
                )

    def test_sanitized_aromatic_spellings_share_aromatic_facts(self) -> None:
        cases = (
            "c1ccccc1",
            "C1=CC=CC=C1",
            "c1ccncc1",
            "c1ccoc1",
            "c1ccccc1C",
            "c1ccncc1C",
            "c1ccoc1C",
        )

        for smiles in cases:
            facts = SouthStarMoleculeFacts.from_mol(parse_smiles(smiles))

            with self.subTest(smiles=smiles):
                self.assertTrue(facts.supported, facts.unsupported_categories)
                self.assertTrue(any(atom.is_aromatic for atom in facts.atom_text_facts))
                self.assertTrue(any(bond.is_aromatic for bond in facts.bond_text_facts))
                self.assertEqual(
                    {"AROMATIC"},
                    {
                        bond.bond_type
                        for bond in facts.bond_text_facts
                        if bond.is_in_ring
                    },
                )
                self.assertTrue(
                    all(
                        bond.bond_type == "SINGLE"
                        for bond in facts.bond_text_facts
                        if not bond.is_in_ring
                    )
                )

    def test_deliberately_kekulized_non_aromatic_facts_are_a_different_contract(
        self,
    ) -> None:
        mol = parse_smiles("c1ccccc1")
        Chem.Kekulize(mol, clearAromaticFlags=True)

        facts = SouthStarMoleculeFacts.from_mol(mol)

        self.assertNotIn("aromatic_ring_surface", facts.unsupported_categories)
        self.assertNotIn("aromatic_directional_surface", facts.unsupported_categories)
        self.assertTrue(facts.supported, facts.unsupported_categories)
        self.assertFalse(any(atom.is_aromatic for atom in facts.atom_text_facts))
        self.assertFalse(any(bond.is_aromatic for bond in facts.bond_text_facts))
        self.assertEqual(
            {"SINGLE", "DOUBLE"},
            {bond.bond_type for bond in facts.bond_text_facts},
        )

    def test_aromatic_directional_surface_remains_a_fail_fast_overlay(self) -> None:
        mol = parse_smiles("c1ccccc1")
        mol.GetBondWithIdx(0).SetBondDir(Chem.BondDir.ENDUPRIGHT)
        facts = SouthStarMoleculeFacts.from_mol(mol)

        self.assertIn("aromatic_directional_surface", facts.unsupported_categories)
        self.assertNotIn("aromatic_ring_surface", facts.unsupported_categories)

    def test_support_gate_reasons_name_active_contract(self) -> None:
        contract = SOUTH_STAR_AROMATIC_TEXT_POLICY_CONTRACT
        mol = parse_smiles("[15te]1cccc1")

        report = south_star_support_gate_report(mol)
        reasons_by_category = {
            feature.category: feature.reason for feature in report.unsupported_features
        }

        self.assertIn(contract.name, reasons_by_category["aromatic_ring_surface"])

    def test_support_gate_directional_reason_names_active_contract(self) -> None:
        contract = SOUTH_STAR_AROMATIC_TEXT_POLICY_CONTRACT
        mol = parse_smiles("c1ccccc1")
        mol.GetBondWithIdx(0).SetBondDir(Chem.BondDir.ENDUPRIGHT)

        report = south_star_support_gate_report(mol)
        reasons_by_category = {
            feature.category: feature.reason for feature in report.unsupported_features
        }

        self.assertIn(
            contract.directional_surface_policy,
            reasons_by_category["aromatic_directional_surface"],
        )


def _expanded_support_case(case_id: str) -> SouthStarExpandedSupportCase:
    return next(
        case
        for case in load_south_star_expanded_support_cases()
        if case.case_id == case_id
    )


if __name__ == "__main__":
    unittest.main()
