from __future__ import annotations

import unittest

from grimace._south_star.component_support_state import (
    SouthStarComponentSupportState,
)
from grimace._south_star.components import extract_south_star_components
from grimace._south_star.molecule_facts import SouthStarMoleculeFacts
from grimace._south_star.support_gates import south_star_support_gate_report
from tests.helpers.south_star_semantic_facts import (
    south_star_semantic_facts_from_case,
)
from tests.helpers.south_star_semantic_oracle import parse_smiles
from tests.helpers.south_star_semantics import load_south_star_semantic_cases


class SouthStarMoleculeFactsTests(unittest.TestCase):
    def test_molecule_facts_share_support_gate_report_with_component_extraction(
        self,
    ) -> None:
        mol = parse_smiles("F/C=C\\Cl")
        direct_report = south_star_support_gate_report(mol)
        facts = SouthStarMoleculeFacts.from_mol(mol)
        extraction = extract_south_star_components(
            mol,
            support_gate_report=facts.support_gate_report,
        )

        self.assertEqual(direct_report.categories, facts.unsupported_categories)
        self.assertIs(facts.support_gate_report, extraction.support_gate_report)
        self.assertEqual(extraction.components, facts.components)

    def test_component_support_state_consumes_molecule_facts_as_ssot(self) -> None:
        facts = SouthStarMoleculeFacts.from_mol(parse_smiles("F/C=C\\Cl"))
        state = SouthStarComponentSupportState.from_molecule_facts(facts)

        self.assertIs(facts, state.molecule_facts)
        self.assertEqual(facts.components, state.components)

    def test_semantic_helper_wraps_runtime_molecule_facts(self) -> None:
        case = next(
            case
            for case in load_south_star_semantic_cases()
            if case.case_id == "isolated_alkene_z"
        )
        semantic_facts = south_star_semantic_facts_from_case(case)

        self.assertIsInstance(semantic_facts.molecule_facts, SouthStarMoleculeFacts)
        self.assertEqual(
            case.eligible_carrier_edges,
            tuple(
                opportunity.edge
                for opportunity in semantic_facts.carrier_opportunities
            ),
        )

    def test_molecule_facts_expose_topology_atom_and_bond_text_inputs(self) -> None:
        facts = SouthStarMoleculeFacts.from_mol(parse_smiles("C1=CCCCC1"))

        self.assertEqual(tuple(range(6)), facts.graph_topology.atom_indices)
        self.assertEqual(1, facts.graph_topology.ring_count)
        self.assertEqual(
            ("C", "C", "C", "C", "C", "C"),
            tuple(atom.symbol for atom in facts.atom_text_facts),
        )
        self.assertIn("DOUBLE", {bond.bond_type for bond in facts.bond_text_facts})

    def test_unsupported_categories_are_available_before_enumeration(self) -> None:
        facts = SouthStarMoleculeFacts.from_mol(parse_smiles("C#N"))

        self.assertIn("unsupported_bond_type", facts.unsupported_categories)
        with self.assertRaisesRegex(NotImplementedError, "unsupported_bond_type"):
            facts.fail_if_unsupported()


if __name__ == "__main__":
    unittest.main()
