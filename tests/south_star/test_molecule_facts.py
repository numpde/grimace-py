from __future__ import annotations

import unittest

from rdkit import Chem

from grimace._south_star.component_support_state import (
    SouthStarComponentSupportState,
)
from grimace._south_star.components import extract_south_star_components
from grimace._south_star.enum_s import (
    _mol_to_smiles_enum_s_graph_native_for_mol,
    mol_to_smiles_enum_s_graph_native,
)
from grimace._south_star.molecule_facts import SouthStarMoleculeFacts
from grimace._south_star.policies import DEFAULT_SOUTH_STAR_POLICY_SET
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
        self.assertEqual(6, facts.graph_topology.atom_count)
        self.assertEqual(6, facts.graph_topology.bond_count)
        self.assertEqual(1, facts.graph_topology.fragment_count)
        self.assertTrue(facts.graph_topology.connected)
        self.assertFalse(facts.graph_topology.acyclic_connected_tree)
        self.assertEqual(1, facts.graph_topology.ring_count)
        self.assertEqual(1, facts.graph_topology.cyclomatic_number)
        self.assertTrue(facts.graph_topology.ring_system.has_rings)
        self.assertTrue(facts.graph_topology.ring_system.simple_monocycle)
        self.assertFalse(facts.graph_topology.ring_system.fused_or_polycyclic)
        self.assertEqual(
            (),
            facts.graph_topology.ring_system.shared_ring_atom_indices,
        )
        self.assertEqual(
            (),
            facts.graph_topology.ring_system.shared_ring_bond_indices,
        )
        self.assertEqual(
            ("C", "C", "C", "C", "C", "C"),
            tuple(atom.symbol for atom in facts.atom_text_facts),
        )
        self.assertIn("DOUBLE", {bond.bond_type for bond in facts.bond_text_facts})

    def test_atom_text_facts_expose_bracket_modifier_inputs(self) -> None:
        facts = SouthStarMoleculeFacts.from_mol(parse_smiles("[CH3:7]C"))
        mapped_atom = facts.atom_text_facts[0]

        self.assertEqual("C", mapped_atom.symbol)
        self.assertEqual(7, mapped_atom.atom_map_number)
        self.assertEqual(3, mapped_atom.explicit_hydrogen_count)
        self.assertEqual("CHI_UNSPECIFIED", mapped_atom.chiral_tag)
        self.assertNotIn("unsupported_atom_map", facts.unsupported_categories)

    def test_atom_text_facts_expose_renderer_capable_modifier_inputs(self) -> None:
        cases = (
            ("[2H][H]", "isotope", 2),
            ("[H+]", "formal_charge", 1),
            ("[Cl-]", "formal_charge", -1),
            ("[CH3:7]C", "atom_map_number", 7),
            ("[SiH3]C", "explicit_hydrogen_count", 3),
            ("[SeH]", "radical_electron_count", 1),
        )

        for smiles, field_name, expected_value in cases:
            facts = SouthStarMoleculeFacts.from_mol(parse_smiles(smiles))
            atom_fact = facts.atom_text_facts[0]

            with self.subTest(smiles=smiles):
                self.assertEqual(expected_value, getattr(atom_fact, field_name))
                self.assertTrue(facts.supported, facts.unsupported_categories)

    def test_atom_text_facts_expose_radical_modifier_inputs(self) -> None:
        facts = SouthStarMoleculeFacts.from_mol(parse_smiles("[CH3]"))
        atom_fact = facts.atom_text_facts[0]

        self.assertEqual(1, atom_fact.radical_electron_count)
        self.assertTrue(facts.supported, facts.unsupported_categories)

    def test_ring_system_facts_expose_polycyclic_witness_shape(self) -> None:
        facts = SouthStarMoleculeFacts.from_mol(parse_smiles("C1CC2CCCC2C1"))

        self.assertEqual(2, facts.graph_topology.cyclomatic_number)
        self.assertEqual(2, facts.graph_topology.ring_system.ring_count)
        self.assertEqual(2, len(facts.graph_topology.ring_system.atom_rings))
        self.assertEqual(2, len(facts.graph_topology.ring_system.bond_rings))
        self.assertFalse(facts.graph_topology.ring_system.simple_monocycle)
        self.assertTrue(facts.graph_topology.ring_system.fused_or_polycyclic)
        self.assertEqual(
            (2, 6),
            facts.graph_topology.ring_system.shared_ring_atom_indices,
        )
        self.assertEqual(
            (8,),
            facts.graph_topology.ring_system.shared_ring_bond_indices,
        )
        self.assertFalse(facts.graph_topology.ring_system.spiro_like)

    def test_ring_system_facts_expose_spiro_guardrail_shape(self) -> None:
        facts = SouthStarMoleculeFacts.from_mol(parse_smiles("C1CCC2(CC1)CCCC2"))

        self.assertEqual(2, facts.graph_topology.cyclomatic_number)
        self.assertEqual(2, facts.graph_topology.ring_system.ring_count)
        self.assertEqual(
            (3,),
            facts.graph_topology.ring_system.shared_ring_atom_indices,
        )
        self.assertEqual(
            (),
            facts.graph_topology.ring_system.shared_ring_bond_indices,
        )
        self.assertTrue(facts.graph_topology.ring_system.spiro_like)

    def test_ring_system_facts_expose_bridged_guardrail_shape(self) -> None:
        facts = SouthStarMoleculeFacts.from_mol(parse_smiles("C1CC2CCC1C2"))

        self.assertEqual(2, facts.graph_topology.cyclomatic_number)
        self.assertEqual(2, facts.graph_topology.ring_system.ring_count)
        self.assertEqual(
            (2, 5, 6),
            facts.graph_topology.ring_system.shared_ring_atom_indices,
        )
        self.assertEqual(
            (5, 7),
            facts.graph_topology.ring_system.shared_ring_bond_indices,
        )
        self.assertFalse(facts.graph_topology.ring_system.spiro_like)

    def test_molecule_facts_topology_distinguishes_tree_and_fragments(self) -> None:
        tree_facts = SouthStarMoleculeFacts.from_mol(parse_smiles("CCO"))
        disconnected_facts = SouthStarMoleculeFacts.from_mol(parse_smiles("CC.O"))

        self.assertTrue(tree_facts.graph_topology.acyclic_connected_tree)
        self.assertEqual(2, disconnected_facts.graph_topology.fragment_count)
        self.assertFalse(disconnected_facts.graph_topology.connected)
        self.assertFalse(disconnected_facts.graph_topology.acyclic_connected_tree)

    def test_graph_native_generation_accepts_precomputed_molecule_facts(self) -> None:
        smiles = "F/C=C\\Cl"
        mol = parse_smiles(smiles)
        facts = SouthStarMoleculeFacts.from_mol(mol)

        with_facts = _mol_to_smiles_enum_s_graph_native_for_mol(
            mol,
            policy_set=DEFAULT_SOUTH_STAR_POLICY_SET,
            molecule_facts=facts,
        )
        direct = mol_to_smiles_enum_s_graph_native(smiles)

        self.assertIs(
            facts,
            SouthStarComponentSupportState.from_molecule_facts(facts).molecule_facts,
        )
        self.assertEqual(direct.outputs, with_facts.outputs)
        self.assertEqual(
            direct.generation_diagnostics,
            with_facts.generation_diagnostics,
        )

    def test_unsupported_categories_are_available_before_enumeration(self) -> None:
        mol = Chem.RWMol()
        begin_idx = mol.AddAtom(Chem.Atom(6))
        end_idx = mol.AddAtom(Chem.Atom(6))
        mol.AddBond(begin_idx, end_idx, Chem.BondType.UNSPECIFIED)
        facts = SouthStarMoleculeFacts.from_mol(mol.GetMol())

        self.assertIn("unsupported_bond_type", facts.unsupported_categories)
        with self.assertRaisesRegex(NotImplementedError, "unsupported_bond_type"):
            facts.fail_if_unsupported()


if __name__ == "__main__":
    unittest.main()
