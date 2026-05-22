from __future__ import annotations

import unittest

import grimace
from rdkit import Chem
from grimace._south_star.api import mol_to_smiles_enum_s_private
from grimace._south_star.api import mol_to_smiles_enum_s_public_shape_private
from grimace._south_star.api import south_star_private_api_contract
from grimace._south_star.api import south_star_proposed_public_api_contract
from grimace._south_star.support_gates import SouthStarUnsupportedFeatureError
from tests.helpers.south_star_exact_support import (
    load_south_star_exact_first_domain_cases,
)
from tests.helpers.south_star_semantic_oracle import parse_smiles


class SouthStarPrivateApiBoundaryTests(unittest.TestCase):
    def unsupported_bond_type_mol(self) -> Chem.Mol:
        mol = Chem.RWMol()
        begin_idx = mol.AddAtom(Chem.Atom(6))
        end_idx = mol.AddAtom(Chem.Atom(6))
        mol.AddBond(begin_idx, end_idx, Chem.BondType.UNSPECIFIED)
        return mol.GetMol()

    def test_private_api_is_not_exported_from_public_package(self) -> None:
        self.assertFalse(hasattr(grimace, "MolToSmilesEnumS"))
        self.assertFalse(hasattr(grimace, "mol_to_smiles_enum_s_private"))
        self.assertFalse(
            hasattr(grimace, "mol_to_smiles_enum_s_public_shape_private")
        )
        self.assertFalse(hasattr(grimace, "south_star_private_api_contract"))

    def test_private_api_contract_names_pre_public_boundary(self) -> None:
        contract = south_star_private_api_contract()

        self.assertEqual("MolToSmilesEnumS", contract.provisional_name)
        self.assertFalse(contract.exported_from_public_package)
        self.assertEqual("rdkit.Chem.Mol", contract.accepted_input)
        self.assertEqual(
            "south_star_declared_subset_grammar_v1",
            contract.grammar_basis,
        )
        self.assertEqual(
            (
                "rdkit_parser_dependency",
                "rdkit_canonical_nonisomeric_parseback",
                "rdkit_canonical_isomeric_parseback",
            ),
            contract.semantic_equivalence_checks,
        )
        self.assertEqual("maximal_eligible_carrier", contract.annotation_policy)
        self.assertEqual("all_fragment_orders", contract.fragment_order_policy)
        self.assertEqual(
            "first_occurrence_deduplication",
            contract.output_order_policy,
        )
        self.assertIn(
            "result_generation_diagnostics",
            contract.diagnostic_boundaries,
        )
        self.assertIn(
            "support_gate_error_evidence",
            contract.diagnostic_boundaries,
        )
        self.assertEqual(
            "SouthStarUnsupportedFeatureError",
            contract.unsupported_error_type,
        )
        self.assertEqual("MolToSmilesEnum", contract.rdkit_parity_surface)

    def test_proposed_public_api_contract_is_explicit_but_not_exported(
        self,
    ) -> None:
        contract = south_star_proposed_public_api_contract()

        self.assertEqual("MolToSmilesEnumS", contract.public_name)
        self.assertFalse(contract.exported_from_public_package)
        self.assertFalse(hasattr(grimace, contract.public_name))
        self.assertEqual("rdkit.Chem.Mol", contract.accepted_input)
        self.assertEqual("tuple[str, ...]", contract.return_type)
        self.assertIn("first_occurrence_deduplication", contract.ordering_contract)
        self.assertIn("support membership", contract.ordering_contract)
        self.assertEqual("none_on_first_public_surface", contract.diagnostics_exposure)
        self.assertIn("maximal_eligible_carrier", contract.policy_surface)
        self.assertIn("all_fragment_orders", contract.policy_surface)
        self.assertIn("first_occurrence_deduplication", contract.policy_surface)
        self.assertEqual(
            "SouthStarUnsupportedFeatureError",
            contract.unsupported_error_type,
        )
        self.assertIn("not RDKit", contract.semantic_contract)
        self.assertEqual("MolToSmilesEnum", contract.rdkit_parity_surface)
        self.assertIn("promotion gates", contract.export_precondition)

    def test_private_api_accepts_rdkit_mol_and_returns_diagnostics(self) -> None:
        case = next(
            case
            for case in load_south_star_exact_first_domain_cases()
            if case.case_id == "isolated_alkene_z"
        )
        result = mol_to_smiles_enum_s_private(parse_smiles(case.source_smiles))

        self.assertEqual(case.expected_support, result.outputs)
        self.assertEqual("maximal_eligible_carrier", result.annotation_policy)
        self.assertEqual("all_fragment_orders", result.fragment_order_policy)
        self.assertEqual(
            "first_occurrence_deduplication",
            result.output_order_policy,
        )
        self.assertIsNotNone(result.generation_diagnostics)

    def test_private_api_fails_fast_for_unsupported_surfaces(self) -> None:
        with self.assertRaisesRegex(
            SouthStarUnsupportedFeatureError,
            "unsupported_bond_type",
        ) as cm:
            mol_to_smiles_enum_s_private(self.unsupported_bond_type_mol())

        self.assertEqual(frozenset({"unsupported_bond_type"}), cm.exception.categories)
        self.assertEqual(
            "unsupported_bond_type",
            cm.exception.unsupported_features[0].category,
        )

    def test_private_api_requires_rdkit_mol(self) -> None:
        with self.assertRaisesRegex(TypeError, "RDKit Mol"):
            mol_to_smiles_enum_s_private("F/C=C\\Cl")  # type: ignore[arg-type]

    def test_public_shape_private_wrapper_returns_only_strings(self) -> None:
        case = next(
            case
            for case in load_south_star_exact_first_domain_cases()
            if case.case_id == "isolated_alkene_z"
        )

        outputs = mol_to_smiles_enum_s_public_shape_private(
            parse_smiles(case.source_smiles)
        )

        self.assertEqual(case.expected_support, outputs)
        self.assertIsInstance(outputs, tuple)
        self.assertTrue(outputs)
        self.assertTrue(all(isinstance(output, str) for output in outputs))

    def test_public_shape_private_wrapper_preserves_fail_fast_boundary(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            SouthStarUnsupportedFeatureError,
            "unsupported_bond_type",
        ):
            mol_to_smiles_enum_s_public_shape_private(
                self.unsupported_bond_type_mol()
            )

    def test_public_shape_private_wrapper_requires_rdkit_mol(self) -> None:
        with self.assertRaisesRegex(TypeError, "RDKit Mol"):
            mol_to_smiles_enum_s_public_shape_private(  # type: ignore[arg-type]
                "F/C=C\\Cl"
            )


if __name__ == "__main__":
    unittest.main()
