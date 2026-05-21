from __future__ import annotations

import unittest

import grimace
from grimace._south_star.api import mol_to_smiles_enum_s_private
from tests.helpers.south_star_exact_support import (
    load_south_star_exact_first_domain_cases,
)
from tests.helpers.south_star_semantic_oracle import parse_smiles


class SouthStarPrivateApiBoundaryTests(unittest.TestCase):
    def test_private_api_is_not_exported_from_public_package(self) -> None:
        self.assertFalse(hasattr(grimace, "MolToSmilesEnumS"))
        self.assertFalse(hasattr(grimace, "mol_to_smiles_enum_s_private"))

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
        with self.assertRaisesRegex(NotImplementedError, "unsupported_bond_type"):
            mol_to_smiles_enum_s_private(parse_smiles("C$C"))

    def test_private_api_requires_rdkit_mol(self) -> None:
        with self.assertRaisesRegex(TypeError, "RDKit Mol"):
            mol_to_smiles_enum_s_private("F/C=C\\Cl")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
