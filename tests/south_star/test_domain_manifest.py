from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest

from rdkit import Chem

from grimace._south_star.annotation_policy import MaximalEligibleCarrierAnnotationPolicy
from grimace._south_star.aromatic_policy import (
    DEFAULT_SOUTH_STAR_AROMATIC_POLICY_CONTRACT,
    SOUTH_STAR_AROMATIC_POLICY_FAMILY_CONTRACTS,
)
from grimace._south_star.fragments import AllFragmentOrderPolicy
from grimace._south_star.output_order import FirstOccurrenceOutputOrderPolicy
from grimace._south_star.support_gates import south_star_support_gate_report
from tests.helpers.south_star_domain_manifest import (
    SOUTH_STAR_AROMATIC_SELENIUM_TEXT_UNIFIED_REFERENCE_AUTHORITY,
    SOUTH_STAR_DISCONNECTED_COMPOSITION_UNIFIED_REFERENCE_AUTHORITY,
    SOUTH_STAR_EXPANDED_SUPPORT_POLICY,
    SOUTH_STAR_FIRST_DOMAIN_UNIFIED_REFERENCE_AUTHORITY,
    SOUTH_STAR_FIRST_DOMAIN_POLICY,
    SOUTH_STAR_MARKERLESS_ACYCLIC_TREE_UNIFIED_REFERENCE_AUTHORITY,
    SOUTH_STAR_NONSTEREO_MONOCYCLE_UNIFIED_REFERENCE_AUTHORITY,
    SOUTH_STAR_PRIVATE_DOMAIN,
    SOUTH_STAR_REGRESSION_WITNESS_AUTHORITIES,
    SOUTH_STAR_RING_STEREO_MONOCYCLE_UNIFIED_REFERENCE_AUTHORITY,
    SOUTH_STAR_SHARED_PIPELINE_ELIGIBLE_EXPANDED_FEATURE_AREAS,
    SOUTH_STAR_SINGLE_ATOM_ATOM_TEXT_UNIFIED_REFERENCE_AUTHORITY,
    SOUTH_STAR_TEMPORARY_WITNESS_FOLD_IN_PLANS,
    SOUTH_STAR_TEMPORARY_WITNESS_AUTHORITIES,
    SOUTH_STAR_TWO_ATOM_BOND_TEXT_UNIFIED_REFERENCE_AUTHORITY,
    SOUTH_STAR_TWO_ATOM_MARKERLESS_ATOM_TEXT_UNIFIED_REFERENCE_AUTHORITY,
    SOUTH_STAR_UNIFIED_REFERENCE_AUTHORITIES,
)
from tests.helpers.south_star_exact_support import (
    EXACT_FIRST_DOMAIN_FIXTURE,
    load_south_star_exact_first_domain_cases,
    load_south_star_expanded_support_cases,
)
from tests.helpers.south_star_semantic_oracle import parse_smiles


class SouthStarDomainManifestTests(unittest.TestCase):
    def test_manifest_names_current_fixture_policies(self) -> None:
        self.assertIn(
            SOUTH_STAR_FIRST_DOMAIN_POLICY,
            SOUTH_STAR_PRIVATE_DOMAIN.fixture_policies,
        )
        self.assertIn(
            SOUTH_STAR_EXPANDED_SUPPORT_POLICY,
            SOUTH_STAR_PRIVATE_DOMAIN.fixture_policies,
        )

    def test_manifest_covers_loaded_fixture_domains(self) -> None:
        first_domain_cases = load_south_star_exact_first_domain_cases()
        self.assertNotEqual((), first_domain_cases)
        for case in first_domain_cases:
            with self.subTest(case_id=case.case_id):
                self.assertIn(
                    case.support_authority,
                    SOUTH_STAR_PRIVATE_DOMAIN.support_authorities,
                )

        for case in load_south_star_expanded_support_cases():
            with self.subTest(case_id=case.case_id):
                self.assertIn(
                    case.feature_area,
                    SOUTH_STAR_PRIVATE_DOMAIN.expanded_feature_areas,
                )
                self.assertIn(
                    case.support_authority,
                    SOUTH_STAR_PRIVATE_DOMAIN.support_authorities,
                )

    def test_shared_pipeline_eligibility_covers_expanded_feature_areas(self) -> None:
        self.assertEqual(
            SOUTH_STAR_PRIVATE_DOMAIN.expanded_feature_areas,
            SOUTH_STAR_SHARED_PIPELINE_ELIGIBLE_EXPANDED_FEATURE_AREAS,
        )

    def test_feature_areas_are_separate_from_support_gate_blockers(self) -> None:
        self.assertFalse(
            SOUTH_STAR_PRIVATE_DOMAIN.expanded_feature_areas
            & SOUTH_STAR_PRIVATE_DOMAIN.support_gate_blocker_categories,
        )
        self.assertIn(
            "simple_saturated_monocycle",
            SOUTH_STAR_PRIVATE_DOMAIN.expanded_feature_areas,
        )
        self.assertIn(
            "ring_molecule",
            SOUTH_STAR_PRIVATE_DOMAIN.support_gate_blocker_categories,
        )

    def test_manifest_classifies_support_evidence_authorities(self) -> None:
        classified_authorities = (
            SOUTH_STAR_REGRESSION_WITNESS_AUTHORITIES
            | SOUTH_STAR_TEMPORARY_WITNESS_AUTHORITIES
            | SOUTH_STAR_UNIFIED_REFERENCE_AUTHORITIES
        )

        self.assertEqual(
            SOUTH_STAR_PRIVATE_DOMAIN.support_authorities,
            classified_authorities,
        )
        for authority in SOUTH_STAR_PRIVATE_DOMAIN.support_authorities:
            with self.subTest(authority=authority):
                self.assertNotIn("independent_", authority)
                self.assertFalse(authority.endswith("_oracle"))

    def test_temporary_witness_authorities_have_fold_in_plans(self) -> None:
        self.assertEqual(
            SOUTH_STAR_TEMPORARY_WITNESS_AUTHORITIES,
            frozenset(SOUTH_STAR_TEMPORARY_WITNESS_FOLD_IN_PLANS),
        )
        for authority, plan in SOUTH_STAR_TEMPORARY_WITNESS_FOLD_IN_PLANS.items():
            with self.subTest(authority=authority):
                self.assertIn("unified-reference", plan)
                self.assertIn("shared", plan)
                self.assertNotIn("permanent", plan.lower())

    def test_first_domain_fixture_declares_unified_reference_authority(self) -> None:
        raw = json.loads(EXACT_FIRST_DOMAIN_FIXTURE.read_text())

        self.assertEqual(
            SOUTH_STAR_FIRST_DOMAIN_UNIFIED_REFERENCE_AUTHORITY,
            raw["support_authority"],
        )

    def test_expanded_fixture_evidence_notes_match_authority_class(self) -> None:
        for case in load_south_star_expanded_support_cases():
            with self.subTest(case_id=case.case_id):
                if case.support_authority in SOUTH_STAR_TEMPORARY_WITNESS_AUTHORITIES:
                    self.assertIn("Temporary witness evidence", case.evidence_notes)
                elif case.support_authority in SOUTH_STAR_REGRESSION_WITNESS_AUTHORITIES:
                    self.assertIn("Pinned graph-native output", case.evidence_notes)
                else:
                    self.assertIn(
                        case.support_authority,
                        SOUTH_STAR_UNIFIED_REFERENCE_AUTHORITIES,
                    )
                    self.assertIn("Unified-reference", case.evidence_notes)

    def test_manifest_names_first_unified_reference_authority(self) -> None:
        self.assertIn(
            SOUTH_STAR_SINGLE_ATOM_ATOM_TEXT_UNIFIED_REFERENCE_AUTHORITY,
            SOUTH_STAR_UNIFIED_REFERENCE_AUTHORITIES,
        )
        self.assertIn(
            SOUTH_STAR_SINGLE_ATOM_ATOM_TEXT_UNIFIED_REFERENCE_AUTHORITY,
            SOUTH_STAR_PRIVATE_DOMAIN.support_authorities,
        )
        self.assertIn(
            SOUTH_STAR_TWO_ATOM_MARKERLESS_ATOM_TEXT_UNIFIED_REFERENCE_AUTHORITY,
            SOUTH_STAR_UNIFIED_REFERENCE_AUTHORITIES,
        )
        self.assertIn(
            SOUTH_STAR_TWO_ATOM_MARKERLESS_ATOM_TEXT_UNIFIED_REFERENCE_AUTHORITY,
            SOUTH_STAR_PRIVATE_DOMAIN.support_authorities,
        )
        self.assertIn(
            SOUTH_STAR_TWO_ATOM_BOND_TEXT_UNIFIED_REFERENCE_AUTHORITY,
            SOUTH_STAR_UNIFIED_REFERENCE_AUTHORITIES,
        )
        self.assertIn(
            SOUTH_STAR_TWO_ATOM_BOND_TEXT_UNIFIED_REFERENCE_AUTHORITY,
            SOUTH_STAR_PRIVATE_DOMAIN.support_authorities,
        )
        self.assertIn(
            SOUTH_STAR_FIRST_DOMAIN_UNIFIED_REFERENCE_AUTHORITY,
            SOUTH_STAR_UNIFIED_REFERENCE_AUTHORITIES,
        )
        self.assertIn(
            SOUTH_STAR_AROMATIC_SELENIUM_TEXT_UNIFIED_REFERENCE_AUTHORITY,
            SOUTH_STAR_UNIFIED_REFERENCE_AUTHORITIES,
        )
        self.assertIn(
            SOUTH_STAR_MARKERLESS_ACYCLIC_TREE_UNIFIED_REFERENCE_AUTHORITY,
            SOUTH_STAR_UNIFIED_REFERENCE_AUTHORITIES,
        )
        self.assertIn(
            SOUTH_STAR_NONSTEREO_MONOCYCLE_UNIFIED_REFERENCE_AUTHORITY,
            SOUTH_STAR_UNIFIED_REFERENCE_AUTHORITIES,
        )
        self.assertIn(
            SOUTH_STAR_RING_STEREO_MONOCYCLE_UNIFIED_REFERENCE_AUTHORITY,
            SOUTH_STAR_UNIFIED_REFERENCE_AUTHORITIES,
        )
        self.assertIn(
            SOUTH_STAR_DISCONNECTED_COMPOSITION_UNIFIED_REFERENCE_AUTHORITY,
            SOUTH_STAR_UNIFIED_REFERENCE_AUTHORITIES,
        )

    def test_manifest_names_fragment_order_policy(self) -> None:
        self.assertIn(
            AllFragmentOrderPolicy().name,
            SOUTH_STAR_PRIVATE_DOMAIN.fragment_order_policies,
        )

    def test_manifest_names_runtime_policy_objects(self) -> None:
        self.assertIn(
            MaximalEligibleCarrierAnnotationPolicy().name,
            SOUTH_STAR_PRIVATE_DOMAIN.annotation_policies,
        )
        self.assertIn(
            DEFAULT_SOUTH_STAR_AROMATIC_POLICY_CONTRACT.name,
            SOUTH_STAR_PRIVATE_DOMAIN.aromatic_policy_contracts,
        )
        self.assertEqual(
            frozenset(
                contract.name
                for contract in SOUTH_STAR_AROMATIC_POLICY_FAMILY_CONTRACTS
                if contract.status == "active"
            ),
            SOUTH_STAR_PRIVATE_DOMAIN.aromatic_policy_contracts,
        )
        self.assertIn(
            FirstOccurrenceOutputOrderPolicy().name,
            SOUTH_STAR_PRIVATE_DOMAIN.output_order_policies,
        )

    def test_manifest_covers_observed_support_gate_blocker_categories(self) -> None:
        reports = (
            south_star_support_gate_report(Chem.MolFromSmarts("[#6]-[#8]")),
            south_star_support_gate_report(parse_smiles("[NH3]->[Cu]")),
            south_star_support_gate_report(parse_smiles("[Na+].O")),
            south_star_support_gate_report(parse_smiles("C1/C=C\\CCCCC1")),
            south_star_support_gate_report(parse_smiles("[SiH3]C")),
            south_star_support_gate_report(_unsupported_bond_type_mol()),
            south_star_support_gate_report(parse_smiles("[2H][H]")),
            south_star_support_gate_report(parse_smiles("[H+]")),
            south_star_support_gate_report(parse_smiles("[H]")),
            south_star_support_gate_report(parse_smiles("[CH3:1]C")),
            south_star_support_gate_report(parse_smiles("c1ccccc1")),
            south_star_support_gate_report(parse_smiles("C1CC2CCCC2C1")),
            south_star_support_gate_report(parse_smiles("F[C@H]1CCCC(C)C1")),
            south_star_support_gate_report(_empty_molecule()),
        )
        observed_categories = frozenset(
            category for report in reports for category in report.categories
        )

        self.assertTrue(
            observed_categories
            <= SOUTH_STAR_PRIVATE_DOMAIN.support_gate_blocker_categories
        )

    def test_manifest_covers_support_gate_literal_blocker_categories(self) -> None:
        support_gate_source = (
            Path(__file__).resolve().parents[2]
            / "python"
            / "grimace"
            / "_south_star"
            / "support_gates.py"
        )
        tree = ast.parse(support_gate_source.read_text())
        literal_categories = frozenset(
            keyword.value.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for keyword in node.keywords
            if keyword.arg == "category"
            and isinstance(keyword.value, ast.Constant)
            and isinstance(keyword.value.value, str)
        )

        self.assertTrue(
            literal_categories
            <= SOUTH_STAR_PRIVATE_DOMAIN.support_gate_blocker_categories
        )


def _empty_molecule() -> Chem.Mol:
    return Chem.Mol()


def _unsupported_bond_type_mol() -> Chem.Mol:
    mol = Chem.RWMol()
    begin_idx = mol.AddAtom(Chem.Atom(6))
    end_idx = mol.AddAtom(Chem.Atom(6))
    mol.AddBond(begin_idx, end_idx, Chem.BondType.UNSPECIFIED)
    return mol.GetMol()


if __name__ == "__main__":
    unittest.main()
