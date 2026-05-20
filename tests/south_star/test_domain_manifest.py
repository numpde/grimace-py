from __future__ import annotations

import ast
from pathlib import Path
import unittest

from rdkit import Chem

from grimace._south_star.annotation_policy import MaximalEligibleCarrierAnnotationPolicy
from grimace._south_star.fragments import AllFragmentOrderPolicy
from grimace._south_star.output_order import FirstOccurrenceOutputOrderPolicy
from grimace._south_star.support_gates import south_star_support_gate_report
from tests.helpers.south_star_domain_manifest import (
    SOUTH_STAR_EXPANDED_SUPPORT_POLICY,
    SOUTH_STAR_FIRST_DOMAIN_POLICY,
    SOUTH_STAR_PRIVATE_DOMAIN,
)
from tests.helpers.south_star_exact_support import (
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
        self.assertNotEqual((), load_south_star_exact_first_domain_cases())

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
            FirstOccurrenceOutputOrderPolicy().name,
            SOUTH_STAR_PRIVATE_DOMAIN.output_order_policies,
        )

    def test_manifest_covers_observed_unsupported_categories(self) -> None:
        reports = (
            south_star_support_gate_report(Chem.MolFromSmarts("[#6]-[#8]")),
            south_star_support_gate_report(parse_smiles("[NH3]->[Cu]")),
            south_star_support_gate_report(parse_smiles("C#N.O")),
            south_star_support_gate_report(parse_smiles("C1/C=C\\CCCCC1")),
            south_star_support_gate_report(parse_smiles("[SiH3]C")),
            south_star_support_gate_report(parse_smiles("C#N")),
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
            <= SOUTH_STAR_PRIVATE_DOMAIN.unsupported_feature_categories
        )

    def test_manifest_covers_support_gate_literal_categories(self) -> None:
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
            <= SOUTH_STAR_PRIVATE_DOMAIN.unsupported_feature_categories
        )


def _empty_molecule() -> Chem.Mol:
    return Chem.Mol()


if __name__ == "__main__":
    unittest.main()
