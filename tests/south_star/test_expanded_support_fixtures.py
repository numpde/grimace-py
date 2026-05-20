from __future__ import annotations

import unittest

from grimace._south_star.enum_s import mol_to_smiles_enum_s_graph_native
from grimace._south_star.support_gates import south_star_support_gate_report
from tests.helpers.south_star_exact_support import (
    load_south_star_expanded_support_cases,
)
from tests.helpers.south_star_domain_manifest import SOUTH_STAR_PRIVATE_DOMAIN
from tests.helpers.south_star_semantic_oracle import graph_signature
from tests.helpers.south_star_semantic_oracle import parse_smiles
from tests.helpers.south_star_semantic_oracle import semantic_signature


class SouthStarExpandedSupportFixtureTests(unittest.TestCase):
    def test_expanded_support_fixture_covers_required_feature_areas(self) -> None:
        feature_areas = {
            case.feature_area for case in load_south_star_expanded_support_cases()
        }

        self.assertTrue(SOUTH_STAR_PRIVATE_DOMAIN.expanded_feature_areas <= feature_areas)

    def test_expanded_support_fixtures_have_explicit_authority(self) -> None:
        cases = load_south_star_expanded_support_cases()

        self.assertNotEqual((), cases)
        for case in cases:
            with self.subTest(case_id=case.case_id):
                self.assertEqual(
                    "graph_native_regression_with_semantic_parseback",
                    case.support_authority,
                )
                self.assertIn(
                    case.support_authority,
                    SOUTH_STAR_PRIVATE_DOMAIN.support_authorities,
                )
                self.assertIn(
                    case.feature_area,
                    SOUTH_STAR_PRIVATE_DOMAIN.expanded_feature_areas,
                )
                self.assertNotEqual("", case.feature_area)
                self.assertNotEqual("", case.evidence_notes)

    def test_graph_native_support_matches_expanded_domain_fixtures(self) -> None:
        for case in load_south_star_expanded_support_cases():
            result = mol_to_smiles_enum_s_graph_native(
                case.source_smiles,
                case_id=case.case_id,
            )

            with self.subTest(case_id=case.case_id):
                self.assertEqual(case.case_id, result.case_id)
                self.assertEqual(case.expected_support, result.outputs)

    def test_expanded_domain_fixture_outputs_are_semantic_evidence(self) -> None:
        for case in load_south_star_expanded_support_cases():
            source_graph = graph_signature(case.source_smiles)
            source_semantics = semantic_signature(case.source_smiles)
            for smiles in case.expected_support:
                with self.subTest(case_id=case.case_id, smiles=smiles):
                    parse_smiles(smiles)
                    self.assertEqual(source_graph, graph_signature(smiles))
                    self.assertEqual(source_semantics, semantic_signature(smiles))

    def test_expanded_domain_sources_are_inside_gate_scope(self) -> None:
        for case in load_south_star_expanded_support_cases():
            report = south_star_support_gate_report(parse_smiles(case.source_smiles))

            with self.subTest(case_id=case.case_id):
                self.assertTrue(report.supported, report.unsupported_features)


if __name__ == "__main__":
    unittest.main()
