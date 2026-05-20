from __future__ import annotations

from dataclasses import dataclass
import unittest

from grimace._south_star.enum_s import mol_to_smiles_enum_s_graph_native
from grimace._south_star.support_gates import south_star_support_gate_report
from tests.helpers.south_star_domain_manifest import (
    SOUTH_STAR_GRAPH_NATIVE_REGRESSION_AUTHORITY,
    SOUTH_STAR_PRIVATE_DOMAIN,
)
from tests.helpers.south_star_exact_support import (
    load_south_star_exact_first_domain_cases,
    load_south_star_expanded_support_cases,
)
from tests.helpers.south_star_semantic_oracle import parse_smiles


@dataclass(frozen=True, slots=True)
class SouthStarReadinessMatrix:
    oracle_backed_case_ids: tuple[str, ...]
    regression_backed_case_ids: tuple[str, ...]
    supported_feature_areas: tuple[str, ...]
    unsupported_categories: tuple[str, ...]
    policy_names: tuple[str, ...]


class SouthStarPackageReadinessTests(unittest.TestCase):
    def test_readiness_matrix_reports_evidence_classes(self) -> None:
        matrix = south_star_package_readiness_matrix()

        self.assertIn("isolated_alkene_z", matrix.oracle_backed_case_ids)
        self.assertIn(
            "explicit_bracket_hydrogen_h2",
            matrix.regression_backed_case_ids,
        )
        self.assertIn("explicit_bracket_hydrogen", matrix.supported_feature_areas)
        self.assertIn("unsupported_atom_charge", matrix.unsupported_categories)
        self.assertEqual(
            (
                "maximal_eligible_carrier",
                "all_fragment_orders",
                "first_occurrence_deduplication",
            ),
            matrix.policy_names,
        )

    def test_every_expanded_case_has_package_readiness_diagnostics(self) -> None:
        for case in load_south_star_expanded_support_cases():
            with self.subTest(case_id=case.case_id):
                report = south_star_support_gate_report(parse_smiles(case.source_smiles))
                self.assertTrue(report.supported, report.unsupported_features)

                result = mol_to_smiles_enum_s_graph_native(
                    case.source_smiles,
                    case_id=case.case_id,
                )
                self.assertEqual(case.expected_support, result.outputs)
                self.assertIsNotNone(result.generation_diagnostics)
                self.assertIn(
                    case.support_authority,
                    SOUTH_STAR_PRIVATE_DOMAIN.support_authorities,
                )


def south_star_package_readiness_matrix() -> SouthStarReadinessMatrix:
    first_domain_cases = load_south_star_exact_first_domain_cases()
    expanded_cases = load_south_star_expanded_support_cases()
    oracle_backed_case_ids = tuple(case.case_id for case in first_domain_cases) + tuple(
        case.case_id
        for case in expanded_cases
        if case.support_authority != SOUTH_STAR_GRAPH_NATIVE_REGRESSION_AUTHORITY
    )
    regression_backed_case_ids = tuple(
        case.case_id
        for case in expanded_cases
        if case.support_authority == SOUTH_STAR_GRAPH_NATIVE_REGRESSION_AUTHORITY
    )
    return SouthStarReadinessMatrix(
        oracle_backed_case_ids=oracle_backed_case_ids,
        regression_backed_case_ids=regression_backed_case_ids,
        supported_feature_areas=tuple(
            sorted(SOUTH_STAR_PRIVATE_DOMAIN.expanded_feature_areas)
        ),
        unsupported_categories=tuple(
            sorted(SOUTH_STAR_PRIVATE_DOMAIN.unsupported_feature_categories)
        ),
        policy_names=(
            "maximal_eligible_carrier",
            "all_fragment_orders",
            "first_occurrence_deduplication",
        ),
    )


if __name__ == "__main__":
    unittest.main()
