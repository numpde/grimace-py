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


@dataclass(frozen=True, slots=True)
class SouthStarPublicApiPromotionGate:
    gate_id: str
    evidence: str
    verification: str


SOUTH_STAR_PUBLIC_API_PROMOTION_GATES: tuple[SouthStarPublicApiPromotionGate, ...] = (
    SouthStarPublicApiPromotionGate(
        gate_id="private_boundary",
        evidence="MolToSmilesEnumS is not exported until all gates pass.",
        verification="PYTHONPATH=python:. python3 -m unittest "
        "tests.south_star.test_private_api_boundary -q",
    ),
    SouthStarPublicApiPromotionGate(
        gate_id="supported_domain_manifest",
        evidence="Supported domains, authorities, policies, and blockers are "
        "declared by the South Star domain manifest.",
        verification="PYTHONPATH=python:. python3 -m unittest "
        "tests.south_star.test_domain_manifest -q",
    ),
    SouthStarPublicApiPromotionGate(
        gate_id="grammar_conformance",
        evidence="Outputs satisfy the declared South Star grammar subset.",
        verification="PYTHONPATH=python:. python3 -m unittest "
        "tests.south_star.test_grammar_conformance -q",
    ),
    SouthStarPublicApiPromotionGate(
        gate_id="semantic_identity",
        evidence="Outputs parse back to the intended graph and stereo identity.",
        verification="PYTHONPATH=python:. python3 -m unittest "
        "tests.south_star.test_semantic_identity "
        "tests.south_star.test_output_correctness_harness -q",
    ),
    SouthStarPublicApiPromotionGate(
        gate_id="independent_support_completeness",
        evidence="Every promoted supported domain has an independent "
        "support-completeness oracle, not graph-native regression authority.",
        verification="PYTHONPATH=python:. python3 -m unittest "
        "tests.south_star.test_first_domain_completeness "
        "tests.south_star.test_expanded_support_fixtures -q",
    ),
    SouthStarPublicApiPromotionGate(
        gate_id="unsupported_category_completeness",
        evidence="Unsupported molecule classes fail before enumeration with "
        "manifested categories.",
        verification="PYTHONPATH=python:. python3 -m unittest "
        "tests.south_star.test_support_gates -q",
    ),
    SouthStarPublicApiPromotionGate(
        gate_id="complexity_guardrails",
        evidence="Generation diagnostics expose product-size and assignment "
        "guardrails for representative promoted domains.",
        verification="PYTHONPATH=python:. python3 -m unittest "
        "tests.south_star.test_package_readiness -q",
    ),
    SouthStarPublicApiPromotionGate(
        gate_id="documentation_contract",
        evidence="Docs name the contract, policy set, supported and unsupported "
        "domains, parser dependency, and RDKit-parity distinction.",
        verification="explicit review: docs/enum-s.md and public API docs match "
        "the exported surface.",
    ),
    SouthStarPublicApiPromotionGate(
        gate_id="release_notes_scope",
        evidence="Release notes state the semantic contract and distinguish it "
        "from MolToSmilesEnum RDKit writer parity.",
        verification="explicit review: release notes match the exact exported "
        "surface and readiness matrix.",
    ),
    SouthStarPublicApiPromotionGate(
        gate_id="full_readiness_runner",
        evidence="The named package-readiness runner passes as the promotion "
        "entry point.",
        verification="PYTHONPATH=python:. python3 -m unittest "
        "tests.run_south_star_package_readiness -q",
    ),
)


class SouthStarPackageReadinessTests(unittest.TestCase):
    def test_public_api_promotion_gate_is_explicit(self) -> None:
        gate_ids = tuple(gate.gate_id for gate in SOUTH_STAR_PUBLIC_API_PROMOTION_GATES)

        self.assertEqual(
            (
                "private_boundary",
                "supported_domain_manifest",
                "grammar_conformance",
                "semantic_identity",
                "independent_support_completeness",
                "unsupported_category_completeness",
                "complexity_guardrails",
                "documentation_contract",
                "release_notes_scope",
                "full_readiness_runner",
            ),
            gate_ids,
        )
        for gate in SOUTH_STAR_PUBLIC_API_PROMOTION_GATES:
            with self.subTest(gate_id=gate.gate_id):
                self.assertTrue(gate.evidence)
                self.assertTrue(
                    gate.verification.startswith("PYTHONPATH=python:.")
                    or gate.verification.startswith("explicit review:")
                )

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
