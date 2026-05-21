from __future__ import annotations

import unittest

from grimace._south_star.enum_s import mol_to_smiles_enum_s_graph_native
from grimace._south_star.support_gates import south_star_support_gate_report
from tests.helpers.south_star_domain_manifest import (
    SOUTH_STAR_DISCONNECTED_COMPOSITION_WITNESS_AUTHORITY,
    SOUTH_STAR_DISCONNECTED_COMPOSITION_UNIFIED_REFERENCE_AUTHORITY,
    SOUTH_STAR_GRAPH_NATIVE_REGRESSION_AUTHORITY,
    SOUTH_STAR_NONSTEREO_MONOCYCLE_UNIFIED_REFERENCE_AUTHORITY,
    SOUTH_STAR_POLYCYCLIC_RING_STEREO_WITNESS_AUTHORITY,
    SOUTH_STAR_PRIVATE_DOMAIN,
    SOUTH_STAR_RING_STEREO_MONOCYCLE_UNIFIED_REFERENCE_AUTHORITY,
    SOUTH_STAR_RING_TETRAHEDRAL_MONOCYCLE_WITNESS_AUTHORITY,
    SOUTH_STAR_TETRAHEDRAL_ATOM_STEREO_WITNESS_AUTHORITY,
)
from tests.helpers.south_star_exact_support import (
    load_south_star_expanded_support_cases,
)
from tests.helpers.south_star_expanded_domain_oracles import (
    disconnected_composition_algebra_proof_for_case,
    ring_core_proof_records_for_case,
    shared_disconnected_composition_support_for_case,
    shared_nonstereo_monocycle_support_for_case,
    shared_ring_stereo_monocycle_support_for_case,
    shared_saturated_monocycle_support_for_case,
    shared_tetrahedral_atom_stereo_support_for_case,
)
from tests.helpers.south_star_semantic_oracle import graph_signature
from tests.helpers.south_star_semantic_oracle import parse_smiles
from tests.helpers.south_star_semantic_oracle import semantic_signature
from tests.helpers.south_star_semantic_identity import south_star_semantic_identity_report
from tests.helpers.south_star_spec_oracle import (
    SOUTH_STAR_SPEC_ORACLE_GENERATION_AUTHORITY,
    south_star_small_support_completeness_report,
)


class SouthStarExpandedSupportFixtureTests(unittest.TestCase):
    DISCONNECTED_COMPOSITION_AUTHORITIES = frozenset(
        {
            SOUTH_STAR_DISCONNECTED_COMPOSITION_WITNESS_AUTHORITY,
            SOUTH_STAR_DISCONNECTED_COMPOSITION_UNIFIED_REFERENCE_AUTHORITY,
        }
    )
    RING_CORE_AUTHORITIES = frozenset(
        {
            SOUTH_STAR_NONSTEREO_MONOCYCLE_UNIFIED_REFERENCE_AUTHORITY,
        }
    )

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

        self.assertTrue(
            any(
                case.support_authority
                == SOUTH_STAR_DISCONNECTED_COMPOSITION_UNIFIED_REFERENCE_AUTHORITY
                for case in cases
            )
        )
        self.assertTrue(
            any(
                case.support_authority == SOUTH_STAR_GRAPH_NATIVE_REGRESSION_AUTHORITY
                for case in cases
            )
        )
        self.assertTrue(
            any(
                case.support_authority
                == SOUTH_STAR_NONSTEREO_MONOCYCLE_UNIFIED_REFERENCE_AUTHORITY
                for case in cases
            )
        )
        self.assertTrue(
            any(
                case.support_authority
                == SOUTH_STAR_RING_STEREO_MONOCYCLE_UNIFIED_REFERENCE_AUTHORITY
                for case in cases
            )
        )
        self.assertTrue(
            any(
                case.support_authority
                == SOUTH_STAR_POLYCYCLIC_RING_STEREO_WITNESS_AUTHORITY
                for case in cases
            )
        )
        self.assertTrue(
            any(
                case.support_authority
                == SOUTH_STAR_RING_TETRAHEDRAL_MONOCYCLE_WITNESS_AUTHORITY
                for case in cases
            )
        )
        self.assertTrue(
            any(
                case.support_authority
                == SOUTH_STAR_TETRAHEDRAL_ATOM_STEREO_WITNESS_AUTHORITY
                for case in cases
            )
        )

    def test_saturated_monocycle_witness_matches_fixtures(self) -> None:
        for case in load_south_star_expanded_support_cases():
            if case.feature_area not in {
                "simple_saturated_monocycle",
                "branched_saturated_monocycle",
            }:
                continue

            with self.subTest(case_id=case.case_id):
                self.assertEqual(
                    SOUTH_STAR_NONSTEREO_MONOCYCLE_UNIFIED_REFERENCE_AUTHORITY,
                    case.support_authority,
                )
                self.assertEqual(
                    frozenset(case.expected_support),
                    frozenset(shared_saturated_monocycle_support_for_case(case)),
                )

    def test_nonstereo_monocycle_witness_matches_fixtures(self) -> None:
        for case in load_south_star_expanded_support_cases():
            if case.feature_area != "unsaturated_nonstereo_monocycle":
                continue

            with self.subTest(case_id=case.case_id):
                self.assertEqual(
                    SOUTH_STAR_NONSTEREO_MONOCYCLE_UNIFIED_REFERENCE_AUTHORITY,
                    case.support_authority,
                )
                self.assertEqual(
                    frozenset(case.expected_support),
                    frozenset(shared_nonstereo_monocycle_support_for_case(case)),
                )

    def test_ring_core_proof_records_expose_closure_model(self) -> None:
        for case in load_south_star_expanded_support_cases():
            if case.support_authority not in self.RING_CORE_AUTHORITIES:
                continue

            records = ring_core_proof_records_for_case(case)
            with self.subTest(case_id=case.case_id):
                self.assertNotEqual((), records)
                self.assertTrue(
                    any(
                        text == "="
                        for record in records
                        for text in record.closure_open_bond_texts
                    )
                    == (case.feature_area == "unsaturated_nonstereo_monocycle")
                )

            for record in records:
                with self.subTest(
                    case_id=case.case_id,
                    root_atom_idx=record.root_atom_idx,
                ):
                    self.assertEqual(1, len(record.closure_edges))
                    self.assertEqual(1, len(record.closure_ids))
                    self.assertEqual(("1",), record.closure_labels)
                    self.assertEqual(("open", "close"), record.closure_endpoint_roles)
                    self.assertEqual(("1", "1"), record.closure_endpoint_labels)
                    self.assertEqual(("ring_open", "ring_close"), record.closure_event_kinds)
                    self.assertEqual(("open", "close"), record.closure_event_roles)
                    self.assertEqual(("1", "1"), record.closure_event_labels)
                    self.assertEqual(1, len(record.closure_open_bond_texts))
                    self.assertEqual(0, record.marker_slot_count)
                    self.assertEqual(0, record.renderer_input_count)

    def test_disconnected_composition_witness_matches_fixtures(self) -> None:
        for case in load_south_star_expanded_support_cases():
            if (
                case.support_authority
                not in self.DISCONNECTED_COMPOSITION_AUTHORITIES
            ):
                continue

            with self.subTest(case_id=case.case_id):
                result = shared_disconnected_composition_support_for_case(case)
                self.assertEqual(case.expected_support, result.outputs)
                self.assertEqual(
                    len(case.expected_support),
                    result.estimated_product_size,
                )
                self.assertEqual("all_fragment_orders", result.fragment_order_policy)
                self.assertEqual(2, result.fragment_order_count)
                self.assertEqual(
                    result.fragment_output_counts,
                    tuple(
                        record.output_count
                        for record in result.fragment_generation_records
                    ),
                )
                self.assertEqual(
                    tuple(
                        f"fragment:{fragment_idx}"
                        for fragment_idx in range(result.fragment_count)
                    ),
                    tuple(
                        record.fragment_id
                        for record in result.fragment_generation_records
                    ),
                )
                self.assertTrue(
                    all(
                        record.source_atom_indices
                        and record.source_fragment_smiles
                        for record in result.fragment_generation_records
                    )
                )

    def test_disconnected_composition_algebra_reconstructs_fixture_support(
        self,
    ) -> None:
        for case in load_south_star_expanded_support_cases():
            if (
                case.support_authority
                not in self.DISCONNECTED_COMPOSITION_AUTHORITIES
            ):
                continue

            with self.subTest(case_id=case.case_id):
                proof = disconnected_composition_algebra_proof_for_case(case)

                self.assertEqual(case.support_authority, proof.support_authority)
                self.assertEqual(
                    case.support_authority
                    == SOUTH_STAR_DISCONNECTED_COMPOSITION_UNIFIED_REFERENCE_AUTHORITY,
                    proof.support_authority_promoted,
                )
                self.assertEqual(case.expected_support, proof.composed_outputs)
                self.assertEqual(proof.graph_native_outputs, proof.composed_outputs)
                self.assertEqual(2, proof.fragment_count)
                self.assertEqual(2, proof.fragment_order_count)
                self.assertEqual("all_fragment_orders", proof.fragment_order_policy)
                self.assertEqual(
                    "first_occurrence_deduplication",
                    proof.output_order_policy,
                )
                self.assertEqual(
                    len(case.expected_support),
                    proof.estimated_product_size,
                )
                self.assertEqual(
                    len(proof.fragment_source_smiles),
                    proof.fragment_count,
                )
                self.assertEqual(
                    len(proof.fragment_output_counts),
                    proof.fragment_count,
                )
                if proof.support_authority_promoted:
                    self.assertTrue(
                        all(
                            authority.startswith("unified_reference_")
                            for authority in proof.fragment_support_authorities
                        )
                    )

    def test_ring_stereo_monocycle_witness_matches_fixtures(self) -> None:
        for case in load_south_star_expanded_support_cases():
            if (
                case.support_authority
                != SOUTH_STAR_RING_STEREO_MONOCYCLE_UNIFIED_REFERENCE_AUTHORITY
            ):
                continue

            with self.subTest(case_id=case.case_id):
                result = shared_ring_stereo_monocycle_support_for_case(case)
                self.assertEqual(case.expected_support, result.outputs)
                self.assertGreater(result.closure_edge_count, 0)
                self.assertEqual(2, result.marker_assignment_count)
                self.assertGreater(len(result.equations), 0)
                self.assertTrue(
                    any(
                        equation.syntax_position == "ring_open"
                        and equation.slot_id.startswith("ring_open:")
                        for equation in result.equations
                    )
                )
                for equation in result.equations:
                    self.assertIn(equation.graph_marker, {"/", "\\"})
                    self.assertIn(equation.emitted_marker, {"/", "\\"})
                    self.assertEqual(
                        equation.traversal_orientation_flip,
                        equation.graph_marker != equation.emitted_marker,
                    )

    def test_tetrahedral_atom_stereo_witness_matches_fixtures(self) -> None:
        for case in load_south_star_expanded_support_cases():
            if (
                case.support_authority
                != SOUTH_STAR_TETRAHEDRAL_ATOM_STEREO_WITNESS_AUTHORITY
            ):
                continue

            with self.subTest(case_id=case.case_id):
                result = shared_tetrahedral_atom_stereo_support_for_case(case)
                self.assertEqual(case.expected_support, result.outputs)
                self.assertEqual(len(case.expected_support), result.output_count)
                self.assertEqual(
                    len(case.expected_support),
                    result.expected_output_count,
                )
                self.assertTrue(result.fixture_cross_check_passed)
                self.assertGreater(len(result.diagnostics), 0)
                self.assertEqual(len(result.diagnostics), len(result.proof_inputs))
                for diagnostic in result.diagnostics:
                    self.assertTrue(diagnostic.preserves_orientation)
                    self.assertEqual(
                        diagnostic.expected_token,
                        diagnostic.emitted_token,
                    )
                    self.assertIn(diagnostic.emitted_token, {"@", "@@"})
                for proof_input, diagnostic in zip(
                    result.proof_inputs,
                    result.diagnostics,
                ):
                    self.assertEqual(
                        proof_input.expected_token,
                        proof_input.renderer_input.value,
                    )
                    self.assertEqual(
                        proof_input.expected_token,
                        diagnostic.expected_token,
                    )
                    self.assertEqual(
                        proof_input.emitted_ligand_order,
                        diagnostic.emitted_ligand_order,
                    )
                    self.assertEqual((), proof_input.ring_closure_ligand_atom_indices)

    def test_tetrahedral_token_obligation_outputs_preserve_semantics(self) -> None:
        case_ids = {
            "implicit_h_tetrahedral_center",
            "quaternary_tetrahedral_center",
        }
        for case in load_south_star_expanded_support_cases():
            if case.case_id not in case_ids:
                continue

            result = shared_tetrahedral_atom_stereo_support_for_case(case)
            with self.subTest(case_id=case.case_id):
                self.assertEqual(result.expected_output_count, result.output_count)
                self.assertTrue(result.fixture_cross_check_passed)

            for output in result.outputs:
                with self.subTest(case_id=case.case_id, output=output):
                    parse_smiles(output)
                    self.assertEqual(
                        graph_signature(case.source_smiles),
                        graph_signature(output),
                    )
                    self.assertEqual(
                        semantic_signature(case.source_smiles),
                        semantic_signature(output),
                    )

    def test_tetrahedral_traversal_support_rejects_wrong_parity_tokens(self) -> None:
        cases = {
            "implicit_h_tetrahedral_center": "C[C@@H](F)Cl",
            "quaternary_tetrahedral_center": "C[C@@](F)(Cl)Br",
        }
        expanded_cases = {
            case.case_id: case for case in load_south_star_expanded_support_cases()
        }

        for case_id, wrong_parity_smiles in cases.items():
            case = expanded_cases[case_id]
            report = south_star_semantic_identity_report(
                source_smiles=case.source_smiles,
                candidate_smiles=wrong_parity_smiles,
            )

            with self.subTest(case_id=case_id):
                self.assertNotIn(
                    wrong_parity_smiles,
                    shared_tetrahedral_atom_stereo_support_for_case(case).outputs,
                )
                self.assertTrue(report.graph_identity.passed)
                self.assertFalse(report.stereo_identity.passed)

    def test_graph_native_support_matches_expanded_domain_fixtures(self) -> None:
        for case in load_south_star_expanded_support_cases():
            result = mol_to_smiles_enum_s_graph_native(
                case.source_smiles,
                case_id=case.case_id,
            )

            with self.subTest(case_id=case.case_id):
                self.assertEqual(case.case_id, result.case_id)
                self.assertEqual(case.expected_support, result.outputs)

    def test_small_graph_native_regressions_have_completeness_evidence(self) -> None:
        feature_areas = {
            "charged_atom_text",
            "explicit_bracket_hydrogen",
            "radical_atom_text",
        }
        for case in load_south_star_expanded_support_cases():
            if (
                case.support_authority != SOUTH_STAR_GRAPH_NATIVE_REGRESSION_AUTHORITY
                or case.feature_area not in feature_areas
            ):
                continue

            result = mol_to_smiles_enum_s_graph_native(
                case.source_smiles,
                case_id=case.case_id,
            )
            report = south_star_small_support_completeness_report(
                source_smiles=case.source_smiles,
                observed_support=result.outputs,
            )

            with self.subTest(case_id=case.case_id):
                self.assertEqual(case.expected_support, result.outputs)
                self.assertEqual(case.expected_support, report.expected_support)
                self.assertTrue(report.complete)
                self.assertEqual(
                    SOUTH_STAR_SPEC_ORACLE_GENERATION_AUTHORITY,
                    report.generation_authority,
                )

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
