from __future__ import annotations

import unittest

from tests.helpers.south_star_semantic_oracle import graph_signature
from tests.helpers.south_star_semantic_oracle import semantic_signature
from tests.helpers.south_star_semantics import load_south_star_semantic_cases


class SouthStarSemanticWitnessTests(unittest.TestCase):
    def test_cases_document_semantic_feature_and_writer_comparison_status(self) -> None:
        allowed_writer_statuses = {"not_checked", "comparison_known_divergence"}
        for case in load_south_star_semantic_cases():
            with self.subTest(case_id=case.case_id):
                self.assertNotEqual("", case.semantic_feature)
                self.assertIn(
                    case.rdkit_writer_membership_status,
                    allowed_writer_statuses,
                )
                self.assertNotEqual("", case.rdkit_writer_membership_notes)

    def test_positive_witnesses_parse_to_intended_graph_and_stereo(self) -> None:
        for case in load_south_star_semantic_cases():
            source_graph = graph_signature(case.source_smiles)
            source_semantics = semantic_signature(case.source_smiles)

            for candidate in case.positive_semantic_smiles:
                with self.subTest(case_id=case.case_id, candidate=candidate):
                    self.assertEqual(source_graph, graph_signature(candidate))
                    self.assertEqual(source_semantics, semantic_signature(candidate))

    def test_negative_witnesses_keep_graph_but_change_or_lose_stereo(self) -> None:
        for case in load_south_star_semantic_cases():
            source_graph = graph_signature(case.source_smiles)
            source_semantics = semantic_signature(case.source_smiles)

            for negative in case.negative_semantic_smiles:
                with self.subTest(
                    case_id=case.case_id,
                    candidate=negative.smiles,
                    reason=negative.reason,
                ):
                    self.assertEqual(source_graph, graph_signature(negative.smiles))
                    self.assertNotEqual(
                        source_semantics,
                        semantic_signature(negative.smiles),
                    )

    def test_fixture_separates_carriers_from_annotation_policy(self) -> None:
        for case in load_south_star_semantic_cases():
            with self.subTest(case_id=case.case_id):
                self.assertGreater(len(case.eligible_carrier_edges), 0)
                self.assertLessEqual(
                    case.maximal_eligible_carrier.required_marker_edge_count,
                    len(case.eligible_carrier_edges),
                )
