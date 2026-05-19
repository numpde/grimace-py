from __future__ import annotations

import unittest

from rdkit import Chem

from tests.helpers.south_star_semantics import load_south_star_semantic_cases


def _parse_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise AssertionError(f"failed to parse SMILES {smiles!r}")
    return mol


def _graph_signature(smiles: str) -> str:
    return Chem.MolToSmiles(
        _parse_smiles(smiles),
        canonical=True,
        isomericSmiles=False,
    )


def _semantic_signature(smiles: str) -> str:
    return Chem.MolToSmiles(
        _parse_smiles(smiles),
        canonical=True,
        isomericSmiles=True,
    )


class SouthStarSemanticWitnessTests(unittest.TestCase):
    def test_positive_witnesses_parse_to_intended_graph_and_stereo(self) -> None:
        for case in load_south_star_semantic_cases():
            source_graph = _graph_signature(case.source_smiles)
            source_semantics = _semantic_signature(case.source_smiles)

            for candidate in case.positive_semantic_smiles:
                with self.subTest(case_id=case.case_id, candidate=candidate):
                    self.assertEqual(source_graph, _graph_signature(candidate))
                    self.assertEqual(source_semantics, _semantic_signature(candidate))

    def test_negative_witnesses_keep_graph_but_change_or_lose_stereo(self) -> None:
        for case in load_south_star_semantic_cases():
            source_graph = _graph_signature(case.source_smiles)
            source_semantics = _semantic_signature(case.source_smiles)

            for negative in case.negative_semantic_smiles:
                with self.subTest(
                    case_id=case.case_id,
                    candidate=negative.smiles,
                    reason=negative.reason,
                ):
                    self.assertEqual(source_graph, _graph_signature(negative.smiles))
                    self.assertNotEqual(
                        source_semantics,
                        _semantic_signature(negative.smiles),
                    )

    def test_fixture_separates_carriers_from_annotation_policy(self) -> None:
        for case in load_south_star_semantic_cases():
            with self.subTest(case_id=case.case_id):
                self.assertGreater(len(case.eligible_carrier_edges), 0)
                self.assertLessEqual(
                    case.maximal_eligible_carrier.required_marker_edge_count,
                    len(case.eligible_carrier_edges),
                )
                self.assertEqual("not_checked", case.rdkit_writer_membership_status)
