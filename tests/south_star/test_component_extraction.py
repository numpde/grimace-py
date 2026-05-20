from __future__ import annotations

import unittest

from tests.helpers.south_star_annotation_policy import normalized_edge
from tests.helpers.south_star_components import extract_south_star_components
from tests.helpers.south_star_semantic_oracle import parse_smiles
from tests.helpers.south_star_semantics import load_south_star_semantic_cases


def _component_carrier_edges(smiles: str) -> tuple[tuple[int, int], ...]:
    extraction = extract_south_star_components(parse_smiles(smiles))
    extraction.fail_if_unsupported()
    return tuple(
        dict.fromkeys(
            edge
            for component in extraction.components
            for edge in component.eligible_carrier_edges
        )
    )


class SouthStarComponentExtractionTests(unittest.TestCase):
    def test_fixture_carrier_edges_are_extracted_outputs(self) -> None:
        for case in load_south_star_semantic_cases():
            extracted_edges = _component_carrier_edges(case.source_smiles)
            expected_edges = tuple(
                normalized_edge(edge) for edge in case.eligible_carrier_edges
            )

            with self.subTest(case_id=case.case_id):
                self.assertEqual(expected_edges, extracted_edges)

    def test_shared_carrier_features_are_one_component(self) -> None:
        extraction = extract_south_star_components(
            parse_smiles("C/C=C/C=C/C"),
        )
        extraction.fail_if_unsupported()

        self.assertEqual(1, len(extraction.components))
        self.assertEqual(2, len(extraction.components[0].source_features))
        self.assertEqual(
            ((0, 1), (2, 3), (4, 5)),
            extraction.components[0].eligible_carrier_edges,
        )
        self.assertEqual(1, len(extraction.components[0].coupling_causes))
        self.assertEqual(
            "shared_carrier_edge",
            extraction.components[0].coupling_causes[0].category,
        )
        self.assertEqual(
            (2, 3),
            extraction.components[0].coupling_causes[0].carrier_edge,
        )
        self.assertEqual(
            ("bond:1", "bond:3"),
            extraction.components[0].coupling_causes[0].feature_ids,
        )

    def test_independent_features_are_separate_components(self) -> None:
        extraction = extract_south_star_components(
            parse_smiles("F/C=C\\CC/C=C\\Cl"),
        )
        extraction.fail_if_unsupported()

        self.assertEqual(2, len(extraction.components))
        self.assertEqual(
            [1, 1],
            [len(component.source_features) for component in extraction.components],
        )
        self.assertEqual(
            [(), ()],
            [component.coupling_causes for component in extraction.components],
        )

    def test_unsupported_gate_prevents_component_extraction(self) -> None:
        extraction = extract_south_star_components(parse_smiles("C[C@H](F)Cl"))

        self.assertFalse(extraction.supported)
        self.assertEqual((), extraction.components)
        self.assertIn(
            "atom_stereo",
            extraction.support_gate_report.categories,
        )
