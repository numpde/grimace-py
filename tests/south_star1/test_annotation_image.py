"""Tests for South Star 1 annotation selection and render-image semantics."""

from __future__ import annotations

import ast
import inspect
import unittest

from grimace._south_star1.annotation import ValidWitness
from grimace._south_star1.annotation import select_annotation_witnesses
from grimace._south_star1.constraints import NamedConstraint
from grimace._south_star1.enumerate import render_image_from_witnesses
from grimace._south_star1.policy import AnnotationMode


class AnnotationImageTest(unittest.TestCase):
    def test_support_image_preserves_rendered_witness_multiplicity(self) -> None:
        witnesses = (
            _witness("w1", "CCO", 0),
            _witness("w2", "CCO", 1),
            _witness("w3", "OCC", 0),
        )

        image = render_image_from_witnesses(witnesses)

        self.assertEqual(image.witness_count, 3)
        self.assertEqual(image.distinct_count, 2)
        self.assertEqual(image.strings, ("CCO", "CCO", "OCC"))

    def test_distinct_count_is_only_instrumentation(self) -> None:
        witnesses = (_witness("w1", "CCO", 0), _witness("w2", "CCO", 1))

        image = render_image_from_witnesses(witnesses)

        self.assertEqual(image.witness_count, 2)
        self.assertEqual(image.distinct_count, 1)
        self.assertEqual(image.strings, ("CCO", "CCO"))

    def test_cardinality_maximal_selects_before_render_image(self) -> None:
        witnesses = (
            _witness("w1", "CCO", 0),
            _witness("w2", "CCO", 2),
            _witness("w3", "OCC", 2),
        )

        selected = select_annotation_witnesses(
            AnnotationMode.CARDINALITY_MAXIMAL,
            witnesses,
        )

        self.assertEqual(tuple(witness.id for witness in selected), ("w2", "w3"))

    def test_support_maximal_preserves_witnesses(self) -> None:
        witnesses = (_witness("w1", "CCO", 0), _witness("w2", "CCO", 1))

        selected = select_annotation_witnesses(
            AnnotationMode.SUPPORT_MAXIMAL,
            witnesses,
        )

        self.assertEqual(selected, witnesses)

    def test_canonical_policy_is_explicitly_unimplemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            select_annotation_witnesses(
                AnnotationMode.CANONICAL,
                (_witness("w1", "CCO", 0),),
            )

    def test_enumeration_image_contains_no_parser_filter(self) -> None:
        tree = ast.parse(inspect.getsource(render_image_from_witnesses))
        imported_names = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }

        self.assertNotIn("rdkit", imported_names)
        self.assertNotIn("Chem", inspect.getsource(render_image_from_witnesses))
        self.assertNotIn("MolFromSmiles", inspect.getsource(render_image_from_witnesses))


def _witness(witness_id: str, rendered: str, annotation_count: int) -> ValidWitness:
    return ValidWitness(
        id=witness_id,
        rendered=rendered,
        annotation_count=annotation_count,
        constraints=(NamedConstraint("semantic_validity", "assignment"),),
    )


if __name__ == "__main__":
    unittest.main()
