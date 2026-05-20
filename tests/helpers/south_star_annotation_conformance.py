from __future__ import annotations

from tests.helpers.south_star_grammar_conformance import (
    SOUTH_STAR_GRAMMAR_CONFORMANCE_BASIS,
    SouthStarGrammarConformance,
    south_star_grammar_conformance,
)


ANNOTATION_CONFORMANCE_BASIS = SOUTH_STAR_GRAMMAR_CONFORMANCE_BASIS
SouthStarAnnotationConformance = SouthStarGrammarConformance


def south_star_annotation_conformance(
    smiles: str,
) -> SouthStarAnnotationConformance:
    """Compatibility shim for the old annotation-conformance helper name."""
    return south_star_grammar_conformance(smiles)
