"""Presentation annotation-policy selectors for the private proof kernel."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .constraints import NamedConstraint
from .policy import AnnotationMode


@dataclass(frozen=True, slots=True)
class ValidWitness:
    """A satisfying assignment after semantic constraints."""

    id: str
    rendered: str
    annotation_count: int
    constraints: tuple[NamedConstraint, ...]

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("witness id must be nonempty")
        if not self.rendered:
            raise ValueError("rendered witness string must be nonempty")
        if self.annotation_count < 0:
            raise ValueError("annotation count must be nonnegative")
        if not self.constraints:
            raise ValueError("valid witness must cite named constraints")


def select_annotation_witnesses(
    mode: AnnotationMode,
    witnesses: Iterable[ValidWitness],
) -> tuple[ValidWitness, ...]:
    """Select from already-valid witnesses according to annotation policy.

    This selector is deliberately downstream of semantic validity. It never
    repairs, parses, or filters by rendered string. Rendered witness
    multiplicity is available through ``WitnessImage`` diagnostics, not through
    the deduplicated support image.
    """

    witness_tuple = tuple(witnesses)
    if mode is AnnotationMode.CANONICAL:
        raise NotImplementedError("canonical annotation policy is not defined yet")
    if mode is AnnotationMode.CARDINALITY_MAXIMAL:
        if not witness_tuple:
            return ()
        max_count = max(witness.annotation_count for witness in witness_tuple)
        return tuple(
            witness
            for witness in witness_tuple
            if witness.annotation_count == max_count
        )
    if mode in (AnnotationMode.HARD, AnnotationMode.SUPPORT_MAXIMAL):
        return witness_tuple
    raise ValueError(f"unknown annotation mode: {mode!r}")


__all__ = (
    "ValidWitness",
    "select_annotation_witnesses",
)
