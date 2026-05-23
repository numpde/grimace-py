"""Rendered witness records and count-only witness diagnostics."""

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


def select_witnesses_by_annotation_count(
    mode: AnnotationMode,
    witnesses: Iterable[ValidWitness],
) -> tuple[ValidWitness, ...]:
    """Select from already-valid witnesses using only annotation counts.

    Support-wise maximality is intentionally not implemented here: at the
    witness layer, carrier-support identities are already gone.  Real
    support-maximal selection belongs in ``stereo_csp.select_stereo_solutions``,
    where ``StereoSolution.marker_support`` is still available.
    """

    witness_tuple = tuple(witnesses)
    if mode is AnnotationMode.CANONICAL:
        raise NotImplementedError("canonical annotation policy is not defined yet")
    if mode is AnnotationMode.SUPPORT_MAXIMAL:
        raise NotImplementedError(
            "support-maximal selection requires StereoSolution.marker_support"
        )
    if mode is AnnotationMode.CARDINALITY_MAXIMAL:
        if not witness_tuple:
            return ()
        max_count = max(witness.annotation_count for witness in witness_tuple)
        return tuple(
            witness
            for witness in witness_tuple
            if witness.annotation_count == max_count
        )
    if mode is AnnotationMode.HARD:
        return witness_tuple
    raise ValueError(f"unknown annotation mode: {mode!r}")


__all__ = (
    "ValidWitness",
    "select_witnesses_by_annotation_count",
)
