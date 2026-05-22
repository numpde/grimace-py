"""Enumeration orchestration for the private proof kernel.

Enumeration must compose facts, policies, semantics, skeletons, slots,
constraints, assignments, annotation policy, and pure rendering. It must not
call RDKit or any other post-hoc parser as a validity filter.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .annotation import ValidWitness


@dataclass(frozen=True, slots=True)
class SupportImage:
    witness_count: int
    distinct_count: int
    strings: tuple[str, ...]


def render_image_from_witnesses(
    witnesses: Iterable[ValidWitness],
) -> SupportImage:
    witness_tuple = tuple(witnesses)
    rendered = tuple(witness.rendered for witness in witness_tuple)
    return SupportImage(
        witness_count=len(witness_tuple),
        distinct_count=len(set(rendered)),
        strings=rendered,
    )


__all__ = (
    "SupportImage",
    "render_image_from_witnesses",
)
