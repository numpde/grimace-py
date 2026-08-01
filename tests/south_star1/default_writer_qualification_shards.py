"""Disjoint fast and slow qualification shards for accepted writer cases."""

from __future__ import annotations

from tests.south_star1.default_writer_capability_ledger import (
    ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES,
)


SLOW_COUPLED_CASE_NAMES = (
    "zero_h_tetrahedral",
    "adjacent_specified_tetrahedral",
    "remote_coupled_tetrahedral_a",
    "remote_coupled_tetrahedral_b",
)

FAST_ACCEPTED_CASES = tuple(
    case
    for case in ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES
    if case.name not in SLOW_COUPLED_CASE_NAMES
)
SLOW_COUPLED_CASES = tuple(
    case
    for case in ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES
    if case.name in SLOW_COUPLED_CASE_NAMES
)
