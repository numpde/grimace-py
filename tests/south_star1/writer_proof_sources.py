"""Reusable branch and terminal proof-source selection."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.ids import BondId
from grimace._south_star1.policy import DirectionMark, SmilesPolicy
from grimace._south_star1.prepared_runtime import SouthStarPreparedMol, SouthStarRuntimeOptions
from grimace._south_star1.writer_frontier import _checked_writer_frontier_branch_supports
from grimace._south_star1.writer_events import WriterRingEndpointEmitted, WriterRingEndpointPaired
from grimace._south_star1.writer_snapshot import (
    WriterDecoderBoundary,
    capture_writer_frontier_snapshot,
)
from tests.south_star1.writer_test_context import (
    initial_writer_snapshot,
    prepare_writer_facts,
    writer_runtime_options,
)
from tests.south_star1.writer_test_fixtures import shared_directional_ring_carrier_facts

SharedRingBranchPhase = Literal["opening", "pair"]


@dataclass(frozen=True, slots=True)
class WriterBranchProofSource:
    phase: SharedRingBranchPhase
    direction_mark: DirectionMark
    facts: MoleculeFacts
    runtime_options: SouthStarRuntimeOptions
    prepared: SouthStarPreparedMol
    snapshot: object
    support: object


@dataclass(frozen=True, slots=True)
class WriterTerminalProofSource:
    facts: MoleculeFacts
    runtime_options: SouthStarRuntimeOptions
    prepared: SouthStarPreparedMol
    snapshot: object
    support: object
    policy: SmilesPolicy | None


@lru_cache(maxsize=1)
def shared_ring_branch_sources() -> tuple[WriterBranchProofSource, ...]:
    facts = shared_directional_ring_carrier_facts()
    options = writer_runtime_options(rooted_at_atom=1)
    prepared = prepare_writer_facts(facts)
    initial = initial_writer_snapshot(prepared, options)
    pending = [(initial.cursor, 0)]
    seen = set()
    found = {}
    while pending and len(found) < 6:
        cursor, depth = pending.pop()
        key = repr(cursor)
        if key in seen:
            continue
        seen.add(key)
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            cursor,
            include_counts=False,
            include_frontier_certificate=True,
            include_count_certificate=False,
        )
        snapshot = (
            initial_writer_snapshot(prepared, options)
            if depth == 0
            else capture_writer_frontier_snapshot(
                prepared=prepared,
                runtime_options=options,
                cursor=cursor,
                decoder_boundary=WriterDecoderBoundary(consumed_token_count=depth),
            )
        )
        for support in batch.supports:
            for event in support.events:
                if isinstance(event, WriterRingEndpointEmitted) and event.bond == BondId(1):
                    found.setdefault(("opening", event.direction_mark), (facts, options, prepared, snapshot, support))
                if isinstance(event, WriterRingEndpointPaired) and event.bond == BondId(1):
                    found.setdefault(("pair", event.first_endpoint_direction_mark), (facts, options, prepared, snapshot, support))
            pending.append((support.successor_cursor, depth + 1))
    if len(found) != 6:
        raise AssertionError(f"missing shared-ring branch sources: {sorted(found)}")
    return tuple(
        WriterBranchProofSource(phase, mark, *found[(phase, mark)])
        for phase in ("opening", "pair")
        for mark in sorted(DirectionMark, key=lambda item: item.value)
    )


def shared_ring_branch_source(
    phase: SharedRingBranchPhase,
    direction_mark: DirectionMark,
) -> WriterBranchProofSource:
    for source in shared_ring_branch_sources():
        if source.phase == phase and source.direction_mark is direction_mark:
            return source
    raise ValueError(f"unknown shared-ring branch source: {phase!r}, {direction_mark!r}")


def first_terminal_proof_source(
    facts: MoleculeFacts,
    runtime_options: SouthStarRuntimeOptions,
    *,
    policy: SmilesPolicy | None = None,
) -> WriterTerminalProofSource:
    prepared = prepare_writer_facts(facts, policy=policy)
    snapshot = initial_writer_snapshot(prepared, runtime_options)
    from grimace._south_star1.writer_snapshot import capture_writer_frontier_snapshot

    for depth in range(256):
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            snapshot.cursor,
            include_counts=False,
            include_frontier_certificate=True,
            include_count_certificate=False,
        )
        if batch.terminal_supports:
            return WriterTerminalProofSource(
                facts, runtime_options, prepared, snapshot, batch.terminal_supports[0], policy
            )
        support = batch.supports[0]
        snapshot = capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=runtime_options,
            cursor=support.successor_cursor,
            decoder_boundary=WriterDecoderBoundary(depth + 1),
        )
    raise AssertionError("terminal support not reached")


__all__ = (
    "WriterBranchProofSource",
    "WriterTerminalProofSource",
    "first_terminal_proof_source",
    "shared_ring_branch_source",
    "shared_ring_branch_sources",
)
