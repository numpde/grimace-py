"""Exhaustive directional-ring rich-artifact diagnostic fixtures."""

from collections import deque
from copy import deepcopy
from functools import lru_cache
from grimace._south_star1.ids import BondId
from grimace._south_star1.policy import DirectionMark
from grimace._south_star1.writer_events import WriterRingEndpointEmitted
from grimace._south_star1.writer_events import WriterRingEndpointPaired
from grimace._south_star1.writer_frontier import _checked_writer_frontier_branch_supports
from grimace._south_star1.writer_snapshot import WriterDecoderBoundary
from grimace._south_star1.writer_snapshot import capture_writer_frontier_snapshot
from grimace._south_star1.writer_support_artifact_envelope import writer_support_artifact_envelope_for_snapshot
from tests.south_star1.writer_test_context import initial_writer_snapshot
from tests.south_star1.writer_test_context import prepare_writer_facts
from tests.south_star1.writer_test_context import writer_runtime_options
from tests.south_star1.writer_test_fixtures import directional_ring_carrier_facts

@lru_cache(maxsize=1)
def _cached_directional_ring_opening_slow_fixture():
    facts = directional_ring_carrier_facts()
    options = writer_runtime_options(rooted_at_atom=0)
    prepared = prepare_writer_facts(facts)
    initial = initial_writer_snapshot(prepared, options)
    frontier = deque([(initial.cursor, 0)])
    seen = set()
    opening_sources = []
    while frontier:
        cursor, depth = frontier.popleft()
        if cursor in seen:
            continue
        seen.add(cursor)
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            cursor,
            include_counts=False,
            include_frontier_certificate=False,
            include_count_certificate=False,
        )
        for support in batch.supports:
            if any(
                isinstance(event, WriterRingEndpointEmitted)
                and event.bond == BondId(3)
                for event in support.events
            ):
                opening_sources.append((cursor, depth))
            frontier.append((support.successor_cursor, depth + 1))
    if not opening_sources:
        raise AssertionError("missing cursor before BondId(3) ring opening")
    source, source_depth = max(opening_sources, key=lambda item: item[1])
    snapshot = capture_writer_frontier_snapshot(
        prepared=prepared,
        runtime_options=options,
        cursor=source,
        decoder_boundary=WriterDecoderBoundary(consumed_token_count=source_depth),
    )
    artifact = writer_support_artifact_envelope_for_snapshot(
        prepared=prepared,
        snapshot=snapshot,
    )
    return facts, options, deepcopy(artifact)


def directional_ring_opening_slow_fixture():
    facts, options, artifact = _cached_directional_ring_opening_slow_fixture()
    return facts, options, deepcopy(artifact)


@lru_cache(maxsize=2)
def _cached_directional_ring_pair_slow_fixture(first_mark: DirectionMark):
    facts = directional_ring_carrier_facts()
    options = writer_runtime_options(rooted_at_atom=0)
    prepared = prepare_writer_facts(facts)
    initial = initial_writer_snapshot(prepared, options)
    frontier = deque([(initial.cursor, 0)])
    seen = set()
    source = None
    source_depth = None
    while frontier and source is None:
        cursor, depth = frontier.popleft()
        if cursor in seen:
            continue
        seen.add(cursor)
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            cursor,
            include_counts=False,
            include_frontier_certificate=False,
            include_count_certificate=False,
        )
        for support in batch.supports:
            if any(
                isinstance(event, WriterRingEndpointPaired)
                and event.bond == BondId(3)
                and event.first_endpoint_direction_mark is first_mark
                for event in support.events
            ):
                source = cursor
                source_depth = depth
                break
            frontier.append((support.successor_cursor, depth + 1))
    if source is None or source_depth is None:
        raise AssertionError(f"missing cursor before BondId(3) pair with first mark {first_mark}")
    snapshot = capture_writer_frontier_snapshot(
        prepared=prepared,
        runtime_options=options,
        cursor=source,
        decoder_boundary=WriterDecoderBoundary(consumed_token_count=source_depth),
    )
    artifact = writer_support_artifact_envelope_for_snapshot(
        prepared=prepared,
        snapshot=snapshot,
    )
    return facts, options, deepcopy(artifact)


def directional_ring_pair_slow_fixture(first_mark: DirectionMark):
    facts, options, artifact = _cached_directional_ring_pair_slow_fixture(first_mark)
    return facts, options, deepcopy(artifact)
