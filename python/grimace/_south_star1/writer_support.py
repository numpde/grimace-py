"""Support-image adapter for the writer-shaped frontier kernel."""

from __future__ import annotations

from collections.abc import Iterator

from . import writer_snapshot
from .enumerate import SupportImage
from .prepared_runtime import SouthStarPreparedMol
from .prepared_runtime import SouthStarRuntimeOptions
from .prepared_runtime import require_writer_shaped_runtime_options
from .prepared_runtime import runtime_root_atom_for_prepared
from . import writer_frontier
from .writer_frontier import count_writer_cursor_completions
from .writer_frontier import count_writer_frontier_support
from .writer_frontier import iter_writer_frontier_support
from .writer_frontier import WriterFrontierCursor


def enumerate_prepared_writer_shaped_support(
    *,
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions,
) -> SupportImage:
    require_writer_shaped_runtime_options(runtime_options)
    runtime_root_atom_for_prepared(runtime_options, prepared=prepared)
    cursor = writer_snapshot._initial_checked_writer_frontier_cursor(
        prepared=prepared,
        runtime_options=runtime_options,
    )
    return _writer_support_image_from_cursor(
        prepared=prepared,
        cursor=cursor,
    )


def _writer_support_image_from_cursor(
    *,
    prepared: SouthStarPreparedMol,
    cursor: WriterFrontierCursor,
) -> SupportImage:
    summary = writer_frontier._writer_frontier_summary(
        prepared,
        cursor,
        include_support_count=True,
        include_completion_count=True,
        include_strings=True,
    )

    strings = summary.require_strings()
    support_count = summary.require_support_count()
    completion_count = summary.require_completion_count()
    if len(strings) != support_count:
        raise AssertionError("writer frontier support stream/count mismatch")
    return SupportImage(
        witness_count=completion_count,
        distinct_count=support_count,
        strings=strings,
    )


def enumerate_writer_snapshot_writer_shaped_support(
    *,
    snapshot: writer_snapshot.WriterSearchSnapshot,
    prepared: SouthStarPreparedMol,
) -> SupportImage:
    cursor = writer_snapshot.writer_frontier_cursor_from_snapshot(
        snapshot,
        prepared=prepared,
    )
    return _writer_support_image_from_cursor(
        prepared=prepared,
        cursor=cursor,
    )


def count_writer_snapshot_writer_shaped_support(
    *,
    snapshot: writer_snapshot.WriterSearchSnapshot,
    prepared: SouthStarPreparedMol,
) -> int:
    cursor = writer_snapshot.writer_frontier_cursor_from_snapshot(
        snapshot,
        prepared=prepared,
    )
    return count_writer_frontier_support(prepared, cursor.support_state)


def count_writer_snapshot_writer_shaped_completions(
    *,
    snapshot: writer_snapshot.WriterSearchSnapshot,
    prepared: SouthStarPreparedMol,
) -> int:
    cursor = writer_snapshot.writer_frontier_cursor_from_snapshot(
        snapshot,
        prepared=prepared,
    )
    return count_writer_cursor_completions(prepared, cursor)


def iter_writer_snapshot_writer_shaped_support(
    *,
    snapshot: writer_snapshot.WriterSearchSnapshot,
    prepared: SouthStarPreparedMol,
) -> Iterator[str]:
    cursor = writer_snapshot.writer_frontier_cursor_from_snapshot(
        snapshot,
        prepared=prepared,
    )
    return iter_writer_frontier_support(prepared, cursor)


__all__ = (
    "enumerate_prepared_writer_shaped_support",
    "enumerate_writer_snapshot_writer_shaped_support",
    "count_writer_snapshot_writer_shaped_support",
    "count_writer_snapshot_writer_shaped_completions",
    "iter_writer_snapshot_writer_shaped_support",
)
