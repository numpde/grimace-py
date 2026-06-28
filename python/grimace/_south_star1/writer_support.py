"""Support-image adapter for the writer-shaped live runtime."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from .enumerate import SupportImage
from .prepared_runtime import SouthStarPreparedMol
from .prepared_runtime import SouthStarRuntimeOptions
from .writer_runtime import WriterRuntimeState
from .writer_runtime import count_writer_runtime_completions
from .writer_runtime import count_writer_runtime_support
from .writer_runtime import initial_writer_runtime_state
from .writer_runtime import iter_writer_runtime_support
from .writer_runtime import writer_runtime_state_from_snapshot

if TYPE_CHECKING:
    from .writer_snapshot import WriterSearchSnapshot


# This module materializes support images; it should not decide support itself.
# Keep it above `writer_runtime`, which is the named boundary for checked live
# writer operations over snapshots/frontiers.
def enumerate_prepared_writer_shaped_support(
    *,
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions,
) -> SupportImage:
    return _writer_support_image_from_runtime_state(
        prepared=prepared,
        state=initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=runtime_options,
        ),
    )


def _writer_support_image_from_runtime_state(
    *,
    prepared: SouthStarPreparedMol,
    state: WriterRuntimeState,
) -> SupportImage:
    strings = tuple(
        iter_writer_runtime_support(
            prepared=prepared,
            state=state,
        )
    )
    support_count = count_writer_runtime_support(
        prepared=prepared,
        state=state,
    )
    if len(strings) != support_count:
        raise AssertionError("writer runtime support stream/count mismatch")
    return SupportImage(
        witness_count=count_writer_runtime_completions(
            prepared=prepared,
            state=state,
        ),
        distinct_count=support_count,
        strings=strings,
    )


def enumerate_writer_snapshot_writer_shaped_support(
    *,
    snapshot: WriterSearchSnapshot,
    prepared: SouthStarPreparedMol,
) -> SupportImage:
    return _writer_support_image_from_runtime_state(
        prepared=prepared,
        state=writer_runtime_state_from_snapshot(snapshot, prepared=prepared),
    )


def count_writer_snapshot_writer_shaped_support(
    *,
    snapshot: WriterSearchSnapshot,
    prepared: SouthStarPreparedMol,
) -> int:
    return count_writer_runtime_support(
        prepared=prepared,
        state=writer_runtime_state_from_snapshot(snapshot, prepared=prepared),
    )


def count_writer_snapshot_writer_shaped_completions(
    *,
    snapshot: WriterSearchSnapshot,
    prepared: SouthStarPreparedMol,
) -> int:
    return count_writer_runtime_completions(
        prepared=prepared,
        state=writer_runtime_state_from_snapshot(snapshot, prepared=prepared),
    )


def iter_writer_snapshot_writer_shaped_support(
    *,
    snapshot: WriterSearchSnapshot,
    prepared: SouthStarPreparedMol,
) -> Iterator[str]:
    return iter_writer_runtime_support(
        prepared=prepared,
        state=writer_runtime_state_from_snapshot(snapshot, prepared=prepared),
    )


__all__ = (
    "enumerate_prepared_writer_shaped_support",
    "enumerate_writer_snapshot_writer_shaped_support",
    "count_writer_snapshot_writer_shaped_support",
    "count_writer_snapshot_writer_shaped_completions",
    "iter_writer_snapshot_writer_shaped_support",
)
