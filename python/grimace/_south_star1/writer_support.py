"""Support-image adapter for the writer-shaped frontier kernel."""

from __future__ import annotations

from .enumerate import SupportImage
from .prepared_runtime import SouthStarPreparedMol
from .prepared_runtime import SouthStarRuntimeOptions
from .prepared_runtime import require_writer_shaped_runtime_options
from .prepared_runtime import runtime_root_atom_for_prepared
from .writer_frontier import count_writer_frontier_support
from .writer_frontier import initial_writer_frontier
from .writer_frontier import iter_writer_frontier_support


def enumerate_prepared_writer_shaped_support(
    *,
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions,
) -> SupportImage:
    require_writer_shaped_runtime_options(runtime_options)
    runtime_root_atom_for_prepared(runtime_options, prepared=prepared)
    frontier = initial_writer_frontier(prepared, runtime_options)
    support_count = count_writer_frontier_support(prepared, frontier)
    strings = tuple(iter_writer_frontier_support(prepared, frontier))
    if len(strings) != support_count:
        raise AssertionError("writer frontier support stream/count mismatch")
    return SupportImage(
        witness_count=support_count,
        distinct_count=support_count,
        strings=strings,
    )


__all__ = ("enumerate_prepared_writer_shaped_support",)
