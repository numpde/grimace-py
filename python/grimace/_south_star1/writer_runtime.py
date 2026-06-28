"""Small public facade for the writer-shaped live runtime.

This module intentionally owns no support logic.  It names the runtime boundary
that public callers should use, while delegating every decision to the existing
checked snapshot/frontier operations.  Keeping this layer thin prevents a second
support authority from growing beside the live writer transition engine.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from .prepared_runtime import SouthStarPreparedMol
from .prepared_runtime import SouthStarRuntimeOptions
from .writer_frontier import WriterFrontierChoices
from .writer_frontier import WriterFrontierTerminal
from .writer_snapshot import WriterDecoderBoundary
from .writer_snapshot import WriterSearchSnapshot
from .writer_snapshot import advance_writer_frontier_snapshot
from .writer_snapshot import capture_initial_writer_frontier_snapshot
from .writer_snapshot import resume_writer_frontier_choices_from_snapshot
from .writer_snapshot import validate_writer_search_snapshot
from .writer_support import count_writer_snapshot_writer_shaped_completions
from .writer_support import count_writer_snapshot_writer_shaped_support
from .writer_support import iter_writer_snapshot_writer_shaped_support


@dataclass(frozen=True, slots=True)
class WriterRuntimeState:
    """Opaque writer-runtime state for public traversal.

    The payload is a writer snapshot rather than a separate state encoding.  A
    snapshot already carries the saved writer cursor plus structural identity;
    checked runtime operations below are responsible for enforcing support by
    running the live frontier.
    """

    snapshot: WriterSearchSnapshot


def initial_writer_runtime_state(
    *,
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions,
    decoder_boundary: WriterDecoderBoundary = WriterDecoderBoundary(),
) -> WriterRuntimeState:
    return WriterRuntimeState(
        capture_initial_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=runtime_options,
            decoder_boundary=decoder_boundary,
        )
    )


def writer_runtime_state_from_snapshot(
    snapshot: WriterSearchSnapshot,
    *,
    prepared: SouthStarPreparedMol,
) -> WriterRuntimeState:
    """Resume a structurally valid snapshot without classifying support.

    Snapshot validation checks that the saved writer state is coherent for the
    prepared molecule.  Calls such as ``writer_runtime_choices`` and
    ``advance_writer_runtime_state`` perform the checked frontier operation that
    can reject unsupported live execution.
    """

    validate_writer_search_snapshot(snapshot, prepared=prepared)
    return WriterRuntimeState(snapshot)


def writer_runtime_choices(
    *,
    prepared: SouthStarPreparedMol,
    state: WriterRuntimeState,
) -> WriterFrontierChoices:
    return resume_writer_frontier_choices_from_snapshot(
        state.snapshot,
        prepared=prepared,
    )


def writer_runtime_terminal(
    *,
    prepared: SouthStarPreparedMol,
    state: WriterRuntimeState,
) -> WriterFrontierTerminal | None:
    return writer_runtime_choices(
        prepared=prepared,
        state=state,
    ).terminal


def writer_runtime_has_eos(
    *,
    prepared: SouthStarPreparedMol,
    state: WriterRuntimeState,
) -> bool:
    return writer_runtime_terminal(
        prepared=prepared,
        state=state,
    ) is not None


def advance_writer_runtime_state(
    *,
    prepared: SouthStarPreparedMol,
    state: WriterRuntimeState,
    emitted_text: str,
) -> WriterRuntimeState:
    return WriterRuntimeState(
        advance_writer_frontier_snapshot(
            state.snapshot,
            prepared=prepared,
            emitted_text=emitted_text,
        )
    )


def count_writer_runtime_support(
    *,
    prepared: SouthStarPreparedMol,
    state: WriterRuntimeState,
) -> int:
    return count_writer_snapshot_writer_shaped_support(
        snapshot=state.snapshot,
        prepared=prepared,
    )


def count_writer_runtime_completions(
    *,
    prepared: SouthStarPreparedMol,
    state: WriterRuntimeState,
) -> int:
    return count_writer_snapshot_writer_shaped_completions(
        snapshot=state.snapshot,
        prepared=prepared,
    )


def iter_writer_runtime_support(
    *,
    prepared: SouthStarPreparedMol,
    state: WriterRuntimeState,
) -> Iterator[str]:
    return iter_writer_snapshot_writer_shaped_support(
        snapshot=state.snapshot,
        prepared=prepared,
    )


__all__ = (
    "WriterRuntimeState",
    "advance_writer_runtime_state",
    "count_writer_runtime_completions",
    "count_writer_runtime_support",
    "initial_writer_runtime_state",
    "iter_writer_runtime_support",
    "writer_runtime_choices",
    "writer_runtime_has_eos",
    "writer_runtime_state_from_snapshot",
    "writer_runtime_terminal",
)
