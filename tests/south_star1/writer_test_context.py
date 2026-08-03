"""Shared preparation vocabulary for writer tests."""

from __future__ import annotations

from dataclasses import dataclass

from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.policy import SerializationLanguageMode, SmilesPolicy
from grimace._south_star1.prepared_runtime import (
    SouthStarPreparedMol,
    SouthStarRuntimeOptions,
    SouthStarWriterSurface,
    prepare_south_star_mol_from_facts,
)
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_snapshot import (
    WriterSearchSnapshot,
    capture_writer_frontier_snapshot,
)


def writer_runtime_options(*, rooted_at_atom: int = -1) -> SouthStarRuntimeOptions:
    return SouthStarRuntimeOptions(
        rooted_at_atom=rooted_at_atom,
        canonical=False,
        do_random=True,
        serialization_language=SerializationLanguageMode.WRITER_SHAPED,
    )


def prepare_writer_facts(
    facts: MoleculeFacts,
    *,
    policy: SmilesPolicy | None = None,
) -> SouthStarPreparedMol:
    return prepare_south_star_mol_from_facts(
        facts,
        writer_surface=SouthStarWriterSurface(),
        policy=policy,
    )


def initial_writer_snapshot(
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions,
) -> WriterSearchSnapshot:
    return capture_writer_frontier_snapshot(
        prepared=prepared,
        runtime_options=runtime_options,
        cursor=initial_writer_frontier_cursor(prepared, runtime_options),
    )


@dataclass(frozen=True, slots=True)
class WriterTestContext:
    runtime_options: SouthStarRuntimeOptions
    prepared: SouthStarPreparedMol
    initial_snapshot: WriterSearchSnapshot


def writer_test_context(
    facts: MoleculeFacts,
    *,
    runtime_options: SouthStarRuntimeOptions | None = None,
    rooted_at_atom: int | None = None,
    policy: SmilesPolicy | None = None,
) -> WriterTestContext:
    if runtime_options is not None and rooted_at_atom is not None:
        raise ValueError("runtime_options and rooted_at_atom are mutually exclusive")
    if runtime_options is None:
        runtime_options = writer_runtime_options(
            rooted_at_atom=-1 if rooted_at_atom is None else rooted_at_atom
        )
    prepared = prepare_writer_facts(facts, policy=policy)
    return WriterTestContext(
        runtime_options,
        prepared,
        initial_writer_snapshot(prepared, runtime_options),
    )


__all__ = (
    "WriterTestContext",
    "initial_writer_snapshot",
    "prepare_writer_facts",
    "writer_runtime_options",
    "writer_test_context",
)
