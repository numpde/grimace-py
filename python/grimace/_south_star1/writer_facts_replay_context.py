"""Facts-derived context shared by repeated local writer proof replay."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .facts import MoleculeFacts
from .prepared_runtime import SouthStarRuntimeOptions
from .prepared_runtime import SouthStarPreparedMol
from .prepared_runtime import SouthStarWriterSurface
from .prepared_runtime import prepare_south_star_mol_from_facts
from .writer_envelope_terms import _identity_envelope
from .writer_envelope_work import default_writer_envelope_work_budget
from .writer_prepared_identity import writer_prepared_identity


@dataclass(frozen=True, slots=True)
class _WriterFactsReplayContext:
    facts: MoleculeFacts
    runtime_options: SouthStarRuntimeOptions
    prepared: SouthStarPreparedMol
    expected_identity: Mapping[str, object]


def _writer_facts_replay_context(
    *, facts, runtime_options: SouthStarRuntimeOptions, policy=None, budget=None
) -> _WriterFactsReplayContext:
    budget = default_writer_envelope_work_budget(budget)
    prepared = prepare_south_star_mol_from_facts(
        facts,
        writer_surface=SouthStarWriterSurface(),
        policy=policy,
    )
    return _WriterFactsReplayContext(
        facts=facts,
        runtime_options=runtime_options,
        prepared=prepared,
        expected_identity=_identity_envelope(
            writer_prepared_identity(prepared, runtime_options),
            budget=budget,
            operation="writer_facts_replay_context.prepared_identity",
        ),
    )


__all__ = (
    "_WriterFactsReplayContext",
    "_writer_facts_replay_context",
)
