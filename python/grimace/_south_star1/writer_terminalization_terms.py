"""Closed proof terms for one writer EOS transition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .ids import AtomId
from .writer_graph_obligations import WriterGraphCompletionStatus


@dataclass(frozen=True, slots=True)
class WriterTerminalizationTerm:
    source_state_digest: str
    finalized_state_digest: str
    active_atom: AtomId
    graph_completion_status: WriterGraphCompletionStatus
    graph_obligation_work_digests: tuple[str, ...]
    stereo_mode: Literal["noop", "tetra_local_order_factor_closure"]
    source_residual_snapshot_digest: str
    finalized_residual_snapshot_digest: str
    terminal_residual_work_digests: tuple[str, ...]
    terminal_stereo_lifecycle_digests: tuple[str, ...]
    terminal_execution_capabilities: tuple[str, ...]


__all__ = ("WriterTerminalizationTerm",)
