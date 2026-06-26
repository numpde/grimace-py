"""Private writer execution evidence records."""

from __future__ import annotations

from dataclasses import dataclass

from .residual_constraints import ResidualFactorKey
from .residual_constraints import ResidualPropagationKind
from .residual_constraints import ResidualPropagationResult
from .residual_constraints import VarId


@dataclass(frozen=True, slots=True)
class WriterResidualPropagationWorkEvidence:
    operation: str
    result_kind: ResidualPropagationKind
    component_variables: tuple[VarId, ...]
    component_factor_keys: tuple[ResidualFactorKey, ...]
    checked_candidate_rows: int
    largest_factor_scope: int
    largest_candidate_row_count: int

    @property
    def component_variable_count(self) -> int:
        return len(self.component_variables)

    @property
    def component_factor_count(self) -> int:
        return len(self.component_factor_keys)


def writer_residual_propagation_work_evidence(
    *,
    operation: str,
    result: ResidualPropagationResult,
) -> WriterResidualPropagationWorkEvidence:
    stats = result.stats
    return WriterResidualPropagationWorkEvidence(
        operation=operation,
        result_kind=result.kind,
        component_variables=stats.component_variables,
        component_factor_keys=stats.component_factor_keys,
        checked_candidate_rows=stats.checked_candidate_rows,
        largest_factor_scope=stats.largest_factor_scope,
        largest_candidate_row_count=stats.largest_candidate_row_count,
    )


__all__ = [
    "WriterResidualPropagationWorkEvidence",
    "writer_residual_propagation_work_evidence",
]
